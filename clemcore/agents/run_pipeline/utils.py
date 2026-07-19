import json
import socket
import subprocess
import time
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path

from clemcore.clemgame.instances import GameInstances
from clemcore.clemgame.registry import GameRegistry

from clemcore.agents.mcp.server import run_clem_mcp_server
from clemcore.agents.model_connection import resolve_agent_model_connection
from clemcore.agents.mcp.bridge import OpenEnvMCPClient

CLEMCORE_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_DIR = CLEMCORE_ROOT / "docker" / "agent-sandbox"
REGISTRY_PATH = SANDBOX_DIR / "agent_registry.json"
ENV_FILE = SANDBOX_DIR / ".env"

DOCKER_IMAGE = "clem-agent-sandbox:dev"
SERVER_PORT = 8001
OPENENV_MCP_URL = f"http://host.docker.internal:{SERVER_PORT}/mcp"
HOST_OPENENV_MCP_URL = f"http://127.0.0.1:{SERVER_PORT}/mcp"
CONTROL_FAILURE_RESPONSE = (
    "CLEM_AGENT_CONTROL_ERROR: external harness ended before completing the game"
)


def _load_game_instances(game_name: str,
                         instances_filename: str | None,
                         experiment_name: str | None) -> GameInstances:
    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_spec = game_registry.get_game_spec(game_name)

    if instances_filename:
        game_spec.instances = instances_filename

    experiment_filter = None

    if experiment_name:
        experiment_filter = lambda row: row["experiment"]["name"] == experiment_name

    game_instances = GameInstances.from_game_spec(game_spec)
    game_instances = game_instances.filter(experiment_filter)

    return game_instances


def _player_ids_from_clemgame(game_name: str) -> list[str] | None:
    """Return player ids declared by the game's clemgame.json, if available."""
    metadata_path = Path.cwd() / game_name / "clemgame.json"
    if not metadata_path.exists():
        return None

    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    specs: list[dict] = []
    if isinstance(data, list):
        specs = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        if data.get("name") == game_name or data.get("game_name") == game_name:
            specs.append(data)
        for key in ("games", "game_specs", "benchmarks"):
            value = data.get(key)
            if isinstance(value, list):
                specs.extend(item for item in value if isinstance(item, dict))

    for spec in specs:
        if spec.get("name") != game_name and spec.get("game_name") != game_name:
            continue

        player_count = (
            spec.get("players")
            or spec.get("num_players")
            or spec.get("n_players")
            or spec.get("number_of_players")
        )

        if isinstance(player_count, int):
            return [f"player_{index}" for index in range(player_count)]

        roles = spec.get("roles")
        if isinstance(roles, list):
            return [f"player_{index}" for index in range(len(roles))]

    return None


def _env_agents_from_models(models: list[str],
                            learner_agent: str,
                            game_name: str | None = None) -> dict[str, str]:
    if game_name is not None:
        player_ids = _player_ids_from_clemgame(game_name)
    else:
        player_ids = None

    if player_ids is None:
        number_of_players = len(models) + 1
        player_ids = [f"player_{index}" for index in range(number_of_players)]

    if learner_agent not in player_ids:
        raise ValueError(
            f"Cannot use learner_agent={learner_agent!r}. "
            f"Valid choices are: {player_ids}"
        )

    env_player_ids = [
        player_id
        for player_id in player_ids
        if player_id != learner_agent
    ]

    if len(models) < len(env_player_ids):
        raise ValueError(
            f"Need {len(env_player_ids)} native model(s) for non-agent players "
            f"{env_player_ids}, but got {len(models)}: {models}"
        )

    return {
        player_id: model
        for player_id, model in zip(env_player_ids, models)
    }



def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _ensure_port_free(port: int) -> None:
    if _port_is_open("127.0.0.1", port):
        raise RuntimeError(
            f"Port {port} is already in use. Stop the existing MCP server before running the pipeline."
        )


def _wait_for_server(port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        if _port_is_open("127.0.0.1", port):
            return

        time.sleep(0.25)

    raise TimeoutError(f"MCP server did not become available on port {port}.")


def _serve(game_name: str,
           agent_name: str,
           agent_player: str,
           env_agent_models: list[str],
           instances_filename: str | None,
           results_dir: str,
           port: int) -> None:
    run_clem_mcp_server(
        game_name=game_name,
        agent_name=agent_name,
        registry_path=REGISTRY_PATH,
        learner_agent=agent_player,
        env_agents=_env_agents_from_models(
            models=env_agent_models,
            learner_agent=agent_player,
            game_name=game_name,
        ),
        game_instance_split=None,
        instances_filename=instances_filename,
        single_pass=False,
        results_dir=results_dir,
        port=port,
    )


def _start_server(game_name: str,
                  agent_name: str,
                  agent_player: str,
                  env_agent_models: list[str],
                  instances_filename: str | None,
                  results_dir: str) -> Process:
    _ensure_port_free(SERVER_PORT)

    process = Process(
        target=_serve,
        kwargs=dict(
            game_name=game_name,
            agent_name=agent_name,
            agent_player=agent_player,
            env_agent_models=env_agent_models,
            instances_filename=instances_filename,
            results_dir=results_dir,
            port=SERVER_PORT,
        ),
    )
    process.start()

    _wait_for_server(SERVER_PORT)

    if not process.is_alive():
        raise RuntimeError("MCP server process exited during startup.")

    return process


def _stop_server(process: Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)

    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def _episode_dirs(results_dir: str,
                  game_name: str,
                  experiment_name: str) -> set[Path]:
    root = Path(results_dir)

    return {
        path
        for path in root.glob(f"*/{game_name}/{experiment_name}/episode_*")
        if path.is_dir()
    }


def _episode_dir_timestamp(episode_dir: Path) -> float:
    candidates = [
        episode_dir / "interactions.json",
        episode_dir / "instance.json",
        episode_dir,
    ]

    return max(
        candidate.stat().st_mtime
        for candidate in candidates
        if candidate.exists()
    )


def _episode_dir_matches_game_id(episode_dir: Path,
                                 game_id: int | str) -> bool:
    instance_path = episode_dir / "instance.json"

    if not instance_path.exists():
        return False

    try:
        instance = json.loads(instance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    return int(instance.get("game_id")) == int(game_id)


def _write_agent_trace(results_dir: str,
                       game_name: str,
                       experiment_name: str,
                       game_id: int | str,
                       before_episode_dirs: set[Path],
                       trace_text: str,
                       metadata: dict) -> Path:
    after_episode_dirs = _episode_dirs(
        results_dir=results_dir,
        game_name=game_name,
        experiment_name=experiment_name,
    )
    new_episode_dirs = sorted(after_episode_dirs - before_episode_dirs)

    if new_episode_dirs:
        episode_dir = max(new_episode_dirs, key=_episode_dir_timestamp)
    else:
        matching_episode_dirs = [
            episode_dir
            for episode_dir in after_episode_dirs
            if _episode_dir_matches_game_id(episode_dir, game_id)
        ]

        if not matching_episode_dirs:
            failure_dir = (
                Path(results_dir)
                / "_agent_failures"
                / game_name
                / experiment_name
                / f"game_id_{game_id}"
                / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            )
            failure_dir.mkdir(parents=True, exist_ok=True)

            trace_path = failure_dir / "agent_trace.log"
            trace_path.write_text(trace_text, encoding="utf-8")

            meta_path = failure_dir / "agent_trace_meta.json"
            meta_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            return trace_path

        episode_dir = max(matching_episode_dirs, key=_episode_dir_timestamp)

    trace_path = episode_dir / "agent_trace.log"
    trace_path.write_text(trace_text, encoding="utf-8")

    meta_path = episode_dir / "agent_trace_meta.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return trace_path


def _run_docker_episode(experiment_name: str,
                        game_id: int | str,
                        agent_name: str,
                        model_connection_path: Path | None = None,
                        shared_state_dir: Path | None = None) -> tuple[str, int]:
    if not ENV_FILE.exists():
        raise FileNotFoundError(f"Missing Docker env file: {ENV_FILE}")

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing agent registry: {REGISTRY_PATH}")

    command = [
        "docker", "run", "--rm", "-i",
        "--env-file", str(ENV_FILE),
        "-e", f"OPENENV_MCP_URL={OPENENV_MCP_URL}",
        "-e", f"CLEM_EXPERIMENT_NAME={experiment_name}",
        "-e", f"CLEM_GAME_ID={game_id}",
        "-e", f"CLEM_AGENT_NAME={agent_name}",
        "-v", f"{CLEMCORE_ROOT}:/opt/clemcore:ro",
        "-v", f"{SANDBOX_DIR}:/app:ro",
    ]

    if shared_state_dir is not None:
        command.extend([
            "-e", "CLEM_OPENENV_SESSION_PATH=/run/clem-agent/openenv_session.json",
            "-v", f"{shared_state_dir}:/run/clem-agent:rw",
        ])

    if model_connection_path is not None:
        command.extend([
            "-e", "CLEM_AGENT_MODEL_CONNECTION_PATH=/run/clem-agent/model_connection.json",
        ])

        if shared_state_dir is None:
            command.extend([
                "-v", f"{model_connection_path}:/run/clem-agent/model_connection.json:ro",
            ])

    command.extend([
        DOCKER_IMAGE,
        "python", "/app/run_agent_container.py",
    ])

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    trace_lines = []

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")
        trace_lines.append(line)

    return_code = process.wait()
    trace_text = "".join(trace_lines)

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            command,
            output=trace_text,
        )

    return trace_text, return_code



def _cleanup_openenv_session_from_file(session_path: Path,
                                       trace_text: str) -> None:
    if not session_path.exists():
        return

    try:
        session_data = json.loads(session_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        print(f"openenv_session_file_invalid: path={session_path} error={error}")
        return

    session_id = session_data.get("session_id")

    if not session_id:
        return

    completed = (
        '"done":true' in trace_text
        or '"done": true' in trace_text
        or "'done': True" in trace_text
    )

    client = OpenEnvMCPClient(HOST_OPENENV_MCP_URL)
    client.session_id = session_id

    if completed:
        return

    try:
        result = client.call_tool(
            "submit_response",
            {"response": CONTROL_FAILURE_RESPONSE},
        )
        print(
            "openenv_control_failure_submitted: "
            f"session_id={session_id} done={result.get('done')}"
        )

    except Exception as error:
        print(
            "openenv_control_failure_failed: "
            f"session_id={session_id} error={error}"
        )

    try:
        client.close_session()
        print(f"openenv_session_closed: {session_id}")

    except Exception as error:
        print(f"openenv_session_close_failed: session_id={session_id} error={error}")


def _write_agent_model_connection(agent_name: str,
                                  output_dir: Path) -> Path | None:
    connection = resolve_agent_model_connection(
        agent_name=agent_name,
        registry_path=REGISTRY_PATH,
    )

    if connection is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_connection.json"
    output_path.write_text(
        json.dumps(connection, indent=2),
        encoding="utf-8",
    )
    output_path.chmod(0o600)

    print(
        "agent_model_connection: "
        f"clem_model={connection.get('clem_model')} "
        f"backend={connection.get('backend')} "
        f"runtime_model={connection.get('model')}"
    )

    return output_path