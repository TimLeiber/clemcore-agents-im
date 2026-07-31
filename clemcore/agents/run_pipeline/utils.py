import json
import socket
import subprocess
import time
import os
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path

from clemcore.clemgame.instances import GameInstances
from clemcore.clemgame.registry import GameRegistry

from clemcore.agents.mcp.server import run_clem_mcp_server
from clemcore.agents.adapters.model_connection import resolve_agent_model_connection
from clemcore.agents.mcp.bridge import OpenEnvMCPClient

CLEMCORE_ROOT = Path(__file__).resolve().parents[3]
SANDBOX_DIR = CLEMCORE_ROOT / "clemcore" / "docker" / "agent-sandbox"
# paths required by the external-agent pipeline
REGISTRY_PATH = Path("agent_registry.json").resolve()
KEYS_PATH = Path("key.json").resolve()
DOCKER_IMAGE = "clem-agent-sandbox:dev"
# By convention I set 8001 to be PORT of the MCP server
SERVER_PORT = 8001
# define variables dependent on port
OPENENV_MCP_URL = f"http://host.docker.internal:{SERVER_PORT}/mcp"
HOST_OPENENV_MCP_URL = f"http://127.0.0.1:{SERVER_PORT}/mcp"
# Debug message
CONTROL_FAILURE_RESPONSE = (
    "CLEM_AGENT_CONTROL_ERROR: external harness ended before completing the game"
)


# ------------------------- MAIN FUNCTIONS RUNNING run_pipeline.py -------------------------


def load_game_instances(game_name: str,
                        instances_filename: str | None = None,
                        experiment_name: str | None = None) -> GameInstances:

    """Load game instances and optionally restrict them to one experiment.

    Args:
        game_name: Name of the game whose instances should be loaded.
        instances_filename: Optional alternative instances filename.
        experiment_name: Optional experiment name used to filter the instances.

    Returns:
        The loaded and optionally filtered game instances.
    """

    # load the game definition from the clembench registry.
    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_spec = game_registry.get_game_spec(game_name)

    # use a custom instances file when one was provided.
    if instances_filename:
        game_spec.instances = instances_filename

    # load all instances defined by the game specification.
    game_instances = GameInstances.from_game_spec(game_spec)

    # Restrict the run to one experiment when requested.
    if experiment_name:
        game_instances = game_instances.filter(
            lambda row: row["experiment"]["name"] == experiment_name
        )

    return game_instances


def start_server(game_name: str,
                 agent_name: str,
                 agent_player: str,
                 env_agent_models: list[str],
                 instances_filename: str | None,
                 results_dir: str) -> Process:

    """Start the clembench MCP server and wait until it accepts connections

    Args:
        game_name: Name of the game served by the MCP server
        agent_name: Name of the external agent
        agent_player: Player slot controlled by the external agent
        env_agent_models: Native models assigned to the remaining player slots
        instances_filename: Optional alternative game instances filename
        results_dir: Root directory where clembench writes results

    Returns:
        The running MCP server process
    """

    # ensure that another server is not already using the configured port
    # should not happen anymore after pipeline automatically closes server
    try:
        with socket.create_connection(("127.0.0.1", SERVER_PORT), timeout=1):
            raise RuntimeError(
                f"Port {SERVER_PORT} is already in use. "
                "Stop the existing MCP server before running the pipeline."
            )
    except OSError:
        pass

    # start the MCP server in a separate process
    process = Process(
        target=run_clem_mcp_server,
        kwargs={
            "game_name": game_name,
            "agent_name": agent_name,
            "registry_path": REGISTRY_PATH,
            "learner_agent": agent_player,
            "env_agents": _env_agents_from_models(models=env_agent_models,
                                                  learner_agent=agent_player,
                                                  game_name=game_name),
            "game_instance_split": None,
            "instances_filename": instances_filename,
            "single_pass": False,
            "results_dir": results_dir,
            "port": SERVER_PORT,
        },
    )
    # start the process
    process.start()

    # wait until the server accepts connections.
    deadline = time.monotonic() + 30

    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError("MCP server process exited during startup.")

        try:
            with socket.create_connection(
                ("127.0.0.1", SERVER_PORT),
                timeout=1,
            ):
                return process
        except OSError:
            time.sleep(0.25)

    # Stop the process when startup fails
    process.terminate()
    process.join(timeout=5)

    raise TimeoutError(
        f"MCP server did not become available on port {SERVER_PORT}."
    )


def run_docker_episode(experiment_name: str,
                        game_id: int | str,
                        agent_name: str,
                        model_connection_path: Path | None = None,
                        shared_state_dir: Path | None = None) -> tuple[str, int]:
    """Run one external-agent episode inside the Docker sandbox.

        Args:
            experiment_name: Name of the experiment containing the game instance.
            game_id: Identifier of the game instance to run.
            agent_name: Name of the external agent from the agent registry.
            model_connection_path: Optional path to the resolved model connection file.
            shared_state_dir: Optional directory shared between the host and container.

        Returns:
            The complete container output and its successful return code.
    """

    # verify that the files required by the container exist
    if not KEYS_PATH.exists():
        raise FileNotFoundError(f"Missing credentials file: {KEYS_PATH}")

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing agent registry: {REGISTRY_PATH}")

    # load the credentials used by external-agent command line tools
    try:
        keys = json.loads(
            KEYS_PATH.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid credentials file: {KEYS_PATH}"
        ) from error

    if not isinstance(keys, dict):
        raise ValueError(
            f"Credentials file must contain a JSON object: {KEYS_PATH}"
        )

    container_environment = {
        "OPENAI_API_KEY": keys.get("openai", {}).get("api_key"),
        "ANTHROPIC_API_KEY": keys.get("anthropic", {}).get("api_key"),
    }

    container_environment = {
        name: value
        for name, value in container_environment.items()
        if value
    }

    # build the base Docker command and provide the episode information.
    command = [
        "docker", "run", "--rm", "-i",
        "-e", f"OPENENV_MCP_URL={OPENENV_MCP_URL}",
        "-e", f"CLEM_EXPERIMENT_NAME={experiment_name}",
        "-e", f"CLEM_GAME_ID={game_id}",
        "-e", f"CLEM_AGENT_NAME={agent_name}",
        "-v", f"{CLEMCORE_ROOT}:/opt/clemcore:ro",
        "-v", f"{SANDBOX_DIR}:/app:ro",
        "-v", f"{REGISTRY_PATH}:/tmp/agent_registry.json:ro",
    ]

    # forward credentials without writing their values into the Docker command
    for name in container_environment:
        command.extend(["-e", name])

    # mount the writable directory used to exchange runtime state
    if shared_state_dir is not None:
        command.extend([
            "-e", "CLEM_OPENENV_SESSION_PATH=/run/clem-agent/openenv_session.json",
            "-v", f"{shared_state_dir}:/run/clem-agent:rw",
        ])

    # tell the container where to find the resolved model connection
    if model_connection_path is not None:
        command.extend([
            "-e", "CLEM_AGENT_MODEL_CONNECTION_PATH=/run/clem-agent/model_connection.json",
        ])

        #  mount the model connection directly when no shared directory is used
        if shared_state_dir is None:
            command.extend([
                "-v", f"{model_connection_path}:/run/clem-agent/model_connection.json:ro",
            ])

    # run the container-side python entrypoint, i.e. run_agent_container
    command.extend([
        DOCKER_IMAGE,
        "python", "/app/run_agent_container.py",
    ])

    # collect the container output for the trace
    trace_lines: list[str] = []

    try:
        process_environment = {
            **os.environ,
            **container_environment,
        }
        # start the container and combine stdout and stderr
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1,
                                   env=process_environment)

        assert process.stdout is not None

        # print and collect the container output
        for line in process.stdout:
            print(line, end="")
            trace_lines.append(line)

        return_code = process.wait()
        trace_text = "".join(trace_lines)

        # treat a nonzero container exit as a failed episode
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code,
                command,
                output=trace_text,
            )

        return trace_text, return_code

    finally:
        # recover an unfinished openenv session
        if shared_state_dir is not None:
            session_path = shared_state_dir / "openenv_session.json"

            if session_path.exists():
                try:
                    session_data = json.loads(session_path.read_text(encoding="utf-8"))
                    session_id = session_data.get("session_id")
                    trace_text = "".join(trace_lines)

                    completed = any(
                        marker in trace_text
                        for marker in ('"done":true', '"done": true', "'done': True")
                    )

                    if session_id and not completed:
                        client = OpenEnvMCPClient(HOST_OPENENV_MCP_URL)
                        client.session_id = session_id

                        try:
                            result = client.call_tool("submit_response",{"response": CONTROL_FAILURE_RESPONSE})
                            print(f"openenv_control_failure_submitted: "
                                  f"session_id={session_id} done={result.get('done')}")
                        except Exception as error:
                            print(f"openenv_control_failure_failed: "
                                  f"session_id={session_id} error={error}")

                        try:
                            client.close_session()
                            print(f"openenv_session_closed: {session_id}")
                        except Exception as error:
                            print(f"openenv_session_close_failed: "
                                  f"session_id={session_id} error={error}")

                except (OSError, json.JSONDecodeError) as error:
                    print(f"openenv_session_file_invalid: path={session_path} error={error}")
                except Exception as error:
                    print(f"openenv_session_cleanup_failed: path={session_path} error={error}")


def write_agent_trace(results_dir: str,
                      run_dir: str,
                      game_name: str,
                      experiment_name: str,
                      game_id: int | str,
                      before_episode_dirs: set[Path],
                      trace_text: str,
                      metadata: dict) -> Path:
    """Write the agent trace into the corresponding clembench episode directory.

    Args:
        results_dir: Root directory containing the clembench results.
        run_dir: Exact dialogue-pair directory used by the MCP server.
        game_name: Name of the game that was run.
        experiment_name: Name of the experiment containing the episode.
        game_id: Identifier of the game instance that was run.
        before_episode_dirs: Episode directories that existed before the run.
        trace_text: Raw output captured from the agent container.
        metadata: Additional information to store alongside the trace.

    Returns:
        Path to the written agent trace file.
    """

    # find all episode directories that exist after the run
    results_path = Path(results_dir)
    after_episode_dirs = {path for path in results_path.glob(f"{run_dir}/{game_name}/{experiment_name}/episode_*")
                          if path.is_dir()}

    try:
        run_started_timestamp = datetime.fromisoformat(
            str(metadata["started_at"])
        ).timestamp()
    except (KeyError, TypeError, ValueError):
        # If the caller cannot establish the run boundary, prefer the failure
        # tree over risking an overwrite of an unrelated historical episode.
        run_started_timestamp = datetime.now(timezone.utc).timestamp()

    # record the modification time of every episode directory
    episode_timestamps = {}

    for episode_dir in after_episode_dirs:
        timestamp_candidates = [
            episode_dir / "interactions.json",
            episode_dir / "instance.json",
            episode_dir,
        ]

        episode_timestamps[episode_dir] = max(path.stat().st_mtime for path in timestamp_candidates if path.exists())

    # first try to identify an episode directory created by this run
    new_episode_dirs = after_episode_dirs - before_episode_dirs

    if new_episode_dirs:
        output_dir = max(new_episode_dirs, key=episode_timestamps.get)

    else:
        # Only reuse an existing episode directory if this run actually
        # modified it. A failed agent may create no episode at all; selecting
        # an old directory by game_id alone can overwrite another agent's
        # trace because all result trees reuse the same episode numbers.
        matching_episode_dirs = []

        for episode_dir in after_episode_dirs:
            instance_path = episode_dir / "instance.json"

            if not instance_path.exists():
                continue

            try:
                instance = json.loads(
                    instance_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                continue

            if (int(instance.get("game_id")) == int(game_id)
                    and episode_timestamps[episode_dir] >= run_started_timestamp):
                matching_episode_dirs.append(episode_dir)

        if matching_episode_dirs:
            output_dir = max(matching_episode_dirs, key=episode_timestamps.get)

        else:
            # preserve the trace separately when clembench created no episode
            output_dir = (
                results_path
                / "_agent_failures"
                / run_dir
                / game_name
                / experiment_name
                / f"game_id_{game_id}"
                / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            )
            output_dir.mkdir(parents=True, exist_ok=True)

    # write the raw trace and its metadata into the selected directory
    trace_path = output_dir / "agent_trace.log"
    trace_path.write_text(trace_text, encoding="utf-8")

    metadata_path = output_dir / "agent_trace_meta.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return trace_path



# ------------------------- HELPER FUNCTIONS -------------------------

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
    """Assign native models to all players except the external agent.

    Args:
        models: Native clembench models supplied for non-agent players.
        learner_agent: Player slot controlled by the external agent.
        game_name: Optional game name used to determine the declared player count.

    Returns:
        A mapping from non-agent player IDs to native model names.

    Raises:
        ValueError: If the external-agent player slot is invalid or too few
            native models were supplied.
    """

    # read player IDs from the game metadata when available
    player_ids = _player_ids_from_clemgame(game_name) if game_name else None

    # fall back to inferring one non-agent player per supplied model
    if player_ids is None:
        player_ids = [f"player_{index}" for index in range(len(models) + 1)]

    if learner_agent not in player_ids:
        raise ValueError(
            f"Cannot use learner_agent={learner_agent!r}. "
            f"Valid choices are: {player_ids}"
        )

    # Remove the player controlled by the external agent
    env_player_ids = [player_id for player_id in player_ids if player_id != learner_agent]

    if len(models) < len(env_player_ids):
        raise ValueError(
            f"Need {len(env_player_ids)} native model(s) for "
            f"{env_player_ids}, but got {len(models)}: {models}"
        )

    return dict(zip(env_player_ids, models))

def _write_agent_model_connection(agent_name: str,
                                  output_dir: Path) -> Path | None:
    """Resolve and write the model connection required by an external agent.

    Args:
        agent_name: name of the external agent from the agent registry
        output_dir: directory where the connection file should be written

    Returns:
        path to the written model connection file, or None when the agent
        does not use a clemcore model connection
    """

    # resolve the clemcore model used by the external agent
    connection = resolve_agent_model_connection(agent_name=agent_name, registry_path=REGISTRY_PATH)

    # agents without an associated clemcore model need no connection file
    if connection is None:
        return None

    # write the connection into the temporary directory shared with the container
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_connection.json"
    output_path.write_text(json.dumps(connection, indent=2), encoding="utf-8")

    # restrict access because the connection may contain credentials
    output_path.chmod(0o600)

    # print the resolved connection without exposing sensitive values
    print("agent_model_connection: "
          f"clem_model={connection.get('clem_model')} "
          f"backend={connection.get('backend')} "
          f"runtime_model={connection.get('model')}")

    return output_path
