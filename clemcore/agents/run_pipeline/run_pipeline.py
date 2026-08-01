import argparse
import tempfile
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from clemcore.agents.mcp.server import result_run_dir_name

# import all functions from utils needed to run main
from .utils import (
    DOCKER_IMAGE, # Docker image used for agent episodes
    REGISTRY_PATH, # external-agent registry used by server and container
    _env_agents_from_models, # map native models to player slots
    _write_agent_model_connection, # write the agent model configuration

    load_game_instances, # load and filter game instances
    run_docker_episode, # run one agent episode in Docker
    start_server, # start the MCP server
    write_agent_trace # store the captured agent trace
)


def main() -> None:
    # define the command line interface
    parser = argparse.ArgumentParser(
        description="Run external agent harnesses through clembench/OpenEnv/MCP.",
    )

    # define the pipeline arguments
    parser.add_argument("-g",
                        "--game",
                        required=True,
                        help="Game name, matching clem run -g.")
    parser.add_argument("-a",
                        "--agent",
                        required=True,
                        help="External agent name from the agent registry.")
    parser.add_argument("-m",
                        "--models",
                        nargs="+",
                        default=[],
                        help="Native clem model(s) for non-external players, matching clem run -m style.")
    parser.add_argument("--agent-player",
                        default="player_0",
                        help="Player slot controlled by the external agent, e.g. player_0 or player_1.")
    parser.add_argument("-e",
                        "--experiment_name",
                        default=None,
                        help="Optional experiment filter, matching clem run -e.")
    parser.add_argument("-i",
                        "--instances_filename",
                        default=None,
                        help="Instances file name without .json, matching clem run -i.")
    parser.add_argument("-r",
                        "--results_dir",
                        default="results/external-agents",
                        help="Results root directory, matching clem run -r style.")
    parser.add_argument("--max-instances",
                        type=int,
                        default=None,
                        help="Optional debugging limit for number of selected instances to run.")

    # parse the command line arguments
    args = parser.parse_args()

    # map native models to non-agent player slots
    env_agents = _env_agents_from_models(
        models=args.models,
        learner_agent=args.agent_player,
        game_name=args.game,
    )

    # ----- main step 1 -----
    # load and filter the selected game instances
    game_instances = load_game_instances(
        game_name=args.game,
        instances_filename=args.instances_filename,
        experiment_name=args.experiment_name,
    )

    # ensure that at least one instance was selected
    total = len(game_instances)

    if total == 0:
        raise RuntimeError("No game instances selected.")

    # print the resolved pipeline configuration
    print(f"game: {args.game}")
    print(f"agent: {args.agent}")
    print(f"agent_player: {args.agent_player}")
    print(f"models: {args.models}")
    print(f"env_agents: {env_agents}")
    print(f"instances_filename: {args.instances_filename or 'instances'}")
    print(f"experiment_name: {args.experiment_name or '<all>'}")
    print(f"results_dir: {args.results_dir}")
    print(game_instances.describe())

    result_run_dir = result_run_dir_name(
        agent_name=args.agent,
        registry_path=REGISTRY_PATH,
        learner_agent=args.agent_player,
        env_agents=env_agents,
    )

    # create temporary shared storage for the agent runtime
    temp_dir = tempfile.TemporaryDirectory(prefix="clem-agent-model-")
    # resolve and write the external agent model connection
    model_connection_path = _write_agent_model_connection(
        agent_name=args.agent,
        output_dir=Path(temp_dir.name),
    )

    # ---- main step 2 -----
    # start the MCP game server, exposing game as tools for the agent
    server_process = start_server(
        game_name=args.game,
        agent_name=args.agent,
        agent_player=args.agent_player,
        env_agent_models=args.models,
        instances_filename=args.instances_filename,
        results_dir=args.results_dir,
    )

    # try block attempts to run external agent in docker container on each episode
    try:
        # apply the optional debugging instance limit
        selected_instances = game_instances

        if args.max_instances is not None:
            selected_instances = list(islice(game_instances, args.max_instances))
            total = len(selected_instances)

        # run each selected game instance
        for index, row in enumerate(selected_instances, start=1):
            experiment = row["experiment"]
            game_instance = row["game_instance"]

            experiment_name = experiment["name"]
            game_id = game_instance["game_id"]
            target = game_instance.get("target_word") or game_instance.get("target")

            print()
            print(f"[{index}/{total}] {experiment_name} / game_id={game_id} / target={target}")

            # record the episode directories that exist before this run
            before_episode_dirs = {path
                                   for path in Path(args.results_dir).glob(
                                       f"{result_run_dir}/{args.game}/{experiment_name}/episode_*"
                                   )
                                   if path.is_dir()}

            # record start time and remove session file from previous episode
            started_at = datetime.now(timezone.utc)
            openenv_session_path = Path(temp_dir.name) / "openenv_session.json"

            if openenv_session_path.exists():
                openenv_session_path.unlink()

            # ----- main step 3 -----
            # run the external agent inside Docker
            trace_text, return_code = run_docker_episode(
                experiment_name=experiment_name,
                game_id=game_id,
                agent_name=args.agent,
                model_connection_path=model_connection_path,
                shared_state_dir=Path(temp_dir.name),
            )
            finished_at = datetime.now(timezone.utc)

            # ----- main step 4 -----
            # write the captured trace and metadata
            trace_path = write_agent_trace(
                results_dir=args.results_dir,
                run_dir=result_run_dir,
                game_name=args.game,
                experiment_name=experiment_name,
                game_id=game_id,
                before_episode_dirs=before_episode_dirs,
                trace_text=trace_text,
                metadata={
                    "agent": args.agent,
                    "agent_player": args.agent_player,
                    "models": args.models,
                    "env_agents": env_agents,
                    "game": args.game,
                    "experiment_name": experiment_name,
                    "game_id": game_id,
                    "target": target,
                    "docker_image": DOCKER_IMAGE,
                    "return_code": return_code,
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                },
            )

            print(f"agent_trace: {trace_path}")


    finally:

        # stop the MCP server
        if server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=5)

        # fallback to kill server process if process fails to exit initially
        if server_process.is_alive():
            server_process.kill()
            server_process.join(timeout=5)

        temp_dir.cleanup()


if __name__ == "__main__":
    main()
