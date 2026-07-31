import json
from pathlib import Path
import uvicorn

from openenv.core import create_app

from clemcore.clemgame import episode_results_folder_callbacks
from clemcore.clemgame.envs.openenv.models import ClemGameAction, ClemGameObservation
from clemcore.agents.mcp.environment import ClemGameMCPEnvironment, SelectableClemGameEnvironment


def result_run_dir_name(agent_name: str,
                        registry_path: str | Path,
                        learner_agent: str = "player_0",
                        env_agents: dict[str, str] | None = None) -> str:
    """Return the deterministic clembench result directory for a player set."""
    registry_path = Path(registry_path).expanduser()
    with open(registry_path, "r", encoding="utf-8") as file:
        registry = json.load(file)

    matches = [entry for entry in registry if entry["agent_name"] == agent_name]
    if not matches:
        known_agents = [entry["agent_name"] for entry in registry]
        raise ValueError(
            f"Unknown agent '{agent_name}'. Known agents: {known_agents}"
        )

    spec = matches[0]
    model_name = spec.get("agent_config", {}).get("model")
    external_name = f"{agent_name}_{model_name}" if model_name else agent_name
    player_names = dict(env_agents or {})
    player_names[learner_agent] = external_name
    ordered_players = []

    for player_id, player_name in player_names.items():
        prefix = "player_"
        suffix = player_id[len(prefix):] if player_id.startswith(prefix) else ""
        sort_key = (0, int(suffix)) if suffix.isdigit() else (1, player_id)
        ordered_players.append((sort_key, player_name))

    ordered_players.sort(key=lambda item: item[0])
    return "--".join(player_name for _, player_name in ordered_players)


def run_clem_mcp_server(game_name: str,
                        agent_name: str,
                        registry_path: str | Path,
                        learner_agent: str = "player_0",
                        env_agents: dict[str, str] | None = None,
                        game_instance_split: str | None = None,
                        instances_filename: str | None = None,
                        single_pass: bool = False,
                        gen_args: dict | None = None,
                        results_dir: str | Path | None = None,
                        run_dir: str | None = None,
                        port: int = 8001) -> None:
    """Create and run the host-side MCP server for a clembench run.

    The server wraps the selected clembench game as an OpenEnv environment and
    exposes it through MCP. External agents do not access the game directly.
    Instead, the container-side bridge server connects to this host process and
    forwards MCP tool calls such as start_game, submit_response, and get_state to
    the underlying OpenEnv session.

    Args:
        game_name: name of the clembench game to serve
        agent_name: name of the external agent from the agent registry
        registry_path: path to the external-agent registry
        learner_agent: player slot controlled by the external agent
        env_agents: native models assigned to the remaining player slots
        game_instance_split: optional game-instance split to serve
        instances_filename: optional alternative instances filename
        single_pass: whether to stop after every selected instance was served
        gen_args: optional generation arguments for native models
        results_dir: root directory where clembench writes results
        run_dir: optional explicit name for the dialogue-pair result directory
        port: port used by the MCP server
        """

    # ----- step 1 -----
    # derive the result directory name from the configured players when omitted
    if run_dir is None:
        run_dir = result_run_dir_name(
            agent_name=agent_name,
            registry_path=registry_path,
            learner_agent=learner_agent,
            env_agents=env_agents,
        )

        # ----- step 2 -----
        # collect the callbacks that make clembench write episode records to disk
        callbacks = None

        if results_dir is not None:
            callbacks = episode_results_folder_callbacks(run_dir=run_dir,
                                                         result_dir_path=results_dir,
                                                         player_model_infos=None)

        # ----- step 3 -----
        # define how OpenEnv builds an environment, called once per agent session
        def make_env():
            base_env = SelectableClemGameEnvironment(game_name,
                                                     instances_filename=instances_filename,
                                                     game_instance_split=game_instance_split,
                                                     single_pass=single_pass,
                                                     learner_agent=learner_agent,
                                                     env_agents=env_agents,
                                                     gen_args=gen_args,
                                                     callbacks=callbacks)

            return ClemGameMCPEnvironment(base_env)

        # ----- step 4 -----
        # build the application exposing that environment over MCP
        app = create_app(make_env,
                         ClemGameAction,
                         ClemGameObservation,
                         env_name="clem_mcp_env")

        # expose the application to the host and Docker container
        uvicorn.run(app, host="0.0.0.0", port=port)
