import json
from pathlib import Path

import uvicorn

from clemcore.agents.mcp.app import create_clem_mcp_app


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

    # derive the result directory name from the configured players when omitted
    if run_dir is None:
        registry_path = Path(registry_path).expanduser()

        with open(registry_path, "r", encoding="utf-8") as file:
            registry = json.load(file)

        matches = [
            entry
            for entry in registry
            if entry["agent_name"] == agent_name
        ]

        if not matches:
            known_agents = [entry["agent_name"] for entry in registry]
            raise ValueError(
                f"Unknown agent '{agent_name}'. Known agents: {known_agents}"
            )

        # include the external agent model in the result directory name
        spec = matches[0]
        model_name = spec.get("agent_config", {}).get("model")
        external_name = f"{agent_name}_{model_name}" if model_name else agent_name

        # collect the model or agent name assigned to every player slot
        player_names = dict(env_agents or {})
        player_names[learner_agent] = external_name

        # order conventional player_N slots numerically before other identifiers
        ordered_players = []

        for player_id, player_name in player_names.items():
            prefix = "player_"
            suffix = player_id[len(prefix):] if player_id.startswith(prefix) else ""

            if suffix.isdigit():
                sort_key = 0, int(suffix)
            else:
                sort_key = 1, player_id

            ordered_players.append((sort_key, player_name))

        ordered_players.sort(key=lambda item: item[0])
        run_dir = "--".join(player_name for _, player_name in ordered_players)

    # create the MCP application containing the selected clembench environment
    app = create_clem_mcp_app(game_name=game_name,
                              learner_agent=learner_agent,
                              env_agents=env_agents,
                              game_instance_split=game_instance_split,
                              instances_filename=instances_filename,
                              single_pass=single_pass,
                              gen_args=gen_args,
                              results_dir=results_dir,
                              run_dir=run_dir)

    # expose the application to the host and Docker container
    uvicorn.run(app, host="0.0.0.0", port=port)