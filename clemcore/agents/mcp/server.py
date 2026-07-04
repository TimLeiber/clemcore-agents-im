import json
from pathlib import Path

import uvicorn

from clemcore.agents.mcp.app import create_clem_mcp_app


def _external_player_name(agent_name: str,
                          registry_path: str | Path) -> str:
    registry_path = Path(registry_path).expanduser()

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    matches = [entry for entry in registry if entry["agent_name"] == agent_name]

    if not matches:
        known_agents = [entry["agent_name"] for entry in registry]
        raise ValueError(f"Unknown agent '{agent_name}'. Known agents: {known_agents}")

    spec = matches[0]
    model_name = spec.get("agent_config", {}).get("model")

    if model_name:
        return f"{agent_name}_{model_name}"

    return agent_name


def _player_sort_key(agent_id: str) -> tuple[int, int | str]:
    prefix = "player_"

    if agent_id.startswith(prefix):
        suffix = agent_id[len(prefix):]

        if suffix.isdigit():
            return 0, int(suffix)

    return 1, agent_id


def _dialogue_pair_name(agent_name: str,
                        registry_path: str | Path,
                        learner_agent: str,
                        env_agents: dict[str, str] | None) -> str:
    external_name = _external_player_name(agent_name, registry_path)

    player_names = dict(env_agents or {})
    player_names[learner_agent] = external_name

    ordered_names = [
        player_names[player_id]
        for player_id in sorted(player_names, key=_player_sort_key)
    ]

    return "--".join(ordered_names)


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
    if run_dir is None:
        run_dir = _dialogue_pair_name(
            agent_name=agent_name,
            registry_path=registry_path,
            learner_agent=learner_agent,
            env_agents=env_agents,
        )

    app = create_clem_mcp_app(
        game_name=game_name,
        learner_agent=learner_agent,
        env_agents=env_agents,
        game_instance_split=game_instance_split,
        instances_filename=instances_filename,
        single_pass=single_pass,
        gen_args=gen_args,
        results_dir=results_dir,
        run_dir=run_dir,
    )

    uvicorn.run(app,
                host="0.0.0.0",
                port=port)
