from pathlib import Path

from clemcore.agents.mcp.server import run_clem_mcp_server


CLEMCORE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = CLEMCORE_ROOT / "docker" / "agent-sandbox" / "agent_registry.json"


if __name__ == "__main__":
    run_clem_mcp_server(
        game_name="taboo",
        agent_name="claude-code-sonnet",
        registry_path=DEFAULT_REGISTRY_PATH,
        learner_agent="player_0",
        env_agents={
            "player_1": "mock",
        },
        game_instance_split=None,
        single_pass=False,
        results_dir="results/external-agents",
        port=8001,
    )
