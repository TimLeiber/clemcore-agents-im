import json
from pathlib import Path
from typing import Any

from clemcore.agents.adapters.claude_code import ClaudeCodeHarness
from clemcore.agents.adapters.manual_mcp import ExternalMCPHarness

# dictionary storing all 'adapters' to the specific agent harnesses which are currently supported
BACKENDS = {
    "manual_mcp": ExternalMCPHarness,
    "claude_code": ClaudeCodeHarness,
}


def load_agent(agent_name: str,
               registry_path: str | Path) -> Any:
    registry_path = Path(registry_path).expanduser()

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    matches = [entry for entry in registry if entry["agent_name"] == agent_name]

    if not matches:
        known_agents = [entry["agent_name"] for entry in registry]
        raise ValueError(f"Unknown agent '{agent_name}'. Known agents: {known_agents}")

    spec = matches[0]
    backend_name = spec["backend"]

    if backend_name not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend_name}' for agent '{agent_name}'.")

    backend_cls = BACKENDS[backend_name]
    config = spec.get("agent_config", {})

    return backend_cls(**config)