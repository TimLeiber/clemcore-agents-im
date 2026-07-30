import json
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from clemcore.agents import adapters
from clemcore.agents.adapters.base import AgentRunResult
from clemcore.agents.adapters.claude_code import ClaudeCodeHarness
from clemcore.agents.adapters.codex import CodexHarness
from clemcore.agents.adapters.hermes import HermesHarness
from clemcore.agents.adapters.manual_mcp import ExternalMCPHarness
from clemcore.agents.adapters.openclaw import OpenClawHarness

# adapters currently supported by the external-agent runner
BACKENDS = {"manual_mcp": ExternalMCPHarness,
            "claude_code": ClaudeCodeHarness,
            "codex": CodexHarness,
            "hermes": HermesHarness,
            "openclaw": OpenClawHarness}
CONFIG_PATH = (Path(adapters.__file__).resolve().parent / "external_agent_config.yaml")


def run_external_agent_episode(agent_name: str,
                               registry_path: str | Path,
                               output_root: str | Path | None,
                               instruction: str | None = None,
                               run_metadata: dict[str, Any] | None = None) -> AgentRunResult:
    """Run one episode with an external agent.

    Args:
        agent_name: name of the agent in the agent registry
        registry_path: path to the external-agent registry
        output_root: directory for run artifacts or None to disable output
        instruction: instruction passed to the agent; loads the shared
            meta prompt when omitted
        run_metadata: optional metadata included in the run summary

    Returns:
        the result returned by the external-agent adapter
    """

    registry_path = Path(registry_path).expanduser()
    output_dir = None

    # load the shared meta prompt when no instruction was provided
    if instruction is None:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Missing external-agent configuration: {CONFIG_PATH}"
            )

        config = yaml.safe_load(
            CONFIG_PATH.read_text(encoding="utf-8")
        )

        if not isinstance(config, dict):
            raise ValueError(
                "External-agent configuration must be a mapping: "
                f"{CONFIG_PATH}"
            )

        instruction = config.get("meta_prompt")

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                "Missing non-empty meta_prompt in configuration: "
                f"{CONFIG_PATH}"
            )

    # create a timestamped output directory when output is enabled
    if output_root is not None:
        output_root = Path(output_root).expanduser()
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / agent_name / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

    # load the requested agent specification from the registry
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [entry for entry in registry if entry["agent_name"] == agent_name]

    if not matches:
        known_agents = [entry["agent_name"] for entry in registry]
        raise ValueError(
            f"Unknown agent '{agent_name}'. Known agents: {known_agents}"
        )

    spec = matches[0]
    backend_name = spec["backend"]

    if backend_name not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend_name}' for agent '{agent_name}'."
        )

    # load the actual agent and store it in this variable
    agent = BACKENDS[backend_name](**spec.get("agent_config", {}))
    result = agent.run_episode(instruction=instruction, output_dir=output_dir)

    # write a machine-readable summary alongside the agent artifacts
    if output_dir is not None:
        summary = {
            "agent_name": agent_name,
            "registry_path": str(registry_path),
            "output_dir": str(output_dir),
            "success": result.success,
            "metadata": result.metadata,
            "artifacts": {key: str(value) for key, value in result.artifacts.items()},
            "run_metadata": run_metadata or {},
        }

        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        result.artifacts["run_summary"] = summary_path

        print(f"Wrote run summary to {summary_path}")

    return result


if __name__ == "__main__":
    result = run_external_agent_episode(
        agent_name=os.environ["CLEM_AGENT_NAME"],
        registry_path="/tmp/agent_registry.json",
        output_root=None,
    )

    print("success:", result.success)

    for name, path in result.artifacts.items():
        print(f"{name}: {path}")
