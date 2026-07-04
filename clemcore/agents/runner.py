import json
from datetime import datetime
from pathlib import Path
from typing import Any

from clemcore.agents.base import AgentRunResult
from clemcore.agents.loader import load_agent


DEFAULT_INSTRUCTION = """
You are connected to a game environment through MCP tools.

Use the available tools to:
1. start the game,
2. read the instructions returned by the environment,
3. submit responses through the available response tool,
4. continue until the environment reports that the episode is done.

Do not assume any game-specific rules before seeing the environment message.
Do not invent hidden state.
Follow the environment's messages exactly.
Try to achieve the best possible score.
"""


def run_external_agent_episode(agent_name: str,
                               registry_path: str | Path,
                               output_root: str | Path | None,
                               instruction: str = DEFAULT_INSTRUCTION,
                               run_metadata: dict[str, Any] | None = None) -> AgentRunResult:
    registry_path = Path(registry_path).expanduser()
    output_dir = None

    if output_root is not None:
        output_root = Path(output_root).expanduser()

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / agent_name / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

    agent = load_agent(agent_name,
                       registry_path=registry_path)

    result = agent.run_episode(instruction=instruction,
                               output_dir=output_dir)

    if output_dir is not None:
        summary = {
            "agent_name": agent_name,
            "registry_path": str(registry_path),
            "output_dir": str(output_dir),
            "success": result.success,
            "metadata": result.metadata,
            "artifacts": {
                key: str(value)
                for key, value in result.artifacts.items()
            },
            "run_metadata": run_metadata or {},
        }

        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        result.artifacts["run_summary"] = summary_path

        print(f"Wrote run summary to {summary_path}")

    return result
