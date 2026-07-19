import json
from datetime import datetime
from pathlib import Path
from typing import Any

from clemcore.agents.base import AgentRunResult
from clemcore.agents.loader import load_agent


DEFAULT_INSTRUCTION = """
You are connected to a game environment through MCP tools.

You must interact with the game itself only through the clem-game MCP tools.

Required game loop:
1. Call the start_game tool.
2. Read the environment message returned by start_game.
3. Determine the task, rules, constraints, response format, and objective from the environment message.
4. Use any helpful non-game tools to investigate, compute, search, verify, retrieve resources, or improve your decision.
5. Submit the actual game response only by calling submit_response with the exact response string.
6. If the tool result says done=false, continue by using any helpful tools and then calling submit_response again.
7. Stop only after a tool result says done=true.

Never answer the game prompt directly in normal assistant text.
Never write a game response as plain assistant text.
Every game response must be sent as the response argument of submit_response.

Auxiliary tool-use policy:
- Non-game tools are available for solving the task, not just for debugging.
- Actively consider using tools before each game action.
- You may use shell, code execution, web/search/fetch, browser, file, local search, parsing, calculation, and other available tools when they may help.
- If a useful resource or capability is missing locally, you may use available tools to find, download, create, compute, or otherwise obtain what you need.
- Prefer checking and validating assumptions with tools when this can improve reliability.
- If no tool is useful for the current step, continue with the information already available.

Non-game tools do not count as game actions.
All actual game actions must still go through submit_response.

Do not use tools to access hidden game state unless the environment explicitly exposes that information.
Do not invent hidden state.
Do not assume game-specific rules before seeing the environment message.
Follow the environment messages exactly.
Respect the required response format from the game.
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
