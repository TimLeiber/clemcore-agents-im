import requests
from typing import Any, Optional

import json
from pathlib import Path

from clemcore.agents.adapters.base import AgentRunResult, ExternalAgentHarness


class ExternalMCPHarness(ExternalAgentHarness):
    """
    Minimal external harness that controls a clembench game through MCP tools.

    This is intentionally simple:
    - it talks to the OpenEnv /mcp endpoint via JSON-RPC
    - it creates one persistent OpenEnv session
    - it calls start_game(), submit_response(...), and get_state()
    """

    def __init__(self, mcp_url: str):
        self.mcp_url = mcp_url
        self.session_id: Optional[str] = None
        self._request_id = 0

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _request(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
            "params": params or {},
        }

        response = requests.post(self.mcp_url, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        return data["result"]

    def create_session(self) -> str:
        result = self._request("openenv/session/create")
        self.session_id = result["session_id"]
        return self.session_id

    def close_session(self) -> None:
        if self.session_id is None:
            return

        self._request(
            "openenv/session/close",
            {
                "session_id": self.session_id,
            },
        )
        self.session_id = None

    def call_tool(self, name: str, arguments: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if self.session_id is None:
            raise RuntimeError("No active session. Call create_session() first.")

        result = self._request(
            "tools/call",
            {
                "session_id": self.session_id,
                "name": name,
                "arguments": arguments or {},
            },
        )

        return result["data"]

    def start_game(self) -> dict[str, Any]:
        return self.call_tool("start_game")

    def submit_response(self, response: str) -> dict[str, Any]:
        return self.call_tool(
            "submit_response",
            {
                "response": response,
            },
        )

    def get_state(self) -> dict[str, Any]:
        return self.call_tool("get_state")

    def choose_response(self, observation: dict[str, Any]) -> str:
        """
        Placeholder decision policy.

        Later this is where Claude Desktop, Claude Code, Codex, or another
        external harness would be called.
        """
        content = observation["context"]["content"]

        if "GUESS:" in content:
            return "CLUE: opposite of present"

        return "CLUE: not nearby"

    def run_episode(self,
                    instruction: str = "",
                    output_dir: Path | str | None = None,
                    max_steps: int = 10) -> AgentRunResult:
        if self.session_id is None:
            self.create_session()

        trajectory = []

        observation = self.start_game()
        trajectory.append(
            {
                "type": "start",
                "observation": observation,
            }
        )

        for step_idx in range(max_steps):
            if observation.get("done"):
                break

            response = self.choose_response(observation)
            next_observation = self.submit_response(response)

            trajectory.append(
                {
                    "type": "step",
                    "step": step_idx,
                    "response": response,
                    "observation": next_observation,
                }
            )

            observation = next_observation

            if observation.get("done"):
                break

        success = bool(observation.get("done"))

        artifacts = {}
        metadata = {
            "adapter": "manual_mcp",
            "session_id": self.session_id,
            "steps": len(trajectory),
            "success": success,
            "final_observation": observation,
        }

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            trajectory_path = output_dir / "trajectory.json"
            trajectory_path.write_text(
                json.dumps(trajectory, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            artifacts["trajectory"] = trajectory_path

            adapter_result_path = output_dir / "adapter_run_result.json"
            adapter_result_path.write_text(
                json.dumps(
                    {
                        "success": success,
                        "metadata": metadata,
                        "artifacts": {
                            key: str(value)
                            for key, value in artifacts.items()
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            artifacts["adapter_run_result"] = adapter_result_path

        return AgentRunResult(
            success=success,
            artifacts=artifacts,
            metadata=metadata,
        )
