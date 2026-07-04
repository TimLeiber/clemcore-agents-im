import asyncio
from typing import Any
from claude_agent_sdk import ClaudeAgentOptions, query

from clemcore.agents.base import AgentRunResult, ExternalAgentHarness


class ClaudeCodeHarness(ExternalAgentHarness):
    """
    Claude Code-style harness accessed through the Claude Agent SDK.

    The SDK owns the agent loop. This class only configures the MCP server,
    sends the instruction, and collects emitted messages.
    """

    def __init__(self,
                 model: str | None = None,
                 mcp_url: str = "http://localhost:8001/mcp",
                 max_turns: int = 20,
                 allowed_tools: list[str] | None = None,
                 permission_mode: str = "bypassPermissions"):
        self.model = model
        self.mcp_url = mcp_url
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools or ["mcp__clem-game__*"]
        self.permission_mode = permission_mode

    async def run_episode_async(self,
                                instruction: str) -> list[Any]:
        options_kwargs = {
            "mcp_servers": {
                "clem-game": {
                    "type": "stdio",
                    "command": "python",
                    "args": [
                        "-m",
                        "clemcore.agents.mcp.bridge_server",
                    ],
                }
            },
            "allowed_tools": self.allowed_tools,
            "max_turns": self.max_turns,
            "permission_mode": self.permission_mode,
        }

        if self.model:
            options_kwargs["model"] = self.model

        options = ClaudeAgentOptions(**options_kwargs)

        messages = []

        async for message in query(prompt=instruction,
                                   options=options):
            messages.append(message)
            print(message)

        return messages

    def run_episode(self,
                    instruction: str,
                    output_dir=None) -> AgentRunResult:
        messages = asyncio.run(self.run_episode_async(instruction))

        metadata = self._extract_metadata(messages)
        artifacts = {}

        if output_dir is not None:
            from pathlib import Path

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            messages_path = output_dir / "adapter_messages.txt"
            messages_path.write_text(
                "\n".join(repr(message) for message in messages),
                encoding="utf-8",
            )
            artifacts["adapter_messages"] = messages_path

        return AgentRunResult(
            success=metadata["success"],
            artifacts=artifacts,
            metadata=metadata,
        )

    def _extract_metadata(self,
                          messages: list[Any]) -> dict[str, Any]:
        metadata = {
            "adapter": "claude_code",
            "model": self.model,
            "success": False,
            "session_id": None,
            "duration_ms": None,
            "total_cost_usd": None,
            "num_turns": None,
            "stop_reason": None,
        }

        for message in messages:
            session_id = getattr(message, "session_id", None)

            if session_id is not None:
                metadata["session_id"] = session_id

            subtype = getattr(message, "subtype", None)

            if subtype == "success":
                metadata["success"] = True

            for field_name in [
                "duration_ms",
                "total_cost_usd",
                "num_turns",
                "stop_reason",
            ]:
                value = getattr(message, field_name, None)

                if value is not None:
                    metadata[field_name] = value

        return metadata

        return messages