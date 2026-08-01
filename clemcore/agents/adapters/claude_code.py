import asyncio
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from clemcore.agents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemcore.agents.adapters.utils import (
    load_model_connection,
    model_connection_environment,
    resolve_runtime_model,
    temporary_environment,
    write_text_artifact,
)


class ClaudeCodeHarness(ExternalAgentHarness):
    """Run Claude Code through the Claude Agent SDK.

    The harness configures the container-side MCP bridge as a Claude Code
    server and collects every SDK message emitted during the episode.
    """

    def __init__(self,
                 model: str | None = None,
                 clem_model: str | None = None,
                 mcp_url: str = "http://localhost:8001/mcp",
                 max_turns: int = 20,
                 allowed_tools: list[str] | None = None,
                 permission_mode: str = "bypassPermissions",
                 model_connection_path: str | None = None):
        """Configure the Claude Code harness.

        Args:
            model: model identifier passed directly to Claude Code
            clem_model: clembench model resolved by the outer pipeline
            mcp_url: URL forwarded to the container-side MCP bridge
            max_turns: maximum number of Claude Code turns
            allowed_tools: tools Claude Code may call
            permission_mode: Claude Code tool-permission policy
            model_connection_path: optional resolved model-connection file
        """

        self.model = model or clem_model
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools or ["mcp__clem_game__*"]
        self.permission_mode = permission_mode
        self._model_connection = load_model_connection("claude_code", model_connection_path)

    async def run_episode_async(self,
                                instruction: str,
                                runtime_model: str | None,
                                runtime_environment: dict[str, str | None]) -> tuple[list[Any], str | None, bool]:
        """Collect the asynchronous Claude SDK message stream.

        This method is the asynchronous boundary required by the Claude Agent
        SDK while the shared harness interface remains synchronous.

        Args:
            instruction: task instruction passed to Claude Code
            runtime_model: resolved model identifier
            runtime_environment: temporary model-provider environment

        Returns:
            the collected SDK messages, an optional runtime error, and whether
            the game completed
        """

        options_kwargs = {
            "mcp_servers": {
                "clem_game": {
                    "type": "stdio",
                    "command": "python",
                    "args": ["-m", "clemcore.agents.mcp.bridge"],
                },
            },
            "allowed_tools": self.allowed_tools,
            "max_turns": self.max_turns,
            "permission_mode": self.permission_mode,
        }

        if runtime_model:
            options_kwargs["model"] = runtime_model

        options = ClaudeAgentOptions(**options_kwargs)
        messages = []
        runtime_error = None
        game_completed = False

        with temporary_environment(runtime_environment):
            try:
                async for message in query(prompt=instruction, options=options):
                    messages.append(message)
                    print(message)

                    tool_use_result = getattr(message, "tool_use_result", None)
                    structured_content = (tool_use_result.get("structuredContent")
                                          if isinstance(tool_use_result, dict) else None)

                    if isinstance(structured_content, dict) and structured_content.get("done") is True:
                        game_completed = True
                        break
            except Exception as error:
                runtime_error = str(error)
                print(f"agent_runtime_error: {error}")

        return messages, runtime_error, game_completed

    def run_episode(self,
                    instruction: str,
                    output_dir: Path | str | None = None) -> AgentRunResult:
        """Run one Claude Code episode.

        Args:
            instruction: task instruction passed to Claude Code
            output_dir: optional directory for adapter artifacts

        Returns:
            the standardized Claude Code run result
        """

        # ----- step 1 -----
        # resolve the model and provider environment
        runtime_model = resolve_runtime_model(model_connection=self._model_connection,
                                              model=self.model,
                                              harness_name="ClaudeCodeHarness",
                                              required=False)
        runtime_environment = model_connection_environment(self._model_connection)

        # ----- step 2 -----
        # run Claude Code and collect the SDK message stream
        messages, runtime_error, game_completed = asyncio.run(
            self.run_episode_async(instruction=instruction,
                                   runtime_model=runtime_model,
                                   runtime_environment=runtime_environment)
        )

        # ----- step 3 -----
        # extract standardized metadata from the SDK messages
        metadata = {
            "adapter": "claude_code",
            "model": self.model,
            "clem_model": self.clem_model,
            "runtime_model": runtime_model,
            "resolved_backend": (self._model_connection or {}).get("backend"),
            "gateway_base_url": runtime_environment.get("ANTHROPIC_BASE_URL"),
            "success": game_completed,
            "session_id": None,
            "duration_ms": None,
            "total_cost_usd": None,
            "num_turns": None,
            "stop_reason": None,
            "runtime_error": runtime_error,
        }

        for message in messages:
            session_id = getattr(message,
                                 "session_id",
                                 None)

            if session_id is not None:
                metadata["session_id"] = session_id

            for field_name in ("duration_ms", "total_cost_usd", "num_turns", "stop_reason"):
                value = getattr(message,
                                field_name,
                                None)

                if value is not None:
                    metadata[field_name] = value

        # ----- step 4 -----
        # write the raw SDK messages when artifact output is enabled
        artifacts = {}
        messages_path = write_text_artifact(output_dir=output_dir,
                                            filename="adapter_messages.txt",
                                            content="\n".join(repr(message) for message in messages))

        if messages_path is not None:
            artifacts["adapter_messages"] = messages_path

        return AgentRunResult(success=bool(metadata["success"]),
                              artifacts=artifacts,
                              metadata=metadata)
