import asyncio
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from clemcore.agents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemcore.agents.adapters.openai_compatible_proxy import proxy_for_model_connection
from clemcore.agents.adapters.utils import (
    load_model_connection,
    model_connection_environment,
    new_game_completion_path,
    read_game_completion,
    resolve_runtime_model,
    temporary_environment,
    write_text_artifact,
)


def _anthropic_proxy_base_url(proxy_base_url: str) -> str:
    """Let Anthropic clients append /v1/messages exactly once."""
    return proxy_base_url[:-3] if proxy_base_url.endswith("/v1") else proxy_base_url


class ClaudeCodeHarness(ExternalAgentHarness):
    """Run Claude Code through the Claude Agent SDK.

    The harness configures the container-side MCP bridge as a Claude Code
    server and collects every SDK message emitted during the episode.
    """

    def __init__(self,
                 model: str | None = None,
                 clem_model: str | None = None,
                 mcp_url: str = "http://localhost:8001/mcp",
                 max_turns: int | None = None,
                 allowed_tools: list[str] | None = None,
                 permission_mode: str = "bypassPermissions",
                 timeout: int = 600,
                 reasoning_effort: str | None = None,
                 model_connection_path: str | None = None):
        """Configure the Claude Code harness.

        Args:
            model: model identifier passed directly to Claude Code
            clem_model: clembench model resolved by the outer pipeline
            mcp_url: URL forwarded to the container-side MCP bridge
            max_turns: optional maximum number of Claude Code turns
            allowed_tools: tool patterns Claude Code pre-approves; this does
                not replace its built-in tool inventory
            permission_mode: Claude Code tool-permission policy
            timeout: maximum duration of one Claude Code episode in seconds
            reasoning_effort: model reasoning effort passed to Claude Code
            model_connection_path: optional resolved model-connection file
        """

        self.model = model or clem_model
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools or ["mcp__clem_game__start_game",
                                               "mcp__clem_game__submit_response"]
        self.permission_mode = permission_mode
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self._model_connection = load_model_connection("claude_code", model_connection_path)

    async def run_episode_async(self,
                                instruction: str,
                                runtime_model: str | None,
                                runtime_environment: dict[str, str | None],
                                completion_path: Path) -> tuple[list[Any], str | None, bool]:
        """Collect the asynchronous Claude SDK message stream.

        This method is the asynchronous boundary required by the Claude Agent
        SDK while the shared harness interface remains synchronous.

        Args:
            instruction: task instruction passed to Claude Code
            runtime_model: resolved model identifier
            runtime_environment: temporary model-provider environment
            completion_path: bridge-to-harness completion marker

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

        if self.reasoning_effort is not None:
            options_kwargs["effort"] = self.reasoning_effort

        options = ClaudeAgentOptions(**options_kwargs)
        messages = []
        runtime_error = None
        game_completed = False

        with temporary_environment(runtime_environment):
            try:
                async for message in query(prompt=instruction, options=options):
                    messages.append(message)
                    print(message)

                    completion = read_game_completion(completion_path)

                    if completion and completion.get("done") is True:
                        game_completed = completion.get("control_failure") is not True
                        break

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
        completion_path = new_game_completion_path()
        proxy = proxy_for_model_connection(self._model_connection, completion_path)

        if proxy is not None:
            with proxy:
                return self._run_episode(
                    instruction,
                    output_dir,
                    completion_path,
                    _anthropic_proxy_base_url(proxy.base_url),
                )

        return self._run_episode(instruction, output_dir, completion_path, None)

    def _run_episode(self,
                     instruction: str,
                     output_dir: Path | str | None,
                     completion_path: Path,
                     proxied_base_url: str | None) -> AgentRunResult:
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
        runtime_model = (
            (self._model_connection or {}).get("runtime_model")
            or runtime_model
        )
        runtime_environment = model_connection_environment(self._model_connection)
        runtime_environment["CLEM_GAME_COMPLETION_PATH"] = str(completion_path)

        if proxied_base_url is not None:
            runtime_environment["ANTHROPIC_BASE_URL"] = proxied_base_url

        # ----- step 2 -----
        # run Claude Code and collect the SDK message stream
        try:
            messages, runtime_error, game_completed = asyncio.run(
                asyncio.wait_for(
                    self.run_episode_async(instruction=instruction,
                                           runtime_model=runtime_model,
                                           runtime_environment=runtime_environment,
                                           completion_path=completion_path),
                    timeout=self.timeout,
                )
            )
        except asyncio.TimeoutError:
            messages = []
            runtime_error = f"Claude Code timed out after {self.timeout}s"
            game_completed = False
            print(f"agent_runtime_error: {runtime_error}")

        # ----- step 3 -----
        # extract standardized metadata from the SDK messages
        completion = read_game_completion(completion_path)
        game_completed = bool(
            game_completed
            or (
                completion
                and completion.get("done") is True
                and completion.get("control_failure") is not True
            )
        )
        metadata = {
            "adapter": "claude_code",
            "model": self.model,
            "clem_model": self.clem_model,
            "runtime_model": runtime_model,
            "resolved_backend": (self._model_connection or {}).get("backend"),
            "gateway_base_url": (
                (self._model_connection or {}).get("base_url")
                or runtime_environment.get("ANTHROPIC_BASE_URL")
            ),
            "compatibility_proxy_base_url": proxied_base_url,
            "tool_choice": (self._model_connection or {}).get("tool_choice"),
            "registry_request_body_overrides": (
                (self._model_connection or {}).get("request_body_overrides") or {}
            ),
            # OpenAI extra_body fields are not injected into Anthropic
            # Messages requests, so this accurately records the wire behavior.
            "request_body_overrides": {},
            "verify_tls": (self._model_connection or {}).get("verify_tls"),
            "timeout": self.timeout,
            "reasoning_effort": self.reasoning_effort,
            "success": game_completed,
            "session_id": None,
            "duration_ms": None,
            "total_cost_usd": None,
            "num_turns": None,
            "stop_reason": None,
            "runtime_error": runtime_error,
            "game_completed": game_completed,
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
