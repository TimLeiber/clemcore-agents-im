import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from clemcore.agents.base import AgentRunResult, ExternalAgentHarness


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _temporary_environment(env: dict[str, str | None]):
    sentinel = object()
    previous_values = {}

    for key, value in env.items():
        previous_values[key] = os.environ.get(key, sentinel)

        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


class ClaudeCodeHarness(ExternalAgentHarness):
    """
    Claude Code harness accessed through the Claude Agent SDK.

    If `clem_model` is configured, run_pipeline resolves it on the host using
    clembench/model_registry.json + key.json and passes a minimal connection
    file into Docker. This adapter translates that resolved OpenRouter
    connection into Claude Code runtime configuration.
    """

    def __init__(self,
                 model: str | None = None,
                 clem_model: str | None = None,
                 mcp_url: str = "http://localhost:8001/mcp",
                 max_turns: int = 20,
                 allowed_tools: list[str] | None = None,
                 permission_mode: str = "bypassPermissions",
                 model_connection_path: str | None = None):
        self.model = model or clem_model
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools or ["mcp__clem-game__*"]
        self.permission_mode = permission_mode
        self.model_connection_path = (
            model_connection_path
            or os.environ.get("CLEM_AGENT_MODEL_CONNECTION_PATH")
        )
        self._model_connection = self._load_model_connection()

    def _load_model_connection(self) -> dict[str, Any] | None:
        if not self.model_connection_path:
            return None

        path = Path(self.model_connection_path)

        if not path.exists():
            raise FileNotFoundError(f"Missing model connection file: {path}")

        connection = _load_json(path)

        if connection.get("harness") != "claude_code":
            raise ValueError(f"Model connection is not for claude_code: {connection}")

        return connection

    def _runtime_model(self) -> str | None:
        if self._model_connection:
            return self._model_connection.get("model")

        return self.model

    def _runtime_env(self) -> dict[str, str | None]:
        if not self._model_connection:
            return {}

        return {
            key: str(value) if value is not None else None
            for key, value in self._model_connection.get("env", {}).items()
        }

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

        runtime_model = self._runtime_model()

        if runtime_model:
            options_kwargs["model"] = runtime_model

        options = ClaudeAgentOptions(**options_kwargs)

        messages = []
        self._runtime_error = None

        with _temporary_environment(self._runtime_env()):
            try:
                async for message in query(prompt=instruction,
                                           options=options):
                    messages.append(message)
                    print(message)

            except Exception as error:
                self._runtime_error = str(error)
                print(f"agent_runtime_error: {error}")

        return messages

    def run_episode(self,
                    instruction: str,
                    output_dir=None) -> AgentRunResult:
        messages = asyncio.run(self.run_episode_async(instruction))

        metadata = self._extract_metadata(messages)
        artifacts = {}

        if output_dir is not None:
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
        connection = self._model_connection or {}

        metadata = {
            "adapter": "claude_code",
            "model": self.model,
            "clem_model": self.clem_model,
            "runtime_model": self._runtime_model(),
            "resolved_backend": connection.get("backend"),
            "gateway_base_url": self._runtime_env().get("ANTHROPIC_BASE_URL"),
            "success": False,
            "session_id": None,
            "duration_ms": None,
            "total_cost_usd": None,
            "num_turns": None,
            "stop_reason": None,
            "runtime_error": getattr(self, "_runtime_error", None),
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
