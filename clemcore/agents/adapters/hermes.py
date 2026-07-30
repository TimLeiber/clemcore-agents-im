import json
import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from clemcore.agents.adapters.base import AgentRunResult, ExternalAgentHarness


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _redact_sensitive(text: str) -> str:
    redacted = text

    # Raw key-like values.
    redacted = re.sub(r"sk-or-[A-Za-z0-9._-]+", "[REDACTED]", redacted)
    redacted = re.sub(r"sk-[A-Za-z0-9._-]+", "[REDACTED]", redacted)

    # Prefix-preserving key assignments / log lines.
    prefix_patterns = [
        r"(?i)(api key:\\s*)\\S+",
        r"(?i)(OPENROUTER_API_KEY=)\\S+",
        r"(?i)(OPENAI_API_KEY=)\\S+",
        r"(?i)(ANTHROPIC_API_KEY=)\\S+",
    ]

    for pattern in prefix_patterns:
        redacted = re.sub(
            pattern,
            lambda match: f"{match.group(1)}[REDACTED]",
            redacted,
        )

    return redacted


def _extract_hermes_session_id(text: str) -> str | None:
    match = re.search(r"^Session:\s*(\S+)", text, flags=re.MULTILINE)

    if match:
        return match.group(1)

    match = re.search(r"hermes --resume\s+(\S+)", text)

    if match:
        return match.group(1)

    return None


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


class HermesHarness(ExternalAgentHarness):
    """
    Hermes Agent harness controlled through the Hermes CLI.

    Hermes supports OpenRouter as a provider and can register stdio MCP servers
    through `hermes mcp add`. We use that path because it was verified in the
    Docker image and avoids guessing the internal config.yaml schema.
    """

    def __init__(self,
                 model: str | None = None,
                 clem_model: str | None = None,
                 provider: str = "openrouter",
                 mcp_url: str = "http://host.docker.internal:8001/mcp",
                 max_turns: int = 20,
                 yolo: bool = True,
                 model_connection_path: str | None = None):
        self.model = model or clem_model
        self.clem_model = clem_model
        self.provider = provider
        self.mcp_url = mcp_url
        self.max_turns = max_turns
        self.yolo = yolo
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

        if connection.get("harness") != "hermes":
            raise ValueError(f"Model connection is not for hermes: {connection}")

        return connection

    def _runtime_model(self) -> str:
        if self._model_connection:
            return str(self._model_connection.get("model"))

        if not self.model:
            raise RuntimeError("HermesHarness requires either model or clem_model.")

        return self.model

    def _runtime_env(self) -> dict[str, str | None]:
        if not self._model_connection:
            return {}

        return {
            key: str(value) if value is not None else None
            for key, value in self._model_connection.get("env", {}).items()
        }

    def _register_mcp_command(self) -> list[str]:
        return [
            "hermes",
            "mcp",
            "add",
            "clem_game",
            "--command",
            "python",
            "--env",
            f"PYTHONPATH={os.environ.get('PYTHONPATH', '/opt/clemcore')}",
            f"OPENENV_MCP_URL={self.mcp_url}",
            f"CLEM_EXPERIMENT_NAME={os.environ.get('CLEM_EXPERIMENT_NAME', '')}",
            f"CLEM_GAME_ID={os.environ.get('CLEM_GAME_ID', '')}",
            f"CLEM_OPENENV_SESSION_PATH={os.environ.get('CLEM_OPENENV_SESSION_PATH', '')}",
            "--args",
            "-m",
            "clemcore.agents.mcp.bridge",
        ]

    def _chat_command(self) -> list[str]:
        command = [
            "hermes",
            "chat",
            "--provider",
            self.provider,
            "--model",
            self._runtime_model(),
            "--max-turns",
            str(self.max_turns),
            "--ignore-rules",
            "--verbose",
        ]

        if self.yolo:
            command.append("--yolo")

        command.extend(["-q", ""])

        return command

    def run_episode(self,
                    instruction: str,
                    output_dir=None) -> AgentRunResult:
        metadata = {
            "adapter": "hermes",
            "model": self.model,
            "clem_model": self.clem_model,
            "runtime_model": None,
            "resolved_backend": (self._model_connection or {}).get("backend"),
            "provider": self.provider,
            "mcp_url": self.mcp_url,
            "max_turns": self.max_turns,
            "success": False,
            "returncode": None,
            "runtime_error": None,
            "tool_call_count_hint": 0,
            "hermes_session_id": None,
            "raw_reasoning_available": None,
            "raw_reasoning_note": (
                "Hermes raw reasoning is preserved only if present in "
                "Hermes session export/log artifacts."
            ),
        }

        artifacts = {}

        try:
            runtime_model = self._runtime_model()
            metadata["runtime_model"] = runtime_model
        except Exception as error:
            metadata["runtime_error"] = str(error)
            return AgentRunResult(success=False, artifacts=artifacts, metadata=metadata)

        register_command = self._register_mcp_command()
        chat_command = self._chat_command()
        chat_command[-1] = instruction

        with _temporary_environment(self._runtime_env()):
            register = subprocess.run(
                register_command,
                input="y\n",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

            config_commands = [
                ["hermes", "config", "set", "display.show_reasoning", "true"],
                ["hermes", "config", "set", "display.streaming", "true"],
                ["hermes", "config", "set", "display.tool_progress", "verbose"],
            ]

            config_results = [
                subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
                for command in config_commands
            ]

            chat = subprocess.run(
                chat_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,
            )

        combined_stdout = _redact_sensitive("\n".join([
            "hermes_mcp_add_command:",
            " ".join(register_command),
            "hermes_mcp_add_stdout:",
            register.stdout,
            "hermes_mcp_add_stderr:",
            register.stderr,
            "hermes_config_commands:",
            "\n".join(" ".join(command) for command in config_commands),
            "hermes_config_stdout:",
            "\n".join(result.stdout for result in config_results),
            "hermes_config_stderr:",
            "\n".join(result.stderr for result in config_results),
            "hermes_chat_command:",
            " ".join(chat_command[:-1] + ["<instruction>"]),
            "hermes_chat_stdout:",
            chat.stdout,
            "hermes_chat_stderr:",
            chat.stderr,
        ]))

        metadata["returncode"] = chat.returncode
        metadata["hermes_session_id"] = _extract_hermes_session_id(
            chat.stdout + "\n" + chat.stderr
        )
        metadata["tool_call_count_hint"] = (
            chat.stdout.count("mcp_clem_game_")
            + chat.stderr.count("mcp_clem_game_")
        )

        metadata["success"] = (
            register.returncode == 0
            and chat.returncode == 0
            and metadata["tool_call_count_hint"] > 0
        )

        if register.returncode != 0:
            metadata["runtime_error"] = "hermes mcp add failed"
        elif chat.returncode != 0:
            metadata["runtime_error"] = "hermes chat failed"
        elif metadata["tool_call_count_hint"] <= 0:
            metadata["runtime_error"] = "hermes completed without visible clem_game MCP tool calls"

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            trace_path = output_dir / "adapter_messages.txt"
            trace_path.write_text(combined_stdout, encoding="utf-8")
            artifacts["adapter_messages"] = trace_path

            session_id = metadata.get("hermes_session_id")

            if session_id:
                session_export_path = output_dir / "hermes_session_export.jsonl"
                session_export = subprocess.run(
                    [
                        "hermes",
                        "sessions",
                        "export",
                        "--session-id",
                        str(session_id),
                        str(session_export_path),
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )

                session_export_meta_path = output_dir / "hermes_session_export_meta.txt"
                session_export_meta_path.write_text(
                    "\n".join([
                        "hermes_sessions_export_returncode:",
                        str(session_export.returncode),
                        "hermes_sessions_export_stdout:",
                        session_export.stdout,
                        "hermes_sessions_export_stderr:",
                        session_export.stderr,
                    ]),
                    encoding="utf-8",
                )

                artifacts["hermes_session_export_meta"] = session_export_meta_path

                if session_export_path.exists():
                    artifacts["hermes_session_export"] = session_export_path

                hermes_log_path = output_dir / "hermes_agent_log.txt"
                hermes_log = subprocess.run(
                    [
                        "hermes",
                        "logs",
                        "--session",
                        str(session_id),
                        "--lines",
                        "1000",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )

                hermes_log_path.write_text(
                    "\n".join([
                        "hermes_logs_returncode:",
                        str(hermes_log.returncode),
                        "hermes_logs_stdout:",
                        hermes_log.stdout,
                        "hermes_logs_stderr:",
                        hermes_log.stderr,
                    ]),
                    encoding="utf-8",
                )
                artifacts["hermes_agent_log"] = hermes_log_path

        print(combined_stdout)

        return AgentRunResult(
            success=metadata["success"],
            artifacts=artifacts,
            metadata=metadata,
        )
