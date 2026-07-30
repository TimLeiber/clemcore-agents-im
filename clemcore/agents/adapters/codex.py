import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from clemcore.agents.adapters.base import AgentRunResult, ExternalAgentHarness


def _toml_string(value: str) -> str:
    return json.dumps(value)


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


def _write_codex_config(mcp_url: str,
                        openai_base_url: str | None = None) -> Path:
    config_dir = Path.home() / ".codex"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "config.toml"

    env = {
        "OPENENV_MCP_URL": mcp_url,
        "PYTHONPATH": os.environ.get("PYTHONPATH", "/opt/clemcore"),
    }

    for name in [
        "CLEM_EXPERIMENT_NAME",
        "CLEM_GAME_ID",
        "CLEM_OPENENV_SESSION_PATH",
    ]:
        value = os.environ.get(name)

        if value is not None:
            env[name] = value

    env_lines = "\n".join(
        f'{key} = {_toml_string(value)}'
        for key, value in sorted(env.items())
    )

    config_lines = [
        'approval_policy = "never"',
        'sandbox_mode = "danger-full-access"',
    ]

    if openai_base_url:
        if "openrouter.ai" in openai_base_url:
            config_lines.extend([
                'model_provider = "openrouter"',
                'model_reasoning_effort = "high"',
                "",
                "[model_providers.openrouter]",
                'name = "openrouter"',
                f"base_url = {_toml_string(openai_base_url)}",
                'env_key = "OPENROUTER_API_KEY"',
            ])
        else:
            provider_name = "openai_api"
            provider_title = "OpenAI API"
            env_key = "OPENAI_API_KEY"

            config_lines.extend([
                f'model_provider = "{provider_name}"',
                "",
                f"[model_providers.{provider_name}]",
                f"name = {_toml_string(provider_title)}",
                f"base_url = {_toml_string(openai_base_url)}",
                f"env_key = {_toml_string(env_key)}",
                'wire_api = "responses"',
            ])

    config_lines.extend([
        "",
        "[mcp_servers.clem_game]",
        'command = "python"',
        'args = ["-m", "clemcore.agents.mcp.bridge"]',
        "startup_timeout_sec = 20",
        "tool_timeout_sec = 120",
        "enabled = true",
        "required = true",
        'enabled_tools = ["start_game", "submit_response", "get_state"]',
        'default_tools_approval_mode = "approve"',
        "",
        "[mcp_servers.clem_game.env]",
        env_lines,
        "",
    ])

    config_path.write_text("\n".join(config_lines), encoding="utf-8")

    return config_path


class CodexHarness(ExternalAgentHarness):
    """
    Codex harness controlled through the Codex CLI.

    Codex MCP support is exposed through the CLI/IDE config layer. The Python
    SDK path does not currently expose the clem_game MCP tools in our run mode,
    so this harness uses `codex exec` and ~/.codex/config.toml.
    """

    def __init__(self,
                 model: str | None = None,
                 clem_model: str | None = None,
                 mcp_url: str = "http://host.docker.internal:8001/mcp",
                 sandbox: str = "full_access",
                 model_connection_path: str | None = None):
        self.model = model or clem_model or "gpt-5.4"
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.sandbox = sandbox
        self.model_connection_path = (
            model_connection_path
            or os.environ.get("CLEM_AGENT_MODEL_CONNECTION_PATH")
        )
        self._model_connection = self._load_model_connection()
        self._runtime_error: str | None = None

    def _load_model_connection(self) -> dict[str, Any] | None:
        if not self.model_connection_path:
            return None

        path = Path(self.model_connection_path)

        if not path.exists():
            raise FileNotFoundError(f"Missing model connection file: {path}")

        connection = _load_json(path)

        if connection.get("harness") != "codex":
            raise ValueError(f"Model connection is not for codex: {connection}")

        return connection

    def _runtime_model(self) -> str:
        if self._model_connection:
            return str(self._model_connection.get("model"))

        return self.model

    def _runtime_env(self) -> dict[str, str | None]:
        if not self._model_connection:
            return {}

        return {
            key: str(value) if value is not None else None
            for key, value in self._model_connection.get("env", {}).items()
        }

    def _runtime_base_url(self) -> str | None:
        if self._model_connection:
            base_url = self._model_connection.get("base_url")

            if base_url:
                return str(base_url)

        return None

    def _cli_sandbox_mode(self) -> str:
        values = {
            "read_only": "read-only",
            "workspace_write": "workspace-write",
            "full_access": "danger-full-access",
        }

        if self.sandbox not in values:
            raise ValueError(
                f"Unknown Codex sandbox '{self.sandbox}'. "
                f"Expected one of: {sorted(values)}"
            )

        return values[self.sandbox]

    def _codex_command(self,
                       last_message_path: Path) -> list[str]:
        command = [
            "codex",
            "exec",
            "--strict-config",
            "--json",
            "--model",
            self._runtime_model(),
            "--cd",
            "/workspace",
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-last-message",
            str(last_message_path),
        ]

        if self.sandbox == "full_access":
            command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            command.extend(["--sandbox", self._cli_sandbox_mode()])

        command.append("-")

        return command

    def run_episode(self,
                    instruction: str,
                    output_dir=None) -> AgentRunResult:
        config_path = _write_codex_config(
            self.mcp_url,
            openai_base_url=self._runtime_base_url(),
        )

        last_message_path = (
            Path.home()
            / ".codex"
            / f"last_message_{os.getpid()}.txt"
        )

        if last_message_path.exists():
            last_message_path.unlink()

        metadata = {
            "adapter": "codex",
            "model": self.model,
            "clem_model": self.clem_model,
            "runtime_model": self._runtime_model(),
            "resolved_backend": (self._model_connection or {}).get("backend"),
            "gateway_base_url": self._runtime_base_url(),
            "mcp_url": self.mcp_url,
            "codex_config": str(config_path),
            "sandbox": self.sandbox,
            "success": False,
            "runtime_error": None,
            "returncode": None,
            "final_response": None,
        }

        codex_instruction = instruction + """

Codex-specific execution rule:
The MCP server named clem_game is already configured.
It exposes exactly these game tools: start_game, submit_response, get_state.
Do not search for tools.
Do not inspect tools.
Do not explain what you will do.
Do not finish the turn with reasoning only.
The first action you produce must be an MCP tool call to clem_game.start_game.
After start_game returns, continue only by calling clem_game.submit_response until the game result says done=true.
"""

        command = self._codex_command(last_message_path)

        mcp_list_stdout = ""
        mcp_list_stderr = ""
        mcp_list_returncode = None

        try:
            with _temporary_environment(self._runtime_env()):
                mcp_list = subprocess.run(
                    ["codex", "mcp", "list"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd="/workspace",
                    check=False,
                )

            mcp_list_stdout = mcp_list.stdout
            mcp_list_stderr = mcp_list.stderr
            mcp_list_returncode = mcp_list.returncode

        except Exception as error:
            mcp_list_stderr = str(error)

        transcript: list[str] = []
        transcript.append(f"codex_config: {config_path}")
        transcript.append("codex_config_toml:")
        transcript.append(config_path.read_text(encoding="utf-8"))
        transcript.append(f"model: {self.model}")
        transcript.append(f"runtime_model: {self._runtime_model()}")
        transcript.append(f"gateway_base_url: {self._runtime_base_url()}")
        transcript.append(f"sandbox: {self.sandbox}")
        transcript.append("codex_mcp_list_returncode:")
        transcript.append(str(mcp_list_returncode))
        transcript.append("codex_mcp_list_stdout:")
        transcript.append(mcp_list_stdout)
        transcript.append("codex_mcp_list_stderr:")
        transcript.append(mcp_list_stderr)
        transcript.append("codex_command:")
        transcript.append(" ".join(command))

        try:
            with _temporary_environment(self._runtime_env()):
                completed = subprocess.run(
                    command,
                    input=codex_instruction,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd="/workspace",
                    check=False,
                )

            metadata["returncode"] = completed.returncode
            metadata["success"] = completed.returncode == 0

            transcript.append("codex_stdout_jsonl:")
            transcript.append(completed.stdout)
            transcript.append("codex_stderr:")
            transcript.append(completed.stderr)

            if last_message_path.exists():
                final_response = last_message_path.read_text(encoding="utf-8")
                metadata["final_response"] = final_response
                transcript.append("final_response:")
                transcript.append(final_response)

        except Exception as error:
            self._runtime_error = str(error)
            metadata["runtime_error"] = self._runtime_error
            transcript.append(f"agent_runtime_error: {error}")
            print(f"agent_runtime_error: {error}")

        trace_text = "\n".join(transcript)

        print(trace_text)

        artifacts = {}

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            messages_path = output_dir / "adapter_messages.txt"
            messages_path.write_text(trace_text, encoding="utf-8")
            artifacts["adapter_messages"] = messages_path

            saved_config_path = output_dir / "codex_config.toml"
            saved_config_path.write_text(
                config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            artifacts["codex_config"] = saved_config_path

        return AgentRunResult(
            success=metadata["success"],
            artifacts=artifacts,
            metadata=metadata,
        )
