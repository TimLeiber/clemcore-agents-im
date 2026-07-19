import json
import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from clemcore.agents.base import AgentRunResult, ExternalAgentHarness


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _redact_sensitive(text: str) -> str:
    redacted = text

    # Raw key-like values.
    redacted = re.sub(r"sk-or-[A-Za-z0-9._-]+", "[REDACTED]", redacted)
    redacted = re.sub(r"sk-[A-Za-z0-9._-]+", "[REDACTED]", redacted)

    # Prefix-preserving key assignments / log lines.
    prefix_patterns = [
        r"(?i)(api key:\s*)\S+",
        r"(?i)(OPENROUTER_API_KEY=)\S+",
        r"(?i)(OPENAI_API_KEY=)\S+",
        r"(?i)(ANTHROPIC_API_KEY=)\S+",
        r"(?i)(--token\s+)\S+",
        r"(?i)(--openrouter-api-key\s+)\S+",
    ]

    for pattern in prefix_patterns:
        redacted = re.sub(
            pattern,
            lambda match: f"{match.group(1)}[REDACTED]",
            redacted,
        )

    return redacted


def _deep_merge_dicts(base: dict[str, Any],
                      patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    for key, value in patch.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


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


class OpenClawHarness(ExternalAgentHarness):
    """
    OpenClaw harness controlled through the OpenClaw CLI.

    The target path is:
      openclaw agent --local --model openrouter/<author>/<slug>

    MCP is configured through OpenClaw's managed mcp.servers config before each
    run. Raw stdout/stderr are preserved in adapter_messages.txt for downstream
    transcript-with-tools parsing.
    """

    def __init__(self,
                 model: str | None = None,
                 clem_model: str | None = None,
                 mcp_url: str = "http://host.docker.internal:8001/mcp",
                 thinking: str | None = None,
                 verbose: bool = True,
                 timeout: int = 600,
                 profile: str = "clembench-openclaw",
                 yolo: bool = True,
                 model_connection_path: str | None = None):
        self.model = model or clem_model
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.thinking = thinking
        self.verbose = verbose
        self.timeout = timeout
        self.profile = profile
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

        if connection.get("harness") != "openclaw":
            raise ValueError(f"Model connection is not for openclaw: {connection}")

        return connection

    def _runtime_model(self) -> str:
        if self._model_connection:
            return str(self._model_connection.get("model"))

        if not self.model:
            raise RuntimeError("OpenClawHarness requires either model or clem_model.")

        return self.model

    def _runtime_env(self) -> dict[str, str | None]:
        if not self._model_connection:
            return {}

        return {
            key: str(value) if value is not None else None
            for key, value in self._model_connection.get("env", {}).items()
        }

    def _openclaw_base_command(self) -> list[str]:
        return [
            "openclaw",
            "--no-color",
            "--profile",
            self.profile,
            "--log-level",
            "trace",
        ]

    def _mcp_add_command(self) -> list[str]:
        env_entries = {
            "OPENENV_MCP_URL": self.mcp_url,
        }

        for name in [
            "CLEM_EXPERIMENT_NAME",
            "CLEM_GAME_ID",
            "CLEM_OPENENV_SESSION_PATH",
        ]:
            value = os.environ.get(name)

            if value is not None:
                env_entries[name] = value

        command = self._openclaw_base_command() + [
            "mcp",
            "add",
            "clem_game",
            "--command",
            "python",
            "--arg",
            "-m",
            "--arg",
            "clemcore.agents.mcp.bridge_server",
            "--cwd",
            "/opt/clemcore",
            "--include",
            "start_game,submit_response,get_state",
            "--no-probe",
        ]

        for key, value in sorted(env_entries.items()):
            command.extend(["--env", f"{key}={value}"])

        return command

    def _session_key(self) -> str:
        parts = [
            "clembench",
            os.environ.get("CLEM_EXPERIMENT_NAME", "experiment"),
            os.environ.get("CLEM_GAME_ID", "game"),
            str(os.getpid()),
        ]
        safe_parts = [
            re.sub(r"[^A-Za-z0-9_.-]+", "-", part).strip("-") or "x"
            for part in parts
        ]
        return "agent:clembench:" + "-".join(safe_parts)

    def _agent_command(self,
                       instruction_path: Path) -> list[str]:
        command = self._openclaw_base_command() + [
            "agent",
            "--local",
            "--session-key",
            self._session_key(),
            "--model",
            self._runtime_model(),
            "--message-file",
            str(instruction_path),
            "--timeout",
            str(self.timeout),
            "--json",
        ]

        if self.thinking is not None:
            command.extend(["--thinking", self.thinking])

        if self.verbose:
            command.extend(["--verbose", "on"])

        return command

    def _yolo_config_patch(self,
                           log_path: Path | None = None) -> str:
        config: dict[str, Any] = {}

        if self._model_connection:
            openclaw_config_patch = self._model_connection.get("openclaw_config_patch", {})

            if isinstance(openclaw_config_patch, dict):
                config = _deep_merge_dicts(config, openclaw_config_patch)

        # Benchmark harness policy: no approval gates, no sandbox restriction.
        if self.yolo:
            config["tools"] = {
                "exec": {
                    "security": "full",
                    "ask": "off",
                },
            }

        # Benchmark trace policy: capture all accepted OpenClaw log surfaces to
        # a known per-run JSONL file, then redact again before agent_trace.log.
        if log_path is not None:
            config["logging"] = {
                "level": "trace",
                "consoleLevel": "trace",
                "consoleStyle": "json",
                "file": str(log_path),
                "redactSensitive": "tools",
            }

        return json.dumps(config)

    def _yolo_config_command(self) -> list[str]:
        return self._openclaw_base_command() + [
            "config",
            "patch",
            "--stdin",
        ]

    def _models_set_command(self) -> list[str]:
        return self._openclaw_base_command() + [
            "models",
            "set",
            self._runtime_model(),
        ]

    def run_episode(self,
                    instruction: str,
                    output_dir=None) -> AgentRunResult:
        metadata = {
            "adapter": "openclaw",
            "model": self.model,
            "clem_model": self.clem_model,
            "runtime_model": None,
            "resolved_backend": (self._model_connection or {}).get("backend"),
            "mcp_url": self.mcp_url,
            "thinking": self.thinking,
            "verbose": self.verbose,
            "timeout": self.timeout,
            "profile": self.profile,
            "yolo": self.yolo,
            "session_key": None,
            "success": False,
            "returncode": None,
            "runtime_error": None,
            "tool_call_count_hint": 0,
            "raw_reasoning_available": None,
            "raw_reasoning_note": (
                "OpenClaw raw reasoning is preserved only if present in "
                "OpenClaw verbose stdout/stderr or session/transcript artifacts."
            ),
        }

        artifacts = {}

        try:
            runtime_model = self._runtime_model()
            metadata["runtime_model"] = runtime_model
            metadata["session_key"] = self._session_key()
        except Exception as error:
            metadata["runtime_error"] = str(error)
            return AgentRunResult(success=False, artifacts=artifacts, metadata=metadata)

        if output_dir is None:
            run_dir = Path("/tmp") / f"openclaw-clembench-{os.getpid()}"
        else:
            run_dir = Path(output_dir)

        run_dir.mkdir(parents=True, exist_ok=True)

        home_dir = run_dir / "openclaw_home"
        home_dir.mkdir(parents=True, exist_ok=True)

        openclaw_log_path = run_dir / "openclaw.log"
        metadata["openclaw_log_path"] = str(openclaw_log_path)

        instruction_path = run_dir / "openclaw_instruction.txt"
        instruction_path.write_text(instruction, encoding="utf-8")

        runtime_env = self._runtime_env()
        runtime_env.update({
            "HOME": str(home_dir),
            "OPENENV_MCP_URL": self.mcp_url,
            "NODE_TLS_REJECT_UNAUTHORIZED": "0", # TODO: remove when connection problem is fixed

            # Maximize OpenClaw-side diagnostics for benchmark trace capture.
            # OpenClaw redacts known secrets, and the harness applies a second
            # redaction pass before writing agent_trace.log.
            "OPENCLAW_LOG_LEVEL": "trace",
            "OPENCLAW_DEBUG_MODEL_TRANSPORT": "1",
            "OPENCLAW_DEBUG_MODEL_PAYLOAD": "full-redacted",
            "OPENCLAW_DEBUG_SSE": "peek",
            "OPENCLAW_DEBUG_CODE_MODE": "1",
        })

        mcp_command = self._mcp_add_command()
        yolo_config_command = self._yolo_config_command()
        yolo_config_patch = self._yolo_config_patch(openclaw_log_path)
        models_set_command = self._models_set_command()
        agent_command = self._agent_command(instruction_path)

        with _temporary_environment(runtime_env):
            mcp_add = subprocess.run(
                mcp_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

            yolo_config = subprocess.run(
                yolo_config_command,
                input=yolo_config_patch,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

            models_set = subprocess.run(
                models_set_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

            agent = subprocess.run(
                agent_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout + 30,
            )

        openclaw_artifact_sections = []

        artifact_candidates = [openclaw_log_path]

        artifact_candidates.extend(sorted(home_dir.glob(".openclaw*/agents/**/*.jsonl")))
        artifact_candidates.extend(sorted(home_dir.glob(".openclaw*/**/*.jsonl")))
        artifact_candidates.extend(sorted(home_dir.glob(".openclaw*/**/*.log")))
        artifact_candidates.extend(sorted(Path("/tmp").glob("openclaw*/openclaw*.log")))
        artifact_candidates.extend(sorted(Path("/tmp").glob("openclaw*/**/*.log")))

        seen_artifacts = set()

        for artifact_path in artifact_candidates:
            try:
                resolved_artifact_path = artifact_path.resolve()
            except Exception:
                resolved_artifact_path = artifact_path

            if resolved_artifact_path in seen_artifacts:
                continue

            seen_artifacts.add(resolved_artifact_path)

            try:
                artifact_text = artifact_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception as error:
                artifact_text = f"[failed to read {artifact_path}: {error}]"

            openclaw_artifact_sections.extend([
                f"openclaw_artifact: {artifact_path}",
                artifact_text,
            ])

        combined_trace = _redact_sensitive("\n".join([
            "openclaw_mcp_add_command:",
            " ".join(mcp_command),
            "openclaw_mcp_add_stdout:",
            mcp_add.stdout,
            "openclaw_mcp_add_stderr:",
            mcp_add.stderr,
            "openclaw_yolo_config_command:",
            " ".join(yolo_config_command),
            "openclaw_yolo_config_patch:",
            yolo_config_patch,
            "openclaw_yolo_config_stdout:",
            yolo_config.stdout,
            "openclaw_yolo_config_stderr:",
            yolo_config.stderr,
            "openclaw_models_set_command:",
            " ".join(models_set_command),
            "openclaw_models_set_stdout:",
            models_set.stdout,
            "openclaw_models_set_stderr:",
            models_set.stderr,
            "openclaw_debug_environment:",
            json.dumps({
                "OPENCLAW_LOG_LEVEL": runtime_env.get("OPENCLAW_LOG_LEVEL"),
                "OPENCLAW_DEBUG_MODEL_TRANSPORT": runtime_env.get("OPENCLAW_DEBUG_MODEL_TRANSPORT"),
                "OPENCLAW_DEBUG_MODEL_PAYLOAD": runtime_env.get("OPENCLAW_DEBUG_MODEL_PAYLOAD"),
                "OPENCLAW_DEBUG_SSE": runtime_env.get("OPENCLAW_DEBUG_SSE"),
                "OPENCLAW_DEBUG_CODE_MODE": runtime_env.get("OPENCLAW_DEBUG_CODE_MODE"),
                "OPENCLAW_LOG_FILE": str(openclaw_log_path),
                "OPENCLAW_THINKING": self.thinking,
            }, indent=2),
            "openclaw_agent_command:",
            " ".join(agent_command),
            "openclaw_agent_stdout:",
            agent.stdout,
            "openclaw_agent_stderr:",
            agent.stderr,
            *openclaw_artifact_sections,
        ]))

        tool_call_count_hint = (
            combined_trace.count("mcp__clem-game__")
            + combined_trace.count("mcp_clem_game_")
            + combined_trace.count("clem_game.start_game")
            + combined_trace.count("clem_game.submit_response")
            + combined_trace.count("clem_game.get_state")
            + combined_trace.count('"start_game"')
            + combined_trace.count('"submit_response"')
            + combined_trace.count('"get_state"')
        )

        openclaw_incomplete_after_tools = (
            '"kind": "incomplete_turn"' in combined_trace
            and '"failures": 0' in combined_trace
            and "clem_game__start_game" in combined_trace
            and "clem_game__submit_response" in combined_trace
        )

        metadata["returncode"] = agent.returncode
        metadata["tool_call_count_hint"] = tool_call_count_hint
        metadata["openclaw_incomplete_after_tools"] = openclaw_incomplete_after_tools

        metadata["success"] = (
            mcp_add.returncode == 0
            and yolo_config.returncode == 0
            and models_set.returncode == 0
            and agent.returncode == 0
            and (
                tool_call_count_hint > 0
                or openclaw_incomplete_after_tools
            )
        )

        if mcp_add.returncode != 0:
            metadata["runtime_error"] = "openclaw mcp add failed"
        elif yolo_config.returncode != 0:
            metadata["runtime_error"] = "openclaw yolo config patch failed"
        elif models_set.returncode != 0:
            metadata["runtime_error"] = "openclaw models set failed"
        elif agent.returncode != 0:
            metadata["runtime_error"] = "openclaw agent failed"
        elif tool_call_count_hint <= 0 and not openclaw_incomplete_after_tools:
            metadata["runtime_error"] = "openclaw completed without visible clem_game MCP tool calls"
        elif openclaw_incomplete_after_tools:
            metadata["runtime_error"] = None

        if output_dir is not None:
            trace_path = run_dir / "adapter_messages.txt"
            trace_path.write_text(combined_trace, encoding="utf-8")
            artifacts["adapter_messages"] = trace_path

            meta_path = run_dir / "openclaw_run_meta.json"
            meta_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            artifacts["openclaw_run_meta"] = meta_path

        print(combined_trace)

        return AgentRunResult(
            success=bool(metadata["success"]),
            artifacts=artifacts,
            metadata=metadata,
        )
