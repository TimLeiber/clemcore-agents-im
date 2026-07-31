import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from clemcore.agents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemcore.agents.adapters.utils import (
    deep_merge_dicts,
    load_model_connection,
    mcp_environment,
    model_connection_environment,
    resolve_runtime_model,
    temporary_environment,
    write_text_artifact,
)


SESSION_PART_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
GAME_TOOL_MARKERS = (
    "mcp__clem-game__",
    "mcp_clem_game_",
    "clem_game.start_game",
    "clem_game.submit_response",
    "clem_game.get_state",
    '"start_game"',
    '"submit_response"',
    '"get_state"',
)


def _validate_openclaw_model_connection(connection: dict[str, Any] | None) -> None:
    """Reject inconsistent resolved OpenRouter connections before CLI startup."""
    if not connection or connection.get("backend") != "openrouter":
        return

    runtime_model = connection.get("model")
    if not isinstance(runtime_model, str) or not runtime_model.startswith("openrouter/"):
        raise ValueError(
            "OpenClaw OpenRouter connections require a canonical "
            f"openrouter/<provider>/<model> reference, got {runtime_model!r}."
        )

    environment = connection.get("env")
    if not isinstance(environment, dict) or not environment.get("OPENROUTER_API_KEY"):
        raise ValueError(
            "OpenClaw OpenRouter connections require OPENROUTER_API_KEY."
        )


def _error_from_output(*outputs: str) -> str | None:
    """Return the most useful OpenClaw error without dumping its full trace."""

    error_keys = ("errorMessage", "error_message", "message", "error")
    parsed_values: list[Any] = []

    for output in outputs:
        for line in output.splitlines():
            try:
                parsed_values.append(json.loads(line))
            except (TypeError, json.JSONDecodeError):
                continue

    def find_error(value: Any) -> str | None:
        if isinstance(value, dict):
            for key in error_keys:
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            for candidate in value.values():
                found = find_error(candidate)
                if found:
                    return found
        elif isinstance(value, list):
            for candidate in value:
                found = find_error(candidate)
                if found:
                    return found
        return None

    for value in parsed_values:
        found = find_error(value)
        if found:
            return found

    for output in outputs:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return None


class OpenClawHarness(ExternalAgentHarness):
    """Run one OpenClaw agent in an isolated home directory."""

    def __init__(self,
                 model: str | None = None,
                 clem_model: str | None = None,
                 mcp_url: str = "http://host.docker.internal:8001/mcp",
                 thinking: str | None = None,
                 verbose: bool = True,
                 timeout: int = 600,
                 profile: str = "clembench-openclaw",
                 yolo: bool = True,
                 debug: bool = False,
                 model_connection_path: str | None = None):
        self.model = model or clem_model
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.thinking = thinking
        self.verbose = verbose
        self.timeout = timeout
        self.profile = profile
        self.yolo = yolo
        self.debug = debug
        self._model_connection = load_model_connection("openclaw", model_connection_path)
        _validate_openclaw_model_connection(self._model_connection)

    def run_episode(self,
                    instruction: str,
                    output_dir: Path | str | None = None) -> AgentRunResult:
        metadata: dict[str, Any] = {
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
            "debug": self.debug,
            "session_key": None,
            "success": False,
            "returncode": None,
            "runtime_error": None,
            "tool_call_count_hint": 0,
        }
        artifacts: dict[str, Path | str | int | float | bool | None] = {}

        try:
            runtime_model = resolve_runtime_model(
                model_connection=self._model_connection,
                model=self.model,
                harness_name="OpenClawHarness",
            )
        except Exception as error:
            metadata["runtime_error"] = str(error)
            return AgentRunResult(False, artifacts, metadata)

        session_parts = (
            "clembench",
            os.environ.get("CLEM_EXPERIMENT_NAME", "experiment"),
            os.environ.get("CLEM_GAME_ID", "game"),
            str(os.getpid()),
        )
        safe_parts = [SESSION_PART_PATTERN.sub("-", part).strip("-") or "x"
                      for part in session_parts]
        session_key = "agent:clembench:" + "-".join(safe_parts)
        metadata.update(runtime_model=runtime_model, session_key=session_key)

        run_dir = (Path(output_dir) if output_dir is not None
                   else Path("/tmp") / f"openclaw-clembench-{os.getpid()}")
        run_dir.mkdir(parents=True, exist_ok=True)
        home_dir = run_dir / "openclaw_home"
        home_dir.mkdir(parents=True, exist_ok=True)
        instruction_path = run_dir / "openclaw_instruction.txt"
        instruction_path.write_text(instruction, encoding="utf-8")

        runtime_environment = model_connection_environment(self._model_connection)
        runtime_environment.update({
            "HOME": str(home_dir),
            "OPENENV_MCP_URL": self.mcp_url,
        })
        log_level = "debug" if self.debug else "warn"
        base_command = [
            "openclaw", "--no-color", "--profile", self.profile,
            "--log-level", log_level,
        ]
        bridge_environment = mcp_environment(self.mcp_url)
        mcp_command = base_command + [
            "mcp", "add", "clem_game",
            "--command", "python",
            "--arg", "-m",
            "--arg", "clemcore.agents.mcp.bridge",
            "--cwd", "/opt/clemcore",
            "--include", "start_game,submit_response,get_state",
            "--no-probe",
        ]
        for name, value in sorted(bridge_environment.items()):
            mcp_command.extend(["--env", f"{name}={value}"])

        openclaw_config: dict[str, Any] = {}
        connection_patch = (self._model_connection or {}).get("openclaw_config_patch", {})
        if isinstance(connection_patch, dict):
            openclaw_config = deep_merge_dicts(openclaw_config, connection_patch)
        if (self._model_connection or {}).get("backend") == "openrouter":
            openclaw_config = deep_merge_dicts(
                openclaw_config,
                {
                    "env": {
                        "OPENROUTER_API_KEY": runtime_environment[
                            "OPENROUTER_API_KEY"
                        ],
                    },
                },
            )
        if self.yolo:
            openclaw_config["tools"] = {"exec": {"security": "full", "ask": "off"}}
        openclaw_config["logging"] = {
            "level": log_level,
            "consoleLevel": log_level,
            "consoleStyle": "json",
        }

        commands = [
            ("mcp add", mcp_command, None, 60),
            ("config patch", base_command + ["config", "patch", "--stdin"],
             json.dumps(openclaw_config), 60),
            ("models set", base_command + ["models", "set", runtime_model], None, 60),
        ]
        agent_command = base_command + [
            "agent", "--local",
            "--session-key", session_key,
            "--model", runtime_model,
            "--message-file", str(instruction_path),
            "--timeout", str(self.timeout),
            "--json",
        ]
        if self.thinking is not None:
            agent_command.extend(["--thinking", self.thinking])
        if self.verbose:
            agent_command.extend(["--verbose", "on"])
        commands.append(("agent", agent_command, None, self.timeout + 30))

        results: dict[str, subprocess.CompletedProcess[str]] = {}
        failed_step: str | None = None
        try:
            with temporary_environment(runtime_environment):
                for name, command, input_text, timeout in commands:
                    result = subprocess.run(
                        command,
                        input=input_text,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=timeout,
                        check=False,
                    )
                    results[name] = result
                    if result.returncode != 0:
                        failed_step = name
                        break
        except subprocess.TimeoutExpired as error:
            failed_step = name
            metadata["runtime_error"] = f"OpenClaw {name} timed out after {error.timeout}s"
        except (OSError, subprocess.SubprocessError) as error:
            failed_step = name
            metadata["runtime_error"] = f"OpenClaw {name} could not run: {error}"

        # Session JSONL is the useful game transcript. Only inspect this run's
        # isolated home; never harvest unrelated global /tmp OpenClaw logs.
        session_paths = sorted(set(home_dir.glob(".openclaw*/agents/**/*.jsonl")))
        trace_sections: list[str] = []
        for name, result in results.items():
            trace_sections.extend([
                f"openclaw_{name.replace(' ', '_')}_stdout:", result.stdout,
                f"openclaw_{name.replace(' ', '_')}_stderr:", result.stderr,
            ])
        for session_path in session_paths:
            trace_sections.extend([
                f"openclaw_session: {session_path}",
                session_path.read_text(encoding="utf-8", errors="replace"),
            ])
        combined_trace = "\n".join(trace_sections)

        agent_result = results.get("agent")
        if agent_result is not None:
            metadata["returncode"] = agent_result.returncode
        metadata["tool_call_count_hint"] = sum(
            combined_trace.count(marker) for marker in GAME_TOOL_MARKERS
        )

        if failed_step and metadata["runtime_error"] is None:
            failed_result = results[failed_step]
            detail = _error_from_output(failed_result.stderr, failed_result.stdout)
            metadata["runtime_error"] = f"OpenClaw {failed_step} failed"
            if detail:
                metadata["runtime_error"] += f": {detail}"
        elif agent_result is not None and metadata["tool_call_count_hint"] <= 0:
            detail = _error_from_output(agent_result.stderr, agent_result.stdout)
            metadata["runtime_error"] = "OpenClaw completed without calling a clem_game tool"
            if detail:
                metadata["runtime_error"] += f": {detail}"
        else:
            metadata["success"] = agent_result is not None

        trace_path = write_text_artifact(
            output_dir=output_dir,
            filename="adapter_messages.txt",
            content=combined_trace,
        )
        metadata_path = write_text_artifact(
            output_dir=output_dir,
            filename="openclaw_run_meta.json",
            content=json.dumps(metadata, indent=2, sort_keys=True),
        )
        if trace_path is not None:
            artifacts["adapter_messages"] = trace_path
        if metadata_path is not None:
            artifacts["openclaw_run_meta"] = metadata_path

        if self.debug:
            print(combined_trace)
        if metadata["runtime_error"]:
            print(metadata["runtime_error"])

        return AgentRunResult(bool(metadata["success"]), artifacts, metadata)
