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
    redact_sensitive,
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
    """Validate the native OpenRouter model and credential expected by OpenClaw."""
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
        raise ValueError("OpenClaw OpenRouter connections require OPENROUTER_API_KEY.")


def _error_detail(result: subprocess.CompletedProcess[str]) -> str | None:
    """Return a short, redacted error from a failed OpenClaw command."""
    lines = [
        line.strip()
        for output in (result.stderr, result.stdout)
        for line in output.splitlines()
        if line.strip()
    ]
    return redact_sensitive(lines[-1]) if lines else None


class OpenClawHarness(ExternalAgentHarness):
    """Run the native OpenClaw CLI with an isolated profile and Clem MCP tools."""

    def __init__(self,
                 model: str | None = None,
                 clem_model: str | None = None,
                 mcp_url: str = "http://host.docker.internal:8001/mcp",
                 thinking: str | None = None,
                 verbose: bool = False,
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
        artifacts: dict[str, Path | str | int | float | bool | None] = {}
        metadata: dict[str, Any] = {
            "adapter": "openclaw",
            "success": False,
            "returncode": None,
            "runtime_error": None,
        }

        try:
            runtime_model = resolve_runtime_model(
                model_connection=self._model_connection,
                model=self.model,
                harness_name="OpenClawHarness",
            )
        except Exception as error:
            metadata["runtime_error"] = str(error)
            return AgentRunResult(False, artifacts, metadata)

        run_dir = (Path(output_dir) if output_dir is not None
                   else Path("/tmp") / f"openclaw-clembench-{os.getpid()}")
        run_dir.mkdir(parents=True, exist_ok=True)
        home_dir = run_dir / "openclaw_home"
        home_dir.mkdir(parents=True, exist_ok=True)
        instruction_path = run_dir / "openclaw_instruction.txt"
        instruction_path.write_text(instruction, encoding="utf-8")

        session_parts = (
            "clembench",
            os.environ.get("CLEM_EXPERIMENT_NAME", "experiment"),
            os.environ.get("CLEM_GAME_ID", "game"),
            str(os.getpid()),
        )
        session_key = "agent:clembench:" + "-".join(
            SESSION_PART_PATTERN.sub("-", part).strip("-") or "x"
            for part in session_parts
        )

        runtime_environment = model_connection_environment(self._model_connection)
        runtime_environment["HOME"] = str(home_dir)
        base_command = ["openclaw", "--no-color", "--profile", self.profile]

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

        config = (self._model_connection or {}).get("openclaw_config_patch", {})
        config = dict(config) if isinstance(config, dict) else {}
        # Restrict the model-facing tool inventory to the three Clem MCP tools.
        # Do not combine this absolute allowlist with the ``minimal`` profile:
        # profiles are applied first, and that profile removes plugin-provided
        # MCP tools before the allowlist can select them.
        config = deep_merge_dicts(config, {
            # The native OpenRouter model catalog is supplied by this plugin,
            # so retain it while excluding unrelated bundled plugins.
            "plugins": {"enabled": True, "allow": ["openrouter"]},
            "tools": {
                "allow": [
                    "clem_game__start_game",
                    "clem_game__submit_response",
                    "clem_game__get_state",
                ],
            },
        })
        if (self._model_connection or {}).get("backend") == "openrouter":
            config = deep_merge_dicts(config, {
                "env": {
                    "OPENROUTER_API_KEY": runtime_environment["OPENROUTER_API_KEY"],
                },
            })
        if self.yolo:
            config = deep_merge_dicts(config, {
                "tools": {"exec": {"security": "full", "ask": "off"}},
            })

        agent_command = base_command + [
            "agent", "--local",
            "--session-key", session_key,
            "--model", runtime_model,
            "--message-file", str(instruction_path),
            "--timeout", str(self.timeout),
            "--verbose", "on" if self.debug else "off",
            "--json",
        ]
        if self.thinking is not None:
            agent_command.extend(["--thinking", self.thinking])

        commands = (
            ("mcp add", mcp_command, None, 60),
            ("config patch", base_command + ["config", "patch", "--stdin"],
             json.dumps(config), 60),
            ("agent", agent_command, None, self.timeout + 30),
        )
        results: dict[str, subprocess.CompletedProcess[str]] = {}

        try:
            with temporary_environment(runtime_environment):
                for name, command, input_text, command_timeout in commands:
                    result = subprocess.run(
                        command,
                        input=input_text,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=command_timeout,
                        check=False,
                    )
                    results[name] = result
                    if result.returncode != 0:
                        detail = _error_detail(result)
                        metadata["runtime_error"] = f"OpenClaw {name} failed"
                        if detail:
                            metadata["runtime_error"] += f": {detail}"
                        break
        except subprocess.TimeoutExpired as error:
            metadata["runtime_error"] = f"OpenClaw timed out after {error.timeout}s"
        except (OSError, subprocess.SubprocessError) as error:
            metadata["runtime_error"] = f"OpenClaw could not run: {error}"

        agent_result = results.get("agent")
        if agent_result is not None:
            metadata["returncode"] = agent_result.returncode

        # The agent output plus native session JSONL is sufficient for scoring and
        # debugging. Setup-command chatter is only included when setup fails.
        trace_parts: list[str] = []
        if agent_result is not None:
            trace_parts.extend([
                "openclaw_agent_stdout:", redact_sensitive(agent_result.stdout),
                "openclaw_agent_stderr:", redact_sensitive(agent_result.stderr),
            ])
        elif results:
            failed_result = list(results.values())[-1]
            trace_parts.extend([
                "openclaw_setup_stdout:", redact_sensitive(failed_result.stdout),
                "openclaw_setup_stderr:", redact_sensitive(failed_result.stderr),
            ])

        for session_path in sorted(home_dir.glob(".openclaw*/agents/**/*.jsonl")):
            trace_parts.extend([
                f"openclaw_session: {session_path}",
                redact_sensitive(session_path.read_text(encoding="utf-8", errors="replace")),
            ])
        combined_trace = "\n".join(trace_parts)

        tool_call_count = sum(
            combined_trace.count(marker) for marker in GAME_TOOL_MARKERS
        )
        if metadata["runtime_error"] is None and agent_result is not None:
            if tool_call_count:
                metadata["success"] = True
            else:
                metadata["runtime_error"] = (
                    "OpenClaw completed without calling a clem_game tool"
                )

        trace_path = write_text_artifact(
            output_dir=output_dir,
            filename="adapter_messages.txt",
            content=combined_trace,
        )
        if trace_path is not None:
            artifacts["adapter_messages"] = trace_path

        if self.debug:
            print(combined_trace)
        if metadata["runtime_error"]:
            print(metadata["runtime_error"])

        return AgentRunResult(bool(metadata["success"]), artifacts, metadata)
