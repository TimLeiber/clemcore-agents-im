import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from clemcore.agents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemcore.agents.adapters.openai_compatible_proxy import proxy_for_model_connection
from clemcore.agents.adapters.utils import (
    deep_merge_dicts,
    load_model_connection,
    mcp_environment,
    model_connection_environment,
    new_game_completion_path,
    parse_openclaw_agent_trace,
    read_game_completion,
    redact_sensitive,
    resolve_runtime_model,
    run_process_until_game_complete,
    temporary_environment,
    write_text_artifact,
)


SESSION_PART_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
GAME_TOOL_MARKERS = (
    "mcp__clem-game__",
    "mcp_clem_game_",
    "clem_game.start_game",
    "clem_game.submit_response",
    '"start_game"',
    '"submit_response"',
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
                 reasoning_effort: str | None = None,
                 model_connection_path: str | None = None,
                 trace_model_io: bool = True):
        self.model = model or clem_model
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.thinking = reasoning_effort if reasoning_effort is not None else thinking
        self.reasoning_effort = self.thinking
        self.verbose = verbose
        self.timeout = timeout
        self.profile = profile
        self.yolo = yolo
        self.debug = debug
        self.trace_model_io = trace_model_io
        self._model_connection = load_model_connection("openclaw", model_connection_path)
        _validate_openclaw_model_connection(self._model_connection)

    @classmethod
    def parse_agent_trace(cls,
                          episode_dir: Path,
                          metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Delegate OpenClaw-specific trace parsing to adapter utilities."""

        return parse_openclaw_agent_trace(episode_dir=episode_dir, metadata=metadata)

    def run_episode(self,
                    instruction: str,
                    output_dir: Path | str | None = None) -> AgentRunResult:
        completion_path = new_game_completion_path()
        proxy = proxy_for_model_connection(self._model_connection,
                                           completion_path,
                                           include_openrouter=True,
                                           trace_responses=self.trace_model_io,
                                           trace_requests=self.trace_model_io)

        if proxy is not None:
            with proxy:
                result = self._run_episode(
                    instruction,
                    output_dir,
                    completion_path,
                    proxy.base_url,
                )
                proxy_trace = redact_sensitive(proxy.captured_trace())
                adapter_messages = result.artifacts.get("adapter_messages")

                if proxy_trace and adapter_messages is not None:
                    trace_path = Path(adapter_messages)
                    trace_path.write_text(
                        trace_path.read_text(encoding="utf-8") + "\n" + proxy_trace,
                        encoding="utf-8"
                    )

                return result

        return self._run_episode(instruction, output_dir, completion_path, None)

    def _run_episode(self,
                     instruction: str,
                     output_dir: Path | str | None,
                     completion_path: Path,
                     proxied_base_url: str | None) -> AgentRunResult:
        artifacts: dict[str, Path | str | int | float | bool | None] = {}
        metadata: dict[str, Any] = {
            "adapter": "openclaw",
            "reasoning_effort": self.reasoning_effort,
            "resolved_backend": (self._model_connection or {}).get("backend"),
            "gateway_base_url": (self._model_connection or {}).get("base_url"),
            "compatibility_proxy_base_url": proxied_base_url,
            "tool_choice": (self._model_connection or {}).get("tool_choice"),
            "request_body_overrides": (
                (self._model_connection or {}).get("request_body_overrides") or {}
            ),
            "verify_tls": (self._model_connection or {}).get("verify_tls"),
            "trace_model_io": self.trace_model_io,
            "success": False,
            "returncode": None,
            "runtime_error": None,
            "game_completed": False,
            "terminated_after_game": False,
        }

        try:
            runtime_model = resolve_runtime_model(
                model_connection=self._model_connection,
                model=self.model,
                harness_name="OpenClawHarness",
            )
            runtime_model = (
                (self._model_connection or {}).get("runtime_model")
                or runtime_model
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
        bridge_environment["CLEM_GAME_COMPLETION_PATH"] = str(completion_path)
        mcp_command = base_command + [
            "mcp", "add", "clem_game",
            "--command", "python",
            "--arg", "-m",
            "--arg", "clemcore.agents.mcp.bridge",
            "--cwd", "/opt/clemcore",
            "--include", "start_game,submit_response",
            "--no-probe",
        ]
        for name, value in sorted(bridge_environment.items()):
            mcp_command.extend(["--env", f"{name}={value}"])

        config = (self._model_connection or {}).get("openclaw_config_patch", {})
        config = dict(config) if isinstance(config, dict) else {}

        if proxied_base_url is not None:
            provider_name = (self._model_connection or {}).get("openclaw_provider")

            if provider_name:
                config = deep_merge_dicts(config, {
                    "models": {
                        "providers": {
                            str(provider_name): {"baseUrl": proxied_base_url},
                        },
                    },
                })
        # Keep OpenClaw's default tool inventory available in YOLO mode so that
        # models can use standard tools in addition to the Clem MCP tools.
        config = deep_merge_dicts(config, {
            # The native OpenRouter model catalog is supplied by this plugin, so retain it while excluding unrelated bundled plugins.
            "plugins": {"enabled": True, "allow": ["openrouter", "duckduckgo", "browser"]},
            "browser": {"enabled": True, "headless": True, "noSandbox": True},
            "tools": {"web": {"search": {"enabled": True, "provider": "duckduckgo"}}},
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
        ]
        if self.trace_model_io:
            agent_command.append("--json")
        if self.thinking is not None:
            agent_command.extend(["--thinking", self.thinking])

        commands = (
            ("mcp add", mcp_command, None, 60),
            ("config patch", base_command + ["config", "patch", "--stdin"],
             json.dumps(config), 60),
            ("agent", agent_command, None, self.timeout + 30),
        )
        results: dict[str, subprocess.CompletedProcess[str]] = {}
        terminated_after_game = False

        try:
            with temporary_environment(runtime_environment):
                for name, command, input_text, command_timeout in commands:
                    if name == "agent":
                        result, terminated_after_game = run_process_until_game_complete(
                            command,
                            completion_path=completion_path,
                            timeout=command_timeout,
                        )
                    else:
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
                    completion = read_game_completion(completion_path)
                    game_completed = bool(
                        completion
                        and completion.get("done") is True
                        and completion.get("control_failure") is not True
                    )
                    if result.returncode != 0 and not (
                        name == "agent" and (terminated_after_game or game_completed)
                    ):
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
        metadata["terminated_after_game"] = terminated_after_game

        # The agent output plus native session JSONL is sufficient for scoring and
        # debugging. Setup-command chatter is only included when setup fails.
        trace_parts: list[str] = [
            "trace_settings:",
            json.dumps({"model_io": self.trace_model_io}),
        ]
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
        metadata["tool_call_count_hint"] = tool_call_count
        completion = read_game_completion(completion_path)
        metadata["game_completed"] = bool(
            completion
            and completion.get("done") is True
            and completion.get("control_failure") is not True
        )
        if metadata["runtime_error"] is None and agent_result is not None:
            if metadata["game_completed"]:
                metadata["success"] = True
            else:
                metadata["runtime_error"] = (
                    "OpenClaw ended before clem_game reported done=true"
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
