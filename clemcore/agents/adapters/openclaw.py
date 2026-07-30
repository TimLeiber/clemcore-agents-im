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


class OpenClawHarness(ExternalAgentHarness):
    """Run OpenClaw through the OpenClaw CLI.

    The harness creates an isolated OpenClaw home, registers the container-side
    MCP bridge, applies benchmark configuration, and captures trace artifacts.
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
        """Configure the OpenClaw harness.

        Args:
            model: model identifier passed directly to OpenClaw
            clem_model: clembench model resolved by the outer pipeline
            mcp_url: URL forwarded to the container-side MCP bridge
            thinking: optional OpenClaw thinking level
            verbose: whether to enable verbose OpenClaw output
            timeout: maximum agent runtime in seconds
            profile: isolated OpenClaw profile name
            yolo: whether to disable OpenClaw approval gates
            model_connection_path: optional resolved model-connection file
        """

        self.model = model or clem_model
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.thinking = thinking
        self.verbose = verbose
        self.timeout = timeout
        self.profile = profile
        self.yolo = yolo
        self._model_connection = load_model_connection("openclaw", model_connection_path)

    def run_episode(self,
                    instruction: str,
                    output_dir: Path | str | None = None) -> AgentRunResult:
        """Run one OpenClaw episode.

        Args:
            instruction: task instruction passed to OpenClaw
            output_dir: optional directory for adapter artifacts

        Returns:
            the standardized OpenClaw run result
        """

        # ----- step 1 -----
        # initialize metadata and resolve the runtime configuration
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
            runtime_model = resolve_runtime_model(model_connection=self._model_connection,
                                                  model=self.model,
                                                  harness_name="OpenClawHarness")
            session_parts = [
                "clembench",
                os.environ.get("CLEM_EXPERIMENT_NAME", "experiment"),
                os.environ.get("CLEM_GAME_ID", "game"),
                str(os.getpid()),
            ]
            safe_session_parts = [SESSION_PART_PATTERN.sub("-", part).strip("-") or "x" for part in session_parts]
            session_key = "agent:clembench:" + "-".join(safe_session_parts)
            metadata["runtime_model"] = runtime_model
            metadata["session_key"] = session_key
        except Exception as error:
            metadata["runtime_error"] = str(error)
            return AgentRunResult(success=False,
                                  artifacts=artifacts,
                                  metadata=metadata)

        # ----- step 2 -----
        # create isolated runtime storage and write the instruction
        run_dir = Path(output_dir) if output_dir is not None else Path("/tmp") / f"openclaw-clembench-{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        home_dir = run_dir / "openclaw_home"
        home_dir.mkdir(parents=True, exist_ok=True)
        openclaw_log_path = run_dir / "openclaw.log"
        instruction_path = run_dir / "openclaw_instruction.txt"
        instruction_path.write_text(instruction, encoding="utf-8")
        metadata["openclaw_log_path"] = str(openclaw_log_path)

        # ----- step 3 -----
        # build the OpenClaw environment and CLI commands
        runtime_environment = model_connection_environment(self._model_connection)
        runtime_environment.update({
            "HOME": str(home_dir),
            "OPENENV_MCP_URL": self.mcp_url,
            # remove after the connection issue is fixed
            "NODE_TLS_REJECT_UNAUTHORIZED": "0",
            # maximize OpenClaw diagnostics for benchmark trace capture
            "OPENCLAW_LOG_LEVEL": "trace",
            "OPENCLAW_DEBUG_MODEL_TRANSPORT": "1",
            "OPENCLAW_DEBUG_MODEL_PAYLOAD": "full-redacted",
            "OPENCLAW_DEBUG_SSE": "peek",
            "OPENCLAW_DEBUG_CODE_MODE": "1",
        })
        base_command = [
            "openclaw",
            "--no-color",
            "--profile",
            self.profile,
            "--log-level",
            "trace",
        ]
        bridge_environment = mcp_environment(self.mcp_url)
        mcp_command = base_command + [
            "mcp",
            "add",
            "clem_game",
            "--command",
            "python",
            "--arg",
            "-m",
            "--arg",
            "clemcore.agents.mcp.bridge",
            "--cwd",
            "/opt/clemcore",
            "--include",
            "start_game,submit_response,get_state",
            "--no-probe",
        ]

        for name, value in sorted(bridge_environment.items()):
            mcp_command.extend(["--env", f"{name}={value}"])

        openclaw_config: dict[str, Any] = {}
        connection_patch = (self._model_connection or {}).get("openclaw_config_patch", {})

        if isinstance(connection_patch, dict):
            openclaw_config = deep_merge_dicts(openclaw_config, connection_patch)

        if self.yolo:
            openclaw_config["tools"] = {
                "exec": {
                    "security": "full",
                    "ask": "off",
                },
            }

        openclaw_config["logging"] = {
            "level": "trace",
            "consoleLevel": "trace",
            "consoleStyle": "json",
            "file": str(openclaw_log_path),
            "redactSensitive": "tools",
        }
        config_command = base_command + ["config", "patch", "--stdin"]
        config_patch = json.dumps(openclaw_config)
        models_command = base_command + ["models", "set", runtime_model]
        agent_command = base_command + [
            "agent",
            "--local",
            "--session-key",
            session_key,
            "--model",
            runtime_model,
            "--message-file",
            str(instruction_path),
            "--timeout",
            str(self.timeout),
            "--json",
        ]

        if self.thinking is not None:
            agent_command.extend(["--thinking", self.thinking])

        if self.verbose:
            agent_command.extend(["--verbose", "on"])

        # ----- step 4 -----
        # configure OpenClaw and run the agent
        with temporary_environment(runtime_environment):
            mcp_add = subprocess.run(mcp_command,
                                     text=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     timeout=60)
            config_result = subprocess.run(config_command,
                                           input=config_patch,
                                           text=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           timeout=60)
            models_result = subprocess.run(models_command,
                                           text=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           timeout=60)
            agent_result = subprocess.run(agent_command,
                                          text=True,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          timeout=self.timeout + 30)

        # ----- step 5 -----
        # collect OpenClaw logs and build the redacted trace
        artifact_candidates = [openclaw_log_path]
        artifact_candidates.extend(sorted(home_dir.glob(".openclaw*/agents/**/*.jsonl")))
        artifact_candidates.extend(sorted(home_dir.glob(".openclaw*/**/*.jsonl")))
        artifact_candidates.extend(sorted(home_dir.glob(".openclaw*/**/*.log")))
        artifact_candidates.extend(sorted(Path("/tmp").glob("openclaw*/openclaw*.log")))
        artifact_candidates.extend(sorted(Path("/tmp").glob("openclaw*/**/*.log")))
        artifact_sections = []
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
                artifact_text = artifact_path.read_text(encoding="utf-8", errors="replace")
            except Exception as error:
                artifact_text = f"[failed to read {artifact_path}: {error}]"

            artifact_sections.extend([
                f"openclaw_artifact: {artifact_path}",
                artifact_text,
            ])

        combined_trace = redact_sensitive("\n".join([
            "openclaw_mcp_add_command:",
            " ".join(mcp_command),
            "openclaw_mcp_add_stdout:",
            mcp_add.stdout,
            "openclaw_mcp_add_stderr:",
            mcp_add.stderr,
            "openclaw_yolo_config_command:",
            " ".join(config_command),
            "openclaw_yolo_config_patch:",
            config_patch,
            "openclaw_yolo_config_stdout:",
            config_result.stdout,
            "openclaw_yolo_config_stderr:",
            config_result.stderr,
            "openclaw_models_set_command:",
            " ".join(models_command),
            "openclaw_models_set_stdout:",
            models_result.stdout,
            "openclaw_models_set_stderr:",
            models_result.stderr,
            "openclaw_debug_environment:",
            json.dumps({
                "OPENCLAW_LOG_LEVEL": runtime_environment.get("OPENCLAW_LOG_LEVEL"),
                "OPENCLAW_DEBUG_MODEL_TRANSPORT": runtime_environment.get("OPENCLAW_DEBUG_MODEL_TRANSPORT"),
                "OPENCLAW_DEBUG_MODEL_PAYLOAD": runtime_environment.get("OPENCLAW_DEBUG_MODEL_PAYLOAD"),
                "OPENCLAW_DEBUG_SSE": runtime_environment.get("OPENCLAW_DEBUG_SSE"),
                "OPENCLAW_DEBUG_CODE_MODE": runtime_environment.get("OPENCLAW_DEBUG_CODE_MODE"),
                "OPENCLAW_LOG_FILE": str(openclaw_log_path),
                "OPENCLAW_THINKING": self.thinking,
            }, indent=2),
            "openclaw_agent_command:",
            " ".join(agent_command),
            "openclaw_agent_stdout:",
            agent_result.stdout,
            "openclaw_agent_stderr:",
            agent_result.stderr,
            *artifact_sections,
        ]))

        # ----- step 6 -----
        # derive the run outcome from command results and visible tool calls
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
        incomplete_after_tools = (
            '"kind": "incomplete_turn"' in combined_trace
            and '"failures": 0' in combined_trace
            and "clem_game__start_game" in combined_trace
            and "clem_game__submit_response" in combined_trace
        )
        metadata["returncode"] = agent_result.returncode
        metadata["tool_call_count_hint"] = tool_call_count_hint
        metadata["openclaw_incomplete_after_tools"] = incomplete_after_tools
        metadata["success"] = (
            mcp_add.returncode == 0
            and config_result.returncode == 0
            and models_result.returncode == 0
            and agent_result.returncode == 0
            and (tool_call_count_hint > 0 or incomplete_after_tools)
        )

        if mcp_add.returncode != 0:
            metadata["runtime_error"] = "openclaw mcp add failed"
        elif config_result.returncode != 0:
            metadata["runtime_error"] = "openclaw yolo config patch failed"
        elif models_result.returncode != 0:
            metadata["runtime_error"] = "openclaw models set failed"
        elif agent_result.returncode != 0:
            metadata["runtime_error"] = "openclaw agent failed"
        elif tool_call_count_hint <= 0 and not incomplete_after_tools:
            metadata["runtime_error"] = "openclaw completed without visible clem_game MCP tool calls"

        # ----- step 7 -----
        # write the redacted trace and metadata
        trace_path = write_text_artifact(output_dir=output_dir,
                                         filename="adapter_messages.txt",
                                         content=combined_trace)
        metadata_path = write_text_artifact(output_dir=output_dir,
                                            filename="openclaw_run_meta.json",
                                            content=json.dumps(metadata,
                                                               indent=2,
                                                               sort_keys=True))

        if trace_path is not None:
            artifacts["adapter_messages"] = trace_path

        if metadata_path is not None:
            artifacts["openclaw_run_meta"] = metadata_path

        print(combined_trace)

        return AgentRunResult(success=bool(metadata["success"]),
                              artifacts=artifacts,
                              metadata=metadata)
