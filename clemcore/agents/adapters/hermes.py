import re
import subprocess
from pathlib import Path

from clemcore.agents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemcore.agents.adapters.utils import (
    load_model_connection,
    mcp_environment,
    model_connection_environment,
    redact_sensitive,
    resolve_runtime_model,
    temporary_environment,
    write_text_artifact,
)


class HermesHarness(ExternalAgentHarness):
    """Run Hermes Agent through the Hermes CLI.

    The harness registers the container-side MCP bridge, enables detailed
    trace output, runs one Hermes chat, and exports its session artifacts.
    """

    def __init__(self,
                 model: str | None = None,
                 clem_model: str | None = None,
                 provider: str = "openrouter",
                 mcp_url: str = "http://host.docker.internal:8001/mcp",
                 max_turns: int = 20,
                 yolo: bool = True,
                 reasoning_effort: str | None = None,
                 model_connection_path: str | None = None):
        """Configure the Hermes harness.

        Args:
            model: model identifier passed directly to Hermes
            clem_model: clembench model resolved by the outer pipeline
            provider: Hermes model provider
            mcp_url: URL forwarded to the container-side MCP bridge
            max_turns: maximum number of Hermes turns
            yolo: whether to disable Hermes approval gates
            reasoning_effort: model reasoning effort configured in Hermes
            model_connection_path: optional resolved model-connection file
        """

        self.model = model or clem_model
        self.clem_model = clem_model
        self.provider = provider
        self.mcp_url = mcp_url
        self.max_turns = max_turns
        self.yolo = yolo
        self.reasoning_effort = reasoning_effort
        self._model_connection = load_model_connection("hermes", model_connection_path)

    def run_episode(self,
                    instruction: str,
                    output_dir: Path | str | None = None) -> AgentRunResult:
        """Run one Hermes episode.

        Args:
            instruction: task instruction passed to Hermes
            output_dir: optional directory for adapter artifacts

        Returns:
            the standardized Hermes run result
        """

        # ----- step 1 -----
        # initialize metadata and resolve the runtime model
        metadata = {
            "adapter": "hermes",
            "model": self.model,
            "clem_model": self.clem_model,
            "runtime_model": None,
            "resolved_backend": (self._model_connection or {}).get("backend"),
            "provider": self.provider,
            "mcp_url": self.mcp_url,
            "max_turns": self.max_turns,
            "reasoning_effort": self.reasoning_effort,
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
            runtime_model = resolve_runtime_model(model_connection=self._model_connection,
                                                  model=self.model,
                                                  harness_name="HermesHarness")
            metadata["runtime_model"] = runtime_model
        except Exception as error:
            metadata["runtime_error"] = str(error)
            return AgentRunResult(success=False,
                                  artifacts=artifacts,
                                  metadata=metadata)

        runtime_environment = model_connection_environment(self._model_connection)

        # ----- step 2 -----
        # build the MCP registration, trace configuration, and chat commands
        bridge_environment = mcp_environment(self.mcp_url, include_pythonpath=True)
        register_command = [
            "hermes",
            "mcp",
            "add",
            "clem_game",
            "--command",
            "python",
            "--env",
            f"PYTHONPATH={bridge_environment['PYTHONPATH']}",
            f"OPENENV_MCP_URL={bridge_environment['OPENENV_MCP_URL']}",
            f"CLEM_EXPERIMENT_NAME={bridge_environment.get('CLEM_EXPERIMENT_NAME', '')}",
            f"CLEM_GAME_ID={bridge_environment.get('CLEM_GAME_ID', '')}",
            f"CLEM_OPENENV_SESSION_PATH={bridge_environment.get('CLEM_OPENENV_SESSION_PATH', '')}",
            "--args",
            "-m",
            "clemcore.agents.mcp.bridge",
        ]
        config_commands = [
            ["hermes", "config", "set", "display.show_reasoning", "true"],
            ["hermes", "config", "set", "display.streaming", "true"],
            ["hermes", "config", "set", "display.tool_progress", "verbose"],
        ]

        if self.reasoning_effort is not None:
            config_commands.append([
                "hermes", "config", "set", "agent.reasoning_effort",
                self.reasoning_effort,
            ])

        chat_command = [
            "hermes",
            "chat",
            "--provider",
            self.provider,
            "--model",
            runtime_model,
            "--max-turns",
            str(self.max_turns),
            "--ignore-rules",
            "--verbose",
        ]

        if self.yolo:
            chat_command.append("--yolo")

        chat_command.extend(["-q", instruction])

        # ----- step 3 -----
        # configure Hermes and run the agent
        with temporary_environment(runtime_environment):
            register = subprocess.run(register_command,
                                      input="y\n",
                                      text=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      timeout=60)
            config_results = []

            for command in config_commands:
                config_results.append(
                    subprocess.run(command,
                                   text=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   timeout=30)
                )

            chat = subprocess.run(chat_command,
                                  text=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  timeout=600)

        # ----- step 4 -----
        # combine the trace and derive the run outcome
        combined_trace = redact_sensitive("\n".join([
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
        session_text = chat.stdout + "\n" + chat.stderr
        session_match = re.search(r"^Session:\s*(\S+)",
                                  session_text,
                                  flags=re.MULTILINE)

        if session_match is None:
            session_match = re.search(r"hermes --resume\s+(\S+)", session_text)

        if session_match is not None:
            metadata["hermes_session_id"] = session_match.group(1)

        metadata["returncode"] = chat.returncode
        metadata["tool_call_count_hint"] = (
                chat.stdout.count("Tool call: mcp__clem_game__")
                + chat.stderr.count("Tool call: mcp__clem_game__")
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

        # ----- step 5 -----
        # write the trace and export available Hermes session artifacts
        trace_path = write_text_artifact(output_dir=output_dir,
                                         filename="adapter_messages.txt",
                                         content=combined_trace)

        if trace_path is not None:
            artifacts["adapter_messages"] = trace_path

        session_id = metadata["hermes_session_id"]

        if output_dir is not None and session_id:
            session_export_path = Path(output_dir) / "hermes_session_export.jsonl"
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
            session_export_meta_path = write_text_artifact(
                output_dir=output_dir,
                filename="hermes_session_export_meta.txt",
                content="\n".join([
                    "hermes_sessions_export_returncode:",
                    str(session_export.returncode),
                    "hermes_sessions_export_stdout:",
                    session_export.stdout,
                    "hermes_sessions_export_stderr:",
                    session_export.stderr,
                ]),
            )

            if session_export_meta_path is not None:
                artifacts["hermes_session_export_meta"] = session_export_meta_path

            if session_export_path.exists():
                artifacts["hermes_session_export"] = session_export_path

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
            hermes_log_path = write_text_artifact(
                output_dir=output_dir,
                filename="hermes_agent_log.txt",
                content="\n".join([
                    "hermes_logs_returncode:",
                    str(hermes_log.returncode),
                    "hermes_logs_stdout:",
                    hermes_log.stdout,
                    "hermes_logs_stderr:",
                    hermes_log.stderr,
                ]),
            )

            if hermes_log_path is not None:
                artifacts["hermes_agent_log"] = hermes_log_path

        print(combined_trace)

        return AgentRunResult(success=bool(metadata["success"]),
                              artifacts=artifacts,
                              metadata=metadata)
