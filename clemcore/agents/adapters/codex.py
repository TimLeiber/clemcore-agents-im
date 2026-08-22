import json
import os
import subprocess
from pathlib import Path
from typing import Any

from clemcore.agents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemcore.agents.adapters.openai_compatible_proxy import proxy_for_model_connection
from clemcore.agents.adapters.utils import (
    load_model_connection,
    mcp_environment,
    model_connection_environment,
    new_game_completion_path,
    parse_codex_agent_trace,
    read_game_completion,
    resolve_runtime_model,
    run_process_until_game_complete,
    temporary_environment,
    write_text_artifact,
)


class CodexHarness(ExternalAgentHarness):
    """Run Codex through the Codex CLI.

    The harness writes an isolated Codex configuration that registers the
    container-side MCP bridge, then runs one non-interactive `codex exec`.
    """

    def __init__(self,
                 model: str | None = None,
                 clem_model: str | None = None,
                 mcp_url: str = "http://host.docker.internal:8001/mcp",
                 sandbox: str = "full_access",
                 reasoning_effort: str | None = None,
                 model_connection_path: str | None = None,
                 trace_model_io: bool = True):
        """Configure the Codex harness.

        Args:
            model: model identifier passed directly to Codex
            clem_model: clembench model resolved by the outer pipeline
            mcp_url: URL forwarded to the container-side MCP bridge
            sandbox: Codex sandbox policy
            reasoning_effort: model reasoning effort passed to Codex
            model_connection_path: optional resolved model-connection file
            trace_model_io: whether to record model requests and responses
        """

        self.model = model or clem_model or "gpt-5.4"
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.sandbox = sandbox
        self.reasoning_effort = reasoning_effort
        self.trace_model_io = trace_model_io
        self._model_connection = load_model_connection("codex", model_connection_path)

    @classmethod
    def parse_agent_trace(cls,
                          episode_dir: Path,
                          metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Delegate Codex-specific trace parsing to adapter utilities."""

        return parse_codex_agent_trace(episode_dir=episode_dir, metadata=metadata)

    def run_episode(self,
                    instruction: str,
                    output_dir: Path | str | None = None) -> AgentRunResult:
        completion_path = new_game_completion_path()
        resolved_backend = (self._model_connection or {}).get("backend")
        proxy = proxy_for_model_connection(self._model_connection,
                                           completion_path,
                                           include_openrouter=True,
                                           trace_responses=self.trace_model_io,
                                           trace_requests=self.trace_model_io)

        if proxy is not None:
            with proxy:
                return self._run_episode(
                    instruction,
                    output_dir,
                    completion_path,
                    proxy.base_url,
                )

        return self._run_episode(instruction, output_dir, completion_path, None)

    def _run_episode(self,
                     instruction: str,
                     output_dir: Path | str | None,
                     completion_path: Path,
                     proxied_base_url: str | None) -> AgentRunResult:
        """Run one Codex episode.

        Args:
            instruction: task instruction passed to Codex
            output_dir: optional directory for adapter artifacts

        Returns:
            the standardized Codex run result
        """

        # ----- step 1 -----
        # resolve the model, provider environment, and sandbox
        runtime_model = resolve_runtime_model(model_connection=self._model_connection,
                                              model=self.model,
                                              harness_name="CodexHarness")
        resolved_backend = (self._model_connection or {}).get("backend")
        codex_model = (
            "clem-openrouter-model"
            if resolved_backend == "openrouter" and proxied_base_url is not None
            else runtime_model
        )
        runtime_environment = model_connection_environment(self._model_connection)
        runtime_base_url = proxied_base_url

        if runtime_base_url is None and self._model_connection and self._model_connection.get("base_url"):
            runtime_base_url = str(self._model_connection["base_url"])

        sandbox_modes = {
            "read_only": "read-only",
            "workspace_write": "workspace-write",
            "full_access": "danger-full-access",
        }

        if self.sandbox not in sandbox_modes:
            raise ValueError(
                f"Unknown Codex sandbox '{self.sandbox}'. "
                f"Expected one of: {sorted(sandbox_modes)}"
            )

        # ----- step 2:
        # define configuration for Codex CLI, e.g. tool access, permissions, model provider, model reasoning effort etc.
        # by creating temporary dedicated config file. This file is called ~/.codex/config.toml for codex.
        # Alternatively one could pass all configurations as parameters to the harness during use
        # -----

        # write the Codex configuration that registers the MCP bridge
        config_dir = Path.home() / ".codex"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.toml"
        bridge_environment = mcp_environment(self.mcp_url, include_pythonpath=True)
        bridge_environment["CLEM_GAME_COMPLETION_PATH"] = str(completion_path)
        environment_lines = "\n".join(
            f"{key} = {json.dumps(value)}" for key, value in sorted(bridge_environment.items())
        )

        # set approvals in ~/.codex/config.toml to allow all tool use and no requirement for permissions
        # (effectively is YOLO, but explicit)
        config_lines = [
            f"model = {json.dumps(codex_model)}",
            'approval_policy = "never"',
            f"sandbox_mode = {json.dumps(sandbox_modes[self.sandbox])}",
            'web_search = "live"',
        ]

        if self.reasoning_effort is not None:
            config_lines.append(
                f"model_reasoning_effort = {json.dumps(self.reasoning_effort)}"
            )

        if runtime_base_url:
            if resolved_backend == "openrouter" and proxied_base_url is not None:
                config_lines.extend([
                    'model_provider = "openrouter_proxy"',
                    "",
                    "[model_providers.openrouter_proxy]",
                    'name = "openrouter_proxy"',
                    f"base_url = {json.dumps(runtime_base_url)}",
                    'env_key = "OPENROUTER_API_KEY"',
                    'wire_api = "responses"',
                ])
            elif "openrouter.ai" in runtime_base_url:
                config_lines.extend([
                    'model_provider = "openrouter"',
                    "",
                    "[model_providers.openrouter]",
                    'name = "openrouter"',
                    f"base_url = {json.dumps(runtime_base_url)}",
                    'env_key = "OPENROUTER_API_KEY"',
                ])
            else:
                config_lines.extend([
                    'model_provider = "openai_api"',
                    "",
                    "[model_providers.openai_api]",
                    f"name = {json.dumps('OpenAI API')}",
                    f"base_url = {json.dumps(runtime_base_url)}",
                    f"env_key = {json.dumps('OPENAI_API_KEY')}",
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
            'enabled_tools = ["start_game", "submit_response"]',
            'default_tools_approval_mode = "approve"',
            "",
            "[mcp_servers.clem_game.env]",
            environment_lines,
            "",
        ])
        config_path.write_text("\n".join(config_lines), encoding="utf-8")

        # ----- step 3 -----
        # build the Codex instruction and command
        last_message_path = config_dir / f"last_message_{os.getpid()}.txt"

        if last_message_path.exists():
            last_message_path.unlink()

        command = ["codex", "exec", "--strict-config", "--json", "--cd", "/workspace", "--skip-git-repo-check",
                   "--ephemeral", "--output-last-message", str(last_message_path), "-"]

        # ----- step 4 -----
        # verify that Codex can read the generated MCP configuration
        mcp_list_stdout = ""
        mcp_list_stderr = ""
        mcp_list_returncode = None

        try:
            with temporary_environment(runtime_environment):
                mcp_list = subprocess.run(["codex", "mcp", "list"],
                                          text=True,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          cwd="/workspace",
                                          check=False)

            mcp_list_stdout = mcp_list.stdout
            mcp_list_stderr = mcp_list.stderr
            mcp_list_returncode = mcp_list.returncode
        except Exception as error:
            mcp_list_stderr = str(error)

        # ----- step 5 -----
        # run Codex and capture its JSONL trace
        metadata = {
            "adapter": "codex",
            "model": self.model,
            "clem_model": self.clem_model,
            "runtime_model": runtime_model,
            "codex_model": codex_model,
            "resolved_backend": resolved_backend,
            "gateway_base_url": (
                (self._model_connection or {}).get("base_url") or runtime_base_url
            ),
            "compatibility_proxy_base_url": proxied_base_url,
            "tool_choice": (self._model_connection or {}).get("tool_choice"),
            "request_body_overrides": (
                (self._model_connection or {}).get("request_body_overrides") or {}
            ),
            "verify_tls": (self._model_connection or {}).get("verify_tls"),
            "mcp_url": self.mcp_url,
            "codex_config": str(config_path),
            "sandbox": self.sandbox,
            "reasoning_effort": self.reasoning_effort,
            "trace_model_io": self.trace_model_io,
            "success": False,
            "runtime_error": None,
            "returncode": None,
            "final_response": None,
            "game_completed": False,
            "terminated_after_game": False,
        }
        transcript = [
            f"codex_config: {config_path}",
            "codex_config_toml:",
            config_path.read_text(encoding="utf-8"),
            f"model: {self.model}",
            f"runtime_model: {runtime_model}",
            f"codex_model: {codex_model}",
            f"gateway_base_url: {metadata['gateway_base_url']}",
            f"compatibility_proxy_base_url: {proxied_base_url}",
            f"sandbox: {self.sandbox}",
            "trace_settings:",
            json.dumps({"model_io": self.trace_model_io}),
            "codex_mcp_list_returncode:",
            str(mcp_list_returncode),
            "codex_mcp_list_stdout:",
            mcp_list_stdout,
            "codex_mcp_list_stderr:",
            mcp_list_stderr,
            "codex_command:",
            " ".join(command),
        ]

        try:
            with temporary_environment(runtime_environment):
                completed, terminated_after_game = run_process_until_game_complete(
                    command,
                    completion_path=completion_path,
                    input_text=instruction,
                    cwd="/workspace",
                )

            metadata["returncode"] = completed.returncode
            metadata["terminated_after_game"] = terminated_after_game
            transcript.extend([
                "codex_stdout_jsonl:",
                completed.stdout,
                "codex_stderr:",
                completed.stderr,
            ])

            if last_message_path.exists():
                final_response = last_message_path.read_text(encoding="utf-8")
                metadata["final_response"] = final_response
                transcript.extend(["final_response:", final_response])
        except Exception as error:
            metadata["runtime_error"] = str(error)
            transcript.append(f"agent_runtime_error: {error}")
            print(f"agent_runtime_error: {error}")

        completion = read_game_completion(completion_path)
        metadata["game_completed"] = bool(
            completion
            and completion.get("done") is True
            and completion.get("control_failure") is not True
        )
        metadata["success"] = metadata["game_completed"]

        if metadata["runtime_error"] is None and not metadata["game_completed"]:
            metadata["runtime_error"] = "Codex ended before clem_game reported done=true"

        # ----- step 6 -----
        # write the trace and generated configuration
        trace_text = "\n".join(transcript)
        print(trace_text)
        artifacts = {}
        messages_path = write_text_artifact(output_dir=output_dir,
                                            filename="adapter_messages.txt",
                                            content=trace_text)
        saved_config_path = write_text_artifact(output_dir=output_dir,
                                                filename="codex_config.toml",
                                                content=config_path.read_text(encoding="utf-8"))

        if messages_path is not None:
            artifacts["adapter_messages"] = messages_path

        if saved_config_path is not None:
            artifacts["codex_config"] = saved_config_path

        return AgentRunResult(success=bool(metadata["success"]),
                              artifacts=artifacts,
                              metadata=metadata)
