import asyncio
import json
import sys
import subprocess
import tempfile
import time
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import requests

from clemcore.agents.adapters import model_connection
from clemcore.agents.adapters.claude_code import (
    ClaudeCodeHarness,
    _anthropic_proxy_base_url,
)
from clemcore.agents.adapters.hermes import HermesHarness
from clemcore.agents.adapters.openai_compatible_proxy import (
    OpenAICompatibleProxy,
    _resolve_namespaced_tool,
)
from clemcore.agents.adapters.openclaw import (
    OpenClawHarness,
    _validate_openclaw_model_connection,
)
from clemcore.agents.mcp.bridge import (
    CONTROL_FAILURE_RESPONSE,
    OpenEnvMCPClient,
    create_mcp_bridge,
)
from clemcore.agents.mcp.server import result_run_dir_name
from clemcore.agents.run_pipeline.utils import write_agent_artifacts
from clemcore.agents.adapters.utils import run_process_until_game_complete


class TestExternalAgentPipeline(unittest.TestCase):
    def test_claude_proxy_base_does_not_duplicate_v1(self):
        self.assertEqual(
            _anthropic_proxy_base_url("http://127.0.0.1:1234/api/v1"),
            "http://127.0.0.1:1234/api",
        )

    def test_claude_code_trace_parser_standardizes_sdk_messages(self):
        trace = "\n".join([
            "agent_loop_instruction_start",
            "Play through MCP tools.",
            "agent_loop_instruction_end",
            "SystemMessage(subtype='init', data={'tools': ['WebSearch', 'mcp__clem_game__start_game'], 'model': 'test-model', 'permissionMode': 'bypassPermissions'})",
            "SystemMessage(subtype='thinking_tokens', data={'estimated_tokens': 4, 'estimated_tokens_delta': 4})",
            "AssistantMessage(content=[ThinkingBlock(thinking='I should search.', signature='')], message_id='message-1')",
            "AssistantMessage(content=[TextBlock(text='I will verify this.')], message_id='message-1')",
            "AssistantMessage(content=[ToolUseBlock(id='call-1', name='WebSearch', input={'query': 'example'})], message_id='message-1')",
            "UserMessage(content=[ToolResultBlock(tool_use_id='call-1', content='search result', is_error=None)])",
            "AssistantMessage(content=[TextBlock(text='DONE')], message_id='message-2')",
            "ResultMessage(subtype='success', duration_ms=20, is_error=False, num_turns=2, result='DONE')",
            "success: True",
        ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(trace, encoding="utf-8")
            parsed = ClaudeCodeHarness.parse_agent_trace(episode_dir)

        event_types = [event["type"] for event in parsed["events"]]
        self.assertEqual(parsed["backend"], "claude_code")
        self.assertIn("instruction", event_types)
        self.assertIn("reasoning", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("tool_result", event_types)
        self.assertIn("tool_preamble", event_types)
        self.assertIn("assistant_text", event_types)
        self.assertEqual(parsed["runtime"]["model"], "test-model")
        self.assertEqual(parsed["result"]["num_turns"], 2)
        self.assertEqual(parsed["capture"]["thinking_tokens"]["estimated_total"], 4)

    def test_hermes_trace_parser_standardizes_verbose_cli_output(self):
        trace = "\n".join([
            "agent_loop_instruction_start",
            "Play through MCP tools.",
            "agent_loop_instruction_end",
            "hermes_chat_command:",
            "hermes chat --provider openrouter --model test-model --yolo -q <instruction>",
            "hermes_chat_stdout:",
            "Query: Play through MCP tools.",
            "Initializing agent...",
            "🤖 AI Agent initialized with model: test-model",
            "🛠️  Final tool selection (2 tools): web_search, mcp__clem_game__start_game",
            "┌─ Reasoning ─────────┐",
            "I should start the game.",
            "└─────────────────────┘",
            "  📞 Tool 1: mcp__clem_game__start_game([])",
            "     Args: {}",
            "  ┊ ⚡ preparing mcp__clem_game__start_game…",
            "  ✅ Tool 1 completed in 0.10s",
            "     Result: {\"structuredContent\": {\"context\": {\"role\": \"user\", \"content\": \"Game prompt\"}, \"done\": false}}",
            "hermes_chat_stderr:",
            "success: True",
        ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(trace, encoding="utf-8")
            parsed = HermesHarness.parse_agent_trace(episode_dir)

        event_types = [event["type"] for event in parsed["events"]]
        self.assertEqual(parsed["backend"], "hermes")
        self.assertIn("instruction", event_types)
        self.assertIn("reasoning", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("tool_result", event_types)
        self.assertEqual(parsed["runtime"]["model"], "test-model")
        self.assertTrue(parsed["result"]["success"])
        self.assertEqual(parsed["capture"]["tool_definitions"]["count"], 2)

    def test_hermes_trace_parser_prefers_session_export(self):
        trace = "\n".join([
            "agent_loop_instruction_start",
            "Play through MCP tools.",
            "agent_loop_instruction_end",
            "hermes_chat_stdout:",
            "Query: Play through MCP tools.",
            "Initializing agent...",
            "hermes_chat_stderr:",
            "success: True",
        ])
        session = {
            "id": "session-1",
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Play through MCP tools."},
                {
                    "role": "assistant",
                    "content": "I will verify this.",
                    "reasoning": "I should search.",
                    "tool_calls": [{
                        "id": "call-1",
                        "function": {
                            "name": "web_search",
                            "arguments": "{\"query\": \"example\"}"
                        }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "tool_name": "web_search",
                    "content": "search result"
                }
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(trace, encoding="utf-8")
            (episode_dir / "hermes_session_export.jsonl").write_text(
                json.dumps(session) + "\n",
                encoding="utf-8",
            )
            parsed = HermesHarness.parse_agent_trace(episode_dir)

        event_types = [event["type"] for event in parsed["events"]]
        self.assertIn("reasoning", event_types)
        self.assertIn("tool_preamble", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("tool_result", event_types)
        self.assertEqual(parsed["runtime"]["session_id"], "session-1")
        self.assertEqual(
            parsed["capture"]["semantic_events"]["source"],
            "hermes_session_export",
        )

    def test_hermes_trace_parser_pairs_concurrent_tools(self):
        trace = "\n".join([
            "hermes_chat_stdout:",
            "┌─ Reasoning ─────────┐",
            "I should search twice.",
            "└─────────────────────┘",
            "  ⚡ Concurrent: 2 tool calls — web_search, web_search",
            "  📞 Tool 1: web_search(['query'])",
            "     Args: {\"query\": \"first\"}",
            "  📞 Tool 2: web_search(['query'])",
            "     Args: {\"query\": \"second\"}",
            "  ┊ 🔍 search first",
            "  ✅ Tool 1 completed in 0.10s",
            "     Result: {\"output\": \"first result\"}",
            "  ┊ 🔍 search second",
            "  ✅ Tool 2 completed in 0.20s",
            "     Result: {\"output\": \"second result\"}",
            "hermes_chat_stderr:",
            "success: True",
        ])

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(trace, encoding="utf-8")
            parsed = HermesHarness.parse_agent_trace(episode_dir)

        calls = [event for event in parsed["events"] if event["type"] == "tool_call"]
        results = [event for event in parsed["events"] if event["type"] == "tool_result"]
        self.assertEqual([event["arguments"]["query"] for event in calls], ["first", "second"])
        self.assertEqual([event["call_id"] for event in results], ["hermes-call-1", "hermes-call-2"])

    def test_openclaw_trace_parser_standardizes_native_session(self):
        stdout = {
            "payloads": [{"text": "DONE"}],
            "meta": {
                "durationMs": 50,
                "aborted": False,
                "agentMeta": {
                    "sessionId": "session-1",
                    "provider": "openrouter",
                    "model": "test-model",
                    "usage": {"input": 10, "output": 5}
                },
                "systemPromptReport": {
                    "systemPrompt": {"chars": 123, "hash": "prompt-hash"}
                }
            }
        }
        session = [
            {"type": "session", "id": "session-1", "cwd": "/workspace"},
            {
                "type": "model_change",
                "provider": "openrouter",
                "modelId": "test-model"
            },
            {"type": "thinking_level_change", "thinkingLevel": "high"},
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Play through MCP tools."}]
                }
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I should search."},
                        {"type": "text", "text": "I will verify this."},
                        {
                            "type": "toolCall",
                            "id": "call-1",
                            "name": "web_search",
                            "arguments": {"query": "example"}
                        }
                    ]
                }
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-1",
                    "toolName": "web_search",
                    "content": [{"type": "text", "text": "search result"}]
                }
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "DONE"}]
                }
            }
        ]
        trajectory = [{
            "type": "context.compiled",
            "data": {
                "systemPrompt": {
                    "truncated": True,
                    "originalChars": 123
                },
                "prompt": "Play through MCP tools.",
                "tools": [{"name": "trajectory_tool"}]
            }
        }]
        request = {
            "messages": [
                {"role": "system", "content": "Native OpenClaw instruction."},
                {"role": "user", "content": "Play through MCP tools."}
            ],
            "tools": [{"type": "function", "function": {"name": "web_search"}}]
        }
        trace_lines = [
            "agent_loop_instruction_start",
            "Play through MCP tools.",
            "agent_loop_instruction_end",
            "openclaw_agent_stdout:",
            json.dumps(stdout),
            "openclaw_agent_stderr:",
            "openclaw_session: /tmp/session-1.jsonl",
            *[json.dumps(record) for record in session],
            "openclaw_session: /tmp/session-1.trajectory.jsonl",
            *[json.dumps(record) for record in trajectory],
            "raw_upstream_request_1_start",
            "path: /api/v1/chat/completions",
            json.dumps(request),
            "raw_upstream_request_1_end",
            "raw_upstream_response_1_start",
            "path: /api/v1/chat/completions",
            "status: 200",
            "content_type: application/json",
            "content_encoding: identity",
            json.dumps({"choices": []}),
            "raw_upstream_response_1_end"
        ]

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(
                "\n".join(trace_lines),
                encoding="utf-8",
            )
            parsed = OpenClawHarness.parse_agent_trace(episode_dir)

        event_types = [event["type"] for event in parsed["events"]]
        instruction_kinds = [event.get("kind") for event in parsed["events"]
                             if event["type"] == "instruction"]
        self.assertEqual(parsed["backend"], "openclaw")
        self.assertEqual(instruction_kinds, ["native_harness", "agent_loop"])
        self.assertIn("reasoning", event_types)
        self.assertIn("tool_preamble", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("tool_result", event_types)
        self.assertIn("assistant_text", event_types)
        self.assertEqual(parsed["runtime"]["thinking_level"], "high")
        self.assertEqual(parsed["result"]["final_text"], "DONE")
        self.assertEqual(parsed["capture"]["tool_definitions"]["source"], "wire_request")
        self.assertEqual(parsed["capture"]["model_requests"]["count"], 1)
        self.assertEqual(parsed["capture"]["model_responses"]["count"], 1)

    def test_openclaw_trace_parser_treats_post_game_abort_as_termination(self):
        session = [
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Play through MCP tools."}]
                }
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "toolCall",
                        "id": "call-1",
                        "name": "clem_game__submit_response",
                        "arguments": {"response": "GUESS: answer"}
                    }]
                }
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-1",
                    "toolName": "clem_game__submit_response",
                    "content": [{"type": "text", "text": "done: true"}],
                    "details": {"structuredContent": {"done": True}}
                }
            },
            {
                "type": "custom",
                "customType": "openclaw:prompt-error",
                "data": {"error": "This operation was aborted | 20"}
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "stopReason": "aborted",
                    "content": [{"type": "thinking", "thinking": "Partial final reasoning."}]
                }
            }
        ]
        trajectory = [{
            "type": "session.ended",
            "data": {
                "status": "error",
                "aborted": True,
                "externalAbort": True,
                "promptError": "This operation was aborted | 20"
            }
        }]
        trace_lines = [
            "openclaw_session: /tmp/session-1.jsonl",
            *[json.dumps(record) for record in session],
            "openclaw_session: /tmp/session-1.trajectory.jsonl",
            *[json.dumps(record) for record in trajectory]
        ]

        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            (episode_dir / "agent_trace.log").write_text(
                "\n".join(trace_lines),
                encoding="utf-8",
            )
            parsed = OpenClawHarness.parse_agent_trace(episode_dir)

        meaningful_events = [event for event in parsed["events"]
                             if event["type"] in {"reasoning", "termination", "error"}]
        self.assertEqual(
            [event["type"] for event in meaningful_events],
            ["reasoning", "termination"],
        )
        self.assertEqual(parsed["result"]["status"], "completed")
        self.assertTrue(parsed["result"]["success"])
        self.assertEqual(
            parsed["result"]["terminal_reason"],
            "terminated_after_game_completion",
        )

    def test_bridge_exposes_start_and_submit_without_get_state(self):
        client = unittest.mock.MagicMock()

        with patch(
            "clemcore.agents.mcp.bridge.OpenEnvMCPClient",
            return_value=client,
        ):
            bridge = create_mcp_bridge("http://example.invalid/mcp")

        tools = asyncio.run(bridge.list_tools())

        self.assertEqual(
            [tool.name for tool in tools], ["start_game", "submit_response"]
        )

    def test_bridge_aborts_repeated_start_game(self):
        client = unittest.mock.MagicMock()
        client.call_tool.side_effect = [
            {"context": {}, "reward": None, "done": False, "metadata": {}},
            {"context": {}, "reward": -1.0, "done": False, "metadata": {}},
        ]

        with tempfile.TemporaryDirectory() as directory:
            completion_path = Path(directory) / "completed.json"

            with patch(
                "clemcore.agents.mcp.bridge.OpenEnvMCPClient",
                return_value=client,
            ), patch.dict(
                "os.environ",
                {"CLEM_GAME_COMPLETION_PATH": str(completion_path)},
            ):
                bridge = create_mcp_bridge("http://example.invalid/mcp")

                async def repeat_start_game():
                    await bridge.call_tool("start_game", {})
                    return await bridge.call_tool("start_game", {})

                result = asyncio.run(repeat_start_game())

            completion = json.loads(completion_path.read_text(encoding="utf-8"))

        self.assertTrue(result.structured_content["done"])
        self.assertEqual(
            client.call_tool.call_args_list,
            [
                unittest.mock.call("start_game", {}),
                unittest.mock.call(
                    "submit_response", {"response": CONTROL_FAILURE_RESPONSE}
                ),
            ],
        )
        client.close_session.assert_called_once_with()
        self.assertTrue(completion["done"])
        self.assertTrue(completion["control_failure"])

    def test_bridge_aborts_submit_response_before_start_game(self):
        client = unittest.mock.MagicMock()

        with tempfile.TemporaryDirectory() as directory:
            completion_path = Path(directory) / "completed.json"

            with patch(
                "clemcore.agents.mcp.bridge.OpenEnvMCPClient",
                return_value=client,
            ), patch.dict(
                "os.environ",
                {"CLEM_GAME_COMPLETION_PATH": str(completion_path)},
            ):
                bridge = create_mcp_bridge("http://example.invalid/mcp")
                result = asyncio.run(
                    bridge.call_tool("submit_response", {"response": "guess"})
                )

            completion = json.loads(completion_path.read_text(encoding="utf-8"))

        self.assertTrue(result.structured_content["done"])
        self.assertTrue(result.structured_content["metadata"]["start_game_required"])
        client.call_tool.assert_not_called()
        self.assertTrue(completion["control_failure"])

    def test_bridge_writes_completion_marker(self):
        client = unittest.mock.MagicMock()
        client.call_tool.side_effect = [
            {"context": {}, "reward": None, "done": False, "metadata": {}},
            {"context": {}, "reward": 1.0, "done": True, "metadata": {}},
        ]

        with tempfile.TemporaryDirectory() as directory:
            completion_path = Path(directory) / "completed.json"

            with patch(
                "clemcore.agents.mcp.bridge.OpenEnvMCPClient",
                return_value=client,
            ), patch.dict(
                "os.environ",
                {"CLEM_GAME_COMPLETION_PATH": str(completion_path)},
            ):
                bridge = create_mcp_bridge("http://example.invalid/mcp")

                async def finish_game():
                    await bridge.call_tool("start_game", {})
                    await bridge.call_tool("submit_response", {"response": "guess"})

                asyncio.run(finish_game())

            completion = json.loads(completion_path.read_text(encoding="utf-8"))

        self.assertTrue(completion["done"])
        self.assertFalse(completion["control_failure"])
        self.assertEqual(completion["reward"], 1.0)

    def test_hermes_timeout_is_a_failed_episode_not_an_exception(self):
        successful_setup = subprocess.CompletedProcess([], 0, "", "")
        calls = [
            successful_setup,
            successful_setup,
            successful_setup,
            successful_setup,
        ]

        with patch(
            "clemcore.agents.adapters.hermes.load_model_connection",
            return_value=None,
        ), patch(
            "clemcore.agents.adapters.hermes.subprocess.run",
            side_effect=calls,
        ), patch(
            "clemcore.agents.adapters.hermes.run_process_until_game_complete",
            side_effect=subprocess.TimeoutExpired(
                ["hermes", "chat"],
                1,
                output="partial Hermes output",
                stderr="",
            ),
        ):
            result = HermesHarness(model="test-model", timeout=1).run_episode(
                "Play the game."
            )

        self.assertFalse(result.success)
        self.assertEqual(
            result.metadata["runtime_error"], "Hermes timed out after 1s"
        )

    def test_hermes_cli_exit_before_game_done_is_not_success(self):
        successful_setup = subprocess.CompletedProcess([], 0, "", "")
        incomplete_chat = subprocess.CompletedProcess(
            [],
            0,
            "Tool call: mcp__clem_game__start_game",
            "",
        )

        with patch(
            "clemcore.agents.adapters.hermes.load_model_connection",
            return_value=None,
        ), patch(
            "clemcore.agents.adapters.hermes.subprocess.run",
            side_effect=[
                successful_setup,
                successful_setup,
                successful_setup,
                successful_setup,
            ],
        ), patch(
            "clemcore.agents.adapters.hermes.run_process_until_game_complete",
            return_value=(incomplete_chat, False),
        ):
            result = HermesHarness(model="test-model").run_episode(
                "Play the game."
            )

        self.assertFalse(result.success)
        self.assertFalse(result.metadata["game_completed"])
        self.assertEqual(
            result.metadata["runtime_error"],
            "Hermes ended before clem_game reported done=true",
        )

    def test_closed_openenv_session_removes_recovery_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            session_path = Path(directory) / "openenv_session.json"
            session_path.write_text(
                json.dumps({"session_id": "session-1"}),
                encoding="utf-8",
            )
            client = OpenEnvMCPClient("http://example.invalid/mcp")
            client.session_id = "session-1"

            with patch.object(client, "_request", return_value={}) as request, patch.dict(
                "os.environ",
                {"CLEM_OPENENV_SESSION_PATH": str(session_path)},
            ):
                client.close_session()

            request.assert_called_once_with(
                "openenv/session/close",
                {"session_id": "session-1"},
            )
            self.assertIsNone(client.session_id)
            self.assertFalse(session_path.exists())

    def test_openclaw_rejects_mismatched_openrouter_connection(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            _validate_openclaw_model_connection({
                "backend": "openrouter",
                "model": "openai/gpt-5-mini",
                "env": {"OPENROUTER_API_KEY": "openrouter-test-key"},
            })

        with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
            _validate_openclaw_model_connection({
                "backend": "openrouter",
                "model": "openrouter/openai/gpt-5-mini",
                "env": {},
            })

    def test_openclaw_openrouter_resolution_uses_canonical_model(self):
        model_spec = {
            "model_name": "gpt-5-mini-openrouter",
            "backend": "openrouter",
            "model_id": "openai/gpt-5-mini",
        }
        with patch.object(
            model_connection, "_find_model_spec", return_value=model_spec
        ), patch.object(
            model_connection,
            "_openrouter_key_config",
            return_value={"api_key": "openrouter-test-key"},
        ):
            connection = model_connection.resolve_clem_model_for_openclaw(
                "gpt-5-mini-openrouter"
            )

        self.assertEqual(connection["harness"], "openclaw")
        self.assertEqual(connection["backend"], "openrouter")
        self.assertEqual(
            connection["model"], "openrouter/openai/gpt-5-mini"
        )
        self.assertEqual(
            connection["env"],
            {"OPENROUTER_API_KEY": "openrouter-test-key"},
        )

    def test_all_harnesses_resolve_openai_compatible_models(self):
        model_spec = {
            "model_name": "Qwen-test-without-reasoning",
            "backend": "openai_compatible",
            "model_id": "Qwen/Qwen-test",
            "context_size": "128k",
            "model_config": {
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            },
        }
        key_config = {
            "api_key": "jarvis-test-key",
            "base_url": "https://jarvis.ling.uni-potsdam.de/api/v1",
        }

        with patch.object(
            model_connection, "_find_model_spec", return_value=model_spec
        ), patch.object(
            model_connection, "_openai_compatible_key_config", return_value=key_config
        ):
            connections = {
                "claude_code": model_connection.resolve_clem_model_for_claude_code(
                    model_spec["model_name"]
                ),
                "codex": model_connection.resolve_clem_model_for_codex(
                    model_spec["model_name"]
                ),
                "hermes": model_connection.resolve_clem_model_for_hermes(
                    model_spec["model_name"]
                ),
                "openclaw": model_connection.resolve_clem_model_for_openclaw(
                    model_spec["model_name"]
                ),
            }

        for harness, connection in connections.items():
            self.assertEqual(connection["harness"], harness)
            self.assertEqual(connection["backend"], "openai_compatible")
            self.assertNotIn("tool_choice", connection)
            self.assertFalse(connection["verify_tls"])
            self.assertEqual(
                connection["request_body_overrides"],
                {"chat_template_kwargs": {"enable_thinking": False}},
            )

        self.assertEqual(connections["hermes"]["provider"], "openai-api")
        self.assertEqual(connections["claude_code"]["runtime_model"], "claude-sonnet-4-5")
        self.assertEqual(connections["claude_code"]["model"], "Qwen/Qwen-test")
        self.assertEqual(
            connections["openclaw"]["openclaw_provider"],
            "clem_openai_compatible",
        )

    def test_proxy_preserves_harness_tool_choice(self):
        received = []

        class UpstreamHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                received.append((self.path, json.loads(self.rfile.read(length))))
                response = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format, *args):
                return

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()

        try:
            with tempfile.TemporaryDirectory() as directory:
                completion_path = Path(directory) / "done.json"
                target = f"http://127.0.0.1:{upstream.server_port}/api/v1"

                with OpenAICompatibleProxy(
                    target,
                    completion_path,
                    {"chat_template_kwargs": {"enable_thinking": False}},
                ) as proxy:
                    openai_body = {
                        "model": "qwen",
                        "tools": [{"type": "function", "function": {"name": "start_game"}}],
                        "tool_choice": "auto",
                    }
                    requests.post(
                        f"{proxy.base_url}/chat/completions",
                        json=openai_body,
                        timeout=5,
                    ).raise_for_status()
                    requests.post(
                        f"{proxy.base_url}/messages",
                        json={
                            "model": "qwen",
                            "tools": openai_body["tools"],
                        },
                        timeout=5,
                    ).raise_for_status()

                    completion_path.write_text('{"done":true}', encoding="utf-8")
                    requests.post(
                        f"{proxy.base_url}/chat/completions",
                        json=openai_body,
                        timeout=5,
                    ).raise_for_status()
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=5)

        self.assertEqual(received[0][0], "/api/v1/chat/completions")
        self.assertEqual(received[0][1]["tool_choice"], "auto")
        self.assertEqual(
            received[0][1]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertNotIn("tool_choice", received[1][1])
        self.assertEqual(received[2][1]["tool_choice"], "auto")

    def test_proxy_rewrites_model_for_anthropic_messages(self):
        received = []

        class UpstreamHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                received.append((self.path, json.loads(self.rfile.read(length))))
                response = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format, *args):
                return

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()

        try:
            with tempfile.TemporaryDirectory() as directory:
                completion_path = Path(directory) / "done.json"
                target = f"http://127.0.0.1:{upstream.server_port}/api/v1"

                with OpenAICompatibleProxy(
                    target,
                    completion_path,
                    {"chat_template_kwargs": {"enable_thinking": False}},
                    upstream_model="Qwen/Qwen3.6-35B-A3B-FP8",
                ) as proxy:
                    payload = {
                        "model": "claude-sonnet-4-5",
                        "tools": [{"type": "function", "function": {"name": "start_game"}}],
                    }
                    requests.post(
                        f"{proxy.base_url}/messages",
                        json=payload,
                        timeout=5,
                    ).raise_for_status()
                    requests.post(
                        f"{proxy.base_url}/chat/completions",
                        json=payload,
                        timeout=5,
                    ).raise_for_status()
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=5)

        self.assertEqual(
            received[0][1]["model"],
            "Qwen/Qwen3.6-35B-A3B-FP8",
        )
        self.assertNotIn("chat_template_kwargs", received[0][1])
        self.assertEqual(
            received[1][1]["model"],
            "Qwen/Qwen3.6-35B-A3B-FP8",
        )
        self.assertEqual(
            received[1][1]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_proxy_flattens_responses_namespaces_and_restores_tool_calls(self):
        received = []

        class UpstreamHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                received.append(json.loads(self.rfile.read(length)))
                flat_name = "mcp__clem_game__start_game"
                events = [
                    "event: response.output_item.done\n"
                    "data: " + json.dumps({
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "name": flat_name,
                            "arguments": "{}",
                            "call_id": "call_1",
                            "status": "completed",
                        },
                    }) + "\n\n",
                    "event: response.completed\n"
                    "data: " + json.dumps({
                        "type": "response.completed",
                        "response": {
                            "status": "completed",
                            "output": [{
                                "type": "function_call",
                                "name": flat_name,
                                "arguments": "{}",
                                "call_id": "call_1",
                                "status": "completed",
                            }],
                        },
                    }) + "\n\n",
                ]
                response = "".join(events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format, *args):
                return

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()

        try:
            with tempfile.TemporaryDirectory() as directory:
                completion_path = Path(directory) / "done.json"
                target = f"http://127.0.0.1:{upstream.server_port}/api/v1"

                with OpenAICompatibleProxy(target, completion_path) as proxy:
                    response = requests.post(
                        f"{proxy.base_url}/responses",
                        json={
                            "model": "qwen",
                            "input": [{
                                "type": "function_call",
                                "name": "start_game",
                                "namespace": "mcp__clem_game__",
                                "arguments": "{}",
                                "call_id": "previous_call",
                            }],
                            "tools": [{
                                "type": "namespace",
                                "name": "mcp__clem_game__",
                                "description": "Game tools",
                                "tools": [{
                                    "type": "function",
                                    "name": "start_game",
                                    "description": "Start the game",
                                    "strict": False,
                                    "parameters": {
                                        "type": "object",
                                        "properties": {},
                                    },
                                }],
                            }],
                            "stream": True,
                        },
                        timeout=5,
                    )
                    response.raise_for_status()
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=5)

        self.assertEqual(
            received[0]["tools"][0]["name"],
            "mcp__clem_game__start_game",
        )
        self.assertEqual(received[0]["tools"][0]["type"], "function")
        self.assertEqual(
            received[0]["input"][0]["name"],
            "mcp__clem_game__start_game",
        )
        self.assertNotIn("namespace", received[0]["input"][0])

        data_events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        calls = [
            data_events[0]["item"],
            data_events[1]["response"]["output"][0],
        ]

        for call in calls:
            self.assertEqual(call["name"], "start_game")
            self.assertEqual(call["namespace"], "mcp__clem_game__")

        namespace_map = {
            "mcp__clem_game__start_game": ("mcp__clem_game__", "start_game")
        }
        self.assertIsNone(_resolve_namespaced_tool("start_game", namespace_map))
        self.assertIsNone(
            _resolve_namespaced_tool("mcp.clem_game.start_game", namespace_map)
        )

    def test_process_is_terminated_after_game_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            completion_path = Path(directory) / "done.json"
            script = (
                "from pathlib import Path; import time; "
                f"Path({str(completion_path)!r}).write_text('{{\"done\":true}}'); "
                "time.sleep(30)"
            )
            started_at = time.monotonic()
            result, terminated = run_process_until_game_complete(
                [sys.executable, "-c", script],
                completion_path=completion_path,
                completion_grace=0.01,
                timeout=5,
            )

        self.assertTrue(terminated)
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - started_at, 2)

    def test_openclaw_writes_openrouter_key_into_native_config(self):
        connection = {
            "backend": "openrouter",
            "model": "openrouter/openai/gpt-5-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "env": {"OPENROUTER_API_KEY": "openrouter-test-key"},
        }
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs.get("input")))
            return subprocess.CompletedProcess(command, 0, "", "")

        def fake_agent_run(command, **kwargs):
            calls.append((command, None))
            kwargs["completion_path"].write_text(
                '{"done":true,"control_failure":false}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, '"submit_response"', ""), False

        with patch(
            "clemcore.agents.adapters.openclaw.load_model_connection",
            return_value=connection,
        ), patch(
            "clemcore.agents.adapters.openclaw.subprocess.run",
            side_effect=fake_run,
        ), patch(
            "clemcore.agents.adapters.openclaw.run_process_until_game_complete",
            side_effect=fake_agent_run,
        ), tempfile.TemporaryDirectory() as directory:
            result = OpenClawHarness(
                reasoning_effort="medium",
                model_connection_path="ignored",
            ).run_episode("Play the game.", output_dir=directory)

        config_input = next(
            input_text
            for command, input_text in calls
            if "config" in command and "patch" in command
        )
        config = json.loads(config_input)
        self.assertEqual(
            config["env"]["OPENROUTER_API_KEY"],
            "openrouter-test-key",
        )
        self.assertEqual(
            config["plugins"],
            {
                "enabled": True,
                "allow": ["openrouter", "duckduckgo", "browser"],
            },
        )
        self.assertEqual(
            config["browser"],
            {"enabled": True, "headless": True, "noSandbox": True},
        )
        self.assertEqual(
            config["tools"]["web"]["search"],
            {"enabled": True, "provider": "duckduckgo"},
        )
        self.assertEqual(
            config["tools"]["exec"],
            {"security": "full", "ask": "off"},
        )
        agent_command = next(
            command for command, _input_text in calls if "agent" in command
        )
        self.assertEqual(
            agent_command[agent_command.index("--thinking") + 1],
            "medium",
        )
        self.assertTrue(result.success)

    def test_result_run_dir_orders_openclaw_and_environment_players(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "agent_registry.json"
            registry_path.write_text(
                json.dumps([
                    {
                        "agent_name": "openclaw-with-gpt-5-mini-openrouter",
                        "backend": "openclaw",
                        "agent_config": {
                            "clem_model": "gpt-5-mini-openrouter"
                        },
                    }
                ]),
                encoding="utf-8",
            )

            run_dir = result_run_dir_name(
                agent_name="openclaw-with-gpt-5-mini-openrouter",
                registry_path=registry_path,
                learner_agent="player_0",
                env_agents={"player_1": "Llama-4-Maverick"},
            )

        self.assertEqual(
            run_dir,
            "openclaw-with-gpt-5-mini-openrouter--Llama-4-Maverick",
        )

    def test_failed_trace_never_reuses_another_harness_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            wrong_episode = (
                results_dir
                / "hermes-with-gpt-5-mini-openrouter--Llama-4-Maverick"
                / "taboo"
                / "high_en"
                / "episode_00001"
            )
            wrong_episode.mkdir(parents=True)
            (wrong_episode / "instance.json").write_text(
                json.dumps({"game_id": 0}), encoding="utf-8"
            )

            run_dir = (
                "openclaw-with-gpt-5-mini-openrouter--Llama-4-Maverick"
            )
            trace_path = write_agent_artifacts(
                results_dir=str(results_dir),
                run_dir=run_dir,
                game_name="taboo",
                experiment_name="high_en",
                game_id=0,
                before_instance_dirs=set(),
                trace_text="openclaw failure",
                metadata={
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            self.assertEqual(
                trace_path.read_text(encoding="utf-8"), "openclaw failure"
            )
            self.assertTrue(
                trace_path.is_relative_to(
                    results_dir / "_agent_failures" / run_dir
                )
            )
            self.assertFalse((wrong_episode / "agent_trace.log").exists())


if __name__ == "__main__":
    unittest.main()
