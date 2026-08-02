import asyncio
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from clemcore.agents.adapters import model_connection
from clemcore.agents.adapters.hermes import HermesHarness
from clemcore.agents.adapters.openclaw import (
    OpenClawHarness,
    _validate_openclaw_model_connection,
)
from clemcore.agents.mcp.bridge import OpenEnvMCPClient, create_mcp_bridge
from clemcore.agents.mcp.server import result_run_dir_name
from clemcore.agents.run_pipeline.utils import write_agent_trace


class TestExternalAgentPipeline(unittest.TestCase):
    def test_bridge_does_not_start_second_episode_after_done(self):
        client = unittest.mock.MagicMock()
        client.call_tool.side_effect = [
            {"context": {"role": "user", "content": "start"},
             "reward": None, "done": False, "metadata": {}},
            {"context": {"role": "user", "content": "finished"},
             "reward": 1.0, "done": True, "metadata": {}},
        ]

        with patch(
            "clemcore.agents.mcp.bridge.OpenEnvMCPClient",
            return_value=client,
        ):
            bridge = create_mcp_bridge("http://example.invalid/mcp")

        async def call_episode_tools():
            await bridge.call_tool("start_game", {})
            await bridge.call_tool(
                "submit_response", {"response": "guess"}
            )
            return await bridge.call_tool("start_game", {})

        repeated_start = asyncio.run(call_episode_tools())

        self.assertEqual(client.call_tool.call_count, 2)
        self.assertTrue(repeated_start.structured_content["done"])
        self.assertTrue(
            repeated_start.structured_content["metadata"]["already_completed"]
        )

    def test_hermes_timeout_is_a_failed_episode_not_an_exception(self):
        successful_setup = subprocess.CompletedProcess([], 0, "", "")
        calls = [
            successful_setup,
            successful_setup,
            successful_setup,
            successful_setup,
            subprocess.TimeoutExpired(
                ["hermes", "chat"],
                1,
                output="partial Hermes output",
                stderr="",
            ),
        ]

        with patch(
            "clemcore.agents.adapters.hermes.load_model_connection",
            return_value=None,
        ), patch(
            "clemcore.agents.adapters.hermes.subprocess.run",
            side_effect=calls,
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
                incomplete_chat,
            ],
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

    def test_openclaw_writes_openrouter_key_into_native_config(self):
        connection = {
            "backend": "openrouter",
            "model": "openrouter/openai/gpt-5-mini",
            "env": {"OPENROUTER_API_KEY": "openrouter-test-key"},
        }
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs.get("input")))
            stdout = '"start_game"' if "agent" in command else ""
            return subprocess.CompletedProcess(command, 0, stdout, "")

        with patch(
            "clemcore.agents.adapters.openclaw.load_model_connection",
            return_value=connection,
        ), patch(
            "clemcore.agents.adapters.openclaw.subprocess.run",
            side_effect=fake_run,
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
            trace_path = write_agent_trace(
                results_dir=str(results_dir),
                run_dir=run_dir,
                game_name="taboo",
                experiment_name="high_en",
                game_id=0,
                before_episode_dirs=set(),
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
