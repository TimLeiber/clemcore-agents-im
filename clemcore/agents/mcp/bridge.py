import requests
import json
from typing import Any, Optional
import atexit
import os
from pathlib import Path
import sys

from fastmcp import FastMCP


class OpenEnvMCPClient:
    """
    Minimal client for OpenEnv's /mcp JSON-RPC endpoint.

    This is not the standard MCP client used by external agent harnesses.
    It talks to OpenEnv's session-based MCP-ish endpoint and is used by the
    bridge below.
    """

    def __init__(self,
                 mcp_url: str):
        self.mcp_url = mcp_url
        self.session_id: Optional[str] = None
        self._request_id = 0

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _request(self,
                 method: str,
                 params: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
            "params": params or {},
        }

        response = requests.post(self.mcp_url,
                                 json=payload,
                                 timeout=30)
        response.raise_for_status()

        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        return data["result"]

    def create_session(self) -> str:
        if self.session_id is not None:
            return self.session_id

        result = self._request("openenv/session/create")
        self.session_id = result["session_id"]

        session_path = os.environ.get("CLEM_OPENENV_SESSION_PATH")
        if session_path:
            Path(session_path).write_text(
                json.dumps({"session_id": self.session_id}),
                encoding="utf-8",
            )

        return self.session_id

    def close_session(self) -> None:
        if self.session_id is None:
            return

        self._request("openenv/session/close",
                      {"session_id": self.session_id})
        self.session_id = None

    def call_tool(self,
                  name: str,
                  arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        session_id = self.create_session()

        result = self._request(
            "tools/call",
            {
                "session_id": session_id,
                "name": name,
                "arguments": arguments or {},
            },
        )

        return result["data"]


def create_mcp_bridge(openenv_mcp_url: str | None = None) -> FastMCP:
    openenv_mcp_url = openenv_mcp_url or os.environ.get(
        "OPENENV_MCP_URL",
        "http://127.0.0.1:8001/mcp",
    )
    """
    Create a standard MCP server that forwards tool calls to OpenEnv /mcp.

    External agent harnesses should connect to this bridge, not directly to
    OpenEnv's /mcp endpoint.
    """
    server_instructions = """
This MCP server exposes clembench game actions as MCP tools.

The available game actions are tools, not resources:
- start_game: call this tool first, with empty arguments.
- submit_response: call this tool with {"response": "..."} for every game move.
- get_state: call this tool only if the current game state is needed.

Do not call read_mcp_resource for start_game, submit_response, or get_state.
Do not treat start_game as a URI.
The first game action must be the start_game tool.
Continue with submit_response until the returned result says done=true.
""".strip()

    mcp = FastMCP("clem_game", instructions=server_instructions)
    client = OpenEnvMCPClient(openenv_mcp_url)

    game_state = {
        "started": False,
        "done": False,
    }

    control_failure_response = (
        "CLEM_AGENT_CONTROL_ERROR: external harness ended before completing the game"
    )

    def cleanup_openenv_session() -> None:
        if client.session_id is None:
            return

        if game_state["started"] and not game_state["done"]:
            try:
                result = client.call_tool(
                    "submit_response",
                    {"response": control_failure_response},
                )

                if result.get("done") is True:
                    game_state["done"] = True

            except Exception as error:
                print(
                    f"failed to submit control failure response: {error}",
                    file=sys.stderr,
                )

        try:
            client.close_session()
        except Exception as error:
            print(
                f"failed to close OpenEnv session: {error}",
                file=sys.stderr,
            )

    atexit.register(cleanup_openenv_session)

    @mcp.tool()
    def start_game() -> dict[str, Any]:
        """
        Start/reset the current clembench game and return the initial observation.
        """
        arguments = {}

        game_id = os.environ.get("CLEM_GAME_ID")
        experiment_name = os.environ.get("CLEM_EXPERIMENT_NAME")

        if game_id is not None:
            arguments["game_id"] = int(game_id)

        if experiment_name is not None:
            arguments["experiment_name"] = experiment_name

        result = client.call_tool("start_game", arguments)
        game_state["started"] = True
        game_state["done"] = result.get("done") is True
        return result

    @mcp.tool()
    def submit_response(response: str) -> dict[str, Any]:
        """
        Submit a response/move to the current clembench game.
        """
        result = client.call_tool("submit_response",
                                  {"response": response})

        if result.get("done") is True:
            game_state["done"] = True
            client.close_session()

        return result

    @mcp.tool()
    def get_state() -> dict[str, Any]:
        """
        Get the current clembench game state.
        """
        return client.call_tool("get_state")

    return mcp