import atexit
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from fastmcp import FastMCP


DEFAULT_OPENENV_MCP_URL = "http://127.0.0.1:8001/mcp"

SERVER_CONFIG_PATH = Path(__file__).parent / "mcp_server_config.yaml"

# response submitted for an agent that stopped before the game was finished
CONTROL_FAILURE_RESPONSE = "CLEM_AGENT_CONTROL_ERROR: external harness ended before completing the game"


class OpenEnvMCPClient:
    """Client for the session-based JSON-RPC endpoint of the host MCP server.

    This is not the standard MCP client used by external agent harnesses. It
    talks to the OpenEnv /mcp endpoint of the host process, which keeps one
    session per episode, and is used by the bridge below to forward tool calls.
    """

    def __init__(self, mcp_url: str):
        self.mcp_url = mcp_url
        self.session_id: Optional[str] = None
        self._request_id = 0

    def _request(self,
                 method: str,
                 params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one JSON-RPC request to the host MCP server

        Args:
            method: name of the JSON-RPC method to invoke
            params: optional parameters for the invoked method

        Returns:
            The result payload of the response
        """
        self._request_id += 1

        payload = {"jsonrpc": "2.0",
                   "id": self._request_id,
                   "method": method,
                   "params": params or {}}

        response = requests.post(self.mcp_url,
                                 json=payload,
                                 timeout=30)
        response.raise_for_status()

        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        return data["result"]

    def create_session(self) -> str:
        """Open an episode session on the host, reusing an already open one

        Returns:
            The identifier of the open session
        """
        if self.session_id is not None:
            return self.session_id

        result = self._request("openenv/session/create")
        self.session_id = result["session_id"]

        # publish the session id through the directory shared with the host
        session_path = os.environ.get("CLEM_OPENENV_SESSION_PATH")

        if session_path:
            Path(session_path).write_text(json.dumps({"session_id": self.session_id}),
                                          encoding="utf-8")

        return self.session_id

    def close_session(self) -> None:
        """Close the open episode session on the host"""
        if self.session_id is None:
            return

        self._request("openenv/session/close", {"session_id": self.session_id})
        self.session_id = None

    def call_tool(self,
                  name: str,
                  arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke one game tool on the host within the episode session

        Args:
            name: name of the tool to invoke
            arguments: optional arguments for the invoked tool

        Returns:
            The data payload returned by the tool
        """
        session_id = self.create_session()

        result = self._request("tools/call",
                               {"session_id": session_id,
                                "name": name,
                                "arguments": arguments or {}})

        return result["data"]


def _load_server_instructions() -> str:
    """Read the agent-facing tool instructions from the bridge configuration

    Returns:
        The instructions presented to the external agent
    """
    with open(SERVER_CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["instructions"].strip()


def create_mcp_bridge(openenv_mcp_url: str | None = None) -> FastMCP:
    """Create the container-side MCP server forwarding game actions to the host.

    External agent harnesses connect to this bridge rather than to the host
    endpoint directly. The bridge exposes the clembench game actions as standard
    MCP tools and translates every call into a session-based request against the
    host MCP server.

    Args:
        openenv_mcp_url: address of the host MCP endpoint, taken from the
            environment when omitted

    Returns:
        The configured MCP server
    """
    openenv_mcp_url = openenv_mcp_url or os.environ.get("OPENENV_MCP_URL", DEFAULT_OPENENV_MCP_URL)

    mcp = FastMCP("clem_game", instructions=_load_server_instructions())
    client = OpenEnvMCPClient(openenv_mcp_url)

    # track the episode progress so an early exit can still be reported
    game_state = {"started": False,
                  "done": False}

    def cleanup_openenv_session() -> None:
        """Report an unfinished game to the host and close the session"""
        if client.session_id is None:
            return

        # a started but unfinished game means the harness quit early
        if game_state["started"] and not game_state["done"]:
            try:
                result = client.call_tool("submit_response",
                                          {"response": CONTROL_FAILURE_RESPONSE})

                if result.get("done") is True:
                    game_state["done"] = True

            except Exception as error:
                print(f"failed to submit control failure response: {error}", file=sys.stderr)

        try:
            client.close_session()
        except Exception as error:
            print(f"failed to close OpenEnv session: {error}", file=sys.stderr)

    atexit.register(cleanup_openenv_session)

    @mcp.tool()
    def start_game() -> dict[str, Any]:
        """Start/reset the current clembench game and return the initial observation."""
        arguments = {}

        # the host selects the instance for this episode through the environment
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
        """Submit a response/move to the current clembench game."""
        result = client.call_tool("submit_response", {"response": response})

        # release the host session as soon as the game reports completion
        if result.get("done") is True:
            game_state["done"] = True
            client.close_session()

        return result

    @mcp.tool()
    def get_state() -> dict[str, Any]:
        """Get the current clembench game state."""
        return client.call_tool("get_state")

    return mcp


if __name__ == "__main__":
    # container-side entry point started by the agent adapters
    create_mcp_bridge().run()