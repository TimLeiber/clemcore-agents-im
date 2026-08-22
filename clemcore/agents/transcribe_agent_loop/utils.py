"""Helpers for transcribing standardized agent-loop JSON to HTML."""

import html
import json
from pathlib import Path
from typing import Any


def load_agent_loop(path: Path) -> dict[str, Any]:
    """Load and validate one standardized agent-loop JSON file.

    Args:
        path: path to an agent_loop.json file

    Returns:
        standardized trace dictionary
    """

    agent_loop = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(agent_loop, dict):
        raise ValueError(f"Agent loop must be a JSON object: {path}")

    return agent_loop


def format_event_content(content: Any) -> str:
    """Return safely escaped content for an HTML preformatted block.

    Args:
        content: event content from a standardized agent-loop trace

    Returns:
        escaped readable content
    """

    if isinstance(content, str):
        return html.escape(content)

    return html.escape(json.dumps(content, indent=2, ensure_ascii=False))


def write_html(path: Path, content: str) -> Path:
    """Write one rendered agent-loop HTML file.

    Args:
        path: target HTML path
        content: complete HTML document

    Returns:
        written HTML path
    """

    path.write_text(content, encoding="utf-8")

    return path
