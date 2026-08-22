"""Render standardized external-agent traces as readable HTML."""

import argparse
import logging
from pathlib import Path
from typing import Any

from clemcore.agents.transcribe_agent_loop.utils import (
    format_event_content,
    load_agent_loop,
    write_html,
)


module_logger = logging.getLogger(__name__)

CSS = """
body {
    background: #f5f7fa;
    color: #1f2937;
    font-family: Arial, sans-serif;
    margin: 0;
}

.page {
    margin: 0 auto;
    max-width: 1040px;
    padding: 28px 20px 48px;
}

.top-info, .capture, .event {
    background: white;
    border: 1px solid #d8dee9;
    border-radius: 8px;
    box-shadow: 0 1px 2px rgb(0 0 0 / 6%);
}

.top-info {
    margin-bottom: 16px;
    padding: 16px 18px;
}

.top-info h1 {
    font-size: 20px;
    margin: 0 0 6px;
}

.top-info p, .capture summary {
    color: #52606d;
    font-size: 13px;
    margin: 0;
}

.capture {
    margin-bottom: 16px;
    padding: 12px 16px;
}

.capture ul {
    font-size: 13px;
    margin: 10px 0 0;
    padding-left: 20px;
}

.event {
    border-left: 5px solid #78909c;
    margin: 12px 0;
    overflow: hidden;
}

.event.instruction { border-left-color: #6b7280; }
.event.reasoning { border-left-color: #8b5cf6; }
.event.tool-call { border-left-color: #d97706; }
.event.tool-result { border-left-color: #059669; }
.event.tool-preamble { border-left-color: #0284c7; }
.event.assistant-text { border-left-color: #2563eb; }
.event.error { border-left-color: #dc2626; }
.event.model-request, .event.model-response { border-left-color: #475569; }

.event-header {
    align-items: baseline;
    background: #f8fafc;
    border-bottom: 1px solid #e5e7eb;
    display: flex;
    gap: 8px;
    padding: 9px 12px;
}

.event-title {
    font-size: 13px;
    font-weight: 700;
}

.event-meta {
    color: #64748b;
    font-family: monospace;
    font-size: 11px;
}

.event-body {
    padding: 12px;
}

.event-note {
    color: #52606d;
    font-size: 12px;
    margin: 0;
}

pre {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    font-family: Menlo, Consolas, monospace;
    font-size: 12px;
    line-height: 1.45;
    margin: 0;
    overflow-x: auto;
    padding: 10px;
    white-space: pre-wrap;
    word-break: break-word;
}

details {
    margin-top: 10px;
}

summary {
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
}

.empty {
    color: #64748b;
    font-style: italic;
    padding: 16px;
}
"""

EVENT_LABELS = {
    "instruction": "Instruction",
    "tool_definitions": "Tool definitions",
    "message": "Message",
    "reasoning": "Reasoning",
    "tool_preamble": "Tool preamble",
    "assistant_text": "Assistant text",
    "assistant_output": "Assistant output",
    "tool_call": "Tool call",
    "tool_result": "Tool result",
    "model_request": "Model request",
    "model_response": "Model response",
    "error": "Error"
}


def build_agent_loop_html(agent_loop: dict[str, Any], source_path: Path) -> str:
    """Create one HTML document from a standardized agent-loop trace.

    Args:
        agent_loop: standardized trace produced by a harness parser
        source_path: path of the source agent_loop.json file

    Returns:
        complete HTML document
    """

    backend = agent_loop.get("backend", "unknown harness")
    schema_version = agent_loop.get("schema_version", "unknown")
    events = agent_loop.get("events", [])

    if not isinstance(events, list):
        raise ValueError(f"Agent-loop events must be a list: {source_path}")

    semantic_events = [event for event in events
                       if isinstance(event, dict)
                       and event.get("type") not in {"model_request", "model_response", "tool_definitions"}]
    tool_definition_events = [event for event in events
                              if isinstance(event, dict)
                              and event.get("type") == "tool_definitions"]
    protocol_events = [event for event in events
                       if isinstance(event, dict)
                       and event.get("type") in {"model_request", "model_response"}]
    title = f"Agent Loop Transcript — {backend}"
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{title}</title>",
        f"<style>{CSS}</style>",
        "</head>",
        "<body>",
        '<main class="page">',
        '<section class="top-info">',
        f"<h1>{title}</h1>",
        f"<p>Schema version {schema_version} · {len(semantic_events)} dialogue events · {source_path.name}</p>",
        "</section>",
        _render_capture(agent_loop.get("capture", {})),
        _render_trace_metadata(agent_loop),
        _render_tool_configuration(tool_definition_events)
    ]

    if semantic_events:
        html_parts.extend(_render_event(event) for event in semantic_events)
    else:
        html_parts.append('<section class="event"><p class="empty">No agent-loop events were recorded</p></section>')

    html_parts.append(_render_protocol_appendix(protocol_events))
    html_parts.extend(["</main>", "</body>", "</html>"])

    return "\n".join(html_parts)


def build_agent_loop_transcripts(top_dir: str | Path) -> tuple[int, int]:
    """Render every standardized agent-loop trace below a result directory.

    Args:
        top_dir: results directory to scan recursively

    Returns:
        number of generated transcripts and failures
    """

    generated = 0
    failures = 0

    for source_path in sorted(Path(top_dir).rglob("agent_loop.json")):
        try:
            agent_loop = load_agent_loop(source_path)
            html_document = build_agent_loop_html(agent_loop, source_path)
            write_html(source_path.with_name("agent_loop.html"), html_document)
            generated += 1
        except Exception:
            module_logger.exception("Cannot transcribe %s", source_path)
            failures += 1

    return generated, failures


def _render_capture(capture: Any) -> str:
    """Render parser capture metadata as a compact expandable section."""

    if not isinstance(capture, dict) or not capture:
        return '<details class="capture"><summary>Capture metadata unavailable</summary></details>'

    entries = []

    for name, status in capture.items():
        entries.append(f"<li><strong>{name}</strong>: {format_event_content(status)}</li>")

    return (
        '<details class="capture" open>'
        '<summary>Capture metadata</summary>'
        f"<ul>{''.join(entries)}</ul>"
        "</details>"
    )


def _render_tool_configuration(events: list[dict[str, Any]]) -> str:
    """Render structured tool schemas outside the dialogue timeline."""

    if not events:
        return ""

    tool_sets = [event.get("tools", []) for event in events]

    return (
        '<details class="capture">'
        '<summary>Available tools sent with the model request</summary>'
        '<p class="event-note">This is the tool inventory or structured tool configuration captured '
        'by the harness. It is request configuration, not natural-language prompt text.</p>'
        f'<pre>{format_event_content(tool_sets[0] if len(tool_sets) == 1 else tool_sets)}</pre>'
        '</details>'
    )


def _render_trace_metadata(agent_loop: dict[str, Any]) -> str:
    """Render optional standardized runtime and result metadata."""

    sections = []

    for key, label in (("runtime", "Harness runtime"), ("result", "Run result")):
        value = agent_loop.get(key)

        if value:
            sections.append(
                '<details class="capture">'
                f'<summary>{label}</summary>'
                f'<pre>{format_event_content(value)}</pre>'
                '</details>'
            )

    return "".join(sections)


def _render_protocol_appendix(events: list[dict[str, Any]]) -> str:
    """Render raw API transport records as an optional diagnostic appendix."""

    if not events:
        return ""

    records = []

    for event in events:
        event_type = str(event.get("type", "unknown"))
        turn = event.get("turn", "unknown")
        status = f" · status {event['status']}" if event.get("status") is not None else ""
        records.append(
            '<details>'
            f'<summary>{EVENT_LABELS.get(event_type, event_type)} · turn {turn}{status}</summary>'
            f'<pre>{format_event_content(event.get("raw", ""))}</pre>'
            '</details>'
        )

    return (
        '<details class="capture">'
        f'<summary>Raw API transport records ({len(events)})</summary>'
        '<p class="event-note">These are proxy-captured HTTP request bodies and streamed response '
        'events. They are retained for diagnostics and are not dialogue turns.</p>'
        f'{"".join(records)}'
        '</details>'
    )


def _render_event(event: Any) -> str:
    """Render one standardized agent-loop event."""

    if not isinstance(event, dict):
        event = {"type": "error", "content": f"Invalid event: {event!r}"}

    event_type = str(event.get("type", "unknown"))
    event_class = event_type.replace("_", "-")
    label = EVENT_LABELS.get(event_type, event_type.replace("_", " ").title())
    details = _event_details(event)
    body = _event_body(event)

    return (
        f'<section class="event {event_class}">'
        '<header class="event-header">'
        f'<span class="event-title">{label}</span>'
        f'<span class="event-meta">{details}</span>'
        "</header>"
        f'<div class="event-body">{body}</div>'
        "</section>"
    )


def _event_details(event: dict[str, Any]) -> str:
    """Return compact event metadata for the HTML header."""

    parts = []

    for key in ("sequence", "turn", "kind", "role", "name", "call_id", "source", "status"):
        value = event.get(key)

        if value is not None:
            parts.append(f"{key}={format_event_content(value)}")

    return " · ".join(parts)


def _event_body(event: dict[str, Any]) -> str:
    """Render visible content and collapsible full snapshots for one event."""

    event_type = event.get("type")

    if event_type in {"model_request", "model_response"}:
        raw = format_event_content(event.get("raw", ""))
        note = (
            "New instructions, messages, and tool results supplied to the model "
            "are shown in the following blocks."
            if event_type == "model_request"
            else "The meaningful generated reasoning, text, and tool calls are shown in the following blocks."
        )
        return (
            f'<p class="event-note">{note}</p>'
            '<details><summary>Full raw snapshot</summary>'
            f"<pre>{raw}</pre>"
            "</details>"
        )

    if event_type == "tool_call":
        content = {
            "name": event.get("name"),
            "arguments": event.get("arguments")
        }
    elif event_type == "tool_definitions":
        content = event.get("tools", [])
    else:
        content = event.get("content", event.get("payload", {}))

    body = f"<pre>{format_event_content(content)}</pre>"

    if "payload" in event and event_type not in {"tool_call", "tool_definitions"}:
        body += (
            '<details><summary>Full structured payload</summary>'
            f"<pre>{format_event_content(event['payload'])}</pre>"
            "</details>"
        )

    return body


def main() -> None:
    """Parse command-line arguments and render standardized agent loops."""

    parser = argparse.ArgumentParser(
        description="Create readable HTML transcripts from agent_loop.json files"
    )
    parser.add_argument("--results_dir",
                        "--results-dir",
                        default="results/external-agents",
                        dest="results_dir",
                        help="Results directory scanned recursively")
    args = parser.parse_args()
    generated, failures = build_agent_loop_transcripts(args.results_dir)
    print(f"Generated {generated} agent-loop HTML transcripts")

    if failures:
        raise SystemExit(f"Could not transcribe {failures} agent-loop traces")


if __name__ == "__main__":
    main()
