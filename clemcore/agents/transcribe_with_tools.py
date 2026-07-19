import argparse
import ast
import glob
import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

import clemcore.clemgame.transcripts.html_templates as html_templates
from clemcore.clemgame.resources import load_json
from clemcore.clemgame.transcripts.builder import (
    _get_class_name,
    get_css,
    get_css_player_dict,
)


module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")


TOOL_TRACE_CSS = """
.msg.agent-reasoning,
.msg.agent-tool-use,
.msg.agent-tool-result,
.msg.agent-summary {
    border-width: 2px !important;
    border-style: solid !important;
    opacity: 0.96;
}

.msg.agent-reasoning {
    border-color: #7b4bb3 !important;
}

.msg.agent-tool-use {
    border-color: #c46f00 !important;
}

.msg.agent-tool-result {
    border-color: #2f8f5b !important;
}

.msg.agent-summary {
    border-color: #2d6ecf !important;
}

.msg.agent-reasoning p::before {
    content: "AGENT thinking\\A";
    white-space: pre;
    font-weight: 700;
}

.msg.agent-tool-use p::before {
    content: "Tool call\\A";
    white-space: pre;
    font-weight: 700;
}

.msg.agent-tool-result p::before {
    content: "Tool result\\A";
    white-space: pre;
    font-weight: 700;
}

.msg.agent-summary p::before {
    content: "AGENT\\A";
    white-space: pre;
    font-weight: 700;
}
"""


def _try_literal_eval(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def _literal_pattern() -> str:
    return r"'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"|None|True|False|-?\d+(?:\.\d+)?"


def _extract_literal_keyword(text: str,
                             keyword: str) -> Any | None:
    match = re.search(
        rf"{re.escape(keyword)}=(?P<value>{_literal_pattern()})",
        text,
    )

    if not match:
        return None

    return _try_literal_eval(match.group("value"))


def _extract_between(text: str,
                     start_marker: str,
                     end_marker: str) -> str | None:
    start = text.find(start_marker)

    if start < 0:
        return None

    start += len(start_marker)
    end = text.find(end_marker, start)

    if end < 0:
        return None

    return text[start:end]


def _trace_event(kind: str,
                 content: str,
                 tool_name: str | None = None,
                 tool_use_id: str | None = None,
                 response: str | None = None,
                 raw_result: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "content": content,
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "response": response,
        "raw_result": raw_result,
    }


def _parse_thinking_block(line: str) -> dict[str, Any] | None:
    match = re.search(
        r"ThinkingBlock\(thinking=(?P<thinking>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"), signature=",
        line,
    )

    if not match:
        return None

    thinking = _try_literal_eval(match.group("thinking"))

    return _trace_event(
        kind="reasoning",
        content=str(thinking).strip(),
    )


def _parse_text_block(line: str) -> dict[str, Any] | None:
    match = re.search(
        r"TextBlock\(text=(?P<text>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")\)",
        line,
    )

    if not match:
        return None

    text = _try_literal_eval(match.group("text"))

    return _trace_event(
        kind="assistant-text",
        content=str(text).strip(),
    )


def _parse_tool_use_block(line: str) -> dict[str, Any] | None:
    match = re.search(
        r"ToolUseBlock\(id=(?P<id>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"), "
        r"name=(?P<name>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"), "
        r"input=(?P<input>\{.*?\})\)",
        line,
    )

    if not match:
        return None

    tool_use_id = _try_literal_eval(match.group("id"))
    tool_name = _try_literal_eval(match.group("name"))
    tool_input = _try_literal_eval(match.group("input"))

    response = None

    if isinstance(tool_input, dict):
        response = tool_input.get("response")

    if tool_name == "ToolSearch":
        content = f"ToolSearch(query={tool_input.get('query')!r}, max_results={tool_input.get('max_results')!r})"
    elif tool_name == "mcp__clem-game__start_game":
        content = "mcp__clem-game__start_game()"
    elif tool_name == "mcp__clem-game__submit_response":
        content = f"mcp__clem-game__submit_response(response={response!r})"
    else:
        content = f"{tool_name}({json.dumps(tool_input, ensure_ascii=False)})"

    return _trace_event(
        kind="tool-use",
        content=content,
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        response=response,
    )


def _parse_tool_result_content(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except Exception:
            return {
                "content": content,
            }

        result = {
            "reward": parsed.get("reward"),
            "done": parsed.get("done"),
            "metadata": parsed.get("metadata"),
        }

        context = parsed.get("context")

        if isinstance(context, dict):
            context_content = context.get("content", "")
            result["context_summary"] = (
                context_content.splitlines()[0]
                if context_content
                else ""
            )

        return result

    if isinstance(content, list):
        tool_names = [
            item.get("tool_name")
            for item in content
            if isinstance(item, dict) and item.get("tool_name")
        ]

        if tool_names:
            return {
                "available_tools": tool_names,
            }

    return {
        "content": content,
    }


def _parse_tool_result_block(line: str) -> dict[str, Any] | None:
    if "ToolResultBlock(" not in line:
        return None

    match = re.search(
        r"ToolResultBlock\(tool_use_id=(?P<id>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"), "
        r"content=(?P<content>.*?), "
        r"is_error=(?P<is_error>None|True|False)\)",
        line,
    )

    if not match:
        return None

    tool_use_id = _try_literal_eval(match.group("id"))
    content = _try_literal_eval(match.group("content"))
    is_error = _try_literal_eval(match.group("is_error"))

    parsed_content = _parse_tool_result_content(content)

    parts = [
        f"tool_use_id: {tool_use_id}",
        f"is_error: {_format_scalar(is_error)}",
    ]

    if "available_tools" in parsed_content:
        parts.append("available_tools: " + ", ".join(parsed_content["available_tools"]))
    else:
        if "reward" in parsed_content:
            parts.append(f"reward: {_format_scalar(parsed_content.get('reward'))}")
        if "done" in parsed_content:
            parts.append(f"done: {_format_scalar(parsed_content.get('done'))}")
        if parsed_content.get("metadata"):
            parts.append("metadata: " + json.dumps(parsed_content["metadata"], ensure_ascii=False))
        if parsed_content.get("context_summary"):
            parts.append("context: returned game observation; full text is shown in the Game Master bubble")

    return _trace_event(
        kind="tool-result",
        content="\n".join(parts),
        tool_use_id=tool_use_id,
        raw_result=parsed_content,
    )


def _parse_result_message(line: str) -> dict[str, Any] | None:
    if not line.startswith("ResultMessage("):
        return None

    result_literal = _extract_between(line, "result=", ", structured_output=")
    result = _try_literal_eval(result_literal) if result_literal else None

    total_cost_usd = _extract_literal_keyword(line, "total_cost_usd")
    num_turns = _extract_literal_keyword(line, "num_turns")
    stop_reason = _extract_literal_keyword(line, "stop_reason")

    content = "\n".join(
        part
        for part in [
            str(result).strip() if result else "",
            "",
            f"num_turns: {num_turns}",
            f"stop_reason: {stop_reason}",
            f"total_cost_usd: {total_cost_usd}",
        ]
        if part is not None
    ).strip()

    return _trace_event(
        kind="summary",
        content=content,
    )


def _canonical_tool_name(tool_name: str | None) -> str | None:
    if tool_name in {"mcp_clem_game_start_game", "clem_game__start_game"}:
        return "mcp__clem-game__start_game"

    if tool_name in {"mcp_clem_game_submit_response", "clem_game__submit_response"}:
        return "mcp__clem-game__submit_response"

    if tool_name in {"mcp_clem_game_get_state", "clem_game__get_state"}:
        return "mcp__clem-game__get_state"

    return tool_name


def _format_tool_call(tool_name: str,
                      tool_input: dict[str, Any]) -> tuple[str, str | None]:
    response = tool_input.get("response") if isinstance(tool_input, dict) else None
    canonical_name = _canonical_tool_name(tool_name)

    if canonical_name == "mcp__clem-game__start_game":
        return "mcp__clem-game__start_game()", response

    if canonical_name == "mcp__clem-game__submit_response":
        return f"mcp__clem-game__submit_response(response={response!r})", response

    return f"{tool_name}({json.dumps(tool_input, ensure_ascii=False)})", response


def _parse_hermes_debug_tool_call(line: str) -> dict[str, Any] | None:
    match = re.search(
        r"Tool call: (?P<name>mcp_clem_game_[a-z_]+) with args: (?P<args>.*?)(?:\.\.\.)?$",
        line,
    )

    if not match:
        return None

    tool_name = match.group("name")
    raw_args = match.group("args").strip()

    try:
        tool_input = json.loads(raw_args)
    except Exception:
        tool_input = {"raw_args": raw_args}

    content, response = _format_tool_call(tool_name, tool_input)

    return _trace_event(
        kind="tool-use",
        content=content,
        tool_name=_canonical_tool_name(tool_name),
        response=response,
    )


def _parse_hermes_debug_reasoning(line: str) -> dict[str, Any] | None:
    match = re.search(r"Captured reasoning \(\d+ chars\): (?P<reasoning>.*)$", line)

    if not match:
        return None

    return _trace_event(
        kind="reasoning",
        content=match.group("reasoning").strip(),
    )


def _parse_hermes_tool_result_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        structured = payload.get("structuredContent")

        if isinstance(structured, dict):
            return {
                "reward": structured.get("reward"),
                "done": structured.get("done"),
                "metadata": structured.get("metadata"),
                "context_summary": (
                    "returned game observation; full text is shown in the Game Master bubble"
                    if structured.get("context")
                    else ""
                ),
            }

        nested_result = payload.get("result")

        if isinstance(nested_result, str):
            return _parse_tool_result_content(nested_result)

    return {
        "content": payload,
    }


def _parse_hermes_debug_tool_result(line: str) -> dict[str, Any] | None:
    match = re.search(r"Tool result \([^)]*\): (?P<payload>\{.*\})$", line)

    if not match:
        return None

    try:
        payload = json.loads(match.group("payload"))
    except Exception:
        return _trace_event(
            kind="tool-result",
            content=match.group("payload"),
        )

    parsed_content = _parse_hermes_tool_result_payload(payload)

    parts = []

    if "reward" in parsed_content:
        parts.append(f"reward: {_format_scalar(parsed_content.get('reward'))}")
    if "done" in parsed_content:
        parts.append(f"done: {_format_scalar(parsed_content.get('done'))}")
    if parsed_content.get("metadata"):
        parts.append("metadata: " + json.dumps(parsed_content["metadata"], ensure_ascii=False))
    if parsed_content.get("context_summary"):
        parts.append("context: " + parsed_content["context_summary"])
    if parsed_content.get("content") and not parts:
        parts.append(str(parsed_content["content"]))

    return _trace_event(
        kind="tool-result",
        content="\n".join(parts),
        raw_result=parsed_content,
    )


def _parse_hermes_trace(text: str) -> list[dict[str, Any]]:
    events = []
    pending_tool_call = None
    pending_reasoning_lines = None

    timestamp_pattern = re.compile(r"^\d{2}:\d{2}:\d{2} - ")

    def flush_reasoning() -> None:
        nonlocal pending_reasoning_lines
        nonlocal pending_tool_call

        if pending_reasoning_lines is None:
            return

        content = "\n".join(pending_reasoning_lines).strip()

        if content:
            # Hermes logs the captured reasoning after the tool-call line.
            # Render it before the tool call, matching the model's actual order.
            events.append(_trace_event(kind="reasoning", content=content))

            if pending_tool_call is not None:
                events.append(pending_tool_call)
                pending_tool_call = None

        pending_reasoning_lines = None

    for line in text.splitlines():
        line = line.rstrip()

        if pending_reasoning_lines is not None:
            if timestamp_pattern.match(line):
                flush_reasoning()
            else:
                pending_reasoning_lines.append(line)
                continue

        if not line:
            continue

        tool_call = _parse_hermes_debug_tool_call(line)

        if tool_call is not None:
            if pending_tool_call is not None:
                events.append(pending_tool_call)

            pending_tool_call = tool_call
            continue

        reasoning = _parse_hermes_debug_reasoning(line)

        if reasoning is not None:
            pending_reasoning_lines = [reasoning["content"]]
            continue

        tool_result = _parse_hermes_debug_tool_result(line)

        if tool_result is not None:
            flush_reasoning()

            if pending_tool_call is not None:
                events.append(pending_tool_call)
                pending_tool_call = None

            events.append(tool_result)
            continue

    flush_reasoning()

    if pending_tool_call is not None:
        events.append(pending_tool_call)

    return events


def _format_openclaw_tool_call(tool_name: str,
                               tool_use_id: str | None,
                               tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
    canonical_name = _canonical_tool_name(tool_name)
    tool_input = tool_input or {}
    content, response = _format_tool_call(canonical_name or tool_name, tool_input)

    if tool_use_id:
        content += f"\ntool_use_id: {tool_use_id}"

    return _trace_event(
        kind="tool-use",
        content=content,
        tool_name=canonical_name,
        tool_use_id=tool_use_id,
        response=response,
    )


def _format_openclaw_tool_result(tool_name: str | None,
                                 tool_use_id: str | None,
                                 payload: Any | None = None,
                                 is_error: bool | None = None) -> dict[str, Any]:
    canonical_name = _canonical_tool_name(tool_name)

    parsed_content = _parse_openclaw_tool_result_payload(payload)

    parts = []

    if tool_use_id:
        parts.append(f"tool_use_id: {tool_use_id}")

    if is_error is not None:
        parts.append(f"is_error: {_format_scalar(is_error)}")

    if "reward" in parsed_content:
        parts.append(f"reward: {_format_scalar(parsed_content.get('reward'))}")
    if "done" in parsed_content:
        parts.append(f"done: {_format_scalar(parsed_content.get('done'))}")
    if parsed_content.get("metadata"):
        parts.append("metadata: " + json.dumps(parsed_content["metadata"], ensure_ascii=False))
    if parsed_content.get("context_summary"):
        parts.append("context: " + parsed_content["context_summary"])
    if parsed_content.get("content") and not any(
        item.startswith(("reward:", "done:", "metadata:", "context:"))
        for item in parts
    ):
        parts.append(str(parsed_content["content"]))

    if not parts:
        parts.append("status: completed")

    return _trace_event(
        kind="tool-result",
        content="\n".join(parts),
        tool_name=canonical_name,
        tool_use_id=tool_use_id,
        raw_result=parsed_content,
    )


def _parse_openclaw_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if not stripped:
        return value

    try:
        return json.loads(stripped)
    except Exception:
        return value


def _parse_openclaw_tool_arguments(value: Any) -> dict[str, Any]:
    value = _parse_openclaw_json_maybe(value)

    if isinstance(value, dict):
        return value

    return {"raw_args": value}


def _parse_openclaw_tool_result_payload(payload: Any) -> dict[str, Any]:
    payload = _parse_openclaw_json_maybe(payload)

    if isinstance(payload, dict):
        # OpenClaw session JSONL has used a few shapes across traces:
        # - {"structuredContent": {...}}
        # - {"content": {"structuredContent": {...}}}
        # - {"result": "...json..."}
        structured = payload.get("structuredContent")

        if structured is None and isinstance(payload.get("details"), dict):
            structured = payload["details"].get("structuredContent")

        if structured is None and isinstance(payload.get("content"), dict):
            structured = payload["content"].get("structuredContent")

        if structured is None and isinstance(payload.get("content"), list):
            for item in payload["content"]:
                if not isinstance(item, dict):
                    continue

                text_value = item.get("text")

                if not isinstance(text_value, str):
                    continue

                marker = "structuredContent:"

                if marker not in text_value:
                    continue

                structured_candidate = _parse_openclaw_json_maybe(
                    text_value.split(marker, 1)[1].strip()
                )

                if isinstance(structured_candidate, dict):
                    structured = structured_candidate
                    break

        if structured is None and isinstance(payload.get("result"), dict):
            structured = payload["result"].get("structuredContent")

        if structured is None and isinstance(payload.get("result"), str):
            return _parse_tool_result_content(payload["result"])

        candidate = structured if isinstance(structured, dict) else payload

        result = {}

        if "reward" in candidate:
            result["reward"] = candidate.get("reward")
        if "done" in candidate:
            result["done"] = candidate.get("done")
        if "metadata" in candidate:
            result["metadata"] = candidate.get("metadata")

        context = candidate.get("context")

        if isinstance(context, dict):
            context_content = context.get("content", "")
            result["context_summary"] = (
                context_content.splitlines()[0]
                if context_content
                else "returned game observation; full text is shown in the Game Master bubble"
            )

        if result:
            return result

    if isinstance(payload, str):
        return _parse_tool_result_content(payload)

    return {
        "content": payload,
    }


def _extract_openclaw_reasoning_from_part(part: dict[str, Any]) -> str | None:
    part_type = str(part.get("type", "")).lower()

    if "reason" not in part_type and "think" not in part_type:
        return None

    for key in [
        "text",
        "content",
        "reasoning",
        "thinking",
        "summary",
        "delta",
    ]:
        value = part.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _parse_openclaw_session_json_line(line: str) -> list[dict[str, Any]]:
    stripped = line.strip()

    if not stripped.startswith("{"):
        return []

    try:
        record = json.loads(stripped)
    except Exception:
        return []

    if record.get("type") != "message":
        return []

    message = record.get("message")

    if not isinstance(message, dict):
        return []

    role = message.get("role")
    content = message.get("content")
    events = []

    if not isinstance(content, list):
        return events

    if role == "assistant":
        for part in content:
            if not isinstance(part, dict):
                continue

            reasoning_text = _extract_openclaw_reasoning_from_part(part)

            if reasoning_text:
                events.append(
                    _trace_event(
                        kind="reasoning",
                        content=reasoning_text,
                    )
                )
                continue

            if part.get("type") == "toolCall":
                tool_name = part.get("name")
                tool_use_id = part.get("id")
                tool_input = _parse_openclaw_tool_arguments(
                    part.get("arguments", part.get("partialArgs", {}))
                )

                events.append(
                    _format_openclaw_tool_call(
                        tool_name=tool_name,
                        tool_use_id=tool_use_id,
                        tool_input=tool_input,
                    )
                )

    elif role in {"tool", "toolResult"}:
        if role == "toolResult":
            events.append(
                _format_openclaw_tool_result(
                    tool_name=message.get("toolName"),
                    tool_use_id=message.get("toolCallId"),
                    payload=message,
                    is_error=message.get("isError"),
                )
            )
            return events

        for part in content:
            if not isinstance(part, dict):
                continue

            if part.get("type") not in {"toolResult", "tool_result"}:
                continue

            events.append(
                _format_openclaw_tool_result(
                    tool_name=part.get("name") or part.get("toolName"),
                    tool_use_id=(
                        part.get("toolCallId")
                        or part.get("tool_call_id")
                        or part.get("id")
                    ),
                    payload=(
                        part.get("content")
                        if "content" in part
                        else part.get("result")
                    ),
                    is_error=part.get("isError"),
                )
            )

    return events


def _parse_openclaw_console_trace(text: str) -> list[dict[str, Any]]:
    events = []

    tool_start_pattern = re.compile(
        r"\[agent/embedded\] embedded run tool start: .*? "
        r"tool=(?P<tool>\S+) toolCallId=(?P<id>\S+)"
    )
    tool_end_pattern = re.compile(
        r"\[agent/embedded\] embedded run tool end: .*? "
        r"tool=(?P<tool>\S+) toolCallId=(?P<id>\S+)"
    )

    for line in text.splitlines():
        line = line.rstrip()

        start_match = tool_start_pattern.search(line)

        if start_match:
            events.append(
                _format_openclaw_tool_call(
                    tool_name=start_match.group("tool"),
                    tool_use_id=start_match.group("id"),
                )
            )
            continue

        end_match = tool_end_pattern.search(line)

        if end_match:
            events.append(
                _format_openclaw_tool_result(
                    tool_name=end_match.group("tool"),
                    tool_use_id=end_match.group("id"),
                )
            )
            continue

        if "incomplete turn detected:" in line:
            events.append(
                _trace_event(
                    kind="summary",
                    content=(
                        "OpenClaw reported incomplete_turn after tool execution. "
                        "This is treated as an adapter/UI completion policy issue "
                        "when native clembench episode artifacts exist."
                    ),
                )
            )

    return events


def _parse_openclaw_trace(text: str) -> list[dict[str, Any]]:
    if (
        "openclaw_agent_command:" not in text
        and "openclaw_agent_stdout:" not in text
        and "openclaw_agent_stderr:" not in text
    ):
        return []

    session_events = []

    for line in text.splitlines():
        session_events.extend(_parse_openclaw_session_json_line(line))

    if session_events:
        # Add the incomplete_turn marker from console logs if present, but use
        # session JSONL for tool calls/results because it contains args/results.
        if "incomplete turn detected:" in text:
            session_events.append(
                _trace_event(
                    kind="summary",
                    content=(
                        "OpenClaw reported incomplete_turn after tool execution. "
                        "This is treated as an adapter/UI completion policy issue "
                        "when native clembench episode artifacts exist."
                    ),
                )
            )

        return session_events

    return _parse_openclaw_console_trace(text)


def parse_agent_trace(trace_path: Path) -> list[dict[str, Any]]:
    trace_text = trace_path.read_text(encoding="utf-8", errors="replace")

    hermes_events = _parse_hermes_trace(trace_text)

    if hermes_events:
        return _repair_hermes_trace_events(hermes_events, trace_text)

    openclaw_events = _normalize_trace_events_for_render(
        _parse_openclaw_trace(trace_text) + _parse_codex_trace(trace_text)
    )

    if openclaw_events:
        return openclaw_events

    events = []

    for line in trace_text.splitlines():
        line = line.rstrip()

        if not line:
            continue

        # Drop noisy/non-qualitative events.
        if line.startswith("SystemMessage("):
            continue

        if line.startswith("success:"):
            continue

        parsed = (
            _parse_thinking_block(line)
            or _parse_tool_use_block(line)
            or _parse_tool_result_block(line)
            or _parse_text_block(line)
            or _parse_result_message(line)
        )

        if parsed is None:
            continue

        # Drop generic assistant narration. The actual benchmark-relevant action
        # is represented by the following MCP tool call and native transcript event.
        if parsed["kind"] == "assistant-text":
            continue

        events.append(parsed)

    return events


def _html_message(content: str) -> str:
    return html.escape(content).replace("\n", "<br/>")



def _parse_codex_trace(text: str) -> list[dict[str, Any]]:
    """Parse Codex `codex exec --json` events from agent_trace.log.

    Codex emits JSONL under a `codex_stdout_jsonl:` marker. We convert MCP
    calls and command executions into the same generic tool transcript events
    used by the renderer.
    """
    if "codex_stdout_jsonl:" not in text:
        return []

    block = text.split("codex_stdout_jsonl:", 1)[1]
    if "\ncodex_stderr:" in block:
        block = block.split("\ncodex_stderr:", 1)[0]
    if "\nsuccess:" in block:
        block = block.split("\nsuccess:", 1)[0]

    events: list[dict[str, Any]] = []

    for raw in block.splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue

        try:
            event = json.loads(raw)
        except Exception:
            continue

        item = event.get("item") or {}
        item_type = item.get("type")
        event_type = event.get("type")

        if event_type == "turn.completed":
            usage = event.get("usage") or {}
            if usage:
                events.append({
                    "kind": "agent_message",
                    "text": "Codex usage: " + json.dumps(usage, ensure_ascii=False),
                })
            continue

        if item_type == "agent_message":
            txt = item.get("text") or ""
            if txt.strip():
                events.append({
                    "kind": "agent_message",
                    "text": txt,
                })
            continue

        if item_type == "reasoning":
            txt = item.get("text") or item.get("summary") or item.get("content") or ""
            if isinstance(txt, list):
                txt = "\n".join(str(x) for x in txt)
            if str(txt).strip():
                events.append({
                    "kind": "agent_reasoning",
                    "text": str(txt),
                })
            continue

        if item_type == "mcp_tool_call":
            tool = item.get("tool")
            server = item.get("server")
            name = f"{server}__{tool}" if server and tool else tool or "mcp_tool_call"
            canonical = _canonical_tool_name(name)

            args = item.get("arguments") or {}
            status = item.get("status")

            if event_type == "item.started":
                events.append({
                    "kind": "agent_tool_use",
                    "tool_name": canonical,
                    "tool_input": args,
                    "tool_use_id": item.get("id"),
                })

            elif event_type == "item.completed":
                result = item.get("result")
                error = item.get("error")
                events.append({
                    "kind": "agent_tool_result",
                    "tool_name": canonical,
                    "tool_use_id": item.get("id"),
                    "is_error": bool(error) or status == "failed",
                    "tool_result": error if error is not None else result,
                })
            continue

        if item_type == "command_execution":
            command = item.get("command") or ""
            if event_type == "item.started":
                events.append({
                    "kind": "agent_tool_use",
                    "tool_name": "command_execution",
                    "tool_input": {"command": command},
                    "tool_use_id": item.get("id"),
                })

            elif event_type == "item.completed":
                out = item.get("aggregated_output")
                exit_code = item.get("exit_code")
                status = item.get("status")
                events.append({
                    "kind": "agent_tool_result",
                    "tool_name": "command_execution",
                    "tool_use_id": item.get("id"),
                    "is_error": bool(exit_code not in (0, None)),
                    "tool_result": {
                        "exit_code": exit_code,
                        "status": status,
                        "output": out,
                    },
                })
            continue

    return events



def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _normalize_trace_events_for_render(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make parsed harness events match the transcript renderer schema.

    Renderer expects:
      kind: reasoning | tool-use | tool-result | summary | ...
      content: string

    Codex parser initially keeps structured fields, so normalize them here while
    preserving tool_name/tool_use_id/tool_input/tool_result for alignment logic.
    """
    normalized: list[dict[str, Any]] = []

    for event in events:
        if "content" in event and event.get("kind") not in {
            "agent_tool_use",
            "agent_tool_result",
            "agent_reasoning",
            "agent_message",
        }:
            normalized.append(event)
            continue

        event = dict(event)
        kind = event.get("kind")

        if kind == "agent_tool_use":
            tool_name = event.get("tool_name") or "tool"
            tool_input = event.get("tool_input")
            event["kind"] = "tool-use"
            event["content"] = (
                f"{tool_name}({_compact_json(tool_input)})\n"
                f"tool_use_id: {event.get('tool_use_id')}"
            )

        elif kind == "agent_tool_result":
            tool_name = event.get("tool_name") or "tool"
            event["kind"] = "tool-result"
            event["content"] = (
                f"{tool_name}\n"
                f"tool_use_id: {event.get('tool_use_id')}\n"
                f"is_error: {event.get('is_error')}\n"
                f"{_compact_json(event.get('tool_result'))}"
            )

        elif kind == "agent_reasoning":
            event["kind"] = "reasoning"
            event["content"] = str(event.get("text") or "")

        elif kind == "agent_message":
            event["kind"] = "summary"
            event["content"] = str(event.get("text") or "")

        normalized.append(event)

    return normalized



def _trace_event_tool_name(event: dict[str, Any]) -> str | None:
    tool_name = event.get("tool_name")
    if tool_name:
        return tool_name

    content = str(event.get("content", ""))
    if content.startswith("mcp__clem-game__start_game"):
        return "mcp__clem-game__start_game"
    if content.startswith("mcp__clem-game__submit_response"):
        return "mcp__clem-game__submit_response"
    if content.startswith("mcp__clem-game__get_state"):
        return "mcp__clem-game__get_state"
    if content.startswith("execute_code"):
        return "execute_code"
    if content.startswith("command_execution"):
        return "command_execution"

    return None


def _format_tool_call_content(tool_name: str, args: Any) -> str:
    if isinstance(args, dict) and len(args) == 0:
        return f"{tool_name}()"

    if tool_name == "mcp__clem-game__submit_response" and isinstance(args, dict):
        return f"{tool_name}(response={json.dumps(args.get('response'), ensure_ascii=False)})"

    if tool_name == "mcp__clem-game__start_game" and isinstance(args, dict):
        return f"{tool_name}({_compact_json(args)})" if args else f"{tool_name}()"

    if tool_name == "execute_code" and isinstance(args, dict) and "code" in args:
        code = str(args.get("code"))
        if len(code) > 4000:
            code = code[:4000] + "\n... [truncated]"
        return f"{tool_name}(code={json.dumps(code, ensure_ascii=False)})"

    return f"{tool_name}({_compact_json(args)})"


def _extract_hermes_debug_tool_calls(trace_text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    for line in trace_text.splitlines():
        if " - root - DEBUG " not in line or " - Tool call: " not in line or " with args: " not in line:
            continue

        part = line.split(" - Tool call: ", 1)[1]
        tool_name, raw_args = part.split(" with args: ", 1)
        tool_name = _canonical_tool_name(tool_name.strip())
        raw_args = raw_args.strip()

        # Hermes may append literal "..." when it truncates debug args.
        if raw_args.endswith("..."):
            raw_args = raw_args[:-3].rstrip()

        args: Any
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {"_raw_args": raw_args}

        calls.append({
            "tool_name": tool_name,
            "args": args,
        })

    return calls


def _repair_hermes_trace_events(events: list[dict[str, Any]], trace_text: str) -> list[dict[str, Any]]:
    """Repair Hermes transcript events.

    Hermes has two console representations:
    - compact CLI lines like Tool 1: mcp_clem_game_submit_response(['response'])
      which only contain argument names;
    - DEBUG lines like Tool call: ... with args: {...}
      which contain real argument values.

    The previous parser often used the compact line and produced response=None.
    This repair overlays DEBUG args where available and adds missing execute_code
    tool-use bubbles before the corresponding raw result.
    """
    debug_calls = _extract_hermes_debug_tool_calls(trace_text)
    debug_pos = 0
    repaired: list[dict[str, Any]] = []
    inserted_execute_code = False

    for event in events:
        event = dict(event)
        kind = event.get("kind")
        content = str(event.get("content", ""))
        tool_name = _trace_event_tool_name(event)

        # Fill missing tool_name for later alignment.
        if tool_name and "tool_name" not in event:
            event["tool_name"] = tool_name

        # Repair lossy Hermes tool-use content using DEBUG call args.
        if kind == "tool-use" and tool_name:
            # Find next DEBUG call for same tool.
            match = None
            while debug_pos < len(debug_calls):
                candidate = debug_calls[debug_pos]
                debug_pos += 1
                if candidate["tool_name"] == tool_name:
                    match = candidate
                    break

            if match is not None:
                event["tool_input"] = match["args"]
                event["content"] = _format_tool_call_content(tool_name, match["args"])

        # Hermes sometimes shows the execute_code result without the call bubble.
        # Insert the missing call immediately before the first execute_code result.
        if (
            not inserted_execute_code
            and kind in {"tool-result", "summary", "reasoning"}
            and (
                "CANDIDATES_COUNT" in content
                or '"tool_calls_made": 1' in content
                or "'tool_calls_made': 1" in content
            )
        ):
            execute_call = next(
                (c for c in debug_calls if c["tool_name"] == "execute_code"),
                None,
            )
            if execute_call is not None:
                repaired.append({
                    "kind": "tool-use",
                    "tool_name": "execute_code",
                    "tool_input": execute_call["args"],
                    "content": _format_tool_call_content("execute_code", execute_call["args"]),
                })
                inserted_execute_code = True

        repaired.append(event)

    return repaired


def _fill_native_submit_response(events: list[dict[str, Any]], response: str) -> list[dict[str, Any]]:
    """Use the native clembench response to repair Hermes response=None bubbles."""
    out: list[dict[str, Any]] = []
    filled = False

    for event in events:
        event = dict(event)
        tool_name = _trace_event_tool_name(event)

        if (
            not filled
            and event.get("kind") == "tool-use"
            and tool_name == "mcp__clem-game__submit_response"
        ):
            content = str(event.get("content", ""))
            tool_input = event.get("tool_input")

            if (
                "response=None" in content
                or not isinstance(tool_input, dict)
                or tool_input.get("response") in {None, ""}
                or "_raw_args" in tool_input
            ):
                event["tool_name"] = tool_name
                event["tool_input"] = {"response": response}
                event["content"] = _format_tool_call_content(tool_name, {"response": response})
                filled = True

        out.append(event)

    return out


def _render_bubble(speaker_attr: str,
                   class_name: str,
                   content: str,
                   extra_class: str) -> str:
    return html_templates.EVENT_TEMPLATE.format(
        html.escape(speaker_attr),
        f"{class_name} {extra_class}",
        "",
        _html_message(content),
    )


def _external_player_name(players: dict[str, Any]) -> str:
    for player_name, player_info in players.items():
        if player_name == "GM":
            continue

        if player_info.get("model_name") == "learner":
            return player_name

    return "Player 1"


def _trace_class_name(event: dict[str, Any],
                      external_player: str,
                      css_player_dict: dict[str, str]) -> str:
    player_css = css_player_dict.get(external_player, "p1")

    if event["kind"] in {"tool-result"}:
        return f"gm-player {player_css}"

    return f"player-gm {player_css}"


def _trace_speaker(event: dict[str, Any],
                   external_player: str,
                   players: dict[str, Any]) -> str:
    role = players.get(external_player, {}).get("game_role", "External Agent")

    if event["kind"] == "tool-result":
        return f"MCP tool result to {external_player} ({role})"

    return f"{external_player} ({role}) to MCP/tool layer"


def _render_trace_events(events: list[dict[str, Any]],
                         external_player: str,
                         players: dict[str, Any],
                         css_player_dict: dict[str, str]) -> str:
    rendered = ""

    for event in events:
        class_name = _trace_class_name(event, external_player, css_player_dict)
        speaker = _trace_speaker(event, external_player, players)
        extra_class = f"agent-{event['kind']}"
        rendered += _render_bubble(
            speaker_attr=speaker,
            class_name=class_name,
            content=event["content"],
            extra_class=extra_class,
        )

    return rendered


def _find_next_submit_index(events: list[dict[str, Any]],
                            start_index: int,
                            response: str) -> int | None:
    """Find the trace span ending at the submit_response result for this native response.

    Hermes sometimes logs only response=null / response=None because the real
    DEBUG args are truncated. In that case, repair the trace event from the
    native clembench response while aligning.
    """
    def repair_event_with_native_response(event: dict[str, Any]) -> None:
        event["tool_name"] = "mcp__clem-game__submit_response"
        event["tool_input"] = {"response": response}
        event["content"] = _format_tool_call_content(
            "mcp__clem-game__submit_response",
            {"response": response},
        )

    def matches_response(event: dict[str, Any]) -> bool:
        if _trace_event_tool_name(event) != "mcp__clem-game__submit_response":
            return False

        tool_input = event.get("tool_input")
        if isinstance(tool_input, dict) and tool_input.get("response") == response:
            return True

        content = str(event.get("content", ""))

        if response in content:
            return True

        try:
            escaped = json.dumps(response, ensure_ascii=False)[1:-1]
            if escaped in content:
                return True
        except Exception:
            pass

        # Hermes fallback: align the next lossy submit bubble with the next
        # native player response, then repair the bubble before rendering.
        if (
            "response=None" in content
            or "response=null" in content
            or "response=None" in str(tool_input)
            or "response': None" in str(tool_input)
            or '"response": null' in str(tool_input)
            or "_raw_args" in str(tool_input)
        ):
            repair_event_with_native_response(event)
            return True

        return False

    for index in range(start_index, len(events)):
        event = events[index]

        if event.get("kind") != "tool-use":
            continue

        if not matches_response(event):
            continue

        tool_use_id = event.get("tool_use_id")

        for result_index in range(index + 1, len(events)):
            result_event = events[result_index]
            if result_event.get("kind") != "tool-result":
                continue
            if _trace_event_tool_name(result_event) != "mcp__clem-game__submit_response":
                continue
            if tool_use_id is None or result_event.get("tool_use_id") == tool_use_id:
                return result_index

        return index

    return None



def build_transcript_with_tools(interactions: dict[str, Any],
                                trace_path: Path) -> str:
    meta = interactions["meta"]
    players = interactions["players"]
    markdown = interactions.get("markdown", False)

    if markdown:
        raise NotImplementedError("transcript_with_tools currently expects non-markdown interactions.")

    css_player_dict = get_css_player_dict(players)
    external_player = _external_player_name(players)
    trace_events = parse_agent_trace(trace_path)
    trace_index = 0

    def render_native_event(event: dict[str, Any]) -> str:
        class_name, player = _get_class_name(event, css_player_dict)

        if player is not None:
            class_name += f" {player}"

        msg_content = event["action"]["content"]
        msg_raw = html.escape(f"{msg_content}").replace("\n", "<br/>")

        if event["from"] == "GM" and event["to"] == "GM":
            speaker_attr = f'Game Master internal: {event["action"]["type"]}'
        else:
            from_player = event["from"]
            to_player = event["to"]

            if (
                from_player in players
                and to_player in players
                and "game_role" in players[from_player]
                and "game_role" in players[to_player]
            ):
                from_game_role = players[from_player]["game_role"]
                to_game_role = players[to_player]["game_role"]
                speaker_attr = f"{from_player} ({from_game_role}) to {to_player} ({to_game_role})"
            else:
                speaker_attr = (
                    f"{event['from'].replace('GM', 'Game Master')} "
                    f"to {event['to'].replace('GM', 'Game Master')}"
                )

        style = "border: dashed" if event["action"].get("label") == "forget" else ""

        return html_templates.EVENT_TEMPLATE.format(
            speaker_attr,
            class_name,
            style,
            msg_raw,
        )

    def is_gm_internal_event(event: dict[str, Any]) -> bool:
        return event.get("from") == "GM" and event.get("to") == "GM"

    transcript = html_templates.HEADER.format(get_css(len(players)) + TOOL_TRACE_CSS)
    pair_descriptor = meta["results_folder"] if "results_folder" in meta else meta["dialogue_pair"]
    title = (
        f"Interaction Transcript with Tools for game '{meta['game_name']}', "
        f"experiment '{meta['experiment_name']}', episode {meta['game_id']} "
        f"with {pair_descriptor}."
    )
    transcript += html_templates.TOP_INFO.format(html.escape(title))

    for turn_idx, turn in enumerate(interactions["turns"]):
        transcript += f'<div class="game-round" data-round="{turn_idx}">'
        pending_submit_result = False
        deferred_gm_internal_events = []

        for event in turn:
            is_external_response = (
                event["from"] == external_player
                and event["to"] == "GM"
                and event["action"].get("label") == "response"
            )

            if is_external_response:
                response = event["action"]["content"]
                submit_index = _find_next_submit_index(trace_events, trace_index, response)

                if submit_index is not None:
                    trace_span = _fill_native_submit_response(
                        trace_events[trace_index:submit_index + 1],
                        response,
                    )
                    transcript += _render_trace_events(
                        trace_span,
                        external_player,
                        players,
                        css_player_dict,
                    )
                    trace_index = submit_index + 1
                    pending_submit_result = True

            elif trace_index == 0 and event["from"] == "GM" and event["to"] == external_player:
                start_result_index = None

                for index, trace_event in enumerate(trace_events):
                    if (
                        trace_event["kind"] == "tool-result"
                        and index > 0
                        and trace_events[index - 1].get("tool_name") == "mcp__clem-game__start_game"
                    ):
                        start_result_index = index
                        break

                if start_result_index is not None:
                    transcript += _render_trace_events(
                        trace_events[:start_result_index + 1],
                        external_player,
                        players,
                        css_player_dict,
                    )
                    trace_index = start_result_index + 1

            if is_gm_internal_event(event):
                deferred_gm_internal_events.append(event)
                continue

            transcript += render_native_event(event)

            if deferred_gm_internal_events and not is_external_response:
                for gm_event in deferred_gm_internal_events:
                    transcript += render_native_event(gm_event)
                deferred_gm_internal_events = []

        if deferred_gm_internal_events:
            for gm_event in deferred_gm_internal_events:
                transcript += render_native_event(gm_event)

        if pending_submit_result:
            next_submit_result = None

            for index in range(trace_index, len(trace_events)):
                trace_event = trace_events[index]

                if trace_event["kind"] == "tool-result":
                    next_submit_result = index
                    break

                if (
                    trace_event["kind"] == "tool-use"
                    and trace_event.get("tool_name") == "mcp__clem-game__submit_response"
                ):
                    break

            if next_submit_result is not None:
                transcript += _render_trace_events(
                    trace_events[trace_index:next_submit_result + 1],
                    external_player,
                    players,
                    css_player_dict,
                )
                trace_index = next_submit_result + 1

        transcript += "</div>"

    if trace_index < len(trace_events):
        transcript += '<div class="game-round" data-round="agent-final">'
        transcript += _render_trace_events(
            trace_events[trace_index:],
            external_player,
            players,
            css_player_dict,
        )
        transcript += "</div>"

    transcript += html_templates.FOOTER

    return transcript

def build_transcripts_with_tools(top_dir: str,
                                 filter_games: list[str] | None = None) -> None:
    if filter_games is None:
        filter_games = []

    interaction_files = glob.glob(
        os.path.join(top_dir, "**", "interactions.json"),
        recursive=True,
    )

    if filter_games:
        interaction_files = [
            interaction_file
            for interaction_file in interaction_files
            if any(game_name in Path(interaction_file).parts for game_name in filter_games)
        ]

    stdout_logger.info(
        "Found %s interaction files for tool transcript generation. Games: %s",
        len(interaction_files),
        filter_games if filter_games else "all",
    )

    written_count = 0
    skipped_count = 0
    error_count = 0

    for interaction_file in tqdm(interaction_files, desc="Building tool transcripts"):
        episode_dir = Path(interaction_file).parent
        trace_path = episode_dir / "agent_trace.log"

        if not trace_path.exists():
            skipped_count += 1
            continue

        try:
            interactions = load_json(interaction_file)
            transcript = build_transcript_with_tools(interactions, trace_path)
            output_path = episode_dir / "transcript_with_tools.html"
            output_path.write_text(transcript, encoding="utf-8")
            written_count += 1
        except Exception:
            module_logger.exception("Cannot build tool transcript for %s", interaction_file)
            error_count += 1

    stdout_logger.info(
        "Tool transcript generation finished: written=%s skipped_without_trace=%s errors=%s",
        written_count,
        skipped_count,
        error_count,
    )

    if error_count > 0:
        stdout_logger.error("'%s' exceptions occurred: see clembench.log for details.", error_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create transcript_with_tools.html files from interactions.json plus agent_trace.log.",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        default="results/external-agents",
        help="Results root directory.",
    )
    parser.add_argument(
        "-g",
        "--game",
        nargs="+",
        default=[],
        help="Optional game name filter, e.g. taboo.",
    )

    args = parser.parse_args()

    build_transcripts_with_tools(
        top_dir=args.results_dir,
        filter_games=args.game,
    )


if __name__ == "__main__":
    main()
