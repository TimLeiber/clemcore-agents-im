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
    content: "Claude thinking\\A";
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
    content: "Claude\\A";
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


def parse_agent_trace(trace_path: Path) -> list[dict[str, Any]]:
    events = []

    for line in trace_path.read_text(encoding="utf-8").splitlines():
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
    for index in range(start_index, len(events)):
        event = events[index]

        if event["kind"] != "tool-use":
            continue

        if event.get("tool_name") != "mcp__clem-game__submit_response":
            continue

        if event.get("response") == response:
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
                    transcript += _render_trace_events(
                        trace_events[trace_index:submit_index + 1],
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
