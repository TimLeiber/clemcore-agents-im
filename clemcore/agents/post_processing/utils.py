import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
import requests

from clemcore.backends.key_registry import KeyRegistry


def scores_by_experiment(game: str | None) -> pd.DataFrame:
    """Compute clembench scores for each experiment.

    Args:
        game: Game name to include, or None to include all games.

    Returns:
        Scores by game, model, and experiment as a dataframe.
    """

    raw_scores = pd.read_csv(Path("results/raw.csv"), index_col=0)

    if game is not None:
        raw_scores = raw_scores[raw_scores["game"] == game]

    raw_scores = raw_scores[raw_scores["metric"].isin(["Played", "Main Score"])].copy()
    raw_scores["value"] = pd.to_numeric(raw_scores["value"])
    episode_scores = raw_scores.pivot(
        index=["game", "model", "experiment", "episode"],
        columns="metric",
        values="value",
    ).reset_index()
    scores = episode_scores.groupby(["game", "model", "experiment"]).agg(
        episodes=("episode", "size"),
        played=("Played", "sum"),
        pct_played=("Played", "mean"),
        quality_score=("Main Score", "mean"),
    ).reset_index()
    scores["pct_played"] *= 100
    scores["total_score"] = scores["pct_played"] / 100 * scores["quality_score"]
    scores[["pct_played", "quality_score", "total_score"]] = scores[
        ["pct_played", "quality_score", "total_score"]
    ].round(2)
    scores[["episodes", "played"]] = scores[["episodes", "played"]].astype(int)

    return scores.sort_values(
        ["game", "experiment", "total_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def count_aborted_games(game: str | None) -> dict[str, dict[str, int]]:
    """Count and classify aborted external-agent games.

    Args:
        game: Game name to include, or None to include all games.

    Returns:
        Abort counts keyed by model and harness combination. Each value contains
        total, game instruction following, and agent failure counts.
    """

    results_dir = Path("results/external-agents")
    counts: dict[str, dict[str, int]] = {}

    for scores_path in sorted(results_dir.glob("**/scores.json")):
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        metadata = scores.get("meta", {})

        if game is not None and metadata.get("game_name") != game:
            continue

        agent = metadata["results_folder"]
        agent_counts = counts.setdefault(agent, {
            "total": 0,
            "game_instruction_following": 0,
            "agent_failure": 0,
        })

        if scores.get("episode scores", {}).get("Aborted") != 1:
            continue

        interactions_text = scores_path.with_name("interactions.json").read_text(
            encoding="utf-8",
            errors="replace",
        )
        category = ("agent_failure" if "CLEM_AGENT_CONTROL_ERROR:" in interactions_text
                    else "game_instruction_following")
        agent_counts["total"] += 1
        agent_counts[category] += 1

    return {agent: abort_counts for agent, abort_counts in sorted(counts.items())}


def count_tool_calls(game: str | None) -> dict[str, dict[str, int]]:
    """Count tool calls by model and harness combination.

    Args:
        game: Game name to include, or None to include all games.

    Returns:
        Tool call counts keyed by model and harness combination, then tool name.
    """

    results_dir = Path("results/external-agents")
    counts: dict[str, dict[str, int]] = {}

    for metadata_path in sorted(results_dir.glob("**/agent_trace_meta.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if game is not None and metadata.get("game") != game:
            continue

        agent = metadata["agent"]
        agent_counts = counts.setdefault(agent, {})
        trace_path = metadata_path.with_name("agent_trace.log")
        trace_text = trace_path.read_text(encoding="utf-8", errors="replace")

        if agent.startswith("codex-with-"):
            tool_names = []

            for line in trace_text.splitlines():
                if not line.startswith("{"):
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                item = event.get("item") or {}

                if event.get("type") != "item.started":
                    continue

                if item.get("type") == "mcp_tool_call":
                    server = item.get("server")
                    tool = item.get("tool")
                    tool_name = f"{server}__{tool}" if server and tool else tool or "mcp_tool_call"
                    tool_names.append(tool_name)
                elif item.get("type") == "command_execution":
                    tool_names.append("command_execution")

        elif agent.startswith("claude-code-with-"):
            tool_names = [next(name for name in match if name)
                          for match in re.findall(
                              r"ToolUseBlock\(id=(?:'[^']*'|\"[^\"]*\"), "
                              r"name=(?:'([^']+)'|\"([^\"]+)\")",
                              trace_text,
                          )]

        elif agent.startswith("hermes-with-"):
            tool_names = re.findall(r"^\s*📞 Tool \d+: ([^\s(]+)", trace_text, re.MULTILINE)

        elif agent.startswith("openclaw-with-"):
            tool_names = []

            for line in trace_text.splitlines():
                if not line.startswith("{"):
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = event.get("message") or {}

                if event.get("type") != "message" or message.get("role") != "assistant":
                    continue

                for part in message.get("content") or []:
                    if (isinstance(part, dict) and part.get("type") == "toolCall"
                            and part.get("name")):
                        tool_names.append(part["name"])

        else:
            tool_names = []

        for tool_name in tool_names:
            for prefix in ("mcp__clem-game__", "mcp__clem_game__", "mcp_clem_game_", "clem_game__"):
                if tool_name.startswith(prefix):
                    tool_name = tool_name.removeprefix(prefix)
                    break

            agent_counts[tool_name] = agent_counts.get(tool_name, 0) + 1

    return {agent: dict(sorted(tool_counts.items())) for agent, tool_counts in sorted(counts.items())}


def count_non_mcp_tool_calls(game: str | None) -> dict[str, dict[str, dict[str, int]]]:
    """Count non-MCP tool calls for each model and harness.

    Args:
        game: Game name to include, or None to include all games.

    Returns:
        Tool call counts keyed by model, harness, and original tool name.
    """

    tool_counts = count_tool_calls(game)
    harnesses = ("codex", "claude-code", "hermes", "openclaw")
    mcp_tools = {
        "get_prompt", "list_prompts", "list_resources", "read_resource",
        "start_game", "submit_response",
    }
    counts: dict[str, dict[str, dict[str, int]]] = {}

    for agent, agent_tool_counts in tool_counts.items():
        harness = next((name for name in harnesses if agent.startswith(f"{name}-with-")), None)

        if harness is None:
            continue

        model = agent.removeprefix(f"{harness}-with-")
        counts.setdefault(model, {})[harness] = {
            tool: count for tool, count in agent_tool_counts.items()
            if tool not in mcp_tools
        }

    return {
        model: {
            harness: dict(sorted(model_counts[harness].items()))
            for harness in harnesses if harness in model_counts
        }
        for model, model_counts in sorted(counts.items())
    }


def count_tokens(game: str | None) -> dict[str, int | None]:
    """Count used tokens by model and harness combination.

    Args:
        game: Game name to include, or None to include all games.

    Returns:
        Token counts keyed by model and harness combination. A value of None
        indicates that at least one matching trace lacks token usage data.
    """

    results_dir = Path("results/external-agents")
    counts: dict[str, int | None] = {}
    generation_usage = {}
    openrouter_key = None

    for metadata_path in sorted(results_dir.glob("**/agent_trace_meta.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if game is not None and metadata.get("game") != game:
            continue

        agent = metadata["agent"]
        trace_text = metadata_path.with_name("agent_trace.log").read_text(encoding="utf-8", errors="replace")
        episode_tokens = None

        if agent.startswith("codex-with-"):
            episode_tokens = 0

            for line in trace_text.splitlines():
                if not line.startswith("{"):
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                    episode_tokens += int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))

            if episode_tokens == 0:
                episode_tokens = None

        elif agent.startswith("claude-code-with-"):
            message_usage = {}

            for line in trace_text.splitlines():
                usage_match = re.search(
                    r"usage=\{'input_tokens': (\d+), 'output_tokens': (\d+), "
                    r"'cache_creation_input_tokens': (None|\d+), 'cache_read_input_tokens': (None|\d+).*?"
                    r"message_id='([^']+)'",
                    line,
                )

                if usage_match:
                    values = [0 if value == "None" else int(value) for value in usage_match.groups()[:4]]
                    message_usage[usage_match.group(5)] = sum(values)

            if any(message_usage.values()):
                episode_tokens = sum(message_usage.values())
            else:
                generation_ids = sorted(set(re.findall(r"message_id='(gen-[^']+)'", trace_text)))

                if generation_ids:
                    if openrouter_key is None:
                        openrouter_key = KeyRegistry.from_json().get_key_for("openrouter")

                    missing_ids = [generation_id for generation_id in generation_ids
                                   if generation_id not in generation_usage]

                    if missing_ids:
                        request_generation = partial(
                            requests.get,
                            f"{openrouter_key.get('base_url', 'https://openrouter.ai/api/v1').rstrip('/')}/generation",
                            headers={"Authorization": f"Bearer {openrouter_key['api_key']}"},
                            timeout=30,
                        )

                        with ThreadPoolExecutor(max_workers=min(8, len(missing_ids))) as executor:
                            responses = executor.map(request_generation, ({"id": generation_id}
                                                                          for generation_id in missing_ids))

                            for generation_id, response in zip(missing_ids, responses):
                                response.raise_for_status()
                                usage = response.json()["data"]
                                prompt_tokens = usage.get("native_tokens_prompt")
                                completion_tokens = usage.get("native_tokens_completion")

                                if prompt_tokens is None or completion_tokens is None:
                                    prompt_tokens = usage.get("tokens_prompt")
                                    completion_tokens = usage.get("tokens_completion")

                                generation_usage[generation_id] = int(prompt_tokens) + int(completion_tokens)

                    episode_tokens = sum(generation_usage[generation_id] for generation_id in generation_ids)

        elif agent.startswith("hermes-with-"):
            totals = re.findall(
                r"Token usage: prompt=[\d,]+, completion=[\d,]+, total=([\d,]+)",
                trace_text,
            )

            if totals:
                episode_tokens = sum(int(total.replace(",", "")) for total in totals)

        elif agent.startswith("openclaw-with-"):
            for line in trace_text.splitlines():
                if not line.startswith("{"):
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") != "trace.artifacts":
                    continue

                usage = (event.get("data") or {}).get("usage") or {}

                if usage.get("total") is not None:
                    episode_tokens = int(usage["total"])

        if agent not in counts:
            counts[agent] = episode_tokens
        elif counts[agent] is None or episode_tokens is None:
            counts[agent] = None
        else:
            counts[agent] += episode_tokens

    return dict(sorted(counts.items()))
