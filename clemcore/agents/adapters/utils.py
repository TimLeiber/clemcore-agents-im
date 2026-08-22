import ast
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence


MODEL_CONNECTION_ENV = "CLEM_AGENT_MODEL_CONNECTION_PATH"
MCP_EPISODE_ENV = (
    "CLEM_EXPERIMENT_NAME",
    "CLEM_GAME_ID",
    "CLEM_GAME_COMPLETION_PATH",
    "CLEM_GAME_STARTED_PATH",
    "CLEM_OPENENV_SESSION_PATH",
)


def new_game_completion_path() -> Path:
    """Return a unique, initially absent bridge-to-harness completion marker."""
    path = Path("/tmp") / f"clem-game-completed-{os.getpid()}-{time.time_ns()}.json"
    path.unlink(missing_ok=True)
    return path


def read_game_completion(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    return value if isinstance(value, dict) else None


def run_process_until_game_complete(
    command: Sequence[str],
    *,
    completion_path: Path,
    input_text: str | None = None,
    cwd: str | Path | None = None,
    timeout: float | None = 600,
    completion_grace: float = 1.0,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    """Run a CLI and terminate it shortly after the bridge reports done=true."""
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )
    started_at = time.monotonic()
    completion_seen_at: float | None = None
    first_communicate = True
    last_timeout: subprocess.TimeoutExpired | None = None

    while True:
        try:
            stdout, stderr = process.communicate(
                input=input_text if first_communicate else None,
                timeout=0.1,
            )
            break
        except subprocess.TimeoutExpired as error:
            first_communicate = False
            last_timeout = error
            now = time.monotonic()

            if completion_path.exists():
                completion_seen_at = completion_seen_at or now

                if now - completion_seen_at >= completion_grace:
                    process.terminate()

                    try:
                        stdout, stderr = process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()

                    return (
                        subprocess.CompletedProcess(
                            list(command), process.returncode, stdout, stderr
                        ),
                        True,
                    )

            if timeout is not None and now - started_at >= timeout:
                process.kill()
                stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(
                    list(command),
                    timeout,
                    output=stdout or (last_timeout.output if last_timeout else None),
                    stderr=stderr or (last_timeout.stderr if last_timeout else None),
                )

    return (
        subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr),
        False,
    )


def load_model_connection(harness: str,
                          model_connection_path: str | Path | None = None) -> dict[str, Any] | None:
    """Load and validate the resolved model connection for one harness.

    Args:
        harness: name of the harness expected in the connection file
        model_connection_path: optional path to the connection file

    Returns:
        the resolved model connection, or None when no file was configured
    """

    connection_path = model_connection_path or os.environ.get(MODEL_CONNECTION_ENV)

    if not connection_path:
        return None

    path = Path(connection_path)

    if not path.exists():
        raise FileNotFoundError(f"Missing model connection file: {path}")

    connection = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(connection, dict):
        raise ValueError(f"Model connection must be a mapping: {path}")

    if connection.get("harness") != harness:
        raise ValueError(f"Model connection is not for {harness}.")

    return connection


def resolve_runtime_model(model_connection: dict[str, Any] | None,
                          model: str | None,
                          harness_name: str,
                          required: bool = True) -> str | None:
    """Select the model identifier used by an external-agent harness.

    Args:
        model_connection: resolved model connection supplied by the pipeline
        model: fallback model configured directly for the harness
        harness_name: harness class name used in validation errors
        required: whether the harness requires a model identifier

    Returns:
        the resolved or directly configured model identifier
    """

    runtime_model = model_connection.get("model") if model_connection else model

    if runtime_model is None:
        if required:
            raise RuntimeError(f"{harness_name} requires either model or clem_model.")

        return None

    return str(runtime_model)


def model_connection_environment(model_connection: dict[str, Any] | None) -> dict[str, str | None]:
    """Convert model-connection environment values into process values.

    Args:
        model_connection: resolved model connection supplied by the pipeline

    Returns:
        environment values ready to apply to a harness process
    """

    if not model_connection:
        return {}

    environment = model_connection.get("env", {})

    if not isinstance(environment, dict):
        raise ValueError("Model connection env must be a mapping.")

    return {key: str(value) if value is not None else None for key, value in environment.items()}


def mcp_environment(mcp_url: str,
                    include_pythonpath: bool = False) -> dict[str, str]:
    """Build the environment passed to the container-side MCP bridge.

    Args:
        mcp_url: URL of the host-side MCP server
        include_pythonpath: whether to include the clemcore import path

    Returns:
        environment variables required by the MCP bridge
    """

    environment = {"OPENENV_MCP_URL": mcp_url}

    if include_pythonpath:
        environment["PYTHONPATH"] = os.environ.get("PYTHONPATH", "/opt/clemcore")

    for name in MCP_EPISODE_ENV:
        value = os.environ.get(name)

        if value is not None:
            environment[name] = value

    return environment


@contextmanager
def temporary_environment(environment: dict[str, str | None]) -> Iterator[None]:
    """Apply environment variables temporarily and restore prior values.

    Args:
        environment: values to set or remove while the context is active

    Yields:
        control while the temporary environment is active
    """

    sentinel = object()
    previous_values: dict[str, str | object] = {}

    for name, value in environment.items():
        previous_values[name] = os.environ.get(name, sentinel)

        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    try:
        yield
    finally:
        for name, previous_value in previous_values.items():
            if previous_value is sentinel:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(previous_value)


def redact_sensitive(text: str) -> str:
    """Replace API credentials in captured harness output.

    Args:
        text: captured output that may contain credentials

    Returns:
        captured output with known credential formats redacted
    """

    redacted = re.sub(r"sk-or-[A-Za-z0-9._-]+",
                      "[REDACTED]",
                      text)
    redacted = re.sub(r"sk-[A-Za-z0-9._-]+",
                      "[REDACTED]",
                      redacted)
    prefix_patterns = (
        r"(?i)(api key:\s*)\S+",
        r"(?i)(OPENROUTER_API_KEY=)\S+",
        r"(?i)(OPENAI_API_KEY=)\S+",
        r"(?i)(ANTHROPIC_API_KEY=)\S+",
        r"(?i)(--token\s+)\S+",
        r"(?i)(--openrouter-api-key\s+)\S+",
    )

    for pattern in prefix_patterns:
        redacted = re.sub(pattern,
                          lambda match: f"{match.group(1)}[REDACTED]",
                          redacted)

    return redacted


def deep_merge_dicts(base: dict[str, Any],
                     patch: dict[str, Any]) -> dict[str, Any]:
    """Merge a nested configuration patch without modifying either input.

    Args:
        base: base configuration
        patch: values that override the base configuration

    Returns:
        a recursively merged configuration
    """

    merged = dict(base)

    for key, value in patch.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


def write_text_artifact(output_dir: str | Path | None,
                        filename: str,
                        content: str) -> Path | None:
    """Write one text artifact when adapter output is enabled.

    Args:
        output_dir: output directory, or None to disable artifact writing
        filename: name of the artifact file
        content: text written to the artifact

    Returns:
        path to the written artifact, or None when output is disabled
    """

    if output_dir is None:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / filename
    artifact_path.write_text(content, encoding="utf-8")

    return artifact_path


def parse_codex_agent_trace(episode_dir: Path,
                            metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse one Codex trace into the common agent-loop schema."""

    trace_path = episode_dir / "agent_trace.log"

    if not trace_path.exists():
        return {}

    trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
    records = _codex_trace_records(trace_text)
    events = []
    seen_input_items = set()
    seen_output_item_ids = set()
    model_request_count = 0
    model_response_count = 0
    has_agent_loop_instruction = False
    has_native_instruction = False

    for record in records:
        record_type = record["type"]

        if record_type == "agent_loop_instruction":
            has_agent_loop_instruction = True
            events.append({
                "type": "instruction",
                "kind": "agent_loop",
                "source": "container",
                "content": record["content"]
            })
            continue

        if record_type == "model_request":
            model_request_count += 1
            request = record["payload"]
            turn = model_request_count
            events.append({
                "type": "model_request",
                "source": "wire_request",
                "turn": turn,
                "path": record["path"],
                "raw": record["raw"],
                "payload": request
            })

            if record["parse_error"] is not None:
                events.append({
                    "type": "error",
                    "source": "wire_request",
                    "turn": turn,
                    "content": f"Could not parse request JSON: {record['parse_error']}"
                })

            native_instruction = request.get("instructions")

            if native_instruction and not has_native_instruction:
                has_native_instruction = True
                events.append({
                    "type": "instruction",
                    "kind": "native_harness",
                    "source": "wire_request",
                    "turn": turn,
                    "content": native_instruction
                })

            if request.get("tools") and turn == 1:
                events.append({
                    "type": "tool_definitions",
                    "source": "wire_request",
                    "turn": turn,
                    "tools": request["tools"]
                })

            request_events, found_agent_loop_instruction = _codex_request_events(
                request=request,
                turn=turn,
                seen_input_items=seen_input_items,
                seen_output_item_ids=seen_output_item_ids
            )
            has_agent_loop_instruction = (
                has_agent_loop_instruction or found_agent_loop_instruction
            )
            events.extend(request_events)
            continue

        if record_type == "model_response":
            model_response_count += 1
            turn = model_response_count
            events.append({
                "type": "model_response",
                "source": "wire_response",
                "turn": turn,
                "path": record["path"],
                "status": record["status"],
                "content_type": record["content_type"],
                "content_encoding": record["content_encoding"],
                "raw": record["raw"]
            })
            events.extend(_codex_response_events(record=record,
                                                 turn=turn,
                                                 seen_output_item_ids=seen_output_item_ids))

            if record["status"] >= 400:
                events.append({
                    "type": "error",
                    "source": "wire_response",
                    "turn": turn,
                    "content": f"Provider response returned HTTP {record['status']}"
                })

    events.extend(_codex_runtime_errors(trace_text))

    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence

    return {
        "schema_version": 1,
        "backend": "codex",
        "capture": {
            "agent_loop_instruction": {
                "status": "complete" if has_agent_loop_instruction else "unavailable",
                "source": "container" if "agent_loop_instruction_start" in trace_text else "wire_request"
            },
            "native_harness_instruction": {
                "status": "complete" if has_native_instruction else "unavailable",
                "source": "wire_request"
            },
            "model_requests": {
                "status": "complete" if model_request_count else "unavailable",
                "source": "raw_upstream_request",
                "count": model_request_count
            },
            "model_responses": {
                "status": "complete" if model_response_count else "unavailable",
                "source": "raw_upstream_response",
                "count": model_response_count
            }
        },
        "events": events
    }


def parse_claude_code_agent_trace(episode_dir: Path,
                                  metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse one Claude Code SDK trace into the common agent-loop schema."""

    trace_path = episode_dir / "agent_trace.log"

    if not trace_path.exists():
        return {}

    if metadata is None:
        metadata_path = episode_dir / "agent_trace_meta.json"

        if metadata_path.exists():
            try:
                loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = loaded_metadata if isinstance(loaded_metadata, dict) else None
            except (json.JSONDecodeError, OSError):
                metadata = None

    trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
    records = _claude_code_trace_records(trace_text)
    events = []
    agent_loop_instruction = _marked_trace_section(
        trace_text,
        "agent_loop_instruction_start",
        "agent_loop_instruction_end",
    )

    if agent_loop_instruction is not None:
        events.append({
            "type": "instruction",
            "kind": "agent_loop",
            "source": "container",
            "content": agent_loop_instruction
        })

    message_ids_with_tools = {
        record.get("message_id")
        for record in records
        if record.get("__type__") == "AssistantMessage"
        and record.get("message_id") is not None
        and any(block.get("__type__") == "ToolUseBlock"
                for block in record.get("content", [])
                if isinstance(block, dict))
    }
    message_turns = {}
    call_turns = {}
    runtime = {}
    result = {}
    thinking_token_updates = 0
    thinking_token_total = 0
    previous_thinking_tokens = 0
    parse_errors = 0

    for record in records:
        record_type = record.get("__type__")

        if record_type == "parse_error":
            parse_errors += 1
            events.append({
                "type": "error",
                "source": "sdk_trace",
                "content": record.get("content", "Could not parse SDK trace line")
            })
            continue

        if record_type == "SystemMessage":
            subtype = record.get("subtype")
            data = record.get("data", {})

            if subtype == "init" and isinstance(data, dict):
                runtime = {
                    key: data.get(key)
                    for key in (
                        "cwd",
                        "session_id",
                        "model",
                        "permissionMode",
                        "mcp_servers",
                        "claude_code_version",
                        "output_style",
                        "agents",
                        "skills",
                        "plugins",
                        "capabilities",
                    )
                    if data.get(key) is not None
                }
                tools = data.get("tools", [])

                if tools:
                    events.append({
                        "type": "tool_definitions",
                        "kind": "sdk_inventory",
                        "source": "sdk_trace",
                        "tools": [{"name": name} for name in tools]
                    })

            elif subtype == "thinking_tokens" and isinstance(data, dict):
                estimated_tokens = data.get("estimated_tokens")

                if isinstance(estimated_tokens, int):
                    thinking_token_updates += 1

                    if estimated_tokens < previous_thinking_tokens:
                        thinking_token_total += previous_thinking_tokens

                    previous_thinking_tokens = estimated_tokens

            continue

        if record_type == "AssistantMessage":
            message_id = record.get("message_id")

            if message_id not in message_turns:
                message_turns[message_id] = len(message_turns) + 1

            turn = message_turns[message_id]

            for block in record.get("content", []):
                if not isinstance(block, dict):
                    continue

                block_type = block.get("__type__")

                if block_type == "ThinkingBlock":
                    events.append({
                        "type": "reasoning",
                        "source": "sdk_trace",
                        "turn": turn,
                        "content": block.get("thinking", ""),
                        "payload": block
                    })
                elif block_type == "ToolUseBlock":
                    call_id = block.get("id")
                    call_turns[call_id] = turn
                    events.append({
                        "type": "tool_call",
                        "source": "sdk_trace",
                        "turn": turn,
                        "call_id": call_id,
                        "name": block.get("name"),
                        "arguments": block.get("input", {}),
                        "payload": block
                    })
                elif block_type == "TextBlock":
                    events.append({
                        "type": (
                            "tool_preamble"
                            if message_id in message_ids_with_tools
                            else "assistant_text"
                        ),
                        "source": "sdk_trace",
                        "turn": turn,
                        "content": block.get("text", ""),
                        "payload": block
                    })

            continue

        if record_type == "UserMessage":
            for block in record.get("content", []):
                if not isinstance(block, dict):
                    continue

                block_type = block.get("__type__")

                if block_type == "ToolResultBlock":
                    call_id = block.get("tool_use_id")
                    events.append({
                        "type": "tool_result",
                        "source": "sdk_trace",
                        "turn": call_turns.get(call_id),
                        "call_id": call_id,
                        "content": block.get("content", ""),
                        "payload": block
                    })
                elif block_type == "TextBlock":
                    events.append({
                        "type": "message",
                        "source": "sdk_trace",
                        "role": "user",
                        "content": block.get("text", ""),
                        "payload": block
                    })

            continue

        if record_type == "TaskNotificationMessage":
            events.append({
                "type": "tool_result",
                "kind": "background_task",
                "source": "sdk_trace",
                "call_id": record.get("tool_use_id"),
                "content": {
                    "status": record.get("status"),
                    "summary": record.get("summary"),
                    "output_file": record.get("output_file")
                },
                "payload": record
            })
            continue

        if record_type == "ResultMessage":
            result = {
                key: record.get(key)
                for key in (
                    "subtype",
                    "duration_ms",
                    "duration_api_ms",
                    "is_error",
                    "num_turns",
                    "session_id",
                    "stop_reason",
                    "total_cost_usd",
                    "usage",
                    "model_usage",
                    "permission_denials",
                    "errors",
                    "api_error_status",
                    "terminal_reason",
                )
                if record.get(key) is not None
            }

    thinking_token_total += previous_thinking_tokens

    for line in trace_text.splitlines():
        if line.startswith("agent_runtime_error:"):
            events.append({
                "type": "error",
                "source": "runtime",
                "content": line.removeprefix("agent_runtime_error:").strip()
            })

    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence

    return {
        "schema_version": 1,
        "backend": "claude_code",
        "capture": {
            "agent_loop_instruction": {
                "status": "complete" if agent_loop_instruction is not None else "unavailable",
                "source": "container" if agent_loop_instruction is not None else None
            },
            "native_harness_instruction": {
                "status": "unavailable",
                "source": "not_exposed_by_claude_agent_sdk"
            },
            "sdk_messages": {
                "status": "complete" if records else "unavailable",
                "source": "claude_agent_sdk",
                "count": len(records),
                "parse_errors": parse_errors
            },
            "reasoning": {
                "status": (
                    "complete"
                    if any(event.get("type") == "reasoning" for event in events)
                    else "unavailable"
                ),
                "source": "ThinkingBlock"
            },
            "thinking_tokens": {
                "status": "estimated" if thinking_token_updates else "unavailable",
                "estimated_total": thinking_token_total,
                "updates": thinking_token_updates
            }
        },
        "runtime": runtime,
        "result": result,
        "metadata": metadata or {},
        "events": events
    }


def parse_hermes_agent_trace(episode_dir: Path,
                             metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse one Hermes session into the common agent-loop schema."""

    trace_path = episode_dir / "agent_trace.log"

    if not trace_path.exists():
        return {}

    if metadata is None:
        metadata_path = episode_dir / "agent_trace_meta.json"

        if metadata_path.exists():
            try:
                loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = loaded_metadata if isinstance(loaded_metadata, dict) else None
            except (json.JSONDecodeError, OSError):
                metadata = None

    trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
    session, session_parse_errors = _hermes_session_export(episode_dir)
    wire_records = [record for record in _codex_trace_records(trace_text)
                    if record["type"] in {"model_request", "model_response"}]
    native_instructions, wire_tools = _hermes_wire_configuration(wire_records)
    events = []

    for instruction in native_instructions:
        events.append({
            "type": "instruction",
            "kind": "native_harness",
            "source": "wire_request",
            "content": instruction
        })

    agent_loop_instruction = _marked_trace_section(
        trace_text,
        "agent_loop_instruction_start",
        "agent_loop_instruction_end",
    )

    if agent_loop_instruction is None:
        agent_loop_instruction = _hermes_query_instruction(trace_text)

    if agent_loop_instruction is not None:
        events.append({
            "type": "instruction",
            "kind": "agent_loop",
            "source": "container",
            "content": agent_loop_instruction
        })

    cli_tools = _hermes_tool_inventory(trace_text)
    tools = wire_tools or cli_tools

    if tools:
        events.append({
            "type": "tool_definitions",
            "kind": "wire_schema" if wire_tools else "cli_inventory",
            "source": "wire_request" if wire_tools else "hermes_cli",
            "tools": tools
        })

    runtime = _hermes_runtime(trace_text, session)
    result = _hermes_result(trace_text, session)
    native_instruction_found = bool(native_instructions)

    if session is not None:
        session_events, session_has_native_instruction = _hermes_session_events(
            session=session,
            agent_loop_instruction=agent_loop_instruction,
        )
        events.extend(session_events)
        native_instruction_found = (
            native_instruction_found or session_has_native_instruction
        )
        semantic_source = "hermes_session_export"
    else:
        events.extend(_hermes_cli_events(trace_text))
        semantic_source = "hermes_cli"

    model_request_count = 0
    model_response_count = 0

    for record in wire_records:
        if record["type"] == "model_request":
            model_request_count += 1
            events.append({
                "type": "model_request",
                "source": "wire_request",
                "turn": model_request_count,
                "path": record["path"],
                "raw": record["raw"],
                "payload": record["payload"]
            })
        else:
            model_response_count += 1
            events.append({
                "type": "model_response",
                "source": "wire_response",
                "turn": model_response_count,
                "path": record["path"],
                "status": record["status"],
                "content_type": record["content_type"],
                "content_encoding": record["content_encoding"],
                "raw": record["raw"]
            })

    for parse_error in session_parse_errors:
        events.append({
            "type": "error",
            "source": "hermes_session_export",
            "content": parse_error
        })

    for match in re.finditer(r"agent_runtime_error:\s*(?P<content>.*)", trace_text):
        events.append({
            "type": "error",
            "source": "runtime",
            "content": match.group("content").strip()
        })

    deduplicated_events = []
    seen_instructions = set()

    for event in events:
        if event.get("type") == "instruction":
            instruction_key = (event.get("kind"), event.get("content"))

            if instruction_key in seen_instructions:
                continue

            seen_instructions.add(instruction_key)

        deduplicated_events.append(event)

    events = deduplicated_events

    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence

    reasoning_count = sum(event.get("type") == "reasoning" for event in events)
    tool_call_count = sum(event.get("type") == "tool_call" for event in events)
    tool_result_count = sum(event.get("type") == "tool_result" for event in events)

    return {
        "schema_version": 1,
        "backend": "hermes",
        "capture": {
            "agent_loop_instruction": {
                "status": "complete" if agent_loop_instruction is not None else "unavailable",
                "source": "container" if agent_loop_instruction is not None else None
            },
            "native_harness_instruction": {
                "status": "complete" if native_instruction_found else "unavailable",
                "source": "wire_request" if native_instruction_found else "not_exposed"
            },
            "session_export": {
                "status": "complete" if session is not None else "unavailable",
                "source": "hermes_sessions_export",
                "parse_errors": len(session_parse_errors)
            },
            "semantic_events": {
                "status": "complete" if reasoning_count or tool_call_count else "unavailable",
                "source": semantic_source,
                "reasoning": reasoning_count,
                "tool_calls": tool_call_count,
                "tool_results": tool_result_count
            },
            "tool_definitions": {
                "status": "complete" if wire_tools else ("partial" if cli_tools else "unavailable"),
                "source": "wire_request" if wire_tools else ("hermes_cli" if cli_tools else None),
                "count": len(tools)
            },
            "model_requests": {
                "status": "complete" if model_request_count else "unavailable",
                "source": "raw_upstream_request",
                "count": model_request_count
            },
            "model_responses": {
                "status": "complete" if model_response_count else "unavailable",
                "source": "raw_upstream_response",
                "count": model_response_count
            }
        },
        "runtime": runtime,
        "result": result,
        "metadata": metadata or {},
        "events": events
    }


def parse_openclaw_agent_trace(episode_dir: Path,
                               metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse one OpenClaw session into the common agent-loop schema."""

    trace_path = episode_dir / "agent_trace.log"

    if not trace_path.exists():
        return {}

    if metadata is None:
        metadata_path = episode_dir / "agent_trace_meta.json"

        if metadata_path.exists():
            try:
                loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = loaded_metadata if isinstance(loaded_metadata, dict) else None
            except (json.JSONDecodeError, OSError):
                metadata = None

    trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
    session, trajectory, session_parse_errors = _openclaw_trace_exports(trace_text)
    stdout_payload, stdout_parse_error = _openclaw_stdout_payload(trace_text)
    wire_records = [record for record in _codex_trace_records(trace_text)
                    if record["type"] in {"model_request", "model_response"}]
    wire_instructions, wire_tools = _openclaw_wire_configuration(wire_records)
    trajectory_instruction, trajectory_tools = _openclaw_trajectory_configuration(
        trajectory
    )
    events = []
    native_instructions = wire_instructions

    if not native_instructions and trajectory_instruction is not None:
        native_instructions = [trajectory_instruction]

    for instruction in native_instructions:
        events.append({
            "type": "instruction",
            "kind": "native_harness",
            "source": "wire_request" if wire_instructions else "openclaw_trajectory",
            "content": instruction
        })

    agent_loop_instruction = _marked_trace_section(
        trace_text,
        "agent_loop_instruction_start",
        "agent_loop_instruction_end",
    )

    if agent_loop_instruction is None:
        agent_loop_instruction = _openclaw_agent_loop_instruction(session, trajectory)

    if agent_loop_instruction is not None:
        events.append({
            "type": "instruction",
            "kind": "agent_loop",
            "source": "container" if "agent_loop_instruction_start" in trace_text else "openclaw_session",
            "content": agent_loop_instruction
        })

    tools = wire_tools or trajectory_tools

    if tools:
        events.append({
            "type": "tool_definitions",
            "kind": "wire_schema" if wire_tools else "trajectory_inventory",
            "source": "wire_request" if wire_tools else "openclaw_trajectory",
            "tools": tools
        })

    events.extend(_openclaw_session_events(
        session=session,
        agent_loop_instruction=agent_loop_instruction,
    ))

    model_request_count = 0
    model_response_count = 0

    for record in wire_records:
        if record["type"] == "model_request":
            model_request_count += 1
            events.append({
                "type": "model_request",
                "source": "wire_request",
                "turn": model_request_count,
                "path": record["path"],
                "raw": record["raw"],
                "payload": record["payload"]
            })

            if record["parse_error"] is not None:
                events.append({
                    "type": "error",
                    "source": "wire_request",
                    "turn": model_request_count,
                    "content": f"Could not parse request JSON: {record['parse_error']}"
                })
        else:
            model_response_count += 1
            events.append({
                "type": "model_response",
                "source": "wire_response",
                "turn": model_response_count,
                "path": record["path"],
                "status": record["status"],
                "content_type": record["content_type"],
                "content_encoding": record["content_encoding"],
                "raw": record["raw"]
            })

            if record["status"] >= 400:
                events.append({
                    "type": "error",
                    "source": "wire_response",
                    "turn": model_response_count,
                    "content": f"Provider response returned HTTP {record['status']}"
                })

    for parse_error in session_parse_errors:
        events.append({
            "type": "error",
            "source": "openclaw_session",
            "content": parse_error
        })

    if stdout_parse_error is not None:
        events.append({
            "type": "error",
            "source": "openclaw_stdout",
            "content": stdout_parse_error
        })

    for match in re.finditer(r"agent_runtime_error:\s*(?P<content>.*)", trace_text):
        events.append({
            "type": "error",
            "source": "runtime",
            "content": match.group("content").strip()
        })

    deduplicated_events = []
    seen_instructions = set()

    for event in events:
        if event.get("type") == "instruction":
            instruction_key = (event.get("kind"), event.get("content"))

            if instruction_key in seen_instructions:
                continue

            seen_instructions.add(instruction_key)

        deduplicated_events.append(event)

    events = deduplicated_events

    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence

    reasoning_count = sum(event.get("type") == "reasoning" for event in events)
    tool_call_count = sum(event.get("type") == "tool_call" for event in events)
    tool_result_count = sum(event.get("type") == "tool_result" for event in events)
    system_prompt_report = _openclaw_system_prompt_report(stdout_payload)

    return {
        "schema_version": 1,
        "backend": "openclaw",
        "capture": {
            "agent_loop_instruction": {
                "status": "complete" if agent_loop_instruction is not None else "unavailable",
                "source": "container" if "agent_loop_instruction_start" in trace_text else "openclaw_session"
            },
            "native_harness_instruction": {
                "status": "complete" if native_instructions else "unavailable",
                "source": (
                    "wire_request"
                    if wire_instructions
                    else ("openclaw_trajectory" if trajectory_instruction is not None else "not_exposed")
                ),
                **system_prompt_report
            },
            "session_export": {
                "status": "complete" if session else "unavailable",
                "source": "openclaw_session_jsonl",
                "parse_errors": len(session_parse_errors)
            },
            "trajectory_export": {
                "status": "complete" if trajectory else "unavailable",
                "source": "openclaw_trajectory_jsonl"
            },
            "semantic_events": {
                "status": "complete" if session else "unavailable",
                "source": "openclaw_session_jsonl",
                "reasoning": reasoning_count,
                "tool_calls": tool_call_count,
                "tool_results": tool_result_count
            },
            "tool_definitions": {
                "status": "complete" if tools else "unavailable",
                "source": "wire_request" if wire_tools else ("openclaw_trajectory" if trajectory_tools else None),
                "count": len(tools)
            },
            "model_requests": {
                "status": "complete" if model_request_count else "unavailable",
                "source": "raw_upstream_request",
                "count": model_request_count
            },
            "model_responses": {
                "status": "complete" if model_response_count else "unavailable",
                "source": "raw_upstream_response",
                "count": model_response_count
            }
        },
        "runtime": _openclaw_runtime(session, trajectory, stdout_payload),
        "result": _openclaw_result(session, trajectory, stdout_payload, metadata),
        "metadata": metadata or {},
        "events": events
    }


def _claude_code_trace_records(trace_text: str) -> list[dict[str, Any]]:
    """Safely parse Claude Agent SDK constructor representations."""

    records = []
    supported_types = {
        "AssistantMessage",
        "ResultMessage",
        "SystemMessage",
        "TaskNotificationMessage",
        "TaskStartedMessage",
        "UserMessage",
    }

    for line_number, line in enumerate(trace_text.splitlines(), start=1):
        constructor_name = line.split("(", 1)[0]

        if constructor_name not in supported_types:
            continue

        try:
            expression = ast.parse(line, mode="eval").body
            record = _constructor_ast_value(expression)
        except (SyntaxError, ValueError) as error:
            records.append({
                "__type__": "parse_error",
                "line": line_number,
                "content": f"Could not parse {constructor_name} on line {line_number}: {error}"
            })
            continue

        if isinstance(record, dict):
            records.append(record)

    return records


def _constructor_ast_value(node: ast.AST) -> Any:
    """Convert a constructor repr AST into JSON-serializable values."""

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.List):
        return [_constructor_ast_value(item) for item in node.elts]

    if isinstance(node, ast.Tuple):
        return [_constructor_ast_value(item) for item in node.elts]

    if isinstance(node, ast.Dict):
        return {
            _constructor_ast_value(key): _constructor_ast_value(value)
            for key, value in zip(node.keys, node.values)
        }

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_constructor_ast_value(node.operand)

    if isinstance(node, ast.Name) and node.id in {"None", "True", "False"}:
        return {"None": None, "True": True, "False": False}[node.id]

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        value = {"__type__": node.func.id}

        if node.args:
            value["__args__"] = [_constructor_ast_value(item) for item in node.args]

        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("Constructor repr contains unsupported keyword expansion")

            value[keyword.arg] = _constructor_ast_value(keyword.value)

        return value

    raise ValueError(f"Unsupported trace expression: {type(node).__name__}")


def _marked_trace_section(trace_text: str,
                          start_marker: str,
                          end_marker: str) -> str | None:
    """Return the first text section enclosed by exact trace markers."""

    pattern = re.compile(
        rf"{re.escape(start_marker)}\n(?P<content>.*?)\n{re.escape(end_marker)}",
        re.DOTALL,
    )
    match = pattern.search(trace_text)
    return match.group("content") if match is not None else None


def _openclaw_trace_exports(
        trace_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Recover OpenClaw session and trajectory JSONL embedded in a trace."""

    blocks = []
    errors = []
    current_path = None
    current_records = []

    def finish_block() -> None:
        nonlocal current_path, current_records

        if current_path is not None:
            blocks.append((current_path, current_records))

        current_path = None
        current_records = []

    for line_number, line in enumerate(trace_text.splitlines(), start=1):
        if line.startswith("openclaw_session: "):
            finish_block()
            current_path = line.removeprefix("openclaw_session: ").strip()
            continue

        if current_path is None or not line.strip():
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            if line.lstrip().startswith("{"):
                errors.append(
                    f"Could not parse OpenClaw JSONL line {line_number}: {error}"
                )

            finish_block()
            continue

        if isinstance(value, dict):
            current_records.append(value)
        else:
            errors.append(f"OpenClaw JSONL line {line_number} is not an object")

    finish_block()
    sessions = [records for path, records in blocks
                if not path.endswith(".trajectory.jsonl")]
    trajectories = [records for path, records in blocks
                    if path.endswith(".trajectory.jsonl")]
    session = max(sessions,
                  key=lambda records: sum(record.get("type") == "message"
                                          for record in records),
                  default=[])
    trajectory = max(trajectories, key=len, default=[])
    return session, trajectory, errors


def _openclaw_stdout_payload(
        trace_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Decode OpenClaw's JSON command result from the combined trace."""

    match = re.search(
        r"openclaw_agent_stdout:\n(?P<content>.*?)(?:\nopenclaw_agent_stderr:|\Z)",
        trace_text,
        flags=re.DOTALL,
    )

    if match is None or not match.group("content").strip():
        return None, None

    try:
        payload = json.loads(match.group("content"))
    except json.JSONDecodeError as error:
        return None, f"Could not parse OpenClaw JSON output: {error}"

    if not isinstance(payload, dict):
        return None, "OpenClaw JSON output is not an object"

    return payload, None


def _openclaw_wire_configuration(
        records: list[dict[str, Any]]) -> tuple[list[str], list[Any]]:
    """Extract native instructions and tool schemas from OpenClaw requests."""

    instructions = []
    seen_instructions = set()
    tools = []

    for record in records:
        if record.get("type") != "model_request":
            continue

        request = record.get("payload", {})

        if not isinstance(request, dict):
            continue

        if not tools and isinstance(request.get("tools"), list):
            tools = request["tools"]

        for message in request.get("messages", []):
            if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
                continue

            content = _openclaw_content_text(message.get("content"))

            if content and content not in seen_instructions:
                seen_instructions.add(content)
                instructions.append(content)

    return instructions, tools


def _openclaw_trajectory_configuration(
        trajectory: list[dict[str, Any]]) -> tuple[str | None, list[Any]]:
    """Recover prompt and tool configuration exposed by OpenClaw trajectory data."""

    for record in trajectory:
        if record.get("type") != "context.compiled":
            continue

        data = record.get("data", {})

        if not isinstance(data, dict):
            return None, []

        system_prompt = data.get("systemPrompt")
        tools = data.get("tools") if isinstance(data.get("tools"), list) else []
        return system_prompt if isinstance(system_prompt, str) else None, tools

    return None, []


def _openclaw_agent_loop_instruction(session: list[dict[str, Any]],
                                     trajectory: list[dict[str, Any]]) -> str | None:
    """Recover the agent-loop instruction from native OpenClaw artifacts."""

    for record in trajectory:
        if record.get("type") not in {"prompt.submitted", "context.compiled"}:
            continue

        data = record.get("data", {})
        prompt = data.get("prompt") if isinstance(data, dict) else None

        if isinstance(prompt, str) and prompt:
            return prompt

    for record in session:
        if record.get("type") != "message":
            continue

        message = record.get("message", {})

        if not isinstance(message, dict) or message.get("role") != "user":
            continue

        content = _openclaw_content_text(message.get("content"))

        if content:
            return content

    return None


def _openclaw_session_events(session: list[dict[str, Any]],
                             agent_loop_instruction: str | None) -> list[dict[str, Any]]:
    """Convert OpenClaw session messages into ordered semantic events."""

    events = []
    call_turns = {}
    current_turn = 0
    synthetic_call_number = 0
    game_completed = False
    pending_prompt_error = None

    def append_prompt_error(record: dict[str, Any]) -> None:
        error = record.get("data", {})

        if game_completed:
            events.append({
                "type": "termination",
                "kind": "after_game_completion",
                "source": "openclaw_session",
                "content": (
                    "OpenClaw stopped after clem_game reported done=true. "
                    f"Native termination detail: {error.get('error', error)}"
                ),
                "payload": record
            })
        else:
            events.append({
                "type": "error",
                "source": "openclaw_session",
                "content": error,
                "payload": record
            })

    for record in session:
        if record.get("type") == "custom" and record.get("customType") == "openclaw:prompt-error":
            pending_prompt_error = record
            continue

        if record.get("type") != "message":
            continue

        message = record.get("message", {})

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        blocks = message.get("content", [])
        blocks = blocks if isinstance(blocks, list) else []

        if role == "assistant":
            current_turn += 1
            has_tool_call = any(
                isinstance(block, dict) and block.get("type") == "toolCall"
                for block in blocks
            )

            for block in blocks:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")

                if block_type == "thinking":
                    content = block.get("thinking") or block.get("text") or ""

                    if content:
                        events.append({
                            "type": "reasoning",
                            "source": "openclaw_session",
                            "turn": current_turn,
                            "content": content,
                            "payload": block
                        })
                elif block_type == "text":
                    content = block.get("text", "")

                    if content.strip():
                        events.append({
                            "type": "tool_preamble" if has_tool_call else "assistant_text",
                            "source": "openclaw_session",
                            "turn": current_turn,
                            "content": content,
                            "payload": block
                        })
                elif block_type == "toolCall":
                    synthetic_call_number += 1
                    call_id = block.get("id") or f"openclaw-call-{synthetic_call_number}"
                    arguments = block.get("arguments")

                    if arguments is None:
                        arguments = _openclaw_json_value(block.get("partialArgs", {}))

                    call_turns[call_id] = current_turn
                    events.append({
                        "type": "tool_call",
                        "source": "openclaw_session",
                        "turn": current_turn,
                        "call_id": call_id,
                        "name": block.get("name"),
                        "arguments": arguments,
                        "payload": block
                    })

            if pending_prompt_error is not None:
                append_prompt_error(pending_prompt_error)
                pending_prompt_error = None

            continue

        content = _openclaw_content_text(message.get("content"))

        if role == "toolResult":
            call_id = message.get("toolCallId")
            events.append({
                "type": "tool_result",
                "source": "openclaw_session",
                "turn": call_turns.get(call_id),
                "call_id": call_id,
                "name": message.get("toolName"),
                "content": content or message.get("details", {}),
                "payload": message
            })
            game_completed = game_completed or _openclaw_tool_result_done(message)
            continue

        if role == "user" and content == agent_loop_instruction:
            continue

        if content:
            events.append({
                "type": "message",
                "source": "openclaw_session",
                "role": role,
                "content": content,
                "payload": message
            })

    if pending_prompt_error is not None:
        append_prompt_error(pending_prompt_error)

    return events


def _openclaw_tool_result_done(message: dict[str, Any]) -> bool:
    """Return whether an OpenClaw tool result completed the game."""

    details = message.get("details", {})
    structured = details.get("structuredContent") if isinstance(details, dict) else None

    if isinstance(structured, dict):
        return structured.get("done") is True

    content = _openclaw_content_text(message.get("content"))
    marker = "structuredContent:"

    if marker not in content:
        return False

    try:
        structured = json.loads(content.split(marker, 1)[1].strip())
    except json.JSONDecodeError:
        return False

    return isinstance(structured, dict) and structured.get("done") is True


def _openclaw_runtime(session: list[dict[str, Any]],
                      trajectory: list[dict[str, Any]],
                      stdout_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Collect concise OpenClaw runtime details from native artifacts."""

    runtime = {}

    for record in session:
        record_type = record.get("type")

        if record_type == "session":
            runtime.update({
                key: record[key]
                for key in ("id", "timestamp", "cwd")
                if record.get(key) is not None
            })
        elif record_type == "model_change":
            runtime.update({
                key: record[key]
                for key in ("provider", "modelId")
                if record.get(key) is not None
            })
        elif record_type == "thinking_level_change" and record.get("thinkingLevel") is not None:
            runtime["thinking_level"] = record["thinkingLevel"]

    for record in trajectory:
        if record.get("type") != "trace.metadata":
            continue

        data = record.get("data", {})
        harness = data.get("harness", {}) if isinstance(data, dict) else {}
        model = data.get("model", {}) if isinstance(data, dict) else {}

        if isinstance(harness, dict):
            runtime["harness_version"] = harness.get("version")

        if isinstance(model, dict):
            runtime["model_api"] = model.get("api")
            runtime["reasoning_level"] = model.get("reasoningLevel")

        break

    stdout_meta = stdout_payload.get("meta", {}) if isinstance(stdout_payload, dict) else {}
    agent_meta = stdout_meta.get("agentMeta", {}) if isinstance(stdout_meta, dict) else {}

    if isinstance(stdout_meta, dict) and stdout_meta.get("durationMs") is not None:
        runtime["duration_ms"] = stdout_meta["durationMs"]

    if isinstance(agent_meta, dict):
        for source_key, target_key in (
            ("sessionId", "session_id"),
            ("provider", "provider"),
            ("model", "model"),
            ("contextTokens", "context_tokens"),
            ("agentHarnessId", "agent_harness_id"),
            ("promptTokens", "prompt_tokens"),
        ):
            if agent_meta.get(source_key) is not None:
                runtime[target_key] = agent_meta[source_key]

    return {key: value for key, value in runtime.items() if value is not None}


def _openclaw_result(session: list[dict[str, Any]],
                     trajectory: list[dict[str, Any]],
                     stdout_payload: dict[str, Any] | None,
                     metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Collect OpenClaw completion status and usage details."""

    result = {}
    stdout_meta = stdout_payload.get("meta", {}) if isinstance(stdout_payload, dict) else {}
    agent_meta = stdout_meta.get("agentMeta", {}) if isinstance(stdout_meta, dict) else {}
    payloads = stdout_payload.get("payloads", []) if isinstance(stdout_payload, dict) else []

    if isinstance(stdout_meta, dict):
        for source_key, target_key in (
            ("durationMs", "duration_ms"),
            ("aborted", "aborted"),
        ):
            if stdout_meta.get(source_key) is not None:
                result[target_key] = stdout_meta[source_key]

    if isinstance(agent_meta, dict):
        for source_key, target_key in (
            ("usage", "usage"),
            ("lastCallUsage", "last_call_usage"),
            ("contextBudgetStatus", "context_budget_status"),
        ):
            if agent_meta.get(source_key) is not None:
                result[target_key] = agent_meta[source_key]

    final_text = [payload.get("text") for payload in payloads
                  if isinstance(payload, dict) and payload.get("text")]

    if final_text:
        result["final_text"] = "\n".join(final_text)

    for record in trajectory:
        if record.get("type") not in {"trace.artifacts", "session.ended"}:
            continue

        data = record.get("data", {})

        if not isinstance(data, dict):
            continue

        for key in (
            "status",
            "finalStatus",
            "aborted",
            "externalAbort",
            "timedOut",
            "idleTimedOut",
            "timedOutDuringCompaction",
            "timedOutDuringToolExecution",
            "promptError",
            "promptErrorSource",
            "usage",
            "compactionCount",
        ):
            if data.get(key) is not None:
                result[key] = data[key]

    if isinstance(metadata, dict):
        for source_key, target_key in (
            ("success", "success"),
            ("runtime_error", "runtime_error"),
            ("game_completed", "game_completed"),
            ("terminated_after_game", "terminated_after_game"),
            ("returncode", "returncode"),
        ):
            if metadata.get(source_key) is not None:
                result[target_key] = metadata[source_key]

    game_completed = any(
        record.get("type") == "message"
        and isinstance(record.get("message"), dict)
        and record["message"].get("role") == "toolResult"
        and _openclaw_tool_result_done(record["message"])
        for record in session
    )

    if game_completed:
        result["game_completed"] = True
        result["success"] = True

    if game_completed and result.get("externalAbort") is True:
        cleanup = {}

        for key in (
            "status",
            "finalStatus",
            "aborted",
            "externalAbort",
            "promptError",
            "promptErrorSource",
        ):
            if key in result:
                cleanup[key] = result.pop(key)

        result["status"] = "completed"
        result["terminal_reason"] = "terminated_after_game_completion"
        result["cleanup"] = cleanup

    return result


def _openclaw_system_prompt_report(
        stdout_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return non-content metadata OpenClaw reports for its system prompt."""

    meta = stdout_payload.get("meta", {}) if isinstance(stdout_payload, dict) else {}
    report = meta.get("systemPromptReport", {}) if isinstance(meta, dict) else {}
    prompt = report.get("systemPrompt", {}) if isinstance(report, dict) else {}

    if not isinstance(prompt, dict):
        return {}

    return {
        target_key: prompt[source_key]
        for source_key, target_key in (
            ("chars", "reported_chars"),
            ("hash", "reported_hash"),
            ("projectContextChars", "reported_project_context_chars"),
            ("nonProjectContextChars", "reported_non_project_context_chars"),
        )
        if prompt.get(source_key) is not None
    }


def _openclaw_content_text(content: Any) -> str:
    """Return readable text from OpenClaw message content blocks."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")

                if text is not None:
                    parts.append(str(text))
            elif isinstance(block, str):
                parts.append(block)

        return "\n".join(parts)

    if isinstance(content, dict):
        return json.dumps(content, indent=2, ensure_ascii=False)

    return str(content)


def _openclaw_json_value(value: Any) -> Any:
    """Decode an OpenClaw JSON argument string when possible."""

    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _hermes_session_export(episode_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Load the single Hermes session exported for one episode."""

    export_path = episode_dir / "hermes_session_export.jsonl"

    if not export_path.exists():
        return None, []

    sessions = []
    errors = []

    for line_number, line in enumerate(export_path.read_text(
            encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"Could not parse session-export line {line_number}: {error}")
            continue

        if isinstance(value, dict):
            sessions.append(value)
        else:
            errors.append(f"Session-export line {line_number} is not an object")

    if not sessions:
        return None, errors

    return sessions[-1], errors


def _hermes_wire_configuration(records: list[dict[str, Any]]) -> tuple[list[str], list[Any]]:
    """Extract native instructions and tool schemas from Hermes requests."""

    instructions = []
    seen_instructions = set()
    tools = []

    for record in records:
        if record.get("type") != "model_request":
            continue

        request = record.get("payload", {})

        if not isinstance(request, dict):
            continue

        if not tools and isinstance(request.get("tools"), list):
            tools = request["tools"]

        for message in request.get("messages", []):
            if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
                continue

            content = _hermes_content_text(message.get("content"))

            if content and content not in seen_instructions:
                seen_instructions.add(content)
                instructions.append(content)

    return instructions, tools


def _hermes_query_instruction(trace_text: str) -> str | None:
    """Recover the user query printed by Hermes before initialization."""

    stdout = _hermes_chat_stdout(trace_text)
    match = re.search(r"(?:^|\n)Query: (?P<content>.*?)(?:\nInitializing agent\.\.\.)",
                      stdout,
                      flags=re.DOTALL)
    return match.group("content").strip() if match is not None else None


def _hermes_tool_inventory(trace_text: str) -> list[dict[str, str]]:
    """Return the names Hermes reports in its final enabled tool set."""

    stdout = _hermes_chat_stdout(trace_text)
    match = re.search(r"Final tool selection \(\d+ tools\): (?P<tools>[^\n]+)", stdout)

    if match is None:
        return []

    return [{"name": name.strip()} for name in match.group("tools").split(",") if name.strip()]


def _hermes_runtime(trace_text: str,
                    session: dict[str, Any] | None) -> dict[str, Any]:
    """Collect concise Hermes runtime details from trace and session data."""

    stdout = _hermes_chat_stdout(trace_text)
    runtime = {}
    model_match = re.search(r"AI Agent initialized with model: (?P<model>[^\n]+)", stdout)
    context_match = re.search(
        r"Context limit: (?P<context>[\d,]+) tokens \(compress at (?P<percent>\d+)% = (?P<threshold>[\d,]+)\)",
        stdout,
    )
    command_match = re.search(
        r"hermes_chat_command:\nhermes chat --provider (?P<provider>\S+) --model (?P<model>\S+).*?(?=\n)",
        trace_text,
    )

    if model_match is not None:
        runtime["model"] = model_match.group("model").strip()

    if command_match is not None:
        runtime["provider"] = command_match.group("provider")
        runtime.setdefault("model", command_match.group("model"))

    if context_match is not None:
        runtime["context_limit"] = int(context_match.group("context").replace(",", ""))
        runtime["compression_percent"] = int(context_match.group("percent"))
        runtime["compression_threshold"] = int(context_match.group("threshold").replace(",", ""))

    if "--yolo" in trace_text:
        runtime["permission_mode"] = "yolo"

    if session is not None:
        for source_key, target_key in (
            ("id", "session_id"),
            ("model", "session_model"),
            ("provider", "session_provider"),
            ("started_at", "started_at"),
            ("ended_at", "ended_at"),
            ("end_reason", "end_reason"),
        ):
            if session.get(source_key) is not None:
                runtime[target_key] = session[source_key]

    return runtime


def _hermes_result(trace_text: str,
                   session: dict[str, Any] | None) -> dict[str, Any]:
    """Collect the Hermes completion status without treating cleanup as failure."""

    result = {}
    success_matches = list(re.finditer(r"(?:^|\n)success:\s*(True|False)", trace_text))
    timeout_match = re.search(r"hermes_timeout:\s*(?P<timeout>[^\n]+)", trace_text)

    if success_matches:
        result["success"] = success_matches[-1].group(1) == "True"

    if timeout_match is not None:
        result["terminal_reason"] = "timeout"
        result["timeout"] = timeout_match.group("timeout")
    elif "Interrupted during API call" in trace_text:
        result["terminal_reason"] = "terminated_after_game"

    if session is not None:
        for key in (
            "end_reason",
            "finish_reason",
            "message_count",
            "tool_call_count",
            "input_tokens",
            "output_tokens",
            "total_cost",
        ):
            if session.get(key) is not None:
                result[key] = session[key]

    return result


def _hermes_session_events(session: dict[str, Any],
                           agent_loop_instruction: str | None) -> tuple[list[dict[str, Any]], bool]:
    """Convert Hermes session messages into ordered semantic events."""

    events = []
    call_turns = {}
    current_turn = 0
    synthetic_call_number = 0
    has_native_instruction = False

    for message in session.get("messages", []):
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = _hermes_content_text(message.get("content"))

        if role in {"system", "developer"}:
            if content:
                has_native_instruction = True
                events.append({
                    "type": "instruction",
                    "kind": "native_harness",
                    "source": "hermes_session_export",
                    "role": role,
                    "content": content,
                    "payload": message
                })
            continue

        if role == "assistant":
            current_turn += 1
            reasoning = _hermes_content_text(
                message.get("reasoning") or message.get("reasoning_content")
            )
            tool_calls = message.get("tool_calls") or []

            if reasoning:
                events.append({
                    "type": "reasoning",
                    "source": "hermes_session_export",
                    "turn": current_turn,
                    "content": reasoning,
                    "payload": {
                        "reasoning_details": message.get("reasoning_details")
                    }
                })

            if content:
                events.append({
                    "type": "tool_preamble" if tool_calls else "assistant_text",
                    "source": "hermes_session_export",
                    "turn": current_turn,
                    "content": content,
                    "payload": message
                })

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue

                function = tool_call.get("function", {})
                function = function if isinstance(function, dict) else {}
                synthetic_call_number += 1
                call_id = tool_call.get("id") or f"hermes-call-{synthetic_call_number}"
                call_turns[call_id] = current_turn
                events.append({
                    "type": "tool_call",
                    "source": "hermes_session_export",
                    "turn": current_turn,
                    "call_id": call_id,
                    "name": function.get("name") or tool_call.get("name"),
                    "arguments": _hermes_json_value(
                        function.get("arguments", tool_call.get("arguments", {}))
                    ),
                    "payload": tool_call
                })
            continue

        if role == "tool":
            call_id = message.get("tool_call_id")
            events.append({
                "type": "tool_result",
                "source": "hermes_session_export",
                "turn": call_turns.get(call_id),
                "call_id": call_id,
                "name": message.get("tool_name"),
                "content": _hermes_readable_tool_result(
                    _hermes_json_value(message.get("content"))
                ),
                "payload": message
            })
            continue

        if role == "user" and content == agent_loop_instruction:
            continue

        if content:
            events.append({
                "type": "message",
                "source": "hermes_session_export",
                "role": role,
                "content": content,
                "payload": message
            })

    return events, has_native_instruction


def _hermes_cli_events(trace_text: str) -> list[dict[str, Any]]:
    """Parse reasoning and tool exchanges from Hermes verbose output."""

    records = _hermes_cli_records(_hermes_chat_stdout(trace_text))
    events = []
    pending_calls = []
    current_turn = 0
    turn_open = False
    call_number = 0

    for record in records:
        record_type = record["type"]

        if record_type == "reasoning":
            if not turn_open:
                current_turn += 1
                turn_open = True

            events.append({
                "type": "reasoning",
                "source": "hermes_cli",
                "turn": current_turn,
                "content": _hermes_unwrap_prose(record["content"])
            })
            continue

        if record_type == "tool_call":
            if not turn_open:
                current_turn += 1
                turn_open = True

            call_number += 1
            call_id = f"hermes-call-{call_number}"
            pending_calls.append((call_id, current_turn, record.get("name")))
            events.append({
                "type": "tool_call",
                "source": "hermes_cli",
                "turn": current_turn,
                "call_id": call_id,
                "name": record.get("name"),
                "arguments": _hermes_json_value(record.get("arguments", "")),
                "payload": {"display": record.get("arguments", "")}
            })
            continue

        if pending_calls:
            call_id, turn, name = pending_calls.pop(0)
        else:
            call_id, turn, name = None, current_turn or None, None

        parsed_result = _hermes_json_value(record.get("content", ""))
        events.append({
            "type": "tool_result",
            "source": "hermes_cli",
            "turn": turn,
            "call_id": call_id,
            "name": name,
            "content": _hermes_readable_tool_result(parsed_result),
            "payload": parsed_result
        })

        if not pending_calls:
            turn_open = False

    return events


def _hermes_cli_records(stdout: str) -> list[dict[str, Any]]:
    """Return Hermes reasoning, tool-call, and tool-result display blocks."""

    records = []
    reasoning_pattern = re.compile(
        r"^┌─ Reasoning [^\n]*\n(?P<content>.*?)\n└[─]+┘",
        flags=re.MULTILINE | re.DOTALL,
    )
    tool_call_pattern = re.compile(
        r"^[ \t]*📞 Tool \d+: (?P<name>[^\s(]+)\([^\n]*\)\n"
        r"[ \t]*Args: (?P<arguments>.*?)(?=\n[ \t]*(?:┊|✅|📞|⚡))",
        flags=re.MULTILINE | re.DOTALL,
    )
    tool_result_pattern = re.compile(
        r"^[ \t]*✅ Tool \d+ completed[^\n]*\n"
        r"[ \t]*Result: (?P<content>.*?)"
        r"(?=\n(?:[ \t]*\n)?(?:┌─ Reasoning|[ \t]*┊|[ \t]*📞 Tool|⚡ Interrupt)|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )

    for match in reasoning_pattern.finditer(stdout):
        records.append({
            "position": match.start(),
            "type": "reasoning",
            "content": match.group("content").strip()
        })

    for match in tool_call_pattern.finditer(stdout):
        records.append({
            "position": match.start(),
            "type": "tool_call",
            "name": match.group("name"),
            "arguments": match.group("arguments").strip()
        })

    for match in tool_result_pattern.finditer(stdout):
        records.append({
            "position": match.start(),
            "type": "tool_result",
            "content": match.group("content").strip()
        })

    return sorted(records, key=lambda record: record["position"])


def _hermes_chat_stdout(trace_text: str) -> str:
    """Return only Hermes chat stdout from the combined adapter trace."""

    match = re.search(
        r"hermes_chat_stdout:\n(?P<content>.*?)(?:\nhermes_chat_stderr:|\Z)",
        trace_text,
        flags=re.DOTALL,
    )
    return match.group("content") if match is not None else trace_text


def _hermes_content_text(content: Any) -> str:
    """Return readable text from Hermes/OpenAI message content."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")

                if text is not None:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))

        return "\n".join(parts)

    if isinstance(content, dict):
        return json.dumps(content, indent=2, ensure_ascii=False)

    return str(content)


def _hermes_json_value(value: Any) -> Any:
    """Decode JSON strings, including Hermes terminal-wrapped JSON."""

    if not isinstance(value, str):
        return value

    stripped = value.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        unwrapped = re.sub(r"\n[ \t]+", " ", stripped)

        try:
            return json.loads(unwrapped)
        except json.JSONDecodeError:
            return stripped


def _hermes_readable_tool_result(result: Any) -> Any:
    """Prefer the meaningful result while retaining the full payload separately."""

    if not isinstance(result, dict):
        return result

    structured = result.get("structuredContent")

    if isinstance(structured, dict):
        context = structured.get("context")

        if isinstance(context, dict):
            sections = []

            if "done" in structured:
                sections.append(f"done: {structured['done']}")

            if structured.get("reward") is not None:
                sections.append(f"reward: {structured['reward']}")

            role = context.get("role")
            content = context.get("content")

            if role:
                sections.append(f"{role}: {content}")
            elif content is not None:
                sections.append(str(content))

            if sections:
                return "\n\n".join(sections)

    if "output" in result:
        sections = [str(result["output"])]

        if result.get("exit_code") is not None:
            sections.append(f"exit_code: {result['exit_code']}")

        if result.get("error"):
            sections.append(f"error: {result['error']}")

        return "\n\n".join(sections)

    if result.get("success") is True and "data" in result:
        return result["data"]

    return result


def _hermes_unwrap_prose(content: str) -> str:
    """Remove Rich terminal wrapping while preserving paragraph breaks."""

    paragraphs = re.split(r"\n\s*\n", content.strip())
    return "\n\n".join(
        " ".join(line.strip() for line in paragraph.splitlines())
        for paragraph in paragraphs
    )


def _codex_trace_records(trace_text: str) -> list[dict[str, Any]]:
    """Return raw Codex trace blocks in their original order."""

    records = []
    request_pattern = re.compile(
        r"raw_upstream_request_(?P<id>\d+)_start\n"
        r"path: (?P<path>[^\n]*)\n"
        r"(?P<body>.*?)\n"
        r"raw_upstream_request_(?P=id)_end",
        re.DOTALL
    )
    response_pattern = re.compile(
        r"raw_upstream_response_(?P<id>\d+)_start\n"
        r"path: (?P<path>[^\n]*)\n"
        r"status: (?P<status>\d+)\n"
        r"content_type: (?P<content_type>[^\n]*)\n"
        r"content_encoding: (?P<content_encoding>[^\n]*)\n"
        r"(?P<body>.*?)\n"
        r"raw_upstream_response_(?P=id)_end",
        re.DOTALL
    )
    instruction_pattern = re.compile(
        r"agent_loop_instruction_start\n"
        r"(?P<content>.*?)\n"
        r"agent_loop_instruction_end",
        re.DOTALL
    )

    for match in request_pattern.finditer(trace_text):
        raw = match.group("body")

        try:
            payload = json.loads(raw)
            parse_error = None
        except json.JSONDecodeError as error:
            payload = {}
            parse_error = str(error)

        records.append({
            "position": match.start(),
            "type": "model_request",
            "path": match.group("path"),
            "raw": raw,
            "payload": payload,
            "parse_error": parse_error
        })

    for match in response_pattern.finditer(trace_text):
        records.append({
            "position": match.start(),
            "type": "model_response",
            "path": match.group("path"),
            "status": int(match.group("status")),
            "content_type": match.group("content_type"),
            "content_encoding": match.group("content_encoding"),
            "raw": match.group("body")
        })

    for match in instruction_pattern.finditer(trace_text):
        records.append({
            "position": match.start(),
            "type": "agent_loop_instruction",
            "content": match.group("content")
        })

    return sorted(records, key=lambda record: record["position"])


def _codex_request_events(request: dict[str, Any],
                          turn: int,
                          seen_input_items: set[str],
                          seen_output_item_ids: set[str]) -> tuple[list[dict[str, Any]], bool]:
    """Extract previously unseen input items from one Responses request."""

    events = []
    found_agent_loop_instruction = False

    for item in request.get("input", []):
        item_key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        item_id = item.get("id") if isinstance(item, dict) else None

        if item_key in seen_input_items or item_id in seen_output_item_ids:
            continue

        seen_input_items.add(item_key)
        item_type = item.get("type") if isinstance(item, dict) else None

        if item_type == "function_call_output":
            events.append({
                "type": "tool_result",
                "source": "wire_request",
                "turn": turn,
                "call_id": item.get("call_id"),
                "content": _codex_tool_result_text(item.get("output")),
                "payload": item
            })
            continue

        if item_type == "function_call":
            continue

        if item_type == "reasoning":
            events.append({
                "type": "reasoning",
                "source": "wire_request",
                "turn": turn,
                "content": _codex_reasoning_text(item),
                "payload": item
            })
            continue

        if item_type != "message":
            events.append({
                "type": "message",
                "source": "wire_request",
                "turn": turn,
                "payload": item
            })
            continue

        role = item.get("role")
        text = _codex_message_text(item.get("content"))
        event_type = "message"
        kind = None

        if role == "developer":
            event_type = "instruction"
            kind = "runtime"
        elif _is_agent_loop_instruction(text):
            event_type = "instruction"
            kind = "agent_loop"
            found_agent_loop_instruction = True
        elif text.startswith("<environment_context>"):
            event_type = "instruction"
            kind = "environment_context"

        event = {
            "type": event_type,
            "source": "wire_request",
            "turn": turn,
            "role": role,
            "content": text,
            "payload": item
        }

        if kind is not None:
            event["kind"] = kind

        events.append(event)

    return events, found_agent_loop_instruction


def _codex_response_events(record: dict[str, Any],
                           turn: int,
                           seen_output_item_ids: set[str]) -> list[dict[str, Any]]:
    """Extract completed Responses API output items from one SSE response."""

    events = []
    completed_items = []

    for line in record["raw"].splitlines():
        if not line.startswith("data: "):
            continue

        try:
            event = json.loads(line.removeprefix("data: "))
        except json.JSONDecodeError:
            continue

        if event.get("type") == "response.output_item.done":
            completed_items.append(event.get("item", {}))

    tool_item_types = {"function_call", "openrouter:web_search", "web_search_call"}

    for item_index, item in enumerate(completed_items):
        item_type = item.get("type")
        item_id = item.get("id")

        if item_id:
            seen_output_item_ids.add(item_id)

        if item_type in {"openrouter:web_search", "web_search_call"}:
            action = item.get("action", {})
            query = action.get("query") if isinstance(action, dict) else None
            sources = action.get("sources", []) if isinstance(action, dict) else []
            source_urls = [source.get("url") for source in sources
                           if isinstance(source, dict) and source.get("url")]
            events.append({
                "type": "tool_call",
                "kind": "hosted_web_search",
                "source": "wire_response",
                "turn": turn,
                "call_id": item_id,
                "name": "web_search",
                "arguments": {"query": query},
                "payload": item
            })
            events.append({
                "type": "tool_result",
                "kind": "hosted_web_search",
                "source": "wire_response",
                "turn": turn,
                "call_id": item_id,
                "name": "web_search",
                "content": (
                    "Search-result context was supplied to the model internally.\n"
                    "The retrieved text is not exposed in this trace.\n\n"
                    "Captured source metadata:\n" + "\n".join(source_urls)
                ),
                "payload": {
                    "status": item.get("status"),
                    "sources": sources
                }
            })
            continue

        if item_type == "function_call":
            events.append({
                "type": "tool_call",
                "source": "wire_response",
                "turn": turn,
                "call_id": item.get("call_id"),
                "name": item.get("name"),
                "arguments": _codex_tool_arguments(item.get("arguments")),
                "payload": item
            })
            continue

        if item_type == "message":
            has_following_tool_call = any(
                later_item.get("type") in tool_item_types
                for later_item in completed_items[item_index + 1:]
            )

            for content in item.get("content", []):
                content_type = content.get("type")

                if content_type == "output_text":
                    events.append({
                        "type": "tool_preamble" if has_following_tool_call else "assistant_text",
                        "source": "wire_response",
                        "turn": turn,
                        "content": content.get("text", ""),
                        "payload": content
                    })
                elif content_type == "reasoning":
                    events.append({
                        "type": "reasoning",
                        "source": "wire_response",
                        "turn": turn,
                        "content": _codex_reasoning_text(content),
                        "payload": content
                    })

            continue

        if item_type == "reasoning":
            events.append({
                "type": "reasoning",
                "source": "wire_response",
                "turn": turn,
                "content": _codex_reasoning_text(item),
                "payload": item
            })
            continue

        events.append({
            "type": "assistant_output",
            "source": "wire_response",
            "turn": turn,
            "payload": item
        })

    return events


def _codex_runtime_errors(trace_text: str) -> list[dict[str, Any]]:
    """Extract adapter-level runtime errors from the native trace."""

    errors = []

    for match in re.finditer(r"agent_runtime_error: (?P<content>.*)", trace_text):
        errors.append({
            "type": "error",
            "source": "native_cli",
            "content": match.group("content")
        })

    return errors


def _codex_message_text(content: Any) -> str:
    """Return the text content from one Responses API message item."""

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    return "\n".join(
        str(part.get("text") or part.get("input_text") or "")
        for part in content
        if isinstance(part, dict)
    )


def _codex_reasoning_text(item: Any) -> str:
    """Return provider-exposed reasoning text from one Responses item."""

    if isinstance(item, str):
        return item

    if not isinstance(item, dict):
        return ""

    text_parts = []

    for field_name in ("content", "summary"):
        content = item.get(field_name, [])

        if isinstance(content, str):
            text_parts.append(content)
            continue

        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, dict):
                continue

            text = part.get("text") or part.get("reasoning_text")

            if text:
                text_parts.append(str(text))

    return "\n\n".join(text_parts)


def _codex_tool_arguments(arguments: Any) -> Any:
    """Parse function arguments when they were encoded as JSON text."""

    if not isinstance(arguments, str):
        return arguments

    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def _codex_tool_result_text(output: Any) -> str:
    """Extract the readable game message from a Codex tool result."""

    if not isinstance(output, str):
        return str(output)

    prefix, separator, possible_json = output.partition("\nOutput:\n")

    if not separator:
        return output

    try:
        result = json.loads(possible_json)
    except json.JSONDecodeError:
        return output

    if not isinstance(result, dict):
        return output

    context = result.get("context")

    if not isinstance(context, dict):
        return output

    sections = []

    if prefix:
        sections.append(prefix)

    if "done" in result:
        sections.append(f"done: {result['done']}")

    if result.get("reward") is not None:
        sections.append(f"reward: {result['reward']}")

    role = context.get("role")
    content = context.get("content")

    if role:
        sections.append(f"{role}: {content}")
    elif content is not None:
        sections.append(str(content))

    return "\n\n".join(sections)


def _is_agent_loop_instruction(text: str) -> bool:
    """Return whether a user message is the universal game-loop instruction."""

    return text.startswith("You are connected to a game environment through MCP tools.")
