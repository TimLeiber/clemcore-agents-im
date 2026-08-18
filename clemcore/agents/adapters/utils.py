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
