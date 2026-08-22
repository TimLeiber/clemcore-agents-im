import json
import threading
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import urllib3


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def _prepare_request_body(
    body: bytes,
    path: str,
    _completion_path: Path,
    request_body_overrides: dict[str, Any],
    upstream_model: str | None = None,
) -> tuple[bytes, dict[str, tuple[str, str]]]:
    """Inject registry options and normalize Responses API tool namespaces."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body, {}

    if not isinstance(payload, dict):
        return body, {}

    request_path = urlsplit(path).path.rstrip("/")
    is_responses = request_path.endswith("/responses")
    is_anthropic_messages = request_path.endswith("/messages")

    if upstream_model is not None:
        payload["model"] = upstream_model

    # Registry extra_body fields follow the OpenAI-compatible wire format.
    # Do not add those provider extensions to Anthropic Messages requests.
    if request_body_overrides and not is_anthropic_messages:
        payload = _deep_merge(payload, request_body_overrides)

    namespace_map = _flatten_namespace_tools(payload) if is_responses else {}

    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), namespace_map


def _namespace_prefix(namespace: str) -> str:
    return namespace if namespace.endswith("__") else f"{namespace}__"


def _flatten_input_function_calls(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _flatten_input_function_calls(item)
        return

    if not isinstance(value, dict):
        return

    if value.get("type") == "function_call":
        namespace = value.get("namespace")
        name = value.get("name")

        if isinstance(namespace, str) and isinstance(name, str):
            value["name"] = f"{_namespace_prefix(namespace)}{name}"
            value.pop("namespace", None)

    for child in value.values():
        _flatten_input_function_calls(child)


def _flatten_namespace_tools(
    payload: dict[str, Any],
) -> dict[str, tuple[str, str]]:
    """Expose Responses namespace tools as ordinary functions for local models."""
    tools = payload.get("tools")

    if not isinstance(tools, list):
        return {}

    flattened_tools = []
    namespace_map: dict[str, tuple[str, str]] = {}

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "namespace":
            flattened_tools.append(tool)
            continue

        namespace = tool.get("name")
        inner_tools = tool.get("tools")

        if not isinstance(namespace, str) or not isinstance(inner_tools, list):
            flattened_tools.append(tool)
            continue

        for inner_tool in inner_tools:
            if not isinstance(inner_tool, dict) or inner_tool.get("type") != "function":
                continue

            inner_name = inner_tool.get("name")

            if not isinstance(inner_name, str):
                continue

            flat_name = f"{_namespace_prefix(namespace)}{inner_name}"
            flattened_tool = dict(inner_tool)
            flattened_tool["name"] = flat_name
            flattened_tools.append(flattened_tool)
            namespace_map[flat_name] = (namespace, inner_name)

    payload["tools"] = flattened_tools
    _flatten_input_function_calls(payload.get("input"))
    return namespace_map


def _resolve_namespaced_tool(
    name: str,
    namespace_map: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    # This is deliberately exact. The proxy translates the wire format but
    # must not repair a model-generated tool name such as `start_game`, a
    # separator variant, or a misspelling.
    return namespace_map.get(name)


def _restore_namespace_function_calls(
    value: Any,
    namespace_map: dict[str, tuple[str, str]],
) -> None:
    if isinstance(value, list):
        for item in value:
            _restore_namespace_function_calls(item, namespace_map)
        return

    if not isinstance(value, dict):
        return

    if value.get("type") == "function_call":
        name = value.get("name")

        if isinstance(name, str):
            match = _resolve_namespaced_tool(name, namespace_map)

            if match is not None:
                namespace, inner_name = match
                value["name"] = inner_name
                value["namespace"] = namespace

    for child in value.values():
        _restore_namespace_function_calls(child, namespace_map)


def _rewrite_responses_body(
    body: bytes,
    content_type: str,
    namespace_map: dict[str, tuple[str, str]],
) -> bytes:
    if not namespace_map:
        return body

    if content_type.startswith("text/event-stream"):
        rewritten_lines = []

        for line in body.decode("utf-8", errors="replace").splitlines(keepends=True):
            stripped = line.rstrip("\r\n")
            newline = line[len(stripped):]

            if not stripped.startswith("data: "):
                rewritten_lines.append(line)
                continue

            try:
                event = json.loads(stripped[6:])
            except json.JSONDecodeError:
                rewritten_lines.append(line)
                continue

            _restore_namespace_function_calls(event, namespace_map)
            rewritten_lines.append(
                f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}{newline}"
            )

        return "".join(rewritten_lines).encode("utf-8")

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body

    _restore_namespace_function_calls(payload, namespace_map)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self,
                 target_origin: str,
                 completion_path: Path,
                 request_body_overrides: dict[str, Any],
                 verify_tls: bool,
                 upstream_model: str | None,
                 trace_responses: bool,
                 trace_requests: bool):
        super().__init__(("127.0.0.1", 0), _ProxyHandler)
        self.target_origin = target_origin
        self.completion_path = completion_path
        self.request_body_overrides = request_body_overrides
        self.verify_tls = verify_tls
        self.upstream_model = upstream_model
        self.trace_responses = trace_responses
        self.trace_requests = trace_requests
        self.response_trace_lock = threading.Lock()
        self.response_trace_count = 0
        self.trace_records: list[str] = []

    def trace_request(self,
                      path: str,
                      body: bytes) -> None:
        if not self.trace_requests:
            return

        request_text = body.decode("utf-8", errors="replace")

        with self.response_trace_lock:
            self.response_trace_count += 1
            trace_id = self.response_trace_count
            trace_record = (
                f"\nraw_upstream_request_{trace_id}_start\n"
                f"path: {path}\n"
                f"{request_text}\n"
                f"raw_upstream_request_{trace_id}_end"
            )
            self.trace_records.append(trace_record)
            print(trace_record, flush=True)

    def trace_response(self,
                       path: str,
                       status_code: int,
                       content_type: str,
                       content_encoding: str,
                       body: bytes) -> None:
        if not self.trace_responses:
            return

        response_text = body.decode("utf-8", errors="replace")

        with self.response_trace_lock:
            self.response_trace_count += 1
            trace_id = self.response_trace_count
            trace_record = (
                f"\nraw_upstream_response_{trace_id}_start\n"
                f"path: {path}\n"
                f"status: {status_code}\n"
                f"content_type: {content_type}\n"
                f"content_encoding: {content_encoding}\n"
                f"{response_text}\n"
                f"raw_upstream_response_{trace_id}_end"
            )
            self.trace_records.append(trace_record)
            print(trace_record, flush=True)

    def captured_trace(self) -> str:
        """Return request and response records in their observed order."""

        with self.response_trace_lock:
            return "\n".join(self.trace_records)


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _proxy(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b""
        body, namespace_map = _prepare_request_body(
            body,
            self.path,
            self.server.completion_path,
            self.server.request_body_overrides,
            self.server.upstream_model,
        )
        self.server.trace_request(self.path, body)
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP_HEADERS
        }
        target_url = f"{self.server.target_origin}{self.path}"

        try:
            upstream = requests.request(
                self.command,
                target_url,
                headers=headers,
                data=body or None,
                stream=True,
                timeout=(30, 600),
                verify=self.server.verify_tls,
            )
        except requests.RequestException as error:
            payload = json.dumps({"error": f"OpenAI-compatible proxy failed: {error}"}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True
            return

        if namespace_map:
            try:
                # The rewritten response must be decoded before we remove the
                # upstream Content-Encoding header and compute a new length.
                upstream.raw.decode_content = True
                response_body = upstream.raw.read()
                content_type = upstream.headers.get("Content-Type", "")
                self.server.trace_response(
                    self.path,
                    upstream.status_code,
                    content_type,
                    upstream.headers.get("Content-Encoding", ""),
                    response_body,
                )
                response_body = _rewrite_responses_body(
                    response_body,
                    content_type,
                    namespace_map,
                )

                self.send_response(upstream.status_code)

                for name, value in upstream.headers.items():
                    if (
                        name.lower() not in HOP_BY_HOP_HEADERS
                        and name.lower() != "content-encoding"
                    ):
                        self.send_header(name, value)

                self.send_header("Content-Length", str(len(response_body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(response_body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                upstream.close()
                self.close_connection = True

            return

        self.send_response(upstream.status_code)

        for name, value in upstream.headers.items():
            if name.lower() not in HOP_BY_HOP_HEADERS:
                self.send_header(name, value)

        self.send_header("Connection", "close")
        self.end_headers()

        response_chunks = []

        try:
            upstream.raw.decode_content = False

            while True:
                chunk = upstream.raw.read(65536)

                if not chunk:
                    break

                response_chunks.append(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.server.trace_response(
                self.path,
                upstream.status_code,
                upstream.headers.get("Content-Type", ""),
                upstream.headers.get("Content-Encoding", ""),
                b"".join(response_chunks),
            )
            upstream.close()
            self.close_connection = True

    do_DELETE = _proxy
    do_GET = _proxy
    do_PATCH = _proxy
    do_POST = _proxy
    do_PUT = _proxy

    def log_message(self, format: str, *args: Any) -> None:
        return


class OpenAICompatibleProxy(AbstractContextManager["OpenAICompatibleProxy"]):
    """Local protocol-preserving proxy for OpenAI-compatible model servers."""

    def __init__(self,
                 target_base_url: str,
                 completion_path: Path,
                 request_body_overrides: dict[str, Any] | None = None,
                 verify_tls: bool = True,
                 upstream_model: str | None = None,
                 trace_responses: bool = True,
                 trace_requests: bool = False):
        parsed = urlsplit(target_base_url)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid OpenAI-compatible base URL: {target_base_url!r}")

        target_origin = f"{parsed.scheme}://{parsed.netloc}"
        self._base_path = parsed.path.rstrip("/")
        self._server = _ProxyServer(
            target_origin=target_origin,
            completion_path=completion_path,
            request_body_overrides=request_body_overrides or {},
            verify_tls=verify_tls,
            upstream_model=upstream_model,
            trace_responses=trace_responses,
            trace_requests=trace_requests,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="clem-openai-compatible-proxy",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}{self._base_path}"

    def captured_trace(self) -> str:
        """Return request and response records captured by the proxy server."""

        return self._server.captured_trace()

    def __enter__(self) -> "OpenAICompatibleProxy":
        if not self._server.verify_tls:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def proxy_for_model_connection(connection: dict[str, Any] | None,
                               completion_path: Path,
                               include_openrouter: bool = False,
                               trace_responses: bool = True,
                               trace_requests: bool = False) -> OpenAICompatibleProxy | None:
    if not connection:
        return None

    backend = connection.get("backend")

    if backend != "openai_compatible" and not (include_openrouter and backend == "openrouter"):
        return None

    base_url = connection.get("base_url")

    if not isinstance(base_url, str) or not base_url:
        raise ValueError("OpenAI-compatible model connections require base_url.")

    overrides = connection.get("request_body_overrides")

    if not isinstance(overrides, dict):
        overrides = {}

    return OpenAICompatibleProxy(
        target_base_url=base_url,
        completion_path=completion_path,
        request_body_overrides=overrides,
        verify_tls=bool(connection.get("verify_tls", True)),
        upstream_model=connection.get("model"),
        trace_responses=trace_responses,
        trace_requests=trace_requests,
    )
