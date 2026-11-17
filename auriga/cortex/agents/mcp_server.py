"""Host-owned streamable-HTTP MCP endpoint for registry tools (Claude Code, OpenCode).

One loopback endpoint serves the central tool registry to every MCP-speaking agent backend (design 04 §3.1-§3.4).
It implements the minimal JSON-RPC subset both pinned clients use, in-repo and without the ``mcp`` package:

==============================  ===================================================================================
POST ``initialize``             protocol negotiation (``SUPPORTED_PROTOCOL_VERSIONS``), ``tools`` capability, no
                                ``instructions`` (Auriga instructions have one channel)
POST ``notifications/*``        202, no body; ``notifications/cancelled`` abandons a call that is still in flight
                                and is ignored once that call's response has been completely written
POST ``ping``                   ``{}``
POST ``tools/list``             the whole catalog, scrubbed once by ``ActorPresentation`` (see ``mcp_tool_catalog``);
                                each ``inputSchema`` is projected to the portable form: top-level ``anyOf``/``oneOf``/
                                ``allOf``/``if``/``then``/``else`` removed (``task_plan``)
POST ``tools/call``             ``call_tool`` -> ``encode_mcp_result``; SSE with ``notifications/progress`` keepalive
                                when the client accepts ``text/event-stream``, else one JSON body
POST anything else              JSON-RPC error -32601 (includes Claude Code's ``server/discover`` probe)
GET / DELETE                    405 (no server-initiated stream, no MCP session to delete)
==============================  ===================================================================================

Facts versus presentation: the server never parses tool text. ``call_tool`` returns the wire envelope
``{"success": bool, "contentItems": [...]}`` that ``Harness.handle_tool`` produces (after the session's second
presentation scrub); ``encode_mcp_result`` maps it item by item, passing image and audio bytes through unchanged.

Delivery certainty: a ``tools/call`` whose answer may not have reached the agent calls
``on_delivery_failure(call_id, reason)`` exactly once. The admitted call itself is never aborted; its late result
is discarded. Reasons (``DELIVERY_FAILURE_REASONS``):

* ``"tool_reply_delivery_uncertain"`` - a write error before the response was completely written, a
  ``notifications/cancelled`` or a peer close while the call was in flight, or ``close()`` with the call in flight.
  A call abandoned before ``call_tool`` was entered is never executed and not reported.
* ``"tool_reply_encoding_failed"`` - the result could not be encoded (e.g. a non-``data:`` image URL); the agent
  received a JSON-RPC error instead of the result.

Raw log: ``raw_log(direction, message)`` (normally ``Store.raw_agent(..., channel="mcp")``) receives every received
JSON-RPC request/notification (``"received"``) and every completely written response (``"sent"``), full content.
The logged copy of a ``tools/call`` request carries one host annotation, ``params._meta.item_id = <call id>``; it is
never sent to anyone. Progress keepalives are not logged, and a response whose write failed is not logged as sent,
so the log proves only what was actually delivered.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import re
import secrets
import select
import socket
import threading
import time
from urllib.parse import urlsplit

from ...core.contracts import canonical, new_id
from ...core.tool_errors import error_response
from ...core.tool_output import InvalidContentItem, validate_content_part

__all__ = ["AurigaMcpServer", "DEFAULT_RESULT_SIZE_HINT_CHARS", "DELIVERY_FAILURE_REASONS", "MCP_PATH",
           "SUPPORTED_PROTOCOL_VERSIONS", "TOP_LEVEL_COMBINATORS", "encode_mcp_result", "mcp_tool_catalog",
           "portable_input_schema"]

MCP_PATH = "/mcp"
# Newest first. Both pinned clients (Claude Code 2.1.321, OpenCode 1.19.5) offer 2025-11-25.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
# Claude Code replaces any text result longer than 50,000 characters by a pointer to a private file unless the tool
# publishes _meta["anthropic/maxResultSizeChars"] (spike S2); 10,000,000 characters covers every registry result.
DEFAULT_RESULT_SIZE_HINT_CHARS = 10_000_000
RESULT_SIZE_META_KEY = "anthropic/maxResultSizeChars"
NATIVE_CALL_ID_META_KEY = "claudecode/toolUseId"
DELIVERY_UNCERTAIN = "tool_reply_delivery_uncertain"
ENCODING_FAILED = "tool_reply_encoding_failed"
DELIVERY_FAILURE_REASONS = (DELIVERY_UNCERTAIN, ENCODING_FAILED)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
# Largest unread request body drained before a rejection is answered (bytes).
DISCARD_LIMIT_BYTES = 1024 * 1024
_MEDIA = {"inputImage": ("imageUrl", "image", "invalid_tool_image_url"),
          "inputAudio": ("audioUrl", "audio", "invalid_tool_audio_url")}
# data:<type>/<subtype>[;param=value...];base64,<payload>
_DATA_URL = re.compile(r"data:([A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+)((?:;[^;,]*)*);base64,(.*)", re.DOTALL)


def _server_version():
    try:
        return package_version("auriga-harness")
    except PackageNotFoundError:
        return "unknown"


def _media_part(item, kind):
    field_name, mcp_type, error_code = _MEDIA[kind]
    match = _DATA_URL.fullmatch(item[field_name])
    if match is None:
        raise ValueError(error_code)
    return {"type": mcp_type, "data": match.group(3), "mimeType": match.group(1)}


def encode_mcp_result(response):
    """Encode one tool wire envelope as an MCP ``CallToolResult`` (pure; never parses text for facts).

    response: ``{"success": bool, "contentItems": [...]}`` as returned by ``Harness.handle_tool`` /
    ``core.tool_errors.error_response`` after presentation. ``success`` is the response-contract flag, not physical
    success. Items map in order (the relay's trim placeholders rely on "text then its images"):

    * ``{"type": "inputText", "text": t}`` -> ``{"type": "text", "text": t}``
    * ``{"type": "inputImage", "imageUrl": "data:<mime>;base64,<b64>"}`` -> ``{"type": "image", "data": <b64>,
      "mimeType": <mime>}``; the base64 text is passed through byte-identically (no decoding, resizing or
      re-encoding).
    * ``{"type": "inputAudio", "audioUrl": "data:<mime>;base64,<b64>"}`` -> ``{"type": "audio", ...}`` likewise.

    ``isError`` is ``not success`` only for results without image/audio items: both clients collapse an ``isError``
    result to its joined text and OpenCode drops its images, so a result carrying evidence media is always sent with
    ``isError: false`` (its failure stays readable in the typed text).

    Raises ``ValueError``: ``invalid_tool_result`` (not an envelope), ``invalid_native_content_item`` (an item outside
    the text/image/audio union), ``invalid_tool_image_url`` / ``invalid_tool_audio_url`` (anything but a base64
    ``data:`` URL; such a URL is never forwarded).

    Example: ``encode_mcp_result({"success": False, "contentItems": [{"type": "inputText", "text": "e"}]})`` ->
    ``{"content": [{"type": "text", "text": "e"}], "isError": True}``.
    """
    if (not isinstance(response, dict) or type(response.get("success")) is not bool
            or not isinstance(response.get("contentItems"), list)):
        raise ValueError("invalid_tool_result")
    content, media = [], False
    for item in response["contentItems"]:
        try:
            validate_content_part(item)
        except (InvalidContentItem, TypeError, ValueError, OverflowError):
            raise ValueError("invalid_native_content_item") from None
        if not isinstance(item, dict):  # e.g. an unserialized StructuredText
            raise ValueError("invalid_native_content_item")
        if item["type"] == "inputText":
            content.append({"type": "text", "text": item["text"]})
        else:
            content.append(_media_part(item, item["type"]))
            media = True
    return {"content": content, "isError": (not response["success"]) and not media}


# Keywords an Anthropic tool input schema does not accept at its top level (Claude Code 2.1.321 drops the tool, OpenCode on
# anthropic_messages forwards it to a provider that rejects it).
TOP_LEVEL_COMBINATORS = ("anyOf", "oneOf", "allOf", "if", "then", "else")


def portable_input_schema(schema):
    """Copy of ``schema`` without top-level ``anyOf``/``oneOf``/``allOf``/``if``/``then``/``else``.

    Only the top level is touched; nested combinators stay. The removed rules remain stated in the tool description
    (the generated ``input: ... Object rules: ...`` clause of the field guide, pinned by
    ``tests/test_tool_registry_prose_bounds.py``) and are enforced at runtime against the registry's full schema.
    """
    return {key: value for key, value in schema.items() if key not in TOP_LEVEL_COMBINATORS}


def mcp_tool_catalog(tools, presentation, *, parallel_hints=None, result_size_hint_chars=None):
    """Build the ``tools/list`` catalog from the registry tool definitions Codex receives.

    tools: ``ToolRegistry.catalog(runtime, role="actor")`` (what ``Workbench.actor_tools()`` passes to
    ``session.start``); each definition needs ``name``, ``description``, ``inputSchema``. ``deferLoading`` and
    ``type`` are dropped: MCP backends publish every tool eagerly (design §3.8).
    presentation: the session's ``ActorPresentation``. ``description`` goes through ``present_text`` and
    ``inputSchema`` through ``present_data``, exactly as Codex ``dynamicTools`` are projected
    (``wire.openai_responses.present_responses_request``), then loses its top-level ``anyOf``/``oneOf``/``allOf``/
    ``if``/``then``/``else`` (``portable_input_schema``; harness difference, the rules stay in the description and
    in the registry's runtime validation).
    parallel_hints: ``{tool name: ToolSpec.parallel as published by ToolRegistry.manifest()}``, i.e. ``True``,
    ``False`` or ``"request_dependent"``; build it with
    ``{t["name"]: t["parallel"] for t in registry.manifest(runtime)["tools"]}``. ``annotations.readOnlyHint`` is
    ``True`` only when the hint is exactly ``True`` and ``False`` otherwise (missing, ``False``, request-dependent).
    Claude Code runs consecutive read-only calls concurrently and everything else serially; the host
    ``OrderedBarrier`` stays the ordering authority. Default ``None``: every tool ``readOnlyHint: false``.
    result_size_hint_chars: ``_meta["anthropic/maxResultSizeChars"]`` published on every tool, in characters of
    text result; positive int, default ``DEFAULT_RESULT_SIZE_HINT_CHARS`` (10,000,000). Mandatory for Claude Code,
    which otherwise rewrites text results over 50,000 characters to a private file pointer.

    Returns a new list of ``{name, description, inputSchema, annotations, _meta}``; raises ``ValueError``
    ``invalid_mcp_tool_definition`` / ``duplicate_mcp_tool:<name>`` / ``invalid_result_size_hint_chars``.
    """
    size = DEFAULT_RESULT_SIZE_HINT_CHARS if result_size_hint_chars is None else result_size_hint_chars
    if type(size) is not int or size <= 0:
        raise ValueError("invalid_result_size_hint_chars")
    hints = dict(parallel_hints or {})
    catalog, names = [], set()
    for tool in tools:
        if (not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"]
                or not isinstance(tool.get("description"), str) or not isinstance(tool.get("inputSchema"), dict)):
            raise ValueError("invalid_mcp_tool_definition")
        if tool["name"] in names:
            raise ValueError("duplicate_mcp_tool:" + tool["name"])
        names.add(tool["name"])
        catalog.append({"name": tool["name"],
                        "description": presentation.present_text(tool["description"]),
                        "inputSchema": portable_input_schema(presentation.present_data(tool["inputSchema"])),
                        "annotations": {"readOnlyHint": hints.get(tool["name"]) is True},
                        "_meta": {RESULT_SIZE_META_KEY: size}})
    return catalog


def _json_bytes(message):
    return json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _rpc_error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _host_name(value):
    """``(host, port)`` of a ``Host`` header value (``"127.0.0.1:8080"`` -> ``("127.0.0.1", 8080)``);
    ``(None, None)`` when unparseable."""
    try:
        parts = urlsplit(value if "//" in value else "//" + value)
        return parts.hostname, parts.port
    except ValueError:
        return None, None


@dataclass(eq=False)
class _Call:
    """One in-flight ``tools/call``. ``state``: running -> answered | abandoned | failed, and ``dispatched`` (set
    once ``call_tool`` is entered), both changed only under ``lock``; ``lock`` also serializes writes."""
    key: str
    rpc_id: object
    call_id: str
    progress_token: object
    sse: bool
    connection: object
    wfile: object
    next_keepalive: float = 0.0
    progress: int = 0
    state: str = "running"
    dispatched: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class AurigaMcpServer:
    """Loopback streamable-HTTP MCP endpoint (``POST http://127.0.0.1:<ephemeral>/mcp``) for registry tools.

    Keyword arguments (all required unless a default is shown):

    tools: registry tool definitions (``ToolRegistry.catalog(runtime, role="actor")``), see ``mcp_tool_catalog``.
    presentation: the session's ``ActorPresentation``; scrubs the catalog once at construction and renders the
        ``error_response`` of a ``call_tool`` exception.
    call_tool: ``call_tool(name, arguments, *, call_id, native_call_id, rpc_id) -> {"success", "contentItems"}``,
        called on the request's handler thread; it blocks until the tool finished (normally
        ``ToolDispatcher.submit(params).result()`` -> ``Harness.handle_tool``, then the presentation scrub).
        ``call_id`` is the host call id: ``native_call_id`` when the client sent one
        (``params._meta["claudecode/toolUseId"]``, Claude Code) else a fresh ``new_id("call")`` (OpenCode); it is
        also the logged ``params._meta.item_id``. ``rpc_id`` is the JSON-RPC request id. An exception is answered
        as ``core.tool_errors.error_response`` (as the Codex transport does).
    raw_log: ``raw_log(direction, message)`` with direction ``"received"``/``"sent"`` (``Store.raw_agent`` with
        ``channel="mcp"``). A failure to log a received ``tools/call`` refuses the call before dispatch.
    on_delivery_failure: ``on_delivery_failure(call_id, reason)``, called at most once per call with a reason from
        ``DELIVERY_FAILURE_REASONS``; the session invalidates ``runtime/image_context_epoch`` and emits
        ``agent.tool_reply_unknown``. Must not raise. Failures of ``raw_log``/``on_delivery_failure`` never stop
        the server; they are kept in ``errors`` as ``(call id or "raw_log", repr(error))``.
    parallel_hints: per-tool ``ToolSpec.parallel`` from ``ToolRegistry.manifest()``; drives
        ``annotations.readOnlyHint`` (see ``mcp_tool_catalog``). Default None: no tool is read-only.
    server_name: ``serverInfo.name`` and the name agents prefix tools with (``mcp__auriga__…``, ``auriga_…``);
        default ``"auriga"``.
    keepalive_s: seconds between SSE keepalives while a call runs, default 5.0: a ``notifications/progress`` for the
        call's ``progressToken`` (progress notifications, not SSE comments, reset Claude Code's MCP idle timer and
        OpenCode's request timeout) or an SSE comment when the client sent no token. Must be > 0.
    peer_check_s: seconds between peer-close checks of in-flight calls (a closed socket abandons the call),
        default 0.25; must be > 0. The keepalive thread wakes at ``min(keepalive_s, peer_check_s)``.
    max_request_bytes: largest accepted request body in bytes (``Content-Length``), default 16 MiB; larger -> 413.
    io_timeout_s: socket timeout in seconds for reading a request and writing a response, default 30.0.
    result_size_hint_chars: ``_meta["anthropic/maxResultSizeChars"]`` on every tool, default 10,000,000.

    After ``start()``: ``url`` (``http://127.0.0.1:<port>/mcp``) and ``headers`` (``{"Authorization": "Bearer
    <per-run token>"}``) configure the agent. Every request must carry that bearer (else 401) and a loopback
    ``Host`` (``127.0.0.1`` or ``localhost`` with this port, else 403). These are the only access controls (loopback
    bind, per-run bearer, Host check, request size cap): the agent's own shell can read the bearer from its
    environment, a documented residual (design §12.3, user decision: document only, no further hardening).
    """

    def __init__(self, *, tools, presentation, call_tool, raw_log, on_delivery_failure, parallel_hints=None,
                 server_name="auriga", keepalive_s=5.0, peer_check_s=0.25, max_request_bytes=16 * 1024 * 1024,
                 io_timeout_s=30.0, result_size_hint_chars=None):
        for name, value in (("keepalive_s", keepalive_s), ("peer_check_s", peer_check_s),
                            ("io_timeout_s", io_timeout_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ValueError("invalid_" + name)
        if type(max_request_bytes) is not int or max_request_bytes <= 0:
            raise ValueError("invalid_max_request_bytes")
        if not isinstance(server_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", server_name):
            raise ValueError("invalid_mcp_server_name")
        self.server_name = server_name
        self.catalog = mcp_tool_catalog(tools, presentation, parallel_hints=parallel_hints,
                                        result_size_hint_chars=result_size_hint_chars)
        self.presentation, self.call_tool, self.raw_log = presentation, call_tool, raw_log
        self.on_delivery_failure = on_delivery_failure
        self.keepalive_s, self.peer_check_s = float(keepalive_s), float(peer_check_s)
        self.max_request_bytes, self.io_timeout_s = max_request_bytes, float(io_timeout_s)
        self.token = secrets.token_urlsafe(32)
        self.headers = {"Authorization": "Bearer " + self.token}
        self.url = None
        self.server = self.thread = self.keepalive_thread = None
        self.closed = False
        self.lock = threading.Lock()
        self.inflight = {}
        self.errors = []
        self._stop = threading.Event()

    @property
    def read_only_tools(self):
        """Names published with ``readOnlyHint: true`` (provenance ``tool_transport.client_parallel_hint``)."""
        return sorted(tool["name"] for tool in self.catalog if tool["annotations"]["readOnlyHint"])

    # lifecycle -------------------------------------------------------------------------------------------------
    def start(self):
        if self.server is not None or self.closed:
            raise ValueError("mcp_server_already_started_or_closed")
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = owner.io_timeout_s
            body_consumed = False

            def log_message(self, *_):
                pass

            def do_GET(self):
                self.close_connection = True
                if self.admitted():
                    self.send_empty(405, {"Allow": "POST"})

            do_DELETE = do_GET

            def do_POST(self):
                self.close_connection = True  # one request per connection, as ModelRequestProxy
                if self.admitted():
                    owner._post(self)

            def admitted(self):
                """Path, loopback Host and bearer checks; sends the rejection and returns False."""
                if urlsplit(self.path).path != MCP_PATH:
                    self.send_empty(404)
                    return False
                host, port = _host_name(self.headers.get("Host") or "")
                if (host or "").lower() not in LOOPBACK_HOSTS or port != owner.server.server_port:
                    self.send_empty(403)
                    return False
                supplied = self.headers.get("Authorization") or ""
                if not hmac.compare_digest(supplied.encode("utf-8", "replace"),
                                           owner.headers["Authorization"].encode()):
                    self.send_empty(401, {"WWW-Authenticate": "Bearer"})
                    return False
                return True

            def discard_body(self):
                """Read and drop an unread request body of at most ``DISCARD_LIMIT_BYTES`` so that closing the
                socket after a rejection does not reset the connection before the client read the status line;
                a larger body is not read (the client may then see a reset instead of the status)."""
                if self.command != "POST" or self.body_consumed or self.headers.get("Transfer-Encoding"):
                    return
                self.body_consumed = True
                try:
                    remaining = int(self.headers.get("Content-Length", "0"))
                    if remaining > DISCARD_LIMIT_BYTES:
                        return
                    while remaining > 0:
                        chunk = self.rfile.read(min(remaining, 65536))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                except (OSError, ValueError):
                    pass

            def send_empty(self, status, headers=None):
                self.discard_body()
                self.send_response(status)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

            def send_json(self, status, message):
                body = _json_bytes(message)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                self.close_connection = True

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False
            request_queue_size = 64

        self.server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}{MCP_PATH}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .05},
                                       name="auriga-mcp-http", daemon=True)
        self.thread.start()
        self.keepalive_thread = threading.Thread(target=self._keepalive_loop, name="auriga-mcp-keepalive",
                                                 daemon=True)
        self.keepalive_thread.start()
        return self

    def close(self):
        """Stop accepting requests; idempotent.

        A call still in flight can no longer be answered: it is abandoned (``on_delivery_failure`` with
        ``tool_reply_delivery_uncertain``) and its socket closed. Its handler thread keeps waiting for
        ``call_tool``; the session joins in-flight tool work through its ``ToolDispatcher`` after the robot closed.
        """
        with self.lock:
            if self.closed:
                return
            self.closed = True
            calls = list(self.inflight.values())
        self._stop.set()
        for call in calls:
            self._abandon(call)
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        for thread in (self.thread, self.keepalive_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(2)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

    # request handling ------------------------------------------------------------------------------------------
    def _log(self, direction, message):
        """Append one record to the raw log; returns False (recorded in ``errors``) when the sink failed."""
        try:
            self.raw_log(direction, message)
        except Exception as error:  # noqa: BLE001 - a logging failure must not stop the endpoint
            self.errors.append(("raw_log", repr(error)))
            return False
        return True

    def _post(self, handler):
        if self.closed:
            handler.send_empty(503)
            return
        if handler.headers.get("Transfer-Encoding"):
            handler.send_empty(411)
            return
        try:
            length = int(handler.headers.get("Content-Length", ""))
        except ValueError:
            handler.send_empty(411)
            return
        if length < 0:
            handler.send_empty(400)
            return
        if length > self.max_request_bytes:
            handler.send_empty(413)
            return
        raw = handler.rfile.read(length)
        handler.body_consumed = True
        try:
            message = json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            self._reply(handler, 400, _rpc_error(None, -32700, "Parse error"))
            return
        if not isinstance(message, dict):
            # A batch or a bare value: one JSON-RPC message per request (protocol 2025-06-18 and later).
            self._reply(handler, 400, _rpc_error(None, -32600, "Invalid Request: one JSON-RPC message per request"))
            return
        method = message.get("method")
        if "id" not in message or (method is None and ("result" in message or "error" in message)):
            # Notification, or a client response to a server request (this server sends none).
            self._log("received", message)
            if method == "notifications/cancelled":
                self._cancelled(message.get("params"))
            handler.send_empty(202)
            return
        request_id = message["id"]
        if not isinstance(method, str) or not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
            self._log("received", message)
            self._reply(handler, 400, _rpc_error(None, -32600, "Invalid Request"))
            return
        if method == "tools/call":
            self._tools_call(handler, message)
            return
        self._log("received", message)
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if method == "initialize":
            offered = params.get("protocolVersion")
            result = {"protocolVersion": offered if offered in SUPPORTED_PROTOCOL_VERSIONS
                      else SUPPORTED_PROTOCOL_VERSIONS[0],
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": self.server_name, "version": _server_version()}}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": deepcopy(self.catalog)}
        else:
            self._reply(handler, 200, _rpc_error(request_id, -32601, "Method not found"))
            return
        self._reply(handler, 200, {"jsonrpc": "2.0", "id": request_id, "result": result})

    def _reply(self, handler, status, message):
        """Send one JSON response and log it as sent once completely written."""
        try:
            handler.send_json(status, message)
        except OSError:
            return False
        self._log("sent", message)
        return True

    def _tools_call(self, handler, message):
        request_id = message["id"]
        params = message.get("params")
        name = params.get("name") if isinstance(params, dict) else None
        arguments = params.get("arguments", {}) if isinstance(params, dict) else None
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            self._log("received", message)
            self._reply(handler, 200, _rpc_error(
                request_id, -32602, "Invalid params: tools/call needs a string name and an object arguments"))
            return
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        native = meta.get(NATIVE_CALL_ID_META_KEY)
        native_call_id = native if isinstance(native, str) and 0 < len(native) <= 256 else None
        call_id = native_call_id or new_id("call")
        logged = {**message, "params": {**params, "_meta": {**meta, "item_id": call_id}}}
        sse = "text/event-stream" in (handler.headers.get("Accept") or "").lower()
        call = _Call(key=canonical(request_id), rpc_id=request_id, call_id=call_id,
                     progress_token=meta.get("progressToken"), sse=sse, connection=handler.connection,
                     wfile=handler.wfile)
        with self.lock:
            duplicate = call.key in self.inflight
            if not duplicate and not self.closed:
                self.inflight[call.key] = call
        if duplicate or self.closed:
            self._log("received", message)
            self._reply(handler, 200 if duplicate else 503, _rpc_error(
                request_id, -32600, "Invalid Request: duplicate in-flight request id" if duplicate
                else "Server closing"))
            return
        try:
            if not self._log("received", logged):  # no audit record, no dispatch
                self._reply(handler, 200, _rpc_error(request_id, -32603, "Internal error: tool call not recorded; "
                                                                         "it was not executed"))
                return
            if sse:
                with call.lock:
                    try:
                        handler.send_response(200)
                        handler.send_header("Content-Type", "text/event-stream")
                        handler.send_header("Cache-Control", "no-cache")
                        handler.send_header("Connection", "close")
                        handler.end_headers()
                        handler.wfile.flush()
                    except OSError:
                        if call.state == "running":
                            call.state = "failed"
                        return  # Peer already gone: nothing was dispatched, nothing to report.
                    call.next_keepalive = time.monotonic() + self.keepalive_s
            with call.lock:
                if call.state != "running":
                    return  # Cancelled or peer gone before dispatch: never executed, nothing to report.
                call.dispatched = True
            self._run_call(handler, call, name, arguments, native_call_id)
        finally:
            with self.lock:
                if self.inflight.get(call.key) is call:
                    del self.inflight[call.key]

    def _run_call(self, handler, call, name, arguments, native_call_id):
        try:
            result = self.call_tool(name, arguments, call_id=call.call_id, native_call_id=native_call_id,
                                    rpc_id=call.rpc_id)
        except Exception as error:  # noqa: BLE001 - answered like the Codex transport (RecordedClient._approval)
            result = self.presentation.present(error_response(error, tool=name, arguments=arguments))
        encoding_failed = False
        try:
            response = {"jsonrpc": "2.0", "id": call.rpc_id, "result": encode_mcp_result(result)}
            body = _json_bytes(response)
        except (ValueError, TypeError) as error:
            encoding_failed = True
            code = str(error) if str(error).startswith(("invalid_tool_", "invalid_native_")) else "invalid_tool_result"
            response = _rpc_error(call.rpc_id, -32603, code + ": the tool call ran but its result could not be "
                                  "delivered; it may have taken effect, so check its status before any retry")
            body = _json_bytes(response)
        with call.lock:
            if call.state != "running":
                return  # Abandoned: the late result is discarded (delivery already reported uncertain).
            gone = self._peer_gone(call)
            if not gone:
                try:
                    if call.sse:
                        call.wfile.write(b"event: message\ndata: " + body + b"\n\n")
                        call.wfile.flush()
                    else:
                        handler.send_response(200)
                        handler.send_header("Content-Type", "application/json")
                        handler.send_header("Content-Length", str(len(body)))
                        handler.send_header("Connection", "close")
                        handler.end_headers()
                        handler.wfile.write(body)
                        handler.wfile.flush()
                except OSError:
                    gone = True
            if gone:
                call.state = "failed"
            else:
                call.state = "answered"
                self._log("sent", response)
        if gone:
            self._report(call.call_id, DELIVERY_UNCERTAIN)
        elif encoding_failed:
            self._report(call.call_id, ENCODING_FAILED)

    # delivery tracking -----------------------------------------------------------------------------------------
    def _report(self, call_id, reason):
        try:
            self.on_delivery_failure(call_id, reason)
        except Exception as error:  # noqa: BLE001 - must not stop the keepalive thread or the handler
            self.errors.append((call_id, repr(error)))

    def _abandon(self, call, *, blocking=True):
        """Mark a running call abandoned (late result discarded) and close its socket; report uncertain delivery
        only when ``call_tool`` was already entered (an undispatched call never runs). ``blocking=False`` (the
        shared keepalive thread) skips a call whose lock is held by a write in progress; returns whether the call
        was abandoned now."""
        if not call.lock.acquire(blocking=blocking):
            return False
        try:
            if call.state != "running":
                return False
            call.state = "abandoned"
            dispatched = call.dispatched
        finally:
            call.lock.release()
        with self.lock:
            if self.inflight.get(call.key) is call:
                del self.inflight[call.key]
        if dispatched:
            self._report(call.call_id, DELIVERY_UNCERTAIN)
        try:
            call.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        return True

    def _cancelled(self, params):
        """``notifications/cancelled``: abandon the call only while its response is not completely written.

        OpenCode's MCP client sends a cancel 6-30 ms after every answered call; that cancel finds no running call
        (the state turned ``answered`` under the call lock right after the final write) and changes nothing.
        """
        if not isinstance(params, dict) or "requestId" not in params:
            return
        try:
            key = canonical(params["requestId"])
        except (TypeError, ValueError):
            return
        with self.lock:
            call = self.inflight.get(key)
        if call is not None:
            self._abandon(call)

    @staticmethod
    def _peer_gone(call):
        """True when the client closed (EOF) or reset the connection; never blocks.

        ``poll`` (not ``select``, which fails for descriptors >= 1024 in a busy gateway) with a zero timeout; only a
        readable socket is peeked, so pending request bytes are never consumed.
        """
        try:
            if hasattr(select, "poll"):
                poller = select.poll()
                poller.register(call.connection, select.POLLIN | select.POLLPRI)
                readable = bool(poller.poll(0))
            else:  # pragma: no cover - platforms without poll()
                readable = bool(select.select([call.connection], [], [], 0)[0])
            return readable and not call.connection.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):
            return True

    def _keepalive_loop(self):
        tick = min(self.keepalive_s, self.peer_check_s)
        while not self._stop.wait(tick):
            with self.lock:
                calls = list(self.inflight.values())
            for call in calls:
                if call.state != "running":
                    continue
                if self._peer_gone(call):
                    self._abandon(call, blocking=False)
                    continue
                if call.sse and call.next_keepalive and time.monotonic() >= call.next_keepalive:
                    self._keepalive(call)

    def _keepalive(self, call):
        # Never wait behind a call's final write (possibly megabytes): this thread serves every call.
        if not call.lock.acquire(blocking=False):
            return
        try:
            if call.state != "running":
                return
            if call.progress_token is not None:
                call.progress += 1
                note = {"jsonrpc": "2.0", "method": "notifications/progress",
                        "params": {"progressToken": call.progress_token, "progress": call.progress}}
                frame = b"event: message\ndata: " + _json_bytes(note) + b"\n\n"
            else:
                frame = b": keepalive\n\n"
            try:
                call.wfile.write(frame)
                call.wfile.flush()
            except OSError:
                failed = True
            else:
                failed = False
                call.next_keepalive = time.monotonic() + self.keepalive_s
        finally:
            call.lock.release()
        if failed:
            self._abandon(call)
