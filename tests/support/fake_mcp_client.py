"""Minimal streamable-HTTP MCP client for testing an MCP server (stdlib only, no host code).

    client = FakeMcpClient('http://127.0.0.1:1234/mcp', token='secret')
    client.handshake()                       # initialize + notifications/initialized
    client.list_tools()
    res = client.call_tool('observe', {'x': 1}, progress_token='p1')   # JSON or SSE reply, both parsed
    res.result, res.progress, res.response.raw

One new HTTP connection per request, so calls from several threads are independent.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

__all__ = ['Exchange', 'FakeMcpClient', 'McpError', 'McpResponse', 'PendingCall', 'ToolCallResult', 'parse_sse']


class McpError(Exception):
    """Raised by the convenience methods on an HTTP failure or JSON-RPC error."""

    def __init__(self, status: int, error: Any) -> None:
        super().__init__(f'MCP error (HTTP {status}): {error}')
        self.status, self.error = status, error


def parse_sse(raw: bytes) -> list[dict]:
    """Split an SSE body into frames ``{event, data, id, comments}``; ``data`` lines are joined with newlines."""
    frames = []
    for block in raw.decode('utf-8', 'replace').replace('\r\n', '\n').split('\n\n'):
        if not block.strip():
            continue
        frame: dict = {'event': None, 'data': None, 'id': None, 'comments': []}
        for line in block.split('\n'):
            if line.startswith(':'):
                frame['comments'].append(line[1:].strip())
                continue
            name, _, value = line.partition(':')
            value = value[1:] if value.startswith(' ') else value
            if name == 'data':
                frame['data'] = value if frame['data'] is None else frame['data'] + '\n' + value
            elif name in ('event', 'id'):
                frame[name] = value
        frames.append(frame)
    return frames


@dataclass
class McpResponse:
    """Raw HTTP response.  ``headers`` is case-insensitive.  ``frames`` are the SSE frames (SSE replies only) and
    ``messages`` the JSON-RPC messages: the single JSON body, or every parseable SSE ``data`` payload in order."""
    status: int
    headers: Any
    raw: bytes
    frames: list = field(default_factory=list)
    messages: list = field(default_factory=list)

    @property
    def is_sse(self) -> bool:
        return 'text/event-stream' in str(self.headers.get('Content-Type', '')).lower()

    def message_for(self, request_id: Any) -> dict | None:
        """The JSON-RPC response (result or error) whose ``id`` equals ``request_id``."""
        for message in self.messages:
            if isinstance(message, dict) and message.get('id') == request_id and ('result' in message
                                                                                  or 'error' in message):
                return message
        return None


@dataclass
class Exchange:
    """One recorded request/response pair, for assertions: ``request_headers``, ``request_body`` (bytes)."""
    method: str
    request_headers: dict
    request_body: bytes
    response: McpResponse | None = None


@dataclass
class ToolCallResult:
    """Outcome of ``tools/call``.  ``request_id`` is the JSON-RPC id used; ``message`` the final JSON-RPC response
    (None if the HTTP reply carried none, e.g. 401); ``progress`` the ``params`` of every ``notifications/progress``
    seen in an SSE reply; ``result``/``error`` shortcuts into ``message``."""
    request_id: Any
    response: McpResponse
    message: dict | None
    progress: list

    @property
    def status(self) -> int:
        return self.response.status

    @property
    def result(self) -> dict | None:
        return (self.message or {}).get('result')

    @property
    def error(self) -> dict | None:
        return (self.message or {}).get('error')

    @property
    def is_sse(self) -> bool:
        return self.response.is_sse


class PendingCall:
    """A ``tools/call`` running in a background thread (see ``FakeMcpClient.start_call``)."""

    def __init__(self, request_id: Any) -> None:
        self.request_id = request_id
        self._done = threading.Event()
        self._result: ToolCallResult | None = None
        self._exc: BaseException | None = None

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float = 10.0) -> ToolCallResult:
        if not self._done.wait(timeout):
            raise TimeoutError('tools/call did not finish')
        if self._exc is not None:
            raise self._exc
        return self._result


class FakeMcpClient:
    """Streamable-HTTP MCP client.

    url: full endpoint URL (e.g. ``http://127.0.0.1:port/mcp``).   token: sent as ``Authorization: Bearer <token>``.
    headers: extra headers on every request (override defaults; a None value removes a header).
    timeout: socket timeout in seconds.   protocol_version: sent in ``initialize`` (``MCP-Protocol-Version`` is then
    sent on later requests).  OpenCode 1.19.5 (MCP SDK 1.29.0) offers ``2025-11-25``; servers should be tested with
    that value as well as the default.  ``exchanges`` records every request made; ``last_response`` is the latest response.
    """

    def __init__(self, url: str, *, token: str | None = None, headers: dict | None = None, timeout: float = 10.0,
                 protocol_version: str = '2025-06-18', client_name: str = 'fake-mcp-client',
                 client_version: str = '0.0.0') -> None:
        parts = urlsplit(url)
        self.host, self.port = parts.hostname, parts.port
        self.path = (parts.path or '/') + ('?' + parts.query if parts.query else '')
        self.token, self.extra_headers, self.timeout = token, dict(headers or {}), timeout
        self.protocol_version, self.client_info = protocol_version, {'name': client_name, 'version': client_version}
        self.session_id: str | None = None
        self.negotiated_version: str | None = None
        self.exchanges: list[Exchange] = []
        self._ids = iter(range(1, 1 << 30))
        self._lock = threading.Lock()

    @property
    def last_response(self) -> McpResponse | None:
        return self.exchanges[-1].response if self.exchanges else None

    def next_id(self) -> int:
        with self._lock:
            return next(self._ids)

    # low level -----------------------------------------------------------------------------------------------------
    def _headers(self, accept: str | None, extra: dict | None) -> dict:
        headers = {'Content-Type': 'application/json', 'Accept': accept or 'application/json, text/event-stream'}
        if self.token is not None:
            headers['Authorization'] = f'Bearer {self.token}'
        if self.session_id:
            headers['Mcp-Session-Id'] = self.session_id
        if self.negotiated_version:
            headers['MCP-Protocol-Version'] = self.negotiated_version
        headers.update(self.extra_headers)
        headers.update(extra or {})
        return {k: v for k, v in headers.items() if v is not None}

    def _open(self, message: dict | bytes, accept: str | None, extra: dict | None, method: str = 'POST'):
        body = message if isinstance(message, bytes) else json.dumps(message).encode()
        headers = self._headers(accept, extra)
        exchange = Exchange(message.get('method', '') if isinstance(message, dict) else method, headers, body)
        with self._lock:
            self.exchanges.append(exchange)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        conn.request(method, self.path, body=body if method == 'POST' else None, headers=headers)
        return conn, exchange

    def _read(self, conn: http.client.HTTPConnection, exchange: Exchange) -> McpResponse:
        try:
            resp = conn.getresponse()
            raw = resp.read()
        finally:
            conn.close()
        out = McpResponse(resp.status, resp.headers, raw)
        if out.is_sse:
            out.frames = parse_sse(raw)
            for frame in out.frames:
                if frame['data'] is not None:
                    try:
                        out.messages.append(json.loads(frame['data']))
                    except ValueError:
                        pass
        elif raw:
            try:
                out.messages.append(json.loads(raw))
            except ValueError:
                pass
        exchange.response = out
        return out

    def post(self, message: dict | bytes, *, accept: str | None = None, headers: dict | None = None) -> McpResponse:
        """POST any JSON-RPC message (or raw bytes) and return the raw ``McpResponse`` (never raises on status)."""
        conn, exchange = self._open(message, accept, headers)
        return self._read(conn, exchange)

    def http(self, method: str, *, headers: dict | None = None) -> McpResponse:
        """Send a bodyless non-POST request (``GET``/``DELETE``) to the endpoint, e.g. to assert 405."""
        conn, exchange = self._open(b'', None, headers, method)
        return self._read(conn, exchange)

    # JSON-RPC ------------------------------------------------------------------------------------------------------
    def request(self, method: str, params: dict | None = None, *, request_id: Any = None, accept: str | None = None,
                headers: dict | None = None) -> McpResponse:
        """Send a JSON-RPC request (id auto-allocated unless ``request_id`` is given); returns the raw response."""
        message: dict = {'jsonrpc': '2.0', 'id': self.next_id() if request_id is None else request_id,
                         'method': method}
        if params is not None:
            message['params'] = params
        return self.post(message, accept=accept, headers=headers)

    def notify(self, method: str, params: dict | None = None) -> McpResponse:
        """Send a JSON-RPC notification (no id); a compliant server answers 202."""
        message: dict = {'jsonrpc': '2.0', 'method': method}
        if params is not None:
            message['params'] = params
        return self.post(message)

    def _result(self, resp: McpResponse, request_id: Any) -> dict:
        message = resp.message_for(request_id)
        if resp.status != 200 or message is None or 'error' in message:
            raise McpError(resp.status, (message or {}).get('error') or resp.raw[:200])
        return message['result']

    def _call(self, method: str, params: dict | None = None) -> dict:
        request_id = self.next_id()
        return self._result(self.request(method, params, request_id=request_id), request_id)

    def initialize(self) -> dict:
        """``initialize``; stores ``Mcp-Session-Id`` (if any) and the negotiated protocol version; returns result."""
        request_id = self.next_id()
        resp = self.request('initialize', {'protocolVersion': self.protocol_version, 'capabilities': {},
                                           'clientInfo': self.client_info}, request_id=request_id)
        result = self._result(resp, request_id)
        self.session_id = resp.headers.get('Mcp-Session-Id') or self.session_id
        self.negotiated_version = result.get('protocolVersion')
        return result

    def initialized(self) -> McpResponse:
        return self.notify('notifications/initialized')

    def discover(self) -> McpResponse:
        """Claude Code 2.1.321 opens every HTTP server with a ``server/discover`` probe (protocol ``2026-07-28``,
        header ``mcp-method: server/discover``, string id) before ``initialize``; a server must answer it with a
        JSON-RPC error (e.g. -32601) and stay usable.  Returns the raw response for assertions."""
        meta = {'io.modelcontextprotocol/protocolVersion': '2026-07-28',
                'io.modelcontextprotocol/clientInfo': self.client_info}
        return self.request('server/discover', {'_meta': meta}, request_id='server-discover-probe-1',
                            headers={'mcp-method': 'server/discover', 'mcp-protocol-version': '2026-07-28'})

    def handshake(self, *, discover_probe: bool = False) -> dict:
        """``initialize`` then ``notifications/initialized``; returns the initialize result.  ``discover_probe`` first
        sends the ``server/discover`` probe exactly as Claude Code does (see ``discover``)."""
        if discover_probe:
            self.discover()
        result = self.initialize()
        self.initialized()
        return result

    def ping(self) -> dict:
        return self._call('ping')

    def list_tools(self) -> list[dict]:
        """``tools/list`` result's ``tools`` array."""
        return self._call('tools/list', {})['tools']

    @staticmethod
    def _call_params(name: str, arguments: dict | None, meta: dict | None, progress_token: Any) -> dict:
        params: dict = {'name': name, 'arguments': {} if arguments is None else arguments}
        merged = dict(meta or {})
        if progress_token is not None:
            merged['progressToken'] = progress_token
        if merged:
            params['_meta'] = merged
        return params

    def call_tool(self, name: str, arguments: dict | None = None, *, meta: dict | None = None,
                  progress_token: Any = None, accept_sse: bool = True, request_id: Any = None,
                  headers: dict | None = None) -> ToolCallResult:
        """``tools/call``.  ``meta`` becomes ``params._meta`` (e.g. ``{"claudecode/toolUseId": "toolu_1"}``);
        ``progress_token`` adds ``_meta.progressToken``; ``accept_sse`` False sends ``Accept: application/json`` only,
        so the server must answer one JSON body.  Handles JSON and SSE replies; never raises on error status."""
        request_id = self.next_id() if request_id is None else request_id
        resp = self.request('tools/call', self._call_params(name, arguments, meta, progress_token),
                            request_id=request_id,
                            accept='application/json, text/event-stream' if accept_sse else 'application/json',
                            headers=headers)
        progress = [m['params'] for m in resp.messages
                    if isinstance(m, dict) and m.get('method') == 'notifications/progress']
        return ToolCallResult(request_id, resp, resp.message_for(request_id), progress)

    def start_call(self, name: str, arguments: dict | None = None, **kw: Any) -> PendingCall:
        """Run ``call_tool`` in a background thread (same keyword args); ``.request_id`` is known immediately, so the
        test can ``cancel()`` it while it runs; ``.wait()`` returns the ``ToolCallResult``."""
        kw['request_id'] = self.next_id() if kw.get('request_id') is None else kw['request_id']
        pending = PendingCall(kw['request_id'])

        def run() -> None:
            try:
                pending._result = self.call_tool(name, arguments, **kw)
            except BaseException as exc:  # noqa: BLE001 - surfaced from wait()
                pending._exc = exc
            finally:
                pending._done.set()

        threading.Thread(target=run, daemon=True).start()
        return pending

    def call_tool_then_cancel(self, name: str, arguments: dict | None = None, **kw: Any) -> ToolCallResult:
        """Real-client fidelity: the MCP TypeScript SDK used by OpenCode 1.19.5 sends ``notifications/cancelled``
        (reason ``AbortError: The operation was aborted.``) for EVERY ``tools/call`` when the model step's abort
        signal fires at step end - i.e. after the result was already received.  A server must ignore a cancel for a
        request id whose response it has fully written.  This helper reproduces that sequence."""
        result = self.call_tool(name, arguments, **kw)
        self.cancel(result.request_id, 'AbortError: The operation was aborted.')
        return result

    def open_standalone_stream(self) -> McpResponse:
        """``GET`` with ``Accept: text/event-stream`` (the optional server->client stream that the MCP SDK opens
        right after ``notifications/initialized``; a server without one must answer 405 and the client continues)."""
        conn, exchange = self._open(b'', 'text/event-stream', None, 'GET')
        return self._read(conn, exchange)

    def cancel(self, request_id: Any, reason: str | None = None) -> McpResponse:
        """``notifications/cancelled`` for ``request_id`` (optional ``reason``)."""
        params: dict = {'requestId': request_id}
        if reason is not None:
            params['reason'] = reason
        return self.notify('notifications/cancelled', params)

    def call_tool_and_disconnect(self, name: str, arguments: dict | None = None, *, meta: dict | None = None,
                                 progress_token: Any = None, accept_sse: bool = True, wait_for: str = 'headers',
                                 request_id: Any = None) -> Any:
        """Send ``tools/call`` then abruptly close the socket while the server is still working; returns the
        request id.  ``wait_for``: ``'headers'`` waits for the response status line first (SSE servers send it
        immediately), ``'sent'`` closes right after sending the request."""
        request_id = self.next_id() if request_id is None else request_id
        message = {'jsonrpc': '2.0', 'id': request_id, 'method': 'tools/call',
                   'params': self._call_params(name, arguments, meta, progress_token)}
        conn, _ = self._open(message, 'application/json, text/event-stream' if accept_sse else 'application/json',
                             None)
        try:
            if wait_for == 'headers':
                conn.getresponse()
        finally:
            sock = conn.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            conn.close()
        return request_id
