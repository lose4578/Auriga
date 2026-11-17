"""Scripted fake model servers for three wire formats (stdlib only, loopback only).

    with FakeAnthropicServer([tool_step('read_file', {'path': 'a'}), text_step('ok')]) as server:
        ...point a client at server.base_url ...
        server.wait_for_requests(2)
        server.find_tool_results()

Servers: ``FakeAnthropicServer`` (``/v1/messages``), ``FakeChatServer`` (``/v1/chat/completions``),
``FakeResponsesServer`` (``/responses``).  Every request is recorded (``RecordedRequest``).  Each generation request
consumes the next ``Step`` of the script; when the script is exhausted the ``default`` step (plain text "done") is
served, so CLIs that issue extra calls never hang.  A script item may be a ``Step``, a plain ``str`` (text step) or a
callable ``request -> Step`` for tests that react to request content.
"""
from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlsplit

__all__ = ['Failure', 'FakeAnthropicServer', 'FakeChatServer', 'FakeModelServer', 'FakeResponsesServer', 'Gate',
           'Image', 'RecordedRequest', 'Step', 'ToolCall', 'ToolResult', 'Usage', 'extract_images', 'text_step',
           'tool_step']


@dataclass
class Usage:
    """Token counts reported by a step.  ``input_tokens`` excludes cache reads/writes on the Anthropic wire (its
    native semantics); on the OpenAI wires the prompt total is ``input_tokens + cache_read_input_tokens`` and
    ``cached_tokens = cache_read_input_tokens``.  ``cache_creation_input_tokens`` only exists on Anthropic;
    ``reasoning_tokens`` only on the OpenAI wires."""
    input_tokens: int = 10
    output_tokens: int = 5
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class ToolCall:
    """One scripted tool call.  ``arguments`` is a dict (serialised to JSON) or a raw ``str`` sent verbatim as the
    argument text (use for malformed JSON).  ``id`` defaults to a generated per-wire id."""
    name: str
    arguments: dict | str = field(default_factory=dict)
    id: str | None = None


@dataclass
class Failure:
    """Injected failure.  ``kind``:
      ``http``                HTTP ``status`` with a wire-shaped error body (``message``, optional ``error_type``,
                              ``retry_after`` seconds for 429, extra ``headers``); sent before any stream bytes.
      ``stream_error``        stream starts, then the wire's mid-stream error event, then the connection closes.
                              Note: the default error types (Anthropic ``overloaded_error``, OpenAI ``server_error``)
                              are retried by OpenCode; pass ``error_type='invalid_request_error'`` to model a relay's
                              non-retryable failed delivery.  Chat: header ``x-fixture-done-after-error`` appends
                              ``data: [DONE]`` after the error chunk.
      ``truncated``           stream ends after the content events with NO terminal event (non-stream: short body).
      ``malformed_tool_json`` tool-call argument text is cut so it no longer parses as JSON.
      ``failed``/``incomplete``  Responses wire only: ``response.failed`` / ``response.incomplete`` terminal
                              (``error_type`` becomes ``response.error.code`` of a failed terminal; default
                              ``server_error``, which OpenCode's message-regex retry matches - use e.g.
                              ``invalid_request`` to model a relay's sanitized, non-retryable failure).
      ``empty``               stream: 200 + SSE headers, then the connection closes with no event at all."""
    kind: str = 'http'
    status: int = 500
    message: str = 'fixture failure'
    error_type: str | None = None
    retry_after: float | None = None
    headers: dict = field(default_factory=dict)


class Gate:
    """Blocks a scripted response until the test calls ``release()``.  ``entered`` is set when the server has admitted
    the request and is waiting (the request is already recorded), so tests can observe "admitted, model blocked"."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self._released = threading.Event()

    def release(self) -> None:
        self._released.set()

    def wait_entered(self, timeout: float = 5.0) -> bool:
        return self.entered.wait(timeout)


@dataclass
class Step:
    """One scripted model response.

    text/thinking: assistant text and thinking text (``thinking_signature`` is the fake signature).
    tool_calls: list of ``ToolCall``.   stop_reason: neutral ``end_turn|tool_use|max_tokens|content_filter|
        stop_sequence|refusal`` (default ``tool_use`` if tool calls else ``end_turn``) or a wire-native value.
    usage: ``Usage``.   model: served model name (default: the request's ``model``).
    delay_s: sleep before responding.   chunk_delay_s: sleep between stream events.
    gate: ``True`` or a ``Gate``; the response waits until ``step.gate.release()``.
    keepalives: number of keepalive events after the stream head (Anthropic ``ping``; other wires SSE comments).
    keepalive_interval_s / keepalive_duration_s: after the response headers, hold the stream open for
        ``keepalive_duration_s`` seconds writing one keepalive frame every ``keepalive_interval_s`` seconds
        (0 = silence), then send the normal stream.  ``keepalive_before_start`` puts the hold before the first event
        (what a buffering relay does: it cannot know ``message_start`` yet); otherwise it follows the first event.
    keepalive_frame: raw bytes of one hold keepalive frame; default the wire's own (Anthropic ``event: ping``, other wires
        an SSE comment). Use it to test frames a client must tolerate or reject (e.g. a custom ``event:`` name).
    stream: force streaming (True) / non-streaming (False); default follows the request's ``stream`` field.
    failure: ``Failure`` to inject.   chunk_size: characters per text / argument delta.
    """
    text: str | None = None
    thinking: str | None = None
    thinking_signature: str = 'fake-signature-0001'
    tool_calls: list = field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    delay_s: float = 0.0
    chunk_delay_s: float = 0.0
    gate: Gate | bool | None = None
    keepalives: int = 0
    keepalive_interval_s: float = 0.0
    keepalive_duration_s: float = 0.0
    keepalive_before_start: bool = False
    keepalive_frame: bytes | None = None
    stream: bool | None = None
    failure: Failure | None = None
    chunk_size: int = 16

    def __post_init__(self) -> None:
        if self.gate is True:
            self.gate = Gate()
        self.tool_calls = [c if isinstance(c, ToolCall) else ToolCall(*c) for c in self.tool_calls]


def text_step(text: str, **kw: Any) -> Step:
    """Step answering with plain text."""
    return Step(text=text, **kw)


def tool_step(name: str, arguments: dict | str | None = None, *, id: str | None = None, text: str | None = None,
              **kw: Any) -> Step:
    """Step with a single tool call (``arguments`` dict or raw str) and optional leading text."""
    return Step(text=text, tool_calls=[ToolCall(name, {} if arguments is None else arguments, id)], **kw)


@dataclass
class Image:
    """An image found in a request body (base64 payload decoded)."""
    media_type: str
    data: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass
class ToolResult:
    """A tool result found in a request: ``call_id`` (tool_use_id / tool_call_id / call_id), joined ``text``,
    ``images`` inside the result, ``is_error`` (Anthropic only, else None), and the ``raw`` JSON node."""
    call_id: str | None
    text: str
    images: list
    is_error: bool | None
    raw: Any


class RecordedRequest:
    """A request as received.  ``path`` includes the query string, ``route`` does not, ``query`` is parsed.
    ``headers`` is case-insensitive (``request.headers['X-Api-Key']``).  ``raw_body`` is as sent, ``body`` is gunzipped
    when ``Content-Encoding: gzip``; ``json`` is the parsed body or None.  ``received_at`` is ``time.time()``.
    ``is_generation`` marks model-generation calls; ``response_status`` / ``step`` / ``client_disconnected`` are filled
    once the server has answered."""

    def __init__(self, method: str, path: str, headers: Any, raw_body: bytes) -> None:
        parts = urlsplit(path)
        self.method, self.path, self.route = method, path, parts.path
        self.query = parse_qs(parts.query)
        self.headers = headers
        self.raw_body = raw_body
        self.received_at = time.time()
        self.decode_error: str | None = None
        self.body = raw_body
        if str(headers.get('Content-Encoding', '')).lower() == 'gzip':
            try:
                self.body = gzip.decompress(raw_body)
            except (OSError, EOFError) as exc:
                self.decode_error = repr(exc)
        try:
            self.json = json.loads(self.body) if self.body else None
        except ValueError:
            self.json = None
        self.is_generation = False
        self.response_status: int | None = None
        self.step: Step | None = None
        self.client_disconnected = False
        self.keepalives_sent = 0

    def header(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name, default)


def _walk(node: Any) -> Iterable[Any]:
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _data_url(url: Any) -> Image | None:
    if isinstance(url, dict):
        url = url.get('url')
    if not isinstance(url, str) or not url.startswith('data:') or ';base64,' not in url:
        return None
    head, _, payload = url.partition(';base64,')
    try:
        return Image(head[5:] or 'application/octet-stream', base64.b64decode(payload, validate=True))
    except (binascii.Error, ValueError, TypeError):
        return None


def extract_images(node: Any) -> list[Image]:
    """All base64 images anywhere inside a JSON value: Anthropic ``image`` blocks (``source.type == base64``), Chat
    ``image_url`` parts and Responses ``input_image`` items carrying ``data:`` URLs.  Remote URLs are skipped."""
    found = []
    for item in _walk(node):
        if not isinstance(item, dict):
            continue
        kind, image = item.get('type'), None
        source = item.get('source')
        if kind == 'image' and isinstance(source, dict) and source.get('type') == 'base64':
            try:
                image = Image(source.get('media_type', 'application/octet-stream'),
                              base64.b64decode(source.get('data', ''), validate=True))
            except (binascii.Error, ValueError, TypeError):
                image = None
        elif kind in ('image_url', 'input_image'):
            image = _data_url(item.get('image_url'))
        if image is not None:
            found.append(image)
    return found


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    parts = []
    for part in content if isinstance(content, list) else []:
        if isinstance(part, dict) and isinstance(part.get('text'), str):
            parts.append(part['text'])
        elif isinstance(part, str):
            parts.append(part)
    return '\n'.join(parts)


def sse(event: str | None, data: Any) -> bytes:
    """One SSE frame; ``event=None`` omits the ``event:`` line (OpenAI Chat style)."""
    head = f'event: {event}\n' if event else ''
    return (head + 'data: ' + (data if isinstance(data, str) else json.dumps(data)) + '\n\n').encode()


def _pieces(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), max(1, size))]


class _Turn:
    """Resolved per-response facts shared by the wire renderers."""

    def __init__(self, n: int, step: Step, req: RecordedRequest) -> None:
        self.n, self.step, self.req = n, step, req
        body = req.json if isinstance(req.json, dict) else {}
        self.model = step.model or body.get('model') or 'fake-model'
        self.calls = []
        for i, call in enumerate(step.tool_calls):
            args = call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments)
            if step.failure and step.failure.kind == 'malformed_tool_json':
                args = args[:-1] if len(args) > 1 else '{'
            self.calls.append((call.id or f'call_fake_{n}_{i}', call.name, args))
        self.stop = step.stop_reason or ('tool_use' if step.tool_calls else 'end_turn')


class FakeModelServer:
    """Base: loopback ``ThreadingHTTPServer`` on an ephemeral port, used as a context manager.

    script: list of steps (see module docstring); mutable afterwards via ``server.script``.
    default: step served once the script is exhausted (default text "done"); may also be a callable
        ``request -> Step`` (so long-running CLIs can be scripted reactively without ``[fn] * N``).
    gate_timeout_s: safety cap on how long a gated response waits for ``release()``.
    local_routes: ``{(method, route): (status, json_body)}`` answered without consuming the script, e.g.
        ``GET /v1/models`` -> 404 by default; override per test.
    """
    wire = 'base'
    generation_routes: frozenset = frozenset()

    def __init__(self, script: Iterable = (), *, default: Step | None = None, gate_timeout_s: float = 30.0) -> None:
        self.script = list(script)
        self.default = default if default is not None else Step(text='done')
        self.gate_timeout_s = gate_timeout_s
        self.local_routes: dict = {('GET', '/v1/models'): (404, self.error_body(404, 'models endpoint not found'))}
        self._requests: list[RecordedRequest] = []
        self._cond = threading.Condition()
        self._generation_count = 0
        self._gates: list[Gate] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # lifecycle -----------------------------------------------------------------------------------------------------
    def start(self) -> 'FakeModelServer':
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                fake._handle(self)

            do_POST = do_PUT = do_DELETE = do_PATCH = do_GET

        class Httpd(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 64

        self._httpd = Httpd(('127.0.0.1', 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self.release_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(5)

    def __enter__(self) -> 'FakeModelServer':
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        """``http://127.0.0.1:<port>`` (no trailing slash, no ``/v1``)."""
        return f'http://127.0.0.1:{self.port}'

    def release_all(self) -> None:
        for gate in list(self._gates):
            gate.release()

    # recording and accessors ---------------------------------------------------------------------------------------
    @property
    def requests(self) -> list[RecordedRequest]:
        """Snapshot of every recorded request, in arrival order."""
        with self._cond:
            return list(self._requests)

    def generation_requests(self) -> list[RecordedRequest]:
        return [r for r in self.requests if r.is_generation]

    def last_json(self, generation: bool = True) -> Any:
        """Parsed body of the last (generation) request, or None."""
        pool = self.generation_requests() if generation else self.requests
        return pool[-1].json if pool else None

    def wait_for_requests(self, n: int, timeout: float = 5.0, *, generation: bool = True) -> list[RecordedRequest]:
        """Block until at least ``n`` (generation) requests were recorded; raise ``TimeoutError`` otherwise."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                pool = [r for r in self._requests if r.is_generation or not generation]
                if len(pool) >= n:
                    return pool
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'expected {n} requests, saw {len(pool)}')
                self._cond.wait(remaining)

    def images(self, request: RecordedRequest | None = None) -> list[Image]:
        """Decoded images anywhere in a request body (default: last generation request)."""
        request = request or self._last_generation()
        return extract_images(request.json) if request else []

    def find_tool_results(self, request: RecordedRequest | None = None) -> list[ToolResult]:
        """Tool results contained in a request's history (default: last generation request)."""
        request = request or self._last_generation()
        return self._tool_results(request.json) if request and isinstance(request.json, dict) else []

    def _last_generation(self) -> RecordedRequest | None:
        pool = self.generation_requests()
        return pool[-1] if pool else None

    def _tool_results(self, body: dict) -> list[ToolResult]:
        raise NotImplementedError

    # request handling ----------------------------------------------------------------------------------------------
    def _read_body(self, h: BaseHTTPRequestHandler) -> bytes:
        if 'chunked' in h.headers.get('Transfer-Encoding', '').lower():
            out = b''
            while True:
                size = int(h.rfile.readline().split(b';')[0].strip() or b'0', 16)
                if size == 0:
                    while h.rfile.readline() not in (b'\r\n', b'\n', b''):
                        pass
                    return out
                out += h.rfile.read(size)
                h.rfile.readline()
        return h.rfile.read(int(h.headers.get('Content-Length') or 0))

    def _handle(self, h: BaseHTTPRequestHandler) -> None:
        req = RecordedRequest(h.command, h.path, h.headers, self._read_body(h))
        req.is_generation = h.command == 'POST' and req.route in self.generation_routes
        with self._cond:
            self._requests.append(req)
            self._cond.notify_all()
        try:
            if req.is_generation:
                self._generate(h, req)
            else:
                self._local(h, req)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            req.client_disconnected = True
            h.close_connection = True

    def _send_json(self, h: BaseHTTPRequestHandler, req: RecordedRequest, status: int, body: Any,
                   headers: dict | None = None) -> None:
        data = json.dumps(body).encode()
        req.response_status = status
        h.send_response(status)
        h.send_header('Content-Type', 'application/json')
        h.send_header('Content-Length', str(len(data)))
        for key, value in (headers or {}).items():
            h.send_header(key, str(value))
        h.end_headers()
        h.wfile.write(data)

    def _local(self, h: BaseHTTPRequestHandler, req: RecordedRequest) -> None:
        status, body = self.local_routes.get((req.method, req.route), (404, self.error_body(404, 'not found')))
        self._send_json(h, req, status, body)

    def _take_step(self, req: RecordedRequest) -> tuple[int, Step]:
        with self._cond:
            n = self._generation_count
            self._generation_count += 1
            item = self.script[n] if n < len(self.script) else self.default
        if callable(item) and not isinstance(item, Step):
            item = item(req)
        if isinstance(item, str):
            item = Step(text=item)
        if isinstance(item.gate, Gate) and item.gate not in self._gates:
            self._gates.append(item.gate)
        return n, item

    def _generate(self, h: BaseHTTPRequestHandler, req: RecordedRequest) -> None:
        n, step = self._take_step(req)
        req.step = step
        if isinstance(step.gate, Gate):
            step.gate.entered.set()
            step.gate._released.wait(self.gate_timeout_s)
        if step.delay_s:
            time.sleep(step.delay_s)
        fail = step.failure
        if fail and fail.kind == 'http':
            headers = dict(fail.headers)
            if fail.status == 429 and fail.retry_after is not None:
                headers.setdefault('retry-after', fail.retry_after)
            return self._send_json(h, req, fail.status, self.error_body(fail.status, fail.message, fail.error_type),
                                   headers)
        turn = _Turn(n, step, req)
        body = req.json if isinstance(req.json, dict) else {}
        stream = step.stream if step.stream is not None else bool(body.get('stream'))
        if not stream:
            if fail and fail.kind == 'stream_error':
                return self._send_json(h, req, fail.status if fail.status >= 400 else 500,
                                       self.error_body(500, fail.message, fail.error_type))
            data = json.dumps(self.json_body(turn)).encode()
            req.response_status = 200
            h.send_response(200)
            h.send_header('Content-Type', 'application/json')
            h.send_header('Content-Length', str(len(data)))
            h.end_headers()
            h.wfile.write(data[:len(data) // 2] if fail and fail.kind == 'truncated' else data)
            h.close_connection = True
            return
        head, content, terminal, error = self.stream_parts(turn)
        if fail and fail.kind == 'empty':
            head = content = terminal = []
        if fail and fail.kind == 'truncated':
            frames = head + content
        elif fail and fail.kind == 'stream_error':
            frames = head + content[:len(content) // 2] + error
        else:
            frames = head + content + terminal
        req.response_status = 200
        h.send_response(200)
        h.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        h.send_header('Cache-Control', 'no-cache')
        h.send_header('Connection', 'close')
        h.end_headers()
        h.wfile.flush()
        if step.keepalive_duration_s > 0:
            if not step.keepalive_before_start and frames:
                h.wfile.write(frames[0])
                h.wfile.flush()
                frames = frames[1:]
            self._hold(h, req, step)
        for frame in frames:
            h.wfile.write(frame)
            h.wfile.flush()
            if step.chunk_delay_s:
                time.sleep(step.chunk_delay_s)
        h.close_connection = True

    def _hold(self, h: BaseHTTPRequestHandler, req: RecordedRequest, step: Step) -> None:
        end = time.monotonic() + step.keepalive_duration_s
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(step.keepalive_interval_s or remaining, remaining))
            if step.keepalive_interval_s and time.monotonic() < end:
                h.wfile.write(step.keepalive_frame if step.keepalive_frame is not None else self.keepalive_frame())
                h.wfile.flush()
                req.keepalives_sent += 1

    # wire hooks ----------------------------------------------------------------------------------------------------
    def keepalive_frame(self) -> bytes:
        """One keepalive frame of this wire (SSE comment by default)."""
        return b': keepalive\n\n'

    def error_body(self, status: int, message: str, error_type: str | None = None) -> dict:
        raise NotImplementedError

    def json_body(self, turn: _Turn) -> dict:
        raise NotImplementedError

    def stream_parts(self, turn: _Turn) -> tuple[list, list, list, list]:
        """Return SSE frames ``(head, content, terminal, error)``: ``error`` is the mid-stream error frame list."""
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------------------
class FakeAnthropicServer(FakeModelServer):
    """Anthropic Messages wire.  ``POST /v1/messages`` (any query string recorded), ``POST /v1/messages/count_tokens``
    -> ``{"input_tokens": count_tokens_value}`` (int, or callable ``request -> int``), ``GET /v1/models`` -> 404."""
    wire = 'anthropic'
    generation_routes = frozenset({'/v1/messages'})
    ERROR_TYPES = {400: 'invalid_request_error', 401: 'authentication_error', 403: 'permission_error',
                   404: 'not_found_error', 413: 'request_too_large', 429: 'rate_limit_error', 500: 'api_error',
                   529: 'overloaded_error'}

    def __init__(self, script: Iterable = (), *, count_tokens_value: int | Callable = 10, **kw: Any) -> None:
        super().__init__(script, **kw)
        self.count_tokens_value = count_tokens_value

    def keepalive_frame(self) -> bytes:
        return sse('ping', {'type': 'ping'})

    def error_body(self, status: int, message: str, error_type: str | None = None) -> dict:
        kind = error_type or self.ERROR_TYPES.get(status, 'api_error')
        return {'type': 'error', 'error': {'type': kind, 'message': message}}

    def _local(self, h: BaseHTTPRequestHandler, req: RecordedRequest) -> None:
        if req.method == 'POST' and req.route == '/v1/messages/count_tokens':
            value = self.count_tokens_value
            return self._send_json(h, req, 200, {'input_tokens': value(req) if callable(value) else value})
        super()._local(h, req)

    def _usage(self, u: Usage, output: int | None = None) -> dict:
        return {'input_tokens': u.input_tokens, 'cache_creation_input_tokens': u.cache_creation_input_tokens,
                'cache_read_input_tokens': u.cache_read_input_tokens,
                'output_tokens': u.output_tokens if output is None else output}

    def _blocks(self, turn: _Turn) -> list[dict]:
        blocks = []
        if turn.step.thinking is not None:
            blocks.append({'type': 'thinking', 'thinking': turn.step.thinking,
                           'signature': turn.step.thinking_signature})
        if turn.step.text:
            blocks.append({'type': 'text', 'text': turn.step.text})
        for call_id, name, args in turn.calls:
            try:
                parsed = json.loads(args)
            except ValueError:
                parsed = {}
            blocks.append({'type': 'tool_use', 'id': call_id, 'name': name, 'input': parsed})
        return blocks

    def _message(self, turn: _Turn, content: list, stop: str | None, usage: dict) -> dict:
        return {'id': f'msg_fake_{turn.n}', 'type': 'message', 'role': 'assistant', 'model': turn.model,
                'content': content, 'stop_reason': stop, 'stop_sequence': None, 'usage': usage}

    def json_body(self, turn: _Turn) -> dict:
        return self._message(turn, self._blocks(turn), turn.stop, self._usage(turn.step.usage))

    def stream_parts(self, turn: _Turn) -> tuple[list, list, list, list]:
        step, u = turn.step, turn.step.usage
        head = [sse('message_start', {'type': 'message_start', 'message': self._message(
            turn, [], None, self._usage(u, output=1))})]
        head += [sse('ping', {'type': 'ping'}) for _ in range(step.keepalives)]
        body, index = [], 0

        def block(start: dict, deltas: list[dict]) -> None:
            nonlocal index
            body.append(sse('content_block_start', {'type': 'content_block_start', 'index': index,
                                                    'content_block': start}))
            body.extend(sse('content_block_delta', {'type': 'content_block_delta', 'index': index, 'delta': d})
                        for d in deltas)
            body.append(sse('content_block_stop', {'type': 'content_block_stop', 'index': index}))
            index += 1

        if step.thinking is not None:
            block({'type': 'thinking', 'thinking': ''},
                  [{'type': 'thinking_delta', 'thinking': p} for p in _pieces(step.thinking, step.chunk_size)]
                  + [{'type': 'signature_delta', 'signature': step.thinking_signature}])
        if step.text:
            block({'type': 'text', 'text': ''},
                  [{'type': 'text_delta', 'text': p} for p in _pieces(step.text, step.chunk_size)])
        for call_id, name, args in turn.calls:
            block({'type': 'tool_use', 'id': call_id, 'name': name, 'input': {}},
                  [{'type': 'input_json_delta', 'partial_json': p} for p in _pieces(args, step.chunk_size)])
        terminal = [sse('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': turn.stop,
                                                                              'stop_sequence': None},
                                          'usage': {'output_tokens': u.output_tokens}}),
                    sse('message_stop', {'type': 'message_stop'})]
        fail = step.failure
        error = [sse('error', self.error_body(529, fail.message if fail else 'fixture stream error',
                                              fail.error_type if fail else None))]
        return head, body, terminal, error

    def _tool_results(self, body: dict) -> list[ToolResult]:
        found = []
        for message in body.get('messages') or []:
            content = message.get('content') if isinstance(message, dict) else None
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get('type') == 'tool_result':
                    found.append(ToolResult(block.get('tool_use_id'), _text_of(block.get('content', '')),
                                            extract_images(block.get('content')), bool(block.get('is_error')), block))
        return found


# ---------------------------------------------------------------------------------------------------------------------
class _OpenAiErrors:
    ERROR_TYPES = {400: ('invalid_request_error', None), 401: ('invalid_request_error', 'invalid_api_key'),
                   403: ('permission_error', None), 404: ('not_found_error', None),
                   413: ('invalid_request_error', 'request_too_large'),
                   429: ('rate_limit_error', 'rate_limit_exceeded'), 500: ('server_error', None),
                   503: ('server_error', None)}

    def error_body(self, status: int, message: str, error_type: str | None = None) -> dict:
        kind, code = self.ERROR_TYPES.get(status, ('server_error', None))
        return {'error': {'message': message, 'type': error_type or kind, 'param': None, 'code': code}}


class FakeChatServer(_OpenAiErrors, FakeModelServer):
    """OpenAI Chat Completions wire.  ``POST /v1/chat/completions`` (also ``/chat/completions``).  The final usage
    chunk (``choices: []``) is sent only when the request has ``stream_options.include_usage``."""
    wire = 'chat'
    generation_routes = frozenset({'/v1/chat/completions', '/chat/completions'})
    FINISH = {'end_turn': 'stop', 'stop_sequence': 'stop', 'tool_use': 'tool_calls', 'max_tokens': 'length',
              'refusal': 'content_filter'}

    def _usage(self, u: Usage) -> dict:
        prompt = u.input_tokens + u.cache_read_input_tokens
        return {'prompt_tokens': prompt, 'completion_tokens': u.output_tokens, 'total_tokens': prompt + u.output_tokens,
                'prompt_tokens_details': {'cached_tokens': u.cache_read_input_tokens},
                'completion_tokens_details': {'reasoning_tokens': u.reasoning_tokens}}

    def _chunk(self, turn: _Turn, delta: dict, finish: str | None = None) -> dict:
        return {'id': f'chatcmpl-fake-{turn.n}', 'object': 'chat.completion.chunk', 'created': int(time.time()),
                'model': turn.model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}

    def json_body(self, turn: _Turn) -> dict:
        step = turn.step
        message: dict = {'role': 'assistant', 'content': step.text}
        if step.thinking is not None:
            message['reasoning_content'] = step.thinking
        if turn.calls:
            message['tool_calls'] = [{'id': i, 'type': 'function', 'function': {'name': n, 'arguments': a}}
                                     for i, n, a in turn.calls]
        return {'id': f'chatcmpl-fake-{turn.n}', 'object': 'chat.completion', 'created': int(time.time()),
                'model': turn.model, 'choices': [{'index': 0, 'message': message,
                                                  'finish_reason': self.FINISH.get(turn.stop, turn.stop)}],
                'usage': self._usage(step.usage)}

    def stream_parts(self, turn: _Turn) -> tuple[list, list, list, list]:
        step = turn.step
        head = [sse(None, self._chunk(turn, {'role': 'assistant', 'content': ''}))]
        head += [b': keepalive\n\n' for _ in range(step.keepalives)]
        body = []
        if step.thinking is not None:
            body += [sse(None, self._chunk(turn, {'reasoning_content': p})) for p in _pieces(step.thinking,
                                                                                            step.chunk_size)]
        if step.text:
            body += [sse(None, self._chunk(turn, {'content': p})) for p in _pieces(step.text, step.chunk_size)]
        for index, (call_id, name, args) in enumerate(turn.calls):
            body.append(sse(None, self._chunk(turn, {'tool_calls': [{'index': index, 'id': call_id, 'type': 'function',
                                                                      'function': {'name': name, 'arguments': ''}}]})))
            body += [sse(None, self._chunk(turn, {'tool_calls': [{'index': index, 'function': {'arguments': p}}]}))
                     for p in _pieces(args, step.chunk_size)]
        terminal = [sse(None, self._chunk(turn, {}, self.FINISH.get(turn.stop, turn.stop)))]
        stream_options = (turn.req.json or {}).get('stream_options') or {}
        if stream_options.get('include_usage'):
            usage_chunk = self._chunk(turn, {})
            usage_chunk.update(choices=[], usage=self._usage(step.usage))
            terminal.append(sse(None, usage_chunk))
        terminal.append(sse(None, '[DONE]'))
        fail = step.failure
        error = [sse(None, self.error_body(500, fail.message if fail else 'fixture stream error',
                                           fail.error_type if fail else None))]
        if fail and fail.headers.get('x-fixture-done-after-error'):
            error.append(sse(None, '[DONE]'))       # the relay's documented Chat failed-delivery shape
        return head, body, terminal, error

    def _tool_results(self, body: dict) -> list[ToolResult]:
        return [ToolResult(m.get('tool_call_id'), _text_of(m.get('content', '')), extract_images(m.get('content')),
                           None, m)
                for m in body.get('messages') or [] if isinstance(m, dict) and m.get('role') == 'tool']


class FakeResponsesServer(_OpenAiErrors, FakeModelServer):
    """OpenAI Responses wire.  Generation: ``POST /responses`` and ``/v1/responses``.  Compaction
    (``/responses/compact``, ``/v1/responses/compact``) is recorded as a non-generation request and answered with
    ``compact_response`` (default: empty ``response.compaction``)."""
    wire = 'responses'
    generation_routes = frozenset({'/responses', '/v1/responses'})
    COMPACT_ROUTES = ('/responses/compact', '/v1/responses/compact')

    def __init__(self, script: Iterable = (), *, compact_response: dict | None = None, **kw: Any) -> None:
        super().__init__(script, **kw)
        self.compact_response = compact_response or {'object': 'response.compaction', 'output': [],
                                                     'usage': {'input_tokens': 3, 'output_tokens': 1}}

    def _local(self, h: BaseHTTPRequestHandler, req: RecordedRequest) -> None:
        if req.method == 'POST' and req.route in self.COMPACT_ROUTES:
            return self._send_json(h, req, 200, self.compact_response)
        super()._local(h, req)

    def _usage(self, u: Usage) -> dict:
        total_in = u.input_tokens + u.cache_read_input_tokens
        return {'input_tokens': total_in, 'input_tokens_details': {'cached_tokens': u.cache_read_input_tokens},
                'output_tokens': u.output_tokens, 'output_tokens_details': {'reasoning_tokens': u.reasoning_tokens},
                'total_tokens': total_in + u.output_tokens}

    def _items(self, turn: _Turn) -> list[dict]:
        step, n, items = turn.step, turn.n, []
        if step.thinking is not None:
            items.append({'type': 'reasoning', 'id': f'rs_fake_{n}', 'encrypted_content': step.thinking_signature,
                          'summary': [{'type': 'summary_text', 'text': step.thinking}]})
        if step.text:
            items.append({'type': 'message', 'id': f'msg_fake_{n}', 'role': 'assistant', 'status': 'completed',
                          'content': [{'type': 'output_text', 'text': step.text, 'annotations': []}]})
        for i, (call_id, name, args) in enumerate(turn.calls):
            items.append({'type': 'function_call', 'id': f'fc_fake_{n}_{i}', 'call_id': call_id, 'name': name,
                          'arguments': args, 'status': 'completed'})
        return items

    def _response(self, turn: _Turn, status: str, output: list) -> dict:
        response = {'id': f'resp_fake_{turn.n}', 'object': 'response', 'created_at': int(time.time()),
                    'status': status, 'model': turn.model, 'output': output}
        if status != 'in_progress':
            response['usage'] = self._usage(turn.step.usage)
        return response

    def _final_status(self, turn: _Turn) -> tuple[str, str]:
        kind = turn.step.failure.kind if turn.step.failure else None
        if kind == 'failed':
            return 'failed', 'response.failed'
        if kind == 'incomplete' or turn.stop == 'max_tokens':
            return 'incomplete', 'response.incomplete'
        return 'completed', 'response.completed'

    def json_body(self, turn: _Turn) -> dict:
        status, _ = self._final_status(turn)
        body = self._response(turn, status, self._items(turn))
        if status == 'incomplete':
            body['incomplete_details'] = {'reason': 'max_output_tokens'}
        if status == 'failed':
            body['error'] = {'code': turn.step.failure.error_type or 'server_error',
                             'message': turn.step.failure.message}
        return body

    def stream_parts(self, turn: _Turn) -> tuple[list, list, list, list]:
        step, seq = turn.step, [0]

        def ev(kind: str, **value: Any) -> bytes:
            seq[0] += 1
            return sse(kind, {'type': kind, 'sequence_number': seq[0], **value})

        head = [ev('response.created', response=self._response(turn, 'in_progress', []))]
        head += [b': keepalive\n\n' for _ in range(step.keepalives)]
        body = []
        for index, item in enumerate(self._items(turn)):
            if item['type'] == 'message':
                text = item['content'][0]['text']
                empty = {**item, 'status': 'in_progress', 'content': []}
                part = {'type': 'output_text', 'text': '', 'annotations': []}
                body += [ev('response.output_item.added', output_index=index, item=empty),
                         ev('response.content_part.added', output_index=index, item_id=item['id'], content_index=0,
                            part=part)]
                body += [ev('response.output_text.delta', output_index=index, item_id=item['id'], content_index=0,
                            delta=p) for p in _pieces(text, step.chunk_size)]
                body += [ev('response.output_text.done', output_index=index, item_id=item['id'], content_index=0,
                            text=text),
                         ev('response.content_part.done', output_index=index, item_id=item['id'], content_index=0,
                            part=item['content'][0])]
            elif item['type'] == 'function_call':
                body.append(ev('response.output_item.added', output_index=index,
                               item={**item, 'arguments': '', 'status': 'in_progress'}))
                body += [ev('response.function_call_arguments.delta', output_index=index, item_id=item['id'],
                            delta=p) for p in _pieces(item['arguments'], step.chunk_size)]
                body.append(ev('response.function_call_arguments.done', output_index=index, item_id=item['id'],
                               arguments=item['arguments']))
            else:
                body.append(ev('response.output_item.added', output_index=index, item={**item, 'summary': []}))
            body.append(ev('response.output_item.done', output_index=index, item=item))
        status, kind = self._final_status(turn)
        final = self._response(turn, status, self._items(turn))
        if status == 'incomplete':
            final['incomplete_details'] = {'reason': 'max_output_tokens'}
        if status == 'failed':
            final['output'] = []
            final['error'] = {'code': step.failure.error_type or 'server_error', 'message': step.failure.message}
        terminal = [ev(kind, response=final)]
        fail = step.failure
        error = [ev('error', code='server_error', message=fail.message if fail else 'fixture stream error',
                    param=None)]
        return head, body, terminal, error

    def _tool_results(self, body: dict) -> list[ToolResult]:
        items = body.get('input')
        return [ToolResult(i.get('call_id'), _text_of(i.get('output', '')), extract_images(i.get('output')), None, i)
                for i in (items if isinstance(items, list) else [])
                if isinstance(i, dict) and i.get('type') in ('function_call_output', 'custom_tool_call_output')]
