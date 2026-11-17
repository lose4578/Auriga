"""Buffered model relay: no request deadline, bounded provider-error retries.

Actor, helper and compact completions share the budget. Attempts and unknown usage
are private evidence; no partial model/tool output is released before validation.
Generic policy lives here; everything that depends on the provider wire format
(routes, SSE terminals, error tables, images, denial payloads) is a ``WireAdapter``
(``wire/``; default ``OpenAIResponsesWire``, the Codex Responses wire). The adapter's client
profile decides the delivery form (early 200 + keepalive, or deferred delivery) and whether the
client-retry runaway guard and the consume guard run; see ``wire.base.ClientProfile``.
"""
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import NamedTuple
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

if __package__:
    from ..agents.context_signal import MAIN as MAIN_CALL, ContinuityTracker
    from .wire import OpenAIResponsesWire, WireAdapter, wire_from_name
else:  # ``--http-worker`` runs this file as a script: import the adapters of this same checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from auriga.cortex.agents.context_signal import MAIN as MAIN_CALL, ContinuityTracker
    from auriga.cortex.session.wire import OpenAIResponsesWire, WireAdapter, wire_from_name
CODEX_PROFILE_NAME = "codex"
# Discontinuity reason when the provider reports a server-side context edit on a delivered main answer
# (``WireAdapter.context_edit_applied``); the other reasons come from ``ContinuityTracker``.
PROVIDER_CONTEXT_EDIT = "provider_context_edit"

HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length"}
# X-Auriga-Thread: native thread key of a host-side client (the relay evidence client), read before the
# wire's own thread key; like every private header it is never forwarded upstream.
PRIVATE_HEADERS = {"x-auriga-deadline-unix-ms", "x-auriga-source", "x-auriga-request-id", "x-auriga-thread"}
AUTH_HEADERS = dict(OpenAIResponsesWire.auth_header_names)
CLIENT_AUTH_HEADERS = set(AUTH_HEADERS) | {"cookie", "x-api-key", "api-key", "openai-organization", "openai-project"}
DEFAULT_MAX_RETRIES = 12  # Pinned RoboProbe joint-policy default; excludes the first attempt.
RETRY_POLICY = "bounded_harness_retry_after_explicit_retryable_provider_error"
TRANSPORT_RECOVERY_KEYS = frozenset(("max_rounds", "initial_delay_s", "backoff_multiplier", "max_delay_s",
                                     "max_episode_delay_s", "transport_incomplete"))
MIN_TRIM_IMAGE_LIMIT = 24
MAX_CLIENT_BODY_BYTES = 128 * 1024 * 1024  # declared client request body bound (bytes)
# Seconds a refused request's declared body may take to arrive while the relay drains it before
# answering (so the client reads the refusal instead of a broken pipe); then the answer is sent anyway.
REFUSAL_DRAIN_TIMEOUT_S = 30.0
# Seconds one auxiliary forward (Anthropic count_tokens) may take end to end; it is never retried.
AUX_REQUEST_TIMEOUT_S = 120.0
ACTOR_UPSTREAM = "actor"          # name of the default binding (constructor wire, URL and credentials)
UPSTREAM_PREFIX = "/_auriga/u/"   # path prefix of a named binding: /_auriga/u/<name><wire route>
UPSTREAM_NAME = re.compile(r"[a-z][a-z0-9_-]{0,31}")


class UpstreamBinding(NamedTuple):
    """One upstream the relay fronts (``ModelRequestProxy.add_upstream``).

    name: ``"actor"`` for the default binding, else the name given to ``add_upstream``.
    wire: the binding's ``WireAdapter`` (routes, terminals, errors, denials of its requests).
    upstream_url: upstream base URL (no trailing slash).
    auth_headers: host credential headers sent upstream (validated against
        ``wire.auth_header_names``), or None to forward none.
    client_auth_headers: lower-case header names stripped from client requests before the
        binding's credentials are added.
    presentation: ``ActorPresentation`` the egress projection of a named binding uses (None for
        the default binding, whose projection is the ``request_projection`` hook).
    prefix: path prefix of the binding on the relay (``""`` for the default binding).
    """
    name: str
    wire: object
    upstream_url: str
    auth_headers: object
    client_auth_headers: frozenset
    presentation: object = None
    prefix: str = ""


class IdleRun(NamedTuple):
    """Consume-guard state of one main thread (``ModelRequestProxy``, "Consume guard"), replaced on every
    delivered usable main completion of the thread.

    items: number of conversation items of that delivered request (``history_fingerprints`` length).
    last_blocks: number of content blocks of its last item (0 when it had no item).
    idle: how many delivered main completions in a row, ending with this one, were requests that added
        no client input to the history of the delivered request before them.
    """
    items: int
    last_blocks: int
    idle: int


class ContinuityAdmission(NamedTuple):
    """``ModelRequestProxy._observe_continuity`` result for one actor generation request.

    call_purpose: ``"main"``, ``"helper"`` or ``"subagent"`` (``context_signal``), committed with the request.
    idle: consume-guard verdict of a compared main request: True when it adds no client input after the
        thread's last delivered main completion, False when it adds some (or rewrote the history); None
        when it was not compared (guard off, not a main request, or no delivered main completion yet).
    refused: the thread's count of consecutive idle delivered completions when the consume guard refuses
        this request, else None.
    """
    call_purpose: str
    idle: bool | None
    refused: int | None


# Compatibility delegates below (module functions and ModelRequestProxy static/class methods)
# keep the c4ff16a names working; they all use this Responses adapter.
_RESPONSES = OpenAIResponsesWire()


def upstream_auth_headers_copy(headers, names=None):
    """Freeze host credentials without putting their values in validation errors.

    names: lower-case → canonical allowed credential header names; default ``AUTH_HEADERS``
    (the Responses wire). A relay passes its wire's ``auth_header_names``.
    """
    names = AUTH_HEADERS if names is None else names
    if headers is None:
        return None
    if not isinstance(headers, dict):
        raise ValueError("invalid_upstream_auth_headers")
    copied = {}
    for name, value in headers.items():
        key = names.get(name.lower()) if isinstance(name, str) else None
        if (key is None or key in copied or not isinstance(value, str) or not 0 < len(value) <= 16384
                or any(ord(char) < 32 or ord(char) > 126 for char in value)):
            raise ValueError("invalid_upstream_auth_headers")
        copied[key] = value
    return copied


def _request_target(url):
    """Request target for ``http.client``: path plus the query string the wire kept (e.g.
    Anthropic ``?beta=true``). Codex routes carry no query, so their target is the bare path."""
    return url.path + ("?" + url.query if url.query else "")


def _terminal_sse_type(block):
    """Compatibility delegate: ``OpenAIResponsesWire.terminal_sse_type``."""
    return _RESPONSES.terminal_sse_type(block)


def _terminal_sse_block(block):
    """Compatibility delegate: a Responses terminal SSE block (used by ``reference_api``)."""
    return _RESPONSES.terminal_sse_type(block) is not None


def _kill(process):
    if hasattr(process, "kill_inprocess"):
        process.kill_inprocess()
        return
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


_DEFAULT_POPEN = subprocess.Popen


class _InProcessAttempt:
    """进程内 HTTP 尝试句柄，保持与子进程句柄兼容的取消和状态查询接口。"""
    def __init__(self):
        self.lock = threading.Lock()
        self.conn = None
        self.killed = False
        self.done = threading.Event()
        self.returncode = None

    def set_connection(self, conn):
        with self.lock:
            self.conn = conn
            if self.killed and conn is not None:
                self._force_close(conn)

    @staticmethod
    def _force_close(conn):
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            conn.close()
        except OSError:
            pass

    def kill_inprocess(self):
        with self.lock:
            self.killed = True
            conn, self.conn = self.conn, None
        if conn is not None:
            self._force_close(conn)

    def poll(self):
        return self.returncode if self.done.is_set() else None

    def wait(self, timeout=None):
        self.done.wait(timeout)
        return self.returncode

    def communicate(self, *_args, **_kwargs):
        self.done.wait(0.2)
        return "", ""


def _http_worker():
    """One HTTP attempt in an independently killable process; stdout is a pipe.

    The JSON payload names the relay's wire (``"wire"``, default ``openai_responses``) and,
    for wires other than Codex Responses, its client profile (``"client"``); the wire decides
    where the buffered stream ends.
    """
    payload = json.load(sys.stdin)
    wire = wire_from_name(payload.get("wire", OpenAIResponsesWire.name), client=payload.get("client"))
    started = time.monotonic()
    def event(kind, **fields):
        print(json.dumps(dict(kind=kind, wall_time=time.time(), elapsed_s=time.monotonic()-started, **fields)), flush=True)
    url = urlsplit(payload["url"])
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=None)
    body = base64.b64decode(payload["body"])
    phase, incoming, first_byte = "connecting_upstream", 0, None
    response = None
    raw = bytearray()
    try:
        event("model_attempt_phase", phase=phase)
        connection.connect()
        event("model_upstream_connected")
        phase = "sending_upstream_body"
        event("model_attempt_phase", phase=phase)
        connection.request("POST", _request_target(url), body=body, headers=payload["headers"])
        event("model_request_forwarded", body_bytes=len(body), body_sha256=hashlib.sha256(body).hexdigest())
        phase = "awaiting_response_headers"
        event("model_attempt_phase", phase=phase)
        response = connection.getresponse()
        event("model_response_headers", http_status=response.status)
        phase = "streaming_response"
        event("model_attempt_phase", phase=phase)
        pending, terminal_seen = b"", False
        while chunk := response.read1(65536):
            event("model_response_bytes_received")
            raw.extend(chunk)
            incoming += len(chunk)
            if first_byte is None:
                first_byte = time.monotonic()-started
                event("model_response_first_byte")
            pending += chunk
            while match := re.search(br"\r?\n\r?\n", pending):
                block, pending = pending[:match.start()], pending[match.end():]
                if wire.terminal_sse_type(block) is not None:
                    terminal_seen = True
                    if pending:
                        del raw[-len(pending):]
                    break
            if terminal_seen:
                # A complete SSE terminal, not HTTP EOF, ends a Responses stream.
                event("model_response_terminal_received")
                break
        if not terminal_seen and response.length not in (None, 0):
            raise http.client.IncompleteRead(b"", response.length)
        result = dict(status=response.status, headers=dict(response.getheaders()), body=base64.b64encode(raw).decode(),
                      terminal_seen=terminal_seen)
    except Exception as error:
        result = dict(error_type=type(error).__name__, phase=phase, body=base64.b64encode(raw).decode(),
                      status=response.status if response is not None else None,
                      headers=dict(response.getheaders()) if response is not None else {})
    finally:
        connection.close()
    print(json.dumps(dict(result=result, incoming_bytes=incoming, first_upstream_byte_s=first_byte)), flush=True)


class ModelRequestProxy:
    """Loopback model relay owning the episode's generic model-call policy.

    wire: ``WireAdapter`` for the agent's provider format (default ``OpenAIResponsesWire()``,
        the Codex Responses wire); it decides routes, terminals, errors, images and denials.
    default_thread_id: thread id attributed to a request whose wire thread key is empty or
        unregistered (non-Codex sessions set their own thread id); default None keeps the
        Codex behaviour (attribution only by the ``thread-id`` header).
    on_context_discontinuity: optional callback ``(reason)`` called when the relay detects that the
        agent's model no longer holds context it was given (see "Continuity tracking"); default
        None (nothing is called, the relay-side effects still apply).
    All other keywords are unchanged from earlier versions.

    Continuity tracking (design §4.3 and Revision 4 R4.3; only for adapters whose client profile is
    not ``codex``, so the Codex relay is unchanged): ``continuity`` is a ``ContinuityTracker``. Its
    tracker inputs (``wire.continuity_inputs``) are computed once per actor-binding generation call,
    on the body after the ``request_context``/``request_projection`` hooks and before the relay's own
    image trim/reuse (a body the relay trimmed itself would look like a rewrite), and admitted under
    ``lock``. The tracker's thread is the native key the client sent (``X-Auriga-Thread``, else
    ``wire.thread_key``); only an empty key falls back to ``default_thread_id``. Attribution
    (``identity.native_thread_id``) is unchanged and still maps an unregistered key to
    ``default_thread_id``, but the tracker keeps such a key (an OpenCode child session) apart. Main
    threads are ``default_thread_id`` and every thread registered with source ``"actor"``; when the
    relay has neither, every thread is a main thread. The admission sets
    ``identity.call_purpose``: ``"helper"`` (no tools declared), ``"main"`` (the main chain of a main
    thread) or ``"subagent"`` (any other request with tools: a different system text on a main
    thread, or any thread that is not a main thread; see ``context_signal``). It is evidence only
    and never changes counting. Because the body is needed, ``call_purpose`` is carried by every
    event of the call recorded after the admission (``model_context_discontinuity``,
    ``model_context_discontinuity_callback_failed``, ``model_images_trimmed``,
    ``model_images_reused``, the per-attempt ``model_request_*`` events, ``model_request_retry``,
    ``model_transport_*``, ``model_client_did_not_consume_completion``, ``model_call_cancelled``,
    ``model_call_completed``, ``model_call_failed``, ``model_call_finished``), but never by
    ``model_call_started`` or ``model_request_body_received`` (recorded before the body is
    decoded), and not by a call refused or failed before the admission
    (runaway denial, undecodable body). Named bindings and Codex calls carry no ``call_purpose``.
    Only main requests are compared. A main request's baseline advances (``commit`` with the same
    inputs and its admitted purpose) only when that request's completion was delivered to the client
    as usable. Helper and subagent requests never report a discontinuity and never move a baseline,
    so a subagent's own compaction never starts an image-context epoch. A discontinuity of a main
    request (``history_shortened``, ``history_rewritten``, ``images_removed``, ``system_changed``, or
    ``provider_context_edit`` when its usable answer reports a server-side context edit) has the
    synchronous effects of a Responses ``/responses/compact`` admission: ``image_context_epoch``
    becomes the call's ``logical_call_id``, the thread's image-trim window is reset, and one
    ``model_context_discontinuity`` event (identity + ``reason``) is recorded, all under ``lock``.
    Then, still before the request is forwarded but outside ``lock`` (the callback may take its
    session's own lock), ``on_context_discontinuity(reason)`` is called; a callback exception is
    recorded as ``model_context_discontinuity_callback_failed`` and never fails the request.

    Routes come from ``wire.route``: generation routes take the full budgeted path; auxiliary
    routes (Anthropic ``count_tokens``) are projected with ``request_projection`` (refused with
    ``wire.relay_error`` when no projection hook is set: a raw body never leaves the host),
    forwarded once with host credentials and an ``AUX_REQUEST_TIMEOUT_S`` deadline (``close()``
    interrupts it), never charged or retried, and recorded as one ``model_aux_request`` event;
    an upstream HTTP error is answered with the sanitised ``wire.relay_error``, never passed
    through. Local/unknown routes get ``wire.not_found()``. Every refusal (denials, unknown
    routes) first drains the declared request body (at most ``MAX_CLIENT_BODY_BYTES``, for at
    most ``REFUSAL_DRAIN_TIMEOUT_S`` seconds) so a client still writing a large body reads the
    refusal instead of a broken pipe.

    Cancellation tombstones on non-``codex`` client profiles: ``cancel_pending(reason)`` also
    tombstones the native thread of every call it cancels (as ``cancel_thread`` does), so a client
    that re-sends after the cancellation gets ``wire.denial("thread_cancelled")`` instead of a new
    forwarded and charged call. Such a tombstone is lifted by ``reopen_thread(thread_id)`` or by
    ``register_thread`` of the same thread (the session does this when it starts its next turn);
    ``cancel_thread`` tombstones stay permanent. ``cancel_pending(reason, threads=[...])`` puts the
    same reopenable tombstone on further threads with no call in flight (a session's own threads at
    an interrupt). The Codex relay keeps its behaviour (no tombstone from ``cancel_pending``).

    Upstream bindings (design §4.1 "Upstream bindings"): the constructor's wire, URL and
    credentials are the ``"actor"`` binding (unprefixed routes). ``add_upstream`` adds named
    bindings served under ``/_auriga/u/<name>`` so that a host-side helper on another wire or
    credential (the relay evidence client) shares this relay's budget, event stream, retry and
    recovery policy and blocked-transport gate. A named binding's requests use only its own wire,
    URL and credentials, always pass ``binding.wire.project_request`` with the binding's
    presentation (the ``request_context`` / ``request_projection`` hooks, ``service_tier``,
    continuity tracking and image trim/reuse belong to the actor binding only), and carry
    ``identity.upstream = <name>`` (actor-binding events carry no ``upstream`` key, so Codex events
    are unchanged). A host-side client names its native thread with the private header
    ``X-Auriga-Thread`` (read before the wire's thread key; never forwarded).

    Client-retry runaway guard (only for adapters whose profile sets ``client_loop_guard``):
    transport protection, not task logic. OpenCode re-sent one generation body 45-970 times
    within 1-14 s (about 10 ms apart) when a stream ended without a terminal (spike S6.9); no
    supported client re-sends a byte-identical generation body more than a few times (Claude
    Code's 400 beta-downgrade re-sends are at most two, and every delivered answer changes the
    next body). When a thread's consecutive generation bodies stay byte-identical for more than
    ``RUNAWAY_MAX_IDENTICAL_ADMISSIONS`` admissions within ``RUNAWAY_WINDOW_S`` seconds, the
    transport is blocked (``error_type="ClientRetryRunaway"``, operator may authorise a manual
    retry) and that request gets ``wire.denial("client_retry_runaway")`` without being forwarded
    or charged. The same profiles get ``SHUTDOWN_TERMINAL_GRACE_S`` seconds on ``close()`` for
    every open client stream to receive its ``relay_error`` before its socket is closed.

    Consume guard (same profiles; transport protection, not task logic): an agent client asks the
    model again only when it has something new for it, a tool result or a user/steer message, so a
    main request (see "Continuity tracking") whose history adds nothing but the model's own earlier
    output (``wire.history_model_output_flags``) to the history of the thread's last delivered main
    completion is *idle*: the client did not act on the completion it was given. Observed (T12, opencode
    1.19.5): when the provider's usage holds a value the client's SDK cannot parse (``input_tokens:
    null``), every answer is validated, delivered and charged, but OpenCode cannot finish the step and
    re-requests at once, on the Responses wire with the delivered text appended and no input (the whole
    180-call budget in about 4 s), on the Anthropic wire with the unchanged history after a backoff.
    Every such re-request has a different body, so the runaway guard cannot see it, and with a real
    provider it is paced by the provider's latency, so no rate window can either. The rule therefore
    counts content, not time: after ``CONSUME_GUARD_MAX_IDLE_COMPLETIONS`` delivered main completions in
    a row whose requests were idle, the next idle main request on the thread blocks the transport
    (``error_type="ClientDidNotConsumeCompletion"``, evidence ``classification=
    "client_did_not_consume_completion"`` with ``idle_completions``, event
    ``model_client_did_not_consume_completion``; an operator may authorise a manual retry) and gets the
    client-loop refusal ``wire.denial("client_retry_runaway")`` without being forwarded or charged. A
    request that adds a client item, appends blocks to the last client item, or has a continuity
    discontinuity (compaction, prune, ...) ends the run; helper and subagent requests are not compared. The
    state is one ``IdleRun`` per main thread, replaced together with the continuity baseline on each
    delivered usable main completion, so failed, denied or cancelled requests never count. Residuals: a
    client that appends a note of its own to every re-request is adding input and is not detected;
    a loop confined to a subagent chain is not compared (both stay bounded by the budget and the
    runaway guard).
    """
    RUNAWAY_MAX_IDENTICAL_ADMISSIONS = 8  # identical consecutive bodies tolerated on one thread
    RUNAWAY_WINDOW_S = 2.0                # seconds the identical admissions must fall within
    # Consume guard: idle delivered main completions in a row tolerated on one thread before the next idle
    # request is refused. Normal sessions produce none (0 of 386 compared main requests of the 89 captured
    # Claude Code / OpenCode sequences in tests/fixtures/continuity, and none in the conformance suites);
    # 3 leaves room for short legitimate continuations without new input (Anthropic ``pause_turn``, which only
    # server-side tools produce and those are removed with online access; a finish reason a client does not map)
    # and bounds an unparseable-answer loop at 4 charged completions (1 + 3) instead of the whole budget.
    CONSUME_GUARD_MAX_IDLE_COMPLETIONS = 3
    SHUTDOWN_TERMINAL_GRACE_S = 2.0       # seconds close() waits for cancelled handlers' terminals
    def __init__(self, upstream_url, *, max_requests=180, request_timeout_s=None, max_retries=None,
                 on_event=None, on_limit=None, metrics_path=None, image_reuse=False,
                 image_trim=True, request_projection=None, upstream_auth_headers=None, request_context=None,
                 max_inline_images=0, transport_recovery=None, recovery_gate=None,
                 wire=None, default_thread_id=None, on_context_discontinuity=None):
        if wire is not None and not isinstance(wire, WireAdapter):
            raise ValueError("model_proxy_wire_requires_adapter")
        self.wire = OpenAIResponsesWire() if wire is None else wire
        if default_thread_id is not None and (not isinstance(default_thread_id, str) or not default_thread_id
                                              or len(default_thread_id) > 256 or "\n" in default_thread_id
                                              or "\r" in default_thread_id):
            raise ValueError("model_proxy_invalid_default_thread_id")
        self.default_thread_id = default_thread_id
        self.service_tier = None
        if on_context_discontinuity is not None and not callable(on_context_discontinuity):
            raise ValueError("model_proxy_context_discontinuity_requires_callable")
        self.on_context_discontinuity = on_context_discontinuity
        self.continuity = None if self.wire.client == CODEX_PROFILE_NAME else ContinuityTracker()
        self._upstream_auth_headers = upstream_auth_headers_copy(upstream_auth_headers, self.wire.auth_header_names)
        self._client_auth_headers = frozenset(CLIENT_AUTH_HEADERS) | frozenset(self.wire.auth_header_names)
        self._check_upstream_url(upstream_url)
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("model_proxy_invalid_request_budget")
        if request_timeout_s is not None:
            raise ValueError("model_proxy_request_deadlines_disabled")
        from ...apps.reference.reference_api import upstream
        self.policy = upstream().joint
        self.max_retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= DEFAULT_MAX_RETRIES:
            raise ValueError("model_proxy_invalid_retry_budget")
        if type(image_reuse) is not bool:
            raise ValueError("model_proxy_image_reuse_requires_boolean")
        self.image_reuse = image_reuse
        if type(image_trim) is not bool:
            raise ValueError("model_proxy_image_trim_requires_boolean")
        self.image_trim = image_trim
        if type(max_inline_images) is not int or not (max_inline_images == 0 or max_inline_images >= MIN_TRIM_IMAGE_LIMIT):
            raise ValueError("model_proxy_invalid_inline_image_limit")
        self.max_inline_images = max_inline_images
        if transport_recovery is not None:
            if not isinstance(transport_recovery, dict) or transport_recovery.keys() != TRANSPORT_RECOVERY_KEYS:
                raise ValueError("model_proxy_invalid_transport_recovery")
            transport_recovery = dict(transport_recovery) if transport_recovery["max_rounds"] > 0 else None
        if recovery_gate is not None and not callable(recovery_gate):
            raise ValueError("model_proxy_recovery_gate_requires_callable")
        self.transport_recovery, self.recovery_gate = transport_recovery, recovery_gate
        if request_projection is not None and not callable(request_projection):
            raise ValueError("model_proxy_request_projection_requires_callable")
        self.request_projection = request_projection
        if request_context is not None and not callable(request_context):
            raise ValueError("model_proxy_request_context_requires_callable")
        self.request_context = request_context
        self.upstream_url = upstream_url.rstrip("/")
        self._upstreams = {}  # name -> UpstreamBinding of add_upstream (the actor binding is built on demand)
        self.max_requests, self.request_timeout_s = max_requests, None
        self.on_event, self.on_limit = on_event, on_limit
        self.metrics_path = Path(metrics_path) if metrics_path else None
        if self.metrics_path:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.touch(exist_ok=False)
            self.metrics_path.chmod(0o600)
        self.lock = threading.RLock()
        self.request_count = self.attempt_count = self.logical_call_count = self.reserved_calls = 0
        self.transport_id = uuid.uuid4().hex
        self.image_context_epoch = None
        self.limit_notified = False
        self.events, self.active = [], {}
        self.thread_owners, self.cancelled_threads = {}, {}
        self._image_trim_watermarks = {}
        self._image_trim_starts = {}
        self._runaway_runs = {}
        self._idle_runs = {}  # consume guard: tracker thread -> IdleRun (client_loop_guard profiles only)
        self._reopenable_threads = set()  # tombstones set by cancel_pending (non-Codex), liftable
        self._aux_connections = set()     # in-flight auxiliary upstream connections (close() interrupts)
        self.aux_request_count = 0
        self._recovery_rounds = self._recovered_calls = self._recovery_denials = 0
        self._recovery_delay_s = 0.0
        self._wait_started_at = None
        self._wait_total_s = 0.0
        self._last_activity_at = None
        self.closed = False
        self.blocked_error = None
        self.server = self.thread = self.base_url = None
        self.callbacks = queue.Queue()
        self.callback_thread = None
        self._conn_pool = {}
        self._conn_pool_lock = threading.Lock()

    @staticmethod
    def _check_upstream_url(upstream_url):
        url = urlsplit(upstream_url) if isinstance(upstream_url, str) else None
        if (url is None or url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("model_proxy_requires_http_base_url_without_credentials_or_query")

    @property
    def actor_binding(self):
        """The default ``UpstreamBinding`` (constructor wire, ``upstream_url`` and credentials). Built
        from the current attributes, so tests that repoint ``upstream_url`` keep working."""
        return UpstreamBinding(ACTOR_UPSTREAM, self.wire, self.upstream_url, self._upstream_auth_headers,
                               self._client_auth_headers)

    def add_upstream(self, name, *, wire, upstream_url, upstream_auth_headers, presentation):
        """Front one more upstream on this relay (host only, before the agent can reach it).

        name: binding name, ``[a-z][a-z0-9_-]{0,31}``, not ``"actor"``, unique.
        wire: the binding's ``WireAdapter`` instance (its own client profile).
        upstream_url: http(s) base URL without credentials, query or fragment.
        upstream_auth_headers: host credential headers for this upstream only (validated against
            ``wire.auth_header_names``; never sent to another binding), or None.
        presentation: the session's ``ActorPresentation``; every request of the binding passes
            ``wire.project_request(request, presentation, registered_tools=frozenset(),
            tool_name_prefix="")`` (no tools are declared on helper bindings).
        Returns the binding's base URL ``http://127.0.0.1:<port>/_auriga/u/<name>`` when the relay is
        started, else None (``upstream_base_url(name)`` gives it after ``start()``).
        """
        if not isinstance(name, str) or not UPSTREAM_NAME.fullmatch(name) or name == ACTOR_UPSTREAM:
            raise ValueError("model_proxy_invalid_upstream_name")
        if not isinstance(wire, WireAdapter):
            raise ValueError("model_proxy_wire_requires_adapter")
        if presentation is None:
            raise ValueError("model_proxy_upstream_requires_presentation")
        self._check_upstream_url(upstream_url)
        binding = UpstreamBinding(name, wire, upstream_url.rstrip("/"),
                                  upstream_auth_headers_copy(upstream_auth_headers, wire.auth_header_names),
                                  frozenset(CLIENT_AUTH_HEADERS) | frozenset(wire.auth_header_names),
                                  presentation, UPSTREAM_PREFIX + name)
        with self.lock:
            if self.closed:
                raise ValueError("model_proxy_already_started_or_closed")
            if name in self._upstreams:
                raise ValueError("model_proxy_upstream_already_bound")
            self._upstreams[name] = binding
        self._record("model_upstream_bound", upstream=name, wire_api=wire.name, relay_profile=wire.client,
                     credentials=sorted(binding.auth_headers or ()))
        return self.upstream_base_url(name)

    def upstream_base_url(self, name):
        """Relay base URL a host-side client uses for binding ``name`` (``"actor"``: the unprefixed
        ``base_url``), or None before ``start()``. Raises ``KeyError`` for an unknown binding."""
        if name != ACTOR_UPSTREAM:
            with self.lock:
                if name not in self._upstreams:
                    raise KeyError(name)
        if self.base_url is None:
            return None
        return self.base_url if name == ACTOR_UPSTREAM else self.base_url + UPSTREAM_PREFIX + name

    def upstream_binding(self, name):
        """``UpstreamBinding`` for ``name`` (``"actor"`` included); ``KeyError`` when unknown."""
        if name == ACTOR_UPSTREAM:
            return self.actor_binding
        with self.lock:
            return self._upstreams[name]

    def _binding_for_path(self, raw_path):
        """``(binding, path_within_binding)`` for a client request target; ``(None, None)`` for a
        ``/_auriga/u/`` path naming no binding."""
        if not isinstance(raw_path, str) or not raw_path.startswith(UPSTREAM_PREFIX):
            return self.actor_binding, raw_path
        name, slash, rest = raw_path[len(UPSTREAM_PREFIX):].partition("/")
        with self.lock:
            binding = self._upstreams.get(name)
        if binding is None or not slash:
            return None, None
        return binding, "/" + rest

    @staticmethod
    def _socket_alive(conn):
        sock = getattr(conn, "sock", None)
        if sock is None:
            return False
        try:
            readable, _, _ = select.select([sock], [], [], 0)
            return not readable
        except (OSError, ValueError):
            return False

    def _acquire_upstream_conn(self, url):
        key = (url.scheme, url.hostname, url.port)
        with self._conn_pool_lock:
            bucket = self._conn_pool.get(key)
            while bucket:
                conn = bucket.pop()
                if self._socket_alive(conn):
                    return conn, True
                try:
                    conn.close()
                except OSError:
                    pass
        connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        return connection_type(url.hostname, url.port, timeout=None), False

    def _release_upstream_conn(self, url, conn, response):
        if conn is None or response is None:
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
            return
        if getattr(response, "will_close", True):
            try:
                conn.close()
            except OSError:
                pass
            return
        sock = getattr(conn, "sock", None)
        if not response.isclosed() and sock is not None:
            try:
                sock.settimeout(0.005)
                response.read()
            except Exception:
                pass
            finally:
                try:
                    sock.settimeout(None)
                except OSError:
                    pass
        if not response.isclosed() or not self._socket_alive(conn):
            try:
                conn.close()
            except OSError:
                pass
            return
        key = (url.scheme, url.hostname, url.port)
        with self._conn_pool_lock:
            if self.closed:
                try:
                    conn.close()
                except OSError:
                    pass
                return
            self._conn_pool.setdefault(key, []).append(conn)

    def _close_conn_pool(self):
        with self._conn_pool_lock:
            buckets = list(self._conn_pool.values())
            self._conn_pool.clear()
        for bucket in buckets:
            for conn in bucket:
                try:
                    conn.close()
                except OSError:
                    pass

    def _inprocess_http(self, attempt, identity, url_str, body, headers, started, wire=None):
        wire = self.wire if wire is None else wire

        def emit(kind, **fields):
            elapsed = time.monotonic() - started
            if kind == "model_response_bytes_received":
                with self.lock:
                    self._last_activity_at = time.monotonic()
                return
            self._record(kind, **identity, elapsed_s=elapsed, **fields)

        url = urlsplit(url_str)
        req_headers = dict(headers)
        if not any(k.lower() == "connection" for k in req_headers):
            req_headers["Connection"] = "keep-alive"
        phase, incoming, first_byte = "connecting_upstream", 0, None
        response = None
        connection = None
        reused_ok = False
        raw = bytearray()
        try:
            emit("model_attempt_phase", phase=phase)
            if identity.get("attempt_index", 1) > 1:
                self._close_conn_pool()
            connection, was_reused = self._acquire_upstream_conn(url)
            attempt.set_connection(connection)
            if not was_reused:
                connection.connect()
            emit("model_upstream_connected")
            phase = "sending_upstream_body"
            emit("model_attempt_phase", phase=phase)
            forwarded_emitted = False
            awaiting_emitted = False
            try:
                connection.request("POST", _request_target(url), body=body, headers=req_headers)
                emit("model_request_forwarded", body_bytes=len(body), body_sha256=hashlib.sha256(body).hexdigest())
                forwarded_emitted = True
                phase = "awaiting_response_headers"
                emit("model_attempt_phase", phase=phase)
                awaiting_emitted = True
                response = connection.getresponse()
            except Exception:
                if not was_reused or attempt.killed:
                    raise
                try:
                    connection.close()
                except OSError:
                    pass
                phase = "connecting_upstream"
                connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
                connection = connection_type(url.hostname, url.port, timeout=None)
                attempt.set_connection(connection)
                connection.connect()
                phase = "sending_upstream_body"
                connection.request("POST", _request_target(url), body=body, headers=req_headers)
                if not forwarded_emitted:
                    emit("model_request_forwarded", body_bytes=len(body), body_sha256=hashlib.sha256(body).hexdigest())
                phase = "awaiting_response_headers"
                if not awaiting_emitted:
                    emit("model_attempt_phase", phase=phase)
                response = connection.getresponse()
            emit("model_response_headers", http_status=response.status)
            phase = "streaming_response"
            emit("model_attempt_phase", phase=phase)
            pending, terminal_seen, terminal_type = b"", False, None
            while chunk := response.read1(65536):
                emit("model_response_bytes_received")
                raw.extend(chunk)
                incoming += len(chunk)
                if first_byte is None:
                    first_byte = time.monotonic() - started
                    emit("model_response_first_byte")
                pending += chunk
                while match := re.search(br"\r?\n\r?\n", pending):
                    block, pending = pending[:match.start()], pending[match.end():]
                    if (t_type := wire.terminal_sse_type(block)) is not None:
                        terminal_seen = True
                        terminal_type = t_type
                        if pending:
                            del raw[-len(pending):]
                        break
                if terminal_seen:
                    emit("model_response_terminal_received")
                    break
            if not terminal_seen and response.length not in (None, 0):
                raise http.client.IncompleteRead(b"", response.length)
            result = dict(status=response.status, headers=dict(response.getheaders()),
                          body=base64.b64encode(raw).decode(), terminal_seen=terminal_seen)
            is_sse = wire.is_event_stream(raw)
            reused_ok = (not attempt.killed and 200 <= response.status < 400
                         and (wire.terminal_allows_reuse(terminal_type) if is_sse else not terminal_seen))
        except Exception as error:
            result = dict(error_type=type(error).__name__, phase=phase, body=base64.b64encode(raw).decode(),
                          status=response.status if response is not None else None,
                          headers=dict(response.getheaders()) if response is not None else {})
        finally:
            attempt.set_connection(None)
            if reused_ok:
                self._release_upstream_conn(url, connection, response)
            elif connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            attempt.returncode = 0
            attempt.done.set()
        return dict(result=result, incoming_bytes=incoming, first_upstream_byte_s=first_byte), phase

    def _record(self, kind, **fields):
        event = {"kind": kind, "wall_time": time.time(), "transport_schema_version": "2", **fields}
        with self.lock:
            if kind in {"model_call_started", "model_request_started", "model_request_body_received",
                        "model_upstream_connected", "model_request_forwarded", "model_response_headers",
                        "model_response_first_byte", "model_response_terminal_received",
                        "model_request_finished", "model_call_completed"}:
                self._last_activity_at = time.monotonic()
            if (kind == "model_call_started" and "upstream" not in fields
                    and self.wire.route_for_endpoint(fields.get("endpoint")).compaction):
                # Synchronous with admission; on_event callbacks run asynchronously.
                self.image_context_epoch = fields.get("logical_call_id") or uuid.uuid4().hex
                thread_id = fields.get("native_thread_id") or ""
                self._image_trim_watermarks.pop(thread_id, None)
                self._image_trim_starts.pop(thread_id, None)
            self.events.append(event)
            if self.metrics_path:
                with self.metrics_path.open("a") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
        if self.on_event:
            self.callbacks.put((self.on_event, event))
        return event

    def _callbacks(self):
        while (work := self.callbacks.get()) is not None:
            try:
                work[0](work[1])
            except Exception:
                pass  # Callback failure must not repeat or block a model request.

    @staticmethod
    def _shutdown(connection):
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass

    def _cancel(self, state, reason):
        with self.lock:
            if state["cancel"].is_set() or state["finished"].is_set():
                return
            if reason == "client_disconnected" and state.get("delivered"):
                # The whole response already reached the client; a helper closing its
                # connection before the call is marked finished is not a transport failure.
                return
            state["cancel_reason"] = reason
            state["cancel"].set()
            if reason == "client_disconnected":
                state["provider_explicit_error"] = False
                state["failure_evidence"] = dict(classification="downstream_disconnected",
                                                response_completion="unknown")
                self._block_after_failure(state, "ClientDisconnected")
            _kill(state.get("process"))
            if state.get("reading_client_body"):
                self._shutdown(state["client"])
        self._record("model_call_cancelled", **state["identity"], reason=reason)

    def cancel_pending(self, reason="episode_stopped", *, threads=()):
        """Cancel every in-flight model call. On non-``codex`` client profiles the native thread of
        each cancelled call is also tombstoned until ``reopen_thread`` / ``register_thread`` (class
        docstring), so a client re-send is denied rather than forwarded and charged.

        threads: further native thread ids (str) to tombstone the same way, reopenable, even when none
        of their calls is in flight; default none. Non-``codex`` profiles only (the Codex relay keeps
        its behaviour). An agent session passes its own threads when it interrupts, so a model call
        its client still makes after the native abort is denied too (OpenCode with a running subagent,
        probe R4-0)."""
        tombstoned = []
        with self.lock:
            active = list(self.active.values())
            if self.wire.client != CODEX_PROFILE_NAME:
                candidates = [state["identity"].get("native_thread_id") for state in active] + list(threads)
                for thread_id in candidates:
                    if isinstance(thread_id, str) and thread_id and thread_id not in self.cancelled_threads:
                        self.cancelled_threads[thread_id] = reason
                        self._reopenable_threads.add(thread_id)
                        tombstoned.append(thread_id)
        for state in active:
            self._cancel(state, reason)
        for thread_id in tombstoned:
            self._record("model_thread_cancelled", native_thread_id=thread_id, reason=reason, reopenable=True)

    def block_threads(self, threads, reason):
        """Put the reopenable ``cancel_pending`` tombstone on ``threads`` (native thread ids, str) only: in-flight
        calls of exactly these threads are cancelled and their later requests are denied until ``reopen_thread`` /
        ``register_thread``; calls of other threads (e.g. an evidence helper) are untouched. Non-``codex``
        profiles only (the Codex relay is unchanged). A session uses it to restore the tombstones it lifted for a
        turn the agent then definitely refused. Returns the list of threads newly tombstoned."""
        if self.wire.client == CODEX_PROFILE_NAME:
            return []
        tombstoned = []
        with self.lock:
            for thread_id in threads:
                if isinstance(thread_id, str) and thread_id and thread_id not in self.cancelled_threads:
                    self.cancelled_threads[thread_id] = reason
                    self._reopenable_threads.add(thread_id)
                    tombstoned.append(thread_id)
            active = [state for state in self.active.values()
                      if state["identity"].get("native_thread_id") in tombstoned]
        for state in active:
            self._cancel(state, reason)
        for thread_id in tombstoned:
            self._record("model_thread_cancelled", native_thread_id=thread_id, reason=reason, reopenable=True)
        return tombstoned

    def reopen_thread(self, thread_id):
        """Lift the tombstone ``cancel_pending`` put on ``thread_id`` (non-``codex`` profiles), so the
        thread's next requests are admitted again. Returns True when a tombstone was lifted; a
        ``cancel_thread`` tombstone is permanent and stays (False)."""
        with self.lock:
            if thread_id not in self._reopenable_threads:
                return False
            self._reopenable_threads.discard(thread_id)
            reason = self.cancelled_threads.pop(thread_id, None)
        self._record("model_thread_reopened", native_thread_id=thread_id, cancel_reason=reason)
        return True

    def register_thread(self, thread_id, source, request_id):
        """Attribute an official native thread; only the harness registers owners. Registering a
        thread lifts a ``cancel_pending`` tombstone on it (``reopen_thread``)."""
        if any(not isinstance(value, str) or not value or len(value) > limit or '\n' in value or '\r' in value
               for value, limit in ((thread_id, 256), (source, 128), (request_id, 128))):
            raise ValueError("model_thread_registration_invalid")
        owner = dict(source=source, caller_request_id=request_id)
        self.reopen_thread(thread_id)
        with self.lock:
            if thread_id in self.cancelled_threads:
                raise ValueError("model_thread_already_cancelled")
            if thread_id in self.thread_owners and self.thread_owners[thread_id] != owner:
                raise ValueError("model_thread_owner_conflict")
            self.thread_owners[thread_id] = owner
        self._record("model_thread_registered", native_thread_id=thread_id, **owner)

    def unregister_thread(self, thread_id):
        # Cancellation tombstones outlive attribution: turn_start may dispatch
        # its first HTTP request only after the caller has already cancelled.
        with self.lock:
            owner = self.thread_owners.pop(thread_id, None)
        if owner is not None:
            self._record("model_thread_unregistered", native_thread_id=thread_id, **owner)

    def cancel_thread(self, thread_id, reason="helper_turn_interrupted"):
        """Cancel this thread only and reject its future HTTP admissions."""
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("model_thread_cancellation_invalid")
        with self.lock:
            self.cancelled_threads.setdefault(thread_id, reason)
            self._reopenable_threads.discard(thread_id)  # an explicit thread cancellation is permanent
            for state in list(self.active.values()):
                if state["identity"].get("native_thread_id") == thread_id:
                    self._cancel(state, reason)
        self._record("model_thread_cancelled", native_thread_id=thread_id, reason=reason)

    @property
    def transport_blocked(self):
        with self.lock:
            return self.blocked_error is not None

    @property
    def pending_requests(self):
        with self.lock:
            return bool(self.active)

    @property
    def last_activity_at(self):
        """Latest actual request/transport activity on the monotonic clock, or None."""
        with self.lock:
            return self._last_activity_at

    @property
    def model_wait_seconds(self):
        """Union of intervals with any pending logical request, including now."""
        with self.lock:
            return self._wait_total_s + (time.monotonic() - self._wait_started_at
                                         if self._wait_started_at is not None else 0.0)

    @property
    def transport_recovery_summary(self):
        with self.lock:
            return dict(policy=dict(self.transport_recovery) if self.transport_recovery else None,
                        rounds=self._recovery_rounds, recovered_calls=self._recovered_calls,
                        denials=self._recovery_denials, cooldown_s=self._recovery_delay_s)

    def _recovery_delay(self, state, rounds, explicit_retryable):
        """Next cooldown before resending the same body, or None to keep the terminal path.

        A resend never reaches the actor or the robot; the gate refuses it unless no
        action is executing or unconfirmed and the episode still accepts actions.
        """
        policy = self.transport_recovery
        evidence = state.get("failure_evidence") or {}
        # Provider credit exhaustion was transient in practice; it waits like a transport failure but
        # never uses the immediate retries.
        if policy is None or not (explicit_retryable or evidence.get("classification") == "quota_exceeded" or (
                policy["transport_incomplete"] and evidence.get("classification") == "transport_incomplete")):
            return None
        delay = policy["initial_delay_s"]
        for _ in range(rounds):
            delay = min(policy["max_delay_s"], delay * policy["backoff_multiplier"])
        delay = min(delay, policy["max_delay_s"])
        reason = "recovery_rounds_exhausted" if rounds >= policy["max_rounds"] else self._recovery_gate_reason()
        if reason is None:
            with self.lock:
                if self._recovery_delay_s + delay > policy["max_episode_delay_s"]:
                    reason = "episode_recovery_budget_exhausted"
                else:
                    self._recovery_delay_s += delay
                    self._recovery_rounds += 1
        if reason is not None:
            self._deny_recovery(state, reason, rounds)
            return None
        return delay

    def _recovery_gate_reason(self):
        with self.lock:
            if self.closed:
                return "proxy_closed"
            if self.blocked_error is not None:
                return "transport_blocked"
        if self.recovery_gate is None:
            return "recovery_gate_unavailable"
        try:
            reason = self.recovery_gate()
        except Exception as error:
            return "recovery_gate_error:" + type(error).__name__
        return reason if reason is None or isinstance(reason, str) else "recovery_gate_invalid"

    def _deny_recovery(self, state, reason, rounds):
        with self.lock:
            self._recovery_denials += 1
        self._record("model_transport_recovery_denied", **state["identity"], reason=reason,
                     recovery_rounds=rounds, error=state.get("failure_evidence"))

    def _block_after_failure(self, state, error_type):
        if state["cancel"].is_set() and state.get("cancel_reason") != "client_disconnected":
            return
        error = dict(logical_call_id=state["identity"]["logical_call_id"], error_type=error_type,
                     manual_retry_allowed=bool(state.get("provider_explicit_error")),
                     evidence=state.get("failure_evidence"))
        with self.lock:
            if self.blocked_error is not None:
                return
            self.blocked_error = error
        self._record("model_transport_blocked", **state["identity"], error=error,
                     automatic_retries=self.max_retries, manual_retry_allowed=error["manual_retry_allowed"])

    def _runaway_admission(self, thread_id, body):
        """Count this generation body in the thread's run of byte-identical consecutive bodies.

        Returns the run length when it exceeds ``RUNAWAY_MAX_IDENTICAL_ADMISSIONS`` within
        ``RUNAWAY_WINDOW_S`` seconds, else None. A different body starts a new run.
        """
        digest = hashlib.sha256(body).digest()
        now = time.monotonic()
        with self.lock:
            previous, times = self._runaway_runs.get(thread_id, (None, ()))
            times = [t for t in times if now - t <= self.RUNAWAY_WINDOW_S] if previous == digest else []
            times.append(now)
            self._runaway_runs[thread_id] = (digest, times)
            return len(times) if len(times) > self.RUNAWAY_MAX_IDENTICAL_ADMISSIONS else None

    def _block_client_loop(self, state, classification, error_type, **facts):
        """Block the transport after a client re-request loop (runaway or consume guard). Nothing of
        this request was forwarded, so an operator may authorise a manual retry after review.

        Sets ``failure_evidence = {classification, **facts}`` and, unless the transport is already
        blocked, ``blocked_error`` (``error_type``, manual retry allowed); records
        ``model_<classification>`` (identity + ``facts``) and, for the first block,
        ``model_transport_blocked``."""
        evidence = dict(classification=classification, **facts)
        state["failure_evidence"] = evidence
        error = dict(logical_call_id=state["identity"]["logical_call_id"], error_type=error_type,
                     manual_retry_allowed=True, evidence=evidence)
        with self.lock:
            first = self.blocked_error is None
            if first:
                self.blocked_error = error
        self._record("model_" + classification, **state["identity"], **facts)
        if first:
            self._record("model_transport_blocked", **state["identity"], error=error,
                         automatic_retries=self.max_retries, manual_retry_allowed=True)

    def _block_runaway(self, state, count):
        """Block the transport after a tight identical-body retry loop (``_block_client_loop``)."""
        self._block_client_loop(state, "client_retry_runaway", "ClientRetryRunaway",
                                identical_admissions=count, window_s=self.RUNAWAY_WINDOW_S)

    def _block_unconsumed(self, state, idle):
        """Block the transport after ``idle`` delivered completions in a row that the client did not act
        on (class docstring, "Consume guard"; ``_block_client_loop``)."""
        self._block_client_loop(state, "client_did_not_consume_completion", "ClientDidNotConsumeCompletion",
                                idle_completions=idle, max_idle_completions=self.CONSUME_GUARD_MAX_IDLE_COMPLETIONS)

    def _idle_request(self, thread, admission, fingerprints, output_flags):
        """Consume-guard verdict of one admitted request (class docstring, "Consume guard"); call with
        ``lock`` held. True: a main request whose history continues the thread's last delivered main
        request and adds only model output; False: it adds a client item, appends blocks to the last
        client item, or has any discontinuity reason (``context_signal.REASONS``: a compaction, prune,
        image removal or system change is new context from the client); None: not compared (not a main
        request, no delivered main completion on the thread yet, or flags that do not match the
        fingerprints)."""
        run = self._idle_runs.get(thread)
        if run is None or admission.call_purpose != MAIN_CALL or len(output_flags) != len(fingerprints):
            return None
        if admission.reason is not None or len(fingerprints) < run.items:
            return False
        n = run.items
        if not all(output_flags[n:]):
            return False  # a new client item: user, system or developer message, tool result
        # Blocks appended to the last item (Claude Code merges steer text and tool results into it).
        return not (n and len(fingerprints[n - 1][1]) > run.last_blocks and not output_flags[n - 1])

    def _commit_idle_run(self, thread, call_purpose, fingerprints, idle):
        """Replace the thread's ``IdleRun`` with a delivered usable main request (no-op for other
        purposes); ``idle`` is that request's ``_idle_request`` verdict. Call with ``lock`` held, next to
        ``continuity.commit``."""
        if call_purpose != MAIN_CALL:
            return
        previous = self._idle_runs.get(thread)
        count = previous.idle + 1 if idle and previous is not None else 0
        self._idle_runs[thread] = IdleRun(len(fingerprints), len(fingerprints[-1][1]) if fingerprints else 0, count)

    def _upstream_headers(self, client_headers, *, decoded, binding=None):
        """Forwarded request headers: hop/private headers and client credentials removed, host
        credentials of ``binding`` (default: the actor binding) added, ``Accept-Encoding:
        identity``; ``Content-Encoding`` dropped when the wire decoded the body (``decoded``)."""
        binding = self.actor_binding if binding is None else binding
        headers = {k: v for k, v in client_headers.items() if k.lower() not in HOP_HEADERS | PRIVATE_HEADERS}
        headers = {k: v for k, v in headers.items() if k.lower() != "accept-encoding"}
        if decoded:
            headers = {k: v for k, v in headers.items() if k.lower() != "content-encoding"}
        if binding.auth_headers is not None:
            headers = {k: v for k, v in headers.items() if k.lower() not in binding.client_auth_headers}
            headers.update(binding.auth_headers)
        elif binding.name != ACTOR_UPSTREAM:
            # A helper binding never forwards a client credential (it may be another binding's).
            headers = {k: v for k, v in headers.items() if k.lower() not in binding.client_auth_headers}
        headers["Accept-Encoding"] = "identity"
        return headers

    def _forward_auxiliary(self, route, body, headers, upstream_url=None):
        """One upstream POST for an auxiliary route: fresh connection, no retry, socket deadline
        ``AUX_REQUEST_TIMEOUT_S``; registered so ``close()`` can interrupt it.

        Returns ``(status, reason, response_headers, body_bytes)``; transport errors propagate.
        """
        url = urlsplit((self.upstream_url if upstream_url is None else upstream_url) + route.upstream_path)
        connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(url.hostname, url.port, timeout=AUX_REQUEST_TIMEOUT_S)
        with self.lock:
            if self.closed:
                raise ConnectionError("model proxy closed")
            self._aux_connections.add(connection)
        try:
            connection.request("POST", _request_target(url), body=body, headers=headers)
            response = connection.getresponse()
            data = response.read()
            return response.status, response.reason, response.getheaders(), data
        finally:
            with self.lock:
                self._aux_connections.discard(connection)
            connection.close()

    def _context_discontinuity(self, state, reason):
        """Apply the synchronous effects of a detected context discontinuity (class docstring,
        "Continuity tracking"), then call ``on_context_discontinuity(reason)`` outside ``lock``."""
        identity = state["identity"]
        with self.lock:
            # The same state change a /responses/compact admission makes (see _record).
            self.image_context_epoch = identity["logical_call_id"]
            thread_id = identity.get("native_thread_id") or ""
            self._image_trim_watermarks.pop(thread_id, None)
            self._image_trim_starts.pop(thread_id, None)
            self._record("model_context_discontinuity", **identity, reason=reason)
        callback = self.on_context_discontinuity
        if callback is None:
            return
        try:
            callback(reason)
        except Exception as error:  # A session callback failure must not fail the model call.
            self._record("model_context_discontinuity_callback_failed", **identity, reason=reason,
                         error_type=type(error).__name__)

    def _continuity_main_thread(self, thread):
        """True when the tracker tracks ``thread``'s main chain (class docstring, "Continuity tracking"):
        ``default_thread_id`` and the threads registered with source ``"actor"``, or every thread when the
        relay has neither. Call with ``lock`` held."""
        actors = [key for key, owner in self.thread_owners.items() if owner.get("source") == "actor"]
        if self.default_thread_id is None and not actors:
            return True
        return thread == self.default_thread_id or thread in actors

    def _observe_continuity(self, state, thread, inputs, output_flags=None):
        """Admit one generation request's tracker ``inputs`` on the tracker ``thread`` (under ``lock``):
        set ``identity.call_purpose``, take the consume-guard verdict when ``output_flags``
        (``wire.history_model_output_flags`` of the same body) is given, apply a discontinuity of a main
        request (never one the consume guard refuses: a refused request is idle, and an idle request has
        no discontinuity), and return a ``ContinuityAdmission`` for the commit. Never moves a baseline."""
        refused = idle = None
        with self.lock:
            admission = self.continuity.admit(thread, main_thread=self._continuity_main_thread(thread), **inputs)
            state["identity"]["call_purpose"] = admission.call_purpose
            if output_flags is not None:
                idle = self._idle_request(thread, admission, inputs["fingerprints"], output_flags)
                run = self._idle_runs.get(thread)
                if idle and run.idle >= self.CONSUME_GUARD_MAX_IDLE_COMPLETIONS:
                    refused = run.idle
        if admission.reason is not None:
            self._context_discontinuity(state, admission.reason)
        return ContinuityAdmission(admission.call_purpose, idle, refused)

    def manual_retry(self):
        """Explicit operator action only; unknown transport outcomes stay blocked."""
        with self.lock:
            if self.active:
                raise ValueError("manual_retry_requires_no_active_model_calls")
            if not self.blocked_error or not self.blocked_error["manual_retry_allowed"]:
                raise ValueError("manual_retry_requires_explicit_provider_error")
            previous = self.blocked_error
            self.blocked_error = None
            # The operator reviewed the loop: identical bodies and idle completions start a fresh count.
            self._runaway_runs.clear()
            self._idle_runs.clear()
        self._record("model_manual_retry_authorized", previous_error=previous)

    def _watch_client(self, state):
        while not state["finished"].wait(.05):
            try:
                readable, _, _ = select.select([state["client"]], [], [], 0)
                if readable and not state["client"].recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                    self._cancel(state, "client_disconnected")
                    return
            except (OSError, ValueError):
                self._cancel(state, "client_disconnected")
                return

    @staticmethod
    def _failed_response_error(response):
        """Compatibility delegate: ``OpenAIResponsesWire.failed_response_error`` (sanitised
        Responses error mapping; never persists the provider message)."""
        return _RESPONSES.failed_response_error(response)

    @staticmethod
    def _response_evidence(raw, endpoint):
        """Compatibility delegate: ``OpenAIResponsesWire.response_evidence`` for ``endpoint``."""
        return _RESPONSES.response_evidence(raw, _RESPONSES.route_for_endpoint(endpoint))

    def _attempt(self, state, body, headers, endpoint, retry_index):
        binding = state.get("binding") or self.actor_binding
        wire = binding.wire
        route = state.get("route") or wire.route_for_endpoint(endpoint)
        for key in ("failed_response", "provider_explicit_error", "failure_evidence", "connect_failure",
                    "stream_incomplete"):
            state.pop(key, None)
        with self.lock:
            if state["cancel"].is_set():
                raise self.policy.InfrastructureFailure("model call cancelled")
            self.attempt_count += 1
            number = self.attempt_count
        identity = dict(**state["identity"], request_index=number, attempt_index=retry_index + 1)
        payload = dict(url=binding.upstream_url + endpoint, body=base64.b64encode(body).decode(), headers=headers,
                       wire=wire.name)
        if wire.client != CODEX_PROFILE_NAME:
            payload["client"] = wire.client
        serialized = json.dumps(payload)
        started = time.monotonic()
        process = None
        result, phase = None, "worker_initialization"
        try:
            if subprocess.Popen is _DEFAULT_POPEN:
                process = _InProcessAttempt()
                with self.lock:
                    state["process"] = process
                    if state["cancel"].is_set():
                        _kill(process)
                self._record("model_request_started", **identity, timeout_s=None, max_retries=self.max_retries,
                             deadline_boundary=None, phase="worker_initialization")
                started = time.monotonic()
                result, phase = self._inprocess_http(process, identity, binding.upstream_url + endpoint, body, headers,
                                                     started, wire=wire)
            else:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--http-worker"],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           text=True, start_new_session=True)
                with self.lock:
                    state["process"] = process
                    if state["cancel"].is_set():
                        _kill(process)
                self._record("model_request_started", **identity, timeout_s=None, max_retries=self.max_retries,
                             deadline_boundary=None, phase="worker_initialization")
                started = time.monotonic()
                process.stdin.write(serialized)
                try:
                    process.stdin.close()
                finally:
                    # A failed flush still closes stdin; communicate must not flush it again.
                    process.stdin = None
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if "result" in event:
                        result = event
                    elif "kind" in event:
                        phase = event.get("phase", phase)
                        kind = event.pop("kind")
                        if kind == "model_response_bytes_received":
                            # Track real upstream bytes without retaining one metric per chunk.
                            with self.lock:
                                self._last_activity_at = time.monotonic()
                            continue
                        self._record(kind, **identity, **event)
                process.wait()
            if state["cancel"].is_set():
                raise self.policy.InfrastructureFailure("model call cancelled")
            if process.returncode != 0 or result is None:
                raise self.policy.InfrastructureFailure("planner HTTP worker failed", retryable=False)
            response = result["result"]
            try:
                _, state["attempt_evidence"], _ = wire.response_evidence(base64.b64decode(response.get("body", "")), route)
            except (ValueError, EOFError, TypeError, AttributeError):
                pass
            raw = base64.b64decode(response["body"])
            status = response.get("status")
            if status is not None and status >= 400:
                state["provider_explicit_error"] = True
                try:
                    error_body = json.loads(raw)
                except ValueError:
                    error_body = None
                detail = error_body if isinstance(error_body, dict) else {}
                state["attempt_evidence"] = dict(usage=None, usage_known=False,
                    response_error=wire.failed_response_error(detail), http_error_status=status)
                state["failure_evidence"] = dict(http_status=status, **state["attempt_evidence"]["response_error"])
                raise wire.classify_http_failure(status, response["headers"], error_body, policy=self.policy)
            if response.get("error_type"):
                phase = response.get("phase", phase)
                # A failure while connecting (DNS resolution, refused or timed-out connect) happens before
                # any request byte reaches the provider, so resending cannot duplicate a generation.
                connect_failure = phase == "connecting_upstream"
                state["failure_evidence"] = dict(
                    classification="transport_connect_failed" if connect_failure else "transport_incomplete",
                    transport_error_type=response["error_type"], phase=phase)
                state["connect_failure"] = connect_failure
                self._record("model_request_transport_error", **identity, error_type=response["error_type"], phase=phase)
                raise self.policy.InfrastructureFailure("model upstream connect failed" if connect_failure
                    else "model upstream transport incomplete", retryable=connect_failure)
            # Nothing is released to the actor before a completed terminal, so a stream that ended
            # without one can be resent like a connect failure.
            def stream_incomplete(message, detail):
                state["failure_evidence"] = dict(classification="stream_incomplete", http_status=status, detail=detail)
                state["stream_incomplete"] = True
                return self.policy.InfrastructureFailure(message, retryable=True)
            try:
                complete, evidence, terminal = wire.response_evidence(raw, route)
            except EOFError:
                raise stream_incomplete("model response terminal missing", "terminal_missing") from None
            except (ValueError, TypeError, AttributeError):
                raise stream_incomplete("model response malformed", "malformed") from None
            state["attempt_evidence"] = evidence
            outcome = wire.completion_outcome(complete, evidence)
            if outcome in ("failed", "incomplete"):
                # Neither failed nor incomplete responses may release tool deltas. A deferred-delivery
                # client never gets an in-band failure (it would retry it); it gets relay_error instead.
                failed_raw = (wire.failed_payload(terminal, streaming=wire.is_event_stream(raw))
                              if wire.early_stream else None)
                if failed_raw is not None:
                    state["failed_response"] = (failed_raw, response["headers"])
                error = evidence.get("response_error", dict(classification="incomplete"))
                state["failure_evidence"] = dict(http_status=status, **error)
                state["provider_explicit_error"] = outcome == "failed"
                raise self.policy.InfrastructureFailure("model response failed: " + error["classification"],
                    retryable=bool(error.get("retryable")), retry_after_s=error.get("retry_after_s"))
            if outcome != "usable":
                raise stream_incomplete("model response ended without completed status", "not_completed")
            if self.continuity is not None:
                state["context_edit_applied"] = wire.context_edit_applied(terminal)
            return raw, response["headers"]
        finally:
            _kill(process)
            if process is not None:
                process.communicate()
            with self.lock:
                state["process"] = None
            info = result or {}
            response = info.get("result", {})
            self._record("model_request_finished", **identity, http_status=response.get("status"),
                         elapsed_s=time.monotonic()-started, first_upstream_byte_s=info.get("first_upstream_byte_s"),
                         incoming_bytes=info.get("incoming_bytes"), outgoing_bytes=0,
                         transport_complete=bool(result and not response.get("error_type")),
                         timed_out=False, phase=phase,
                         **state.pop("attempt_evidence", dict(usage=None, usage_known=False)))

    # Compatibility delegates (Responses request shape); the relay itself uses ``self.wire``.
    @staticmethod
    def _part_observation_labels(parts):
        """Delegate: ``OpenAIResponsesWire.part_observation_labels``."""
        return OpenAIResponsesWire.part_observation_labels(parts)

    @classmethod
    def _inline_image_locations(cls, request, *, include_labels=False):
        """Delegate: ``OpenAIResponsesWire.image_locations``."""
        return _RESPONSES.image_locations(request, include_labels=include_labels)

    @classmethod
    def _step_image_boundaries(cls, request):
        """Delegate: ``OpenAIResponsesWire.step_image_boundaries``."""
        return _RESPONSES.step_image_boundaries(request)

    @classmethod
    def _trim_image_range(cls, request, start_idx, end_idx):
        """Delegate: ``WireAdapter.trim_image_range`` on the Responses wire."""
        return _RESPONSES.trim_image_range(request, start_idx, end_idx)

    @classmethod
    def _trim_prefix_images(cls, request, prefix_count):
        """Delegate: ``WireAdapter.trim_prefix_images`` on the Responses wire."""
        return _RESPONSES.trim_prefix_images(request, prefix_count)

    @classmethod
    def _trim_oldest_images(cls, body, keep_latest=24):
        """Delegate: ``WireAdapter.trim_oldest_images`` on the Responses wire."""
        return _RESPONSES.trim_oldest_images(body, keep_latest)

    def _prepare_responses_body(self, state, base_request, base_body, *, force_advance=False, retry_index=0):
        """Image-trim window state machine for one generation body (any wire; historic name).

        Returns ``(request, body_bytes)`` to forward. Image locations, step boundaries,
        placeholders, reuse and the byte limits come from ``self.wire``.
        """
        wire = self.wire
        identity = state["identity"]
        thread_key = identity.get("native_thread_id") or ""
        locations, step_boundaries = wire.step_image_boundaries(base_request)
        total_images = len(locations)
        with self.lock:
            watermark, last_total = self._image_trim_watermarks.get(thread_key, (0, 0))
            start_idx = self._image_trim_starts.get(thread_key, 0)
            if total_images < last_total or watermark > total_images or start_idx > watermark:
                start_idx, watermark = 0, 0
            previous = (start_idx, watermark)
            if force_advance and total_images > 0:
                keep = max(8, 16 - retry_index * 4)
                watermark = min(total_images, max(watermark + 4, total_images - keep))

        def render(s_idx, wm):
            working = base_request
            if wm > s_idx:
                working, _ = wire.trim_image_range(working, s_idx, wm)
            image_stats = None
            if self.image_reuse:
                rewritten, image_stats = wire.reuse_exact_images(working)
                if image_stats["replaced"]:
                    working = rewritten
            if wm <= s_idx and not (image_stats and image_stats["replaced"]):
                encoded = base_body
            else:
                encoded = json.dumps(working, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            return working, encoded, image_stats

        working, encoded, image_stats = render(start_idx, watermark)
        limit = self.max_inline_images
        def image_count(request):
            return len(wire.image_locations(request))
        # The provider rejects requests by inline image count, independent of bytes.
        bytes_over = len(encoded) > wire.body_limit_bytes
        count_over = bool(limit) and image_count(working) > limit
        reason = "body_bytes_and_image_count" if bytes_over and count_over else "image_count" if count_over else "body_bytes"
        if self.image_trim and (bytes_over or count_over) and total_images > 8:
            target_bytes = wire.trim_target_bytes
            target_images = limit * 2 // 3
            tail_candidates = [b for b in step_boundaries if b <= total_images - 12]
            new_wm = max(watermark, tail_candidates[-1] if tail_candidates else max(0, total_images - 12))
            if watermark == 0 and new_wm > 0:
                need_bytes = len(encoded) - target_bytes if bytes_over else 0
                need_images = image_count(working) - target_images if count_over else 0
                saved, removed, raw_start = 0, 0, new_wm
                for idx in range(new_wm - 1, -1, -1):
                    parts, part_idx = locations[idx]
                    saved += wire.image_part_size(parts[part_idx])
                    removed += 1
                    raw_start = idx
                    if saved >= need_bytes and removed >= need_images:
                        break
                prefix_candidates = [b for b in step_boundaries if 0 < b <= raw_start and b < new_wm]
                start_idx = prefix_candidates[-1] if prefix_candidates else 0
            watermark = new_wm
            working, encoded, image_stats = render(start_idx, watermark)
            while ((bytes_over and len(encoded) > target_bytes)
                   or (count_over and image_count(working) > target_images)):
                if start_idx > 0:
                    lower = [b for b in step_boundaries if 0 < b < start_idx]
                    start_idx = lower[-1] if lower else 0
                elif total_images - watermark > 8:
                    watermark = min(total_images - 8, max(watermark + 4, total_images - 8))
                else:
                    break
                working, encoded, image_stats = render(start_idx, watermark)
        with self.lock:
            self._image_trim_watermarks[thread_key] = (watermark, total_images)
            self._image_trim_starts[thread_key] = start_idx
        if (start_idx, watermark) != previous:
            self._record("model_images_trimmed", **identity,
                         reason="provider_rejected" if force_advance else reason,
                         images_total=total_images, images_forwarded=image_count(working),
                         window_start=start_idx, window_end=watermark,
                         previous_window=list(previous), original_body_bytes=len(base_body),
                         forwarded_body_bytes=len(encoded))
        if not force_advance and image_stats and image_stats["replaced"]:
            self._record("model_images_reused", **identity, **image_stats,
                         original_body_bytes=len(base_body), forwarded_body_bytes=len(encoded),
                         forwarded_body_sha256=hashlib.sha256(encoded).hexdigest())
        return working, encoded

    def _complete(self, state, body, headers, endpoint):
        binding = state.get("binding") or self.actor_binding
        wire = binding.wire
        # Image trim and reuse belong to the actor binding only (design §4.1 bindings rules).
        image_trim = self.image_trim and binding.name == ACTOR_UPSTREAM
        route = state.get("route") or wire.route_for_endpoint(endpoint)
        if (image_trim and route.kind == "generation" and not route.compaction and "base_request" not in state
                and len(body) > wire.body_limit_bytes):
            body = wire.trim_oldest_images(body, 16)
        retry_index = rounds = 0
        while True:
            try:
                result = self._attempt(state, body, headers, endpoint, retry_index)
                if rounds:
                    with self.lock:
                        self._recovered_calls += 1
                    self._record("model_transport_recovered", **state["identity"], recovery_rounds=rounds,
                                 attempt_index=retry_index + 1)
                return result
            except self.policy.InfrastructureFailure as failure:
                if state["cancel"].is_set():
                    raise
                image_rejected = wire.image_rejected(state.get("failure_evidence") or {})
                # Without trimming, a retry would resend the body the provider just rejected.
                explicit_retryable = bool((state.get("provider_explicit_error") or state.get("connect_failure")
                                           or state.get("stream_incomplete"))
                                          and failure.retryable
                                          and not (image_rejected and not image_trim))
                recovering = not (explicit_retryable and retry_index < self.max_retries)
                delay = self._recovery_delay(state, rounds, explicit_retryable) if recovering else None
                if recovering and delay is None:
                    return self._terminal_failure(state, failure)
                if image_rejected:
                    if "base_request" in state:
                        _, body = self._prepare_responses_body(
                            state, state["base_request"], state["base_body"],
                            force_advance=True, retry_index=retry_index,
                        )
                    else:
                        body = wire.trim_oldest_images(body, max(8, 16 - retry_index * 4))
                if recovering:
                    rounds += 1
                    self._record("model_transport_recovery_scheduled", **state["identity"], recovery_round=rounds,
                                 attempt_index=retry_index + 1, delay_s=delay, http_status=failure.status,
                                 episode_cooldown_s=self.transport_recovery_summary["cooldown_s"],
                                 error=state.get("failure_evidence"))
                else:
                    delay = self.policy._retry_delay_seconds(retry_index, failure)
                    self._record("model_request_retry", **state["identity"], attempt_index=retry_index + 1,
                                 delay_s=delay, http_status=failure.status,
                                 error=state.get("failure_evidence"))
                if state["cancel"].wait(delay):
                    raise self.policy.InfrastructureFailure("model call cancelled") from None
                if recovering and (reason := self._recovery_gate_reason()) is not None:
                    # Actions may have started during the cooldown; keep the original failure.
                    self._deny_recovery(state, reason, rounds)
                    return self._terminal_failure(state, failure)
                retry_index += 1

    def _terminal_failure(self, state, failure):
        if "failed_response" in state:
            self._block_after_failure(state, "ModelResponseFailed")
            state["delivering_failed_response"] = True
            return state["failed_response"]
        raise failure

    def start(self):
        if self.server is not None or self.closed:
            raise ValueError("model_proxy_already_started_or_closed")
        self.callback_thread = threading.Thread(target=self._callbacks, daemon=True)
        self.callback_thread.start()
        owner = self

        wire = self.wire

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *_): pass

            def send_reply(self, reply):
                """Send a relay-generated ``WireReply`` as a complete HTTP answer."""
                if reply.body is None:
                    self.send_error(reply.status, reply.reason)
                    return
                self.send_response(reply.status, reply.reason)
                for key, value in reply.headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(reply.body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(reply.body)
                self.wfile.flush()

            def content_length(self):
                """Declared body length within the relay bound, or None after answering 400."""
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    self.send_error(400)
                    return None
                if not 0 <= length <= MAX_CLIENT_BODY_BYTES or self.headers.get("Transfer-Encoding"):
                    self.send_error(400, "Content-Length required within model input bound")
                    return None
                return length

            def drain_body(self, length):
                """Read and discard a refused request's declared body (bounded by ``length`` and
                ``REFUSAL_DRAIN_TIMEOUT_S``) so the client finishes writing and reads the refusal."""
                remaining = length
                try:
                    self.connection.settimeout(REFUSAL_DRAIN_TIMEOUT_S)
                    while remaining > 0 and (chunk := self.rfile.read(min(remaining, 1 << 16))):
                        remaining -= len(chunk)
                    self.connection.settimeout(None)
                except OSError:
                    pass

            def refuse(self, reply, length):
                """Drain the declared body, then send the relay-generated ``reply``."""
                if length:
                    self.drain_body(length)
                self.send_reply(reply)

            def declared_length(self):
                """Declared body length when valid and within the bound, else 0 (nothing to drain)."""
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    return 0
                return length if 0 < length <= MAX_CLIENT_BODY_BYTES and not self.headers.get("Transfer-Encoding") else 0

            def wire_thread(self, wire=wire):
                """The native thread key the client sent: the private ``X-Auriga-Thread`` header of a
                host-side client, else the wire's own thread key; ``""`` when absent or invalid."""
                key = self.headers.get("X-Auriga-Thread") or wire.thread_key(self.headers)
                return key if isinstance(key, str) and len(key) <= 256 else ""

            def attribution(self, wire=wire):
                """``(native_thread_id, owner_attribution_or_None)``; call with ``owner.lock`` held.

                The key is ``wire_thread``; an empty or unregistered key falls back to ``default_thread_id``.
                """
                native_thread_id = self.wire_thread(wire)
                if (owner.default_thread_id is not None and native_thread_id not in owner.thread_owners
                        and native_thread_id not in owner.cancelled_threads):
                    native_thread_id = owner.default_thread_id
                return native_thread_id, owner.thread_owners.get(native_thread_id)

            def do_POST(self):
                binding, path = owner._binding_for_path(self.path)
                if binding is None:
                    self.refuse(wire.not_found(), self.declared_length())
                    return
                route = binding.wire.route(self.command, path)
                if route is not None and route.kind == "generation":
                    self.generation(route, binding)
                elif route is not None and route.kind == "auxiliary":
                    self.auxiliary(route, binding)
                else:
                    self.refuse(binding.wire.not_found(), self.declared_length())

            def binding_projection(self, binding):
                """Egress projection of a request on ``binding``: the ``request_projection`` hook for the
                actor binding (None when unset), the binding wire's walker for a named binding."""
                if binding.name == ACTOR_UPSTREAM:
                    return owner.request_projection
                return lambda request: binding.wire.project_request(
                    request, binding.presentation, registered_tools=frozenset(), tool_name_prefix="")

            def auxiliary(self, route, binding):
                """Projected, forwarded once, never charged or retried; one ``model_aux_request``."""
                wire = binding.wire
                projection = self.binding_projection(binding)
                length = self.content_length()
                if length is None:
                    return
                with owner.lock:
                    native_thread_id, attribution = self.attribution(wire)
                    attribution = attribution or dict(
                        source=self.headers.get("X-Auriga-Source", "actor")[:128],
                        caller_request_id=self.headers.get("X-Auriga-Request-Id", "")[:128])
                    owner.aux_request_count += 1
                    record = dict(aux_request_index=owner.aux_request_count, endpoint=route.endpoint,
                                  native_thread_id=native_thread_id, **attribution)
                    if binding.name != ACTOR_UPSTREAM:
                        record["upstream"] = binding.name
                    denial = ("thread_cancelled" if native_thread_id in owner.cancelled_threads else
                              "transport_blocked" if owner.blocked_error is not None or owner.closed else None)
                started, phase = time.monotonic(), "reading_client_body"
                try:
                    if denial is not None:
                        record["denied"] = denial
                        self.refuse(wire.denial(denial), length)
                        return
                    if projection is None:
                        # An auxiliary body carries raw history; without the egress projection it
                        # never leaves the host.
                        record["denied"] = "projection_unavailable"
                        self.refuse(wire.relay_error(streaming=False, evidence=None, manual_retry_allowed=False), length)
                        return
                    self.connection.settimeout(None)
                    received = self.rfile.read(length)
                    if len(received) != length:
                        raise ConnectionError("incomplete client request")
                    record["body_bytes"] = len(received)
                    phase = "projecting"
                    body = wire.decode_body(self.headers, received)
                    decoded = body is not received
                    request = json.loads(body)
                    request = projection(request)
                    body = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
                    record["requested_model"] = wire.requested_model(request) if isinstance(request, dict) else None
                    headers = owner._upstream_headers(self.headers, decoded=decoded, binding=binding)
                    phase = "forwarding"
                    record.update(forwarded_body_bytes=len(body), forwarded_body_sha256=hashlib.sha256(body).hexdigest())
                    status, reason, response_headers, data = owner._forward_auxiliary(route, body, headers,
                                                                                      binding.upstream_url)
                    record.update(http_status=status, response_bytes=len(data))
                    phase = "delivering"
                    if status >= 400:
                        try:
                            detail = json.loads(data)
                        except ValueError:
                            detail = None
                        record["response_error"] = wire.failed_response_error(detail if isinstance(detail, dict) else {})
                        # Provider error text never reaches the client: the wire's sanitised error instead.
                        self.send_reply(wire.relay_error(streaming=False, evidence=None, manual_retry_allowed=False))
                        record["delivered"] = True
                        return
                    self.send_response(status, reason)
                    for key, value in response_headers:
                        if key.lower() not in HOP_HEADERS:
                            self.send_header(key, value)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                    record["delivered"] = True
                except (Exception, owner.policy.InfrastructureFailure) as error:
                    record.update(error_type=type(error).__name__, phase=phase)
                    if phase != "delivering":
                        try:
                            self.send_reply(wire.relay_error(streaming=False, evidence=None, manual_retry_allowed=False))
                        except OSError:
                            pass
                finally:
                    self.close_connection = True
                    owner._record("model_aux_request", **record, elapsed_s=time.monotonic()-started)

            def generation(self, route, binding):
                wire = binding.wire
                actor = binding.name == ACTOR_UPSTREAM
                length = self.content_length()
                if length is None:
                    return
                refusal, denied, exhausted, notify_limit = None, False, False, False
                with owner.lock:  # one atomic admission decision; any refusal is sent after the lock
                    native_thread_id, registered = self.attribution(wire)
                    # The continuity tracker's thread: the client's own key, never remapped when unregistered.
                    chain_thread = self.wire_thread(wire) or owner.default_thread_id or ""
                    if native_thread_id in owner.cancelled_threads:
                        owner._record("model_thread_request_rejected", native_thread_id=native_thread_id,
                                      reason=owner.cancelled_threads[native_thread_id])
                        refusal = "thread_cancelled"
                    elif owner.blocked_error is not None:
                        # Blocks SDK reconnect, automatic compaction recovery and
                        # any new logical request until an operator authorizes it.
                        refusal = "transport_blocked"
                    else:
                        exhausted = owner.request_count >= owner.max_requests
                        denied = owner.closed or owner.request_count + owner.reserved_calls >= owner.max_requests
                        if not denied:
                            owner.reserved_calls += 1
                            owner.logical_call_count += 1
                            number = owner.logical_call_count
                            attribution = registered or dict(
                                source=self.headers.get("X-Auriga-Source", "actor")[:128],
                                caller_request_id=self.headers.get("X-Auriga-Request-Id", "")[:128])
                            identity = dict(logical_call_index=number, logical_call_id=f"{owner.transport_id}:{number}",
                                            endpoint=route.endpoint, native_thread_id=native_thread_id, **attribution)
                            if not actor:
                                identity["upstream"] = binding.name
                            state = dict(identity=identity, client=self.connection, process=None,
                                         cancel=threading.Event(), finished=threading.Event(), cancel_reason=None,
                                         reading_client_body=True, route=route)
                            if not actor:
                                state["binding"] = binding
                            if not owner.active:
                                owner._wait_started_at = time.monotonic()
                            owner.active[number] = state
                        notify_limit = exhausted and not owner.limit_notified
                        if notify_limit:
                            owner.limit_notified = True
                if refusal is not None:
                    self.refuse(wire.denial(refusal), length)
                    return
                if denied:
                    event = owner._record("model_budget_exhausted" if exhausted else "model_budget_reserved",
                                          completed_calls=owner.request_count, reserved_calls=owner.reserved_calls,
                                          max_requests=owner.max_requests)
                    self.refuse(wire.denial("budget"), length)
                    if notify_limit and owner.on_limit:
                        owner.callbacks.put((owner.on_limit, event))
                    return
                started, streaming, delivered, charged = time.monotonic(), False, False, False
                continuity = None  # ContinuityTracker inputs of this call (non-Codex profiles only)
                call_purpose = None  # its admitted call_purpose, committed with the same inputs
                idle = None  # its consume-guard verdict (client_loop_guard profiles only)
                # stream_open: the 200 event-stream head was already sent (early_stream profiles only);
                # a deferred profile sends nothing until the attempt is validated.
                stream_open = False
                io_lock = threading.Lock()
                heartbeat_stop = threading.Event()
                keepalive = wire.keepalive_frame()
                def heartbeat():
                    while not heartbeat_stop.wait(wire.keepalive_interval_s):
                        try:
                            with io_lock:
                                self.wfile.write(keepalive)
                                self.wfile.flush()
                        except OSError:
                            owner._cancel(state, "client_disconnected")
                            return
                try:
                    owner._record("model_call_started", **identity, phase="reading_client_body")
                    self.connection.settimeout(None)
                    received = self.rfile.read(length)
                    with owner.lock:
                        state["reading_client_body"] = False
                    if len(received) != length:
                        raise ConnectionError("incomplete client request")
                    owner._record("model_request_body_received", **identity, body_bytes=len(received),
                                  elapsed_s=time.monotonic()-started, body_sha256=hashlib.sha256(received).hexdigest())
                    body = wire.decode_body(self.headers, received)
                    decoded = body is not received  # the wire changed the encoding: drop Content-Encoding
                    if wire.client_loop_guard and (runaway := owner._runaway_admission(native_thread_id, body)):
                        owner._block_runaway(state, runaway)
                        self.send_reply(wire.denial("client_retry_runaway"))
                        return
                    request = json.loads(body)
                    projected = False
                    if actor and owner.request_context is not None:
                        # Snapshot once per logical request; retries reuse the filtered body.
                        request = owner.request_context(request, dict(identity))
                        projected = True
                    projection = self.binding_projection(binding)
                    if projection is not None:
                        request = projection(request)
                        projected = True
                    if actor and owner.continuity is not None and isinstance(request, dict):
                        # The body as the agent sent it (after the egress projection), before the
                        # relay's own image trim/reuse below.
                        continuity = wire.continuity_inputs(request)
                        flags = wire.history_model_output_flags(request) if wire.client_loop_guard else None
                        admitted = owner._observe_continuity(state, chain_thread, continuity, flags)
                        call_purpose, idle = admitted.call_purpose, admitted.idle
                        if admitted.refused is not None:
                            # Consume guard: refused before forwarding, never charged.
                            owner._block_unconsumed(state, admitted.refused)
                            self.send_reply(wire.denial("client_retry_runaway"))
                            return
                    tier = owner.service_tier if actor else None
                    if tier is not None and wire.supports_service_tier and not route.compaction and isinstance(request, dict):
                        request = dict(request, service_tier=tier)
                    if projected or tier is not None:
                        body = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
                    if actor and not route.compaction:
                        state["base_request"] = request
                        state["base_body"] = body
                        request, body = owner._prepare_responses_body(state, request, body)
                    identity["requested_model"] = wire.requested_model(request)
                    streaming = wire.is_streaming(request)
                    self.connection.settimeout(None)
                    if streaming and wire.early_stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        stream_open = True
                        if keepalive is not None:
                            threading.Thread(target=heartbeat, daemon=True).start()
                    threading.Thread(target=owner._watch_client, args=(state,), daemon=True).start()
                    headers = owner._upstream_headers(self.headers, decoded=decoded, binding=binding)
                    raw, response_headers = owner._complete(state, body, headers, route.upstream_path)
                    with owner.lock:
                        if state["cancel"].is_set() or owner.closed:
                            raise owner.policy.InfrastructureFailure("model call cancelled before delivery")
                    if call_purpose == MAIN_CALL and state.get("context_edit_applied"):
                        # The provider edited the main chain's context server-side: what it saw is not the history.
                        owner._context_discontinuity(state, PROVIDER_CONTEXT_EDIT)
                    heartbeat_stop.set()
                    with io_lock:
                        if not stream_open:
                            self.send_response(200)
                            for key, value in response_headers.items():
                                if key.lower() not in HOP_HEADERS:
                                    self.send_header(key, value)
                            self.send_header("Content-Length", str(len(raw)))
                            self.send_header("Connection", "close")
                            self.end_headers()
                        self.wfile.write(raw)
                        self.wfile.flush()
                        delivered = True
                        state["delivered"] = True
                    if state.get("delivering_failed_response"):
                        owner._block_after_failure(state, "ModelResponseFailed")
                        owner._record("model_call_failed", **identity, error_type="ModelResponseFailed",
                                      cancelled=False, completed_call_charged=False, outgoing_bytes=len(raw),
                                      manual_retry_required=True, error=state.get("failure_evidence"))
                    else:
                        with owner.lock:
                            # Keep the reservation until delivery succeeds; usage is
                            # already recorded even when a downstream write fails.
                            owner.request_count += 1
                            owner.reserved_calls -= 1
                            charged = True
                            if continuity is not None:
                                # Only a delivered usable main completion becomes the thread's baseline.
                                owner.continuity.commit(chain_thread, call_purpose=call_purpose, **continuity)
                                if wire.client_loop_guard:
                                    owner._commit_idle_run(chain_thread, call_purpose, continuity["fingerprints"],
                                                           idle)
                        owner._record("model_call_completed", **identity, completed_calls=owner.request_count,
                                      outgoing_bytes=len(raw))
                except (Exception, owner.policy.InfrastructureFailure) as error:
                    heartbeat_stop.set()
                    owner._block_after_failure(state, type(error).__name__)
                    owner._record("model_call_failed", **identity, error_type=type(error).__name__,
                                  cancelled=state["cancel"].is_set(), completed_call_charged=charged,
                                  manual_retry_required=not state["cancel"].is_set() or state["cancel_reason"] == "client_disconnected",
                                  error=state.get("failure_evidence"))
                    try:
                        reply = wire.relay_error(streaming=stream_open, evidence=state.get("failure_evidence"),
                                                 manual_retry_allowed=bool(state.get("provider_explicit_error")))
                        with io_lock:
                            if stream_open:
                                # Never end an open client stream without a terminal event.
                                self.wfile.write(reply.body)
                                self.wfile.flush()
                            else:
                                self.send_reply(reply)
                    except OSError:
                        pass
                finally:
                    heartbeat_stop.set()
                    state["finished"].set()
                    self.close_connection = True
                    with owner.lock:
                        if not charged:
                            owner.reserved_calls -= 1
                        owner.active.pop(number, None)
                        if not owner.active and owner._wait_started_at is not None:
                            owner._wait_total_s += time.monotonic() - owner._wait_started_at
                            owner._wait_started_at = None
                    owner._record("model_call_finished", **identity, completed_calls=owner.request_count,
                                  attempt_count=owner.attempt_count, delivered=delivered,
                                  elapsed_s=time.monotonic()-started, cancelled=state["cancel"].is_set(),
                                  failed_response_delivered=bool(delivered and state.get("delivering_failed_response")))

        if "GET" in wire.methods:
            # Codex keeps the standard 501 page for GET (no do_GET attribute); wires that list
            # GET answer it through the same routing (``GET /v1/models`` → local 404).
            Handler.do_GET = Handler.do_POST

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.block_on_close = False
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
        self.thread.start()
        return self

    def close(self):
        with self.lock:
            self.closed = True
            auxiliary = list(self._aux_connections)
        for connection in auxiliary:
            _InProcessAttempt._force_close(connection)
        self.cancel_pending("proxy_closed")
        with self.lock:
            active = list(self.active.values())
        if self.wire.client_loop_guard:
            # Each cancelled handler writes its relay_error terminal first: a client whose stream
            # ends without a terminal re-requests in a tight loop.
            deadline = time.monotonic() + self.SHUTDOWN_TERMINAL_GRACE_S
            for state in active:
                state["finished"].wait(max(0.0, deadline - time.monotonic()))
        for state in active:
            self._shutdown(state["client"])
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(1)
        for state in active:
            state["finished"].wait(1)
        self._close_conn_pool()
        self.callbacks.put(None)
        if self.callback_thread and self.callback_thread is not threading.current_thread():
            self.callback_thread.join(1)

    def __enter__(self): return self.start()
    def __exit__(self, *_): self.close()


if __name__ == "__main__":
    _http_worker()
