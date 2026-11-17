"""Blocking HTTP client for ``opencode serve`` 1.19.5 (plan 07 section 3.1, design 04 section 2.2).

Stdlib ``http.client`` only, HTTP basic auth (user ``opencode`` + the per-run server password), one short-lived
connection per request so any number of threads may call it concurrently.  The client never retries anything.

Error model (the only thing callers need to decide whether a request may have been admitted):

* ``OpenCodeHttpError``       the server answered with a non-2xx status.  ``definite_rejection`` is True for 4xx:
                              OpenCode explicitly refused the request and nothing was admitted.  5xx is *not* a
                              definite rejection (the request may have been processed before the failure).
* ``OpenCodeTransportError``  no usable HTTP answer (connection refused/reset, timeout, truncated or unparsable
                              answer).  It is an ``OSError``.  For ``prompt_async`` this means *unknown admission*:
                              the caller must stop, never resend.
* ``OpenCodeStreamEnded``     the ``/event`` stream ended (a transport error subclass).

``/event`` is advisory only (OpenCode keeps no replay buffer and sends no SSE ``id:``); ``event_stream`` reconnects
with backoff and reports every gap so the caller can reconcile from ``status()`` / ``messages()``.
"""
from __future__ import annotations

import base64
import http.client
import json
import re
import socket
import threading
from typing import Callable, Iterator, Mapping, Sequence
from urllib.parse import quote, urlsplit

__all__ = [
    "OpenCodeError", "OpenCodeHttpError", "OpenCodeTransportError", "OpenCodeStreamEnded", "OpenCodeClient",
    "is_definite_rejection", "routes_from_openapi", "OPENCODE_USERNAME",
]

OPENCODE_USERNAME = "opencode"          # fixed basic-auth user name of `opencode serve` (OPENCODE_SERVER_PASSWORD)
_SESSION_ID = re.compile(r"ses[A-Za-z0-9_]*\Z")
_HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
_MAX_BODY_BYTES = 256 << 20


class OpenCodeError(Exception):
    """Base class of the client errors."""


class OpenCodeHttpError(OpenCodeError):
    """Non-2xx HTTP answer.  ``status`` is the HTTP status, ``body`` the decoded JSON body (else text, else None)."""

    def __init__(self, status: int, body: object = None, *, method: str = "", path: str = ""):
        super().__init__(f"opencode_http_{status}")
        self.status, self.body, self.method, self.path = status, body, method, path

    @property
    def definite_rejection(self) -> bool:
        """True for 4xx: the server refused the request, nothing was admitted (safe to treat as rejected)."""
        return 400 <= self.status < 500


class OpenCodeTransportError(OpenCodeError, OSError):
    """No usable HTTP answer; admission of a state-changing request is unknown.  ``__cause__`` has the original."""

    admission_unknown = True

    def __init__(self, reason: str):
        OpenCodeError.__init__(self, reason)
        self.reason = reason


class OpenCodeStreamEnded(OpenCodeTransportError):
    """The ``/event`` stream reached EOF (server closed it or the connection dropped)."""


def is_definite_rejection(error: BaseException) -> bool:
    """True only for an HTTP 4xx answer.  Everything else (5xx, transport error, timeout) is unknown admission."""
    return isinstance(error, OpenCodeHttpError) and error.definite_rejection


def routes_from_openapi(document: Mapping) -> list[str]:
    """Sorted ``"METHOD /path"`` operation list of an OpenAPI document (the format of ``opencode_routes.json``)."""
    paths = document.get("paths") if isinstance(document, Mapping) else None
    if not isinstance(paths, Mapping):
        raise ValueError("openapi_paths_missing")
    routes = []
    for path, item in paths.items():
        if isinstance(item, Mapping):
            routes.extend(f"{method.upper()} {path}" for method in item if method in _HTTP_METHODS)
    return sorted(routes)


class _UnixHTTPConnection(http.client.HTTPConnection):
    """``HTTPConnection`` whose transport is an ``AF_UNIX`` stream socket (the host side of an inbound socket)."""

    def __init__(self, host: str, port: int, unix_socket: str, timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._unix_socket = unix_socket

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self._unix_socket)
        except BaseException:
            sock.close()
            raise
        self.sock = sock


class OpenCodeClient:
    """Typed wrapper over the ~10 OpenCode v1 routes Auriga uses.  Thread-safe; no retries.

    Parameters
    ----------
    base_url:
        ``http://127.0.0.1:<port>`` of the server (no credentials, no path).
    password:
        The per-run ``OPENCODE_SERVER_PASSWORD``.  ``username`` defaults to the fixed ``opencode``.
    timeout_s:
        Socket timeout (s) of every non-streaming request, default 30.  A timeout is a transport error.
    stream_timeout_s:
        Maximum silence (s) on the ``/event`` stream before it counts as dropped, default 35 (the server sends a
        heartbeat every 10 s).
    unix_socket:
        Host path of a Unix socket that reaches the server (``AgentSandbox.inbound_socket_path`` for a sandboxed
        server).  Default ``None``: plain TCP to ``base_url``.  With a socket the connection goes there; ``base_url``
        still supplies the ``Host`` header (``127.0.0.1:<port>``, the in-sandbox listen address).  The API, the
        authentication and the error model are identical.
    """

    def __init__(self, base_url: str, password: str, *, username: str = OPENCODE_USERNAME, timeout_s: float = 30.0,
                 stream_timeout_s: float = 35.0, unix_socket: str | None = None):
        parts = urlsplit(base_url)
        if parts.scheme != "http" or not parts.hostname or parts.username or parts.path not in ("", "/") \
                or parts.query or parts.fragment:
            raise ValueError("opencode_base_url_invalid")
        self.host, self.port = parts.hostname, parts.port or 80
        self.timeout_s, self.stream_timeout_s = float(timeout_s), float(stream_timeout_s)
        self.unix_socket = None if unix_socket is None else str(unix_socket)
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        self._auth = "Basic " + token
        self._streams: set = set()
        self._lock = threading.Lock()
        self.invalid_event_frames = 0

    # ------------------------------------------------------------------------------------------ transport
    def _connect(self, timeout: float) -> http.client.HTTPConnection:
        if self.unix_socket is not None:
            return _UnixHTTPConnection(self.host, self.port, self.unix_socket, timeout)
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict:
        headers = {"Authorization": self._auth, "Accept": "application/json"}
        headers.update(extra or {})
        return headers

    def _request(self, method: str, path: str, body: object = None, *, timeout: float | None = None,
                 want_body: bool = True) -> object:
        """One request.  Returns decoded JSON (None for empty).  Raises OpenCodeHttpError / OpenCodeTransportError."""
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = self._headers({"Content-Type": "application/json"} if data is not None else None)
        connection = self._connect(self.timeout_s if timeout is None else timeout)
        try:
            try:
                connection.request(method, path, body=data, headers=headers)
                response = connection.getresponse()
                status = response.status
                raw = response.read(_MAX_BODY_BYTES + 1)
            except (OSError, http.client.HTTPException) as error:
                raise OpenCodeTransportError(type(error).__name__) from error
        finally:
            connection.close()
        if len(raw) > _MAX_BODY_BYTES:
            raise OpenCodeTransportError("response_too_large")
        if not 200 <= status < 300:
            raise OpenCodeHttpError(status, _decode_loose(raw), method=method, path=path)
        if not want_body or not raw.strip():
            return None
        try:
            return json.loads(raw)
        except ValueError as error:
            raise OpenCodeTransportError("response_not_json") from error

    @staticmethod
    def _session_path(session_id: str) -> str:
        if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
            raise ValueError("opencode_session_id_invalid")
        return "/session/" + quote(session_id, safe="")

    # ------------------------------------------------------------------------------------------------ routes
    def health(self, *, timeout: float | None = None) -> dict:
        """``GET /global/health`` -> ``{"healthy": true, "version": "1.19.5"}``.

        ``timeout`` (s): per-request socket timeout override, used by startup polling so one stalled early
        connection (observed on Linux, ~30 s) does not consume the whole startup budget. Default ``None`` keeps
        the client's configured ``timeout_s``.
        """
        return self._expect_mapping(self._request("GET", "/global/health", timeout=timeout))

    def create_session(self, title: str) -> str:
        """``POST /session`` with an explicit non-default ``title`` (a default title triggers a title-model call).
        Returns the session id (``ses_...``)."""
        if not isinstance(title, str) or not title.strip():
            raise ValueError("opencode_session_title_required")
        session_id = self._expect_mapping(self._request("POST", "/session", {"title": title})).get("id")
        if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
            raise OpenCodeTransportError("session_id_missing")
        return session_id

    def session_exists(self, session_id: str) -> bool:
        """``GET /session/{id}``: True on 200, False on 404; any other failure raises."""
        try:
            self._request("GET", self._session_path(session_id))
        except OpenCodeHttpError as error:
            if error.status == 404:
                return False
            raise
        return True

    def prompt_async(self, session_id: str, text: str | None = None, *, variant: str | None = None,
                     agent: str | None = None, model: tuple[str, str] | Mapping[str, str] | None = None,
                     parts: Sequence[Mapping] | None = None) -> None:
        """``POST /session/{id}/prompt_async`` (204: accepted; the turn runs in the background; while busy the
        message joins at the next step boundary).  No ``system`` field is offered: instructions travel in the config
        ``instructions`` file because ``system`` vanishes after compaction and steers (design R3-5).

        Parameters
        ----------
        text:
            Plain user text, sent as one text part.  Exactly one of ``text`` / ``parts`` is required.
        variant:
            Reasoning-effort variant of the model (must exist in ``config_providers()``); null = model default.
        agent:
            OpenCode agent name; null = the default primary agent (``build``).
        model:
            ``(providerID, modelID)`` or ``{"providerID":..,"modelID":..}``; null = the configured model.
        parts:
            Explicit OpenCode input parts instead of ``text`` (e.g. ``[{"type": "text", "text": "..."}]``).

        Raises ``OpenCodeHttpError`` for any HTTP answer other than 2xx (4xx = definite rejection) and
        ``OpenCodeTransportError`` for socket errors/timeouts (unknown admission).  Never retried.
        """
        if (text is None) == (parts is None):
            raise ValueError("opencode_prompt_text_or_parts")
        body: dict = {"parts": [dict(part) for part in parts] if parts is not None else [{"type": "text", "text": text}]}
        if agent is not None:
            body["agent"] = agent
        if model is not None:
            provider, model_id = (model["providerID"], model["modelID"]) if isinstance(model, Mapping) else model
            body["model"] = {"providerID": provider, "modelID": model_id}
        if variant is not None:
            body["variant"] = variant
        self._request("POST", self._session_path(session_id) + "/prompt_async", body, want_body=False)

    def abort(self, session_id: str, *, timeout_s: float | None = None) -> bool:
        """``POST /session/{id}/abort`` (also kills running tool shells).  True when the server accepted it
        (a no-op on an idle session)."""
        return self._request("POST", self._session_path(session_id) + "/abort", {}, timeout=timeout_s) is True

    def status(self) -> dict:
        """``GET /session/status`` -> ``{}`` (all idle) or ``{session_id: {"type": "busy"|"retry", ...}}``."""
        return self._expect_mapping(self._request("GET", "/session/status"))

    def messages(self, session_id: str) -> list:
        """``GET /session/{id}/message`` -> ``[{"info": {...}, "parts": [...]}]`` in creation order."""
        result = self._request("GET", self._session_path(session_id) + "/message")
        if not isinstance(result, list):
            raise OpenCodeTransportError("response_not_list")
        return result

    def config_providers(self) -> dict:
        """``GET /config/providers`` -> ``{"providers": [...], "default": {...}}`` (model variants live here)."""
        return self._expect_mapping(self._request("GET", "/config/providers"))

    def tool_list(self, provider_id: str, model_id: str) -> list:
        """``GET /experimental/tool?provider=<id>&model=<id>`` -> the tools OpenCode offers that provider/model
        (``[{"id", "description", "parameters"}, ...]``; applies the model-id rule of its tool registry, e.g.
        ``apply_patch`` only for non-``gpt-`` ids; includes the internal ``invalid`` repair tool, which is never
        offered to the model).  Both query parameters are required by the API."""
        result = self._request("GET", "/experimental/tool?provider=" + quote(provider_id, safe="")
                               + "&model=" + quote(model_id, safe=""))
        if not isinstance(result, list):
            raise OpenCodeTransportError("tool_list_not_list")
        return result

    def routes(self) -> list[str]:
        """``GET /doc`` reduced to the sorted ``"METHOD /path"`` operation list."""
        try:
            return routes_from_openapi(self._expect_mapping(self._request("GET", "/doc")))
        except ValueError as error:
            raise OpenCodeTransportError(str(error)) from error

    @staticmethod
    def _expect_mapping(result: object) -> dict:
        if not isinstance(result, dict):
            raise OpenCodeTransportError("response_not_object")
        return result

    # --------------------------------------------------------------------------------------------- event stream
    def events(self) -> Iterator[dict]:
        """One ``GET /event`` connection: yields decoded frames (``{"id","type","properties"}``; the first is
        ``server.connected``, then heartbeats every 10 s).  Comment lines and undecodable frames are skipped
        (counted in ``invalid_event_frames``).  Raises ``OpenCodeHttpError`` for a non-200 answer,
        ``OpenCodeStreamEnded`` at EOF and ``OpenCodeTransportError`` on errors or ``stream_timeout_s`` of silence.
        ``close_streams()`` ends it from another thread.  Events are lost while disconnected (no replay)."""
        connection = self._connect(self.stream_timeout_s)
        with self._lock:
            self._streams.add(connection)
        try:
            try:
                connection.request("GET", "/event", headers=self._headers({"Accept": "text/event-stream"}))
                response = connection.getresponse()
                if response.status != 200:
                    raise OpenCodeHttpError(response.status, _decode_loose(response.read(1 << 16)), method="GET",
                                            path="/event")
                data_lines: list[str] = []
                while True:
                    line = response.fp.readline(1 << 22)
                    if not line:
                        raise OpenCodeStreamEnded("event_stream_eof")
                    text = line.decode("utf-8", "replace").rstrip("\r\n")
                    if text:
                        if text.startswith("data:"):
                            data_lines.append(text[5:].lstrip(" "))
                        continue
                    if not data_lines:
                        continue
                    payload, data_lines = "\n".join(data_lines), []
                    try:
                        frame = json.loads(payload)
                    except ValueError:
                        self.invalid_event_frames += 1
                        continue
                    if isinstance(frame, dict):
                        yield frame
                    else:
                        self.invalid_event_frames += 1
            except OpenCodeError:
                raise
            except (OSError, http.client.HTTPException, ValueError) as error:
                raise OpenCodeTransportError(type(error).__name__) from error
        finally:
            with self._lock:
                self._streams.discard(connection)
            connection.close()

    def event_stream(self, stop: threading.Event, *, on_gap: Callable[[BaseException], None] | None = None,
                     initial_backoff_s: float = 0.1, max_backoff_s: float = 2.0) -> Iterator[dict]:
        """Reconnecting ``/event`` reader for a reader thread: yields frames until ``stop`` is set.

        After every drop or failed connect ``on_gap(error)`` is called (the caller reconciles from ``status()`` and
        ``messages()``, because events missed while disconnected are not replayed), then the reader waits
        ``backoff`` (doubling from ``initial_backoff_s`` to ``max_backoff_s``; reset by the next received frame)
        and reconnects.  ``on_gap`` exceptions propagate.
        """
        backoff = initial_backoff_s
        while not stop.is_set():
            try:
                for frame in self.events():
                    backoff = initial_backoff_s
                    yield frame
                    if stop.is_set():
                        return
            except OpenCodeError as error:
                if stop.is_set():
                    return
                if on_gap is not None:
                    on_gap(error)
            if stop.wait(backoff):
                return
            backoff = min(backoff * 2, max_backoff_s)

    def close_streams(self) -> None:
        """Shut down every open ``/event`` connection (unblocks readers in other threads)."""
        with self._lock:
            streams = list(self._streams)
        for connection in streams:
            sock = connection.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


def _decode_loose(raw: bytes) -> object:
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text[:2000]
