"""Scripted fake ``opencode serve`` (1.19.5 wire shapes) for client and session tests.  Loopback only, no model.

Routes (basic auth ``opencode:<password>``): ``GET /global/health``, ``GET /doc`` (OpenAPI with ``routes``),
``POST /session``, ``GET /session/{id}``, ``POST /session/{id}/prompt_async``, ``POST /session/{id}/abort``,
``GET /session/status``, ``GET /session/{id}/message``, ``GET /config/providers``, ``GET /event`` (SSE).  The event
and message shapes are those captured from the real binary (``tests/fixtures/opencode/events_turn_with_tool.json``).

Behaviour is scripted from the test thread:

* ``prompt_script``: list of per-prompt behaviours consumed in order (default ``"accept"``): ``"accept"`` (204, the
  session turns busy and stays busy until ``finish_turn`` / ``abort``), ``"auto"`` (204 + immediate assistant answer
  ``auto_text`` and idle), ``("status", 400)`` / ``("status", 500)`` (HTTP error, nothing admitted), ``"hang"`` (the
  server reads the request, then stalls ``hang_s`` without answering: client timeout = unknown admission),
  ``"drop"`` (closes the socket without answering, but the prompt *is* recorded as admitted).
* ``finish_turn(sid, text, finish="stop")`` adds the assistant answer and idles the session; ``abort`` marks the
  assistant message aborted exactly like the real server (``MessageAbortedError``, ``session.idle`` twice).
* ``push(event)`` / ``drop_event_streams()`` / ``comment()`` drive ``/event``; events emitted while no stream is
  connected are lost (no replay), like the real server.
* ``busy_gate``: a ``threading.Event`` the ``status`` / ``message`` routes wait on when set to an Event (hold GETs).
* ``health_stalls`` / ``health_stall_s``: stall the first N ``GET /global/health`` requests (default 0: none) for
  ``health_stall_s`` seconds before answering 200, so a client's per-request timeout for that request is exercised.
* ``requests``: list of ``{method, path, body, headers}`` for every HTTP request except the event stream.
"""
from __future__ import annotations

import base64
import itertools
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PIN = "1.19.5"
ROUTES_JSON = Path(__file__).resolve().parents[2] / "auriga" / "cortex" / "agents" / "opencode" / "opencode_routes.json"
PASSWORD = "fake-opencode-password"


def _new_id(prefix: str, counter=itertools.count(1)) -> str:
    return f"{prefix}_{next(counter):012d}FakeIdAbcdef"


class FakeOpenCodeServer:
    def __init__(self, *, password: str = PASSWORD, version: str = PIN, routes: list[str] | None = None,
                 auto_text: str = "done", providers: dict | None = None):
        self.password, self.version, self.auto_text = password, version, auto_text
        self.routes = list(routes) if routes is not None else _checked_in_routes()
        self.providers = providers if providers is not None else {"providers": [], "default": {}}
        self.prompt_script: list = []
        self.hang_s = 2.0
        # Startup-health regression (D1): stall the first `health_stalls` GET /global/health requests for
        # `health_stall_s` before answering 200, reproducing a just-started real `opencode serve` whose first
        # health request can take far longer than later ones.
        self.health_stalls = 0
        self.health_stall_s = 2.0
        self.busy_gate: threading.Event | None = None
        self.requests: list[dict] = []
        self.sessions: dict[str, dict] = {}
        self.busy: dict[str, dict] = {}
        self.event_log: list[dict] = []
        self.lock = threading.RLock()
        self._subscribers: list = []
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url = ""
        self.heartbeat_s: float | None = None

    # ------------------------------------------------------------------------------------------- lifecycle
    def start(self) -> "FakeOpenCodeServer":
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                fake._handle(self)

            do_POST = do_DELETE = do_PUT = do_GET

        class Httpd(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 64

        self._httpd = Httpd(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self.drop_event_streams()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------------------------------ scripted state
    def request_paths(self, method: str | None = None) -> list[str]:
        with self.lock:
            return [item["path"] for item in self.requests if method in (None, item["method"])]

    def push(self, event_type: str, properties: dict | None = None) -> dict:
        """Emit one event to every connected ``/event`` stream (lost if none is connected)."""
        event = {"id": _new_id("evt"), "type": event_type, "properties": properties or {}}
        with self.lock:
            self.event_log.append(event)
            subscribers = list(self._subscribers)
        frame = ("data: " + json.dumps(event) + "\n\n").encode()
        for subscriber in subscribers:
            subscriber.send(frame)
        return event

    def comment(self, text: str = "keepalive") -> None:
        with self.lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.send(f": {text}\n\n".encode())

    def send_raw_event(self, raw: bytes) -> None:
        with self.lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.send(raw)

    def wait_subscribers(self, count: int = 1, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if len(self._subscribers) >= count:
                    return True
            time.sleep(0.01)
        return False

    def drop_event_streams(self) -> None:
        with self.lock:
            subscribers, self._subscribers = list(self._subscribers), []
        for subscriber in subscribers:
            subscriber.close()

    def add_session(self, title: str = "auriga-test") -> str:
        session_id = _new_id("ses")
        with self.lock:
            self.sessions[session_id] = {"id": session_id, "title": title, "messages": []}
        return session_id

    def set_busy(self, session_id: str, status: str = "busy") -> None:
        with self.lock:
            self.busy[session_id] = {"type": status}
        self.push("session.status", {"sessionID": session_id, "status": {"type": status}})

    def set_idle(self, session_id: str) -> None:
        with self.lock:
            self.busy.pop(session_id, None)
        self.push("session.status", {"sessionID": session_id, "status": {"type": "idle"}})
        self.push("session.idle", {"sessionID": session_id})
        self.push("session.idle", {"sessionID": session_id})  # the real server emits it twice

    def _add_message(self, session_id: str, role: str, parent: str | None = None, **info) -> dict:
        message = {"info": {"id": _new_id("msg"), "sessionID": session_id, "role": role,
                            "time": {"created": int(time.time() * 1000)}, **info}, "parts": []}
        if parent:
            message["info"]["parentID"] = parent
        with self.lock:
            self.sessions[session_id]["messages"].append(message)
        self.push("message.updated", {"sessionID": session_id, "info": message["info"]})
        return message

    def _add_part(self, message: dict, part: dict) -> dict:
        part = {"id": _new_id("prt"), "messageID": message["info"]["id"], "sessionID": message["info"]["sessionID"],
                **part}
        with self.lock:
            message["parts"].append(part)
        self.push("message.part.updated", {"sessionID": part["sessionID"], "part": part, "time": int(time.time() * 1000)})
        return part

    def finish_turn(self, session_id: str, text: str = "done", finish: str = "stop", error: dict | None = None) -> dict:
        """Add the assistant answer to the last user message and idle the session."""
        with self.lock:
            users = [m for m in self.sessions[session_id]["messages"] if m["info"]["role"] == "user"]
        extra = {"finish": finish} if error is None else {"error": error}
        message = self._add_message(session_id, "assistant", users[-1]["info"]["id"] if users else None,
                                    modelID="fake-model", providerID="relay", agent="build", mode="build", **extra)
        if text and error is None:
            self._add_part(message, {"type": "step-start"})
            self._add_part(message, {"type": "text", "text": text})
            self._add_part(message, {"type": "step-finish", "reason": finish})
        message["info"]["time"]["completed"] = int(time.time() * 1000)
        self.push("message.updated", {"sessionID": session_id, "info": message["info"]})
        self.set_idle(session_id)
        return message

    def abort_session(self, session_id: str) -> None:
        with self.lock:
            was_busy = session_id in self.busy
            users = [m for m in self.sessions[session_id]["messages"] if m["info"]["role"] == "user"]
        if was_busy:
            error = {"name": "MessageAbortedError", "data": {"message": "Aborted"}}
            self._add_message(session_id, "assistant", users[-1]["info"]["id"] if users else None,
                              modelID="fake-model", providerID="relay", agent="build", mode="build", error=error)
            self.push("session.error", {"sessionID": session_id, "error": error})
        self.set_idle(session_id)

    # ------------------------------------------------------------------------------------------------ HTTP
    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = {"_unparsable": raw[:200].decode("latin1")}
        path = handler.path.split("?", 1)[0]
        expected = "Basic " + base64.b64encode(f"opencode:{self.password}".encode()).decode()
        if path != "/event":
            with self.lock:
                self.requests.append({"method": handler.command, "path": path, "body": body,
                                      "headers": dict(handler.headers)})
        if handler.headers.get("Authorization") != expected:
            return self._json(handler, 401, {"error": "unauthorized"})
        method = handler.command
        parts = path.strip("/").split("/")
        try:
            if method == "GET" and path == "/global/health":
                with self.lock:
                    stall = self.health_stalls > 0
                    if stall:
                        self.health_stalls -= 1
                if stall:
                    time.sleep(self.health_stall_s)
                return self._json(handler, 200, {"healthy": True, "version": self.version})
            if method == "GET" and path == "/doc":
                return self._json(handler, 200, self._openapi())
            if method == "GET" and path == "/config/providers":
                return self._json(handler, 200, self.providers)
            if method == "GET" and path == "/event":
                return self._event_stream(handler)
            if method == "GET" and path == "/session/status":
                self._wait_gate()
                with self.lock:
                    return self._json(handler, 200, dict(self.busy))
            if method == "POST" and path == "/session":
                title = (body or {}).get("title") or "New session"
                session_id = self.add_session(title)
                self.push("session.created", {"sessionID": session_id, "info": {"id": session_id, "title": title}})
                return self._json(handler, 200, {"id": session_id, "title": title, "version": self.version})
            if parts[0] == "session" and len(parts) >= 2:
                session_id = parts[1]
                if session_id not in self.sessions:
                    return self._json(handler, 404, {"name": "NotFoundError", "data": {"message": "Session not found"}})
                tail = "/".join(parts[2:])
                if method == "GET" and not tail:
                    return self._json(handler, 200, {"id": session_id, "title": self.sessions[session_id]["title"]})
                if method == "GET" and tail == "message":
                    self._wait_gate()
                    with self.lock:
                        return self._json(handler, 200, json.loads(json.dumps(self.sessions[session_id]["messages"])))
                if method == "POST" and tail == "abort":
                    self.abort_session(session_id)
                    return self._json(handler, 200, True)
                if method == "POST" and tail == "prompt_async":
                    return self._prompt(handler, session_id, body)
            return self._json(handler, 404, {"error": "not found"})
        except (BrokenPipeError, ConnectionError):
            return

    def _wait_gate(self) -> None:
        gate = self.busy_gate
        if gate is not None:
            gate.wait(10)

    def _prompt(self, handler, session_id: str, body) -> None:
        with self.lock:
            behaviour = self.prompt_script.pop(0) if self.prompt_script else "accept"
        if isinstance(behaviour, tuple) and behaviour[0] == "status":
            return self._json(handler, behaviour[1], {"name": "InvalidRequestError", "data": {"message": "rejected"}})
        parts = (body or {}).get("parts") if isinstance(body, dict) else None
        if not isinstance(parts, list) or not parts:
            return self._json(handler, 400, {"name": "BadRequest", "data": {"message": "parts required"}})
        if behaviour == "hang":
            time.sleep(self.hang_s)
            handler.close_connection = True
            return
        user = self._add_message(session_id, "user", agent="build", model={"providerID": "relay", "modelID": "fake-model"})
        for part in parts:
            self._add_part(user, {"type": part.get("type", "text"), "text": part.get("text", "")})
        self.set_busy(session_id)
        if behaviour == "drop":
            handler.close_connection = True
            handler.connection.shutdown(socket.SHUT_RDWR)
            return
        handler.send_response(204)
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        if behaviour == "auto":
            self.finish_turn(session_id, self.auto_text)

    def _json(self, handler, status: int, payload) -> None:
        data = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def _openapi(self) -> dict:
        paths: dict = {}
        for route in self.routes:
            method, path = route.split(" ", 1)
            paths.setdefault(path, {})[method.lower()] = {"operationId": route}
        return {"openapi": "3.1.1", "info": {"title": "opencode", "version": "1.0.0"}, "paths": paths}

    def _event_stream(self, handler) -> None:
        subscriber = _Subscriber(handler)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        handler.close_connection = True
        with self.lock:
            self._subscribers.append(subscriber)
        self.push("server.connected", {})
        started = time.monotonic()
        while not subscriber.closed.wait(0.05):
            if self.heartbeat_s and time.monotonic() - started >= self.heartbeat_s:
                started = time.monotonic()
                self.push("server.heartbeat", {})
        with self.lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)


class _Subscriber:
    def __init__(self, handler):
        self.handler = handler
        self.closed = threading.Event()
        self._lock = threading.Lock()

    def send(self, data: bytes) -> None:
        with self._lock:
            if self.closed.is_set():
                return
            try:
                self.handler.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.handler.wfile.flush()
            except OSError:
                self.closed.set()

    def close(self) -> None:
        self.closed.set()
        try:
            self.handler.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _checked_in_routes() -> list[str]:
    if ROUTES_JSON.exists():
        return list(json.loads(ROUTES_JSON.read_text())["routes"])
    return []
