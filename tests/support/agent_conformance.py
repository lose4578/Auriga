"""Backend conformance suite shared by every agent backend (design 04 §10.2 items 1-28; plan 07 §4.2).

``AgentBackendConformance`` is a ``unittest`` mixin of contract tests. Each test drives the REAL pinned agent binary
of one backend through the real relay (``ModelRequestProxy`` + the backend's wire adapter) and the real MCP endpoint
(``AurigaMcpServer`` via ``McpToolBridge``); the model is always a scripted loopback fake from
``fake_model_servers`` (never a real model, never the network). The fake Harness is ``ToolBench`` (records every
``tool_handler`` call and answers per tool). A binding module adds one small class::

    @requires_claude_cli
    class ClaudeCodeConformance(AgentBackendConformance, unittest.TestCase):
        backend_kind = "claude_code"
        fake_model_class = FakeAnthropicServer
        ...
        def make_session(self, run_dir, store, tool_handler, fake_model, *, model=MODEL, compact=False): ...
        def classify_request(self, request): ...

Binding contract (class attributes and hooks; each default below documents its exact semantics):

* identity of the binding: ``backend_kind`` (str), ``fake_model_class`` (the ``FakeModelServer`` subclass of the
  backend's wire), ``view`` (a wire view, ``AnthropicMessagesView`` or ``OpenAIResponsesView``: system text, tool
  names, last user text, error flag and normalised usage of one request body; ``flagged_error`` is None on a wire
  without an error flag, and item 5 then checks the host's failed transcript row instead), ``tool_prefix``
  (model-visible prefix of registry tools).
* capability expectations: ``expect_steer``, ``expect_native_wakeup``, ``expect_subagents``; a test of a capability
  the backend lacks checks the refusal or skips with that reason.
* expectations: ``forbidden_tool_names`` (online / human-waiting native tools), ``identity_fields``
  (backend-specific ``runtime/agent`` keys), ``native_call_id_source`` (``callIdSource`` when the agent forwards the
  model's call id, else None), ``auxiliary_prompt`` / ``auxiliary_routes`` (a prompt that makes the agent send
  non-generation requests, item 7), ``private_markers`` (agent-private text that must never reach the provider),
  ``compaction_usage`` + ``compaction_trigger_steps`` (usage that makes an agent started with ``compact=True``
  auto-compact after that many consecutive main steps).
* ``make_session(run_dir, store, tool_handler, fake_model, *, model, compact, sandbox)``: construct (not start) the
  backend session wired to ``fake_model.base_url`` with the host credential in the environment variable
  ``UPSTREAM_KEY_ENV``. ``model``: ``model.name``; ``compact``: configure native auto-compaction (items 18, 24);
  ``sandbox``: ``features.sandbox_enabled`` (item 28; default False).
* ``classify_request(request) -> "main" | "subagent" | "compaction" | "helper"``: which agent loop sent a
  generation request. ``main`` = the actor loop, recognised by tools + ``INSTRUCTIONS_MARKER`` in the system text,
  so an actor request that lost the instructions is classified ``helper`` (item 24 relies on it).
* native tool calls in the model's vocabulary: ``shell_call(command, background=False)``, ``write_call(path,
  content)``, ``subagent_call(prompt)`` -> ``ToolCall``.
* fault injection: ``definite_rejection(session)`` / ``unknown_admission(session)`` (item 10),
  ``route_outside_traffic(proxy_url)`` (item 8: the agent's own environment uses the deny-all proxy),
  ``default_tool_set()`` (item 25: launch without the host denylist, to read the harness default set).
* readers: ``runtime_record(provenance)`` -> ``{"version", "executable", "sha256", "pinned_version"}`` of the real
  agent binary; ``denied_tools(provenance)``; ``init_tool_names(provenance)`` (native tools the agent reported at
  start, in request naming); ``interrupted_result(tool_result)`` (the agent's own "interrupted / outcome unknown"
  answer after a crash, item 21); ``await_history_persisted(session, native_call_id)`` (wait until the agent's own
  resumable history records a tool call; default no-op); ``agent_workdir``, ``mcp_endpoint``, ``agent_pid``
  (defaults read the ``McpSessionMixin`` attributes); ``unknown_usage()``.

A binding marks a precisely diagnosed product defect of its backend by overriding the test with a wrapper next to a
comment naming the defect::

    @unittest.expectedFailure
    def test_...(self):
        super().test_...()

(never ``unittest.expectedFailure(AgentBackendConformance.test_...)``: that marks the shared function object, i.e. the
item of every binding loaded in the same process).

Timing: each test starts one agent process (about 1 s for Claude Code, 0.5-1.5 s for ``opencode serve`` on macOS)
and finishes in seconds (measured on this Mac: Claude Code binding ~60 s, OpenCode ~90 s per wire); the only
slow item (22, a 330 s provider delay) needs ``AURIGA_SLOW_TESTS=1``. Item 28 (sandbox) runs only on a host where
production ``AgentSandbox.prepare`` can start a bubblewrap namespace (``host_capabilities.bubblewrap_namespaces``:
Linux with bubblewrap, unprivileged or through the rootful ``sudo -n`` fallback); elsewhere it is skipped with the
probe's reason, and ``AURIGA_REQUIRE_SANDBOX_TESTS=1`` turns that skip into a failure. Its fixture
(``SandboxBoundary``) is shared with the Codex sandbox test (``tests/test_codex_sandbox_conformance.py``).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from base64 import b64encode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from .fake_mcp_client import FakeMcpClient
from .fake_model_servers import Failure, Gate, Step, ToolCall, Usage, text_step, tool_step
from .requirements import requires_bwrap, requires_slow_tests

__all__ = ["AgentBackendConformance", "AnthropicMessagesView", "DenyAllProxy", "INSTRUCTIONS", "INSTRUCTIONS_MARKER",
           "MODEL", "OpenAIResponsesView", "Router", "SandboxBoundary", "TOOLS", "ToolBench", "UPSTREAM_KEY_ENV",
           "body_text", "make_png"]

# ---- fixed conformance inputs --------------------------------------------------------------------------------
INSTRUCTIONS_MARKER = "AURIGA-CONFORMANCE-INSTRUCTIONS-5c1e"
INSTRUCTIONS = ("Conformance robot instructions. " + INSTRUCTIONS_MARKER +
                "\nUse the registry tools when the user asks for them and answer briefly.")
MODEL = "conformance-model"
UPSTREAM_KEY_ENV = "AURIGA_CONFORMANCE_UPSTREAM_KEY"
HOST_KEY = "host-secret-conformance-upstream-key"
IDENTITY = dict(run_id="conformance", episode_id="ep", env_instance_id="env", env_generation=1)
COMMON_IDENTITY_FIELDS = ("backend", "thread_id", "instance_id", "pid", "model", "provider", "effort", "isolation",
                          "tools_digest", "instructions_digest", "counting", "wire_api", "clean_shutdown")
GENERATION_COUNTING = "delivered_usable_generation_completions_v1"
MODEL_REQUEST_LIMIT = 180
TURN_TIMEOUT_S = 120
INTERRUPT_BOUND_S = 45      # gateway bound on interrupt() -> terminal state (design §2.2)
SETTLE_S = 2.0              # quiet period in which a client retry would have arrived

_X = {"type": "number", "description": "Target x position in m (metres) along the table; default 0; any finite value."}
_EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
# Registry-shaped actor tool definitions (what Workbench.actor_tools() passes to start()).
TOOLS = [
    {"type": "function", "name": "look", "description": "Read-only view of the scene: returns a text summary and one "
     "camera image. Parallel-safe.", "inputSchema": {"type": "object", "properties": {"x": _X},
                                                     "additionalProperties": False}},
    {"type": "function", "name": "act", "description": "Stand-in for a motion that changes the world; ordered after "
     "every earlier call.", "inputSchema": {"type": "object", "properties": {"x": _X}, "additionalProperties": False}},
    {"type": "function", "name": "slow", "description": "Long-running motion stand-in: returns when the robot "
     "finishes (the test decides when).", "inputSchema": _EMPTY},
    {"type": "function", "name": "boom", "description": "Always fails with a runtime error (failure path).",
     "inputSchema": _EMPTY},
    {"type": "function", "name": "camera_fault", "description": "Returns a failed result that still carries one "
     "camera frame.", "inputSchema": _EMPTY},
]
# ToolSpec.parallel of each tool as ToolRegistry.manifest() would publish it (drives MCP readOnlyHint).
TOOL_PARALLEL = {"look": True, "act": False, "slow": False, "boom": False, "camera_fault": False}


def make_png(seed=0, size=16):
    """A deterministic ``size`` x ``size`` RGB PNG (bytes, well below every client re-encoding threshold)."""
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            row += bytes(((x * 13 + seed) % 256, (y * 17 + seed * 7) % 256, (x * y + seed * 29) % 256))
        rows.append(bytes(row))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def image_item(data):
    return {"type": "inputImage", "imageUrl": "data:image/png;base64," + b64encode(data).decode()}


def text_item(text):
    return {"type": "inputText", "text": text}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def body_text(request):
    """Decoded request body (gunzipped by the fake) as text, for absence/presence checks."""
    return request.body.decode("utf-8", "replace") if isinstance(request.body, bytes) else str(request.body)


def process_alive(pid):
    """True while ``pid`` exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = subprocess.run(["/bin/ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True,
                               timeout=5, env={"PATH": os.defpath, "LC_ALL": "C"}).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(state) and not state.startswith("Z")


def kill_quietly(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def inet_endpoints(pids):
    """``lsof`` NAME fields of every inet socket of ``pids`` (e.g. ``127.0.0.1:5->127.0.0.1:6``), or None when
    ``lsof`` is not installed."""
    lsof = shutil.which("lsof", path=os.defpath + os.pathsep + "/usr/sbin" + os.pathsep + "/sbin")
    if not lsof or not pids:
        return None
    completed = subprocess.run([lsof, "-nP", "-a", "-i", "-p", ",".join(str(pid) for pid in pids), "-Fn"],
                               capture_output=True, text=True, timeout=20, env={"PATH": os.defpath, "LC_ALL": "C"})
    return [line[1:] for line in completed.stdout.splitlines() if line.startswith("n")]


def is_loopback_endpoint(name):
    for part in name.split("->"):
        host = part.rsplit(":", 1)[0].strip("[]")
        if not (host.startswith("127.") or host in ("::1", "localhost")):
            return False
    return True


# ---- wire views -------------------------------------------------------------------------------------------------
class AnthropicMessagesView:
    """Reads Anthropic Messages request bodies (Claude Code; OpenCode on ``anthropic_messages``)."""
    name = "anthropic_messages"

    @staticmethod
    def system_text(body):
        system = (body or {}).get("system")
        if isinstance(system, list):
            return "\n".join(block.get("text", "") for block in system if isinstance(block, dict))
        return system if isinstance(system, str) else ""

    @staticmethod
    def tools(body):
        return [tool for tool in (body or {}).get("tools") or [] if isinstance(tool, dict)]

    def tool_names(self, body):
        return [tool.get("name") for tool in self.tools(body)]

    @staticmethod
    def hosted_tools(body):
        """Provider-hosted (server) tools: Anthropic marks them with a versioned ``type`` (``web_search_…``)."""
        return [tool for tool in (body or {}).get("tools") or [] if isinstance(tool, dict)
                and tool.get("type") not in (None, "custom")]

    @staticmethod
    def last_user_text(body):
        """Text blocks of the last ``user`` message (tool_result contents excluded)."""
        for message in reversed((body or {}).get("messages") or []):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            return "\n".join(block.get("text", "") for block in content or []
                             if isinstance(block, dict) and block.get("type") == "text")
        return ""

    @staticmethod
    def flagged_error(result):
        """True when the tool result is marked as an error on the wire."""
        return result.is_error is True

    @staticmethod
    def expected_usage(usage):
        """Normalised counters (``TOKEN_FIELDS`` order semantics) of one scripted ``Usage`` on this wire; a field the
        fake sends as null is unknown (None), never zero."""
        def known(value):
            return value if type(value) is int and value >= 0 else None
        base, read, write, out = (known(usage.input_tokens), known(usage.cache_read_input_tokens),
                                  known(usage.cache_creation_input_tokens), known(usage.output_tokens))
        total_input = None if base is None or read is None or write is None else base + read + write
        return dict(input_tokens=total_input, cached_input_tokens=read, cache_write_input_tokens=write,
                    output_tokens=out, reasoning_output_tokens=None,
                    total_tokens=None if total_input is None or out is None else total_input + out)


class OpenAIResponsesView:
    """Reads OpenAI Responses request bodies (OpenCode on ``openai_responses``). Same interface as
    ``AnthropicMessagesView``."""
    name = "openai_responses"
    SYSTEM_ROLES = ("system", "developer")      # leading ``input`` items that carry the system text

    @staticmethod
    def _content_text(content):
        """Text of a Responses ``content`` (a string, or a list of ``input_text``/``output_text`` parts)."""
        if isinstance(content, str):
            return content
        return "\n".join(part.get("text", "") for part in content or []
                         if isinstance(part, dict) and isinstance(part.get("text"), str))

    def system_text(self, body):
        """``instructions`` plus the content of the leading ``system``/``developer`` input items."""
        body = body or {}
        texts = [body["instructions"]] if isinstance(body.get("instructions"), str) else []
        for item in body.get("input") or []:
            if not (isinstance(item, dict) and item.get("role") in self.SYSTEM_ROLES):
                break
            texts.append(self._content_text(item.get("content")))
        return "\n".join(texts)

    @staticmethod
    def tools(body):
        return [tool for tool in (body or {}).get("tools") or [] if isinstance(tool, dict)]

    def tool_names(self, body):
        """Function/custom tool names; a hosted tool (no name) is listed by its ``type``."""
        return [tool.get("name") or tool.get("type") for tool in self.tools(body)]

    def hosted_tools(self, body):
        """Provider-hosted tools: every tool type other than ``function``/``custom`` (``web_search``, ...)."""
        return [tool for tool in self.tools(body) if tool.get("type") not in ("function", "custom")]

    def last_user_text(self, body):
        """Text of the last ``user`` input item (``function_call_output`` items have no role and are skipped)."""
        for item in reversed((body or {}).get("input") or []):
            if isinstance(item, dict) and item.get("role") == "user":
                return self._content_text(item.get("content"))
        return ""

    @staticmethod
    def flagged_error(result):
        """None: ``function_call_output`` carries no error flag on this wire (a failure is its text only)."""
        return None

    @staticmethod
    def expected_usage(usage):
        """Normalised counters of one scripted ``Usage`` as ``FakeResponsesServer`` renders it (``input_tokens`` =
        input + cache reads, ``cached_tokens`` = cache reads, no cache-write field). A counter that is not a
        non-negative int is unknown (None) and makes every sum it enters None."""
        def known(value):
            return value if type(value) is int and value >= 0 else None
        base, read, out = known(usage.input_tokens), known(usage.cache_read_input_tokens), known(usage.output_tokens)
        total_input = None if base is None or read is None else base + read
        return dict(input_tokens=total_input, cached_input_tokens=read, cache_write_input_tokens=None,
                    output_tokens=out, reasoning_output_tokens=known(usage.reasoning_tokens),
                    total_tokens=None if total_input is None or out is None else total_input + out)


# ---- fake model routing -----------------------------------------------------------------------------------------
class Router:
    """``default`` callable of the fake model server: classifies each generation request with the binding and
    answers from per-kind scripts (``script(kind, *steps)``; a step is a ``Step``, a ``str`` or a callable
    ``request -> Step``). An exhausted script serves ``defaults[kind]``. ``requests[kind]`` lists the requests in
    arrival order."""
    KINDS = ("main", "subagent", "compaction", "helper")

    def __init__(self, classify):
        self.classify = classify
        self.lock = threading.Lock()
        self.scripts = {kind: [] for kind in self.KINDS}
        self.requests = {kind: [] for kind in self.KINDS}
        self.defaults = {"main": text_step("done"), "subagent": text_step("subagent report: done"),
                         "compaction": text_step("<analysis>Scripted conformance session.</analysis>\n<summary>"
                                                 "Conformance summary: the user asked for scripted tool steps; they "
                                                 "ran.</summary>"),
                         "helper": text_step("side-ok")}

    def script(self, kind, *steps):
        with self.lock:
            self.scripts[kind].extend(steps)

    def __call__(self, request):
        kind = self.classify(request)
        with self.lock:
            self.requests[kind].append(request)
            queue = self.scripts[kind]
            item = queue.pop(0) if queue else self.defaults[kind]
        if callable(item) and not isinstance(item, Step):
            item = item(request)
        return Step(text=item) if isinstance(item, str) else item


# ---- fake Harness ---------------------------------------------------------------------------------------------
class ToolBench:
    """The test's ``tool_handler`` (stands in for ``Harness.handle_tool``).

    Records every call as ``{"params", "started", "finished", "entered_at", "epoch"}`` (``started``/``finished``:
    ``time.monotonic()``; ``entered_at``: ``time.time()``, comparable with ``RecordedRequest.received_at``; ``epoch`` =
    ``runtime/image_context_epoch`` when the call entered, i.e. before its result was produced). ``behaviour[tool]`` overrides a tool's answer
    (``params -> result`` or raise). ``slow`` blocks until ``release_slow()`` (or 90 s); ``robot_closed`` is the
    fake robot's close event that also releases it.
    """

    def __init__(self, store):
        self.store = store
        self.calls = []
        self.condition = threading.Condition()
        self.behaviour = {}
        self.slow_release = threading.Event()
        self.image = make_png(1)

    @staticmethod
    def image_for(x):
        """The camera frame ``look`` returns at position ``x`` (distinct per x, so the relay's exact-duplicate image
        reuse never replaces one observation by a reference to another)."""
        return make_png(3 + int(round(float(x) * 100)) % 200)

    def __call__(self, params):
        record = dict(params=params, started=time.monotonic(), entered_at=time.time(), finished=None,
                      epoch=self.store.read_state("runtime", "image_context_epoch"))
        with self.condition:
            self.calls.append(record)
            self.condition.notify_all()
        try:
            handler = self.behaviour.get(params.get("tool")) or getattr(self, "_" + str(params.get("tool")), None)
            if handler is None:
                raise ValueError("conformance_unknown_tool")
            return handler(params)
        finally:
            record["finished"] = time.monotonic()

    def wait_calls(self, count, timeout=60):
        deadline = time.monotonic() + timeout
        with self.condition:
            while len(self.calls) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            return True

    def release_slow(self):
        self.slow_release.set()

    def calls_of(self, tool):
        return [call for call in self.calls if call["params"].get("tool") == tool]

    # default answers
    def _look(self, params):
        x = (params.get("arguments") or {}).get("x", 0)
        return {"success": True, "contentItems": [text_item(f"scene: one red cube at x={x}"),
                                                  image_item(self.image_for(x))]}

    def _act(self, params):
        x = (params.get("arguments") or {}).get("x", 0)
        return {"success": True, "contentItems": [text_item(f"moved to x={x}")]}

    def _slow(self, params):
        self.slow_release.wait(90)
        return {"success": True, "contentItems": [text_item("slow motion finished")]}

    def _boom(self, params):
        raise RuntimeError("conformance_boom_reason: the fake robot refused")

    def _camera_fault(self, params):
        return {"success": False, "contentItems": [text_item("camera fault: frame attached"),
                                                   image_item(make_png(2))]}


class DenyAllProxy:
    """Recording HTTP(S) proxy that refuses everything: CONNECT and absolute-URI requests get 403. ``attempts``
    lists ``(method, target, host header)``."""

    def __init__(self):
        self.attempts = []
        self._httpd = self._thread = None

    def start(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def deny(self):
                proxy.attempts.append((self.command, self.path, self.headers.get("Host")))
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
            do_CONNECT = do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_PATCH = do_OPTIONS = deny

        class Server(ThreadingHTTPServer):
            daemon_threads = True

        self._httpd = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    @property
    def url(self):
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def close(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


# ---- item 28: the sandbox boundary ----------------------------------------------------------------------------
# Non-loopback TCP endpoints (public anycast DNS / HTTPS) the in-namespace probe tries to reach. A network namespace
# that only has ``lo`` refuses them at once (ENETUNREACH); a host network would connect, a filtering one time out.
SANDBOX_OUTSIDE_ADDRESSES = (("1.1.1.1", 443), ("8.8.8.8", 53))
SANDBOX_CONNECT_TIMEOUT_S = 10.0   # probe connect timeout per address, s
SANDBOX_FAIL_FAST_S = 5.0          # a refused connect must fail faster than this, s (a silent drop would time out)
SANDBOX_CLOSE_WAIT_S = 20.0        # after close(): time allowed for every sandbox process to disappear, s
SANDBOX_PROBE_MARKER = "SANDBOX-PROBE-28"

_SANDBOX_PROBE = r'''import hashlib, json, os, socket, subprocess, sys, time
P = json.loads(%(params)r)


def entry(path):
    item = {"exists": os.path.lexists(path)}
    try:
        if os.path.isdir(path):
            item["entries"] = sorted(os.listdir(path))
        else:
            with open(path, "rb") as stream:
                stream.read(1)
        item["readable"] = True
    except OSError as error:
        item.update(readable=False, error=type(error).__name__)
    return item


def connect(host, port, timeout):
    started = time.monotonic()
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        outcome = "connected"
    except OSError as error:
        outcome = "%%s:%%s" %% (type(error).__name__, error.errno)
    return {"outcome": outcome, "elapsed_s": round(time.monotonic() - started, 3)}


report = {"paths": {path: entry(path) for path in P["paths"]}, "harness_files": {}}
for base, _dirs, names in os.walk(P["harness"]):
    for name in names:
        with open(os.path.join(base, name), "rb") as stream:
            report["harness_files"][os.path.relpath(os.path.join(base, name), P["harness"])] = (
                hashlib.sha256(stream.read()).hexdigest())
try:
    open(os.path.join(P["harness"], "write-probe-28"), "w").close()
    report["harness_writable"] = True
except OSError as error:
    report["harness_writable"] = type(error).__name__
report["open_ports"] = {str(port): connect("127.0.0.1", port, 5) for port in P["open_ports"]}
report["closed_ports"] = {str(port): connect("127.0.0.1", port, 5) for port in P["closed_ports"]}
report["outside"] = {"%%s:%%s" %% (host, port): connect(host, port, P["timeout_s"]) for host, port in P["outside"]}
report["interfaces"] = sorted(name for _index, name in socket.if_nameindex())
report["namespaces"] = {kind: os.readlink("/proc/self/ns/" + kind) for kind in ("net", "pid", "mnt")}
# A child left behind in its own session (re-parented away from the agent's tree); close() must still end it.
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", P["sleeper_marker"]],
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 start_new_session=True, close_fds=True)
with open(P["report"], "w") as stream:
    json.dump(report, stream)
print(P["done"])
'''


def proc_start_ticks(pid):
    """Start time of ``pid`` in clock ticks since boot (Linux ``/proc/<pid>/stat`` field 22), or None when the process
    is gone or a zombie. Identifies a process across pid reuse."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return None if fields[0] in ("Z", "X") else int(fields[19])


def proc_pids_matching(*needles):
    """``{pid: start ticks}`` of the live host processes (Linux procfs) whose command line contains one of ``needles``
    (str), excluding this process. Root-owned processes of the rootful sandbox path are included."""
    encoded = [needle.encode() for needle in needles]
    found = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if any(needle in cmdline for needle in encoded):
            ticks = proc_start_ticks(int(entry.name))
            if ticks is not None:
                found[int(entry.name)] = ticks
    return found


def _file_digests(root):
    root = Path(root)
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class SandboxBoundary:
    """Item 28 fixture (Linux only): host-side plants, the in-namespace probe and the host-side checks.

    Shared by ``AgentBackendConformance.test_sandbox_namespace`` and the Codex sandbox test
    (``tests/test_codex_sandbox_conformance.py``). Construct it BEFORE the session (``test``: the running TestCase,
    every cleanup is registered on it); it sets up:

    * a private, short ``XDG_RUNTIME_DIR`` (0700 under /tmp): every backend keeps its run IPC directory at
      ``$XDG_RUNTIME_DIR/auriga/<key>``, which must exist while the sandbox runs and be gone after close;
    * a planted fake RoboDojo root OUTSIDE the run directory, next to it (``<root>/../fake-robodojo``, so mounting the
      run's parent would expose it), holding a stand-in simulator reward file: the simulator source the model must
      not read;
    * a host-only loopback TCP listener (``closed_port``) that is not forwarded into the namespace;
    * a unique ``sleeper_marker`` for a detached child the probe leaves behind.

    ``write_probe(workdir, open_ports)`` writes the probe script into the agent's working directory; the agent's own
    native shell runs ``command`` (``python3 <script>``, the system python of the sandbox userland). The probe writes
    its findings to ``<workdir>/sandbox-probe-28.json`` (read on the host, never through the model, so the egress
    scrub cannot alter them) and prints ``SANDBOX_PROBE_MARKER``. ``assert_inside`` checks (a) ``/harness`` holds
    exactly the published source snapshot, read-only; (b) the checkout, the user's home and the fake RoboDojo root
    are not readable; (c) the only interface is ``lo`` in a namespace of its own, the relay/MCP ports connect, the
    host-only port and the outside addresses fail fast. ``assert_running`` / ``assert_closed`` bracket close() for
    (e): no process of the run (bwrap, supervisor, agent, the detached child) and no IPC directory survives.
    """
    script_name = "sandbox_probe_28.py"
    report_name = "sandbox-probe-28.json"
    harness_mount = "/harness"

    def __init__(self, test, root):
        """``root``: the run directory (inside a private temporary directory owned by the test)."""
        import auriga
        self.test, self.root = test, Path(root).resolve()
        self.checkout = Path(auriga.__file__).resolve().parents[1]
        # Short and private: Unix socket paths below it must fit in 107 bytes.
        self.runtime_dir = Path(tempfile.mkdtemp(prefix="auriga-c28-", dir="/tmp" if os.path.isdir("/tmp") else None))
        test.addCleanup(shutil.rmtree, self.runtime_dir, True)
        os.chmod(self.runtime_dir, 0o700)
        environment = mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(self.runtime_dir)})
        environment.start()
        test.addCleanup(environment.stop)
        self.fake_robodojo = self.root.parent / "fake-robodojo"
        test.addCleanup(shutil.rmtree, self.fake_robodojo, True)
        self.secret = f"SIMULATOR-REWARD-SOURCE-{secrets.token_hex(6)}"
        self.secret_file = self.fake_robodojo / "robodojo" / "envs" / "reward_secret.py"
        self.secret_file.parent.mkdir(parents=True)
        self.secret_file.write_text(f"REWARD = {self.secret!r}\n")
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(4)
        test.addCleanup(self.listener.close)
        self.closed_port = self.listener.getsockname()[1]
        self.sleeper_marker = f"auriga-c28-sleeper-{secrets.token_hex(6)}"
        home = Path.home()
        # Regular files directly in the user's home: never a mount or a mount ancestor, so none may be visible.
        self.home_files = sorted(path for path in home.iterdir() if path.is_file() and not path.is_symlink())[:5]
        self.forbidden_dirs = [self.fake_robodojo, self.checkout / "auriga", Path("~/code/robodojo").expanduser()]
        if os.environ.get("AURIGA_ROBODOJO_ROOT"):
            self.forbidden_dirs.append(Path(os.environ["AURIGA_ROBODOJO_ROOT"]).expanduser())
        self.forbidden_files = [self.secret_file, self.checkout / "auriga" / "runtime" / "harness.py",
                                *[path for path in (self.checkout / "pyproject.toml",) if path.is_file()],
                                *self.home_files]
        self.home = home

    @property
    def command(self):
        return f"python3 {self.script_name}"

    @property
    def ipc_base(self):
        return self.runtime_dir / "auriga"

    def write_probe(self, workdir, open_ports):
        """Write the probe into ``workdir`` (the agent's cwd); ``open_ports``: loopback ports the agent must reach
        (relay, MCP endpoint)."""
        workdir = Path(workdir)
        params = dict(paths=[str(path) for path in [*self.forbidden_dirs, *self.forbidden_files, self.home]],
                      harness=self.harness_mount, open_ports=sorted(int(port) for port in open_ports),
                      closed_ports=[self.closed_port], outside=[list(item) for item in SANDBOX_OUTSIDE_ADDRESSES],
                      timeout_s=SANDBOX_CONNECT_TIMEOUT_S, sleeper_marker=self.sleeper_marker,
                      report=str(workdir / self.report_name), done=SANDBOX_PROBE_MARKER + " done")
        (workdir / self.script_name).write_text(_SANDBOX_PROBE % {"params": json.dumps(params)})
        self.workdir = workdir

    def report(self):
        path = self.workdir / self.report_name
        self.test.assertTrue(path.is_file(), "the in-namespace probe wrote no report (the native shell did not run it)")
        return json.loads(path.read_text())

    def assert_inside(self, report, snapshot):
        """(a)-(c) on the probe ``report``; ``snapshot``: the host copy of the published source (``<run>/private/
        actor-source``) that the sandbox mounts at /harness."""
        test = self.test
        from auriga.cortex.session.actor_source import SOURCE_FILES
        # (a) /harness: exactly the published snapshot, byte-identical to the checkout's files, read-only.
        published = {"auriga/" + name for name in SOURCE_FILES}
        test.assertEqual(set(report["harness_files"]), published, "/harness is not the published source snapshot")
        test.assertEqual(report["harness_files"], _file_digests(snapshot))
        test.assertEqual(report["harness_files"]["auriga/runtime/harness.py"],
                         hashlib.sha256((self.checkout / "auriga" / "runtime" / "harness.py").read_bytes()).hexdigest())
        test.assertIsNot(report["harness_writable"], True, "/harness is writable")
        # (b) no checkout, no home content, no RoboDojo root.
        paths = report["paths"]
        for path in self.forbidden_dirs:
            test.assertFalse(paths[str(path)]["exists"], f"{path} exists inside the sandbox")
        for path in self.forbidden_files:
            test.assertFalse(paths[str(path)]["readable"], f"{path} is readable inside the sandbox")
        visible = set(paths[str(self.home)].get("entries") or ())
        test.assertFalse(visible & {path.name for path in self.home_files}, "the user's home files are visible")
        # (c) a network namespace of its own with loopback only; relay/MCP reachable, nothing else.
        test.assertEqual(report["interfaces"], ["lo"])
        test.assertNotEqual(report["namespaces"]["net"], os.readlink("/proc/self/ns/net"), "host network namespace")
        test.assertNotEqual(report["namespaces"]["pid"], os.readlink("/proc/self/ns/pid"), "host pid namespace")
        for port, result in report["open_ports"].items():
            test.assertEqual(result["outcome"], "connected", f"loopback port {port} (relay/MCP) unreachable: {result}")
        for name, result in [*report["closed_ports"].items(), *report["outside"].items()]:
            test.assertNotEqual(result["outcome"], "connected", f"{name} is reachable from the sandbox")
            test.assertLess(result["elapsed_s"], SANDBOX_FAIL_FAST_S, f"{name} did not fail fast: {result}")
        # The probe's own connect outcome is the evidence. Do not also check that the host listener stayed unreached:
        # other same-user tools on a shared host may probe newly opened loopback ports (seen on a benchmark host).

    def assert_running(self, tree_pids):
        """Before close: the IPC directory exists, the detached child and the run's sandbox processes are visible from
        the host (so ``assert_closed`` is meaningful). ``tree_pids``: the agent process tree seen by the host. Returns
        ``{pid: start ticks}`` of every process that must be gone after close."""
        test = self.test
        ipc = [path for path in self.ipc_base.iterdir()] if self.ipc_base.is_dir() else []
        test.assertEqual(len(ipc), 1, f"expected one run IPC directory under {self.ipc_base}: {ipc}")
        self.ipc = ipc[0]
        test.assertTrue(self.wait_for(lambda: proc_pids_matching(self.sleeper_marker), 15),
                        "the detached child the probe started is not running")
        run = proc_pids_matching(str(self.root), str(self.ipc))
        test.assertTrue(run, "no process of the run is visible from the host")
        tracked = {pid: proc_start_ticks(pid) for pid in tree_pids}
        tracked.update(run)
        tracked.update(proc_pids_matching(self.sleeper_marker))
        return {pid: ticks for pid, ticks in tracked.items() if ticks is not None}

    def assert_closed(self, tracked):
        """After close: none of the ``tracked`` processes (``{pid: start ticks}`` from ``assert_running``), and no
        other process naming the run, its IPC directory or the detached child, survives; the run IPC directory is
        removed."""
        def survivors():
            alive = {pid for pid, ticks in tracked.items() if proc_start_ticks(pid) == ticks}
            return alive | set(proc_pids_matching(str(self.root), str(self.ipc), self.sleeper_marker))
        self.wait_for(lambda: not survivors(), SANDBOX_CLOSE_WAIT_S)
        self.test.assertEqual(sorted(survivors()), [], "sandbox processes survived close")
        self.test.assertFalse(self.ipc.exists(), "the run IPC directory was not removed")
        self.test.assertEqual(list(self.ipc_base.iterdir()) if self.ipc_base.is_dir() else [], [])

    @staticmethod
    def wait_for(predicate, timeout):
        """Poll ``predicate`` every 0.1 s for up to ``timeout`` s; return its last truth value."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.1)
        return bool(predicate())


# ---- the mixin ------------------------------------------------------------------------------------------------
class AgentBackendConformance:
    """Contract tests every agent backend must pass (design 04 §10.2). Mix into ``unittest.TestCase``."""

    # -- binding contract ------------------------------------------------------------------------------------
    backend_kind = None
    fake_model_class = None
    view = AnthropicMessagesView()
    tool_prefix = ""
    expect_steer = True
    expect_native_wakeup = False
    expect_subagents = True
    # Native tool names that must never be published to the model (online access, waiting for a human).
    forbidden_tool_names = frozenset()
    # Backend-specific keys of runtime/agent (in addition to COMMON_IDENTITY_FIELDS).
    identity_fields = ()
    # "native_tool_use_id" when the agent forwards the model's own call id to the MCP endpoint.
    native_call_id_source = None
    # A host prompt that makes the agent send auxiliary (non-generation) requests, and their routes; None = none.
    auxiliary_prompt = None
    auxiliary_routes = ()
    # Agent-private text markers that must never reach the provider (e.g. local image-cache paths).
    private_markers = ()
    # Scripted usage of a main step that, on ``compaction_trigger_steps`` consecutive steps, makes an agent started
    # with ``compact=True`` auto-compact before its next model request.
    compaction_usage = Usage()
    compaction_trigger_steps = 1

    def make_session(self, run_dir, store, tool_handler, fake_model, *, model=MODEL, compact=False, sandbox=False):
        raise NotImplementedError

    def classify_request(self, request):
        raise NotImplementedError

    def shell_call(self, command, *, background=False):
        raise NotImplementedError

    def write_call(self, path, content):
        raise NotImplementedError

    def subagent_call(self, prompt):
        raise NotImplementedError

    def runtime_record(self, provenance):
        """``{"version", "executable", "sha256", "pinned_version"}`` of the real agent binary in ``provenance``."""
        raise NotImplementedError

    def denied_tools(self, provenance):
        """Names of the native tools the host removed (recorded denylist)."""
        raise NotImplementedError

    def init_tool_names(self, provenance):
        """Native tool names the agent itself reported at start (``native_tools.enabled``), in request naming."""
        raise NotImplementedError

    def interrupted_result(self, result):
        """True when ``result`` (a ``ToolResult``) is the agent's own "interrupted / outcome unknown" answer."""
        raise NotImplementedError

    def unknown_usage(self):
        """A ``Usage`` whose input counter the provider does not report."""
        return Usage(input_tokens=None, output_tokens=6)

    @contextlib.contextmanager
    def definite_rejection(self, session):
        """While active, ``run_turn`` fails BEFORE anything reaches the agent (a definite rejection)."""
        raise NotImplementedError
        yield  # pragma: no cover

    @contextlib.contextmanager
    def unknown_admission(self, session):
        """While active, the prompt submission fails in a way that leaves its delivery unknown."""
        raise NotImplementedError
        yield  # pragma: no cover

    @contextlib.contextmanager
    def route_outside_traffic(self, proxy_url):
        """While active, sessions started route every non-loopback request of the agent through ``proxy_url``."""
        raise NotImplementedError
        yield  # pragma: no cover

    @contextlib.contextmanager
    def default_tool_set(self):
        """While active, sessions start without the host's native-tool denylist (the harness default set)."""
        raise NotImplementedError
        yield  # pragma: no cover

    def await_history_persisted(self, session, native_call_id):
        """Block until the agent's own history (needed to resume) records the model's tool call
        ``native_call_id``; default: nothing to wait for."""

    def agent_workdir(self, session):
        """The agent's working directory (its shell's cwd)."""
        return session.actor_dir

    def mcp_endpoint(self, session):
        """``(url, headers)`` of the session's host MCP endpoint."""
        return session.bridge.url, dict(session.bridge.headers)

    def agent_pid(self, session):
        (_, process), = session.owned_processes()
        return process.pid

    def sandbox_ports(self, session):
        """Loopback TCP ports the sandboxed agent must reach (forwarded into its namespace): relay and MCP endpoint."""
        from urllib.parse import urlsplit
        return [urlsplit(session.model_proxy.base_url).port, urlsplit(self.mcp_endpoint(session)[0]).port]

    # -- fixtures ------------------------------------------------------------------------------------------
    def setUp(self):
        super().setUp()
        from auriga.core.storage.store import Store
        directory = tempfile.TemporaryDirectory(prefix="auriga-conformance-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve() / "run"
        self.root.mkdir()
        self.router = Router(self.classify_request)
        self.server = self.fake_model_class([], default=self.router, gate_timeout_s=90).start()
        self.addCleanup(self.server.close)
        environment = mock.patch.dict(os.environ, {UPSTREAM_KEY_ENV: HOST_KEY})
        environment.start()
        self.addCleanup(environment.stop)
        self.store = Store(self.root, IDENTITY)
        self.addCleanup(self.store.close)
        self.bench = ToolBench(self.store)
        self.addCleanup(self.bench.release_slow)
        self.disconnects, self.resets = [], []

    def new_session(self, **options):
        session = self.make_session(self.root, self.store, self.bench, self.server, **options)
        session.on_disconnect = lambda: self.disconnects.append(session.state)
        session.on_context_reset = self.resets.append
        session.tool_parallel_hints = dict(TOOL_PARALLEL)
        session.tool_parallel_safe = lambda tool, arguments: TOOL_PARALLEL.get(tool) is True
        self.addCleanup(self._close_session, session)
        return session

    def _close_session(self, session):
        self.bench.release_slow()        # the fake robot is closed before the session (Workbench order)
        self.server.release_all()
        try:
            session.close()
        except Exception:  # noqa: BLE001 - cleanup only
            pass

    def started_session(self, **options):
        session = self.new_session(**options)
        session.start(TOOLS, INSTRUCTIONS)
        return session

    def name(self, tool):
        return self.tool_prefix + tool

    def turn(self, session, prompt, *, timeout=TURN_TIMEOUT_S, status="completed"):
        turn_id = session.run_turn(prompt)
        self.assertTrue(session.turn_done.wait(timeout), f"turn did not end within {timeout} s")
        if status is not None:
            self.assertEqual((session.turn_result or {}).get("status"), status, session.turn_result)
        return turn_id

    def wait_for(self, predicate, timeout=60, message="condition not reached"):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.assertTrue(predicate(), message)

    def events(self, *types):
        return self.store.events_by_type(list(types), limit=1000)

    def main_requests(self):
        return list(self.router.requests["main"])

    def results_for(self, call_id, requests=None):
        """``(request, ToolResult)`` of the first request whose history carries the result of ``call_id``."""
        for request in requests if requests is not None else self.server.generation_requests():
            for result in self.server.find_tool_results(request):
                if result.call_id == call_id:
                    return request, result
        self.fail(f"no request carries the result of {call_id}")

    def turn_rows(self, turn_id):
        return [row for row in self.store.states("codex_item") if row.get("turn_id") == turn_id]

    def completed_per_turn(self):
        counts = {}
        for event in self.events("agent.turn_completed"):
            turn = event["correlation"].get("turn_id")
            counts[turn] = counts.get(turn, 0) + 1
        return counts

    @staticmethod
    def call_id(prefix="toolu_conf"):
        return f"{prefix}_{secrets.token_hex(6)}"

    # -- 1 ------------------------------------------------------------------------------------------------------
    def test_start_identity(self):
        """Item 1: start() returns a thread id, state idle, runtime/agent carries every identity field."""
        from auriga.core.contracts import digest
        from auriga.cortex.agents.base import read_agent_identity
        session = self.new_session()
        thread = session.start(TOOLS, INSTRUCTIONS)
        self.assertIsInstance(thread, str)
        self.assertTrue(thread)
        self.assertEqual((session.thread_id, session.state), (thread, "idle"))
        identity = self.store.read_state("runtime", "agent")
        for key in COMMON_IDENTITY_FIELDS + tuple(self.identity_fields):
            self.assertIn(key, identity, f"runtime/agent lacks {key}")
        self.assertEqual((identity["backend"], identity["thread_id"], identity["model"], identity["counting"],
                          identity["wire_api"], identity["tools_digest"], identity["clean_shutdown"]),
                         (self.backend_kind, thread, MODEL, GENERATION_COUNTING, self.view.name, digest(TOOLS), False))
        self.assertEqual(identity["pid"], self.agent_pid(session))
        self.assertTrue(identity["instance_id"])
        self.assertEqual(read_agent_identity(self.store), identity)
        self.assertEqual(self.server.generation_requests(), [], "start() must not spend a model call")

    # -- 2 ------------------------------------------------------------------------------------------------------
    def test_admission_returns_before_completion(self):
        """Item 2: run_turn returns on admission while the model call is still blocked."""
        session = self.started_session()
        gate = Gate()
        self.router.script("main", Step(text="ADMITTED-FINAL", gate=gate))
        turn = session.run_turn("Answer once the gate opens.")
        self.assertEqual((session.state, session.turn_id, session.turn_result), ("running", turn, None))
        self.assertFalse(session.turn_done.is_set())
        self.assertTrue(gate.wait_entered(60), "the model call never arrived")
        self.assertEqual(session.state, "running")
        self.assertFalse(session.turn_done.is_set(), "turn_done set while the model is still blocked")
        gate.release()
        self.assertTrue(session.turn_done.wait(TURN_TIMEOUT_S))
        self.assertEqual((session.state, session.turn_result["id"]), ("completed", turn))
        self.assertEqual(session.completed_final_answer(), "ADMITTED-FINAL")

    # -- 3 ------------------------------------------------------------------------------------------------------
    def test_turn_completes_with_final_answer(self):
        """Item 3: completion sets turn_done/state/turn_result; the final answer excludes commentary."""
        session = self.started_session()
        self.router.script("main", tool_step(self.name("look"), {"x": 0.5}, text="COMMENTARY-BEFORE-TOOL"),
                           text_step("FINAL-ANSWER-3"))
        turn = self.turn(session, "Look at the scene, then give the final answer.")
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual((session.state, session.turn_result["id"], session.turn_result["error"]),
                         ("completed", turn, None))
        self.assertEqual(session.completed_final_answer(), "FINAL-ANSWER-3")
        commentary = [row for row in self.turn_rows(turn) if row.get("type") == "agentMessage"
                      and "COMMENTARY-BEFORE-TOOL" in (row.get("text") or "")]
        self.assertTrue(commentary, "commentary text was not recorded")
        self.assertTrue(all(row.get("phase") != "final_answer" for row in commentary))

    # -- 4 ------------------------------------------------------------------------------------------------------
    def test_tool_round_trip(self):
        """Item 4: handler params carry the session's ids; the model gets the scrubbed text and the exact image."""
        session = self.started_session()
        hidden = secrets.token_hex(32)
        self.bench.behaviour["look"] = lambda params: {"success": True, "contentItems": [
            text_item(f"scene: red cube; sha256: {hidden}"), image_item(self.bench.image)]}
        native_id = self.call_id()
        self.router.script("main", tool_step(self.name("look"), {"x": 1.25}, id=native_id),
                           text_step("ROUND-TRIP-DONE"))
        turn = self.turn(session, "Look at the scene.")
        (call,) = self.bench.calls
        params = call["params"]
        self.assertTrue({"threadId", "turnId", "callId", "tool", "arguments"} <= set(params), params)
        self.assertEqual((params["threadId"], params["turnId"], params["tool"], params["arguments"]),
                         (session.thread_id, turn, "look", {"x": 1.25}))
        if self.native_call_id_source:
            self.assertEqual((params["callId"], params["callIdSource"]), (native_id, self.native_call_id_source))
        row = self.store.transcript_item(params["callId"])
        self.assertIsNotNone(row, "no transcript row with id == callId")
        self.assertEqual((row["type"], row["status"], row["success"]), ("dynamicToolCall", "completed", True))
        request, result = self.results_for(native_id)
        self.assertIn("scene: red cube", result.text)
        self.assertNotIn(hidden, body_text(request))
        self.assertEqual([image.sha256 for image in result.images], [sha256(self.bench.image)])
        self.assertEqual(session.completed_final_answer(), "ROUND-TRIP-DONE")

    # -- 5 ------------------------------------------------------------------------------------------------------
    def test_tool_failure_and_failed_result_with_image(self):
        """Item 5: a raising handler reaches the model as a flagged error; a failed result with an image keeps it."""
        session = self.started_session()
        boom, fault = self.call_id(), self.call_id()
        self.router.script("main", tool_step(self.name("boom"), {}, id=boom),
                           tool_step(self.name("camera_fault"), {}, id=fault), text_step("FAILURE-PATHS-DONE"))
        self.turn(session, "Call boom, then camera_fault.")
        _, error_result = self.results_for(boom)
        flagged = self.view.flagged_error(error_result)
        if flagged is None:
            # The wire has no error flag (Responses): the failure reaches the model as its text only, and the host
            # transcript records the call as failed.
            (call,) = self.bench.calls_of("boom")
            row = self.store.transcript_item(call["params"]["callId"])
            self.assertEqual((row["status"], row["success"]), ("failed", False), row)
        else:
            self.assertTrue(flagged, error_result.raw)
        self.assertIn("conformance_boom_reason", error_result.text)
        _, image_result = self.results_for(fault)
        self.assertFalse(self.view.flagged_error(image_result), image_result.raw)
        self.assertIn("camera fault", image_result.text)
        self.assertEqual([image.sha256 for image in image_result.images], [sha256(make_png(2))])
        self.assertEqual(session.completed_final_answer(), "FAILURE-PATHS-DONE")

    # -- 6 ------------------------------------------------------------------------------------------------------
    def test_ordering_barrier(self):
        """Item 6: two parallel-safe calls overlap; the stateful call starts after both (arrival order of ordered
        calls preserved); every result reaches the model."""
        session = self.started_session()

        def slow_look(params):
            time.sleep(0.8)
            return self.bench._look(params)
        self.bench.behaviour["look"] = slow_look
        ids = [self.call_id() for _ in range(3)]
        self.router.script("main", Step(tool_calls=[ToolCall(self.name("look"), {"x": 1}, ids[0]),
                                                    ToolCall(self.name("look"), {"x": 2}, ids[1]),
                                                    ToolCall(self.name("act"), {"x": 3}, ids[2])]),
                           text_step("ORDER-DONE"))
        self.turn(session, "Look twice, then act.")
        by_x = {call["params"]["arguments"]["x"]: call for call in self.bench.calls}
        first, second, stateful = by_x[1], by_x[2], by_x[3]
        self.assertTrue(first["started"] < second["finished"] and second["started"] < first["finished"],
                        "the two parallel-safe calls did not overlap")
        self.assertGreaterEqual(stateful["started"], max(first["finished"], second["finished"]))
        self.assertIs(self.bench.calls[-1], stateful, "the ordered call did not run in arrival order")
        # Every result reaches the next request (the client may list concurrent results in completion order).
        request, _ = self.results_for(ids[2])
        self.assertEqual({result.call_id for result in self.server.find_tool_results(request)} & set(ids), set(ids))
        self.assertEqual(session.completed_final_answer(), "ORDER-DONE")

    # -- 7 ------------------------------------------------------------------------------------------------------
    def test_information_boundary(self):
        """Item 7: planted hashes / long ids / paths never reach any recorded request (incl. auxiliary requests,
        native tool output and the resumed history)."""
        hashes = [secrets.token_hex(32) for _ in range(6)]
        ids = [secrets.token_hex(12) for _ in range(2)]
        secret_path = "/opt/conformance-sim/secret_eval.py"
        instructions = INSTRUCTIONS + f"\nCalibration digest {hashes[5]}."
        self.bench.behaviour["look"] = lambda params: {"success": True, "contentItems": [
            text_item(f"scene ok sha256: {hashes[1]} observation-{ids[1]} raw {hashes[2]}"),
            text_item(json.dumps({"frame": 3, "source_path": secret_path})), image_item(self.bench.image)]}
        session = self.new_session()
        session.start(TOOLS, instructions)
        # Host-side data the agent reads with its own shell (e.g. a program log): only the egress scrub stands
        # between it and the provider. The model's own command carries no planted value.
        (Path(self.agent_workdir(session)) / "sim-log.txt").write_text(f"NATIVE-OUTPUT-{hashes[3]}\n")
        self.router.script("main", tool_step(self.name("look"), {"x": 0}),
                           Step(tool_calls=[self.shell_call("cat sim-log.txt")]), text_step("BOUNDARY-DONE"))
        self.turn(session, f"Inspect the scene. checksum={hashes[0]} for action-{ids[0]}.")
        if self.auxiliary_prompt:
            session.run_turn(self.auxiliary_prompt)
            self.assertTrue(session.turn_done.wait(TURN_TIMEOUT_S))
        thread = session.thread_id
        self.assertTrue(session.close()["clean_shutdown"])
        resumed = self.new_session()
        resumed.start(TOOLS, instructions, thread)
        before = len(self.main_requests())
        self.router.script("main", text_step("BOUNDARY-RESUMED"))
        self.turn(resumed, f"Continue. digest {hashes[4]}")
        planted = [*hashes, f"action-{ids[0]}", f"observation-{ids[1]}", secret_path, *self.private_markers]
        for request in self.server.requests:
            text = body_text(request)
            for value in planted:
                self.assertNotIn(value, text, f"{value!r} reached the provider in {request.method} {request.route}")
        bodies = [body_text(request) for request in self.server.requests]
        self.assertTrue(any("scene ok" in text for text in bodies), "the tool result never reached the model")
        self.assertTrue(any("NATIVE-OUTPUT-" in text for text in bodies), "the shell output never reached the model")
        self.assertIn("Inspect the scene.", body_text(self.main_requests()[before]), "resumed history missing")
        if self.auxiliary_routes:
            self.assertTrue(any(request.route in self.auxiliary_routes for request in self.server.requests),
                            "the auxiliary prompt sent no auxiliary request")

    # -- 8 ------------------------------------------------------------------------------------------------------
    def test_no_outside_information(self):
        """Item 8: no web/hosted tool is published, nothing leaves loopback, the agent shell holds no host secret."""
        host_secrets = {name: f"host-credential-{name.lower()}-{secrets.token_hex(8)}" for name in (
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "HF_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN", "OPENCODE_SERVER_PASSWORD")}
        environment = mock.patch.dict(os.environ, host_secrets)
        environment.start()
        self.addCleanup(environment.stop)
        proxy = DenyAllProxy().start()
        self.addCleanup(proxy.close)
        with self.route_outside_traffic(proxy.url):
            session = self.started_session()
        self.router.script("main", Step(tool_calls=[self.shell_call("env | sort > env-dump.txt")]),
                           text_step("ENV-DONE"))
        self.turn(session, "Dump your environment.")
        endpoints = inet_endpoints([member.pid for member in self._tree(session)])
        dump = (Path(self.agent_workdir(session)) / "env-dump.txt").read_text()
        for value in [*host_secrets.values(), HOST_KEY]:
            self.assertNotIn(value, dump, "a host credential reached the agent environment")
        for request in self.server.requests:
            text = body_text(request)
            for value in [*host_secrets.values(), HOST_KEY]:
                self.assertNotIn(value, text)
        for request in self.server.generation_requests():
            names = set(self.view.tool_names(request.json))
            self.assertFalse(names & set(self.forbidden_tool_names), names & set(self.forbidden_tool_names))
            self.assertEqual(self.view.hosted_tools(request.json), [], "a provider-hosted tool was published")
        self.assertEqual(proxy.attempts, [], "the agent tried to reach a non-loopback host")
        if endpoints is not None:
            self.assertEqual([name for name in endpoints if not is_loopback_endpoint(name)], [],
                             "the agent holds a non-loopback socket")

    def _tree(self, session):
        from auriga.cortex.agents.isolation import process_tree_snapshot
        return process_tree_snapshot(self.agent_pid(session)).members

    # -- 9 ------------------------------------------------------------------------------------------------------
    def test_interrupt_model_and_tool(self):
        """Item 9: interrupt during a blocked model call and during a blocked tool call; a cancel notification for
        an already answered call changes nothing."""
        session = self.started_session()
        gate = Gate()
        self.router.script("main", Step(text="LATE", gate=gate))
        session.run_turn("Wait for the gate.")
        self.assertTrue(gate.wait_entered(60))
        started = time.monotonic()
        session.interrupt()
        self.assertTrue(session.turn_done.wait(INTERRUPT_BOUND_S), "interrupt during a model call did not end the turn")
        self.assertLess(time.monotonic() - started, INTERRUPT_BOUND_S)
        self.assertEqual((session.state, session.turn_result["status"]), ("interrupted", "interrupted"))
        self.assertFalse(session.model_proxy.transport_blocked)
        gate.release()

        slow_id = self.call_id()
        self.router.script("main", tool_step(self.name("slow"), {}, id=slow_id))
        session.run_turn("Run the slow motion.")
        self.assertTrue(self.bench.wait_calls(1, 60), "the slow tool was never called")
        call_id = self.bench.calls[-1]["params"]["callId"]
        started = time.monotonic()
        session.interrupt()
        self.assertTrue(session.turn_done.wait(INTERRUPT_BOUND_S), "interrupt during a tool call did not end the turn")
        self.assertLess(time.monotonic() - started, INTERRUPT_BOUND_S)
        self.assertEqual((session.state, session.turn_result["status"]), ("interrupted", "interrupted"))
        self.assertFalse(session.model_proxy.transport_blocked)
        self.wait_for(lambda: any(event["correlation"].get("item_id") == call_id
                                  for event in self.events("agent.tool_reply_unknown")), 15,
                      "the interrupted tool call was not recorded as uncertain delivery")
        self.assertEqual(self.store.read_state("runtime", "image_context_epoch")["source"],
                         "tool_reply_delivery_uncertain")
        self.bench.release_slow()

        answered = self.call_id()
        self.router.script("main", tool_step(self.name("look"), {"x": 4}, id=answered), text_step("AFTER-INTERRUPTS"))
        self.turn(session, "Look once more.")
        self.assertEqual(session.completed_final_answer(), "AFTER-INTERRUPTS")
        look_id = self.bench.calls[-1]["params"]["callId"]
        rpc_id = self._mcp_request_id(look_id)
        epoch = self.store.read_state("runtime", "image_context_epoch")
        unknown = len(self.events("agent.tool_reply_unknown"))
        url, headers = self.mcp_endpoint(session)
        cancel = FakeMcpClient(url, headers=headers).notify("notifications/cancelled", {"requestId": rpc_id,
                                                                                         "reason": "late cancel"})
        self.assertEqual(cancel.status, 202, "the endpoint did not accept the cancel notification")
        time.sleep(0.5)
        self.assertEqual(len(self.events("agent.tool_reply_unknown")), unknown)
        self.assertEqual(self.store.read_state("runtime", "image_context_epoch"), epoch)
        self.assertEqual(self.disconnects, [])

    def _mcp_request_id(self, call_id):
        for line in (self.root / "agent-raw.jsonl").read_text().splitlines():
            record = json.loads(line)
            message = record.get("message") or {}
            if (record.get("channel") == "mcp" and record.get("direction") == "received"
                    and message.get("method") == "tools/call"
                    and ((message.get("params") or {}).get("_meta") or {}).get("item_id") == call_id):
                return message["id"]
        self.fail(f"no MCP tools/call logged for {call_id}")

    # -- 10 -----------------------------------------------------------------------------------------------------
    def test_rejection_and_unknown_admission(self):
        """Item 10: a definite rejection restores state; unknown admission disconnects once, never retried."""
        from auriga.cortex.agents.base import AgentRejected, AgentTransportUnknown
        session = self.started_session()
        self.router.script("main", text_step("BEFORE-REJECTION"))
        self.turn(session, "Answer.")
        before = (session.state, session.turn_id, dict(session.turn_result))
        with self.definite_rejection(session):
            with self.assertRaises(AgentRejected):
                session.run_turn("REJECTED-PROMPT-10")
        self.assertEqual((session.state, session.turn_id, session.turn_result), before)
        self.assertTrue(session.turn_done.is_set())
        self.router.script("main", text_step("AFTER-REJECTION"))
        self.turn(session, "Answer again.")
        self.assertEqual(session.completed_final_answer(), "AFTER-REJECTION")
        self.assertFalse(any("REJECTED-PROMPT-10" in body_text(r) for r in self.server.requests))
        generations = len(self.server.generation_requests())
        with self.unknown_admission(session):
            with self.assertRaises(AgentTransportUnknown):
                session.run_turn("UNKNOWN-PROMPT-10")
        self.assertEqual(session.state, "disconnected")
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(session.turn_result["status"], "failed")
        with self.assertRaisesRegex(ValueError, "session_not_connected"):
            session.run_turn("Retry?")
        time.sleep(SETTLE_S)
        self.assertEqual(self.disconnects, ["disconnected"])
        self.assertEqual(len(self.server.generation_requests()), generations, "the prompt was retried")

    # -- 11 -----------------------------------------------------------------------------------------------------
    def test_process_death_mid_turn(self):
        """Item 11: killing the agent process mid-turn -> disconnected, synthetic failed turn_result, turn_done."""
        session = self.started_session()
        gate = Gate()
        self.router.script("main", Step(text="NEVER", gate=gate))
        turn = session.run_turn("Wait for the gate.")
        self.assertTrue(gate.wait_entered(60))
        os.kill(self.agent_pid(session), signal.SIGKILL)
        self.assertTrue(session.turn_done.wait(30), "the agent death was not noticed")
        self.assertEqual(session.state, "disconnected")
        self.assertEqual((session.turn_result["id"], session.turn_result["status"]), (turn, "failed"))
        self.assertIn("upstream_connection_lost", session.turn_result["error"]["message"])
        self.assertEqual(self.disconnects, ["disconnected"])
        gate.release()
        self.assertFalse(session.close()["clean_shutdown"])

    # -- 12 -----------------------------------------------------------------------------------------------------
    def test_budget_denial_single_attempt(self):
        """Item 12: with N calls of budget left, call N+1 is denied once (no client retry), the turn fails."""
        remaining = 2
        self.store.set_state("runtime", "model_requests", dict(
            completed_calls=MODEL_REQUEST_LIMIT - remaining, http_attempt_count=MODEL_REQUEST_LIMIT - remaining,
            counting=GENERATION_COUNTING))
        session = self.started_session()
        self.router.script("main", tool_step(self.name("look"), {"x": 1}), tool_step(self.name("look"), {"x": 2}),
                           text_step("NEVER-SERVED"))
        self.turn(session, "Look twice.", status="failed")
        self.assertTrue(session.model_limit_reached.is_set())
        time.sleep(SETTLE_S)
        self.assertEqual(len(self.server.generation_requests()), remaining)
        self.assertEqual(len(self.events("model.model_budget_exhausted")), 1, "the denied call was re-sent")
        self.assertEqual(self.store.read_state("runtime", "model_requests")["completed_calls"], MODEL_REQUEST_LIMIT)

    # -- 13 -----------------------------------------------------------------------------------------------------
    def test_blocked_transport(self):
        """Item 13: a fatal provider error blocks the transport; later requests never reach the provider."""
        session = self.started_session()
        self.router.script("main", Step(failure=Failure("http", status=400, message="fatal fixture error",
                                                        error_type="invalid_request_error")))
        self.turn(session, "Answer.", status="failed")
        self.assertTrue(session.model_proxy.transport_blocked)
        self.assertTrue(self.store.read_state("runtime", "model_requests")["transport_blocked"])
        self.assertEqual(len(self.server.generation_requests()), 1, "a non-retryable error was retried")
        self.turn(session, "Try again.", status="failed")
        time.sleep(SETTLE_S)
        self.assertEqual(len(self.server.generation_requests()), 1, "a request passed the blocked transport")

    # -- 14 -----------------------------------------------------------------------------------------------------
    def test_resume_clean(self):
        """Item 14: close + resume keeps the history; a changed model, tool catalog or backend is refused."""
        first = self.started_session()
        thread = first.thread_id
        self.router.script("main", text_step("FIRST-TURN-ANSWER-14"))
        self.turn(first, "REMEMBER-PROMPT-ALPHA-14: answer.")
        self.assertTrue(first.close()["clean_shutdown"])
        with self.assertRaisesRegex(ValueError, "resume_agent_identity_conflict:model"):
            self.new_session(model="conformance-other-model").start(TOOLS, INSTRUCTIONS, thread)
        with self.assertRaisesRegex(ValueError, "resume_agent_identity_conflict:tools_digest"):
            self.new_session().start(TOOLS[:-1], INSTRUCTIONS, thread)
        recorded = self.store.read_state("runtime", "agent")
        self.store.set_state("runtime", "agent", dict(recorded, backend="conformance_other_backend"))
        with self.assertRaisesRegex(ValueError, "resume_agent_identity_conflict:backend"):
            self.new_session().start(TOOLS, INSTRUCTIONS, thread)
        self.store.set_state("runtime", "agent", recorded)
        second = self.new_session()
        self.assertEqual(second.start(TOOLS, INSTRUCTIONS, thread), thread)
        before = len(self.main_requests())
        self.router.script("main", text_step("SECOND-TURN-ANSWER-14"))
        self.turn(second, "Continue.")
        history = body_text(self.main_requests()[before])
        self.assertIn("REMEMBER-PROMPT-ALPHA-14", history)
        self.assertIn("FIRST-TURN-ANSWER-14", history)
        self.assertEqual(self.resets, [], "a clean resume must not invalidate the model context")

    # -- 15 -----------------------------------------------------------------------------------------------------
    # Sleepers are ``perl -e 'sleep N'``: a portable way to call setsid from a shell, and Claude Code refuses Bash
    # commands that start with ``sleep``.
    def _perl(self):
        perl = shutil.which("perl", path=os.defpath)
        if perl is None:
            self.skipTest("perl not found on the default PATH (used for portable sleeper/setsid shell children)")
        return perl

    def test_close_joins_tools_last_and_is_idempotent(self):
        """Item 15 (a): close stops the whole agent process tree (incl. a native background task), then joins the
        in-flight tool handler only after the (fake) robot closed; close is idempotent."""
        self._perl()
        session = self.started_session()
        self.router.script("main", Step(tool_calls=[self.shell_call("perl -e 'sleep 600'", background=True)]),
                           tool_step(self.name("slow"), {}))
        session.run_turn("Start a background task, then run the slow motion.")
        self.assertTrue(self.bench.wait_calls(1, 60), "the slow tool was never called")
        agent = self.agent_pid(session)
        members = [member.pid for member in self._tree(session)]
        for pid in members:
            self.addCleanup(kill_quietly, pid)
        self.assertIn(agent, members)
        outcome = {}
        closer = threading.Thread(target=lambda: outcome.setdefault("report", session.close()), daemon=True)
        closer.start()
        # The agent is stopped while the tool handler still blocks; close() then waits for the handler, which
        # returns only once the (fake) robot is closed.
        self.wait_for(lambda: not process_alive(agent), 40, "close() did not stop the agent process")
        time.sleep(0.5)
        self.assertTrue(closer.is_alive(), "close() returned before the in-flight tool handler finished")
        self.assertIsNone(self.bench.calls_of("slow")[0]["finished"])
        robot_closed = time.monotonic()
        self.bench.release_slow()
        closer.join(30)
        self.assertFalse(closer.is_alive(), "close() did not return after the robot closed")
        report = outcome["report"]
        self.assertEqual(session.state, "closed")
        self.assertTrue(session.turn_done.is_set())
        steps = list(report["steps"])
        self.assertLess(steps.index("backend"), steps.index("mcp"))
        self.assertLess(steps.index("mcp"), steps.index("tool_workers"))
        self.assertEqual(report["steps"]["tool_workers"], "completed")
        self.assertGreaterEqual(self.bench.calls_of("slow")[0]["finished"], robot_closed)
        self.assertEqual([pid for pid in members if process_alive(pid)], [], "agent processes survived close")
        self.assertFalse(report["clean_shutdown"], "a close during a running turn is not a clean boundary")
        self.assertIs(session.close(), report)

    def test_close_kills_detached_child(self):
        """Item 15 (b): no process survives close, including children a foreground shell call left behind: one
        backgrounded with ``&`` and one that detached itself with setsid (new session and group). Their parent shell
        exits at once, so both are re-parented away from the agent's process tree."""
        self._perl()
        session = self.started_session()
        workdir = Path(self.agent_workdir(session))
        command = ("perl -e 'sleep 600' >/dev/null 2>&1 & echo $! > orphan.pid; "
                   "perl -MPOSIX -e 'POSIX::setsid(); sleep 600' >/dev/null 2>&1 & echo $! > detached.pid")
        self.router.script("main", Step(tool_calls=[self.shell_call(command)]), text_step("DETACHED"))
        self.turn(session, "Leave two sleepers behind.")
        children = {name: int((workdir / f"{name}.pid").read_text().strip()) for name in ("orphan", "detached")}
        for pid in children.values():
            self.addCleanup(kill_quietly, pid)
        self.assertTrue(all(process_alive(pid) for pid in children.values()))
        report = session.close()
        survivors = {name: pid for name, pid in children.items() if process_alive(pid)}
        self.assertEqual(survivors, {}, f"shell children survived close (clean_shutdown={report['clean_shutdown']})")
        self.assertTrue(report["clean_shutdown"], report)

    # -- 16 -----------------------------------------------------------------------------------------------------
    def test_usage_accumulates(self):
        """Item 16: runtime/token_usage is the sum of the served usage; an unknown counter stays None."""
        session = self.started_session()
        self.router.script("main", tool_step(self.name("look"), {"x": 1}, usage=Usage(101, 7, 13, 3)),
                           text_step("USAGE-ONE", usage=Usage(211, 11, 17, 0)))
        self.turn(session, "Look, then answer.")
        expected = self._served_usage()
        usage = self.store.read_state("runtime", "token_usage")
        self.assertEqual(usage["total"], expected)
        self.assertEqual(self.store.read_state("runtime", "model_usage_relay")["calls"],
                         len(self.server.generation_requests()))
        known_output = expected["output_tokens"]
        self.router.script("main", text_step("USAGE-UNKNOWN", usage=self.unknown_usage()))
        self.turn(session, "Answer without input usage.")
        self.router.script("main", text_step("USAGE-AFTER", usage=Usage(5, 3, 0, 0)))
        self.turn(session, "Answer once more.")
        total = self.store.read_state("runtime", "token_usage")["total"]
        self.assertIsNone(total["input_tokens"], "an unknown input counter became a number")
        self.assertIsNone(total["total_tokens"])
        self.assertGreater(total["output_tokens"], known_output)
        self.assertEqual(total["output_tokens"], self._served_usage()["output_tokens"])

    def _served_usage(self):
        """Expected totals: the sum over every generation request the fake answered, unknown -> None."""
        totals = None
        for request in self.server.generation_requests():
            if request.step is None or request.response_status != 200 or request.step.failure is not None:
                continue
            values = self.view.expected_usage(request.step.usage)
            totals = values if totals is None else {key: None if totals[key] is None or values[key] is None
                                                    else totals[key] + values[key] for key in totals}
        return totals

    # -- 17 -----------------------------------------------------------------------------------------------------
    def test_provenance(self):
        """Item 17: provenance() names kind, versions, executable + sha256, wire, counting and prompt channel."""
        session = self.started_session()
        self.router.script("main", text_step("PROVENANCE"))
        self.turn(session, "Answer.")
        provenance = session.provenance()
        text = json.dumps(provenance)
        self.assertEqual((provenance["kind"], provenance["counting"], provenance["wire_api"]),
                         (self.backend_kind, GENERATION_COUNTING, self.view.name))
        runtime = self.runtime_record(provenance)
        self.assertEqual(runtime["version"], runtime["pinned_version"])
        self.assertTrue(os.path.isabs(runtime["executable"]))
        self.assertEqual(runtime["sha256"], sha256(Path(runtime["executable"]).read_bytes()))
        self.assertIn("prompt_channel", provenance)
        self.assertEqual(provenance["capabilities"], session.capabilities.to_dict())
        self.assertEqual((provenance["capabilities"]["steer"], provenance["capabilities"]["native_wakeup"]),
                         (self.expect_steer, self.expect_native_wakeup))
        self.assertTrue(provenance["native_tools"]["disabled"])
        self.assertNotIn(HOST_KEY, text)

    # -- 18 -----------------------------------------------------------------------------------------------------
    def test_compaction_changes_epoch(self):
        """Item 18: a native compaction changes runtime/image_context_epoch before the next tool result is
        produced; with full delivery every observation carries its image."""
        session = self.started_session(compact=True)
        self.assertEqual(session.context_delivery, "full")
        ids = {x: self.call_id() for x in range(1, 5)}
        high = self.compaction_trigger_steps
        self.router.script("main", *[tool_step(self.name("look"), {"x": x}, id=ids[x],
                                               **(dict(usage=self.compaction_usage) if x <= high else {}))
                                     for x in ids], text_step("COMPACTED-DONE"))
        self.turn(session, "Look four times.")
        self.assertTrue(self.router.requests["compaction"], "the agent never compacted")
        compacted_at = self.router.requests["compaction"][0].received_at
        looks = self.bench.calls_of("look")
        after = [call for call in looks if call["entered_at"] > compacted_at]
        self.assertTrue(after, "no tool call after the compaction")
        epoch = after[0]["epoch"]
        self.assertIsNotNone(epoch, "the epoch was not changed before the next tool result was produced")
        self.assertNotEqual(epoch, looks[0]["epoch"])
        self.assertIn(epoch["source"], ("native_context_compaction", "relay_context_discontinuity"))
        self.assertTrue(self.events("agent.compaction"))
        self.assertTrue(set(self.resets) & {"native_context_compaction", "relay_context_discontinuity"}, self.resets)
        x = after[0]["params"]["arguments"]["x"]
        _, result = self.results_for(ids[x])
        self.assertEqual([image.sha256 for image in result.images], [sha256(self.bench.image_for(x))])
        self.assertEqual(session.completed_final_answer(), "COMPACTED-DONE")

    # -- 19 -----------------------------------------------------------------------------------------------------
    def test_self_briefing_via_relay(self):
        """Item 19: self-briefing = exactly one tool-free generation before the first actor request, charged, and
        its notes appear in the first actor prompt (the Workbench sequence)."""
        from auriga.cortex.context import self_briefing as briefing
        session = self.started_session()
        notes = "Grasp the red cube first; check the gripper image before lifting. NOTES-MARKER-19"
        self.router.script("helper", text_step(json.dumps({"notes": notes})))
        agent = session.evidence_agent(model=session.model, base_instructions=briefing.BASE_INSTRUCTIONS,
                                       developer_instructions=briefing.DEVELOPER_INSTRUCTIONS, source=briefing.SOURCE)
        record = briefing.generate(agent, self.store, None, "Task: stack the cubes.")
        self.assertEqual(record["status"], "applied", record)
        self.assertEqual(record["notes"], notes)
        self.assertEqual(self.store.read_state("runtime", "model_requests")["completed_calls"], 1)
        self.router.script("main", text_step("BRIEFED"))
        self.turn(session, "Task: stack the cubes." + briefing.section(record["notes"]))
        generations = self.server.generation_requests()
        first_actor = generations.index(self.main_requests()[0])
        self.assertEqual(first_actor, 1, "exactly one generation must precede the first actor request")
        self.assertEqual(self.view.tool_names(generations[0].json), [])
        self.assertIn("NOTES-MARKER-19", body_text(generations[first_actor]))
        self.assertEqual(self.store.read_state("runtime", "model_requests")["completed_calls"], len(generations))

    # -- 20 -----------------------------------------------------------------------------------------------------
    def test_steer_joins_turn(self):
        """Item 20: steer during a blocked model call and during a blocked tool call joins the same turn."""
        from auriga.cortex.agents.base import AgentCapabilityError
        session = self.started_session()
        if not self.expect_steer:
            with self.assertRaises(AgentCapabilityError):
                session.steer("unsupported")
            return
        hidden = secrets.token_hex(32)
        gate = Gate()
        self.router.script("main", Step(tool_calls=[ToolCall(self.name("look"), {"x": 1})], gate=gate),
                           tool_step(self.name("slow"), {}), text_step("STEER-DONE"))
        turn = session.run_turn("Start the steer test.")
        self.assertTrue(gate.wait_entered(60))
        self.assertEqual(session.steer(f"STEER-ONE-20 checksum={hidden}"), turn)
        gate.release()
        self.assertTrue(self.bench.wait_calls(2, 60), "the slow tool was never called")
        self.assertEqual(session.steer("STEER-TWO-20"), turn)
        self.bench.release_slow()
        self.assertTrue(session.turn_done.wait(TURN_TIMEOUT_S))
        self.assertEqual((session.state, session.turn_id), ("completed", turn), session.turn_result)
        mains = self.main_requests()
        self.assertTrue(any("STEER-ONE-20" in body_text(request) for request in mains[1:]))
        self.assertTrue(any("STEER-TWO-20" in body_text(request) for request in mains[2:]))
        self.assertFalse(any(hidden in body_text(request) for request in self.server.requests))
        steers = [row for row in self.turn_rows(turn) if row.get("type") == "userMessage" and row.get("steer")]
        texts = sorted(row["content"][0]["text"].split()[0] for row in steers)
        self.assertEqual(texts, ["STEER-ONE-20", "STEER-TWO-20"])
        self.assertEqual(self.completed_per_turn(), {turn: 1})
        self.assertEqual(len(self.events("agent.turn_started")), 1)

    # -- 21 -----------------------------------------------------------------------------------------------------
    def test_unclean_resume(self):
        """Item 21: kill the agent during a tool call and resume: the agent's own interrupted result is sent, no
        tool call is re-executed, the image epoch changed."""
        first = self.started_session()
        thread = first.thread_id
        slow_id = self.call_id()
        self.router.script("main", tool_step(self.name("slow"), {}, id=slow_id))
        first.run_turn("Run the slow motion.")
        self.assertTrue(self.bench.wait_calls(1, 60))
        self.await_history_persisted(first, slow_id)
        os.kill(self.agent_pid(first), signal.SIGKILL)
        self.assertTrue(first.turn_done.wait(30))
        self.assertEqual(first.state, "disconnected")
        self.bench.release_slow()
        self.assertFalse(first.close()["clean_shutdown"])
        calls = len(self.bench.calls)
        second = self.new_session()
        second.start(TOOLS, INSTRUCTIONS, thread)
        self.assertEqual(self.store.read_state("runtime", "image_context_epoch")["source"],
                         "agent_resume_after_unclean_exit")
        self.assertIn("agent_resume_after_unclean_exit", self.resets)
        before = len(self.main_requests())
        self.router.script("main", text_step("RECOVERED-21"))
        self.turn(second, "Continue after the crash.")
        _, result = self.results_for(slow_id, self.main_requests()[before:])
        self.assertTrue(self.interrupted_result(result), result.raw)
        self.assertEqual(len(self.bench.calls), calls, "a tool call was re-executed on resume")
        self.assertEqual(second.completed_final_answer(), "RECOVERED-21")

    # -- 22 -----------------------------------------------------------------------------------------------------
    @requires_slow_tests
    def test_long_model_wait(self):
        """Item 22: a provider that answers after 330 s -> one generation request, the turn completes."""
        session = self.started_session()
        self.router.script("main", text_step("LONG-WAIT-DONE", delay_s=330))
        self.turn(session, "Answer slowly.", timeout=480)
        self.assertEqual(len(self.main_requests()), 1)
        self.assertEqual(session.completed_final_answer(), "LONG-WAIT-DONE")

    # -- 23 -----------------------------------------------------------------------------------------------------
    def test_truncated_upstream_no_storm(self):
        """Item 23: a stream without terminal -> at most 1 + max_retries relay attempts for the one client call,
        one well-formed failure at the agent, no client retry storm. (Relay backoff patched to 0 s.)"""
        from auriga.apps.reference.reference_api import upstream
        session = self.started_session()
        self.router.defaults["main"] = Step(text="cut short", failure=Failure("truncated"))
        with mock.patch.object(upstream().joint, "_retry_delay_seconds", return_value=0.0):
            self.turn(session, "Answer.", status="failed")
            time.sleep(SETTLE_S)
        started = self.events("model.model_call_started")
        self.assertEqual(len(started), 1, "the agent re-sent the failed call")
        attempts = len(self.server.generation_requests())
        self.assertGreaterEqual(attempts, 1)
        self.assertLessEqual(attempts, 1 + session.model_proxy.max_retries, "more upstream attempts than the relay "
                             "retry budget allows")
        self.assertEqual(len(self.events("model.model_call_failed")), 1)
        self.assertEqual(self.disconnects, [])
        self.assertEqual(session.state, "failed")

    # -- 24 -----------------------------------------------------------------------------------------------------
    def test_instructions_persist(self):
        """Item 24: the instructions are in the first request, after a steer and after a compaction. The steer is
        sent during the first (gated) model call, whose usage is normal; the compaction usage starts with the next
        step, so the steer joins before the compaction is triggered (an agent that compacts at the step boundary
        where a steer joins would otherwise fold the steer into its summary request)."""
        session = self.started_session(compact=True)
        gate = Gate()
        steps = [tool_step(self.name("look"), {"x": 1}, gate=gate)]
        steps += [tool_step(self.name("look"), {"x": x}, usage=self.compaction_usage)
                  for x in range(2, self.compaction_trigger_steps + 2)]
        self.router.script("main", *steps, text_step("PERSIST-DONE"))
        session.run_turn("Look several times.")
        self.assertTrue(gate.wait_entered(60))
        if self.expect_steer:
            session.steer("PERSIST-STEER-24")
        gate.release()
        self.assertTrue(session.turn_done.wait(TURN_TIMEOUT_S))
        self.assertEqual(session.state, "completed", session.turn_result)
        self.assertEqual(session.completed_final_answer(), "PERSIST-DONE")
        self.assertTrue(self.router.requests["compaction"], "the agent never compacted")
        # Every actor-loop request carries the instructions: one that lost them would be classified "helper" (the
        # main classification requires the marker) and still declare tools.
        self.assertEqual([request for request in self.router.requests["helper"] if self.view.tools(request.json)],
                         [], "an actor request without the Auriga instructions")
        mains = self.main_requests()
        self.assertIn(INSTRUCTIONS_MARKER, self.view.system_text(mains[0].json))
        if self.expect_steer:
            steered = [request for request in mains if "PERSIST-STEER-24" in body_text(request)]
            self.assertTrue(steered, "the steer never reached the model")
            self.assertIn(INSTRUCTIONS_MARKER, self.view.system_text(steered[0].json))
        compaction = self.router.requests["compaction"][0]
        if self.expect_steer:
            self.assertLess(steered[0].received_at, compaction.received_at, "the steer joined after the compaction")
        after = [request for request in mains if request.received_at > compaction.received_at]
        self.assertTrue(after, "no actor request after the compaction")
        self.assertIn(INSTRUCTIONS_MARKER, self.view.system_text(after[0].json))

    # -- 25 -----------------------------------------------------------------------------------------------------
    def test_default_tools(self):
        """Item 25 (a): the published native tools are the harness default set minus exactly the recorded denylist;
        a native file write and a shell run of agent-written code work without a prompt."""
        with self.default_tool_set():
            probe = self.started_session()
        self.router.script("main", text_step("DISCOVERED"))
        self.turn(probe, "Answer.")
        default = self._native_names(self.main_requests()[0])
        probe.close()
        session = self.started_session()
        before = len(self.main_requests())
        script = Path(self.agent_workdir(session)) / "agent_code.sh"
        write, run = self.write_call(script, "echo AGENT-CODE-OK-25\n"), self.shell_call("sh agent_code.sh")
        write.id, run.id = self.call_id(), self.call_id()
        self.router.script("main", Step(tool_calls=[write]), Step(tool_calls=[run]), text_step("TOOLS-DONE"))
        self.turn(session, "Write the script and run it.")
        configured = self._native_names(self.main_requests()[before])
        provenance = session.provenance()
        denied = set(self.denied_tools(provenance))
        self.assertEqual(configured, default - denied, "native tool set drifted from default minus denylist")
        self.assertFalse(configured & set(self.forbidden_tool_names))
        self.assertEqual(script.read_text(), "echo AGENT-CODE-OK-25\n")
        _, written = self.results_for(write.id)
        self.assertFalse(self.view.flagged_error(written), written.raw)
        _, ran = self.results_for(run.id)
        self.assertIn("AGENT-CODE-OK-25", ran.text)
        self.assertEqual(self.events("agent.permission_denied"), [])

    def test_native_tools_provenance(self):
        """Item 25 (b): provenance ``native_tools.enabled`` (read back from the running agent) names exactly the
        native tools the model is offered."""
        session = self.started_session()
        self.router.script("main", text_step("NATIVE-TOOLS-25"))
        self.turn(session, "Answer.")
        published = self._native_names(self.main_requests()[0])
        self.assertEqual(set(self.init_tool_names(session.provenance())), published)

    def _native_names(self, request):
        return {name for name in self.view.tool_names(request.json) if not name.startswith(self.tool_prefix)}

    # -- 26 -----------------------------------------------------------------------------------------------------
    def test_subagent_tool_call(self):
        """Item 26: a subagent's registry tool call runs once through the Harness; child rows carry parent_item_id;
        completions are charged; no discontinuity; the final answer comes from the main agent."""
        if not self.expect_subagents:
            self.skipTest("the backend runs without subagents")
        session = self.started_session()
        spawn = self.subagent_call("SUBAGENT-TASK-26: call the look tool once, then report.")
        spawn.id = self.call_id()
        self.router.script("main", Step(tool_calls=[spawn]), text_step("MAIN-FINAL-26"))
        self.router.script("subagent", tool_step(self.name("look"), {"x": 2.5}), text_step("SUB-REPORT-26"))
        turn = self.turn(session, "Delegate the inspection to a subagent.")
        (call,) = self.bench.calls
        self.assertEqual((call["params"]["turnId"], call["params"]["tool"]), (turn, "look"))
        self.assertGreaterEqual(len(self.router.requests["subagent"]), 2)
        children = [row for row in self.turn_rows(turn) if row.get("parent_item_id")]
        self.assertTrue(children, "no subagent row carries parent_item_id")
        self.assertTrue(all(row.get("phase") != "final_answer" for row in children))
        self.assertEqual(self.store.read_state("runtime", "model_requests")["completed_calls"],
                         len(self.server.generation_requests()))
        self.assertEqual(self.events("model.model_context_discontinuity"), [])
        self.assertEqual(self.resets, [])
        purposes = {event["payload"].get("call_purpose") for event in self.events("model.model_call_completed")}
        self.assertIn("subagent", purposes)
        self.assertEqual(session.completed_final_answer(), "MAIN-FINAL-26")

    # -- 27 -----------------------------------------------------------------------------------------------------
    def test_native_wakeup_adopted(self):
        """Item 27: a background shell finishing after the turn ended opens an adopted turn; run_turn during it
        joins it; one turn_done per turn."""
        if not self.expect_native_wakeup:
            self.skipTest("the backend never starts a turn by itself (capabilities.native_wakeup=False)")
        session = self.started_session()
        gate = Gate()
        self.router.script("main", Step(tool_calls=[self.shell_call("sleep 1; echo BG-FINISHED-27", background=True)]),
                           text_step("HOST-TURN-DONE-27"), Step(text="WAKE-ANSWER-27", gate=gate),
                           text_step("JOINED-ANSWER-27"))
        host = self.turn(session, "Start the background job and end your turn.")
        self.assertEqual(session.completed_final_answer(), "HOST-TURN-DONE-27")
        self.assertTrue(gate.wait_entered(60), "the background job never woke the agent")
        self.wait_for(lambda: session.state == "running" and session.turn_id != host, 15,
                      "the native wake-up turn was not adopted")
        adopted = session.turn_id
        self.assertEqual(session.run_turn("JOIN-PROMPT-27: answer this too."), adopted)
        gate.release()
        self.wait_for(lambda: session.turn_done.is_set() and session.state != "running", TURN_TIMEOUT_S,
                      "the adopted turn did not end")
        self.assertEqual((session.turn_id, session.state), (adopted, "completed"), session.turn_result)
        origins = [event["payload"]["origin"] for event in self.events("agent.turn_started")]
        self.assertEqual(origins, ["host", "native_wakeup"])
        self.assertEqual(self.completed_per_turn(), {host: 1, adopted: 1})
        self.assertTrue(any("JOIN-PROMPT-27" in body_text(request) for request in self.main_requests()[2:]))

    # -- 28 -----------------------------------------------------------------------------------------------------
    @requires_bwrap
    def test_sandbox_namespace(self):
        """Item 28: with ``features.sandbox_enabled`` the agent runs in the bubblewrap namespace. A scripted native
        shell call proves (a) /harness holds the published source snapshot, (b) the checkout, the user's home and a
        planted RoboDojo root are unreadable, (c) only the loopback relay/MCP ports are reachable (outside addresses
        and a host-only loopback port fail fast); (d) a registry tool call still round-trips through the relay and
        MCP; (e) close leaves no bwrap / supervisor / agent / detached child process and removes the IPC directory;
        (f) runtime/agent and provenance record the sandbox isolation."""
        from auriga.cortex.session.agent_sandbox import ACTOR_ISOLATION
        boundary = SandboxBoundary(self, self.root)
        session = self.started_session(sandbox=True)
        # (f) recorded isolation
        self.assertEqual(session.isolation, ACTOR_ISOLATION)
        self.assertEqual(self.store.read_state("runtime", "agent")["isolation"], ACTOR_ISOLATION)
        self.assertEqual(session.provenance()["isolation"], ACTOR_ISOLATION)
        boundary.write_probe(self.agent_workdir(session), self.sandbox_ports(session))
        probe = self.shell_call(boundary.command)
        probe.id, look_id = self.call_id(), self.call_id()
        self.router.script("main", Step(tool_calls=[probe]), tool_step(self.name("look"), {"x": 2.8}, id=look_id),
                           text_step("SANDBOX-DONE-28"))
        turn = self.turn(session, "Probe the sandbox, then look at the scene.")
        self.assertEqual(session.completed_final_answer(), "SANDBOX-DONE-28")
        # (a)-(c) measured inside the namespace by the agent's own shell
        _, ran = self.results_for(probe.id)
        self.assertIn(SANDBOX_PROBE_MARKER, ran.text)
        boundary.assert_inside(boundary.report(), self.root / "private" / "actor-source")
        # (d) the registry tool call went through the forwarded MCP endpoint and its result through the relay
        (call,) = self.bench.calls
        self.assertEqual((call["params"]["threadId"], call["params"]["turnId"], call["params"]["tool"],
                          call["params"]["arguments"]), (session.thread_id, turn, "look", {"x": 2.8}))
        _, looked = self.results_for(look_id)
        self.assertIn("scene: one red cube at x=2.8", looked.text)
        self.assertEqual([image.sha256 for image in looked.images], [sha256(self.bench.image_for(2.8))])
        # (e) teardown
        tracked = boundary.assert_running([member.pid for member in self._tree(session)])
        report = session.close()
        boundary.assert_closed(tracked)
        self.assertTrue(report["clean_shutdown"], report)
