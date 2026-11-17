"""OpenCode agent backend session (design 04 §2.2-§2.3, §3.7, R4.3; plan 07 §1, §3).

``OpenCodeSession`` drives one pinned ``opencode serve`` 1.19.5 for one run over its HTTP API (``OpenCodeClient``).
Registry tools reach OpenCode only through the host MCP endpoint (``McpToolBridge``, model-visible prefix
``auriga_``); model requests only through the run's relay (``ModelRequestProxy`` with the ``opencode`` client profile
of ``AnthropicMessagesWire`` or ``OpenAIResponsesWire`` per ``agent.opencode.upstream.wire_api``; egress scrub = the
wire's ``project_request``). OpenCode sees a placeholder credential, a clean environment and read-only config dirs
(``config.build_launch``). Instructions travel in the config ``instructions`` file, never in a prompt ``system`` field.

Threads
-------
* callers (gateway ``asyncio.to_thread``): the synchronous facade; blocking HTTP calls are never made under the
  session lock.
* ``opencode-events`` (daemon): the ``/event`` reader (``OpenCodeClient.event_stream``, reconnecting). Events are
  hints only (no replay, duplicate ``session.idle``): they feed the live transcript (``project_event``), compaction
  signals, subagent discovery and reconcile requests; nothing on this thread decides a turn.
* ``opencode-reconcile`` (daemon): the single worker that decides turns from authoritative ``GET /session/status`` +
  ``GET /session/{id}/message`` snapshots. Requests are merged while waiting, and a request that arrives while a
  reconcile runs causes one more run afterwards (never dropped: OpenCode's first ``session.idle`` after an abort or a
  provider failure precedes the write of the message error; a second idle follows the write). A watchdog reconcile
  runs every ``watchdog_s`` while a turn runs (covers a silently lost ``/event`` stream).
* ``opencode-watch`` (daemon): ``process.wait()``; an exit that is not part of ``close()`` -> ``disconnected``.
* MCP handler threads + ``agent-tools`` workers (``McpToolBridge``) and relay threads (``_model_event``,
  ``_relay_discontinuity``).

Turn contract (design §2.2, OpenCode row)
-----------------------------------------
* Admission: ``POST /session/{id}/prompt_async`` answered 204. A 4xx is a definite rejection (state restored,
  ``AgentRejected``); a 5xx, timeout or socket error is unknown admission -> ``disconnected``, never retried. The
  host ``userMessage`` row is written BEFORE the request (and withdrawn on a 4xx).
* Turn end: ``verdict.turn_verdict`` on the main session's snapshot (the anchor is the native id of the latest host
  prompt/steer of the turn, re-read under the session lock that ``steer()`` takes, so a steer written after the
  snapshot keeps the turn running); ``IncompleteTurnGuard`` turns a verdict that stays non-terminal while idle into
  ``failed: opencode_turn_incomplete``.
* Steer: ``prompt_async`` while busy (joins at the next step boundary; after a final text OpenCode runs one more
  model call in the same busy period).
* Interrupt: relay block first (``cancel_pending`` plus a reopenable tombstone on the main session and every known
  subagent session, even with no model call in flight) -> ``POST /abort`` on the main session -> abort of every
  subagent session of the turn; the turn ends ``interrupted`` once idle. The next turn reopens those threads first.
  (Probe R4-0: after an abort during a running ``task`` the main agent still makes one more model call; the
  tombstone denies it.)

Subagents (design R4.3): child sessions (``session.created`` with a known ``parentID``, or a ``task`` tool part whose
``state.metadata.sessionId`` names them) are followed: their rows are written with ``thread_id`` = the child session
id and ``parent_item_id`` = the ``task`` tool part that started them, so they are never final-answer text; the turn
verdict reads only the main session.

Nothing written here is model-visible: transcript rows, events, raw logs, identity and close reports are observer data.
"""
import hashlib
import os
import re
import secrets
import socket
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from ....core.contracts import digest, new_id
from ....core.storage.images import store_image_settings
from ...session.actor_source import snapshot_actor_source
from ...session.agent_sandbox import ACTOR_ISOLATION, AgentSandbox
from ...session.model_proxy import upstream_auth_headers_copy
from ...session.wire import AnthropicMessagesWire, OpenAIResponsesWire
from ..base import (AgentCapabilities, AgentCapabilityError, AgentRejected, AgentSession, AgentTransportUnknown,
                    STATE_CLOSED, STATE_DISCONNECTED, STATE_IDLE, STATE_NOT_STARTED, STATE_RUNNING, UNCONNECTED_STATES,
                    read_agent_identity)
from ..isolation import build_snapshot, process_table, stop_agent_tree
from ..mcp_session import GENERATION_COUNTING, STALE_TOOL_STATES, McpSessionMixin, McpToolBridge
from ..transcript import TranscriptWriter
from . import config as opencode_config
from .client import OpenCodeClient, OpenCodeHttpError, is_definite_rejection
from .projector import TOOL_NAME_PREFIX, OpenCodeProjector
from .verdict import IncompleteTurnGuard, TurnVerdict, status_type, turn_verdict

__all__ = ["OpenCodeSession", "BACKEND_KIND", "MCP_SERVER_NAME", "NATIVE_DATA_DIR", "WIRES", "naming_note",
           "compose_instructions"]

BACKEND_KIND = "opencode"
MCP_SERVER_NAME = "auriga"          # OpenCode names MCP tools "<server>_<tool>" -> TOOL_NAME_PREFIX "auriga_"
# Run-relative directory holding OpenCode's own session database (needed for resume; hidden from the actor).
NATIVE_DATA_DIR = "private/opencode/xdg/data"
# Upstream wires the relay can serve for OpenCode (``openai_chat`` has no relay wire adapter in this checkout).
WIRES = {"anthropic_messages": AnthropicMessagesWire, "openai_responses": OpenAIResponsesWire}
SESSION_TITLE = "auriga-actor"      # explicit non-default title (a default title triggers a title-model call)
NATIVE_AGENT = "build"              # OpenCode's default primary agent; every prompt runs it (no ``agent`` field)
_SESSION_ID = re.compile(r"ses[A-Za-z0-9_]*\Z")
_MESSAGE_EVENTS = ("message.updated", "message.part.updated", "message.part.delta")
_IDLE_EVENTS = ("session.idle", "session.error")


def naming_note():
    """The one-sentence tool-name-prefix note appended to the instructions (design §3.5)."""
    return (f"Tool names in these instructions are written without the prefix '{TOOL_NAME_PREFIX}' that your tool "
            "list shows; they refer to the same tools.")


def compose_instructions(instructions):
    """Instructions file text: the (already presented) ``instructions`` followed by the naming note, separated by a
    blank line. An empty ``instructions`` yields the note alone."""
    text = (instructions or "").rstrip()
    return (text + "\n\n" if text else "") + naming_note() + "\n"


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _frame_session(properties):
    """Session id a ``/event`` frame belongs to (``properties.sessionID``, else the info's/part's), or None."""
    return (properties.get("sessionID") or _mapping(properties.get("info")).get("sessionID")
            or _mapping(properties.get("part")).get("sessionID"))


def _task_child(part):
    """``(child session id, task part id)`` of a ``task`` tool part that names its subagent session, else None.
    OpenCode 1.19.5 sets ``state.metadata.sessionId`` while the subagent runs and keeps it when it ends."""
    if not isinstance(part, dict) or part.get("type") != "tool" or part.get("tool") != "task":
        return None
    child = _mapping(_mapping(part.get("state")).get("metadata")).get("sessionId")
    if isinstance(child, str) and _SESSION_ID.match(child) and isinstance(part.get("id"), str):
        return child, part["id"]
    return None


def _task_children(messages):
    """Every ``(child session id, task part id)`` named by the ``task`` tool parts of a message snapshot."""
    found = []
    for message in messages or ():
        for part in _mapping(message).get("parts") or ():
            child = _task_child(part)
            if child is not None:
                found.append(child)
    return found


def _free_loopback_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _SubagentTranscript(TranscriptWriter):
    """Transcript writer of one OpenCode subagent (child) session (design R4.3).

    Rows carry ``thread_id`` = the child session id, ``parent_item_id`` = the ``task`` tool part of the parent session
    that started the child (unless the projector supplies a more specific parent, e.g. a compaction row) and the main
    writer's CURRENT turn when first written. A row keeps the parent it was first written with: a subagent resumed by
    a later ``task`` call (``task_id``) hangs its new rows under the new task part (``parent_item_id`` is re-pointed by
    the session). They are never stamped as final answer (``stamp_final_answer`` reads only the main thread's
    top-level rows). ``OpenCodeProjector(_SubagentTranscript(...))`` projects the child.
    """

    def __init__(self, main, child_id, parent_item_id):
        self._main = main
        super().__init__(main.store, backend=main.backend, thread_id=child_id)
        self.parent_item_id = parent_item_id

    @property
    def turn_id(self):
        return self._main.turn_id

    @turn_id.setter
    def turn_id(self, value):
        """Turns are owned by the main writer; the base constructor's reset is ignored."""

    def _row(self, identifier, fields, *, parent_item_id=None, **keywords):
        with self.lock:
            if parent_item_id is None:
                previous = self.store.transcript_item(identifier, None)
                if not (previous and previous.get("parent_item_id")):
                    parent_item_id = self.parent_item_id
            return super()._row(identifier, fields, parent_item_id=parent_item_id, **keywords)


class OpenCodeSession(McpSessionMixin, AgentSession):
    """OpenCode backend (``agent.backend = "opencode"``). Public contract: ``AgentSession``.

    Constructor (called by ``factory.create_session``): ``run_dir`` (run directory), ``store`` (run Store),
    ``tool_handler`` (``Harness.handle_tool``), ``price_card`` (frozen card or None), ``config`` (``Config``; default
    ``Config.load()``), ``task_planning`` (unused: the decision reminder reaches this backend through
    ``on_context_reset``). Test seams: ``client_factory`` (``(base_url, password) -> OpenCodeClient``-like; default
    ``OpenCodeClient``) and ``launch_options`` (extra keywords for ``config.build_launch``: ``ripgrep``,
    ``managed_config_paths``, ``check_version``, ``base_env``). Constructing a session starts nothing (a replay viewer
    of an OpenCode archive needs no binary).
    """
    backend_kind = BACKEND_KIND
    counting = GENERATION_COUNTING
    # steer: prompt_async while busy. live_effort: the effort travels per prompt as the model ``variant``.
    capabilities = AgentCapabilities(steer=True, live_effort=True, sandbox=True, compaction_signal="native+relay")
    context_delivery = "full"
    tool_name_prefix = TOOL_NAME_PREFIX
    # Timeouts and periods in seconds (plan 07 §3). startup: health poll cap; health_poll: poll interval;
    # request: every non-streaming API call; interrupt: POST /abort from interrupt() (below the gateway's 45 s bound);
    # close_abort: POST /abort during close(); watchdog: reconcile period while a turn runs; incomplete_gap: minimum
    # gap between the two idle observations of IncompleteTurnGuard; thread_join: per daemon thread at close;
    # events_ready: wait for the first /event connection at start (events before it are recovered by reconcile).
    startup_timeout_s = 30.0
    health_poll_s = 0.05
    # per-request cap for each startup health poll (s): a connection made while `opencode serve` is still coming up
    # can stall for the server's own long internal timeout (observed ~30 s on Linux) before answering; capping each
    # poll keeps the deadline below under `startup_timeout_s` regardless, instead of one stalled poll consuming it.
    health_poll_request_timeout_s = 1.0
    request_timeout_s = 30.0
    interrupt_timeout_s = 15.0
    close_abort_timeout_s = 2.0
    watchdog_s = 10.0
    incomplete_gap_s = 2.0
    thread_join_timeout_s = 2.0
    events_ready_timeout_s = 2.0
    # Interval (s) at which the server's process tree is re-snapshotted while it runs (also right after a native bash
    # tool part changes), so the detached tool shells of a server that dies can still be found and stopped at close.
    tree_refresh_s = 2.0
    # Consecutive failed reconciles after which a dead server is declared disconnected (plan 07 §3.9).
    reconcile_failure_limit = 3

    def __init__(self, run_dir, store, tool_handler, *, price_card=None, config=None, task_planning=None,
                 client_factory=None, launch_options=None):
        from ....core.config import Config
        from ....core.presentation.actor_presentation import ActorPresentation
        self.config = config if config is not None else Config.load()
        self.oc_config = self.config.agent.opencode
        self.root = Path(run_dir).resolve()
        self.store, self.tool_handler, self.task_planning = store, tool_handler, task_planning
        self.presentation = ActorPresentation(store)
        self.model = self.config.model.name
        self.active_turn_model = self.model
        self.effort = self.config.model.effort
        upstream = self.oc_config.upstream
        self.provider = upstream.provider
        self.wire_api = upstream.wire_api or ("openai_responses" if upstream.source == "codex_home" else None)
        self.price_card = price_card
        self.sandbox_enabled = bool(self.config.features.sandbox_enabled)
        self.isolation = ACTOR_ISOLATION if self.sandbox_enabled else "disabled"
        self.sandbox = None
        self.ipc = None
        # Design R4.3: a subagent may receive Auriga images in a context the Harness does not track.
        self.context_delivery = "full" if self.oc_config.subagents else self.oc_config.context_delivery
        self.actor_dir, self.private = self.root / "actor", self.root / "private"
        self.wire = None
        self.lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._projector_lock = threading.Lock()
        self.state, self.thread_id, self.turn_id = STATE_NOT_STARTED, None, None
        self.turn_done, self.turn_result = threading.Event(), None
        self.last_activity_at = time.monotonic()
        self.model_limit_reached = threading.Event()
        self.model_proxy = self.bridge = self.transcript = self.projector = self.launch = self.client = None
        self.process = None
        self.closing, self.close_report = False, None
        self._client_factory = client_factory or (
            lambda base_url, password, unix_socket=None: OpenCodeClient(
                base_url, password, timeout_s=self.request_timeout_s, unix_socket=unix_socket))
        self._launch_options = dict(launch_options or {})
        self._started = False
        self._identity = None
        self._instance_id = None          # runtime/agent.instance_id; also the process marker of the server env
        self._teardown_confirmed = True
        self._process_gone = False
        self._interrupt_requested = False
        self._submitting = 0
        self._guard = IncompleteTurnGuard(min_gap_s=self.incomplete_gap_s)
        self._children = {}             # child session id -> {parent_session, parent_item_id, turn_id, projector}
        self._compaction_keys = set()
        self._compaction_pending = False
        self._route_digest = None
        self._native_tools = None
        self._variants = ()
        self._instructions_sha256 = None
        self._tool_names = frozenset()
        self._stop = threading.Event()
        self._tree_wake = threading.Event()
        self._known_tree = {}             # pid -> ProcessInfo of every server-tree member seen while the server ran
        # Native status of the main session from /event and reconciles (busy|retry -> True). ``_busy_seq`` counts busy
        # EVENTS (a reconcile whose idle read predates a newer busy event is stale); ``_idle_after_abort`` is set when
        # a reconcile saw the session idle after the interrupt's abort (a later busy means the loop started after
        # it); ``_recheck_at`` schedules a reconcile (monotonic s) or None.
        self._native_busy = False
        self._busy_seq = 0
        self._idle_after_abort = False
        self._recheck_at = None
        self._events_ready = threading.Event()
        self._stream_connections = 0
        self._gap_open = False
        self._native_busy_unsolicited = False
        self._reconcile_cond = threading.Condition()
        self._reconcile_reasons = []
        self._reconcile_failures = 0
        self._threads = []

    # ---- start ---------------------------------------------------------------------------------------------
    def start(self, tools, instructions, resume_thread_id=None):
        """Start ``opencode serve`` and create (or resume ``resume_thread_id``) the main session; return its id.

        Order: resume identity check -> upstream credentials -> relay -> MCP endpoint -> launch (managed-config
        check, exact ``--version``, sha256, ``rg``, private dirs, read-only instructions file and config dirs, clean
        environment) -> spawn on a free loopback port -> ``/global/health`` (pinned version) -> ``/doc`` route list
        -> effective native tool ids of the ``build`` agent (provenance) -> model variants -> session create / existence check -> history adoption (``projector.sync``) -> identity
        record -> event, reconcile and watch threads -> ``idle``. Any failure closes the session and re-raises (not
        ``disconnected``: no turn was admitted).
        """
        with self.lock:
            if self._started or self.closing:
                raise ValueError("session_already_started")
            self._started = True
        try:
            return self._start(tools, instructions, resume_thread_id)
        except BaseException:
            try:
                self.close()
            except Exception:  # noqa: BLE001 - the start failure is the error to report
                pass
            raise

    def _start(self, tools, instructions, resume_thread_id):
        resume = resume_thread_id is not None
        previous = read_agent_identity(self.store)
        executable = self.oc_config.executable
        if not executable:
            raise ValueError("opencode_executable_not_set: set agent.opencode.executable")
        if resume:
            self._validate_resume(previous, resume_thread_id, tools)
            self._stop_previous_instance(previous)     # a crashed gateway leaves its detached server running
        # Minted before the launch: the server environment carries it as the process marker of this session.
        self._instance_id = new_id("agent")
        upstream_url, upstream_headers = self._upstream()
        self._tool_names = frozenset(tool["name"] for tool in tools)
        self.store.set_state("runtime", "price_card", self.price_card)
        self._start_relay(upstream_url, upstream_auth_headers=upstream_headers,
                          request_projection=self._project_request, default_thread_id=resume_thread_id,
                          image_settings=store_image_settings(self.store, self.config.images.to_dict()))
        self.bridge = McpToolBridge(self, tools, parallel_hints=self.tool_parallel_hints,
                                    server_name=MCP_SERVER_NAME).start()
        text = compose_instructions(self.presentation.present_text(instructions))
        self._instructions_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        port, password = _free_loopback_port(), secrets.token_urlsafe(32)
        self.root.mkdir(parents=True, exist_ok=True)
        options = dict(self._launch_options)
        options.setdefault("base_env", {"LANG": "C.UTF-8"})
        self.launch = opencode_config.build_launch(
            self.root, executable=executable, model=self.model, wire_api=self.wire_api,
            relay_base_url=self.model_proxy.base_url, instructions_text=text, port=port, password=password,
            context_limit_tokens=self.oc_config.context_limit_tokens,
            output_limit_tokens=self.oc_config.output_limit_tokens, mcp_url=self.bridge.url,
            mcp_headers=self.bridge.headers, subagents=self.oc_config.subagents, sandbox=self.sandbox_enabled,
            variants=self._declared_variants(), instance_id=self._instance_id, **options)
        argv, env, unix_socket = list(self.launch.argv), self.launch.env, None
        if self.sandbox_enabled:
            prefix, env, unix_socket = self._prepare_sandbox(port)
            argv = prefix + argv
        with open(self.private / "agent.log", "ab") as log:     # the server keeps its own descriptor
            self.process = subprocess.Popen(
                argv, env=env, cwd=self.launch.cwd, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        base_url = f"http://127.0.0.1:{port}"
        self.client = (self._client_factory(base_url, password) if unix_socket is None
                       else self._client_factory(base_url, password, unix_socket=unix_socket))
        self._await_health()
        self._route_digest = opencode_config.check_route_list(self.client.routes())
        # The build agent's effective native tool ids (tools offered for relay/<model> minus the permission denials), read back from the
        # running server (design R4.1); a malformed reply of the pinned API refuses the start like a route change.
        self._native_tools = opencode_config.read_effective_tool_ids(
            self.client, agent=NATIVE_AGENT, provider=opencode_config.RELAY_PROVIDER_ID, model=self.model)
        self._select_variants()
        if resume:
            if not self.client.session_exists(resume_thread_id):
                raise ValueError("resume_native_session_missing: the OpenCode session of this run is absent")
            thread_id = resume_thread_id
        else:
            thread_id = self.client.create_session(SESSION_TITLE)
            # The relay had to start before the session id existed (the id comes from the server, whose config
            # names the relay); no model request can precede the session, so the attribution fallback is set now.
            self.model_proxy.default_thread_id = thread_id
        self.thread_id = thread_id
        self.model_proxy.register_thread(thread_id, "actor", thread_id)
        self.transcript = TranscriptWriter(self.store, backend=self.backend_kind, thread_id=thread_id)
        self.projector = OpenCodeProjector(self.transcript)
        history = self.client.messages(thread_id)
        children = self._fetch_children(history, all_turns=True)
        with self._projector_lock:
            self.projector.sync(history)
            self._sync_children(children)
        identity = dict(
            backend=self.backend_kind, thread_id=thread_id, instance_id=self._instance_id, pid=self.process.pid,
            model=self.model, provider=self.provider, effort=self.effort, isolation=self.isolation,
            tools_digest=digest(tools), instructions_digest=self._instructions_sha256, counting=self.counting,
            wire_api=self.wire_api, cli_version=opencode_config.OPENCODE_VERSION,
            cli_executable=self.launch.executable, cli_sha256=self.launch.sha256,
            native=dict(data_dir=NATIVE_DATA_DIR), resumed=resume, clean_shutdown=False)
        self.store.set_state("runtime", "agent", identity)
        self._identity = identity
        if resume:
            clean = (previous or {}).get("clean_shutdown") is True
            if not clean:
                # OpenCode repairs its own dangling tool call ("[Tool execution was interrupted]"); evidence delivered
                # before the crash can no longer be assumed in context. The Harness unknown-action gate stays
                # authoritative for physical state.
                self._invalidate_context("agent_resume_after_unclean_exit", marker=new_id("resume"))
            self.transcript.event("agent.history_recovered", dict(clean=clean, delta_history_complete=False,
                                                                  messages=len(history)))
        self._start_threads()
        self._events_ready.wait(self.events_ready_timeout_s)
        with self.lock:
            self.state = STATE_IDLE
        return thread_id

    def _prepare_sandbox(self, port):
        """Build and prepare the bubblewrap sandbox of the server (design R4.4); return
        ``(command prefix, environment, host path of the server's inbound unix socket)``.

        Outbound: the relay and the MCP endpoint (same loopback port numbers inside and outside). Inbound: the
        server's own ``port`` only, as ``<ipc>/inbound/<port>.sock``; no host TCP listener. ``/harness`` is the
        ``actor-source`` snapshot (the Workbench writes it too; the copy is idempotent)."""
        ports = [urlsplit(self.model_proxy.base_url).port, urlsplit(self.bridge.url).port]
        if port in ports:
            raise ValueError("opencode_server_port_conflicts_with_relay_port")
        self._prepare_ipc()
        source = snapshot_actor_source(self.private / "actor-source")
        spec = opencode_config.sandbox_spec(self.launch, harness_source=source)
        self.sandbox = AgentSandbox(self.private / "sandbox", self.ipc, self.actor_dir, ports=ports,
                                    inbound_ports=[port], packages=True, **spec)
        prefix, overrides = self.sandbox.prepare()
        return list(prefix), {"PATH": os.defpath, **overrides}, str(self.sandbox.inbound_socket_path(port))

    def _prepare_ipc(self):
        """Short run-owned IPC directory (Unix socket paths are bounded to 107 bytes), same place as Codex's."""
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        base = (Path(runtime) / "auriga" if runtime and os.path.isabs(runtime)
                else Path(tempfile.gettempdir()) / f"auriga-ipc-{os.getuid()}")
        self.ipc = base / hashlib.sha256(os.fsencode(self.root)).hexdigest()[:24]
        for path in (base, self.ipc):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("unsafe_opencode_ipc_directory")
            path.chmod(0o700)

    def _validate_resume(self, previous, thread_id, tools):
        if not previous:
            raise ValueError("resume_agent_identity_missing: the run has no recorded agent session")
        expected = dict(backend=self.backend_kind, thread_id=thread_id, isolation=self.isolation,
                        tools_digest=digest(tools), model=self.model, provider=self.provider, counting=self.counting,
                        wire_api=self.wire_api, cli_version=opencode_config.OPENCODE_VERSION)
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(f"resume_agent_identity_conflict:{key}: resume with the recorded configuration "
                                 "or start a fresh run")

    def _upstream(self):
        """``(base_url, host credential headers)`` of ``agent.opencode.upstream`` and the relay wire (``self.wire``).

        ``source="explicit"``: ``wire_api`` (anthropic_messages | openai_responses), ``base_url`` and the key read
        from the gateway's environment variable ``api_key_env`` (never stored, never given to the agent).
        ``source="codex_home"``: the Codex credential source (``model.codex_home``) on the Responses wire, resolved
        exactly as the Codex backend resolves it."""
        upstream = self.oc_config.upstream
        if self.wire_api is None:
            raise ValueError("opencode_upstream_incomplete: set agent.opencode.upstream.wire_api, base_url and "
                             "api_key_env, or source=codex_home")
        if self.wire_api not in WIRES:
            raise ValueError(f"opencode_upstream_wire_unsupported:{self.wire_api}: set agent.opencode.upstream."
                             "wire_api to anthropic_messages or openai_responses")
        self.wire = WIRES[self.wire_api](client=BACKEND_KIND)
        if upstream.source == "codex_home":
            import tomllib
            from ...session.codex import _model_provider, _upstream_auth
            if self.wire_api != "openai_responses":
                raise ValueError("opencode_upstream_codex_home_requires_openai_responses")
            source = Path(self.config.model.codex_home).expanduser()
            native = tomllib.loads((source / "config.toml").read_text())
            _name, provider, url = _model_provider(native, source)
            headers = _upstream_auth(provider, source)
            return url.rstrip("/"), upstream_auth_headers_copy(headers, self.wire.auth_header_names)
        if not upstream.base_url:
            raise ValueError("opencode_upstream_base_url_missing: set agent.opencode.upstream.base_url")
        if not upstream.api_key_env or not os.environ.get(upstream.api_key_env):
            raise ValueError(f"opencode_upstream_key_missing: set the environment variable {upstream.api_key_env}")
        key = os.environ[upstream.api_key_env]
        headers = {"Authorization": "Bearer " + key} if upstream.auth_scheme == "bearer" else {"x-api-key": key}
        return upstream.base_url.rstrip("/"), upstream_auth_headers_copy(headers, self.wire.auth_header_names)

    def _project_request(self, request):
        """Relay ``request_projection``: the wire's egress scrub (design §4.4) for this session's tools."""
        return self.wire.project_request(request, self.presentation, registered_tools=self._tool_names,
                                         tool_name_prefix=TOOL_NAME_PREFIX, on_unknown_part=self._egress_unknown_part)

    def _egress_unknown_part(self, type_name):
        """A content block of a type the egress walker does not know (still scrubbed generically): one
        ``model.egress_unknown_part`` event per type, so a client change is visible."""
        known = self.__dict__.setdefault("_unknown_parts", set())
        if type_name in known:
            return
        known.add(type_name)
        try:
            self.store.event("model.egress_unknown_part", dict(type=type_name), source="private_model_transport")
        except Exception:  # noqa: BLE001 - observer record only
            pass

    def _await_health(self):
        """Poll ``GET /global/health`` every ``health_poll_s`` until it answers (``startup_timeout_s`` cap); require
        the pinned version. A server that exits first -> ``opencode_exited_during_startup:<code>``."""
        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(f"opencode_exited_during_startup:{code}")
            remaining = deadline - time.monotonic()
            poll_timeout = max(0.001, min(self.health_poll_request_timeout_s, remaining))
            try:
                health = self.client.health(timeout=poll_timeout)
            except OSError:
                health = None
            except OpenCodeHttpError as error:
                raise RuntimeError(f"opencode_health_http_{error.status}") from error
            if health is not None:
                opencode_config.check_health_version(health)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("opencode_startup_timeout")
            time.sleep(self.health_poll_s)

    def _declared_variants(self):
        """The effort variants the generated config declares for this wire (``config.effort_variants``)."""
        return opencode_config.effort_variants(self.wire_api)

    def _select_variants(self):
        """Effort -> prompt ``variant`` (design §6, review T11 task 2). The generated provider declares one variant per
        ``EFFORT_CHOICES[wire]`` name (``config.effort_variants``); the running server must report exactly that
        declaration for ``relay/<model>`` (``GET /config/providers``), else the start is refused with
        ``OpenCodeConfigError("opencode_variants_mismatch")`` (OpenCode would silently apply nothing for an unknown
        variant). ``model.effort`` must be one of them (``agent_effort_unsupported:opencode:<effort>``; the launch gate
        already refuses it)."""
        providers = self.client.config_providers().get("providers")
        variants = {}
        for provider in providers if isinstance(providers, list) else ():
            if isinstance(provider, dict) and provider.get("id") == opencode_config.RELAY_PROVIDER_ID:
                model = _mapping(_mapping(provider.get("models")).get(self.model))
                variants = _mapping(model.get("variants"))
        declared = self._declared_variants()
        if variants != declared:
            error = opencode_config.OpenCodeConfigError("opencode_variants_mismatch")
            error.declared, error.served = sorted(declared), sorted(variants)
            raise error
        self._variants = tuple(declared)
        if self.effort not in self._variants:
            raise ValueError(f"agent_effort_unsupported:{self.backend_kind}:{self.effort}: choose model.effort from "
                             f"{', '.join(self._variants)}")

    @property
    def effort_choices(self):
        """``model.effort`` values this session accepts (validated by ``set_effort``): the declared variants of the
        configured wire (``config.EFFORT_CHOICES``), () when the wire has none (unsupported; start refuses it)."""
        return self._variants or opencode_config.EFFORT_CHOICES.get(self.wire_api, ())

    def _variant(self):
        """Prompt ``variant`` for the current effort (a declared variant name), or None before the start."""
        return self.effort if self.effort in self._variants else None

    # ---- threads ---------------------------------------------------------------------------------------------
    def _start_threads(self):
        for name, target in (("opencode-events", self._events_loop), ("opencode-reconcile", self._reconcile_loop),
                             ("opencode-watch", self._watch_loop), ("opencode-tree", self._tree_loop)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(thread)
            thread.start()

    def _handler_error(self, where, error):
        try:
            if self.transcript is not None:
                self.transcript.event("agent.native_handler_error", dict(handler=where, error=type(error).__name__,
                                                                         detail=str(error)[:200]))
        except Exception:  # noqa: BLE001
            pass

    def _watch_loop(self):
        process = self.process
        try:
            code = process.wait()
        except Exception:  # noqa: BLE001
            return
        if self.closing or self._stop.is_set():
            return
        try:
            with self._projector_lock:
                for projector in self._projectors():
                    projector.abandon_open_items()
        except Exception as error:  # noqa: BLE001
            self._handler_error("abandon_open_items", error)
        self._disconnected(ProcessLookupError(f"opencode_exited:{code}"))
        self._halt_threads()

    def _halt_threads(self):
        self._stop.set()
        self._tree_wake.set()
        with self._reconcile_cond:
            self._reconcile_cond.notify_all()

    def _refresh_tree(self):
        """Add the server's current descendants to ``_known_tree`` and drop members that no longer exist (one process
        table read). No-op once the server exited (its detached children are no longer reachable by ppid)."""
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            table = process_table()
        except Exception:  # noqa: BLE001 - a failed read only loses one refresh
            return
        present = {(row.pid, row.pgid) for row in table}
        known = {pid: row for pid, row in self._known_tree.items() if (row.pid, row.pgid) in present}
        for row in build_snapshot(process.pid, table).members:
            known[row.pid] = row
        self._known_tree = known

    def _tree_loop(self):
        last = 0.0
        while not self._stop.is_set():
            pause = 0.25 - (time.monotonic() - last)     # at most four table reads per second when woken often
            if pause > 0 and self._stop.wait(pause):
                return
            self._refresh_tree()
            last = time.monotonic()
            self._tree_wake.wait(self.tree_refresh_s)
            self._tree_wake.clear()

    def _note_shell_part(self, part):
        """A native ``bash`` tool part changed: its shell (detached into its own process group) may have just started;
        refresh the known process tree soon."""
        if isinstance(part, dict) and part.get("type") == "tool" and part.get("tool") == "bash":
            self._tree_wake.set()

    def _projectors(self):
        with self.lock:
            children = [entry["projector"] for entry in self._children.values() if entry["projector"] is not None]
        return ([self.projector] if self.projector is not None else []) + children

    # ---- events (hints only) ---------------------------------------------------------------------------------
    def _events_loop(self):
        try:
            for frame in self.client.event_stream(self._stop, on_gap=self._event_gap):
                try:
                    self._on_event(frame)
                except Exception as error:  # noqa: BLE001 - never kill the reader
                    self._handler_error("event", error)
        except Exception as error:  # noqa: BLE001 - the reader ends; reconcile and the watchdog continue
            if not self._stop.is_set():
                self._handler_error("event_stream", error)

    def _event_gap(self, error):
        """``/event`` dropped or a reconnect failed (no replay, so events are being lost): reconcile from the API.

        The first failure of an outage records ``agent.event_stream_gap``; later failed reconnects (every <= 2 s) are
        not recorded again but still request a reconcile while a turn runs; the reconnect requests one more
        (``server.connected``)."""
        if self._stop.is_set() or self.closing:
            return
        first, self._gap_open = not self._gap_open, True
        if first:
            try:
                self.transcript.event("agent.event_stream_gap", dict(error=type(error).__name__))
            except Exception:  # noqa: BLE001
                pass
        if first or self.state == STATE_RUNNING:
            self._request_reconcile("event_stream_gap")

    def _on_event(self, frame):
        kind = frame.get("type")
        if kind == "server.heartbeat":
            return
        self.store.raw_agent("received", frame, channel="native")
        properties = _mapping(frame.get("properties"))
        if kind == "server.connected":
            if self._stream_connections:
                self._request_reconcile("event_stream_reconnected")
            self._stream_connections += 1
            self._gap_open = False
            self._events_ready.set()
            return
        if kind in ("session.created", "session.updated"):
            info = _mapping(properties.get("info"))
            parent = info.get("parentID")
            if isinstance(info.get("id"), str) and parent and (parent == self.thread_id or parent in self._children):
                self._register_child(info["id"], parent)
            return
        session = _frame_session(properties)
        if session is None:
            return
        if session == self.thread_id:
            self._on_main_event(kind, properties, frame)
        elif session in self._children:
            self._on_child_event(session, kind, properties, frame)

    def _on_main_event(self, kind, properties, frame):
        if kind in _MESSAGE_EVENTS:
            self.last_activity_at = time.monotonic()
            with self._projector_lock:
                projected = self.projector.project_event(frame)
            if kind == "message.part.updated":
                self._note_task_part(properties.get("part"), self.thread_id)
                self._note_shell_part(properties.get("part"))
            if projected.kind == "compaction" and projected.data.get("new"):
                self._compaction_signal("compaction_part", key=projected.data.get("part_id"))
        elif kind == "session.compacted":
            self._compaction_signal("session_compacted")
        elif kind == "session.status":
            if status_type(_mapping(properties.get("status"))) == "idle":
                with self.lock:
                    self._native_busy = False
                self._native_busy_unsolicited = False
                self._request_reconcile("idle")
            else:
                self.last_activity_at = time.monotonic()
                with self.lock:
                    self._note_busy()
                    interrupting = self.state == STATE_RUNNING and self._interrupt_requested
                self._note_unsolicited_busy()
                if interrupting:          # a loop that started after the abort (e.g. after a 204, before its loop)
                    self._request_reconcile("busy_after_interrupt")
        elif kind in _IDLE_EVENTS:
            self._request_reconcile("idle" if kind == "session.idle" else "error")

    def _note_busy(self, *, event=True):
        """The main session is busy (call with ``self.lock`` held): the incomplete-turn guard is disarmed
        (``prompt_async`` answers 204 before OpenCode turns busy, so an earlier idle observation is stale). ``event``:
        seen on ``/event`` (counted in ``_busy_seq``) rather than read by a reconcile."""
        self._native_busy = True
        if event:
            self._busy_seq += 1
        self._guard.reset()

    def _native_active(self):
        """``McpSessionMixin`` hook: the main session was last seen busy (a close now is not a clean boundary)."""
        return self._native_busy

    def _note_unsolicited_busy(self):
        """OpenCode 1.19.5 never starts a loop by itself (design R4.2); a busy main session while no host turn runs
        is recorded once per busy period (``agent.unsolicited_native_turn``) and never attributed to a turn."""
        with self.lock:
            if self.state == STATE_RUNNING or self.closing or self._native_busy_unsolicited or self._submitting:
                return
            self._native_busy_unsolicited = True
            host_state = self.state
        try:
            self.transcript.event("agent.unsolicited_native_turn", dict(host_state=host_state))
        except Exception:  # noqa: BLE001
            pass

    def _on_child_event(self, session, kind, properties, frame):
        self.last_activity_at = time.monotonic()
        with self.lock:
            if self.state == STATE_RUNNING:          # a subagent active in this turn (also a resumed one)
                self._children[session]["turn_id"] = self.turn_id
        if kind not in _MESSAGE_EVENTS:
            return
        if kind == "message.part.updated":
            self._note_task_part(properties.get("part"), session)
            self._note_shell_part(properties.get("part"))
        with self.lock:
            projector = self._children[session]["projector"]
        if projector is not None:            # rows wait until the parent task part is known (the next sync has them)
            with self._projector_lock:
                projector.project_event(frame)

    # ---- subagent sessions (design R4.3) ---------------------------------------------------------------------
    def _register_child(self, child_id, parent_session):
        with self.lock:
            if child_id in self._children or child_id == self.thread_id:
                return
            self._children[child_id] = dict(parent_session=parent_session, parent_item_id=None,
                                            turn_id=self.turn_id, projector=None, parts=set())

    def _note_task_part(self, part, owner_session):
        child = _task_child(part)
        if child is not None:
            self._map_child(child[0], child[1], owner_session)

    def _map_child(self, child_id, part_id, owner_session):
        """Link subagent session ``child_id`` to the ``task`` tool part ``part_id`` of ``owner_session`` and create its
        projector (rows: ``thread_id`` = child, ``parent_item_id`` = the task part). A known child named by another
        task part (resumed with ``task_id``) gets that part as the parent of its later rows and joins the current
        turn. Returns True when the link is new."""
        with self.lock:
            if child_id == self.thread_id or self.transcript is None:
                return False
            entry = self._children.setdefault(child_id, dict(parent_session=owner_session, parent_item_id=None,
                                                             turn_id=self.turn_id, projector=None, parts=set()))
            entry.setdefault("parts", set())
            if part_id in entry["parts"]:
                return False
            entry["parts"].add(part_id)
            if entry["projector"] is None:
                entry["projector"] = OpenCodeProjector(_SubagentTranscript(self.transcript, child_id, part_id))
            else:
                entry["projector"].transcript.parent_item_id = part_id
                entry["turn_id"] = self.turn_id
            entry.update(parent_session=owner_session, parent_item_id=part_id)
        try:
            self.transcript.event("agent.subagent_session", dict(session_id=child_id, parent_session_id=owner_session),
                                  item_id=part_id)
        except Exception:  # noqa: BLE001
            pass
        return True

    def _fetch_children(self, messages, *, all_turns=False):
        """Link the subagent sessions named by the ``task`` parts of the main snapshot ``messages`` and fetch the
        snapshots of the linked children of the current turn (every linked child with ``all_turns``), following
        nested subagents named in those snapshots. Returns ``[(child id, messages)]``; a child whose fetch fails is
        recorded and skipped (observer rows only: the verdict reads the main session)."""
        for child, part in _task_children(messages):
            self._map_child(child, part, self.thread_id)
        with self.lock:
            queue = [child for child, entry in self._children.items() if entry["projector"] is not None
                     and (all_turns or entry["turn_id"] == self.turn_id)]
        fetched, done = [], set()
        while queue:
            child = queue.pop(0)
            if child in done:
                continue
            done.add(child)
            try:
                snapshot = self.client.messages(child)
            except Exception as error:  # noqa: BLE001
                self._handler_error("subagent_messages", error)
                continue
            fetched.append((child, snapshot))
            for grandchild, part in _task_children(snapshot):
                self._map_child(grandchild, part, child)
                queue.append(grandchild)
        return fetched

    def _sync_children(self, fetched):
        """Project fetched subagent snapshots (call with ``_projector_lock`` held)."""
        for child, snapshot in fetched:
            with self.lock:
                projector = self._children.get(child, {}).get("projector")
            if projector is not None:
                projector.sync(snapshot)

    # ---- compaction ------------------------------------------------------------------------------------------
    def _compaction_signal(self, signal, key=None):
        """Native compaction evidence of the main session: once per compaction (a compaction part, identified by
        ``key``, and the later ``session.compacted`` of the same compaction signal once): new
        ``runtime/image_context_epoch``, ``agent.compaction`` and the decision reminder (``on_context_reset``)."""
        with self.lock:
            if key is None:
                if self._compaction_pending:
                    self._compaction_pending = False
                    return
            else:
                if key in self._compaction_keys:
                    return
                self._compaction_keys.add(key)
                self._compaction_pending = True
        try:
            self._invalidate_context("native_context_compaction", reason="native_context_compaction", signal=signal,
                                     marker=new_id("compaction"))
            self.transcript.event("agent.compaction", dict(signal=signal, source="native"))
        except Exception as error:  # noqa: BLE001
            self._handler_error("compaction", error)

    # ---- reconciliation (authoritative) ----------------------------------------------------------------------
    def _request_reconcile(self, reason):
        with self._reconcile_cond:
            if reason not in self._reconcile_reasons:
                self._reconcile_reasons.append(reason)
            self._reconcile_cond.notify_all()

    def _reconcile_loop(self):
        last_run = time.monotonic()
        while not self._stop.is_set():
            with self._reconcile_cond:
                if not self._reconcile_reasons and not self._stop.is_set():
                    now = time.monotonic()
                    timeout = self.watchdog_s - (now - last_run if self.state == STATE_RUNNING else 0.0)
                    if self._recheck_at is not None:
                        timeout = min(timeout, self._recheck_at - now)
                    self._reconcile_cond.wait(max(0.05, timeout))
                reasons, self._reconcile_reasons = self._reconcile_reasons, []
            if self._stop.is_set():
                return
            if not reasons:
                now = time.monotonic()
                if self._recheck_at is not None and now >= self._recheck_at:
                    reasons = ["recheck"]
                elif self.state == STATE_RUNNING and now - last_run >= self.watchdog_s:
                    reasons = ["watchdog"]
                else:
                    continue
            self._recheck_at = None
            last_run = time.monotonic()
            try:
                self._reconcile(reasons)
                self._reconcile_failures = 0
            except Exception as error:  # noqa: BLE001 - retried by the next request or the watchdog
                self._reconcile_failed(error)

    def _reconcile_failed(self, error):
        if self.closing or self._stop.is_set():
            return
        self._reconcile_failures += 1
        try:
            self.transcript.event("agent.reconcile_failed", dict(error=type(error).__name__, detail=str(error)[:200],
                                                                 consecutive=self._reconcile_failures))
        except Exception:  # noqa: BLE001
            pass
        process = self.process
        if (self._reconcile_failures >= self.reconcile_failure_limit and process is not None
                and process.poll() is not None):
            self._disconnected(ProcessLookupError(f"opencode_exited:{process.returncode}"))
            self._halt_threads()

    def _reconcile(self, reasons):
        """One authoritative pass: status -> (idle only) messages -> projection (main + subagents) -> verdict.

        While the main session is busy nothing else is read. While no turn runs the snapshot is still projected
        (rows OpenCode writes after the turn's terminal, e.g. the message error that follows an abort's first idle).
        """
        with self.lock:
            if self.closing or self.state in STALE_TOOL_STATES or self.thread_id is None:
                return
            running, turn_id, seq = self.state == STATE_RUNNING, self.turn_id, self._busy_seq
        entry = self.client.status().get(self.thread_id)
        if status_type(entry) != "idle":
            with self.lock:
                self._note_busy(event=False)
                reabort = (running and self.state == STATE_RUNNING and self.turn_id == turn_id and not self.closing
                           and self._interrupt_requested and self._idle_after_abort)
                if reabort:
                    self._idle_after_abort = False
            if reabort:
                self._repeat_abort(turn_id)
            return
        with self.lock:
            if self._busy_seq == seq:
                self._native_busy = False
                if self._interrupt_requested and self.state == STATE_RUNNING and self.turn_id == turn_id:
                    self._idle_after_abort = True
        messages = self.client.messages(self.thread_id)
        children = self._fetch_children(messages)
        with self._projector_lock:
            snapshot = self.projector.sync(messages)
            self._sync_children(children)
        for part_id in snapshot.data.get("compactions") or ():
            self._compaction_signal("compaction_part", key=part_id)
        if not running:
            return
        with self.lock:
            if self.state != STATE_RUNNING or self.turn_id != turn_id or self.closing:
                return
            if self._busy_seq != seq:       # the session turned busy after this pass read idle: stale snapshot
                return
            if self._submitting:            # a prompt/steer is on its way: its message may not exist yet
                self._guard.reset()
                return
            interrupt = self._interrupt_requested
            last_user, _pending = self.projector.submissions()
            verdict = turn_verdict(messages, entry, last_user_message_id=last_user, interrupt_requested=interrupt)
            if interrupt and verdict.status == "interrupted" and verdict.message_id is None:
                # Idle with no answer to the turn's prompt yet: the abort may have landed after the 204 but before
                # OpenCode started the loop (an abort then is a no-op and the loop starts afterwards). The interrupt
                # is confirmed only by a second idle observation incomplete_gap_s later with no busy in between; a
                # loop that does start is aborted again (``busy_after_interrupt``).
                verdict = TurnVerdict("running", reason="interrupt_unconfirmed", idle=True)
            verdict = self._guard.observe(verdict, time.monotonic(), interrupt_requested=interrupt)
            if not verdict.terminal:
                if interrupt and verdict.idle:
                    self._recheck_at = time.monotonic() + self._guard.min_gap_s
                return
            finished = self._finish_turn(verdict.status, verdict.error)
        if finished:
            # Written after the terminal, so a Store failure here can never delay it (it is recorded instead).
            try:
                self.transcript.event("agent.turn_verdict", dict(
                    status=verdict.status, reason=verdict.reason, message_id=verdict.message_id,
                    signals=list(reasons)), turn_id=turn_id)
            except Exception as error:  # noqa: BLE001
                self._handler_error("turn_verdict", error)

    def _repeat_abort(self, turn_id):
        """The main session turned busy after an interrupt of ``turn_id`` was requested (the loop started after the
        first abort, or kept going): abort it again. The relay tombstones from ``interrupt`` stay in force."""
        try:
            self.store.raw_agent("sent", dict(command="interrupt_repeat", thread_id=self.thread_id, turn_id=turn_id),
                                 channel="control")
            self.client.abort(self.thread_id, timeout_s=self.interrupt_timeout_s)
            self.transcript.event("agent.interrupt_repeated", {}, turn_id=turn_id)
        except Exception as error:  # noqa: BLE001 - the next busy observation tries again
            self._handler_error("interrupt_repeat", error)

    # ---- turns -----------------------------------------------------------------------------------------------
    def _connected(self):
        process = self.process
        return self.client is not None and process is not None and process.poll() is None

    def run_turn(self, prompt):
        """Submit ``prompt`` as a new turn; returns the host-minted ``turn_id`` once ``prompt_async`` answered 204
        (admission), not at completion. 4xx: definite rejection (``AgentRejected``, previous state restored); any
        other failure: unknown admission (``disconnected`` + ``AgentTransportUnknown``), never retried."""
        self._check_turn_admissible()
        # An interrupt tombstoned the relay threads (non-Codex profiles); this turn's model calls must be admitted.
        # They are lifted before the prompt is sent (OpenCode answers 204 from a forked loop that may call the model
        # before the answer reaches us) and put back if nothing was admitted (any refusal before or at the submit).
        lifted = self._reopen_relay_threads()
        unknown = False
        try:
            with self.lock:
                previous = self._begin_turn()
                turn_id = self.turn_id
                self._guard.reset()
                self._recheck_at = None
                self._idle_after_abort = False
                if not self._connected():
                    self._reject_turn(previous, "not_connected", "opencode_not_connected")
                try:
                    text = self.presentation.present_text(prompt)
                    self.store.raw_agent("sent", dict(command="submit_prompt", thread_id=self.thread_id,
                                                      turn_id=turn_id, input=[dict(type="text", text=text)]),
                                         channel="control")
                    item = self.transcript.user_message(text, turn_id=turn_id)
                except Exception as error:  # noqa: BLE001 - nothing was sent: a definite rejection
                    self._reject_turn(previous, "submit_not_recorded",
                                      f"opencode_submit_not_recorded:{type(error).__name__}")
                self._submitting += 1
                variant = self._variant()
            try:
                self.client.prompt_async(self.thread_id, text, variant=variant)
            except Exception as error:  # noqa: BLE001
                self._submission_failed(item, reconcile=False)
                if is_definite_rejection(error):
                    self._reject_turn(previous, error.status, "opencode_prompt_rejected")
                unknown = True                  # _disconnected blocks the relay again
                self._disconnected(error)
                code = f"status_{error.status}" if isinstance(error, OpenCodeHttpError) else type(error).__name__
                raise AgentTransportUnknown(f"opencode_prompt_unknown:{code}") from error
        except BaseException:
            if not unknown:
                self._restore_tombstones(lifted)
            raise
        with self.lock:
            self._submitting -= 1
        self._request_reconcile("submitted")
        return turn_id

    def _submission_failed(self, item, *, reconcile=True):
        """A prompt/steer request failed: the host row never becomes the verdict anchor (withdrawn) and the submission
        counter drops; for a steer (``reconcile``) the still running turn is re-evaluated without it."""
        try:
            with self._projector_lock:
                self.projector.withdraw_submission(item)
        except Exception as error:  # noqa: BLE001
            self._handler_error("withdraw_submission", error)
        with self.lock:
            self._submitting -= 1
        if reconcile:
            self._request_reconcile("submission_withdrawn")

    def steer(self, text):
        """Add operator ``text`` to the ACTIVE turn (``prompt_async`` while busy; it joins at the next step
        boundary). Returns ``turn_id`` once OpenCode answered 204; ``ValueError("no_active_turn_to_steer")`` unless
        running, ``ValueError("turn_interrupt_requested")`` once ``interrupt()`` was called for the turn. A 4xx is a
        definite rejection (``AgentRejected``; the turn continues); any other failure is unknown
        delivery -> ``disconnected`` + ``AgentTransportUnknown``."""
        with self.lock:
            if self.state != STATE_RUNNING or not self.turn_id or self.closing or not self._connected():
                raise ValueError("no_active_turn_to_steer")
            if self._interrupt_requested:
                # A steer now would start a new native loop after the abort; the interrupted turn is ending.
                raise ValueError("turn_interrupt_requested: the turn is being interrupted; send the text as the next "
                                 "prompt")
            turn_id = self.turn_id
            text = self.presentation.present_text(text)
            self.store.raw_agent("sent", dict(command="steer", thread_id=self.thread_id, turn_id=turn_id,
                                              input=[dict(type="text", text=text)]), channel="control")
            item = self.transcript.steer(text, turn_id=turn_id)
            self._submitting += 1
            variant = self._variant()
        try:
            self.client.prompt_async(self.thread_id, text, variant=variant)
        except Exception as error:  # noqa: BLE001
            self._submission_failed(item)
            if is_definite_rejection(error):
                self.transcript.event("agent.steer_rejected", dict(code=error.status), item_id=item)
                raise AgentRejected(error.status, "opencode_steer_rejected") from error
            self._disconnected(error)
            code = f"status_{error.status}" if isinstance(error, OpenCodeHttpError) else type(error).__name__
            raise AgentTransportUnknown(f"opencode_steer_unknown:{code}") from error
        with self.lock:
            self._submitting -= 1
        self._request_reconcile("steered")
        return turn_id

    def interrupt(self):
        """Interrupt the active turn, or stop a native loop that runs without one; returns before it ends (the
        reconcile worker finishes a host turn as ``interrupted``).

        Acts when a host turn runs or, with no host turn, when a live ``GET /session/status`` shows the main session
        busy (an unreadable status counts as busy), also after an unknown admission
        (``disconnected``) while the server lives. Order (relay block first, native abort second, as Codex): mark the
        interrupt (the verdict takes it first); ``cancel_pending`` on the relay plus a reopenable tombstone on the
        main session and every known subagent session, even when none of them has a model call in flight (probe
        R4-0: aborting the main session while a ``task`` runs still lets the main agent make one more model call; the
        tombstone denies it instead of forwarding and charging it); ``POST /abort`` on the main session, then on every
        subagent session of the turn (design R4.3; OpenCode also cascades the cancel itself). The next ``run_turn``
        reopens those threads. Nothing is sent to an idle session with no host turn (in-flight relay calls are still
        cancelled). A failed main abort is unknown -> ``disconnected`` + ``AgentTransportUnknown``; a failed subagent
        abort is recorded only."""
        with self.lock:
            running = self.state == STATE_RUNNING
            if running:
                self._interrupt_requested = True
            turn_id = self.turn_id
            children = [child for child, entry in self._children.items() if entry["turn_id"] == turn_id]
            closing = self.closing
            reachable = not closing and self.thread_id is not None and self._connected()
        # With no host turn the live status decides (the cached one may lag the events either way).
        busy = False if running or not reachable else self._live_native_busy(self.interrupt_timeout_s)
        act = running or busy
        self._block_relay("turn_interrupted", block=act)
        if not act or not reachable:
            return
        with self.lock:
            self._idle_after_abort = False
        self.store.raw_agent("sent", dict(command="interrupt", thread_id=self.thread_id, turn_id=turn_id,
                                          host_turn=running, subagent_sessions=children), channel="control")
        try:
            accepted = self.client.abort(self.thread_id, timeout_s=self.interrupt_timeout_s)
        except Exception as error:  # noqa: BLE001
            self.transcript.event("agent.interrupt_failed", dict(error=type(error).__name__, host_turn=running))
            self._disconnected(error)
            code = f"status_{error.status}" if isinstance(error, OpenCodeHttpError) else type(error).__name__
            raise AgentTransportUnknown(f"opencode_abort_unknown:{code}") from error
        if not accepted:
            self.transcript.event("agent.interrupt_not_accepted", {})
        for child in children:
            try:
                self.client.abort(child, timeout_s=self.interrupt_timeout_s)
            except Exception as error:  # noqa: BLE001
                self.transcript.event("agent.subagent_abort_failed", dict(session_id=child,
                                                                          error=type(error).__name__))
        self._request_reconcile("abort")

    def _live_native_busy(self, timeout_s):
        """``GET /session/status`` for the main session now (``timeout_s`` s): True when busy/retry or when the status
        cannot be read (unknown counts as busy); updates the cached native status."""
        try:
            status = self.client._request("GET", "/session/status", timeout=timeout_s)
            busy = status_type(_mapping(status).get(self.thread_id)) != "idle"
        except Exception:  # noqa: BLE001 - unknown: treat as busy (stop it rather than let it run)
            return True
        with self.lock:
            if busy:
                self._note_busy(event=False)
            else:
                self._native_busy = False
        return busy

    def _block_relay(self, reason, *, block=True):
        """``model_proxy.cancel_pending(reason)``; with ``block`` also a reopenable tombstone on the main session and
        every known subagent session (an unregistered child id falls back to the main session for attribution, so
        the main tombstone alone already denies a child created later)."""
        proxy = self.model_proxy
        if proxy is None:
            return
        threads = ()
        if block and self.thread_id is not None:
            with self.lock:
                threads = (self.thread_id, *self._children)
        proxy.cancel_pending(reason, threads=threads)

    def _reopen_relay_threads(self):
        """Before a new turn: lift the interrupt tombstones of the main session and of every known subagent session
        (a subagent resumed with ``task_id`` keeps its session id). Returns ``{thread: tombstone reason}`` of the
        threads lifted (for ``_restore_tombstones``)."""
        proxy = self.model_proxy
        if proxy is None or self.thread_id is None:
            return {}
        with self.lock:
            threads = (self.thread_id, *self._children)
        lifted = {}
        for thread in threads:
            reason = proxy.cancelled_threads.get(thread)
            if proxy.reopen_thread(thread):
                lifted[thread] = reason or "turn_interrupted"
        return lifted

    def _reopen_relay_thread(self):
        """``McpSessionMixin`` hook: ``_reopen_relay_threads``; True when one tombstone was lifted."""
        return bool(self._reopen_relay_threads())

    def _restore_tombstones(self, lifted):
        """Put back the tombstones ``_reopen_relay_threads`` lifted for a turn that was not admitted (``{thread:
        reason}``), without touching other threads' calls. Never raises."""
        proxy = self.model_proxy
        if proxy is None or not lifted:
            return
        by_reason = {}
        for thread, reason in lifted.items():
            by_reason.setdefault(reason, []).append(thread)
        for reason, threads in by_reason.items():
            try:
                proxy.block_threads(threads, reason)
            except Exception as error:  # noqa: BLE001
                self._handler_error("restore_tombstones", error)

    def _disconnected(self, error):
        """``McpSessionMixin._disconnected`` after a relay block of the main session and every known subagent
        session: an admitted turn (unknown admission) or a loop still running on the server cannot reach the model
        any more. ``interrupt()`` and ``close()`` still abort it while the server lives."""
        with self.lock:
            if self.closing or self.state in (STATE_DISCONNECTED, STATE_CLOSED):
                return
        try:
            self._block_relay("agent_disconnected")
        except Exception as failure:  # noqa: BLE001 - the disconnect below must still happen
            self._handler_error("relay_block", failure)
        McpSessionMixin._disconnected(self, error)

    def set_model(self, model):
        """OpenCode fixes the model at session start (the model id selects its native base prompt and edit tool set
        and is the only model the generated provider declares): the same model is accepted, another one is refused
        with ``AgentCapabilityError("set_model_unsupported_by_backend:opencode")``."""
        if model == self.model:
            return
        raise AgentCapabilityError("set_model_unsupported_by_backend:opencode: start a new run with model.name")

    def set_effort(self, effort):
        """Select the effort for the NEXT prompt (sent as the model ``variant``). Validated against the model's
        declared variants when it declares any (``effort_choices``), else against every ``model.effort`` choice."""
        AgentSession.set_effort(self, effort)

    # ---- protocol members ------------------------------------------------------------------------------------
    def owned_processes(self):
        """``[("opencode", the server Popen)]`` from spawn until close confirmed it gone, else ``[]``."""
        process = self.process
        return [("opencode", process)] if process is not None and not self._process_gone else []

    def provenance(self):
        """Manifest ``agent_backend`` block (design §7.3) from ``config.provenance_block``; observer data only."""
        from ..factory import evidence_client
        features = self.config.features
        common = dict(upstream_provider=self.provider, model_configured=self.model, effort=self.effort,
                      isolation=self.isolation, counting=self.counting, context_delivery=self.context_delivery,
                      capabilities=self.capabilities.to_dict(), subagents=self.oc_config.subagents,
                      evidence_client=(evidence_client(self.config, self.backend_kind)
                                       if features.visual_review_model or features.self_briefing else None))
        if self.launch is None:
            return dict(AgentSession.provenance(self), wire_api=self.wire_api, **common)
        block = opencode_config.provenance_block(
            self.launch, route_list_sha256=self._route_digest, wire_api=self.wire_api, model=self.model,
            effort=self._variant(), instructions_sha256=self._instructions_sha256, tool_name_prefix=TOOL_NAME_PREFIX,
            subagents=self.oc_config.subagents, enabled_tools=self._native_tools)
        block.update(common)
        variant = self._variant()
        block["native_effort"] = dict(
            variant=variant, applied=variant is not None, channel="prompt_variant",
            provider_options=self._declared_variants().get(variant) if variant else None,
            declared_variants=list(self._variants), wire_field=opencode_config.EFFORT_WIRE_FIELDS.get(self.wire_api))
        block["output_limit_tokens"] = self.oc_config.output_limit_tokens
        block["context_limit_tokens"] = self.oc_config.context_limit_tokens
        block["prompt_channel"].update(tool_name_prefix=TOOL_NAME_PREFIX, naming_note=naming_note())
        if self.bridge is not None:
            block["tool_transport"]["read_only_tools"] = self.bridge.read_only_tools
        return block

    # ---- close -----------------------------------------------------------------------------------------------
    def _close_sandbox(self, report):
        """Remove the sandbox IPC (inbound socket, relay forwarders) after the process tree is gone. Closing the
        owner socket also makes a surviving in-namespace supervisor end its tree (defence in depth)."""
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is None:
            return
        try:
            sandbox.close()
            if self.ipc is not None:
                self.ipc.rmdir()
        except OSError as error:
            if not isinstance(error, FileNotFoundError):
                report["sandbox_close_error"] = type(error).__name__
        except Exception as error:  # noqa: BLE001
            report["sandbox_close_error"] = type(error).__name__

    def close(self):
        """``McpSessionMixin.close`` after one live read of the main session's native status (so a loop the events
        did not report still makes the close unclean; an unreadable status of a live server counts as busy)."""
        with self.lock:
            check = (not self.closing and self.state not in (STATE_CLOSED, STATE_NOT_STARTED)
                     and self.thread_id is not None and self.client is not None)
        if check and self._connected():
            self._live_native_busy(self.close_abort_timeout_s)
        return McpSessionMixin.close(self)

    def _stop_backend(self):
        """Stop the server and every process it started, then verify (design §3.7).

        While the server lives: relay block (main + known subagent sessions) and ``POST /abort`` on the main session
        whatever the host state (kills running tool shells; a no-op on an idle session). Native tool rows left open
        by a turn that was running (host or native) become failed + interrupted. Then the event stream ends and the
        process tree is stopped: a fresh process table right before the kill (members re-validated, so a recycled pid
        or group is not signalled) plus every descendant and process group seen in the periodic refreshes, plus every
        process carrying this session's marker (``isolation.AGENT_MARKER_ENV``, inherited by every tool shell and its
        detached children; on macOS also orphans in the actor dir whose environment the OS withholds), all through the
        shared ``isolation.stop_agent_tree`` -> killed -> re-scanned. Teardown is confirmed only when nothing targeted
        survived, the final re-scan found no marked process and every scan succeeded; this also holds after a server
        crash (``agent_exited_before_teardown`` is recorded as a fact). The daemon threads are joined last."""
        report = {}
        self._halt_threads()
        process, client = self.process, self.client
        alive = process is not None and process.poll() is None
        if alive:
            self._refresh_tree()
        elif process is not None and self.thread_id is not None:
            report["agent_exited_before_teardown"] = True
        at_close = (self.close_report or {}).get("state_at_close")
        native_active = self._native_busy            # refreshed by close() right before the mixin's close
        if alive and client is not None and self.thread_id is not None:
            try:   # relay block before the abort (see interrupt): a model call made after it is denied, not charged
                self._block_relay("session_closed")
            except Exception as error:  # noqa: BLE001
                report["relay_block"] = "failed:" + type(error).__name__
            try:
                client.abort(self.thread_id, timeout_s=self.close_abort_timeout_s)
                report["abort"] = "completed"
            except Exception as error:  # noqa: BLE001 - the process-tree teardown follows anyway
                report["abort"] = "failed:" + type(error).__name__
        if at_close in (STATE_RUNNING, STATE_DISCONNECTED) or native_active:
            # The turn ends with the session: native tool rows it left open become failed + interrupted (registry
            # rows keep what the MCP transport wrote).
            try:
                with self._projector_lock:
                    report["abandoned_items"] = sum(len(projector.abandon_open_items())
                                                    for projector in self._projectors())
            except Exception as error:  # noqa: BLE001
                report["abandon_error"] = type(error).__name__
        if client is not None:
            try:
                client.close_streams()
            except Exception:  # noqa: BLE001
                pass
        confirmed = True
        if process is not None or self._instance_id is not None:
            try:
                result = stop_agent_tree(process.pid if process is not None else None,
                                         process is not None and process.poll() is None,
                                         list(self._known_tree.values()), self._instance_id,
                                         cwd_roots=[self.actor_dir])
                report["agent_processes"] = result.to_dict()
                confirmed = result.confirmed
            except Exception as error:  # noqa: BLE001
                report["teardown_error"] = type(error).__name__
                confirmed = False
            try:
                if process is not None:
                    process.poll()   # the teardown may have reaped it: let the Popen record its end
            except Exception:  # noqa: BLE001
                pass
        if not confirmed:
            report["agent_process_exit_unconfirmed"] = True
        elif process is not None:
            self._process_gone = True
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(self.thread_join_timeout_s)
        alive_threads = [thread.name for thread in self._threads if thread.is_alive()]
        if alive_threads:
            report["threads_alive"] = alive_threads
        self._teardown_confirmed = confirmed
        self._close_sandbox(report)
        if self.close_report is not None:
            self.close_report["backend"] = report
