"""Claude Code agent backend session (design 04 §2.2-§2.3, §3.3, §3.7, §5.2-§5.3, §7.3; plan 07 §2; T0 report 08).

``ClaudeCodeSession`` drives one pinned Claude Code CLI (``claude-agent-sdk==0.2.180``, bundled CLI 2.1.321) through
``ClaudeSDKClient`` for one run. Registry tools reach the CLI only through the host MCP endpoint (``McpToolBridge``);
model requests only through the run's relay (``ModelRequestProxy`` with ``AnthropicMessagesWire(client="claude_code")``,
egress scrub = the wire's ``project_request``); the CLI sees a placeholder credential and an ``env -i`` environment.

Threads
-------
* callers (gateway ``asyncio.to_thread``): the synchronous facade; each SDK call is posted to the private loop with
  ``asyncio.run_coroutine_threadsafe`` and awaited with a timeout (``_call``). The session lock is never held across
  such a wait (the loop thread takes it to deliver messages).
* ``claude-loop`` (daemon): one private asyncio loop that runs every SDK coroutine, the single reader task over
  ``receive_messages()`` (never abandoned mid-iteration) and the ``PreCompact`` hook. Handlers there only take the
  session lock and write Store rows.
* MCP handler threads + ``agent-tools`` workers (``McpToolBridge``): tool calls; they never touch the loop.
* relay threads: ``_model_event`` / ``_relay_discontinuity``.

Turn contract (T0 report 08 §7 and design R4.2, normative)
----------------------------------------------------------
* Admission: ``run_turn`` returns once ``client.query(text)`` returned (the stdin write is done). There is no definite
  rejection after the write: a write error or timeout is unknown admission -> ``disconnected``, never retried. The
  definite rejections happen before anything is sent: no client or reader, the turn could not be recorded, or an
  ``interrupt()`` arrived while the submission was still being prepared (``AgentRejected("interrupted", ...)``).
* Inputs: every prompt and steer written into a turn waits for its ``--replay-user-messages`` echo. A turn ends at
  the first ``session_state_changed: idle`` at which none of its inputs is still unconsumed; an idle before that
  (a native busy period that ended before the CLI read the input: a steer written as the model finished, a prompt
  written while a native wake-up was finishing) is bridged and recorded as ``agent.turn_end_deferred`` — the CLI
  runs the input as a further native turn inside the same Auriga turn (design §2.2). An interrupted turn ends at its
  idle regardless. An input that outlives its turn (interrupt, guard) belongs to whichever turn consumes it later
  (``agent.input_delivered_late``).
* Status comes from the LAST ``ResultMessage`` since the turn began: host interrupt requested or ``terminal_reason``
  in ``aborted_tools``/``aborted_streaming`` -> ``interrupted``; ``is_error`` false -> ``completed``; else ``failed``
  with ``claude_code_turn_failed:<terminal_reason or subtype>[:<api_error_status>]`` (no provider text). No result
  before the end -> ``failed`` / ``claude_code_turn_failed:idle_without_result``.
* Guards, not mechanisms (each records ``agent.turn_end_fallback {signal}``; T0 never needed them): a result
  followed by ``result_quiet_guard_s`` (5 s) of silence while no delegated native task (subagent, workflow) runs
  (``result_quiet_fallback``); a bridged idle followed by the same silence (``input_echo_missing``; the turn ends
  and its unconsumed inputs belong to whichever native turn consumes them later). With a background subagent the
  CLI stays ``running`` past the main Result and reports an interrupt with no new Result, so the interrupt flag, not
  the last Result, decides ``interrupted`` (probe R4-0); subagent rows that arrive after the idle stay with that turn.
* Adoption (``capabilities.native_wakeup``): a native ``running`` while the host turn is terminal or idle opens an
  adopted Auriga turn under the session lock. Sources seen with the pinned CLI (T0, probe R4-0): a background shell,
  monitor or background subagent finishing; ``ScheduleWakeup``/``CronCreate`` timers; a durable cron's "missed task"
  turn right after a resume connects (before the host's first ``run_turn``). Wake-up turns carry no origin marker the
  SDK passes on and no prompt echo, so adoption keys on the state event alone. The adopted turn
  (``agent.turn_started {origin: "native_wakeup"}``, new turn id, ``turn_done`` cleared) ends at its idle by the
  same rules; its registry tool calls carry that turn id. ``run_turn`` during an adopted turn that has no host prompt
  yet joins it (``control: submit_prompt`` with the adopted turn id; returns it); a host turn still raises
  ``turn_already_active``.
* Registry tool calls are refused as stale unless the host state is ``running``; ``interrupt()`` sends the control
  whenever a host turn runs or the CLI reports itself busy; a native turn in progress at ``close()`` makes the
  shutdown unclean.
"""
import asyncio
import concurrent.futures
import dataclasses
import os
import signal
import sys
import threading
import time
import uuid
from pathlib import Path

from ....core.contracts import canonical, digest, new_id
from ....core.storage.images import store_image_settings
from urllib.parse import urlsplit

from ...session.actor_source import snapshot_actor_source
from ...session.agent_sandbox import ACTOR_ISOLATION
from ...session.model_proxy import upstream_auth_headers_copy
from ...session.wire import AnthropicMessagesWire
from ..base import (AgentCapabilities, AgentRejected, AgentSession, AgentTransportUnknown, STATE_IDLE,
                    STATE_NOT_STARTED, STATE_REPLAY, STATE_RUNNING, UNCONNECTED_STATES, read_agent_identity)
from ..isolation import build_snapshot, private_dirs, process_table, stop_agent_tree
from ..mcp_session import GENERATION_COUNTING, McpSessionMixin, McpToolBridge
from ..transcript import TranscriptWriter
from . import options as claude_options
from .projector import ABORT_REASONS, ClaudeProjector, classify_result

__all__ = ["ClaudeCodeSession", "NATIVE_SESSION_DIR"]

# Run-relative directory holding the CLI's own session JSONL (needed for resume; hidden from the actor).
NATIVE_SESSION_DIR = "private/claude-config"
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "stopped", "killed"})   # SDK TERMINAL_TASK_STATUSES
# Native task types that keep the CLI ``running`` after a ResultMessage until they settle (SDK 0.2.180
# ``_internal.query.DEFERRING_TASK_TYPES``: subagents and workflows). Background shells (``local_bash``) and monitors
# do not: the CLI goes idle and later starts a wake-up turn of its own (adopted, design R4.2).
_DEFERRING_TASK_TYPES = frozenset({"local_agent", "local_workflow"})
# ``session_state_changed`` states in which the CLI is working on a turn.
_NATIVE_BUSY_STATES = frozenset({"running", "requires_action"})
# Error classes (SDK 0.2.180 ``_errors``) that mean the CLI process is gone: a control request pending when it died
# fails with the reader's ProcessError/ResultError, a later write with CLIConnectionError.
_CLI_GONE_ERRORS = frozenset({"ProcessError", "ResultError", "CLIConnectionError"})


def _class_names(value):
    return {cls.__name__ for cls in type(value).__mro__}


def _record(value):
    """JSON-safe copy of an SDK message: dataclasses become dicts with ``_type`` (the spike capture format read by
    ``tests/support/fake_claude_sdk.message_from_capture``); unknown objects become their ``repr``."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {"_type": type(value).__name__}
        for item in dataclasses.fields(value):
            out[item.name] = _record(getattr(value, item.name))
        return out
    if isinstance(value, dict):
        return {str(key): _record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else repr(value)
    return repr(value)


class _AnyioProcess:
    """Popen-like view (``pid``, ``returncode``, ``poll``, ``terminate``, ``kill``) of the anyio ``Process`` behind
    ``ClaudeSDKClient._transport._process`` (the one private SDK attribute this backend reads, pinned to 0.2.180).

    One instance per session, so callers that key on the object (``Workbench.abort_startup``) see the same process.
    Signals go by pid and only while ``returncode`` is None and the session has not confirmed the process gone
    (``gone``; set after a verified teardown, when the private loop that would record the exit code is stopped), so
    a recycled pid is never signalled.
    """

    def __init__(self, process):
        self._process = process
        self.pid = process.pid
        self.gone = False

    @property
    def returncode(self):
        return self._process.returncode

    def poll(self):
        return self._process.returncode

    def _signal(self, number):
        if self._process.returncode is None and not self.gone:
            os.kill(self.pid, number)

    def terminate(self):
        self._signal(signal.SIGTERM)

    def kill(self):
        self._signal(signal.SIGKILL)


class ClaudeCodeSession(McpSessionMixin, AgentSession):
    """Claude Code backend (``agent.backend = "claude_code"``). Public contract: ``AgentSession``.

    Constructor (called by ``factory.create_session``): ``run_dir`` (run directory), ``store`` (run Store),
    ``tool_handler`` (``Harness.handle_tool``), ``price_card`` (frozen card or None), ``config`` (``Config``; default
    ``Config.load()``), ``task_planning`` (unused: the decision reminder reaches this backend through
    ``on_context_reset``). Test seams: ``client_factory`` (``options -> ClaudeSDKClient``-like; default the SDK class)
    and ``runtime`` (a pre-resolved ``options.ClaudeRuntime``; default: resolved and version-checked in ``start``).
    Constructing a session never imports ``claude_agent_sdk`` (a replay viewer of a Claude Code archive needs no
    runtime).
    """
    backend_kind = claude_options.BACKEND_KIND
    counting = GENERATION_COUNTING
    capabilities = AgentCapabilities(steer=True, live_effort=claude_options.LIVE_EFFORT, native_wakeup=True,
                                     sandbox=True, compaction_signal="native+relay")
    effort_choices = claude_options.EFFORT_CHOICES
    context_delivery = "full"
    wire_api = "anthropic_messages"
    # Facade timeouts in seconds (plan 07 §2.1). connect: SDK initialize waits 60 s. interrupt: below the gateway's
    # 45 s bound on interrupt(), so a wedged CLI is declared disconnected before the gateway gives up waiting.
    connect_timeout_s = 120.0
    submit_timeout_s = 15.0
    interrupt_timeout_s = 40.0
    set_model_timeout_s = 70.0
    disconnect_timeout_s = 30.0
    loop_join_timeout_s = 5.0
    # Turn-end guard (s): silence after a ResultMessage with no idle, or after a bridged idle (see module doc).
    result_quiet_guard_s = 5.0
    # Interval (s) at which the CLI process tree is re-snapshotted while it runs, so the children of a CLI that dies
    # can still be found and stopped at close (their parent link is gone after the crash).
    tree_refresh_s = 5.0

    def __init__(self, run_dir, store, tool_handler, *, price_card=None, config=None, task_planning=None,
                 client_factory=None, runtime=None):
        from ....core.config import Config
        from ....core.presentation.actor_presentation import ActorPresentation
        self.config = config if config is not None else Config.load()
        self.cc_config = self.config.agent.claude_code
        self.root = Path(run_dir).resolve()
        self.store, self.tool_handler, self.task_planning = store, tool_handler, task_planning
        self.presentation = ActorPresentation(store)
        self.model = self.config.model.name
        self.active_turn_model = self.model
        self.effort = self.config.model.effort
        self.provider = self.cc_config.upstream.provider
        self.price_card = price_card
        # features.sandbox_enabled runs the CLI in the generic bubblewrap AgentSandbox (design R4.4; verified on
        # Linux by conformance item 28); recorded in runtime/agent.isolation, and resume across modes is refused.
        self.sandbox_enabled = bool(self.config.features.sandbox_enabled)
        self.isolation = ACTOR_ISOLATION if self.sandbox_enabled else "disabled"
        self._sandbox_ipc = None
        # Design R4.3: a subagent that calls an Auriga tool receives images in a context the Harness does not track,
        # so with subagents enabled every actor tool result is self-contained ("full"); "delta" needs subagents off.
        self.context_delivery = "full" if self.cc_config.subagents else self.cc_config.context_delivery
        self.actor_dir, self.private = self.root / "actor", self.root / "private"
        self.wire = AnthropicMessagesWire(client="claude_code")
        self.lock = threading.RLock()
        self._close_lock = threading.Lock()
        self.state, self.thread_id, self.turn_id = STATE_NOT_STARTED, None, None
        self.turn_done, self.turn_result = threading.Event(), None
        self.last_activity_at = time.monotonic()
        self.model_limit_reached = threading.Event()
        self.model_proxy = self.bridge = self.transcript = self.projector = self.launch = None
        self.closing, self.close_report = False, None
        self.runtime = runtime
        self._client_factory = client_factory
        self._client = self._reader = self._loop = self._loop_thread = self._process = None
        self._started = False
        self._identity = None
        self._teardown_confirmed = True
        self._interrupt_requested = False
        self.turn_origin = None
        self._host_prompted = False      # the running turn carries a host prompt (host-started or joined)
        self._native_state = None
        self._last_result = None
        self._end_deferred = False       # an idle arrived while an input of the running turn was unconsumed
        self._inputs = {}                # transcript item id -> (turn id, "prompt" | "steer", text), awaiting echo
        self._orphans = {}               # item id -> (turn id, kind, text): inputs that outlived their turn
        self._write_future = None        # the latest scheduled prompt/steer stdin write (interrupt orders after it)
        self._steer_outcomes = {}        # steer item id -> turn id that consumed it, or None when it was dropped
        self._steer_unsettled = set()    # steer item ids whose steer() call has not determined its turn yet
        self._submission = None          # run_turn before its write: {"cancelled": bool}
        self._deferring_tasks = set()
        self._quiet_handle = None
        self._quiet_token = 0
        self._process_lock = threading.Lock()
        self._connect_failed = False     # connect raised after a client existed (a CLI may have been spawned)
        self._instance_id = None         # runtime/agent.instance_id; also the process marker of the CLI env
        self._known_tree = {}            # pid -> ProcessInfo of every CLI-tree member seen while the CLI ran
        self._tree_stop = threading.Event()
        self._tree_thread = None
        self._native_model = None
        self._init_tools = self._init_model = None
        self._tool_names = frozenset()
        self._unknown_parts = set()
        self._log = None
        self._log_lock = threading.Lock()

    # ---- start ---------------------------------------------------------------------------------------------
    def start(self, tools, instructions, resume_thread_id=None):
        """Start (or resume ``resume_thread_id``) the CLI session; return the session id (``thread_id``).

        Fresh start: a host-chosen uuid4 passed as ``session_id``. Resume: the recorded id passed as ``resume``,
        refused unless ``runtime/agent`` matches this session (backend, thread, isolation, tools digest, model,
        provider, counting, wire, SDK/CLI versions) and the CLI's session JSONL exists under
        ``private/claude-config``; after an unclean exit the model context epoch is invalidated and the decision
        reminder is scheduled (``on_context_reset``). Any failure closes the session and re-raises (not
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
        dirs = private_dirs(self.root, ["private", "private/claude-config", "private/tmp"])
        self.actor_dir.mkdir(exist_ok=True)
        self._log = open(self.private / "agent.log", "a", encoding="utf-8")
        if self.runtime is None:
            self.runtime = claude_options.resolve_runtime(self.cc_config.cli_path, dict(
                PATH=os.defpath, LANG="C.UTF-8", HOME=os.fspath(self.actor_dir), TMPDIR=os.fspath(dirs["private/tmp"]),
                CLAUDE_CONFIG_DIR=os.fspath(dirs["private/claude-config"])))
        thread_id = resume_thread_id if resume else str(uuid.uuid4())
        if resume:
            self._validate_resume(previous, thread_id, tools)
            self._stop_previous_instance(previous)
        # Minted before the launch: the CLI environment carries it as the process marker of this session.
        self._instance_id = new_id("agent")
        self.thread_id = thread_id
        self.transcript = TranscriptWriter(self.store, backend=self.backend_kind, thread_id=thread_id)
        self.projector = ClaudeProjector(self.transcript)
        self._tool_names = frozenset(tool["name"] for tool in tools)
        upstream_url, upstream_headers = self._upstream()
        self.store.set_state("runtime", "price_card", self.price_card)
        self._start_relay(upstream_url, upstream_auth_headers=upstream_headers,
                          request_projection=self._project_request, default_thread_id=thread_id,
                          image_settings=store_image_settings(self.store, self.config.images.to_dict()))
        self.model_proxy.register_thread(thread_id, "actor", thread_id)
        self.bridge = McpToolBridge(self, tools, parallel_hints=self.tool_parallel_hints,
                                    server_name=claude_options.MCP_SERVER_NAME).start()
        self.launch = claude_options.build_launch(
            self.root, model=self.model, effort=self.effort, thread_id=thread_id,
            instructions=self.presentation.present_text(instructions), relay_base_url=self.model_proxy.base_url,
            mcp_url=self.bridge.url, mcp_headers=self.bridge.headers, resume=resume, config=self.cc_config,
            native_memories=self.config.model.native_memories,
            stderr=self._stderr_line, hooks=self._hooks(), runtime=self.runtime, sandbox=self._sandbox_plan(),
            instance_id=self._instance_id, settings_env=self._settings_env())
        self._start_loop()
        self._connect_and_watch(self.launch.options)
        self._native_model = self.model
        runtime = self.runtime
        identity = dict(
            backend=self.backend_kind, thread_id=thread_id, instance_id=self._instance_id,
            pid=self._process.pid if self._process is not None else None, model=self.model, provider=self.provider,
            effort=self.effort, isolation=self.isolation, tools_digest=digest(tools),
            instructions_digest=self.launch.instructions_sha256, counting=self.counting, wire_api=self.wire_api,
            sdk_version=runtime.sdk_version, cli_version=runtime.cli_version, cli_executable=runtime.executable,
            cli_sha256=runtime.sha256, native=dict(session_dir=NATIVE_SESSION_DIR), resumed=resume,
            clean_shutdown=False)
        self.store.set_state("runtime", "agent", identity)
        self._identity = identity
        if resume:
            clean = (previous or {}).get("clean_shutdown") is True
            if not clean:
                # The CLI repairs its own dangling tool call ("outcome unknown"); evidence delivered before the
                # crash can no longer be assumed in context. The Harness unknown-action gate stays authoritative.
                self._invalidate_context("agent_resume_after_unclean_exit", marker=new_id("resume"))
            self.transcript.event("agent.history_recovered", dict(clean=clean, delta_history_complete=False))
        with self.lock:
            if self.state == STATE_NOT_STARTED:   # a native wake-up during connect may already be adopted
                self.state = STATE_IDLE
        return thread_id

    def _sandbox_plan(self):
        """``options.SandboxPlan`` of this launch, or None without ``features.sandbox_enabled``.

        The CLI reaches the model relay and the MCP endpoint through in-namespace forwarders on the same loopback
        ports; the published Harness source (``private/actor-source``, refreshed here like Codex does) is mounted
        read-only at ``/harness``. Nothing else of the host is visible (see ``options`` module doc).
        """
        if not self.sandbox_enabled:
            return None
        ports = [urlsplit(self.model_proxy.base_url).port, urlsplit(self.bridge.url).port]
        if None in ports:
            raise ValueError("claude_code_sandbox_requires_loopback_ports")
        snapshot = snapshot_actor_source(self.private / "actor-source")
        self._sandbox_ipc = claude_options.prepare_sandbox_ipc(self.root)
        return claude_options.SandboxPlan(ipc=self._sandbox_ipc, ports=tuple(ports), packages=True,
                                          read_paths=((snapshot, claude_options.HARNESS_SOURCE_MOUNT),))

    def _validate_resume(self, previous, thread_id, tools):
        if not previous:
            raise ValueError("resume_agent_identity_missing: the run has no recorded agent session")
        if previous.get("isolation") != self.isolation:
            raise ValueError("resume_isolation_changed: use a fresh run when changing sandbox mode")
        expected = dict(backend=self.backend_kind, thread_id=thread_id, isolation=self.isolation,
                        tools_digest=digest(tools), model=self.model, provider=self.provider, counting=self.counting,
                        wire_api=self.wire_api, sdk_version=self.runtime.sdk_version,
                        cli_version=self.runtime.cli_version)
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(f"resume_agent_identity_conflict:{key}: resume with the recorded configuration "
                                 "or start a fresh run")
        sessions = self.root / NATIVE_SESSION_DIR / "projects"
        if not any(sessions.glob(f"*/{thread_id}.jsonl")):
            raise ValueError("resume_native_session_missing: the Claude Code session history of this run is absent")

    def _upstream(self):
        """``(base_url, host credential headers)`` of ``agent.claude_code.upstream``; the key is read from the
        named environment variable of the gateway process (never stored, never given to the agent)."""
        upstream = self.cc_config.upstream
        if upstream.source == "claude_settings":
            base_url, headers = claude_options.read_claude_settings_upstream(upstream.settings)
            return base_url, upstream_auth_headers_copy(headers, self.wire.auth_header_names)
        if upstream.source != "explicit" or upstream.wire_api not in (None, self.wire_api) or not upstream.base_url:
            raise ValueError("claude_code_upstream_requires_anthropic_messages_base_url")
        if not upstream.api_key_env or not os.environ.get(upstream.api_key_env):
            raise ValueError(f"claude_code_upstream_key_missing: set the environment variable {upstream.api_key_env}")
        key = os.environ[upstream.api_key_env]
        headers = {"Authorization": "Bearer " + key} if upstream.auth_scheme == "bearer" else {"x-api-key": key}
        return upstream.base_url.rstrip("/"), upstream_auth_headers_copy(headers, self.wire.auth_header_names)

    def _settings_env(self):
        """Provider variables of the ``claude_settings`` file forwarded to the CLI (``read_claude_settings_env``;
        names only in provenance), ``{}`` for every other upstream source."""
        upstream = self.cc_config.upstream
        if upstream.source != "claude_settings":
            return {}
        return claude_options.read_claude_settings_env(upstream.settings)

    def _project_request(self, request):
        """Relay ``request_projection``: the Anthropic egress scrub (design §4.4) for this session's tools."""
        return self.wire.project_request(request, self.presentation, registered_tools=self._tool_names,
                                         tool_name_prefix=claude_options.TOOL_NAME_PREFIX,
                                         on_unknown_part=self._egress_unknown_part)

    def _egress_unknown_part(self, type_name):
        """A content block of a type the egress walker does not know (still scrubbed generically); recorded once
        per type as ``model.egress_unknown_part`` so a CLI change is visible."""
        if type_name in self._unknown_parts:
            return
        self._unknown_parts.add(type_name)
        try:
            self.store.event("model.egress_unknown_part", dict(type=type_name), source="private_model_transport")
        except Exception:  # noqa: BLE001 - observer record only
            pass

    def _hooks(self):
        from claude_agent_sdk import HookMatcher
        return {"PreCompact": [HookMatcher(hooks=[self._pre_compact])]}

    def _stderr_line(self, line):
        with self._log_lock:
            if self._log is not None and not self._log.closed:
                self._log.write(line.rstrip("\n") + "\n")
                self._log.flush()

    # ---- private loop ----------------------------------------------------------------------------------------
    def _start_loop(self):
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run():
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)   # set from inside run_forever: loop.is_running() is already True then
            try:
                loop.run_forever()
            finally:
                try:
                    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.wait(pending, timeout=2))
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:  # noqa: BLE001 - best-effort loop teardown
                    pass
                finally:
                    loop.close()

        self._loop = loop
        self._loop_thread = threading.Thread(target=run, name="claude-loop", daemon=True)
        self._loop_thread.start()
        ready.wait(5)

    def _loop_alive(self):
        loop = self._loop
        return loop is not None and loop.is_running() and not loop.is_closed()

    def _submit(self, coroutine):
        """Schedule ``coroutine`` on the private loop and return its ``concurrent.futures.Future`` (non-blocking).

        Callers that must order two SDK writes (a prompt before an interrupt) schedule them while holding the
        session lock: the loop runs them in scheduling order and the SDK transport serialises writes FIFO.
        Raises ``RuntimeError("claude_code_loop_not_running")`` (the coroutine is closed) when the loop is gone."""
        if not self._loop_alive():
            coroutine.close()
            raise RuntimeError("claude_code_loop_not_running")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    @staticmethod
    def _wait(future, timeout):
        """Wait ``timeout`` seconds for a ``_submit`` future; ``TimeoutError("claude_code_command_timeout")`` (the
        coroutine is cancelled) or the coroutine's own exception. Never call from the loop thread."""
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError("claude_code_command_timeout") from None

    def _call(self, coroutine, timeout):
        """``_wait(_submit(coroutine), timeout)``."""
        return self._wait(self._submit(coroutine), timeout)

    def _connect_and_watch(self, options):
        """Connect within ``connect_timeout_s`` while exposing the CLI process as soon as the SDK spawned it (the SDK
        ``initialize`` control can take up to 60 s; ``Workbench.abort_startup`` must be able to signal the process
        during it)."""
        future = self._submit(self._connect(options))
        deadline = time.monotonic() + self.connect_timeout_s
        while not future.done():
            self._discover_process()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                raise TimeoutError("claude_code_command_timeout")
            concurrent.futures.wait([future], timeout=min(0.02, remaining))
        self._discover_process()
        try:
            return future.result()
        except BaseException:
            self._connect_failed = True
            raise

    async def _connect(self, options):
        if self._client_factory is not None:
            factory = self._client_factory
        else:
            from claude_agent_sdk import ClaudeSDKClient as factory
        client = factory(options)
        self._client = client
        await client.connect()
        self._discover_process()
        self._reader = asyncio.get_running_loop().create_task(self._read(client), name="claude-reader")

    def _discover_process(self):
        """Wrap the SDK transport's CLI process (``ClaudeSDKClient._transport._process``, SDK 0.2.180) once it exists;
        start the tree refresher on first sight. Safe from any thread; returns the ``_AnyioProcess`` or None."""
        with self._process_lock:
            if self._process is None and self._client is not None:
                process = getattr(getattr(self._client, "_transport", None), "_process", None)
                if process is not None and getattr(process, "pid", None):
                    self._process = _AnyioProcess(process)
                    self._refresh_tree()
                    self._tree_thread = threading.Thread(target=self._tree_loop, name="claude-tree", daemon=True)
                    self._tree_thread.start()
            return self._process

    def _refresh_tree(self):
        """Add the CLI's current descendants to ``_known_tree`` and drop members that no longer exist (one process
        table read). No-op once the CLI exited."""
        process = self._process
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
        while not self._tree_stop.wait(self.tree_refresh_s):
            self._refresh_tree()

    async def _read(self, client):
        """The single consumer of ``receive_messages()`` for the whole session (never breaks early)."""
        error = None
        try:
            async for message in client.receive_messages():
                self._on_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as caught:  # noqa: BLE001 - ProcessError / ResultError / CLIConnectionError ...
            error = caught
        self._on_stream_end(error)

    # ---- message handling (loop thread) ----------------------------------------------------------------------
    def _handler_error(self, where, error):
        try:
            self.transcript.event("agent.native_handler_error", dict(handler=where, error=type(error).__name__,
                                                                     detail=str(error)[:200]))
        except Exception:  # noqa: BLE001
            pass

    def _on_message(self, message):
        with self.lock:
            self.last_activity_at = time.monotonic()
            self._cancel_quiet()
            reference = projected = None
            try:
                record = _record(message)
                self.store.raw_agent("received", record, channel="native")
                reference = self.store.put_artifact(canonical(record).encode(), "agent_event", "application/json",
                                                    "observer")
                self.transcript.event("agent.native", dict(raw_ref=reference, kind=type(message).__name__))
            except Exception as error:  # noqa: BLE001 - never kill the reader
                self._handler_error("raw_log", error)
            try:
                projected = self.projector.project(message, raw_ref=reference)
            except Exception as error:  # noqa: BLE001
                self._handler_error("projector", error)
            try:
                if projected is not None and projected.kind in ("prompt_echo", "steer_echo"):
                    self._input_consumed(projected.data.get("item_id"), projected.kind)
                attributed = self._apply(message)
                self._arm_guard(result=attributed)
            except Exception as error:  # noqa: BLE001
                self._handler_error("session", error)

    def _apply(self, message):
        """Session effect of one message; returns True when it was a ResultMessage of the running host turn."""
        names = _class_names(message)
        if "ResultMessage" in names:
            return self._on_result(message)
        if "SystemMessage" not in names:
            return False
        subtype = getattr(message, "subtype", None)
        data = message.data if isinstance(getattr(message, "data", None), dict) else {}
        if subtype == "init":
            self._init_tools, self._init_model = data.get("tools"), data.get("model")
            if data.get("session_id") not in (None, self.thread_id):
                self._disconnected(RuntimeError("claude_code_session_id_mismatch"))
        elif subtype == "session_state_changed":
            self._on_state(data.get("state"))
        elif subtype == "status" and data.get("status") == "compacting":
            self._compaction_signal("compacting")
        elif subtype in ("compact_boundary", "microcompact_boundary"):
            self._compaction_signal(subtype)
        elif subtype in ("task_started", "task_notification", "task_updated"):
            self._track_task(subtype, data)
        return False

    def _track_task(self, subtype, data):
        """Delegated native work (subagents, workflows) that keeps the CLI running after a result (SDK
        ``_track_task_lifecycle``); only it suppresses the result guard."""
        task_id = data.get("task_id")
        if not task_id:
            return
        if subtype == "task_started":
            if data.get("task_type") in _DEFERRING_TASK_TYPES:
                self._deferring_tasks.add(task_id)
        elif subtype == "task_notification":
            self._deferring_tasks.discard(task_id)
        else:
            patch = data.get("patch") if isinstance(data.get("patch"), dict) else {}
            if patch.get("status") in _TERMINAL_TASK_STATUSES:
                self._deferring_tasks.discard(task_id)

    def _input_consumed(self, item_id, kind):
        """The CLI echoed a prompt/steer it consumed (``--replay-user-messages``). The running turn no longer waits
        for it. An input consumed by another turn than the one it was written for (it outlived that turn, e.g. a
        steer racing an interrupt) moves to the consuming turn and ``agent.input_delivered_late`` is recorded."""
        entry = self._inputs.pop(item_id, None) or self._orphans.pop(item_id, None)
        if entry is None:
            return
        registered, input_kind, _ = entry
        consumer = self.turn_id if self.state == STATE_RUNNING else None
        if consumer is not None and consumer != registered:
            row = self.store.transcript_item(item_id, None)
            if row:
                self.store.set_transcript_item(item_id, dict(row, turn_id=consumer))
            self.transcript.event("agent.input_delivered_late", dict(kind=input_kind, written_turn_id=registered),
                                  turn_id=consumer, item_id=item_id)
        if kind == "steer_echo" and item_id in self._steer_unsettled:
            self._steer_outcomes[item_id] = consumer

    def _pending_inputs(self):
        return [item for item, (turn, _, _) in self._inputs.items() if turn == self.turn_id]

    def _expect_input(self, item, text, turn_id, kind):
        """Register a written prompt/steer: it waits for its echo in ``turn_id``. An older input with the same text
        that outlived its turn and was never echoed is withdrawn first, so the coming echo is matched to the new row
        (the projector matches equal texts first-in-first-out). Call with the lock held."""
        for old, (_, _, old_text) in list(self._orphans.items()):
            if old_text.strip() == text.strip():
                del self._orphans[old]
                self.projector.forget_echo(old)
        self.projector.expect_echo(text, item, steer=kind == "steer")
        self._inputs[item] = (turn_id, kind, text)

    def _on_state(self, state):
        self._native_state = state
        if state == "running":
            self._end_deferred = False
            if self.state != STATE_RUNNING:
                self._adopt()
        elif state == "idle" and self.state == STATE_RUNNING:
            pending = self._pending_inputs()
            if pending and not self._interrupted():
                if not self._end_deferred:
                    self._end_deferred = True
                    self.transcript.event("agent.turn_end_deferred", dict(reason="input_not_consumed",
                                                                          inputs=len(pending)))
                return
            self._end_turn()

    def _adopt(self):
        """A native turn started while the host turn is terminal or idle (design R4.2): open an adopted Auriga turn.
        Never adopts while closing, disconnected, closed or in replay (recorded as unsolicited instead); a Store
        failure leaves the native turn unadopted (recorded as a handler error; its tool calls are then refused as
        stale and ``interrupt()`` still reaches it)."""
        if self.transcript is None:
            return
        if self.closing or self.state in UNCONNECTED_STATES or self.state == STATE_REPLAY:
            self.transcript.event("agent.unsolicited_native_turn", dict(host_state=self.state))
            return
        try:
            self._begin_turn(origin="native_wakeup")
        except Exception as error:  # noqa: BLE001
            self._handler_error("adopt", error)
            return
        self._last_result, self._host_prompted, self._end_deferred = None, False, False

    def _on_result(self, message):
        if self.state != STATE_RUNNING:
            status, _ = classify_result(message)
            self.transcript.event("agent.unsolicited_native_result", dict(
                status=status, origin=_record(getattr(message, "origin", None)), host_state=self.state))
            return False
        self._last_result = message
        return True

    def _interrupted(self):
        reason = getattr(self._last_result, "terminal_reason", None)
        return self._interrupt_requested or reason in ABORT_REASONS

    def _end_turn(self, signal=None):
        if self._last_result is None:
            status, error = "failed", "claude_code_turn_failed:idle_without_result"
        else:
            status, error = classify_result(self._last_result, interrupt_requested=self._interrupt_requested)
        self._cancel_quiet()
        pending = self._pending_inputs()
        for item in pending:
            # Not consumed by this turn: whichever native turn consumes it later owns it (_input_consumed).
            self._orphans[item] = self._inputs.pop(item)
        if signal is not None:
            self.transcript.event("agent.turn_end_fallback", dict(signal=signal, quiet_s=self.result_quiet_guard_s,
                                                                  status=status, unconsumed_inputs=len(pending)))
        self._end_deferred = False
        self._finish_turn(status, error)

    def _arm_guard(self, *, result):
        """Re-arm the turn-end guard after a message (any message cancels it first)."""
        if self.state != STATE_RUNNING or self.turn_done.is_set() or self.result_quiet_guard_s is None:
            return
        if result and not self._deferring_tasks:
            signal = "result_quiet_fallback"
        elif self._end_deferred:
            signal = "input_echo_missing"
        else:
            return
        self._quiet_token += 1
        self._quiet_handle = self._loop.call_later(self.result_quiet_guard_s, self._quiet_fired, self._quiet_token,
                                                   signal)

    def _cancel_quiet(self):
        if self._quiet_handle is not None:
            self._quiet_handle.cancel()
            self._quiet_handle = None

    def _quiet_fired(self, token, signal):
        with self.lock:
            if token != self._quiet_token or self._quiet_handle is None:
                return
            self._quiet_handle = None
            if self.state != STATE_RUNNING or self.turn_done.is_set():
                return
            if signal == "result_quiet_fallback" and self._deferring_tasks:
                return
            self._end_turn(signal=signal)

    def _compaction_signal(self, signal):
        """Native compaction evidence (PreCompact hook, ``status: compacting``, compact boundaries): new
        ``runtime/image_context_epoch``, ``agent.compaction`` and the decision reminder (``on_context_reset``)."""
        self._invalidate_context("native_context_compaction", reason="native_context_compaction", signal=signal,
                                 marker=new_id("compaction"))
        self.transcript.event("agent.compaction", dict(signal=signal, source="native"))

    async def _pre_compact(self, input_data, tool_use_id, context):
        """SDK ``PreCompact`` hook (loop thread): the earliest compaction signal; returns ``{}`` (no opinion)."""
        try:
            with self.lock:
                self._compaction_signal("pre_compact")
        except Exception as error:  # noqa: BLE001 - a hook failure must not block compaction
            self._handler_error("pre_compact", error)
        return {}

    def _on_stream_end(self, error):
        """The receive stream ended (CLI exit, transport error, or our own disconnect while closing)."""
        with self.lock:
            if self.closing:
                return
            try:
                self.projector.abandon_open_items()
            except Exception as failure:  # noqa: BLE001
                self._handler_error("abandon_open_items", failure)
            self._disconnected(error if error is not None else EOFError("claude_code_stream_closed"))

    def _native_active(self):
        return self._native_state in _NATIVE_BUSY_STATES

    # ---- turns ---------------------------------------------------------------------------------------------------
    def _joinable(self):
        """An adopted turn without a host prompt is running: ``run_turn`` joins it (design R4.2 rule 3)."""
        return (self.state == STATE_RUNNING and self.turn_origin == "native_wakeup" and not self._host_prompted
                and not self.closing)

    def _connected(self):
        reader = self._reader
        return self._client is not None and reader is not None and not reader.done() and self._loop_alive()

    def run_turn(self, prompt):
        """Submit ``prompt`` as a new turn, or into a running adopted turn (``capabilities.native_wakeup``); returns
        the turn id once the CLI accepted the stdin write (admission), not at completion. See the module doc for
        the definite rejections and unknown admission."""
        with self.lock:
            if not self._joinable():
                self._check_turn_admissible()
            token = self._submission = dict(cancelled=False)
        try:
            # An interrupt tombstoned the relay thread (non-Codex profiles); this turn, including the set_model
            # validation request below, must be admitted again.
            self._reopen_relay_thread()
            self._apply_pending_model(token)
            with self.lock:
                if token["cancelled"]:
                    self.transcript.event("agent.turn_start_rejected", dict(
                        code="interrupted", message="claude_code_submission_cancelled_by_interrupt", busy=False,
                        state_restored=True))
                    raise AgentRejected("interrupted", "claude_code_submission_cancelled_by_interrupt")
                previous, joined = None, self._joinable()
                if joined:
                    if not self._connected():
                        raise AgentRejected("not_connected", "claude_code_not_connected")
                else:
                    try:
                        previous = self._begin_turn()
                    except ValueError:
                        raise
                    except Exception as error:  # noqa: BLE001 - the Store refused the turn record; nothing changed
                        raise AgentRejected("turn_not_recorded",
                                            f"claude_code_turn_not_recorded:{type(error).__name__}") from error
                    self._last_result, self._end_deferred = None, False
                    if not self._connected():
                        self._reject_turn(previous, "not_connected", "claude_code_not_connected")
                turn_id = self.turn_id
                try:
                    text = self.presentation.present_text(prompt)
                    self.store.raw_agent("sent", dict(command="submit_prompt", thread_id=self.thread_id,
                                                      turn_id=turn_id, input=[dict(type="text", text=text)]),
                                         channel="control")
                    item = self.transcript.user_message(text, turn_id=turn_id)
                    future = self._submit(self._client.query(text))
                except Exception as error:  # noqa: BLE001 - nothing was written: a definite rejection
                    code, message = "submit_not_recorded", f"claude_code_submit_not_recorded:{type(error).__name__}"
                    if previous is not None:
                        self._reject_turn(previous, code, message)
                    raise AgentRejected(code, message) from error
                self._write_future = future
                self._expect_input(item, text, turn_id, "prompt")
                self._host_prompted = True
                if joined:
                    self.transcript.event("agent.turn_joined", dict(origin=self.turn_origin), item_id=item)
        finally:
            with self.lock:
                if self._submission is token:
                    self._submission = None
        try:
            self._wait(future, self.submit_timeout_s)
        except Exception as error:  # noqa: BLE001 - the write may or may not have reached the CLI
            self._disconnected(error)
            raise AgentTransportUnknown(f"claude_code_submit_unknown:{type(error).__name__}") from error
        return turn_id

    def steer(self, text):
        """Add operator ``text`` to the ACTIVE turn (plain ``query()``; the CLI injects it at its next model call, or
        runs it as a further native turn that this Auriga turn waits for when the model had already ended).

        Returns the id of the turn the text belongs to once the stdin write is done: the targeted running turn; or,
        when that turn ended without consuming the text (an interrupt, a guard), the turn that consumed it, else the
        turn running now (the text is attached to it), else the targeted id. Every such case records
        ``agent.steer_not_delivered {delivered_turn_id}``. ``ValueError("no_active_turn_to_steer")`` unless a turn
        runs. A write failure is unknown delivery -> ``disconnected`` + ``AgentTransportUnknown``."""
        with self.lock:
            if self.state != STATE_RUNNING or not self.turn_id or self._client is None:
                raise ValueError("no_active_turn_to_steer")
            turn_id = self.turn_id
            text = self.presentation.present_text(text)
            self.store.raw_agent("sent", dict(command="steer", thread_id=self.thread_id, turn_id=turn_id,
                                              input=[dict(type="text", text=text)]), channel="control")
            item = self.transcript.user_message(text, steer=True, turn_id=turn_id)
            future = self._write_future = self._submit(self._client.query(text))
            self._expect_input(item, text, turn_id, "steer")
            self._steer_unsettled.add(item)
        try:
            self._wait(future, self.submit_timeout_s)
        except Exception as error:  # noqa: BLE001
            with self.lock:
                self._steer_unsettled.discard(item)
            self._disconnected(error)
            raise AgentTransportUnknown(f"claude_code_steer_unknown:{type(error).__name__}") from error
        with self.lock:
            return self._steer_turn(item, turn_id)

    def _steer_turn(self, item, targeted):
        """The turn a written steer belongs to (see ``steer``); call with the lock held."""
        self._steer_unsettled.discard(item)
        if item in self._inputs:
            return self._inputs[item][0]
        if item in self._steer_outcomes:
            delivered = self._steer_outcomes.pop(item)
            if delivered == targeted:
                return targeted
        elif item in self._orphans and self.state == STATE_RUNNING:
            delivered = self.turn_id                 # queued in the CLI: the running turn will consume it
            _, kind, text = self._orphans.pop(item)
            self._inputs[item] = (delivered, kind, text)
        else:
            delivered = None
        self.transcript.event("agent.steer_not_delivered", dict(delivered_turn_id=delivered), turn_id=targeted,
                              item_id=item)
        return delivered or targeted

    def interrupt(self):
        """Interrupt the native turn in progress; returns before it ends (the reader finishes it as ``interrupted``).

        In this order: a ``run_turn`` still preparing its submission is cancelled (it raises
        ``AgentRejected("interrupted")`` and writes nothing); the running host turn is marked interrupted
        (classification); in-flight relay calls are cancelled; a prompt/steer write already scheduled is awaited, so
        the CLI reads the input before the interrupt and interrupts it; then the SDK interrupt control is sent when a
        host turn runs or the CLI reports itself busy. Nothing is sent to an idle CLI with no turn. A control timeout
        is unknown -> ``disconnected`` + ``AgentTransportUnknown``; a control that fails because the CLI is gone
        disconnects the session (no raise); any other failure is re-raised only while the targeted turn runs.
        """
        with self.lock:
            if self._submission is not None:
                self._submission["cancelled"] = True
            host_running = self.state == STATE_RUNNING
            if host_running:
                self._interrupt_requested = True
            target, client, write = (self.turn_id if host_running else None), self._client, self._write_future
            send = (client is not None and not self.closing and self.state not in UNCONNECTED_STATES
                    and (host_running or self._native_active()))
        if self.model_proxy is not None:
            self.model_proxy.cancel_pending("turn_interrupted")
        if not send:
            return
        if write is not None and not write.done():
            concurrent.futures.wait([write], timeout=self.submit_timeout_s)
        self.store.raw_agent("sent", dict(command="interrupt", thread_id=self.thread_id, turn_id=target),
                             channel="control")
        try:
            self._call(client.interrupt(), self.interrupt_timeout_s)
        except TimeoutError as error:
            self._disconnected(error)
            raise AgentTransportUnknown("claude_code_interrupt_timeout") from error
        except Exception as error:  # noqa: BLE001
            self.transcript.event("agent.interrupt_failed", dict(error=type(error).__name__), turn_id=target)
            process = self._process
            if _class_names(error) & _CLI_GONE_ERRORS or (process is not None and process.poll() is not None):
                self._disconnected(error)
                return
            with self.lock:
                still_running = self.state == STATE_RUNNING and self.turn_id == target
            if still_running:
                raise

    def set_model(self, model):
        """Select the model for the NEXT turn. Never blocks: the SDK ``set_model`` control (which first sends one
        charged ``max_tokens: 1`` validation request through the relay) runs at the next ``run_turn`` before the
        prompt is written; a refused change keeps the previous model (``agent.model_change_rejected``)."""
        AgentSession.set_model(self, model)

    def _apply_pending_model(self, token=None):
        with self.lock:
            wanted, native, client = self.model, self._native_model, self._client
        if client is None or native is None or wanted == native:
            return
        try:
            self._call(client.set_model(wanted), self.set_model_timeout_s)
        except TimeoutError as error:
            self._disconnected(error)
            raise AgentTransportUnknown("claude_code_set_model_timeout") from error
        except Exception as error:  # noqa: BLE001 - e.g. check_rejected: the probe request was denied
            if token is not None and token["cancelled"]:
                # The interrupt cancelled the validation request: the change stays pending for the next turn.
                self.transcript.event("agent.model_change_deferred", dict(model=wanted, reason="interrupted"))
                return
            with self.lock:
                if self.model == wanted:
                    self.model = native
            self.transcript.event("agent.model_change_rejected", dict(model=wanted, kept=native,
                                                                      error=type(error).__name__))
            return
        with self.lock:
            self._native_model = wanted
        self.transcript.event("agent.model_changed", dict(model=wanted, previous=native))

    # ---- protocol members ------------------------------------------------------------------------------------
    def owned_processes(self):
        """``[("claude_code", the CLI process)]`` from the moment the SDK spawned it (also during ``connect``) until
        close confirmed it gone, else ``[]``."""
        process = self._discover_process()
        return [("claude_code", process)] if process is not None and not process.gone else []

    def provenance(self):
        """Manifest ``agent_backend`` block (design §7.3) from ``options.provenance_block``; observer data only."""
        from ..factory import evidence_client
        delivery = dict(effective=self.context_delivery, configured=self.cc_config.context_delivery,
                        subagents=self.cc_config.subagents)
        if self.launch is None:
            return dict(AgentSession.provenance(self), wire_api=self.wire_api, upstream_provider=self.provider,
                        model_configured=self.model, effort=self.effort, isolation=self.isolation,
                        context_delivery=self.context_delivery, context_delivery_policy=delivery)
        block = claude_options.provenance_block(
            self.launch, model=self.model, counting=self.counting, wire_api=self.wire_api,
            upstream_provider=self.provider, isolation=self.isolation, context_delivery=self.context_delivery,
            init_tools=self._init_tools, init_model=self._init_model, capabilities=self.capabilities.to_dict(),
            evidence_client=evidence_client(self.config, self.backend_kind))
        block["context_delivery_policy"] = delivery
        if self.bridge is not None:
            block["tool_transport"]["read_only_tools"] = self.bridge.read_only_tools
        return block


    # ---- close -----------------------------------------------------------------------------------------------
    def _stop_backend(self):
        """Stop the CLI and every process it started, then verify (design §3.7).

        Snapshot of the CLI tree while it runs -> SDK disconnect -> loop stop -> ``isolation.stop_agent_tree``
        (shared with OpenCode): a fresh process table right before the kill (members re-validated, so a recycled pid
        or group is not signalled; descendants and groups seen in the periodic refreshes are included), the tree plus
        every process carrying this session's marker (``AGENT_MARKER_ENV`` = instance id, inherited by every native
        tool child; on macOS also orphans in the actor dir whose environment the OS withholds) -> killed ->
        re-scanned. This also finds children a foreground shell detached (``cmd &``, setsid) that were re-parented
        away from the tree, and it works after a CLI crash or a failed connect (``agent_exited_before_teardown`` /
        ``agent_process_unobserved`` are recorded as facts). Teardown is confirmed only when nothing targeted survived, the final re-scan found no
        marked process and every scan succeeded; else ``agent_process_exit_unconfirmed``.
        """
        report = {}
        self._tree_stop.set()
        process = self._discover_process()
        alive = process is not None and process.poll() is None
        if alive:
            self._refresh_tree()
        elif process is not None:
            report["agent_exited_before_teardown"] = True
        if process is None and self._client is not None and self._connect_failed:
            # connect failed before the spawned process (if any) could be observed: only the marker can find it.
            report["agent_process_unobserved"] = True
        client = self._client
        if client is not None and self._loop_alive():
            try:
                self._call(client.disconnect(), self.disconnect_timeout_s)
                report["disconnect"] = "completed"
            except Exception as error:  # noqa: BLE001 - fall back to the process-tree teardown
                report["disconnect"] = "failed:" + type(error).__name__
        self._stop_loop()
        if self._tree_thread is not None and self._tree_thread is not threading.current_thread():
            self._tree_thread.join(2)
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
        if not confirmed:
            report["agent_process_exit_unconfirmed"] = True
        elif process is not None:
            process.gone = True
        self._teardown_confirmed = confirmed
        self._close_sandbox(report)
        with self._log_lock:
            if self._log is not None:
                self._log.close()
        if self.close_report is not None:
            self.close_report["backend"] = report

    def _close_sandbox(self, report):
        """Stop the sandbox's host-side relays and owner socket (after the CLI tree is gone: closing the owner socket
        also makes the in-namespace supervisor kill whatever is left) and remove the run-owned IPC directory."""
        sandbox = self.launch.sandbox if self.launch is not None else None
        try:
            if sandbox is not None:
                sandbox.close()
                report["sandbox"] = "closed"
        except Exception as error:  # noqa: BLE001 - reported, never blocks the rest of close
            report["sandbox"] = "close_failed:" + type(error).__name__
        ipc, self._sandbox_ipc = self._sandbox_ipc, None
        if ipc is not None:
            try:
                ipc.rmdir()
            except OSError:
                pass

    def _stop_loop(self):
        loop, thread = self._loop, self._loop_thread
        if loop is None:
            return
        if self._loop_alive():
            async def drain():
                self._cancel_quiet()
                reader = self._reader
                if reader is not None and not reader.done():
                    reader.cancel()
                    await asyncio.wait([reader], timeout=2)
            try:
                self._call(drain(), 5)
            except Exception:  # noqa: BLE001
                pass
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(self.loop_join_timeout_s)
