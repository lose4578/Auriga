"""Scripted stand-in for ``claude_agent_sdk.ClaudeSDKClient`` (SDK 0.2.180) for ``ClaudeCodeSession`` unit tests.

No process, no model, no network. ``FakeClaudeSDK`` is the controller and the ``client_factory``; every client it
creates is a ``FakeClaudeSDKClient`` that follows the real client's observable semantics (read from the installed
package, ``client.py`` and ``_internal/query.py``):

* ``connect(prompt=None)`` makes the client usable; before it, ``query``/``interrupt``/``set_model``/
  ``receive_messages`` raise ``CLIConnectionError("Not connected. Call connect() first.")``. A failing ``connect``
  disconnects before raising (like the real one).
* ``receive_messages()`` is ONE consumer's async iterator: it yields messages in order and ends cleanly on the end
  sentinel (CLI exited cleanly, or ``disconnect()``); a scripted error (``ProcessError``, ``ResultError``,
  ``CLIConnectionError`` ...) is raised from it exactly once and ends it. A second concurrent consumer is a test
  error (``RuntimeError("fake_second_receive_consumer")``): the real stream would silently split messages.
* ``query(prompt, session_id="default")`` returns after the (fake) stdin write; it only records and runs the
  ``on_query`` handlers. ``interrupt()`` / ``set_model()`` are control requests (return when "answered").
* ``disconnect()`` is idempotent, ends the receive stream and makes later calls raise ``CLIConnectionError``.
* After ``kill_process()`` (CLI death) a control request or write that was waiting fails with the reader's
  ``ProcessError`` (the real ``Query`` fails pending controls with it), and a new ``query``/``interrupt``/
  ``set_model`` fails at once with ``CLIConnectionError("Cannot write to terminated process ...")`` (the real
  ``SubprocessCLITransport.write``).
* ``_transport._process`` is a ``FakeAnyioProcess`` (``pid``, ``returncode``, ``terminate``, ``kill``) so code that
  adapts the one private SDK attribute (plan 07 section 2.2.6) can be tested.

Thread model: the client coroutines run on the loop of whoever awaits them (the session's ``claude-loop``); the
scripting methods (``emit``/``end``/``fail``/``kill_process``/``release`` ...) may be called from ANY thread.

Scripting: ``emit(*messages)``; ``on_query`` handlers ``(client, text) -> None`` (sync, run on the loop, may ``emit``);
fault injection attributes ``connect_error``, ``query_error``, ``interrupt_error``, ``set_model_error``,
``disconnect_error`` (exception instance or ``(call_index) -> exception | None``) and ``gate_<call>`` events
(``hold("query")`` makes that call await until ``release("query")`` -- emulates the SDK's 15/60/70 s waits).
``client.calls`` is the ordered ``[(name, args...)]`` record; ``wait_for_call`` blocks a test thread until a call
happened. Stdin writes are serialised FIFO like the real transport's write lock: a held ``query`` keeps a later
``interrupt``/``set_model`` write behind it, and ``client.writes`` lists the writes in the order they reached stdin.
The CLI process is spawned (``_transport._process`` set) before the ``connect`` gate, as in the real SDK, so a test
can observe it while ``initialize`` is pending.

Message builders (``init_message``, ``assistant_message``, ``tool_use``, ``tool_result_message``, ``result_message``,
``state_message``, ``turn_messages``) construct the real SDK dataclasses with real-shaped defaults;
``load_capture``/``message_from_capture`` rebuild messages from the spike evidence (``sdk-messages.jsonl``).
"""
import asyncio
import dataclasses
import itertools
import json
import threading
import time
from pathlib import Path

try:
    import claude_agent_sdk as sdk
    from claude_agent_sdk import (AssistantMessage, ResultMessage, SystemMessage, TextBlock, ThinkingBlock,
                                  ToolResultBlock, ToolUseBlock, UserMessage)
    from claude_agent_sdk._errors import CLIConnectionError, ProcessError, ResultError
    SDK_AVAILABLE = True
except ImportError:  # optional extra not installed: importers skip their tests
    sdk = None
    SDK_AVAILABLE = False

NOT_CONNECTED = "Not connected. Call connect() first."
# Default pid of the fake CLI process: above every OS pid range (Linux pid_max <= 2**22, macOS < 100000), so code
# that snapshots or signals the "CLI" tree can never touch a real process.
FAKE_PID = 2 ** 30 - 1
_END = ("end", None)
_ids = itertools.count(1)


def requires_sdk(test):
    """``unittest.skipUnless`` decorator: skip when ``claude-agent-sdk`` is not installed."""
    import unittest
    return unittest.skipUnless(SDK_AVAILABLE, "claude-agent-sdk (optional extra) is not installed")(test)


# ------------------------------------------------------------------------------------------------ builders

def init_message(session_id, *, tools=(), model="fake-model", cwd="/run/actor", mcp_servers=None):
    """``SystemMessage`` init frame (real shape: tools, model, permissionMode, claude_code_version ...)."""
    return SystemMessage(subtype="init", data=dict(
        type="system", subtype="init", cwd=cwd, session_id=session_id, tools=list(tools), model=model,
        mcp_servers=mcp_servers if mcp_servers is not None else [dict(name="auriga", status="connected")],
        permissionMode="dontAsk", apiKeySource="ANTHROPIC_API_KEY", claude_code_version="2.1.321",
        uuid=f"init-{next(_ids)}", slash_commands=[], skills=[], plugins=[]))


def state_message(state, session_id="s"):
    """``SystemMessage`` session_state_changed (``running`` | ``idle`` | ``requires_action``)."""
    return SystemMessage(subtype="session_state_changed", data=dict(
        type="system", subtype="session_state_changed", state=state, uuid=f"state-{next(_ids)}", session_id=session_id))


def system_message(subtype, session_id="s", **data):
    """Any other ``SystemMessage`` (``status``, ``compact_boundary``, ``permission_denied``, ``api_retry`` ...)."""
    return SystemMessage(subtype=subtype, data=dict(type="system", subtype=subtype, session_id=session_id,
                                                    uuid=f"sys-{next(_ids)}", **data))


def tool_use(tool_use_id, name, **arguments):
    return ToolUseBlock(id=tool_use_id, name=name, input=arguments)


def usage_dict(input_tokens=10, output_tokens=1, cache_read=0, cache_creation=0):
    return dict(input_tokens=input_tokens, cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read, output_tokens=output_tokens)


def assistant_message(*blocks, message_id=None, parent_tool_use_id=None, session_id="s", usage=None, error=None,
                      model="fake-model"):
    """``AssistantMessage``; a ``str`` block becomes a ``TextBlock``. Like the CLI it carries ONE content block
    per message in practice, but any number is allowed."""
    content = [TextBlock(text=block) if isinstance(block, str) else block for block in blocks]
    return AssistantMessage(content=content, model=model, parent_tool_use_id=parent_tool_use_id, error=error,
                            usage=usage if usage is not None else usage_dict(), message_id=message_id or f"msg_{next(_ids)}",
                            session_id=session_id, uuid=f"asst-{next(_ids)}")


def tool_result_message(tool_use_id, content="ok", *, is_error=None, parent_tool_use_id=None, extra=()):
    """``UserMessage`` carrying one ``ToolResultBlock`` (``content``: str or MCP-style item list)."""
    block = ToolResultBlock(tool_use_id=tool_use_id, content=content, is_error=is_error)
    return UserMessage(content=[block, *extra], uuid=f"user-{next(_ids)}", parent_tool_use_id=parent_tool_use_id)


def echo_message(text):
    """The ``--replay-user-messages`` echo of a prompt/steer: a ``UserMessage`` with string content."""
    return UserMessage(content=text, uuid=f"echo-{next(_ids)}", parent_tool_use_id=None)


def result_message(*, result="done", is_error=False, subtype="success", terminal_reason="completed",
                   api_error_status=None, session_id="s", permission_denials=None, usage=None, stop_reason="end_turn"):
    return ResultMessage(
        subtype=subtype, duration_ms=1, duration_api_ms=1, is_error=is_error, num_turns=1, session_id=session_id,
        stop_reason=stop_reason, total_cost_usd=0.0, usage=usage or {}, result=None if is_error else result,
        permission_denials=permission_denials if permission_denials is not None else [],
        api_error_status=api_error_status, uuid=f"result-{next(_ids)}", terminal_reason=terminal_reason)


def turn_messages(answer="done", *, session_id="s", tools=(), model="fake-model", with_init=True):
    """The messages of one plain turn as the real CLI emits them with state events on:
    ``running``, ``init``, assistant text, ``ResultMessage``, ``idle``."""
    out = [state_message("running", session_id)]
    if with_init:
        out.append(init_message(session_id, tools=tools, model=model))
    out += [assistant_message(answer, session_id=session_id), result_message(result=answer, session_id=session_id),
            state_message("idle", session_id)]
    return out


# ------------------------------------------------------------------------------------- spike evidence loader

def message_from_capture(record):
    """Rebuild one SDK message object from a spike ``sdk-messages.jsonl`` record (``_type`` + dataclass fields;
    system subclasses such as ``TaskStartedMessage`` become plain ``SystemMessage``)."""
    kind = record["_type"]
    if "subtype" in record and "data" in record:
        return SystemMessage(subtype=record["subtype"], data=record["data"])

    def convert(value):
        if isinstance(value, dict) and "_type" in value:
            return message_from_capture(value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    cls = getattr(sdk, kind)
    names = {item.name for item in dataclasses.fields(cls)}
    return cls(**{key: convert(value) for key, value in record.items() if key in names})


def load_capture(path):
    """All messages of a captured ``sdk-messages.jsonl`` (or fixture) file, in order."""
    return [message_from_capture(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------------------------- fake process

class FakeAnyioProcess:
    """Stand-in for the anyio ``Process`` behind ``client._transport._process`` (``pid``, ``returncode``,
    ``terminate``, ``kill``). ``returncode`` is None until ``exit(code)``/``kill``/``terminate``."""

    def __init__(self, pid=FAKE_PID):
        self.pid, self.returncode, self.signals = pid, None, []

    def exit(self, code=0):
        self.returncode = code

    def terminate(self):
        self.signals.append("TERM")
        self.returncode = -15 if self.returncode is None else self.returncode

    def kill(self):
        self.signals.append("KILL")
        self.returncode = -9 if self.returncode is None else self.returncode


class _FakeTransport:
    def __init__(self, process):
        self._process = process


# --------------------------------------------------------------------------------------------------- client

class FakeClaudeSDKClient:
    """See the module doc. Constructed by ``FakeClaudeSDK.factory`` (or directly with the real signature)."""

    def __init__(self, options=None, transport=None, *, controller=None):
        self.options = options
        self.controller = controller
        self.calls = []                    # [("connect",), ("query", text, session_id), ("interrupt",), ...]
        self.writes = []                   # "query" | "interrupt" | "set_model" in the order they reached stdin
        self.on_connect = []               # messages emitted right after connect()
        self.on_query = []                 # handlers (client, text) -> None
        self.on_interrupt = []             # handlers (client) -> None
        for name in ("connect", "query", "interrupt", "set_model", "disconnect"):
            setattr(self, f"{name}_error", None)
        self.process = FakeAnyioProcess()
        self._transport = None
        self._connected = False
        self._loop = None
        self._queue = None
        self._consumer_active = False
        self._gates = {}
        self._cond = threading.Condition()
        self._counts = {}
        self._dead_code = None             # exit code once kill_process() ran
        self._write_lock = None            # asyncio.Lock: stdin writes are serialised FIFO like the real transport

    # ----- scripting (any thread) -----
    def emit(self, *messages):
        """Deliver messages to the receive stream (in order). Raises RuntimeError before connect()."""
        self._push(*[("msg", message) for message in messages])

    def end(self):
        """Clean end of the stream (the CLI exited with code 0 / EOF)."""
        self._push(_END)

    def fail(self, error):
        """The next ``receive_messages`` step raises ``error`` (e.g. ``ProcessError``) and the stream ends."""
        self._push(("error", error))

    def kill_process(self, code=-9, error=None):
        """CLI death: process gone, stream raises ``error`` (default ``ProcessError``); waiting and later SDK calls
        fail like the real ones (module doc)."""
        self.process.exit(code)
        self._dead_code = code
        self.fail(error or ProcessError(f"Command failed with exit code {code}", exit_code=code))

    def hold(self, call):
        """Make every ``call`` (``query``/``interrupt``/``set_model``/``connect``/``disconnect``) await ``release``."""
        self._gates[call] = threading.Event()

    def release(self, call):
        gate = self._gates.get(call)
        if gate is not None:
            gate.set()

    def call_names(self):
        return [item[0] for item in self.calls]

    def queries(self):
        """Texts passed to ``query`` in order."""
        return [item[1] for item in self.calls if item[0] == "query"]

    def wait_for_call(self, name, count=1, timeout=5.0):
        """Block (test thread) until ``name`` was called ``count`` times; False on timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._counts.get(name, 0) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    # ----- internals -----
    def _push(self, *items):
        if self._loop is None or self._queue is None:
            raise RuntimeError("fake_client_not_connected")
        for item in items:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def _record(self, name, *args):
        with self._cond:
            self.calls.append((name, *args))
            self._counts[name] = self._counts.get(name, 0) + 1
            index = self._counts[name]
            self._cond.notify_all()
        return index

    def _require_connected(self):
        if not self._connected:
            raise CLIConnectionError(NOT_CONNECTED)

    def _require_alive(self):
        if self._dead_code is not None:
            raise CLIConnectionError(f"Cannot write to terminated process (exit code: {self._dead_code})")

    async def _gate_and_fault(self, name, index):
        gate = self._gates.get(name)
        if gate is not None:
            while not gate.is_set():
                if self._dead_code is not None and name in ("query", "interrupt", "set_model"):
                    raise ProcessError(f"Command failed with exit code {self._dead_code}", exit_code=self._dead_code)
                await asyncio.sleep(0.005)
        error = getattr(self, f"{name}_error")
        error = error(index) if callable(error) else error
        if error is not None:
            raise error

    # ----- SDK surface -----
    async def connect(self, prompt=None):
        index = self._record("connect")
        # Like the real SDK: the transport spawns the CLI first, then the initialize control runs (the gate); a
        # failing connect disconnects, which terminates the process and drops the transport.
        self._transport = _FakeTransport(self.process)
        self._write_lock = asyncio.Lock()
        try:
            await self._gate_and_fault("connect", index)
        except BaseException:
            self.process.terminate()
            await self.disconnect()
            self._transport = None
            raise
        self._loop, self._queue = asyncio.get_running_loop(), asyncio.Queue()
        self._connected = True
        controller = self.controller
        for message in [*(controller.initial_messages(self) if controller else []), *self.on_connect]:
            self._queue.put_nowait(("msg", message))

    async def receive_messages(self):
        self._require_connected()
        if self._consumer_active:
            raise RuntimeError("fake_second_receive_consumer")
        self._consumer_active = True
        try:
            while True:
                kind, payload = await self._queue.get()
                if kind == "end":
                    return
                if kind == "error":
                    raise payload
                yield payload
        finally:
            self._consumer_active = False

    async def query(self, prompt, session_id="default"):
        self._require_connected()
        index = self._record("query", prompt if isinstance(prompt, str) else "<stream>", session_id)
        self._require_alive()
        async with self._write_lock:   # held "query" = a stdin write in progress; later writes queue behind it
            await self._gate_and_fault("query", index)
            self.writes.append("query")
            for handler in [*(self.controller.on_query if self.controller else []), *self.on_query]:
                handler(self, prompt)

    async def interrupt(self):
        self._require_connected()
        index = self._record("interrupt")
        self._require_alive()
        async with self._write_lock:   # the control request's own write
            self.writes.append("interrupt")
        await self._gate_and_fault("interrupt", index)   # held "interrupt" = waiting for the control response
        for handler in [*(self.controller.on_interrupt if self.controller else []), *self.on_interrupt]:
            handler(self)

    async def set_model(self, model=None):
        self._require_connected()
        index = self._record("set_model", model)
        self._require_alive()
        async with self._write_lock:
            self.writes.append("set_model")
        await self._gate_and_fault("set_model", index)

    async def disconnect(self):
        index = self._record("disconnect")
        was_connected, self._connected = self._connected, False
        if was_connected and self._queue is not None:
            self._queue.put_nowait(_END)           # the real Query always sends the end sentinel on close
        await self._gate_and_fault("disconnect", index)


# ------------------------------------------------------------------------------------------------ controller

class FakeClaudeSDK:
    """Factory and registry of fake clients: pass ``fake.factory`` as the session's ``client_factory``.

    ``mcp_tools``: bare registry names listed as ``mcp__auriga__<name>`` in the init frame of
    ``emit_init_on_connect``. ``connect_hook(client) -> list[message]`` (default None: connect emits nothing) and the
    ``on_query``/``on_interrupt`` handler lists are inherited by every client; ``connect_error`` is copied to each
    new client; ``factory_error`` makes ``factory`` itself raise. ``clients``: every client created, oldest first;
    ``client``: the latest.
    """

    def __init__(self, *, mcp_tools=(), model="fake-model"):
        self.clients = []
        self.mcp_tools = tuple(mcp_tools)
        self.model = model
        self.on_query = []
        self.on_interrupt = []
        self.connect_hook = None
        self.connect_error = None
        self.factory_error = None

    @property
    def client(self):
        return self.clients[-1]

    def initial_messages(self, client):
        """Messages a client emits right after connect (``connect_hook``; default none)."""
        return self.connect_hook(client) if self.connect_hook else []

    def emit_init_on_connect(self):
        """Every client emits an init frame (session id = options.session_id or options.resume) right after connect."""
        def hook(client):
            options = client.options
            session_id = getattr(options, "session_id", None) or getattr(options, "resume", None) or "s"
            tools = [*(getattr(options, "tools", None) or []), *(f"mcp__auriga__{name}" for name in self.mcp_tools)]
            return [init_message(session_id, tools=tools, model=getattr(options, "model", None) or self.model)]
        self.connect_hook = hook
        return self

    def factory(self, options=None, transport=None):
        """The ``ClaudeSDKClient`` replacement: same constructor signature."""
        if self.factory_error is not None:
            raise self.factory_error
        client = FakeClaudeSDKClient(options, transport, controller=self)
        client.connect_error = self.connect_error
        self.clients.append(client)
        return client
