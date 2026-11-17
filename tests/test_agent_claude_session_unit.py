"""ClaudeCodeSession with the scripted fake SDK client (plan 07 §2, §2.8 failure paths; T0 report 08 shapes).

No CLI process and no model: ``tests/support/fake_claude_sdk.FakeClaudeSDK`` replaces ``ClaudeSDKClient`` (injected
as ``client_factory``), a pre-resolved ``ClaudeRuntime`` skips the version probe, the relay points at a closed port
(no test here sends a model request) and the MCP endpoint is the real ``AurigaMcpServer`` driven by FakeMcpClient.
"""
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from support.fake_claude_sdk import (FAKE_PID, SDK_AVAILABLE, FakeClaudeSDK, assistant_message, echo_message,
                                     init_message, load_capture, requires_sdk, result_message, state_message,
                                     system_message, tool_result_message, tool_use)
from support.fake_mcp_client import FakeMcpClient

from auriga.core.config import Config, ConfigError
from auriga.core.storage.store import Store
from auriga.cortex.agents import factory
from auriga.cortex.agents.base import AgentRejected, AgentTransportUnknown

if SDK_AVAILABLE:
    from claude_agent_sdk._errors import CLIConnectionError, ProcessError
    from auriga.cortex.agents.claude_code import options as claude_options
    from auriga.cortex.agents.claude_code.session import ClaudeCodeSession
    from auriga.cortex.agents.isolation import VersionMismatch

FIXTURES = Path(__file__).parent / "fixtures" / "claude_sdk"
IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
KEY_ENV = "AURIGA_TEST_CLAUDE_UPSTREAM_KEY"
SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x in m, default 0"}}}
TOOLS = [{"type": "function", "name": "look", "description": "Read-only view.", "inputSchema": SCHEMA},
         {"type": "function", "name": "move", "description": "Stateful motion.", "inputSchema": SCHEMA}]
CONFIG = {"agent": {"backend": "claude_code", "claude_code": {"upstream": {
              "base_url": "http://127.0.0.1:9", "api_key_env": KEY_ENV}}},
          "model": {"name": "claude-test-model", "effort": "high"},
          "features": {"self_briefing": False, "visual_review_model": None}}


def text_result(text, success=True):
    return {"success": success, "contentItems": [{"type": "inputText", "text": text}]}


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def plain_turn(answer="FINAL"):
    """Messages of one plain native turn as the CLI emits them (state events on, prompt echo on)."""
    def build(sid, text):
        return [state_message("running", sid), init_message(sid), echo_message(text),
                assistant_message(answer, session_id=sid), result_message(result=answer, session_id=sid),
                state_message("idle", sid)]
    return build


def opening(sid, text):
    return [state_message("running", sid), init_message(sid), echo_message(text)]


def ending(sid, *, answer="FINAL", **result):
    return [assistant_message(answer, session_id=sid), result_message(session_id=sid, **result),
            state_message("idle", sid)]


@requires_sdk
class SessionCase(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve() / "run"
        self.root.mkdir()
        self.store = Store(self.root, IDENTITY)
        self.addCleanup(self.store.close)
        environment = mock.patch.dict(os.environ, {KEY_ENV: "host-secret"})
        environment.start()
        self.addCleanup(environment.stop)
        self.config = Config.from_dict(CONFIG)
        self.fake = FakeClaudeSDK()
        self.scripts = []
        self.fake.on_query.append(self._on_query)
        self.handler_calls, self.disconnects, self.resets = [], [], []
        self.runtime = claude_options.ClaudeRuntime(sdk_version=claude_options.SDK_VERSION,
                                                    cli_version=claude_options.CLI_VERSION,
                                                    executable="/usr/bin/true", sha256="0" * 64)

    # ---- helpers --------------------------------------------------------------------------------------------
    def _on_query(self, client, text):
        if self.scripts:
            build = self.scripts.pop(0)
            messages = build(self.session.thread_id, text) if callable(build) else build
            if messages:
                client.emit(*messages)

    def script(self, *builds):
        self.scripts.extend(builds)

    def tool_handler(self, params):
        self.handler_calls.append(params)
        gate = getattr(self, "tool_gate", None)
        if gate is not None:
            gate.wait(5)
        return text_result("ok " + params["tool"])

    def make(self, **overrides):
        session = ClaudeCodeSession(self.root, self.store, self.tool_handler, config=self.config,
                                    client_factory=self.fake.factory, runtime=self.runtime)
        session.on_disconnect = lambda: self.disconnects.append(session.state)
        session.on_context_reset = self.resets.append
        session.tool_parallel_hints = {"look": True, "move": False}
        session.tool_parallel_safe = lambda tool, arguments: tool == "look"
        session.submit_timeout_s = session.interrupt_timeout_s = session.set_model_timeout_s = 2.0
        session.result_quiet_guard_s = 0.5
        for key, value in overrides.items():
            setattr(session, key, value)
        self.session = session
        self.addCleanup(session.close)
        return session

    def started(self, resume=None, **overrides):
        session = self.make(**overrides)
        session.start(TOOLS, "Be careful.", resume)
        return session

    @property
    def client(self):
        return self.fake.client

    def events(self, kind):
        return self.store.events_by_type([kind], limit=1000)

    def raw(self, channel):
        path = self.root / "agent-raw.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [record for record in records if record["channel"] == channel]

    def finish(self, session, timeout=5.0):
        self.assertTrue(session.turn_done.wait(timeout), "turn did not end")
        return session.state

    def mcp(self, session):
        client = FakeMcpClient(session.bridge.url, headers=session.bridge.headers)
        client.handshake()
        return client


class StartTests(SessionCase):
    def test_native_memory_follows_model_native_memories(self):
        session = self.started()
        self.assertNotIn("CLAUDE_CODE_DISABLE_AUTO_MEMORY", session.launch.env)
        self.assertNotIn("CLAUDE_CODE_DISABLE_CLAUDE_MDS", session.launch.env)
        session.close()
        self.root = self.root.parent / "run-no-memory"
        self.root.mkdir()
        self.store = Store(self.root, IDENTITY)
        self.addCleanup(self.store.close)
        self.config = Config.from_dict({**CONFIG, "model": {**CONFIG["model"], "native_memories": False}})
        session = self.started()
        self.assertEqual(session.launch.env.get("CLAUDE_CODE_DISABLE_AUTO_MEMORY"), "1")

    def test_fresh_start_writes_the_identity_and_wires_relay_mcp_and_hooks(self):
        session = self.started()
        self.assertEqual(session.state, "idle")
        self.assertEqual(str(uuid.UUID(session.thread_id)), session.thread_id)
        options = self.client.options
        self.assertEqual((options.session_id, options.resume), (session.thread_id, None))
        self.assertEqual(options.mcp_servers["auriga"]["url"], session.bridge.url)
        self.assertEqual(options.mcp_servers["auriga"]["headers"], session.bridge.headers)
        self.assertIn("PreCompact", options.hooks)
        self.assertEqual(options.env, {})  # the launcher, not options.env, carries the environment
        launcher = Path(options.cli_path).read_text()
        self.assertIn("ANTHROPIC_BASE_URL=" + session.model_proxy.base_url, launcher)
        self.assertNotIn("host-secret", launcher)
        identity = self.store.read_state("runtime", "agent")
        self.assertEqual({key: identity[key] for key in ("backend", "thread_id", "pid", "model", "provider", "effort",
                                                         "isolation", "counting", "wire_api", "sdk_version",
                                                         "cli_version", "clean_shutdown", "resumed")},
                         dict(backend="claude_code", thread_id=session.thread_id, pid=FAKE_PID, model="claude-test-model",
                              provider="anthropic", effort="high", isolation="disabled",
                              counting="delivered_usable_generation_completions_v1", wire_api="anthropic_messages",
                              sdk_version="0.2.180", cli_version="2.1.321", clean_shutdown=False, resumed=False))
        self.assertEqual(identity["native"], {"session_dir": "private/claude-config"})
        self.assertEqual(session.model_proxy.thread_owners[session.thread_id]["source"], "actor")
        self.assertEqual(session.model_proxy.default_thread_id, session.thread_id)
        (role, process), = session.owned_processes()
        self.assertEqual((role, process.pid, process.poll()), ("claude_code", FAKE_PID, None))
        self.assertIs(session.owned_processes()[0][1], process)  # one stable object for abort_startup
        self.assertTrue((self.root / "private" / "agent.log").exists())
        self.assertEqual(session.context_delivery, "full")
        self.assertFalse(session.capabilities.live_effort)
        self.assertTrue(session.capabilities.steer)

    def test_the_private_process_attribute_exists_in_the_pinned_sdk(self):
        # owned_processes() reads ClaudeSDKClient._transport._process (the one private SDK attribute, plan 07 §2.2.6).
        import importlib.metadata
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

        async def empty():
            return
            yield {}
        self.assertEqual(importlib.metadata.version("claude-agent-sdk"), "0.2.180")
        stream = empty()
        transport = SubprocessCLITransport(prompt=stream, options=ClaudeAgentOptions(cli_path="/usr/bin/true"))
        self.assertIsNone(transport._process)
        asyncio.run(stream.aclose())
        # Without the attribute (another SDK) the session owns nothing rather than failing.
        self.fake.connect_hook = None
        session = self.make()
        original = self.fake.factory

        def factory(options=None, transport=None):
            client = original(options, transport)
            client.process.pid = None
            return client
        session._client_factory = factory
        session.start(TOOLS, "x")
        self.assertEqual(session.owned_processes(), [])

    def test_start_twice_is_refused(self):
        session = self.started()
        with self.assertRaisesRegex(ValueError, "^session_already_started$"):
            session.start(TOOLS, "again")

    def test_connect_failure_closes_everything_and_writes_no_identity(self):
        self.fake.connect_error = CLIConnectionError("cli did not start")
        session = self.make()
        with self.assertRaises(CLIConnectionError):
            session.start(TOOLS, "x")
        self.assertEqual(session.state, "closed")
        self.assertIsNone(self.store.read_state("runtime", "agent"))
        self.assertTrue(session.model_proxy.closed)
        self.assertTrue(session.bridge.server.closed)
        self.assertEqual(self.disconnects, [])

    def test_factory_failure_and_version_mismatch_close_the_session(self):
        self.fake.factory_error = RuntimeError("no client")
        session = self.make()
        with self.assertRaisesRegex(RuntimeError, "no client"):
            session.start(TOOLS, "x")
        self.assertEqual(session.state, "closed")
        self.fake.factory_error = None
        session = self.make(runtime=None)
        with mock.patch.object(claude_options, "resolve_runtime",
                               side_effect=VersionMismatch("claude_code", "2.1.999", "2.1.321")):
            with self.assertRaisesRegex(VersionMismatch, "^claude_code_version_mismatch$"):
                session.start(TOOLS, "x")
        self.assertEqual(session.state, "closed")
        self.assertIsNone(session.model_proxy)

    def test_missing_upstream_key_refuses_start(self):
        del os.environ[KEY_ENV]
        session = self.make()
        with self.assertRaisesRegex(ValueError, "^claude_code_upstream_key_missing"):
            session.start(TOOLS, "x")
        self.assertEqual(session.state, "closed")

    def test_connect_timeout_closes(self):
        self.fake.connect_hook = None
        session = self.make(connect_timeout_s=0.2)
        original = self.fake.factory

        def factory(options=None, transport=None):
            client = original(options, transport)
            client.hold("connect")
            return client
        session._client_factory = factory
        with self.assertRaisesRegex(TimeoutError, "claude_code_command_timeout"):
            session.start(TOOLS, "x")
        self.assertEqual(session.state, "closed")

    def test_provenance_records_the_runtime_and_the_init_tools(self):
        session = self.started()
        self.script(plain_turn())
        session.run_turn("go")
        self.finish(session)
        block = session.provenance()
        self.assertEqual((block["kind"], block["cli"]["version"], block["sdk"]["version"], block["wire_api"],
                          block["context_delivery"], block["tool_transport"]["tool_name_prefix"]),
                         ("claude_code", "2.1.321", "0.2.180", "anthropic_messages", "full", "mcp__auriga__"))
        self.assertEqual(block["native_model_id"], "fake-model")  # what the CLI's init frame reported
        self.assertEqual(block["tool_transport"]["read_only_tools"], ["look"])
        self.assertFalse(block["capabilities"]["live_effort"])


class TurnTests(SessionCase):
    def test_admission_returns_before_completion_and_idle_ends_the_turn(self):
        session = self.started()
        self.script(opening)
        turn = session.run_turn("Do the task sha256 " + "ab" * 32)
        self.assertTrue(turn.startswith("turn-"))
        self.assertEqual(session.state, "running")
        self.assertFalse(session.turn_done.is_set())
        sid = session.thread_id
        self.client.emit(assistant_message("thinking", message_id="m1", session_id=sid),
                         assistant_message(tool_use("toolu_1", "mcp__auriga__look", x=1), message_id="m1",
                                           session_id=sid))
        call = self.mcp(session).call_tool("look", {"x": 1}, meta={"claudecode/toolUseId": "toolu_1"})
        self.assertFalse(call.result["isError"])
        self.client.emit(tool_result_message("toolu_1", "ok look"), *ending(sid, answer="All done."))
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.turn_result, dict(id=turn, status="completed", error=None))
        self.assertEqual(session.completed_final_answer(), "All done.")
        params, = self.handler_calls
        self.assertEqual((params["threadId"], params["turnId"], params["callId"], params["callIdSource"]),
                         (sid, turn, "toolu_1", "native_tool_use_id"))
        row = self.store.transcript_item("toolu_1")
        self.assertEqual((row["type"], row["message_id"], row["turn_id"]), ("dynamicToolCall", "m1", turn))
        submit, = self.raw("control")
        self.assertEqual((submit["direction"], submit["message"]["command"], submit["message"]["turn_id"]),
                         ("sent", "submit_prompt", turn))
        # The prompt went through the actor presentation before submission (digest removed).
        self.assertNotIn("ab" * 32, submit["message"]["input"][0]["text"])
        self.assertEqual(self.client.queries(), [submit["message"]["input"][0]["text"]])
        prompt_rows = [r for r in self.store.states(self.store.TRANSCRIPT_KIND) if r["type"] == "userMessage"]
        self.assertTrue(prompt_rows[0]["native_id"].startswith("echo-"))  # consumed echo
        self.assertEqual(len(self.events("agent.turn_completed")), 1)
        self.assertTrue(self.raw("native"))
        self.assertTrue(self.events("agent.native"))
        self.assertEqual(self.events("agent.turn_end_fallback"), [])

    def test_run_turn_refusals(self):
        session = self.make()
        with self.assertRaisesRegex(ValueError, "^session_not_connected$"):
            session.run_turn("before start")
        session.start(TOOLS, "x")
        self.script(opening)
        session.run_turn("one")
        with self.assertRaisesRegex(ValueError, "^turn_already_active$"):
            session.run_turn("two")
        self.assertEqual(len(self.client.queries()), 1)

    def test_preflight_rejection_restores_the_previous_turn(self):
        session = self.started()
        self.script(plain_turn())
        first = session.run_turn("one")
        self.finish(session)
        kept = session.turn_result
        session._reader = SimpleNamespace(done=lambda: True)
        with self.assertRaises(AgentRejected) as raised:
            session.run_turn("two")
        self.assertEqual(raised.exception.code, "not_connected")
        self.assertEqual((session.state, session.turn_id, session.turn_result), ("completed", first, kept))
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(len(self.events("agent.turn_start_rejected")), 1)
        self.assertEqual(len(self.client.queries()), 1)
        self.assertEqual(self.disconnects, [])

    def test_write_error_is_unknown_admission_never_retried(self):
        session = self.started()
        self.client.query_error = CLIConnectionError("broken pipe")
        with self.assertRaisesRegex(AgentTransportUnknown, "claude_code_submit_unknown:CLIConnectionError"):
            session.run_turn("go")
        self.assertEqual(session.state, "disconnected")
        self.assertEqual(session.turn_result["status"], "failed")
        self.assertEqual(session.turn_result["error"]["message"], "upstream_connection_lost: CLIConnectionError")
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(self.disconnects, ["disconnected"])
        self.assertEqual(len(self.client.queries()), 1)
        with self.assertRaisesRegex(ValueError, "^session_not_connected$"):
            session.run_turn("again")
        # Tool calls of the lost connection are stale.
        result = self.mcp(session).call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_x"})
        self.assertIn("native_tool_call_stale_connection", result.result["content"][0]["text"])

    def test_write_timeout_is_unknown_admission(self):
        session = self.started(submit_timeout_s=0.2)
        self.client.hold("query")
        with self.assertRaisesRegex(AgentTransportUnknown, "TimeoutError"):
            session.run_turn("go")
        self.client.release("query")
        self.assertEqual((session.state, self.disconnects), ("disconnected", ["disconnected"]))

    def test_cli_exit_mid_turn_disconnects_once(self):
        session = self.started()
        self.script(opening)
        session.run_turn("go")
        self.client.emit(assistant_message(tool_use("toolu_b", "Bash", command="sleep 9"),
                                           session_id=session.thread_id))
        self.client.kill_process(code=1)
        self.assertEqual(self.finish(session), "disconnected")
        self.assertEqual(session.turn_result["error"]["message"], "upstream_connection_lost: ProcessError")
        self.assertEqual(self.disconnects, ["disconnected"])
        lost, = self.events("agent.connection_lost")
        self.assertEqual(lost["payload"]["reason"], "ProcessError")
        # The native Bash row that never got a result is closed as failed.
        self.assertEqual(self.store.transcript_item("toolu_b")["status"], "failed")

    def test_cli_exit_while_idle_disconnects_and_keeps_the_previous_result(self):
        session = self.started()
        self.script(plain_turn())
        session.run_turn("go")
        self.finish(session)
        kept = session.turn_result
        self.client.end()
        self.assertTrue(wait_until(lambda: session.state == "disconnected"))
        self.assertIs(session.turn_result, kept)
        self.assertEqual(self.disconnects, ["disconnected"])
        with self.assertRaisesRegex(ValueError, "^session_not_connected$"):
            session.run_turn("again")

    def test_relay_denial_and_provider_failure_fail_the_turn_without_disconnect(self):
        session = self.started()
        for status, expected in ((403, "claude_code_turn_failed:api_error:403"), (None, "claude_code_turn_failed:api_error")):
            def denial(sid, text, status=status):
                return [*opening(sid, text),
                        assistant_message("Failed to authenticate. API Error: 403 auriga_model_budget_exhausted",
                                          session_id=sid, error="authentication_failed"),
                        result_message(is_error=True, subtype="success", terminal_reason="api_error",
                                       api_error_status=status, session_id=sid),
                        state_message("idle", sid)]
            self.script(denial)
            session.run_turn("go")
            self.assertEqual(self.finish(session), "failed")
            self.assertEqual(session.turn_result["error"], {"message": expected})
            self.assertIsNone(session.completed_final_answer())
        self.assertEqual(self.disconnects, [])
        # Provider prose never reaches the turn result; the assistant error is an error row, not an answer.
        error_rows = [r for r in self.store.states(self.store.TRANSCRIPT_KIND) if r["type"] == "error"]
        self.assertTrue(error_rows)

    def test_several_results_in_one_turn_end_once_at_idle(self):
        session = self.started()
        sid_turn = []

        def first(sid, text):
            sid_turn.append(sid)
            return [*opening(sid, text), assistant_message("answer one", session_id=sid),
                    result_message(result="answer one", session_id=sid)]
        self.script(first)
        session.run_turn("go")
        time.sleep(0.05)
        self.assertEqual(session.state, "running")  # a result alone never ends the turn
        sid = sid_turn[0]
        session.steer("also report the time")
        self.client.emit(init_message(sid), echo_message("also report the time"),
                         assistant_message("answer two", session_id=sid), result_message(session_id=sid),
                         state_message("idle", sid))
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "answer two")
        self.assertEqual(len(self.events("agent.turn_completed")), 1)
        steered, = self.events("agent.turn_steered")
        self.assertEqual(steered["correlation"]["turn_id"], session.turn_id)
        self.assertEqual(self.events("agent.turn_end_fallback"), [])

    def test_steer_records_control_and_requires_a_running_turn(self):
        session = self.started()
        with self.assertRaisesRegex(ValueError, "^no_active_turn_to_steer$"):
            session.steer("too early")
        self.script(opening, None)
        turn = session.run_turn("go")
        self.assertEqual(session.steer("STEER-1"), turn)
        self.assertEqual(self.client.queries()[-1], "STEER-1")
        commands = [record["message"]["command"] for record in self.raw("control")]
        self.assertEqual(commands, ["submit_prompt", "steer"])
        row, = [r for r in self.store.states(self.store.TRANSCRIPT_KIND) if r.get("steer")]
        self.assertEqual((row["type"], row["turn_id"]), ("userMessage", turn))
        self.client.query_error = CLIConnectionError("gone")
        with self.assertRaisesRegex(AgentTransportUnknown, "claude_code_steer_unknown"):
            session.steer("STEER-2")
        self.assertEqual((session.state, self.disconnects), ("disconnected", ["disconnected"]))

    def test_idle_without_result_is_a_failed_turn(self):
        session = self.started()
        self.script(lambda sid, text: [*opening(sid, text), state_message("idle", sid)])
        session.run_turn("go")
        self.assertEqual(self.finish(session), "failed")
        self.assertEqual(session.turn_result["error"]["message"], "claude_code_turn_failed:idle_without_result")

    def test_quiet_guard_finishes_a_turn_whose_idle_never_comes(self):
        session = self.started(result_quiet_guard_s=0.2)
        self.script(lambda sid, text: [*opening(sid, text), assistant_message("done", session_id=sid),
                                       result_message(session_id=sid)])
        session.run_turn("go")
        self.assertEqual(self.finish(session, 3), "completed")
        fallback, = self.events("agent.turn_end_fallback")
        self.assertEqual(fallback["payload"]["signal"], "result_quiet_fallback")

    def test_quiet_guard_is_not_armed_while_a_native_task_runs(self):
        session = self.started(result_quiet_guard_s=0.1)
        self.script(lambda sid, text: [*opening(sid, text),
                                       system_message("task_started", sid, task_id="t1", task_type="local_agent"),
                                       assistant_message("done", session_id=sid), result_message(session_id=sid)])
        session.run_turn("go")
        time.sleep(0.4)
        self.assertEqual(session.state, "running")
        self.client.emit(system_message("task_notification", session.thread_id, task_id="t1", status="completed"),
                         state_message("idle", session.thread_id))
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(self.events("agent.turn_end_fallback"), [])

    def test_run_turn_store_failure_keeps_the_previous_turn(self):
        # Review (b)1: the turn record is written before any state changes, so a refused record loses nothing.
        session = self.started()
        self.script(plain_turn())
        first = session.run_turn("go")
        self.finish(session)
        kept = session.turn_result
        original = session.transcript.event

        def refuse(event_type, *args, **kwargs):
            if event_type == "agent.turn_started":
                raise RuntimeError("database is locked")
            return original(event_type, *args, **kwargs)
        session.transcript.event = refuse
        with self.assertRaises(AgentRejected) as raised:
            session.run_turn("again")
        self.assertEqual(raised.exception.code, "turn_not_recorded")
        self.assertEqual((session.state, session.turn_id, session.turn_result, session.transcript.turn_id),
                         ("completed", first, kept, first))
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(self.client.queries(), ["go"])
        session.transcript.event = original
        self.script(plain_turn("second"))
        session.run_turn("again")
        self.assertEqual(self.finish(session), "completed")

    def test_init_with_another_session_id_disconnects(self):
        session = self.started()
        self.script(lambda sid, text: [state_message("running", sid), init_message(str(uuid.uuid4()))])
        session.run_turn("go")
        self.assertEqual(self.finish(session), "disconnected")
        self.assertEqual(session.turn_result["error"]["message"], "upstream_connection_lost: RuntimeError")

    def test_handler_errors_are_recorded_and_the_reader_continues(self):
        session = self.started()
        self.script(plain_turn())
        with mock.patch.object(session.projector, "project", side_effect=RuntimeError("projector bug")):
            session.run_turn("go")
            self.assertEqual(self.finish(session), "completed")
        self.assertTrue(self.events("agent.native_handler_error"))
        self.assertEqual(self.disconnects, [])


class InterruptTests(SessionCase):
    def interrupted_by(self, messages_after_interrupt):
        session = self.started()
        self.script(opening)
        self.fake.on_interrupt.append(lambda client: client.emit(*messages_after_interrupt(session.thread_id)))
        cancelled = []
        original = session.model_proxy.cancel_pending
        session.model_proxy.cancel_pending = lambda reason: (cancelled.append(reason), original(reason))
        turn = session.run_turn("go")
        session.interrupt()
        return session, turn, cancelled

    def test_interrupt_during_a_model_call(self):
        session, turn, cancelled = self.interrupted_by(lambda sid: [
            echo_message("[Request interrupted by user]"),
            result_message(is_error=True, subtype="error_during_execution", terminal_reason="aborted_streaming",
                           stop_reason=None, session_id=sid),
            state_message("idle", sid)])
        self.assertEqual(self.finish(session), "interrupted")
        self.assertEqual(session.turn_result, dict(id=turn, status="interrupted", error=None))
        self.assertEqual(cancelled, ["turn_interrupted"])
        commands = [(r["message"]["command"], r["message"]["turn_id"]) for r in self.raw("control")]
        self.assertEqual(commands, [("submit_prompt", turn), ("interrupt", turn)])
        self.assertEqual(self.client.call_names().count("interrupt"), 1)
        self.assertEqual(self.disconnects, [])

    def test_interrupt_during_a_tool_call_reports_the_tool_reply_uncertain(self):
        self.tool_gate = threading.Event()
        self.addCleanup(self.tool_gate.set)
        session = self.started()
        self.script(opening)
        session.run_turn("go")
        mcp = self.mcp(session)
        pending = mcp.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_slow"})
        self.assertTrue(wait_until(lambda: self.handler_calls))
        sid = session.thread_id

        def on_interrupt(client):
            mcp.cancel(pending.request_id, "AbortError: remote-cancel")
            client.emit(tool_result_message("toolu_slow", "The user doesn't want to proceed", is_error=True),
                        echo_message("[Request interrupted by user for tool use]"),
                        result_message(is_error=True, subtype="error_during_execution", terminal_reason="aborted_tools",
                                       session_id=sid),
                        state_message("idle", sid))
        self.fake.on_interrupt.append(on_interrupt)
        session.interrupt()
        self.assertEqual(self.finish(session), "interrupted")
        self.assertTrue(wait_until(lambda: self.resets))
        self.assertEqual(self.store.read_state("runtime", "image_context_epoch")["source"],
                         "tool_reply_delivery_uncertain")
        self.assertEqual(self.resets, ["tool_reply_delivery_uncertain"])
        self.tool_gate.set()

    def test_late_relay_failure_after_an_interrupt_counts_as_interrupted(self):
        # T0 int-relay-cancel-late: the cancelled relay call surfaces as an api_error result before idle.
        session, _, _ = self.interrupted_by(lambda sid: [
            assistant_message("API Error: auriga_model_request_failed", session_id=sid, error="unknown"),
            result_message(is_error=True, subtype="success", terminal_reason="api_error", session_id=sid),
            state_message("idle", sid)])
        self.assertEqual(self.finish(session), "interrupted")

    def test_interrupt_control_timeout_disconnects(self):
        session = self.started(interrupt_timeout_s=0.2)
        self.script(opening)
        session.run_turn("go")
        self.client.hold("interrupt")
        with self.assertRaisesRegex(AgentTransportUnknown, "claude_code_interrupt_timeout"):
            session.interrupt()
        self.client.release("interrupt")
        self.assertEqual(session.state, "disconnected")
        self.assertEqual(session.turn_result["status"], "failed")
        self.assertEqual(self.disconnects, ["disconnected"])

    def test_interrupt_control_error_is_raised_only_while_the_turn_runs(self):
        session = self.started()
        self.script(opening)
        session.run_turn("go")
        self.client.interrupt_error = RuntimeError("control failed")
        with self.assertRaisesRegex(RuntimeError, "control failed"):
            session.interrupt()
        self.assertEqual(session.state, "running")

        def error_after_turn_end(index):
            # The turn ends (native terminal) while the control request is in flight; then the control fails.
            session._finish_turn("interrupted")
            return RuntimeError("turn already over")
        self.client.interrupt_error = error_after_turn_end
        session.interrupt()  # swallowed: the turn it targeted is over
        self.assertEqual(self.finish(session), "interrupted")
        self.assertEqual(len(self.events("agent.interrupt_failed")), 2)

    def test_interrupt_while_idle_sends_nothing(self):
        session = self.started()
        session.interrupt()
        self.assertNotIn("interrupt", self.client.call_names())
        self.assertEqual(self.raw("control"), [])


class ModelAndEffortTests(SessionCase):
    def test_set_model_applies_before_the_next_prompt(self):
        session = self.started()
        session.set_model("claude-other")
        self.assertNotIn("set_model", self.client.call_names())  # never blocks the caller
        self.script(plain_turn())
        session.run_turn("go")
        self.finish(session)
        names = self.client.call_names()
        self.assertLess(names.index("set_model"), names.index("query"))
        self.assertEqual(session.active_turn_model, "claude-other")
        changed, = self.events("agent.model_changed")
        self.assertEqual(changed["payload"], {"model": "claude-other", "previous": "claude-test-model"})
        with self.assertRaisesRegex(ValueError, "^invalid_model$"):
            session.set_model("")

    def test_each_turn_reopens_the_relay_thread_before_anything_reaches_the_relay(self):
        session = self.started()
        self.assertIs(session.model_proxy.on_context_discontinuity.__func__,
                      ClaudeCodeSession._relay_discontinuity)
        order = []
        original = session.model_proxy.reopen_thread
        session.model_proxy.reopen_thread = lambda thread: (order.append(("reopen", thread)), original(thread))[1]
        self.fake.on_query.insert(0, lambda client, text: order.append(("query", text)))
        session.set_model("claude-other")
        original_set_model = self.client.set_model

        async def set_model(model=None):
            order.append(("set_model", model))
            await original_set_model(model)
        self.client.set_model = set_model
        self.script(plain_turn())
        session.run_turn("go")
        self.finish(session)
        self.assertEqual(order, [("reopen", session.thread_id), ("set_model", "claude-other"), ("query", "go")])

    def test_rejected_model_change_keeps_the_previous_model(self):
        session = self.started()
        self.client.set_model_error = Exception("API error: 403 auriga_model_budget_exhausted · model not changed")
        session.set_model("claude-other")
        self.script(plain_turn())
        session.run_turn("go")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual((session.model, session.active_turn_model), ("claude-test-model", "claude-test-model"))
        self.assertEqual(self.events("agent.model_change_rejected")[0]["payload"]["model"], "claude-other")
        self.assertEqual(self.disconnects, [])

    def test_set_model_timeout_disconnects(self):
        session = self.started(set_model_timeout_s=0.2)
        self.client.hold("set_model")
        session.set_model("claude-other")
        with self.assertRaisesRegex(AgentTransportUnknown, "claude_code_set_model_timeout"):
            session.run_turn("go")
        self.client.release("set_model")
        self.assertEqual((session.state, self.disconnects), ("disconnected", ["disconnected"]))
        self.assertEqual(self.client.queries(), [])

    def test_effort_applies_at_the_next_session_start(self):
        session = self.started()
        session.set_effort("max")
        self.assertEqual(session.effort, "max")
        with self.assertRaisesRegex(ValueError, "^invalid_effort$"):
            session.set_effort("ultra")
        self.assertEqual(session.effort_choices, ("low", "medium", "high", "xhigh", "max"))


class CompactionTests(SessionCase):
    def test_native_compaction_signals_change_the_epoch(self):
        session = self.started()
        self.script(opening)
        session.run_turn("go")
        sid = session.thread_id
        self.assertEqual(session._call(session._pre_compact({"hook_event_name": "PreCompact"}, None, None), 2), {})
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual((epoch["source"], epoch["signal"]), ("native_context_compaction", "pre_compact"))
        self.client.emit(system_message("status", sid, status="compacting"),
                         system_message("compact_boundary", sid, compact_metadata={"trigger": "auto",
                                                                                   "pre_tokens": 190000}))
        self.assertTrue(wait_until(lambda: len(self.events("agent.compaction")) == 3))
        signals = [event["payload"]["signal"] for event in self.events("agent.compaction")]
        self.assertEqual(signals, ["pre_compact", "compacting", "compact_boundary"])
        self.assertEqual(self.resets, ["native_context_compaction"] * 3)
        rows = [r for r in self.store.states(self.store.TRANSCRIPT_KIND) if r["type"] == "contextCompaction"]
        self.assertEqual(len(rows), 1)
        self.client.emit(*ending(sid))
        self.assertEqual(self.finish(session), "completed")


class CloseTests(SessionCase):
    def test_close_disconnects_stops_the_loop_and_records_a_clean_shutdown(self):
        session = self.started()
        self.script(plain_turn())
        session.run_turn("go")
        self.finish(session)
        loop_thread = session._loop_thread
        report = session.close()
        self.assertIn("disconnect", self.client.call_names())
        self.assertFalse(loop_thread.is_alive())
        self.assertEqual(report["backend"]["disconnect"], "completed")
        self.assertTrue(report["clean_shutdown"])
        self.assertEqual(report["steps"], {"relay_cancel": "completed", "backend": "completed", "relay": "completed",
                                           "mcp": "completed", "tool_workers": "completed", "identity": "completed",
                                           "event": "completed"})
        self.assertTrue(self.store.read_state("runtime", "agent")["clean_shutdown"])
        self.assertEqual((session.state, session.turn_done.is_set()), ("closed", True))
        self.assertTrue(session.model_proxy.closed)
        self.assertIs(session.close(), report)
        self.assertEqual(self.disconnects, [])
        # A confirmed-gone process is no longer owned and is never signalled again (pid reuse).
        self.assertEqual(session.owned_processes(), [])
        with mock.patch("os.kill") as kill:
            session._process.terminate()
        kill.assert_not_called()

    def test_close_during_a_turn_is_unclean_and_never_reports_a_disconnect(self):
        session = self.started()
        self.script(opening)
        session.run_turn("go")
        report = session.close()
        self.assertFalse(report["clean_shutdown"])
        self.assertFalse(self.store.read_state("runtime", "agent")["clean_shutdown"])
        self.assertEqual(self.disconnects, [])
        self.assertTrue(session.turn_done.is_set())

    def test_close_survives_a_hung_disconnect(self):
        session = self.started(disconnect_timeout_s=0.2)
        self.client.hold("disconnect")
        report = session.close()
        self.client.release("disconnect")
        self.assertEqual(report["backend"]["disconnect"], "failed:TimeoutError")
        self.assertEqual(session.state, "closed")

    def test_close_before_start_and_in_replay(self):
        session = self.make()
        session.enter_replay("recorded-thread")
        self.assertEqual(session.state, "replay")
        with self.assertRaisesRegex(ValueError, "^session_not_connected$"):
            session.run_turn("x")
        session.close()
        self.assertEqual(session.state, "closed")
        self.assertIsNone(self.store.read_state("runtime", "agent"))


class ResumeTests(SessionCase):
    def recorded(self, clean=True):
        first = self.started()
        thread = first.thread_id
        first.close()
        (self.root / "private" / "claude-config" / "projects" / "-run-actor").mkdir(parents=True, exist_ok=True)
        (self.root / "private" / "claude-config" / "projects" / "-run-actor" / f"{thread}.jsonl").write_text("{}\n")
        if not clean:
            identity = self.store.read_state("runtime", "agent")
            self.store.set_state("runtime", "agent", dict(identity, clean_shutdown=False))
        self.fake.clients.clear()
        return thread

    def test_clean_resume_uses_the_recorded_session(self):
        thread = self.recorded()
        session = self.started(resume=thread)
        self.assertEqual(session.thread_id, thread)
        self.assertEqual((self.client.options.resume, self.client.options.session_id), (thread, None))
        recovered, = self.events("agent.history_recovered")
        self.assertEqual(recovered["payload"]["clean"], True)
        self.assertEqual(self.resets, [])
        self.assertTrue(self.store.read_state("runtime", "agent")["resumed"])

    def test_unclean_resume_invalidates_the_context(self):
        thread = self.recorded(clean=False)
        self.started(resume=thread)
        self.assertEqual(self.store.read_state("runtime", "image_context_epoch")["source"],
                         "agent_resume_after_unclean_exit")
        self.assertEqual(self.resets, ["agent_resume_after_unclean_exit"])
        self.assertEqual(self.events("agent.history_recovered")[0]["payload"]["clean"], False)

    def test_resume_identity_checks(self):
        thread = self.recorded()
        identity = self.store.read_state("runtime", "agent")
        for key, value in (("model", "other-model"), ("tools_digest", "x"), ("backend", "opencode"),
                           ("cli_version", "2.1.0"), ("counting", "other")):
            with self.subTest(key):
                self.store.set_state("runtime", "agent", dict(identity, **{key: value}))
                session = self.make()
                with self.assertRaisesRegex(ValueError, f"^resume_agent_identity_conflict:{key}"):
                    session.start(TOOLS, "x", thread)
                self.assertEqual(session.state, "closed")
        self.store.set_state("runtime", "agent", identity)
        session = self.make()
        with self.assertRaisesRegex(ValueError, "^resume_agent_identity_conflict:thread_id"):
            session.start(TOOLS, "x", str(uuid.uuid4()))
        for path in (self.root / "private" / "claude-config" / "projects").rglob("*.jsonl"):
            path.unlink()
        session = self.make()
        with self.assertRaisesRegex(ValueError, "^resume_native_session_missing"):
            session.start(TOOLS, "x", thread)


class CapturedSequenceTests(SessionCase):
    """Replay the real CLI message sequences captured in spikes S3 and T0; the quiet guard must never fire."""

    CASES = {
        "t0-int-model-gated.jsonl": ("interrupted", "completed"),
        "t0-int-model-streaming.jsonl": ("interrupted", "completed"),
        "t0-int-tool.jsonl": ("interrupted", "completed"),
        "t0-err-403.jsonl": ("failed", "completed"),
        "t0-err-529.jsonl": ("failed", "completed"),
        "t0-err-truncated.jsonl": ("failed", "completed"),
        "s3-final-gated-state-replay.jsonl": ("completed",),
        "s3-during-tool-state-replay.jsonl": ("completed",),
    }
    ERRORS = {"t0-err-403.jsonl": "claude_code_turn_failed:api_error:403",
              "t0-err-529.jsonl": "claude_code_turn_failed:api_error:529",
              "t0-err-truncated.jsonl": "claude_code_turn_failed:api_error"}

    @staticmethod
    def segments(messages):
        """Split a capture into native busy periods, each starting at a ``running`` state frame."""
        out = []
        for message in messages:
            if (type(message).__name__ == "SystemMessage" and message.subtype == "session_state_changed"
                    and message.data.get("state") == "running"):
                out.append([])
            if out:
                out[-1].append(message)
        return out

    def test_captured_sequences_end_turns_at_idle(self):
        for name, statuses in self.CASES.items():
            with self.subTest(name):
                self.setUp()
                session = self.started(result_quiet_guard_s=0.1)
                segments = self.segments(load_capture(FIXTURES / name))
                self.assertEqual(len(segments), len(statuses))
                for segment, expected in zip(segments, statuses):
                    for message in segment:
                        if type(message).__name__ == "SystemMessage" and "session_id" in message.data:
                            message.data["session_id"] = session.thread_id
                    self.script(segment)
                    # The prompt the capture answered (its --replay-user-messages echo).
                    prompt = next(m.content for m in segment if type(m).__name__ == "UserMessage"
                                  and isinstance(m.content, str))
                    session.run_turn(prompt)
                    self.assertEqual(self.finish(session), expected)
                    if expected == "failed":
                        self.assertEqual(session.turn_result["error"]["message"], self.ERRORS[name])
                time.sleep(0.15)
                self.assertEqual(self.events("agent.turn_end_fallback"), [])
                self.assertEqual(self.disconnects, [])
                session.close()
                self.doCleanups()


class AdoptionTests(SessionCase):
    """Design R4.2: a native turn that starts while the host turn is terminal or idle becomes an adopted Auriga turn."""

    def completed_first_turn(self, **overrides):
        session = self.started(**overrides)
        self.script(plain_turn("first"))
        first = session.run_turn("go")
        self.assertEqual(self.finish(session), "completed")
        return session, first

    def test_capability_flag(self):
        from auriga.cortex.agents.base import AgentCapabilities
        self.assertIs(AgentCapabilities().native_wakeup, False)
        self.assertIs(ClaudeCodeSession.capabilities.native_wakeup, True)

    def test_background_wakeup_opens_an_adopted_turn_that_owns_its_tool_calls(self):
        session, first = self.completed_first_turn()
        sid = session.thread_id
        self.client.emit(system_message("task_notification", sid, task_id="b1", status="completed"),
                         state_message("running", sid))
        self.assertTrue(wait_until(lambda: session.state == "running"))
        adopted = session.turn_id
        self.assertNotEqual(adopted, first)
        self.assertFalse(session.turn_done.is_set())
        started = self.events("agent.turn_started")
        self.assertEqual([event["payload"]["origin"] for event in started], ["host", "native_wakeup"])
        self.assertEqual(started[-1]["correlation"]["turn_id"], adopted)
        self.client.emit(assistant_message(tool_use("toolu_w", "mcp__auriga__look"), session_id=sid))
        call = self.mcp(session).call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_w"})
        self.assertFalse(call.result["isError"])
        self.assertEqual(self.handler_calls[-1]["turnId"], adopted)
        self.client.emit(tool_result_message("toolu_w", "ok look"), *ending(sid, answer="noticed"))
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.turn_result, dict(id=adopted, status="completed", error=None))
        self.assertEqual(session.completed_final_answer(), "noticed")
        self.assertEqual(self.events("agent.unsolicited_native_turn"), [])
        self.assertEqual(self.disconnects, [])
        # A later host turn is a normal host turn again.
        self.script(plain_turn("third"))
        self.assertNotEqual(session.run_turn("again"), adopted)
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "third")

    def test_captured_background_bash_wakeup(self):
        # T0 bg-bash: Result, idle; the background shell ends; the CLI starts a turn of its own (Result origin
        # task-notification), then the follow-up host prompt.
        session = self.started(result_quiet_guard_s=0.1)
        busy = CapturedSequenceTests.segments(load_capture(FIXTURES / "t0-bg-bash.jsonl"))
        self.assertEqual(len(busy), 3)
        for message in (m for segment in busy for m in segment):
            if type(message).__name__ == "SystemMessage" and "session_id" in message.data:
                message.data["session_id"] = session.thread_id
        tail = load_capture(FIXTURES / "t0-bg-bash.jsonl")
        between = [m for m in tail if type(m).__name__ == "SystemMessage" and m.subtype in
                   ("task_updated", "task_notification", "background_tasks_changed")][-3:]
        self.script(busy[0])
        first = session.run_turn("Start the scripted task.")
        self.assertEqual(self.finish(session), "completed")
        self.client.emit(*between, *busy[1])
        self.assertTrue(wait_until(lambda: session.turn_id != first and session.turn_done.is_set()))
        self.assertEqual(session.state, "completed")
        self.assertEqual(session.completed_final_answer(), "noticed background completion")
        self.script(busy[2])
        session.run_turn("FOLLOWUP turn after the probe.")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual([event["payload"]["origin"] for event in self.events("agent.turn_started")],
                         ["host", "native_wakeup", "host"])
        self.assertEqual([event["payload"]["status"] for event in self.events("agent.turn_completed")],
                         ["completed"] * 3)
        time.sleep(0.15)
        self.assertEqual(self.events("agent.turn_end_fallback"), [])

    def test_run_turn_joins_an_adopted_turn(self):
        session, first = self.completed_first_turn()
        sid = session.thread_id
        self.client.emit(state_message("running", sid), init_message(sid))
        self.assertTrue(wait_until(lambda: session.state == "running"))
        adopted = session.turn_id
        self.script(None)
        self.assertEqual(session.run_turn("operator prompt"), adopted)
        submit = self.raw("control")[-1]["message"]
        self.assertEqual((submit["command"], submit["turn_id"]), ("submit_prompt", adopted))
        joined, = self.events("agent.turn_joined")
        self.assertEqual(joined["payload"]["origin"], "native_wakeup")
        # The wake-up turn ends before the CLI read the prompt; the Auriga turn waits for it (one turn_done).
        self.client.emit(assistant_message("background", session_id=sid), result_message(session_id=sid),
                         state_message("idle", sid))
        self.assertTrue(wait_until(lambda: self.events("agent.turn_end_deferred")))
        self.assertEqual(session.state, "running")
        self.assertFalse(session.turn_done.is_set())
        self.client.emit(state_message("running", sid), echo_message("operator prompt"),
                         *ending(sid, answer="answered the prompt"))
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.turn_result["id"], adopted)
        self.assertEqual(session.completed_final_answer(), "answered the prompt")
        self.assertEqual(len(self.events("agent.turn_completed")), 2)
        # A second prompt while the joined (now host-prompted) turn runs is refused like any host turn.
        self.client.emit(state_message("running", sid))
        self.assertTrue(wait_until(lambda: session.state == "running"))
        self.script(None)
        session.run_turn("first prompt of the new wake-up")
        with self.assertRaisesRegex(ValueError, "^turn_already_active$"):
            session.run_turn("second prompt")

    def test_host_turn_still_refuses_a_second_prompt(self):
        session = self.started()
        self.script(opening)
        session.run_turn("go")
        with self.assertRaisesRegex(ValueError, "^turn_already_active$"):
            session.run_turn("again")

    def test_interrupt_reaches_an_adopted_turn(self):
        session, _ = self.completed_first_turn()
        sid = session.thread_id
        self.fake.on_interrupt.append(lambda client: client.emit(
            result_message(is_error=True, subtype="error_during_execution", terminal_reason="aborted_streaming",
                           session_id=sid), state_message("idle", sid)))
        self.client.emit(state_message("running", sid), init_message(sid))
        self.assertTrue(wait_until(lambda: session.state == "running"))
        adopted = session.turn_id
        session.interrupt()
        self.assertEqual(self.finish(session), "interrupted")
        self.assertEqual(self.raw("control")[-1]["message"], dict(command="interrupt", thread_id=sid,
                                                                  turn_id=adopted))

    def test_close_during_an_adopted_turn_is_unclean(self):
        session, _ = self.completed_first_turn()
        self.client.emit(state_message("running", session.thread_id))
        self.assertTrue(wait_until(lambda: session.state == "running"))
        report = session.close()
        self.assertFalse(report["clean_shutdown"])
        self.assertTrue(report["native_active_at_close"])
        self.assertFalse(self.store.read_state("runtime", "agent")["clean_shutdown"])

    def test_native_busy_without_adoption_still_makes_close_unclean_and_interrupt_reach_it(self):
        # A wake-up whose adoption the Store refused: the CLI runs, the host state stays terminal.
        session, first = self.completed_first_turn()
        original = session.transcript.event

        def refuse(event_type, *args, **kwargs):
            if event_type == "agent.turn_started":
                raise RuntimeError("database is locked")
            return original(event_type, *args, **kwargs)
        session.transcript.event = refuse
        self.client.emit(state_message("running", session.thread_id))
        self.assertTrue(wait_until(lambda: session._native_state == "running"))
        self.assertEqual((session.state, session.turn_id), ("completed", first))
        session.transcript.event = original
        result = self.mcp(session).call_tool("move", {}, meta={"claudecode/toolUseId": "toolu_orphan"})
        self.assertIn("native_tool_call_stale_connection", result.result["content"][0]["text"])
        session.interrupt()
        self.assertEqual(self.raw("control")[-1]["message"]["command"], "interrupt")
        self.assertIsNone(self.raw("control")[-1]["message"]["turn_id"])
        self.assertFalse(session.close()["clean_shutdown"])

    def test_no_adoption_after_disconnect_or_close(self):
        session, _ = self.completed_first_turn()
        session.closing = True
        session._on_state("running")
        self.assertEqual(session.state, "completed")
        self.assertEqual(len(self.events("agent.unsolicited_native_turn")), 1)
        session.closing = False


class LateWriteTests(SessionCase):
    """Review A1: a stdin write that lands after its turn ended, tool calls outside a running turn."""

    def test_tool_calls_are_refused_outside_a_running_turn(self):
        session = self.started()
        self.script(plain_turn())
        turn = session.run_turn("go")
        self.finish(session)
        result = self.mcp(session).call_tool("move", {"x": 1}, meta={"claudecode/toolUseId": "toolu_late"})
        self.assertTrue(result.result["isError"])
        self.assertIn("native_tool_call_stale_connection", result.result["content"][0]["text"])
        self.assertEqual(self.handler_calls, [])
        self.assertIsNone(self.store.transcript_item("toolu_late"))
        self.assertEqual(session.turn_id, turn)

    def test_steer_written_as_the_model_finishes_keeps_the_turn_open(self):
        session = self.started()
        self.script(opening, None)
        turn = session.run_turn("go")
        sid = session.thread_id
        self.client.hold("query")
        out = {}
        worker = threading.Thread(target=lambda: out.setdefault("turn", session.steer("STEER-LATE")))
        worker.start()
        self.assertTrue(self.client.wait_for_call("query", 2))
        self.client.emit(*ending(sid, answer="first answer"))     # result + idle before the steer is read
        self.assertTrue(wait_until(lambda: self.events("agent.turn_end_deferred")))
        self.assertEqual(session.state, "running")
        self.client.release("query")
        worker.join(5)
        self.assertEqual(out["turn"], turn)
        # The CLI runs the steer as a further native turn inside the same Auriga turn.
        self.client.emit(state_message("running", sid), init_message(sid), echo_message("STEER-LATE"))
        self.client.emit(assistant_message(tool_use("toolu_s", "mcp__auriga__look"), session_id=sid))
        self.mcp(session).call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_s"})
        self.assertEqual(self.handler_calls[-1]["turnId"], turn)
        self.client.emit(tool_result_message("toolu_s", "ok"), *ending(sid, answer="after steer"))
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.turn_result["id"], turn)
        self.assertEqual(session.completed_final_answer(), "after steer")
        self.assertEqual(len(self.events("agent.turn_started")), 1)
        self.assertEqual(self.events("agent.steer_not_delivered"), [])

    def test_steer_outliving_an_interrupt_is_reported_and_adopted_later(self):
        session = self.started()
        self.script(opening, None)
        turn = session.run_turn("go")
        sid = session.thread_id
        self.client.hold("query")
        out = {}
        worker = threading.Thread(target=lambda: out.setdefault("turn", session.steer("STEER-AFTER-INT")))
        worker.start()
        self.assertTrue(self.client.wait_for_call("query", 2))
        self.client.emit(result_message(is_error=True, subtype="error_during_execution",
                                        terminal_reason="aborted_streaming", session_id=sid),
                         state_message("idle", sid))
        self.assertEqual(self.finish(session), "interrupted")       # an interrupted turn never waits for input
        self.client.release("query")
        worker.join(5)
        self.assertEqual(out["turn"], turn)
        missed, = self.events("agent.steer_not_delivered")
        self.assertEqual((missed["correlation"]["turn_id"], missed["payload"]["delivered_turn_id"]), (turn, None))
        # The CLI later runs the queued steer by itself: adopted, and the steer row moves to that turn.
        self.client.emit(state_message("running", sid), init_message(sid), echo_message("STEER-AFTER-INT"),
                         *ending(sid, answer="did the steer"))
        self.assertTrue(wait_until(lambda: session.turn_id != turn and session.turn_done.is_set()))
        late, = self.events("agent.input_delivered_late")
        self.assertEqual((late["payload"]["kind"], late["payload"]["written_turn_id"]), ("steer", turn))
        row, = [r for r in self.store.states(self.store.TRANSCRIPT_KIND) if r.get("steer")]
        self.assertEqual(row["turn_id"], session.turn_id)

    def test_a_repeated_text_after_an_unechoed_input_gets_its_own_echo(self):
        # An interrupted turn leaves its unconsumed steer behind; the same text written later must not be matched
        # to that old row (identical continuation prompts are common).
        session = self.started()
        self.script(opening, None)
        session.run_turn("go")
        sid = session.thread_id
        session.steer("Continue.")
        self.client.emit(result_message(is_error=True, subtype="error_during_execution",
                                        terminal_reason="aborted_streaming", session_id=sid),
                         state_message("idle", sid))
        self.assertEqual(self.finish(session), "interrupted")
        self.script(lambda s, text: [state_message("running", s), echo_message(text), *ending(s, answer="again")])
        turn = session.run_turn("Continue.")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.turn_result["id"], turn)
        self.assertEqual(self.events("agent.turn_end_fallback"), [])
        self.assertEqual(self.events("agent.input_delivered_late"), [])

    def test_guard_ends_a_turn_whose_input_echo_never_comes(self):
        session = self.started(result_quiet_guard_s=0.2)
        self.script(opening, None)
        session.run_turn("go")
        sid = session.thread_id
        session.steer("never echoed")
        self.client.emit(*ending(sid))
        self.assertEqual(self.finish(session, 3), "completed")
        fallback, = self.events("agent.turn_end_fallback")
        self.assertEqual((fallback["payload"]["signal"], fallback["payload"]["unconsumed_inputs"]),
                         ("input_echo_missing", 1))

    def test_prompt_written_while_a_wakeup_ends_is_not_ended_by_its_result(self):
        # Review (b)6: the wake-up's Result and idle already in the pipe must not end the prompted turn early.
        session = self.started()
        self.script(plain_turn("first"))
        session.run_turn("go")
        self.finish(session)
        sid = session.thread_id
        self.client.emit(state_message("running", sid))
        self.assertTrue(wait_until(lambda: session.state == "running"))
        adopted = session.turn_id
        self.script(None)
        self.assertEqual(session.run_turn("REAL PROMPT"), adopted)
        self.client.emit(assistant_message("bg", session_id=sid), result_message(result="bg", session_id=sid),
                         state_message("idle", sid))
        time.sleep(0.1)
        self.assertFalse(session.turn_done.is_set())
        self.client.emit(state_message("running", sid), echo_message("REAL PROMPT"), *ending(sid, answer="real"))
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "real")


class InterruptOrderingTests(SessionCase):
    """Review (b)2: an interrupt racing the prompt submission."""

    def test_interrupt_before_the_prompt_write_cancels_the_submission(self):
        session = self.started()
        session.set_model("claude-other")
        self.client.hold("set_model")
        out = {}

        def submit():
            try:
                out["turn"] = session.run_turn("go")
            except Exception as error:  # noqa: BLE001
                out["error"] = error
        worker = threading.Thread(target=submit)
        worker.start()
        self.assertTrue(self.client.wait_for_call("set_model", 1))
        kept = (session.state, session.turn_id)
        session.interrupt()
        self.assertNotIn("interrupt", self.client.call_names())     # an idle CLI with no turn: nothing to stop
        self.client.release("set_model")
        worker.join(5)
        self.assertIsInstance(out.get("error"), AgentRejected)
        self.assertEqual(out["error"].code, "interrupted")
        self.assertEqual(self.client.queries(), [])
        self.assertEqual((session.state, session.turn_id), kept)
        self.assertEqual(self.events("agent.turn_started"), [])
        self.assertEqual(self.events("agent.turn_start_rejected")[0]["payload"]["code"], "interrupted")

    def test_interrupt_during_the_prompt_write_is_sent_after_it(self):
        session = self.started()
        self.client.hold("query")
        worker = threading.Thread(target=lambda: session.run_turn("go"))
        worker.start()
        self.assertTrue(self.client.wait_for_call("query", 1))
        sid = session.thread_id
        self.fake.on_interrupt.append(lambda client: client.emit(
            state_message("running", sid), echo_message("go"),
            result_message(is_error=True, subtype="error_during_execution", terminal_reason="aborted_streaming",
                           session_id=sid), state_message("idle", sid)))
        interrupter = threading.Thread(target=session.interrupt)
        interrupter.start()
        time.sleep(0.05)
        self.assertNotIn("interrupt", self.client.call_names())     # waits for the prompt write
        self.client.release("query")
        worker.join(5)
        interrupter.join(5)
        self.assertEqual(self.client.writes, ["query", "interrupt"])
        self.assertEqual(self.finish(session), "interrupted")


class ProcessTeardownTests(SessionCase):
    """Review (b)3-(b)5: the CLI process is visible during connect, a crash is not a confirmed teardown, the kill
    list is re-read right before the kill."""

    def test_cli_process_is_owned_while_connect_waits(self):
        original = self.fake.factory
        held = []

        def factory(options=None, transport=None):
            client = original(options, transport)
            client.hold("connect")
            held.append(client)
            return client
        session = self.make()
        session._client_factory = factory
        worker = threading.Thread(target=lambda: session.start(TOOLS, "x"))
        worker.start()
        self.assertTrue(wait_until(lambda: held and held[0].wait_for_call("connect", 1)))
        self.assertTrue(wait_until(lambda: session.owned_processes()))
        (role, process), = session.owned_processes()
        self.assertEqual((role, process.pid), ("claude_code", FAKE_PID))
        held[0].release("connect")
        worker.join(5)
        self.assertEqual(session.state, "idle")

    def test_connect_failure_after_spawn_is_verified_by_the_marker_scan(self):
        original, held = self.fake.factory, []

        def factory(options=None, transport=None):
            client = original(options, transport)
            client.hold("connect")
            client.connect_error = CLIConnectionError("initialize failed")
            held.append(client)
            return client
        session = self.make()
        session._client_factory = factory
        errors = []

        def start():
            try:
                session.start(TOOLS, "x")
            except CLIConnectionError as error:
                errors.append(error)
        worker = threading.Thread(target=start)
        worker.start()
        self.assertTrue(wait_until(lambda: session.owned_processes()))   # spawned, initialize pending
        held[0].release("connect")
        worker.join(5)
        self.assertEqual(len(errors), 1)
        self.assertEqual(session.state, "closed")
        backend = session.close_report["backend"]
        self.assertTrue(backend["agent_exited_before_teardown"])
        # The CLI's children would carry the session marker: the re-scan is what confirms nothing is left.
        self.assertTrue(backend["agent_processes"]["marker"]["instance_id_set"])
        self.assertTrue(backend["agent_processes"]["confirmed"])
        self.assertNotIn("agent_process_exit_unconfirmed", backend)
        self.assertFalse(session.close_report["clean_shutdown"])   # a failed start is never a clean boundary

    def test_connect_failure_before_the_process_was_seen_is_verified_by_the_marker_scan(self):
        self.fake.connect_error = CLIConnectionError("initialize failed")
        session = self.make()
        with self.assertRaises(CLIConnectionError):
            session.start(TOOLS, "x")
        backend = session.close_report["backend"]
        self.assertTrue(backend.get("agent_process_unobserved") or backend.get("agent_exited_before_teardown"))
        self.assertTrue(backend["agent_processes"]["confirmed"])
        # An unreadable process table leaves it unverifiable.
        from auriga.cortex.agents import isolation
        self.fake.connect_error = CLIConnectionError("initialize failed")
        session = self.make()
        with mock.patch.object(isolation, "_marker_rows", side_effect=OSError("ps failed")):
            with self.assertRaises(CLIConnectionError):
                session.start(TOOLS, "x")
        backend = session.close_report["backend"]
        self.assertTrue(backend["agent_process_exit_unconfirmed"])
        self.assertEqual(backend["agent_processes"]["scan_errors"][0], "marker_scan:OSError")
        # A factory failure spawns nothing: trivially confirmed.
        self.fake.connect_error, self.fake.factory_error = None, RuntimeError("no client")
        session = self.make()
        with self.assertRaises(RuntimeError):
            session.start(TOOLS, "x")
        self.assertNotIn("agent_process_exit_unconfirmed", session.close_report["backend"])

    def test_crash_teardown_targets_revalidated_members_and_the_marker(self):
        from auriga.cortex.agents import isolation
        from auriga.cortex.agents.isolation import AgentTeardownReport, ProcessInfo, TerminationReport
        session = self.started()
        own_group = os.getpgrp()
        # Two descendants seen while the CLI ran: one detached into its own group, one in the CLI's group.
        session._known_tree = {7001: ProcessInfo(7001, FAKE_PID, 7001), 7002: ProcessInfo(7002, FAKE_PID, FAKE_PID)}
        self.client.kill_process(code=-9)
        self.assertTrue(wait_until(lambda: session.state == "disconnected"))
        table = [ProcessInfo(7001, 1, 7001), ProcessInfo(7005, 1, 7001),   # still alive, its group grew
                 ProcessInfo(7002, 1, 9999),                               # pid reused by another group
                 ProcessInfo(os.getpid(), 1, own_group)]
        captured = []

        def fake_teardown(snapshot, instance_id, *, cwd_roots, **kwargs):
            captured.append((snapshot, instance_id, [os.fspath(root) for root in cwd_roots]))
            termination = TerminationReport(root_pid=snapshot.root_pid, members=tuple(m.pid for m in snapshot.members))
            return AgentTeardownReport(termination=termination, rounds=[termination],
                                       remaining=(7009,) if len(captured) > 1 else ())
        # The shared path (isolation.stop_agent_tree): a fresh table re-validates the known tree, then the marker.
        with mock.patch.object(isolation, "process_table", return_value=table), \
                mock.patch.object(isolation, "terminate_agent_processes", side_effect=fake_teardown):
            report = session.close()
        (target, instance, roots), = captured
        self.assertEqual(sorted(member.pid for member in target.members), [7001, 7005])
        self.assertEqual((instance, roots), (self.store.read_state("runtime", "agent")["instance_id"],
                                             [os.fspath(session.actor_dir)]))
        backend = report["backend"]
        self.assertTrue(backend["agent_exited_before_teardown"])
        self.assertNotIn("agent_process_exit_unconfirmed", backend)   # the marker re-scan verified it
        self.assertFalse(report["clean_shutdown"])                      # a crash is never a clean boundary
        # A marked process that survives the re-scan makes the teardown unconfirmed.
        captured.clear()
        captured.append(None)
        session = self.started()
        with mock.patch.object(isolation, "terminate_agent_processes", side_effect=fake_teardown):
            report = session.close()
        self.assertTrue(report["backend"]["agent_process_exit_unconfirmed"])
        self.assertEqual(report["backend"]["agent_processes"]["remaining"], [7009])
        self.assertFalse(report["clean_shutdown"])

    def test_teardown_snapshot_rules(self):
        # The rule is shared with OpenCode (isolation.teardown_snapshot, through isolation.stop_agent_tree).
        from auriga.cortex.agents.claude_code import session as session_module
        from auriga.cortex.agents.isolation import ProcessInfo, stop_agent_tree, teardown_snapshot as _teardown_snapshot
        self.assertIs(session_module.stop_agent_tree, stop_agent_tree)
        me, own = os.getpid(), os.getpgrp()
        table = [ProcessInfo(100, me, own), ProcessInfo(101, 100, 100), ProcessInfo(102, 1, 102),
                 ProcessInfo(103, 102, 102), ProcessInfo(200, me, own), ProcessInfo(me, 1, own)]
        known = [ProcessInfo(101, 100, 100), ProcessInfo(102, 101, 102), ProcessInfo(200, 100, own),
                 ProcessInfo(300, 100, 300)]
        snapshot = _teardown_snapshot(100, True, known, table)
        self.assertEqual([m.pid for m in snapshot.members], [100, 101, 102, 103])
        # 200: a direct child of the gateway in the gateway's group (pid reuse) is never selected; the root of a
        # dead CLI is not either.
        self.assertEqual([m.pid for m in _teardown_snapshot(100, False, known, table).members], [101, 102, 103])


class FakeDeathTests(SessionCase):
    def test_interrupt_after_cli_death_disconnects_instead_of_hanging(self):
        session = self.started()
        self.script(opening)
        session.run_turn("go")
        self.client.hold("interrupt")
        worker = threading.Thread(target=session.interrupt)
        worker.start()
        self.assertTrue(self.client.wait_for_call("interrupt", 1))
        self.client.kill_process(code=-9)
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(session.state, "disconnected")
        self.assertEqual(self.disconnects, ["disconnected"])
        self.assertTrue(self.events("agent.interrupt_failed"))


class SubagentTests(SessionCase):
    """Design R4.3: subagent output is linked by parent_tool_use_id, its registry calls run in the host turn, it is
    never the final answer; subagents force full context delivery."""

    def test_subagent_rows_and_tool_calls(self):
        session = self.started()
        self.script(opening)
        turn = session.run_turn("go")
        sid = session.thread_id
        self.client.emit(assistant_message(tool_use("toolu_task", "Agent", description="look", prompt="look"),
                                           session_id=sid, message_id="m_main"),
                         system_message("task_started", sid, task_id="t1", task_type="local_agent"),
                         assistant_message("subagent says hi", parent_tool_use_id="toolu_task", session_id=sid),
                         assistant_message(tool_use("toolu_sub", "mcp__auriga__look"), parent_tool_use_id="toolu_task",
                                           session_id=sid))
        call = self.mcp(session).call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_sub"})
        self.assertFalse(call.result["isError"])
        self.assertEqual(self.handler_calls[-1]["turnId"], turn)
        self.client.emit(tool_result_message("toolu_sub", "ok", parent_tool_use_id="toolu_task"),
                         assistant_message("subagent final", parent_tool_use_id="toolu_task", session_id=sid),
                         system_message("task_notification", sid, task_id="t1", status="completed"),
                         tool_result_message("toolu_task", "subagent report"),
                         *ending(sid, answer="main final"))
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "main final")
        rows = self.store.states(self.store.TRANSCRIPT_KIND)
        sub = [r for r in rows if r.get("parent_item_id") == "toolu_task"]
        self.assertEqual({r["type"] for r in sub}, {"agentMessage", "dynamicToolCall"})
        self.assertTrue(all(r.get("phase") != "final_answer" for r in sub))
        self.assertEqual(self.store.transcript_item("toolu_sub")["parent_item_id"], "toolu_task")
        self.assertEqual(self.store.transcript_item("toolu_task")["type"], "mcpToolCall")

    def test_result_guard_waits_for_a_running_subagent(self):
        session = self.started(result_quiet_guard_s=0.1)
        sid = []
        self.script(lambda s, text: (sid.append(s), [*opening(s, text),
                    system_message("task_started", s, task_id="t9", task_type="local_agent"),
                    assistant_message("main paused", session_id=s), result_message(session_id=s)])[1])
        session.run_turn("go")
        time.sleep(0.3)
        self.assertEqual(session.state, "running")
        # A background shell (local_bash) does not hold the turn: only delegated agent work does.
        self.client.emit(system_message("task_started", sid[0], task_id="b1", task_type="local_bash"),
                         system_message("task_updated", sid[0], task_id="t9", patch={"status": "completed"}),
                         result_message(session_id=sid[0]))
        self.assertEqual(self.finish(session, 3), "completed")
        self.assertEqual(self.events("agent.turn_end_fallback")[0]["payload"]["signal"], "result_quiet_fallback")

    def test_subagents_force_full_context_delivery(self):
        for subagents, configured, effective in ((True, "delta", "full"), (False, "delta", "delta"),
                                                 (False, "full", "full")):
            with self.subTest(subagents=subagents, configured=configured):
                document = json.loads(json.dumps(CONFIG))
                document["agent"]["claude_code"].update(subagents=subagents, context_delivery=configured)
                self.config = Config.from_dict(document)
                session = self.make()
                self.assertEqual(session.context_delivery, effective)
                self.assertEqual(session.provenance()["context_delivery_policy"],
                                 dict(effective=effective, configured=configured, subagents=subagents))
        session = self.started()
        block = session.provenance()
        self.assertEqual((block["context_delivery"], block["evidence_client"]), ("full", "relay"))

    def test_relay_attributes_unregistered_requests_to_this_main_thread(self):
        session = self.started()
        proxy = session.model_proxy
        self.assertEqual(proxy.default_thread_id, session.thread_id)
        self.assertEqual(proxy.thread_owners[session.thread_id]["source"], "actor")


class ProbeR40SequenceTests(SessionCase):
    """Real CLI sequences captured by probe R4-0 (ScheduleWakeup/CronCreate, resume catch-up, interrupt of a
    background subagent). Wake-up turns carry no origin marker and no prompt echo: adoption keys on the native
    ``running`` alone."""

    def capture(self, name, session):
        messages = load_capture(FIXTURES / name)
        for message in messages:
            if type(message).__name__ == "SystemMessage" and "session_id" in message.data:
                message.data["session_id"] = session.thread_id
        return CapturedSequenceTests.segments(messages)

    def test_scheduled_wakeups_are_adopted(self):
        session = self.started(result_quiet_guard_s=0.1)
        host, wakeup = self.capture("r4-cc-cron.jsonl", session)
        self.assertFalse(any(type(m).__name__ == "UserMessage" for m in wakeup))   # no echo, no origin
        self.script(host)
        first = session.run_turn("Schedule the probe timers.")
        self.assertEqual(self.finish(session), "completed")
        self.client.emit(*wakeup)
        self.assertTrue(wait_until(lambda: session.turn_id != first and session.turn_done.is_set()))
        self.assertEqual(session.state, "completed")
        self.assertEqual(session.completed_final_answer(), "native answer 1")   # two Results, one Auriga turn
        self.assertEqual([e["payload"]["origin"] for e in self.events("agent.turn_started")],
                         ["host", "native_wakeup"])
        time.sleep(0.15)
        self.assertEqual(self.events("agent.turn_end_fallback"), [])

    def test_catch_up_turn_at_resume_before_the_first_prompt(self):
        thread = ResumeTests.recorded(self)
        session = self.make()
        messages = load_capture(FIXTURES / "r4-cc-cron-disconnect-resume.jsonl")
        for message in messages:
            if type(message).__name__ == "SystemMessage" and "session_id" in message.data:
                message.data["session_id"] = thread
        catch_up, host = CapturedSequenceTests.segments(messages)
        self.fake.connect_hook = lambda client: list(catch_up)
        session.start(TOOLS, "Be careful.", thread)
        self.assertTrue(wait_until(lambda: session.turn_done.is_set() and session.turn_id is not None))
        self.assertEqual((session.state, session.completed_final_answer()), ("completed", "resumed answer 0"))
        self.assertEqual(self.events("agent.turn_started")[-1]["payload"]["origin"], "native_wakeup")
        self.script(host)
        session.run_turn("Anything pending?")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "resumed answer 1")

    def test_first_prompt_joins_a_catch_up_turn_still_running_at_resume(self):
        thread = ResumeTests.recorded(self)
        messages = load_capture(FIXTURES / "r4-cc-cron-disconnect-resume.jsonl")
        for message in messages:
            if type(message).__name__ == "SystemMessage" and "session_id" in message.data:
                message.data["session_id"] = thread
        catch_up, host = CapturedSequenceTests.segments(messages)
        self.fake.connect_hook = lambda client: catch_up[:2]          # running, init: still in progress
        session = self.make()
        session.start(TOOLS, "Be careful.", thread)
        self.assertEqual(session.state, "running")
        adopted = session.turn_id
        self.script(None)
        self.assertEqual(session.run_turn("Anything pending?"), adopted)
        self.client.emit(*catch_up[2:])                                # its Result and idle: the prompt is unread
        self.assertTrue(wait_until(lambda: self.events("agent.turn_end_deferred")))
        self.assertFalse(session.turn_done.is_set())
        self.client.emit(*host)
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual((session.turn_result["id"], session.completed_final_answer()), (adopted, "resumed answer 1"))
        self.assertEqual(len(self.events("agent.turn_completed")), 1)

    def test_interrupt_of_a_background_subagent(self):
        session = self.started(result_quiet_guard_s=0.1)
        (host, followup) = self.capture("r4-cc-agent-int-bg.jsonl", session)
        cut = next(i for i, m in enumerate(host) if type(m).__name__ == "ResultMessage") + 1
        before, after = host[:cut], host[cut:]
        self.script(before)
        turn = session.run_turn("Run the subagent and wait.")
        time.sleep(0.3)                                    # main Result while the local_agent task runs: no guard
        self.assertEqual(session.state, "running")
        self.fake.on_interrupt.append(lambda client: client.emit(*after))
        session.interrupt()
        self.assertEqual(self.finish(session), "interrupted")   # last Result is success: the interrupt flag decides
        self.assertEqual(session.turn_result["id"], turn)
        def late_rows():
            return [r for r in self.store.states(self.store.TRANSCRIPT_KIND) if r.get("parent_item_id")]
        self.assertTrue(wait_until(late_rows))             # the subagent's rows arrive after the idle
        late = late_rows()
        self.assertTrue(all(r["turn_id"] == turn for r in late))   # late subagent rows stay with that turn
        self.assertEqual(len(self.events("agent.turn_started")), 1)  # and open no turn
        self.script(followup)
        session.run_turn("FOLLOWUP after interrupt.")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(self.events("agent.turn_end_fallback"), [])


@requires_sdk
class WorkbenchAdoptionTests(unittest.IsolatedAsyncioTestCase):
    """The gateway's `prompt` command while an adopted turn runs: refused like any running turn (design R4.2: no caller
    change); `steer` joins it."""

    async def asyncSetUp(self):
        from aiohttp.test_utils import TestClient, TestServer
        from auriga.apps.gateway.workbench import Workbench
        from test_harness import FixtureRobot
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        environment = mock.patch.dict(os.environ, {KEY_ENV: "host-secret"})
        environment.start()
        self.addCleanup(environment.stop)
        self.fake = FakeClaudeSDK()
        runtime = claude_options.ClaudeRuntime(sdk_version=claude_options.SDK_VERSION,
                                               cli_version=claude_options.CLI_VERSION,
                                               executable="/usr/bin/true", sha256="0" * 64)
        created = []

        def create(config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
            created.append(ClaudeCodeSession(run_dir, store, tool_handler, config=config, price_card=price_card,
                                             client_factory=self.fake.factory, runtime=runtime))
            return created[-1]
        robot = FixtureRobot(root)
        robot.close = lambda: None
        document = json.loads(json.dumps(CONFIG))
        document["features"].update(grasp_memory_dir=str(root / "memory"), observation_profile="rgb_proprio_v1",
                                    motion_review=False)
        with mock.patch("auriga.apps.gateway.workbench.create_session", side_effect=create), \
                mock.patch("auriga.apps.gateway.workbench.backend_requirements"):
            self.workbench = Workbench(root / "run", robot=robot, config=Config.from_dict(document))
        self.session, = created
        self.workbench.harness.start()
        self.workbench.starting, self.workbench.status = False, "running"
        await asyncio.to_thread(self.session.start, TOOLS, "instructions")
        self.client = TestClient(TestServer(self.workbench.app()))
        await self.client.start_server()
        self.origin = str(self.client.make_url("")).rstrip("/")
        owner = await (await self.client.post("/v1/control", headers={"Origin": self.origin}, json=dict(
            request_id="claim", command="claim_input", client_id="fixture"))).json()
        self.lease = dict(client_id="fixture", writer_token=owner["writer_token"])

    async def asyncTearDown(self):
        await self.client.close()
        await self.workbench.close()

    async def control(self, identifier, command, **fields):
        response = await self.client.post("/v1/control", headers={"Origin": self.origin}, json=dict(
            request_id=identifier, command=command, **self.lease, **fields))
        return response.status, await response.json()

    async def test_prompt_is_refused_and_steer_joins_while_an_adopted_turn_runs(self):
        sid = self.session.thread_id
        self.fake.client.emit(state_message("running", sid), init_message(sid))
        self.assertTrue(await asyncio.to_thread(wait_until, lambda: self.session.state == "running"))
        adopted = self.session.turn_id
        status, body = await self.control("p1", "prompt", prompt="new task")
        self.assertEqual((status, body["error"]), (400, "run_not_ready_for_new_task"))
        self.assertEqual(self.fake.client.queries(), [])
        status, body = await self.control("s1", "steer", prompt="operator text")
        self.assertEqual((status, body["status"], body["turn_id"]), (200, "steered", adopted))
        self.fake.client.emit(echo_message("operator text"), *ending(sid, answer="done"))
        self.assertTrue(await asyncio.to_thread(self.session.turn_done.wait, 5))
        self.assertEqual((self.session.state, self.session.turn_result["id"]), ("completed", adopted))
        status, body = await self.control("p2", "prompt", prompt="next task")
        self.assertEqual((status, body["status"]), (200, "running"))
        self.assertNotEqual(body["turn_id"], adopted)
        self.fake.client.emit(state_message("running", sid), echo_message("next task"), *ending(sid))
        self.assertTrue(await asyncio.to_thread(self.session.turn_done.wait, 5))


@requires_sdk
class FactoryAndGateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.store = Store(self.root / "run", IDENTITY)
        self.addCleanup(self.store.close)

    def test_create_session_builds_a_claude_code_session_without_the_sdk(self):
        config = Config.from_dict(CONFIG)
        card, planning = {"input_per_million": 1.0}, object()
        with mock.patch.dict(sys.modules, {"claude_agent_sdk": None}):
            session = factory.create_session(config, self.root / "run", self.store, None, price_card=card,
                                             task_planning=planning)
            session.enter_replay("recorded")
            session.close()
        self.assertIsInstance(session, ClaudeCodeSession)
        self.assertEqual((session.price_card, session.task_planning, session.model, session.backend_kind),
                         (card, planning, "claude-test-model", "claude_code"))

    def test_requirements_pass_with_the_pinned_runtime_and_refuse_without_it(self):
        config = Config.from_dict(CONFIG)
        with mock.patch.object(factory, "claude_code_runtime_problem", return_value=None):
            self.assertIsNone(factory.backend_requirements(config))
        with mock.patch.object(factory, "claude_code_runtime_problem", return_value="claude_code_version_mismatch"):
            with self.assertRaisesRegex(ConfigError, "^agent_backend_unavailable:claude_code: "
                                                     "claude_code_version_mismatch; install auriga-harness\\[claude\\]$"):
                factory.backend_requirements(config)

    def test_requirements_check_the_real_runtime(self):
        self.assertIsNone(factory.claude_code_runtime_problem())
        self.assertEqual(factory.claude_code_runtime_problem("relative/claude"), "claude_code_cli_path_must_be_absolute")
        self.assertEqual(factory.claude_code_runtime_problem("/usr/bin/false"), "claude_code_cli_version_probe_failed")
        with mock.patch.dict(sys.modules, {"claude_agent_sdk": None}):
            self.assertTrue(factory.claude_code_runtime_problem().startswith("claude_code_sdk_missing"))

    def test_requirement_order_and_upstream(self):
        quiet = {"self_briefing": False, "visual_review_model": None}
        cases = (
            # features.sandbox_enabled is accepted (R4-8): the configuration checks after it still apply.
            ({"features": {**quiet, "sandbox_enabled": True, "lean_projection": True}},
             "lean_projection_requires_codex_backend"),
            ({"features": {**quiet, "lean_projection": True}}, "lean_projection_requires_codex_backend"),
            ({"features": quiet, "model": {"effort": "ultra"}}, "agent_effort_unsupported:claude_code:ultra"),
            ({"features": quiet, "agent": {"claude_code": {"upstream": {"wire_api": "openai_responses"}}}},
             "claude_code_upstream_requires_anthropic_messages"),
            ({"features": quiet, "agent": {"claude_code": {"upstream": {"source": "codex_home"}}}},
             "claude_code_upstream_requires_anthropic_messages"),
        )
        for document, code in cases:
            with self.subTest(code=code, document=document), \
                    mock.patch.object(factory, "claude_code_runtime_problem", return_value=None) as probe:
                agent = {"backend": "claude_code", **document.pop("agent", {})}
                config = Config.from_dict({**document, "agent": agent})
                with self.assertRaises(ConfigError) as raised:
                    factory.backend_requirements(config)
                self.assertTrue(str(raised.exception).startswith(code + ": "), str(raised.exception))
                probe.assert_not_called()

    def test_opencode_is_implemented(self):
        # Implemented since P3 (tests/test_agent_opencode_session_unit.py FactoryTests).
        session = factory.create_session(Config.from_dict({"agent": {"backend": "opencode"}}), self.root, self.store, None)
        self.assertEqual(type(session).__name__, "OpenCodeSession")


class ArchiveToolTests(unittest.TestCase):
    """Non-Codex archives open in replay without the runtime, resume is gated on the archived config, and the
    Codex-only checkpoint/probe tools refuse them (P0-3 review follow-ups)."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()

    def archive(self, backend):
        run = self.root / ("run-" + backend)
        store = Store(run, IDENTITY)
        store.set_state("runtime", "agent", dict(backend=backend, thread_id="t", instance_id="agent-1"))
        store.close()
        return run

    def test_checkpoint_and_probes_refuse_non_codex_archives(self):
        from auriga.apps.evaluation import checkpoint, turn_probe
        run = self.archive("claude_code")
        target = self.root / "slice"
        with self.assertRaisesRegex(ValueError, "^checkpoint_requires_codex_archive:claude_code$"):
            checkpoint.prepare_checkpoint_slice(run, target, until_call_id="c")
        self.assertFalse(target.exists())
        for probe in (lambda: turn_probe.probe_forked_turn(run, "c", work_dir=target),
                      lambda: turn_probe.run_live_slice(run, until_call_id="c", work_dir=target)):
            with self.assertRaisesRegex(ValueError, "^checkpoint_requires_codex_archive:claude_code$"):
                probe()
        self.assertFalse(target.exists())
        checkpoint.require_codex_archive(self.archive("codex"))
        legacy = self.root / "legacy"
        Store(legacy, IDENTITY).close()
        checkpoint.require_codex_archive(legacy)  # no identity record: an older Codex run

    def test_resume_is_gated_by_the_workbench_on_the_archived_config(self):
        from auriga.apps.gateway import server
        run = self.root / "run"
        run.mkdir()
        invocation = Config.from_dict({"agent": {"backend": "claude_code"}, "features": {"sandbox_enabled": True}})
        args = SimpleNamespace(run_dir=str(run), resume=True, replay=False, price_card=None,
                               experiment_root=str(self.root), category="diagnostic", task="fixture", title="fixture",
                               tag=[], resolved_config=invocation, gpu=0, max_actions=None, observation_profile=None,
                               seed=0, layout_id=0)

        class Reached(Exception):
            pass
        with mock.patch.object(server, "Workbench", side_effect=Reached) as workbench, \
                mock.patch.object(server, "register_run"):
            with self.assertRaises(Reached):
                asyncio.run(server.serve(args))
        workbench.assert_called_once()

    @requires_sdk
    def test_replay_of_a_claude_code_archive_opens_without_the_runtime(self):
        from unittest.mock import Mock
        from auriga.apps.gateway.workbench import Workbench
        from support.fake_agent import FakeAgentSession
        from test_agent_workbench import fixture_harness, terminal_double

        run = self.root / "claude-run"
        config = Config.from_dict(CONFIG)
        created = []

        def create(config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
            created.append(FakeAgentSession(run_dir, store, tool_handler, price_card=price_card))
            return created[-1]

        async def record():
            robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
            with mock.patch("auriga.apps.gateway.workbench.create_session", side_effect=create), \
                    mock.patch("auriga.apps.gateway.workbench.backend_requirements"):
                workbench = Workbench(run, robot=robot, visual_review_model=None, config=config)
            workbench.harness = fixture_harness(workbench)
            with mock.patch("auriga.apps.gateway.workbench.Terminal", return_value=terminal_double()):
                await workbench.start("unused", benchmark=True)
            workbench.store.set_state("runtime", "agent", dict(backend="claude_code", thread_id="session-uuid",
                                                               instance_id="agent-1"))
            workbench.store.set_state("runtime", "capabilities", dict(data_origin="synthetic_fixture"))
            await workbench.close()

        async def replay():
            with mock.patch.dict(sys.modules, {"claude_agent_sdk": None}):
                viewer = Workbench(run, robot=SimpleNamespace(close=Mock()), replay=True)
                try:
                    await viewer.start("must not start a turn")
                    return type(viewer.session), viewer.session.state, viewer.session.thread_id, viewer.snapshot()
                finally:
                    await viewer.close()

        asyncio.run(record())
        kind, state, thread, snapshot = asyncio.run(replay())
        self.assertIs(kind, ClaudeCodeSession)
        self.assertEqual((state, thread), ("replay", "session-uuid"))
        self.assertEqual((snapshot["agent_backend"], snapshot["agent_status"]), ("claude_code", "replay"))


if __name__ == "__main__":
    unittest.main()
