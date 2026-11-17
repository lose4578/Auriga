"""Shared MCP-backend session skeleton (plan 07 §1; design 04 §2.2, §3.3, §3.7, §5.5): McpToolBridge + McpSessionMixin.

Loopback only (real AurigaMcpServer, FakeMcpClient, real Store); no agent binary, no model.
"""
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

from support.fake_mcp_client import FakeMcpClient

from auriga.core.config import Config
from auriga.core.presentation.actor_presentation import ActorPresentation
from auriga.core.storage.store import Store
from auriga.cortex.agents.base import AgentRejected, AgentSession
from auriga.cortex.agents.mcp_session import GENERATION_COUNTING, McpSessionMixin, McpToolBridge
from auriga.cortex.agents.transcript import TranscriptWriter
from auriga.cortex.session.wire import AnthropicMessagesWire

IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
PLANTED_HASH = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x in m, default 0"}}}
TOOLS = [{"type": "function", "name": "look", "description": "Read-only view.", "inputSchema": SCHEMA},
         {"type": "function", "name": "move", "description": "Stateful motion.", "inputSchema": SCHEMA}]


def text_result(text, success=True):
    return {"success": success, "contentItems": [{"type": "inputText", "text": text}]}


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class StubSession(McpSessionMixin, AgentSession):
    """Minimal MCP-backend session over the mixin: real Store/transcript/presentation, no native process."""
    backend_kind = "stub_backend"
    counting = GENERATION_COUNTING

    def __init__(self, root, store, handler, *, parallel_safe=None):
        self.lock = threading.RLock()
        self.root, self.store, self.private = root, store, root / "private"
        self.private.mkdir(parents=True, exist_ok=True)
        self.presentation = ActorPresentation(store)
        self.tool_handler, self.tool_parallel_safe = handler, parallel_safe
        self.state, self.thread_id, self.turn_id = "idle", "thread-1", None
        self.turn_done, self.turn_result = threading.Event(), None
        self.transcript = TranscriptWriter(store, backend=self.backend_kind, thread_id=self.thread_id)
        self.closing, self.close_report = False, None
        self.model, self.provider, self.price_card = "model-a", "provider-a", None
        self.config, self.wire = Config(), AnthropicMessagesWire(client="claude_code")
        self.model_proxy, self.model_limit_reached, self.bridge = None, threading.Event(), None
        self.last_activity_at = 0.0
        self.resets, self.disconnects, self.order = [], [], []
        self.on_context_reset = self.resets.append
        self.on_disconnect = lambda: self.disconnects.append(self.state)
        self._identity = None
        self._teardown_confirmed = True

    def _stop_backend(self):
        self.order.append("backend")


class RecordingHandler:
    def __init__(self, results=None, error=None, gate=None):
        self.calls, self.results, self.error, self.gate = [], results or {}, error, gate

    def __call__(self, params):
        self.calls.append(params)
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return self.results.get(params["tool"], text_result("ok " + params["tool"]))


class Case(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "run"
        self.store = Store(self.root, IDENTITY)
        self.addCleanup(self.store.close)

    def session(self, handler=None, **kwargs):
        self.handler = handler or RecordingHandler()
        return StubSession(self.root, self.store, self.handler, **kwargs)

    def bridge(self, session, hints=None):
        bridge = McpToolBridge(session, TOOLS, parallel_hints=hints).start()
        session.bridge = bridge
        self.addCleanup(bridge.join)
        self.addCleanup(bridge.stop_accepting)
        client = FakeMcpClient(bridge.url, headers=bridge.headers)
        client.handshake()
        return bridge, client

    def events(self, kind):
        return self.store.events_by_type([kind], limit=1000)


class BridgeTests(Case):
    def running(self, session):
        session._begin_turn()
        return session.turn_id

    def test_native_call_id_reaches_the_handler_and_names_the_transcript_row(self):
        session = self.session(RecordingHandler(results={"look": text_result("seen sha256 " + PLANTED_HASH)}))
        bridge, client = self.bridge(session, hints={"look": True})
        turn = self.running(session)
        result = client.call_tool("look", {"x": 1}, meta={"claudecode/toolUseId": "toolu_01"})
        self.assertFalse(result.result["isError"])
        params, = self.handler.calls
        self.assertEqual(params, dict(threadId="thread-1", turnId=turn, callId="toolu_01", tool="look",
                                      arguments={"x": 1}, callIdSource="native_tool_use_id"))
        row = self.store.transcript_item("toolu_01")
        self.assertEqual((row["type"], row["status"], row["success"], row["turn_id"], row["native_id"]),
                         ("dynamicToolCall", "completed", True, turn, "toolu_01"))
        # The second presentation scrub removed the planted digest from the delivered text and the row.
        self.assertNotIn(PLANTED_HASH, result.result["content"][0]["text"])
        self.assertNotIn(PLANTED_HASH, row["contentItems"][0]["text"])
        self.assertEqual(bridge.read_only_tools, ["look"])
        self.assertGreater(session.last_activity_at, 0)

    def test_call_without_native_id_gets_a_host_call_id(self):
        session = self.session()
        _, client = self.bridge(session)
        self.running(session)
        client.call_tool("move", {"x": 2})
        params, = self.handler.calls
        self.assertTrue(params["callId"].startswith("call-"))
        self.assertEqual(params["callIdSource"], "mcp_request")
        self.assertEqual(self.store.transcript_item(params["callId"])["tool"], "move")

    def test_stale_connection_is_refused_before_dispatch(self):
        for label, mutate in (("closing", lambda s: setattr(s, "closing", True)),
                              ("disconnected", lambda s: setattr(s, "state", "disconnected")),
                              ("closed", lambda s: setattr(s, "state", "closed")),
                              ("not_started", lambda s: setattr(s, "state", "not_started")),
                              ("replay", lambda s: setattr(s, "state", "replay")),
                              ("no_turn", lambda s: setattr(s, "turn_id", None)),
                              ("no_thread", lambda s: setattr(s, "thread_id", None)),
                              # Review A1: after its turn's terminal a call has no turn that owns it.
                              ("completed", lambda s: s._finish_turn("completed")),
                              ("interrupted", lambda s: s._finish_turn("interrupted")),
                              ("idle", lambda s: setattr(s, "state", "idle"))):
            with self.subTest(label):
                session = self.session()
                _, client = self.bridge(session)
                self.running(session)
                mutate(session)
                result = client.call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_" + label})
                self.assertTrue(result.result["isError"])
                self.assertIn("native_tool_call_stale_connection", result.result["content"][0]["text"])
                self.assertEqual(self.handler.calls, [])
                self.assertIsNone(self.store.transcript_item("toolu_" + label))

    def test_handler_exception_is_an_error_result_and_a_failed_row(self):
        session = self.session(RecordingHandler(error=ValueError("tool_exploded")))
        _, client = self.bridge(session)
        self.running(session)
        result = client.call_tool("move", {}, meta={"claudecode/toolUseId": "toolu_err"})
        self.assertTrue(result.result["isError"])
        self.assertIn("tool_exploded", result.result["content"][0]["text"])
        row = self.store.transcript_item("toolu_err")
        self.assertEqual((row["status"], row["success"]), ("failed", False))

    def test_uncertain_delivery_changes_the_epoch_and_reports_a_context_reset(self):
        gate = threading.Event()
        session = self.session(RecordingHandler(gate=gate))
        _, client = self.bridge(session)
        self.running(session)
        pending = client.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_cancel"})
        self.assertTrue(wait_until(lambda: self.handler.calls))
        client.cancel(pending.request_id, reason="AbortError: remote-cancel")
        self.assertTrue(wait_until(lambda: session.resets))
        gate.set()
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual((epoch["source"], epoch["tool_call_id"], epoch["thread_id"]),
                         ("tool_reply_delivery_uncertain", "toolu_cancel", "thread-1"))
        self.assertTrue(epoch["marker"].startswith("image-delivery-failure-"))
        self.assertEqual(session.resets, ["tool_reply_delivery_uncertain"])
        unknown, = self.events("agent.tool_reply_unknown")
        self.assertEqual((unknown["payload"], unknown["correlation"]["tool_call_id"]),
                         ({"reason": "tool_reply_delivery_uncertain"}, "toolu_cancel"))

    def test_delivery_failure_never_raises(self):
        session = self.session()
        bridge = McpToolBridge(session, TOOLS)
        self.addCleanup(bridge.join)
        session.store = None  # every write fails
        bridge._delivery_failure("call-x", "tool_reply_delivery_uncertain")

    def test_join_waits_for_the_handler_threads_transcript_writes(self):
        # Close joins the tool workers, then the gateway closes the Store: a handler thread still writing its
        # transcript row after join returned would use a closed sqlite connection (seen as a segfault).
        gate = threading.Event()
        session = self.session(RecordingHandler(gate=gate))
        bridge, client = self.bridge(session)
        self.running(session)
        original, written = session.transcript.tool_completed, []

        def slow_completion(*args, **kwargs):
            time.sleep(0.3)
            written.append(original(*args, **kwargs))
        session.transcript.tool_completed = slow_completion
        pending = client.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_join"})
        self.assertTrue(wait_until(lambda: self.handler.calls))
        gate.set()
        bridge.join()
        self.assertEqual(written, ["toolu_join"])
        pending.wait()
        session.transcript.tool_completed = lambda *args, **kwargs: time.sleep(5)
        with mock.patch.object(bridge, "_active", 1):
            with self.assertRaisesRegex(RuntimeError, "^tool_handlers_still_running$"):
                bridge.join(timeout_s=0.1)

    def test_parallel_safe_calls_overlap_and_ordered_calls_wait(self):
        gate, active, peak = threading.Event(), [], []
        lock = threading.Lock()

        def handler(params):
            with lock:
                active.append(params["tool"])
                peak.append(len(active))
            if params["tool"] == "look":
                gate.wait(5)
            with lock:
                active.remove(params["tool"])
            return text_result("ok")

        session = self.session(handler, parallel_safe=lambda tool, arguments: tool == "look")
        _, client = self.bridge(session)
        self.running(session)
        first = client.start_call("look", {}, meta={"claudecode/toolUseId": "toolu_a"})
        second = client.start_call("look", {}, meta={"claudecode/toolUseId": "toolu_b"})
        self.assertTrue(wait_until(lambda: max(peak or [0]) == 2))
        ordered = client.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_c"})
        time.sleep(0.1)
        self.assertIsNone(self.store.transcript_item("toolu_c").get("success"))
        gate.set()
        for call in (first, second, ordered):
            self.assertFalse(call.wait().result["isError"])
        self.assertEqual(max(peak), 2)


class TurnHelperTests(Case):
    def test_begin_enters_running_with_a_host_turn_id(self):
        session = self.session()
        previous = session._begin_turn()
        self.assertEqual(previous[0], "idle")
        self.assertEqual(session.state, "running")
        self.assertTrue(session.turn_id.startswith("turn-"))
        self.assertEqual(session.transcript.turn_id, session.turn_id)
        self.assertEqual(session.active_turn_model, "model-a")
        self.assertFalse(session.turn_done.is_set())
        started, = self.events("agent.turn_started")
        self.assertEqual(started["correlation"]["turn_id"], session.turn_id)

    def test_begin_refuses_while_running_and_when_not_connected(self):
        session = self.session()
        session._begin_turn()
        with self.assertRaisesRegex(ValueError, "^turn_already_active$"):
            session._begin_turn()
        for state in ("disconnected", "closed", "unknown", "not_started", "replay"):
            session.state = state
            with self.subTest(state), self.assertRaisesRegex(ValueError, "^session_not_connected$"):
                session._begin_turn()

    def test_store_failure_in_begin_changes_nothing(self):
        # Review (b)1: the turn record is written first; a refusal keeps turn_done set and the previous result.
        session = self.session()
        session._begin_turn()
        session._finish_turn("completed")
        kept = (session.state, session.turn_id, session.turn_result, session.transcript.turn_id)
        original = session.transcript.event
        session.transcript.event = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("database is locked"))
        with self.assertRaisesRegex(RuntimeError, "database is locked"):
            session._begin_turn()
        session.transcript.event = original
        self.assertEqual((session.state, session.turn_id, session.turn_result, session.transcript.turn_id), kept)
        self.assertTrue(session.turn_done.is_set())

    def test_adopted_turn_origin(self):
        session = self.session()
        session._begin_turn()
        session._finish_turn("completed")
        session._begin_turn(origin="native_wakeup")   # no admissibility check: the native runtime already runs it
        self.assertEqual((session.state, session.turn_origin), ("running", "native_wakeup"))
        self.assertEqual([event["payload"]["origin"] for event in self.events("agent.turn_started")],
                         ["host", "native_wakeup"])
        with self.assertRaisesRegex(ValueError, "^invalid_turn_origin$"):
            session._begin_turn(origin="other")

    def test_reject_restores_the_previous_turn(self):
        session = self.session()
        session.turn_id, session.turn_result, session.state = "turn-old", {"status": "completed"}, "completed"
        session.transcript.turn_id = "turn-old"
        session.active_turn_model = "model-old"
        previous = session._begin_turn()
        rejected = session.turn_id
        with self.assertRaises(AgentRejected) as raised:
            session._reject_turn(previous, "not_connected", "backend_not_connected")
        self.assertEqual((raised.exception.code, raised.exception.message, raised.exception.busy),
                         ("not_connected", "backend_not_connected", False))
        self.assertEqual((session.state, session.turn_id, session.turn_result, session.active_turn_model,
                          session.transcript.turn_id),
                         ("completed", "turn-old", {"status": "completed"}, "model-old", "turn-old"))
        self.assertTrue(session.turn_done.is_set())
        event, = self.events("agent.turn_start_rejected")
        self.assertEqual(event["correlation"]["turn_id"], rejected)
        self.assertEqual(event["payload"]["code"], "not_connected")

    def test_finish_stamps_the_final_answer_once(self):
        session = self.session()
        session._begin_turn()
        session.transcript.assistant_text("thinking aloud", message_id="m1")
        session.transcript.tool_started("call-1", "look", {}, message_id="m1")
        session.transcript.assistant_text("FINAL", message_id="m2")
        self.assertTrue(session._finish_turn("completed"))
        self.assertFalse(session._finish_turn("failed", "late"))
        self.assertEqual(session.turn_result, dict(id=session.turn_id, status="completed", error=None))
        self.assertEqual(session.state, "completed")
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(session.completed_final_answer(), "FINAL")
        completed, = self.events("agent.turn_completed")
        self.assertEqual(completed["payload"], {"status": "completed", "error": None})

    def test_finish_with_error_keeps_a_host_message(self):
        session = self.session()
        session._begin_turn()
        session._finish_turn("failed", "claude_code_turn_failed:api_error:403")
        self.assertEqual(session.turn_result["error"], {"message": "claude_code_turn_failed:api_error:403"})
        self.assertIsNone(session.completed_final_answer())

    def test_disconnected_is_reported_once_with_a_synthetic_failed_turn(self):
        session = self.session()
        session._begin_turn()
        session._disconnected(ConnectionResetError("pipe"))
        session._disconnected(OSError("again"))
        self.assertEqual(session.state, "disconnected")
        self.assertEqual(session.disconnects, ["disconnected"])
        self.assertEqual(session.turn_result["status"], "failed")
        self.assertEqual(session.turn_result["error"]["message"], "upstream_connection_lost: ConnectionResetError")
        self.assertTrue(session.turn_done.is_set())
        lost, = self.events("agent.connection_lost")
        self.assertEqual(lost["payload"]["reason"], "ConnectionResetError")
        self.assertFalse(lost["payload"]["history_complete"])
        # A terminal after the disconnect changes nothing.
        self.assertFalse(session._finish_turn("completed"))
        with self.assertRaisesRegex(ValueError, "session_not_connected"):
            session._begin_turn()

    def test_disconnected_while_idle_keeps_the_previous_turn_result(self):
        session = self.session()
        session._begin_turn()
        session._finish_turn("completed")
        kept = session.turn_result
        session._disconnected(EOFError("stream_closed"))
        self.assertIs(session.turn_result, kept)
        self.assertEqual(session.disconnects, ["disconnected"])

    def test_disconnected_is_ignored_while_closing(self):
        session = self.session()
        session.closing = True
        session._disconnected(EOFError())
        self.assertEqual((session.state, session.disconnects), ("idle", []))

    def test_disconnect_listener_failure_still_sets_turn_done(self):
        session = self.session()
        session._begin_turn()
        session.on_disconnect = lambda: (_ for _ in ()).throw(RuntimeError("listener"))
        with self.assertRaises(RuntimeError):
            session._disconnected(EOFError())
        self.assertTrue(session.turn_done.is_set())

    def test_context_invalidation_and_relay_discontinuity(self):
        session = self.session()
        session._invalidate_context("agent_resume_after_unclean_exit")
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual(epoch["source"], "agent_resume_after_unclean_exit")
        session._relay_discontinuity("history_rewritten")
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual((epoch["source"], epoch["signal"]), ("relay_context_discontinuity", "history_rewritten"))
        self.assertEqual(session.resets, ["agent_resume_after_unclean_exit", "relay_context_discontinuity"])
        compaction, = self.events("agent.compaction")
        self.assertEqual(compaction["payload"], {"signal": "history_rewritten", "source": "relay"})
        session.on_context_reset = lambda reason: (_ for _ in ()).throw(RuntimeError("listener"))
        session._context_reset("ignored")  # never raises


class FakeProxy:
    def __init__(self, order):
        self.order, self.request_count, self.attempt_count = order, 0, 0
        self.transport_blocked, self.blocked_error = False, None
        self.transport_recovery_summary = dict(policy=None, rounds=0)

    def cancel_pending(self, reason):
        self.order.append(("cancel_pending", reason))

    def reopen_thread(self, thread_id):
        self.order.append(("reopen_thread", thread_id))
        return True

    def close(self):
        self.order.append("relay_close")


class RelayAndCloseTests(Case):
    def test_close_order_and_clean_shutdown(self):
        session = self.session()
        proxy = session.model_proxy = FakeProxy(session.order)
        bridge = McpToolBridge(session, TOOLS).start()
        session.bridge = bridge
        original_stop, original_join = bridge.stop_accepting, bridge.join
        bridge.stop_accepting = lambda: (session.order.append("mcp_close"), original_stop())
        bridge.join = lambda: (session.order.append("tool_join"), original_join())
        session._identity = dict(backend="stub_backend", thread_id="thread-1", clean_shutdown=False)
        report = session.close()
        self.assertEqual(session.order, [("cancel_pending", "session_closed"), "backend", "relay_close", "mcp_close",
                                         "tool_join"])
        self.assertIs(proxy, session.model_proxy)
        self.assertTrue(report["clean_shutdown"])
        self.assertEqual(self.store.read_state("runtime", "agent")["clean_shutdown"], True)
        self.assertEqual((session.state, session.turn_done.is_set()), ("closed", True))
        self.assertIs(session.close(), report)  # idempotent
        self.assertEqual(len(self.events("agent.session_closed")), 1)

    def test_close_during_a_turn_is_not_clean_and_collects_step_errors(self):
        session = self.session()
        session._identity = dict(backend="stub_backend", clean_shutdown=False)
        session._begin_turn()

        def broken():
            raise RuntimeError("backend_stuck")
        session._stop_backend = broken
        report = session.close()
        self.assertFalse(report["clean_shutdown"])
        self.assertEqual(report["steps"]["backend"], "failed")
        self.assertIn("backend_stuck", report["errors"][0]["error"])
        self.assertFalse(self.store.read_state("runtime", "agent")["clean_shutdown"])
        self.assertTrue(session.turn_done.is_set())
        # Tool calls after close are stale.
        with self.assertRaisesRegex(ValueError, "native_tool_call_stale_connection"):
            McpToolBridge(session, TOOLS)._call_tool("look", {}, call_id="c", native_call_id=None, rpc_id=1)

    def test_native_turn_in_progress_at_close_is_not_clean(self):
        session = self.session()
        session._native_active = lambda: True   # e.g. a native wake-up the host has not adopted
        report = session.close()
        self.assertFalse(report["clean_shutdown"])
        self.assertTrue(report["native_active_at_close"])

    def test_unconfirmed_teardown_is_not_clean(self):
        session = self.session()
        session._teardown_confirmed = False
        self.assertFalse(session.close()["clean_shutdown"])

    def test_model_events_account_relay_usage_of_this_thread_only(self):
        session = self.session()
        session._begin_turn()
        proxy = session.model_proxy = FakeProxy([])
        session._relay_used, session._relay_attempts, session._attempt_usage = 3, 4, {}
        usage = {"input_tokens": 100, "output_tokens": 7, "cache_read_input_tokens": 50,
                 "cache_creation_input_tokens": 0}
        proxy.request_count, proxy.attempt_count = 1, 1
        session._model_event(dict(kind="model_request_finished", logical_call_id="t:1", usage=usage, usage_known=True))
        session._model_event(dict(kind="model_call_completed", logical_call_id="t:1", native_thread_id="thread-1"))
        session._model_event(dict(kind="model_request_finished", logical_call_id="t:2", usage=usage, usage_known=True))
        session._model_event(dict(kind="model_call_completed", logical_call_id="t:2", native_thread_id="other"))
        requests = self.store.read_state("runtime", "model_requests")
        self.assertEqual((requests["completed_calls"], requests["http_attempt_count"], requests["counting"],
                          requests["limit"]), (4, 5, GENERATION_COUNTING, 180))
        relay = self.store.read_state("runtime", "model_usage_relay")
        self.assertEqual(relay["calls"], 1)
        self.assertEqual(relay["total"]["output_tokens"], 7)
        accounting = self.store.read_state("runtime", "token_accounting")
        self.assertEqual(accounting["source"], "relay.anthropic_messages.usage.sum")
        # A delivered call without usage makes the totals unknown, never zero.
        session._model_event(dict(kind="model_request_finished", logical_call_id="t:3", usage=None, usage_known=False))
        session._model_event(dict(kind="model_call_completed", logical_call_id="t:3", native_thread_id="thread-1"))
        relay = self.store.read_state("runtime", "model_usage_relay")
        # The Anthropic wire never reports reasoning tokens separately, so every call has an unknown counter.
        self.assertEqual((relay["calls"], relay["unknown_calls"]), (2, 2))
        self.assertIsNone(relay["total"]["output_tokens"])
        self.assertIsNone(relay["total"]["reasoning_output_tokens"])
        self.assertEqual(len(self.events("model.model_call_completed")), 3)

    def test_reopen_relay_thread_lifts_the_interrupt_tombstone(self):
        session = self.session()
        self.assertFalse(session._reopen_relay_thread())  # no relay yet
        session.model_proxy = FakeProxy(session.order)
        self.assertTrue(session._reopen_relay_thread())
        self.assertEqual(session.order, [("reopen_thread", "thread-1")])

    def test_budget_resume_rules(self):
        session = self.session()
        self.assertEqual(session._relay_budget(), (0, 0, session.config.model.transport_recovery.to_dict()))
        self.store.set_state("runtime", "model_requests", dict(counting=GENERATION_COUNTING, completed_calls=12,
                                                               http_attempt_count=15,
                                                               transport_recovery=dict(policy=None)))
        self.assertEqual(session._relay_budget(), (12, 15, None))
        self.store.set_state("runtime", "model_requests", dict(counting="other", completed_calls=1))
        with self.assertRaisesRegex(ValueError, "^resume_model_budget_semantics_changed"):
            session._relay_budget()
        self.store.set_state("runtime", "model_requests", dict(counting=GENERATION_COUNTING, completed_calls=180))
        with self.assertRaisesRegex(ValueError, "^episode_model_request_limit_exhausted$"):
            session._relay_budget()

    def test_start_relay_builds_the_budgeted_proxy(self):
        session = self.session()
        self.store.set_state("runtime", "model_requests", dict(counting=GENERATION_COUNTING, completed_calls=170,
                                                               transport_recovery=dict(policy=None)))
        settings = dict(reuse_exact_images=False, trim_oversized_requests=True, trim_max_inline_images=0)
        proxy = session._start_relay("http://127.0.0.1:9", upstream_auth_headers={"x-api-key": "host-secret"},
                                     request_projection=lambda request: request, default_thread_id="thread-1",
                                     image_settings=settings)
        self.addCleanup(proxy.close)
        self.assertIs(session.model_proxy, proxy)
        self.assertEqual((proxy.max_requests, proxy.default_thread_id, proxy.wire.name), (10, "thread-1",
                                                                                          "anthropic_messages"))
        self.assertTrue(proxy.base_url.startswith("http://127.0.0.1:"))
        self.assertEqual(self.store.read_state("runtime", "model_image_policy")["oversized_trim"], True)


if __name__ == "__main__":
    unittest.main()
