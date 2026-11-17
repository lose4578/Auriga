"""One real-binary smoke run of ClaudeCodeSession (plan 07 §4; the full conformance suite is a separate package).

Drives the pinned bundled Claude Code CLI 2.1.321 through the real relay (ModelRequestProxy +
AnthropicMessagesWire, upstream = FakeAnthropicServer on loopback) and the real AurigaMcpServer: start -> one turn
with one registry tool call -> completed with the final answer; an interrupt during a gated model call; close leaves
no process. No real model, no network: the CLI's HOME/CLAUDE_CONFIG_DIR/TMPDIR are inside the temporary run
directory and its only credential is the relay placeholder. Skipped (with the reason) when the pinned runtime is absent.
"""
import json
import os
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from support.fake_model_servers import FakeAnthropicServer, Gate, Step, text_step, tool_step

from auriga.core.config import Config
from auriga.core.storage.store import Store
from auriga.cortex.agents import factory
from auriga.cortex.agents.isolation import process_tree_snapshot

RUNTIME_PROBLEM = factory.claude_code_runtime_problem()
IDENTITY = dict(run_id="smoke", episode_id="ep", env_instance_id="env", env_generation=1)
KEY_ENV = "AURIGA_SMOKE_UPSTREAM_KEY"
HOST_KEY = "host-secret-smoke-key"
MARKER = "AURIGA-SMOKE-INSTRUCTIONS-7f3a"
TOOL_USE_ID = "toolu_smoke_01"
SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x position in m, default 0"}},
          "additionalProperties": False}
TOOLS = [{"type": "function", "name": "look", "description": "Read-only view of the scene.", "inputSchema": SCHEMA}]


def system_text(body):
    system = body.get("system")
    if isinstance(system, list):
        return "\n".join(block.get("text", "") for block in system if isinstance(block, dict))
    return system if isinstance(system, str) else ""


def last_user_text(body):
    for message in reversed(body.get("messages") or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content, False
        texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        results = any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
        return "\n".join(texts), results
    return "", False


class Router:
    """Actor-loop requests (tools + the instructions marker): a GATE prompt is held on ``gate``; otherwise the
    first answer calls ``mcp__auriga__look`` and the answer after its tool_result is the final text. Anything else
    (side calls) gets a short text."""

    def __init__(self):
        self.gate = Gate()
        self.main = []

    def __call__(self, request):
        body = request.json if isinstance(request.json, dict) else {}
        if not (body.get("tools") and MARKER in system_text(body)):
            return text_step("side-ok")
        self.main.append(request)
        text, has_result = last_user_text(body)
        if has_result:
            return text_step("FINAL")
        if "PLAIN" in text:  # checked first: the CLI merges an interrupted prompt into the next user message
            return text_step("FINAL-2")
        if "GATE" in text:
            return Step(text="late", gate=self.gate)
        return tool_step("mcp__auriga__look", {"x": 1}, id=TOOL_USE_ID)


@unittest.skipIf(RUNTIME_PROBLEM is not None, f"Claude Code runtime unavailable: {RUNTIME_PROBLEM}")
class ClaudeCodeRealBinarySmoke(unittest.TestCase):
    def test_tool_turn_interrupt_and_close(self):
        from auriga.cortex.agents.claude_code.session import ClaudeCodeSession
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve() / "run"
        root.mkdir()
        router = Router()
        server = FakeAnthropicServer([], default=router, gate_timeout_s=60).start()
        self.addCleanup(server.close)
        self.addCleanup(router.gate.release)
        environment = mock.patch.dict(os.environ, {KEY_ENV: HOST_KEY})
        environment.start()
        self.addCleanup(environment.stop)
        store = Store(root, IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({
            "agent": {"backend": "claude_code", "claude_code": {"upstream": {
                "base_url": server.base_url, "api_key_env": KEY_ENV, "auth_scheme": "x-api-key"}}},
            "model": {"name": "claude-smoke-model", "effort": "low"},
            "features": {"self_briefing": False, "visual_review_model": None}})
        calls, disconnects = [], []

        def handler(params):
            calls.append(params)
            return {"success": True, "contentItems": [{"type": "inputText", "text": "scene: one red cube"}]}

        session = ClaudeCodeSession(root, store, handler, config=config)
        session.on_disconnect = lambda: disconnects.append(session.state)
        session.tool_parallel_hints = {"look": True}
        self.addCleanup(session.close)

        thread = session.start(TOOLS, "Robot instructions. " + MARKER)
        self.assertEqual(session.state, "idle")
        (_, process), = session.owned_processes()
        pid = process.pid

        # 1. One turn with one registry tool call ends completed with the final answer.
        turn = session.run_turn("Look at the scene, then answer.")
        self.assertTrue(session.turn_done.wait(120), "tool turn did not end")
        self.assertEqual((session.state, session.turn_result["status"]), ("completed", "completed"),
                         session.turn_result)
        self.assertEqual(session.completed_final_answer(), "FINAL")
        params, = calls
        self.assertEqual((params["threadId"], params["turnId"], params["tool"], params["callIdSource"]),
                         (thread, turn, "look", "native_tool_use_id"))
        self.assertEqual(params["callId"], TOOL_USE_ID)  # the model's tool_use id, via the CLI's MCP _meta
        row = store.transcript_item(params["callId"])
        self.assertEqual((row["type"], row["status"], row["success"]), ("dynamicToolCall", "completed", True))
        # Upstream saw the host credential (never the agent's placeholder), the configured model and the tool result.
        actor = router.main
        self.assertGreaterEqual(len(actor), 2)
        self.assertTrue(all(request.header("x-api-key") == HOST_KEY for request in server.requests))
        self.assertEqual(actor[0].json["model"], "claude-smoke-model")
        self.assertIn("scene: one red cube", " ".join(result.text for result in server.find_tool_results(actor[1])))
        self.assertEqual(store.read_state("runtime", "model_requests")["counting"],
                         "delivered_usable_generation_completions_v1")
        self.assertGreaterEqual(store.read_state("runtime", "model_usage_relay")["calls"], 2)

        # 2. An interrupt during a gated model call ends the turn as interrupted.
        session.run_turn("GATE: wait for the gate.")
        self.assertTrue(router.gate.wait_entered(60), "gated model call never arrived")
        started = time.monotonic()
        session.interrupt()
        self.assertTrue(session.turn_done.wait(45), "interrupt did not end the turn")
        self.assertLess(time.monotonic() - started, 45)
        self.assertEqual(session.state, "interrupted", session.turn_result)
        self.assertEqual(disconnects, [])

        # 3. The interrupt tombstoned the relay thread; the next turn reopens it and completes.
        session.run_turn("PLAIN: answer directly.")
        self.assertTrue(session.turn_done.wait(120), "turn after the interrupt did not end")
        self.assertEqual(session.state, "completed", session.turn_result)
        self.assertEqual(session.completed_final_answer(), "FINAL-2")

        # Both turns ended at the CLI's idle state event, never by the quiet guard.
        self.assertEqual(store.events_by_type(["agent.turn_end_fallback"]), [])
        self.assertEqual([event["payload"]["status"] for event in store.events_by_type(["agent.turn_completed"])],
                         ["completed", "interrupted", "completed"])
        commands = [record["message"]["command"] for record in map(json.loads, (root / "agent-raw.jsonl").read_text()
                    .splitlines()) if record["channel"] == "control"]
        self.assertEqual(commands, ["submit_prompt", "submit_prompt", "interrupt", "submit_prompt"])
        self.assertTrue(store.events_by_type(["model.model_thread_reopened"]))

        # 4. Close leaves no process behind.
        members = [item.pid for item in process_tree_snapshot(pid).members]
        self.assertIn(pid, members)
        report = session.close()
        self.assertTrue(report["clean_shutdown"], report)
        self.assertEqual(session.state, "closed")
        for member in members:
            with self.assertRaises(ProcessLookupError, msg=f"pid {member} survived close"):
                os.kill(member, 0)
        self.assertTrue(store.read_state("runtime", "agent")["clean_shutdown"])
        self.assertNotIn(HOST_KEY, (root / "private" / "claude-launcher.sh").read_text())


@unittest.skipIf(RUNTIME_PROBLEM is not None, f"Claude Code runtime unavailable: {RUNTIME_PROBLEM}")
class ClaudeCodeRealBinaryResume(unittest.TestCase):
    """Real CLI: clean resume, CLI killed during a tool call (disconnect, marker-verified teardown), unclean resume
    (context epoch invalidated, next turn completes), and a stdin write that lands after the turn ended becomes an
    adopted turn whose tool call carries that turn's id (review A1)."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve() / "run"
        self.root.mkdir()
        self.router = Router()
        self.server = FakeAnthropicServer([], default=self.router, gate_timeout_s=60).start()
        self.addCleanup(self.server.close)
        self.addCleanup(self.router.gate.release)
        environment = mock.patch.dict(os.environ, {KEY_ENV: HOST_KEY})
        environment.start()
        self.addCleanup(environment.stop)
        self.store = Store(self.root, IDENTITY)
        self.addCleanup(self.store.close)
        self.calls, self.tool_gate = [], threading.Event()
        self.tool_gate.set()
        self.addCleanup(self.tool_gate.set)

    def handler(self, params):
        self.calls.append(params)
        self.tool_gate.wait(60)
        return {"success": True, "contentItems": [{"type": "inputText", "text": "scene: one red cube"}]}

    def session(self):
        from auriga.cortex.agents.claude_code.session import ClaudeCodeSession
        config = Config.from_dict({
            "agent": {"backend": "claude_code", "claude_code": {"upstream": {
                "base_url": self.server.base_url, "api_key_env": KEY_ENV, "auth_scheme": "x-api-key"}}},
            "model": {"name": "claude-smoke-model", "effort": "low"},
            "features": {"self_briefing": False, "visual_review_model": None}})
        session = ClaudeCodeSession(self.root, self.store, self.handler, config=config)
        session.disconnects, session.resets = [], []
        session.on_disconnect = lambda: session.disconnects.append(session.state)
        session.on_context_reset = session.resets.append
        session.tool_parallel_hints = {"look": True}
        self.addCleanup(session.close)
        return session

    def wait(self, predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return predicate()

    def test_resume_crash_and_late_write(self):
        first = self.session()
        thread = first.start(TOOLS, "Robot instructions. " + MARKER)
        first.run_turn("PLAIN: answer directly.")
        self.assertTrue(first.turn_done.wait(120))
        self.assertEqual(first.state, "completed")

        # A stdin write after the turn ended (what a racing steer or a late prompt does): adopted, owns its tool.
        completed = first.turn_id
        first._call(first._client.query("Look at the scene, then answer."), 15)
        self.assertTrue(self.wait(lambda: first.turn_id != completed and first.turn_done.is_set()))
        adopted = first.turn_id
        self.assertEqual(first.state, "completed", first.turn_result)
        self.assertEqual(self.calls[-1]["turnId"], adopted)
        self.assertEqual([event["payload"]["origin"] for event in self.store.events_by_type(["agent.turn_started"])],
                         ["host", "native_wakeup"])
        self.assertTrue(first.close()["clean_shutdown"])

        # Clean resume.
        second = self.session()
        self.assertEqual(second.start(TOOLS, "Robot instructions. " + MARKER, thread), thread)
        self.assertEqual(second.resets, [])
        second.run_turn("PLAIN: answer directly.")
        self.assertTrue(second.turn_done.wait(120))
        self.assertEqual((second.state, second.completed_final_answer()), ("completed", "FINAL-2"))

        # The CLI dies during a tool call: one disconnect, synthetic failed turn; teardown verified by the marker.
        self.tool_gate.clear()
        count = len(self.calls)
        second.run_turn("Look at the scene, then answer.")
        self.assertTrue(self.wait(lambda: len(self.calls) > count))
        os.kill(second.owned_processes()[0][1].pid, signal.SIGKILL)
        self.assertTrue(second.turn_done.wait(30))
        self.assertEqual((second.state, second.turn_result["status"], second.disconnects),
                         ("disconnected", "failed", ["disconnected"]))
        self.tool_gate.set()
        report = second.close()
        self.assertFalse(report["clean_shutdown"])
        self.assertTrue(report["backend"]["agent_exited_before_teardown"])
        self.assertTrue(report["backend"]["agent_processes"]["confirmed"], report["backend"])
        self.assertFalse(self.store.read_state("runtime", "agent")["clean_shutdown"])

        # Unclean resume: epoch invalidated, reminder scheduled, the next turn completes.
        third = self.session()
        third.start(TOOLS, "Robot instructions. " + MARKER, thread)
        self.assertEqual(third.resets, ["agent_resume_after_unclean_exit"])
        self.assertEqual(self.store.read_state("runtime", "image_context_epoch")["source"],
                         "agent_resume_after_unclean_exit")
        third.run_turn("PLAIN: answer directly.")
        self.assertTrue(third.turn_done.wait(120))
        self.assertEqual(third.state, "completed", third.turn_result)
        self.assertTrue(third.close()["clean_shutdown"])

if __name__ == "__main__":
    unittest.main()
