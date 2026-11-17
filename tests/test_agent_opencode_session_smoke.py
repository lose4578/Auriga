"""One real-binary smoke run of OpenCodeSession (plan 07 §4; the full conformance suite is package T12).

Drives the pinned ``opencode serve`` 1.19.5 named by ``AURIGA_OPENCODE_EXECUTABLE`` through the real relay
(ModelRequestProxy + AnthropicMessagesWire with the ``opencode`` profile, upstream = FakeAnthropicServer on loopback)
and the real AurigaMcpServer: start -> one turn with one registry tool call -> completed with the final answer; a steer
during a gated model call joins the turn; an interrupt during a gated model call ends it interrupted and the next turn
reopens the relay thread; close during a running native bash call leaves no process (the bash tool's shell is
detached into its own process group). No real model, no network: OpenCode's XDG dirs, HOME and TMPDIR are inside the
temporary run directory and its only credential is the relay placeholder. Skipped (with the reason) unless the
variable names the pinned binary and ``rg`` is installed.
"""
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from support.fake_model_servers import FakeAnthropicServer, Gate, Step, text_step, tool_step

from auriga.core.config import Config
from auriga.core.storage.store import Store
from auriga.cortex.agents import factory
from auriga.cortex.agents.isolation import process_tree_snapshot

EXECUTABLE = os.environ.get("AURIGA_OPENCODE_EXECUTABLE")
PROBLEM = (factory.opencode_runtime_problem(EXECUTABLE) if EXECUTABLE else
           "set AURIGA_OPENCODE_EXECUTABLE to the opencode-ai@1.19.5 platform binary")
if PROBLEM is None and shutil.which("rg") is None:
    PROBLEM = "ripgrep (rg) is not installed"
IDENTITY = dict(run_id="smoke", episode_id="ep", env_instance_id="env", env_generation=1)
KEY_ENV = "AURIGA_SMOKE_OPENCODE_UPSTREAM_KEY"
HOST_KEY = "host-secret-opencode-smoke-key"
MARKER = "AURIGA-OPENCODE-SMOKE-INSTRUCTIONS-5c1d"
SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x position in m, default 0"}},
          "additionalProperties": False}
TOOLS = [{"type": "function", "name": "look", "description": "Read-only view of the scene.", "inputSchema": SCHEMA}]
DETACH_COMMAND = ("perl -e 'sleep 600' >/dev/null 2>&1 & echo $! > orphan.pid; "
                  "perl -MPOSIX -e 'POSIX::setsid(); sleep 600' >/dev/null 2>&1 & echo $! > detached.pid")


def system_text(body):
    system = body.get("system")
    if isinstance(system, list):
        return "\n".join(block.get("text", "") for block in system if isinstance(block, dict))
    return system if isinstance(system, str) else ""


def last_user(body):
    """``(text, has_tool_result)`` of the last user message of an Anthropic Messages body."""
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
    """Actor-loop requests (tools declared + the instructions marker in the system text) are routed on the last user
    message; anything else (none expected: the title agent is disabled) gets a short text."""

    def __init__(self):
        self.gate_steer, self.gate_abort = Gate(), Gate()
        self.main = []

    def __call__(self, request):
        body = request.json if isinstance(request.json, dict) else {}
        if not (body.get("tools") and MARKER in system_text(body)):
            return text_step("side-ok")
        self.main.append(request)
        text, has_result = last_user(body)
        if has_result:
            return text_step("FINAL")
        if "STEER-NOW" in text:
            return text_step("FINAL-STEER")
        if "PLAIN" in text:        # checked before GATE-A: the aborted prompt merges into the next user message
            return text_step("FINAL-2")
        if "DETACH" in text:       # conformance item 15 (b): two children outlive the shell that started them
            return tool_step("bash", {"command": DETACH_COMMAND, "description": "Leave two sleepers behind"})
        if "BASH" in text:
            return tool_step("bash", {"command": "sleep 600", "description": "Wait ten minutes", "timeout": 600000})
        if "GATE-S" in text:
            return Step(text="late answer", gate=self.gate_steer)
        if "GATE-A" in text:
            return Step(text="never delivered", gate=self.gate_abort)
        return tool_step("auriga_look", {"x": 1})


def bodies_text(request):
    return json.dumps(request.json)


@unittest.skipIf(PROBLEM is not None, f"OpenCode runtime unavailable: {PROBLEM}")
class OpenCodeRealBinarySmoke(unittest.TestCase):
    def wait(self, predicate, timeout, message):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail(message)

    def test_tool_turn_steer_interrupt_and_close(self):
        from auriga.cortex.agents.opencode.session import OpenCodeSession
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve() / "run"
        root.mkdir()
        router = Router()
        server = FakeAnthropicServer([], default=router, gate_timeout_s=60).start()
        self.addCleanup(server.close)
        self.addCleanup(router.gate_steer.release)
        self.addCleanup(router.gate_abort.release)
        environment = mock.patch.dict(os.environ, {KEY_ENV: HOST_KEY})
        environment.start()
        self.addCleanup(environment.stop)
        store = Store(root, IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({
            "agent": {"backend": "opencode", "opencode": {"executable": EXECUTABLE, "upstream": {
                "wire_api": "anthropic_messages", "base_url": server.base_url, "api_key_env": KEY_ENV,
                "auth_scheme": "x-api-key", "provider": "fake"}}},
            "model": {"name": "claude-smoke-model", "effort": "high"},
            "features": {"self_briefing": False, "visual_review_model": None}})
        self.assertIsNone(factory.backend_requirements(config))
        calls, disconnects = [], []

        def handler(params):
            calls.append(params)
            return {"success": True, "contentItems": [{"type": "inputText", "text": "scene: one red cube"}]}

        session = factory.create_session(config, root, store, handler)
        self.assertIsInstance(session, OpenCodeSession)
        session.on_disconnect = lambda: disconnects.append(session.state)
        session.tool_parallel_hints = {"look": True}
        self.addCleanup(session.close)

        thread = session.start(TOOLS, "Robot instructions. " + MARKER)
        self.assertEqual(session.state, "idle")
        self.assertTrue(thread.startswith("ses"))
        (role, process), = session.owned_processes()
        self.assertEqual(role, "opencode")
        identity = store.read_state("runtime", "agent")
        self.assertEqual((identity["backend"], identity["thread_id"], identity["cli_version"], identity["wire_api"]),
                         ("opencode", thread, "1.19.5", "anthropic_messages"))
        provenance = session.provenance()
        self.assertIsNotNone(provenance["route_list_sha256"])
        enabled = provenance["native_tools"]["enabled"]           # read back from the running server
        self.assertTrue({"bash", "read", "task"} <= set(enabled), enabled)
        self.assertFalse({"webfetch", "websearch", "codesearch", "question"} & set(enabled), enabled)

        # 1. One turn with one registry tool call ends completed with the final answer.
        turn = session.run_turn("Look at the scene, then answer.")
        self.assertTrue(session.turn_done.wait(120), "tool turn did not end")
        self.assertEqual((session.state, session.turn_result["status"]), ("completed", "completed"),
                         session.turn_result)
        self.assertEqual(session.completed_final_answer(), "FINAL")
        params, = calls
        self.assertEqual((params["threadId"], params["turnId"], params["tool"], params["callIdSource"]),
                         (thread, turn, "look", "mcp_request"))
        row = store.transcript_item(params["callId"])
        self.assertEqual((row["type"], row["status"], row["success"]), ("dynamicToolCall", "completed", True))
        self.assertTrue(row.get("native_id"), row)        # the projector attached OpenCode's callID
        # Only relayed model requests carry the host key; stray loopback probes from unrelated local tools (seen on
        # Linux: keyless `GET /`) can reach the fake upstream port and must not fail the test.
        self.assertTrue(all(request.header("x-api-key") == HOST_KEY for request in server.generation_requests()))
        self.assertEqual(router.main[0].json["model"], "claude-smoke-model")
        self.assertIn("scene: one red cube", " ".join(result.text for result in server.find_tool_results(router.main[1])))
        self.assertNotIn("Instructions from:", bodies_text(router.main[0]))   # egress neutralises the file path
        self.assertEqual(store.read_state("runtime", "model_requests")["counting"],
                         "delivered_usable_generation_completions_v1")
        self.assertGreaterEqual(store.read_state("runtime", "model_usage_relay")["calls"], 2)

        # 2. A steer during a gated model call joins the running turn; the turn completes once, with its answer.
        steered = session.run_turn("GATE-S: wait for the gate.")
        self.assertTrue(router.gate_steer.wait_entered(60), "gated model call never arrived")
        self.assertEqual(session.steer("STEER-NOW: answer the steer."), steered)
        router.gate_steer.release()
        self.assertTrue(session.turn_done.wait(120), "steered turn did not end")
        self.assertEqual(session.state, "completed", session.turn_result)
        self.assertEqual(session.completed_final_answer(), "FINAL-STEER")
        self.assertTrue(any("STEER-NOW" in last_user(request.json)[0] for request in router.main))
        steered_rows = [row for row in store.states(store.TRANSCRIPT_KIND)
                        if row.get("turn_id") == steered and row["type"] == "userMessage"]
        self.assertEqual([row.get("steer", False) for row in steered_rows], [False, True])
        self.assertTrue(all(row.get("native_id") for row in steered_rows), steered_rows)

        # 3. An interrupt during a gated model call ends the turn interrupted; the next turn reopens the relay thread.
        session.run_turn("GATE-A: wait for the gate.")
        self.assertTrue(router.gate_abort.wait_entered(60), "gated model call never arrived")
        started = time.monotonic()
        session.interrupt()
        self.assertTrue(session.turn_done.wait(45), "interrupt did not end the turn")
        self.assertLess(time.monotonic() - started, 45)
        self.assertEqual(session.state, "interrupted", session.turn_result)
        session.run_turn("PLAIN: answer directly.")
        self.assertTrue(session.turn_done.wait(120), "turn after the interrupt did not end")
        self.assertEqual(session.state, "completed", session.turn_result)
        self.assertEqual(session.completed_final_answer(), "FINAL-2")
        self.assertTrue(store.events_by_type(["model.model_thread_reopened"]))
        self.assertEqual([event["payload"]["status"] for event in store.events_by_type(["agent.turn_completed"])],
                         ["completed", "completed", "interrupted", "completed"])
        self.assertEqual(disconnects, [])

        # 4. Close while the native bash tool runs (its shell is detached into its own group) leaves no process.
        session.run_turn("BASH: wait in the shell.")
        server_pgid = os.getpgid(process.pid)

        def detached_shell():
            members = process_tree_snapshot(process.pid).members
            return [item for item in members if item.pgid != server_pgid] or None

        self.wait(detached_shell, 60, "the bash tool's shell never started")
        members = [item.pid for item in process_tree_snapshot(process.pid).members]
        report = session.close()
        self.assertEqual(session.state, "closed")
        self.assertFalse(report["clean_shutdown"])        # a turn was running at close
        self.assertTrue(report["backend"]["agent_processes"]["confirmed"], report)
        for member in members:
            with self.assertRaises(ProcessLookupError, msg=f"pid {member} survived close"):
                os.kill(member, 0)
        self.assertEqual(session.owned_processes(), [])
        self.assertEqual(disconnects, [])
        self.assertFalse(store.read_state("runtime", "agent")["clean_shutdown"])
        commands = [record["message"]["command"] for record in map(json.loads, (root / "agent-raw.jsonl").read_text()
                    .splitlines()) if record["channel"] == "control"]
        self.assertEqual(commands, ["submit_prompt", "submit_prompt", "steer", "submit_prompt", "interrupt",
                                    "submit_prompt", "submit_prompt"])
        self.assertNotIn(HOST_KEY, (root / "private" / "agent.log").read_text(errors="replace"))


@unittest.skipIf(PROBLEM is not None, f"OpenCode runtime unavailable: {PROBLEM}")
@unittest.skipIf(shutil.which("perl") is None, "perl is not installed")
class OpenCodeDetachedChildren(unittest.TestCase):
    """Conformance item 15 (b) with the real server: a bash call leaves ``cmd &`` and a setsid child behind (their
    shell exits at once, both are re-parented away from the server's tree); close kills both and reports a clean,
    confirmed teardown only after the marker re-scan verified it."""

    def test_close_kills_children_a_bash_call_left_behind(self):
        from auriga.cortex.agents.opencode.session import OpenCodeSession
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve() / "run"
        root.mkdir()
        server = FakeAnthropicServer([], default=Router(), gate_timeout_s=60).start()
        self.addCleanup(server.close)
        environment = mock.patch.dict(os.environ, {KEY_ENV: HOST_KEY})
        environment.start()
        self.addCleanup(environment.stop)
        store = Store(root, IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({
            "agent": {"backend": "opencode", "opencode": {"executable": EXECUTABLE, "upstream": {
                "wire_api": "anthropic_messages", "base_url": server.base_url, "api_key_env": KEY_ENV,
                "auth_scheme": "x-api-key", "provider": "fake"}}},
            "model": {"name": "claude-smoke-model", "effort": "high"},
            "features": {"self_briefing": False, "visual_review_model": None}})
        session = factory.create_session(config, root, store, lambda params: {"success": True, "contentItems": []})
        self.assertIsInstance(session, OpenCodeSession)
        self.addCleanup(session.close)
        session.start(TOOLS, "Robot instructions. " + MARKER)
        session.run_turn("DETACH: leave two sleepers.")
        self.assertTrue(session.turn_done.wait(120), "the bash turn did not end")
        self.assertEqual(session.state, "completed", session.turn_result)
        actor = session.actor_dir
        children = {name: int((actor / f"{name}.pid").read_text().strip()) for name in ("orphan", "detached")}
        for pid in children.values():
            self.addCleanup(lambda pid=pid: _kill_quietly(pid))
        self.assertTrue(all(_alive(pid) for pid in children.values()), children)
        report = session.close()
        survivors = {name: pid for name, pid in children.items() if _alive(pid)}
        self.assertEqual(survivors, {}, f"bash children survived close: {report['backend']}")
        self.assertTrue(report["backend"]["agent_processes"]["confirmed"], report["backend"])
        self.assertTrue(report["clean_shutdown"], report)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill_quietly(pid):
    try:
        os.kill(pid, 9)
    except OSError:
        pass


if __name__ == "__main__":
    unittest.main()
