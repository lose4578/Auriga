"""Conformance item 28 (sandbox boundary) for the Codex backend (design 04 §10.2; WP R4-8).

Codex has no binding of the shared ``AgentBackendConformance`` mixin (its contract is covered by the existing Codex
suites), so this module runs item 28 alone with the same fixture (``support.agent_conformance.SandboxBoundary``): the
real pinned Codex app-server in the real bubblewrap ``CodexSandbox``, the real relay, and a scripted loopback
Responses upstream (``FakeResponsesServer``; no real model, no network). One turn: the model runs the in-namespace
probe with the native ``exec_command`` shell tool, calls the registry tool ``look`` (a Codex dynamic tool), then
answers. Checked: (a) /harness holds the published source snapshot, read-only; (b) the checkout, the user's home and
a planted RoboDojo root are unreadable; (c) only the forwarded loopback relay is reachable; (d) the tool round trip;
(e) close leaves no bwrap / supervisor / app-server / detached child and removes the IPC directory; (f) the recorded
isolation. Linux with bubblewrap only (``requires_bwrap``): skipped with the probe's reason elsewhere, a failure with
``AURIGA_REQUIRE_SANDBOX_TESTS=1``.
"""
import tempfile
import unittest
from pathlib import Path

from support.agent_conformance import TOOLS, SANDBOX_PROBE_MARKER, SandboxBoundary
from support.fake_model_servers import FakeResponsesServer, Step, ToolCall, text_step
from support.requirements import requires_bwrap, requires_codex_binary

from auriga.core.config import Config
from auriga.core.storage.store import Store

PROBE_ID, LOOK_ID = "call_sandbox_probe_28", "call_sandbox_look_28"
EXEC_YIELD_MS = 20000      # exec_command yield time, ms: the probe finishes in about a second
INSTRUCTIONS = "Codex sandbox conformance (item 28). Use the tools when asked and answer briefly."


class CodexResponsesServer(FakeResponsesServer):
    """``FakeResponsesServer`` whose ``exec_command`` calls carry Codex's ``functions`` namespace (as the native
    shell tool is published to the model)."""

    def _items(self, turn):
        items = super()._items(turn)
        for item in items:
            if item.get("type") == "function_call" and item.get("name") == "exec_command":
                item["namespace"] = "functions"
        return items


def _output_ids(request):
    body = request.json if isinstance(request.json, dict) else {}
    return {item.get("call_id") for item in body.get("input") or ()
            if isinstance(item, dict) and item.get("type") == "function_call_output"}


@requires_codex_binary
@requires_bwrap
class CodexSandboxConformance(unittest.TestCase):
    def test_sandbox_namespace(self):
        from auriga.cortex.agents.factory import backend_requirements
        from auriga.cortex.agents.isolation import process_tree_snapshot
        from auriga.cortex.session.agent_sandbox import ACTOR_ISOLATION
        from auriga.cortex.session.codex import CodexSession
        directory = tempfile.TemporaryDirectory(prefix="auriga-c28-codex-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve() / "run"
        root.mkdir()
        boundary = SandboxBoundary(self, root)          # before the session: it fixes XDG_RUNTIME_DIR
        probe_command = []

        def router(request):
            outputs = _output_ids(request)
            if LOOK_ID in outputs:
                return text_step("SANDBOX-DONE-28")
            if PROBE_ID in outputs:
                return Step(tool_calls=[ToolCall("look", {"x": 1.5}, LOOK_ID)])
            if not probe_command:
                return text_step("side-ok")
            return Step(tool_calls=[ToolCall("exec_command", dict(cmd=probe_command[0], yield_time_ms=EXEC_YIELD_MS,
                                                                  max_output_tokens=2000), PROBE_ID)])

        server = CodexResponsesServer([], default=router, gate_timeout_s=60).start()
        self.addCleanup(server.close)
        source = root.parent / "codex-source"
        source.mkdir()
        (source / "config.toml").write_text(
            'model_provider="fixture"\n[model_providers.fixture]\nname="Loopback fixture"\n'
            f'base_url="{server.base_url}/v1"\nwire_api="responses"\nrequires_openai_auth=false\n'
            'supports_websockets=false\n')
        store = Store(root, dict(run_id="codex-sandbox-28", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        config = Config.from_dict({"model": {"codex_home": str(source), "native_memories": False},
                                   "features": {"sandbox_enabled": True, "self_briefing": False,
                                                "visual_review_model": None}})
        self.assertIsNone(backend_requirements(config))
        calls = []

        def handler(params):
            calls.append(params)
            return {"success": True, "contentItems": [{"type": "inputText", "text": "scene: one red cube at x=1.5"}]}

        session = CodexSession(root, store, handler, config=config)
        closed = []

        def close():
            if closed:
                return
            closed.append(True)
            bridge = session.client._proc if session.client else None
            session.close()
            if bridge:
                for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                    if stream and not stream.closed:
                        stream.close()
        self.addCleanup(close)
        session.start([TOOLS[0]], INSTRUCTIONS)
        # (f) recorded isolation
        self.assertEqual(session.isolation, ACTOR_ISOLATION)
        self.assertEqual(store.read_state("runtime", "agent")["isolation"], ACTOR_ISOLATION)
        self.assertEqual(store.read_state("runtime", "codex")["isolation"], ACTOR_ISOLATION)
        self.assertEqual(session.provenance()["isolation"], ACTOR_ISOLATION)
        relay_port = int(session.model_proxy.base_url.rsplit(":", 1)[1].split("/")[0])
        boundary.write_probe(session.actor_dir, [relay_port])
        probe_command.append(boundary.command)
        session.run_turn("Probe the sandbox, then look at the scene.")
        self.assertTrue(session.turn_done.wait(180), "the sandboxed turn did not end")
        self.assertEqual(session.state, "completed", session.turn_result)
        # (a)-(c) measured inside the namespace by the native shell tool
        self.assertTrue(boundary.wait_for(lambda: (session.actor_dir / boundary.report_name).is_file(), 30))
        outputs = [result.text for request in server.generation_requests() for result in server.find_tool_results(request)
                   if result.call_id == PROBE_ID]
        self.assertTrue(any(SANDBOX_PROBE_MARKER in text for text in outputs), outputs)
        boundary.assert_inside(boundary.report(), root / "private" / "actor-source")
        # (d) the registry tool round trip through the dynamic-tool path and the relay
        (call,) = calls
        self.assertEqual((call["threadId"], call["tool"], call["arguments"]), (session.thread_id, "look", {"x": 1.5}))
        looked = [result.text for request in server.generation_requests() for result in server.find_tool_results(request)
                  if result.call_id == LOOK_ID]
        self.assertTrue(any("scene: one red cube at x=1.5" in text for text in looked), looked)
        # (e) teardown
        tree = [member.pid for _role, process in session.owned_processes()
                for member in process_tree_snapshot(process.pid).members]
        tracked = boundary.assert_running(tree)
        close()
        boundary.assert_closed(tracked)


if __name__ == "__main__":
    unittest.main()
