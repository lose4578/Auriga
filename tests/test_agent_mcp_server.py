"""AurigaMcpServer + encode_mcp_result (design 04 §3.2-§3.4, §5.3; loopback only, no binaries, no model)."""
import base64
import hashlib
import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

from support.fake_mcp_client import FakeMcpClient, parse_sse

from auriga.apps.evaluation.native_evidence import load_native_evidence
from auriga.core.config import Config
from auriga.core.presentation.actor_presentation import ActorPresentation
from auriga.core.storage.store import Store
from auriga.cortex.agents.dispatch import ToolDispatcher
from auriga.cortex.agents.mcp_server import (DEFAULT_RESULT_SIZE_HINT_CHARS, SUPPORTED_PROTOCOL_VERSIONS,
                                             AurigaMcpServer, encode_mcp_result, mcp_tool_catalog,
                                             portable_input_schema)

IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
TIMEOUT = 5
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
PNG_B64 = base64.b64encode(PNG).decode()
PLANTED_HASH = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x in m, default 0"}}}
TOOLS = [
    {"type": "function", "name": "look", "description": "Read-only view. build sha256: " + PLANTED_HASH,
     "inputSchema": SCHEMA},
    {"type": "function", "name": "move", "description": "Stateful motion.", "inputSchema": SCHEMA,
     "deferLoading": True},
    {"type": "function", "name": "mixed", "description": "Request dependent.", "inputSchema": SCHEMA},
    {"type": "function", "name": "plain", "description": "No hint.", "inputSchema": SCHEMA},
]
HINTS = {"look": True, "move": False, "mixed": "request_dependent"}


def text_result(text, success=True):
    return {"success": success, "contentItems": [{"type": "inputText", "text": text}]}


def wait_until(predicate, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeCallTool:
    """Stands in for the session's §3.3 closure: records calls, optional gates, scripted results."""

    def __init__(self):
        self.calls, self.started, self.finished = [], {}, {}
        self.gates, self.results, self.errors = {}, {}, {}
        self.lock = threading.Lock()

    def gate(self, name):
        self.gates[name] = threading.Event()
        return self.gates[name]

    def __call__(self, name, arguments, *, call_id, native_call_id, rpc_id):
        with self.lock:
            self.calls.append(dict(name=name, arguments=arguments, call_id=call_id, native_call_id=native_call_id,
                                   rpc_id=rpc_id))
            self.started.setdefault(name, threading.Event()).set()
        try:
            if name in self.gates:
                self.gates[name].wait(TIMEOUT * 2)
            if name in self.errors:
                raise self.errors[name]
            return self.results.get(name, text_result("ok:" + name))
        finally:
            with self.lock:
                self.finished.setdefault(name, threading.Event()).set()


class ServerCase(unittest.TestCase):
    server_kwargs = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name, IDENTITY)
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)
        self.fake = FakeCallTool()
        self.failures = []
        self.server = self.make_server(**self.server_kwargs)
        self.client = FakeMcpClient(self.server.url, token=self.server.token, protocol_version="2025-11-25")

    def make_server(self, **kwargs):
        options = dict(tools=TOOLS, presentation=self.presentation, call_tool=self.fake,
                       raw_log=lambda direction, message: self.store.raw_agent(direction, message, channel="mcp"),
                       on_delivery_failure=lambda call_id, reason: self.failures.append((call_id, reason)),
                       parallel_hints=HINTS, keepalive_s=0.05, peer_check_s=0.02)
        options.update(kwargs)
        server = AurigaMcpServer(**options).start()
        self.addCleanup(server.close)
        return server

    def raw_records(self):
        path = self.store.root / "agent-raw.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ---------------------------------------------------------------------------------------------------------------------
class HandshakeTest(ServerCase):
    def test_claude_code_opening_sequence(self):
        probe = self.client.discover()
        self.assertEqual(probe.status, 200)
        self.assertEqual(probe.messages[0], {"jsonrpc": "2.0", "id": "server-discover-probe-1",
                                             "error": {"code": -32601, "message": "Method not found"}})
        init = self.client.initialize()
        self.assertEqual(init["protocolVersion"], "2025-11-25")
        self.assertEqual(init["capabilities"], {"tools": {"listChanged": False}})
        self.assertEqual(init["serverInfo"]["name"], "auriga")
        self.assertIsInstance(init["serverInfo"]["version"], str)
        self.assertNotIn("instructions", init)
        self.assertEqual(self.client.initialized().status, 202)
        stream = self.client.open_standalone_stream()
        self.assertEqual((stream.status, stream.headers.get("Allow")), (405, "POST"))
        self.assertEqual(self.client.http("DELETE").status, 405)
        self.assertEqual(self.client.ping(), {})
        self.assertEqual(len(self.client.list_tools()), len(TOOLS))

    def test_protocol_negotiation(self):
        for offered in SUPPORTED_PROTOCOL_VERSIONS:
            client = FakeMcpClient(self.server.url, token=self.server.token, protocol_version=offered)
            self.assertEqual(client.initialize()["protocolVersion"], offered)
        for offered in ("2099-01-01", None):
            client = FakeMcpClient(self.server.url, token=self.server.token, protocol_version=offered)
            self.assertEqual(client.initialize()["protocolVersion"], "2025-11-25")

    def test_unknown_method_parse_error_batch_and_notifications(self):
        response = self.client.request("resources/list", {})
        self.assertEqual(response.messages[0]["error"]["code"], -32601)
        self.assertEqual(self.client.post(b"{not json").status, 400)
        self.assertEqual(self.client.last_response.messages[0]["error"]["code"], -32700)
        batch = self.client.post(json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]).encode())
        self.assertEqual((batch.status, batch.messages[0]["error"]["code"]), (400, -32600))
        self.assertEqual(self.client.notify("notifications/roots/list_changed").status, 202)
        self.assertEqual(self.client.post({"jsonrpc": "2.0", "id": 4, "result": {}}).status, 202)
        self.assertEqual(self.client.request("ping", request_id=True).status, 400)


class CatalogTest(ServerCase):
    def test_list_tools_scrub_annotations_and_meta(self):
        tools = {tool["name"]: tool for tool in self.client.list_tools()}
        self.assertEqual(list(tools), ["look", "move", "mixed", "plain"])
        for name, tool in tools.items():
            self.assertEqual(set(tool), {"name", "description", "inputSchema", "annotations", "_meta"})
            self.assertEqual(tool["_meta"], {"anthropic/maxResultSizeChars": DEFAULT_RESULT_SIZE_HINT_CHARS})
        self.assertEqual({n: t["annotations"]["readOnlyHint"] for n, t in tools.items()},
                         {"look": True, "move": False, "mixed": False, "plain": False})
        self.assertNotIn(PLANTED_HASH, json.dumps(tools))
        self.assertEqual(tools["look"]["description"], self.presentation.present_text(TOOLS[0]["description"]))
        self.assertEqual(tools["move"]["inputSchema"], self.presentation.present_data(SCHEMA))
        self.assertEqual(self.server.read_only_tools, ["look"])

    def test_top_level_combinators_are_removed_and_nested_ones_stay(self):
        nested = {"type": "object", "anyOf": [{"required": ["a"]}], "properties": {"a": {"type": "string"}}}
        schema = {"type": "object", "properties": {"op": {"type": "string"}, "n": nested}, "required": [],
                  "allOf": [{"if": {"required": ["op"]}}], "anyOf": [{"required": ["op"]}], "oneOf": [{}],
                  "if": {"required": ["op"]}, "then": {"required": ["n"]}, "else": {}, "additionalProperties": False}
        tool = {"name": "t", "description": "d", "inputSchema": schema}
        published = mcp_tool_catalog([tool], self.presentation)[0]["inputSchema"]
        self.assertEqual(published, {"type": "object", "properties": {"op": {"type": "string"}, "n": nested},
                                     "required": [], "additionalProperties": False})
        self.assertIn("allOf", schema)   # the registry definition is not mutated

    def test_task_plan_is_published_without_top_level_allof_but_rules_stay_in_prose(self):
        from auriga.runtime.tools.builtins import create_registry
        config = Config()
        runtime = SimpleNamespace(config=config, disabled_tools=config.features.disabled_tools,
                                  observation_profile=config.features.observation_profile,
                                  motion_review=config.features.motion_review, enable_wait=False,
                                  vision_compare=config.features.vision_compare, visual_reviewer=None,
                                  action_policy=None, capabilities=None, programs=True)
        tools = create_registry(runtime=runtime).catalog(runtime, role="actor")
        plan = next(tool for tool in tools if tool["name"] == "task_plan")
        self.assertIn("allOf", plan["inputSchema"])
        published = next(tool for tool in mcp_tool_catalog(tools, self.presentation) if tool["name"] == "task_plan")
        self.assertFalse({"anyOf", "oneOf", "allOf", "if", "then", "else"} & set(published["inputSchema"]))
        self.assertEqual(published["description"], self.presentation.present_text(plan["description"]))
        self.assertIn("Object rules:", published["description"])

    def test_custom_size_hint_and_invalid_definitions(self):
        catalog = mcp_tool_catalog(TOOLS[:1], self.presentation, result_size_hint_chars=123)
        self.assertEqual(catalog[0]["_meta"], {"anthropic/maxResultSizeChars": 123})
        self.assertFalse(catalog[0]["annotations"]["readOnlyHint"])
        for bad in ([{"name": "x"}], [TOOLS[0], TOOLS[0]]):
            with self.assertRaises(ValueError):
                mcp_tool_catalog(bad, self.presentation)
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                mcp_tool_catalog(TOOLS, self.presentation, result_size_hint_chars=size)

    def test_registry_catalog_matches_codex_dynamic_tools_presentation(self):
        from auriga.cortex.session.codex import DYNAMIC_TOOL_NAMESPACE, _native_dynamic_tools
        from auriga.cortex.session.wire.openai_responses import present_responses_request
        from auriga.runtime.tools.builtins import create_registry
        config = Config()
        runtime = SimpleNamespace(config=config, disabled_tools=config.features.disabled_tools,
                                  observation_profile=config.features.observation_profile,
                                  motion_review=config.features.motion_review, enable_wait=False,
                                  vision_compare=config.features.vision_compare, visual_reviewer=None,
                                  action_policy=None, capabilities=None, programs=True)
        registry = create_registry(runtime=runtime)
        tools = registry.catalog(runtime, role="actor")
        manifest = registry.manifest(runtime)
        hints = {tool["name"]: tool["parallel"] for tool in manifest["tools"]}
        catalog = mcp_tool_catalog(tools, self.presentation, parallel_hints=hints)
        deferred = {tool["name"] for tool in tools if tool.get("deferLoading")}
        codex = present_responses_request(dict(tools=_native_dynamic_tools(tools)), self.presentation,
                                          deferred_tools=deferred, namespace=DYNAMIC_TOOL_NAMESPACE)["tools"]
        flat = {tool["name"]: tool for entry in codex for tool in (entry.get("tools") or [entry])}
        self.assertEqual({tool["name"] for tool in catalog}, set(flat))
        for tool in catalog:
            self.assertEqual(tool["description"], flat[tool["name"]]["description"])
            # the MCP projection differs from the Codex schema only by the removed top-level combinators
            self.assertEqual(tool["inputSchema"], portable_input_schema(flat[tool["name"]]["inputSchema"]))
            self.assertEqual(tool["annotations"]["readOnlyHint"], hints[tool["name"]] is True)
        self.assertEqual({t["name"] for t in catalog if t["annotations"]["readOnlyHint"]},
                         {name for name, value in hints.items() if value is True})


class CallTest(ServerCase):
    def test_json_call_with_native_id(self):
        result = self.client.call_tool("look", {"x": 1}, meta={"claudecode/toolUseId": "toolu_1"},
                                       accept_sse=False)
        self.assertFalse(result.is_sse)
        self.assertEqual(result.result, {"content": [{"type": "text", "text": "ok:look"}], "isError": False})
        self.assertEqual(self.fake.calls, [dict(name="look", arguments={"x": 1}, call_id="toolu_1",
                                                native_call_id="toolu_1", rpc_id=result.request_id)])
        self.assertEqual(self.failures, [])

    def test_sse_call_with_progress_and_generated_call_id(self):
        self.fake.gate("move")
        timer = threading.Timer(0.4, self.fake.gates["move"].set)
        timer.start()
        self.addCleanup(timer.cancel)
        result = self.client.call_tool("move", {}, progress_token="tok-7")
        self.assertTrue(result.is_sse)
        self.assertGreaterEqual(len(result.progress), 3)
        self.assertEqual([p["progress"] for p in result.progress], list(range(1, len(result.progress) + 1)))
        self.assertEqual({p["progressToken"] for p in result.progress}, {"tok-7"})
        self.assertTrue(all(not frame["comments"] for frame in result.response.frames))
        self.assertEqual(result.response.frames[-1]["event"], "message")
        self.assertEqual(result.result["content"][0]["text"], "ok:move")
        call = self.fake.calls[0]
        self.assertIsNone(call["native_call_id"])
        self.assertTrue(call["call_id"].startswith("call-"))

    def test_sse_without_progress_token_uses_comments(self):
        self.fake.gate("move")
        timer = threading.Timer(0.3, self.fake.gates["move"].set)
        timer.start()
        self.addCleanup(timer.cancel)
        result = self.client.call_tool("move", {})
        self.assertEqual(result.progress, [])
        self.assertGreaterEqual(sum(frame["comments"] == ["keepalive"] for frame in result.response.frames), 2)
        self.assertEqual(result.result["content"][0]["text"], "ok:move")

    def test_call_tool_exception_becomes_presented_error_result(self):
        self.fake.errors["move"] = ValueError("native_tool_call_stale_connection")
        result = self.client.call_tool("move", {"x": 2}, progress_token=1)
        self.assertTrue(result.result["isError"])
        payload = json.loads(result.result["content"][0]["text"])
        self.assertEqual(payload["error"], "native_tool_call_stale_connection")
        self.assertEqual(payload["tool"], "move")
        self.assertEqual(self.failures, [])

    def test_invalid_params_are_not_dispatched(self):
        for params in ({"name": "look", "arguments": [1]}, {"arguments": {}}, {"name": 3}):
            response = self.client.request("tools/call", params)
            self.assertEqual(response.messages[-1]["error"]["code"], -32602)
        self.assertEqual(self.fake.calls, [])

    def test_missing_arguments_default_to_empty_object(self):
        response = self.client.request("tools/call", {"name": "look"}, accept="application/json")
        self.assertFalse(response.messages[0]["result"]["isError"])
        self.assertEqual(self.fake.calls[0]["arguments"], {})

    def test_duplicate_in_flight_request_id_is_refused(self):
        self.fake.gate("move")
        pending = self.client.start_call("move", {}, request_id=42, progress_token=1)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        second = self.client.call_tool("look", {}, request_id=42, accept_sse=False)
        self.assertEqual(second.error["code"], -32600)
        self.fake.gates["move"].set()
        self.assertFalse(pending.wait(TIMEOUT).result["isError"])
        self.assertEqual([call["name"] for call in self.fake.calls], ["move"])

    def test_encoding_failure_is_a_jsonrpc_error_and_a_delivery_failure(self):
        self.fake.results["look"] = {"success": True, "contentItems": [
            {"type": "inputText", "text": "see"}, {"type": "inputImage", "imageUrl": "https://example.invalid/a.png"}]}
        result = self.client.call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_9"}, progress_token=1)
        self.assertEqual(result.error["code"], -32603)
        self.assertTrue(result.error["message"].startswith("invalid_tool_image_url"))
        self.assertNotIn(b"example.invalid", result.response.raw)
        self.assertEqual(self.failures, [("toolu_9", "tool_reply_encoding_failed")])


class KeepaliveCadenceTest(ServerCase):
    server_kwargs = dict(keepalive_s=0.2, peer_check_s=0.02)

    def test_long_call_headers_immediately_then_progress_every_keepalive(self):
        self.fake.gate("move")
        body = json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                           "params": {"name": "move", "arguments": {}, "_meta": {"progressToken": 5}}})
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server.server_port, timeout=TIMEOUT)
        self.addCleanup(conn.close)
        started = time.monotonic()
        conn.request("POST", "/mcp", body=body, headers={**self.server.headers, "Content-Type": "application/json",
                                                         "Accept": "application/json, text/event-stream"})
        response = conn.getresponse()  # headers arrive while the call is still gated
        self.assertFalse(self.fake.finished.get("move", threading.Event()).is_set())
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        arrivals = []
        while len(arrivals) < 5:
            line = response.fp.readline()
            if line.startswith(b"data:"):
                arrivals.append(time.monotonic())
                self.assertEqual(json.loads(line[5:])["method"], "notifications/progress")
        self.fake.gates["move"].set()
        rest = response.read()
        self.assertIn(b'"ok:move"', rest)
        gaps = [later - earlier for earlier, later in zip(arrivals, arrivals[1:])]
        self.assertGreater(arrivals[0] - started, 0.1)
        for gap in gaps:
            self.assertGreater(gap, 0.1)
            self.assertLess(gap, 1.0)


class CancelAndDisconnectTest(ServerCase):
    def test_cancel_after_answer_is_ignored(self):
        result = self.client.call_tool_then_cancel("look", {}, progress_token=3)
        self.assertEqual(self.client.last_response.status, 202)
        self.assertFalse(result.result["isError"])
        time.sleep(0.1)
        self.assertEqual(self.failures, [])
        records = [r["message"] for r in self.raw_records()]
        sent = next(i for i, m in enumerate(records) if m.get("id") == result.request_id and "result" in m)
        cancel = next(i for i, m in enumerate(records) if m.get("method") == "notifications/cancelled")
        self.assertLess(sent, cancel)

    def test_in_flight_cancel_abandons_without_aborting(self):
        self.fake.gate("move")
        pending = self.client.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_c"}, progress_token=9)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        self.assertEqual(self.client.cancel(pending.request_id, "AbortError: interrupt").status, 202)
        self.assertEqual(self.failures, [("toolu_c", "tool_reply_delivery_uncertain")])
        outcome = pending.wait(TIMEOUT)  # the stream is closed without a result
        self.assertIsNone(outcome.message)
        self.assertFalse(self.fake.finished.get("move", threading.Event()).is_set())  # never aborted
        self.fake.gates["move"].set()
        self.assertTrue(self.fake.finished.setdefault("move", threading.Event()).wait(TIMEOUT))
        time.sleep(0.1)
        self.assertEqual(len(self.failures), 1)
        self.assertFalse(any(r["direction"] == "sent" and r["message"].get("id") == pending.request_id
                             for r in self.raw_records()))

    def open_call(self, request_id, name, *, sse=True, meta=None):
        """Raw socket ``tools/call`` so the test decides exactly when the peer goes away."""
        body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                           "params": {"name": name, "arguments": {}, "_meta": {"progressToken": 1, **(meta or {})}}})
        port = self.server.server.server_port
        sock = socket.create_connection(("127.0.0.1", port), timeout=TIMEOUT)
        self.addCleanup(sock.close)
        accept = "application/json, text/event-stream" if sse else "application/json"
        sock.sendall((f"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                      f"Authorization: {self.server.headers['Authorization']}\r\nContent-Type: application/json\r\n"
                      f"Accept: {accept}\r\nContent-Length: {len(body)}\r\n\r\n{body}").encode())
        return sock

    @staticmethod
    def hang_up(sock):
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()

    def test_peer_close_during_sse_call(self):
        self.fake.gate("move")
        sock = self.open_call(11, "move", meta={"claudecode/toolUseId": "toolu_d"})
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        self.hang_up(sock)
        self.assertTrue(wait_until(lambda: self.failures))
        self.assertEqual(self.failures, [("toolu_d", "tool_reply_delivery_uncertain")])
        self.fake.gates["move"].set()
        self.assertTrue(self.fake.finished.setdefault("move", threading.Event()).wait(TIMEOUT))
        time.sleep(0.1)
        self.assertEqual(len(self.failures), 1)
        self.assertFalse(any(r["direction"] == "sent" and r["message"].get("id") == 11 for r in self.raw_records()))

    def test_peer_close_before_dispatch_is_never_executed_nor_reported(self):
        logged, release = threading.Event(), threading.Event()

        def raw_log(direction, message):
            self.store.raw_agent(direction, message, channel="mcp")
            if message.get("method") == "tools/call":
                logged.set()
                release.wait(TIMEOUT)  # the peer hangs up while the server is still recording the call

        server = self.make_server(raw_log=raw_log)
        self.server = server
        sock = self.open_call(12, "move")
        self.assertTrue(logged.wait(TIMEOUT))
        self.hang_up(sock)
        self.assertTrue(wait_until(lambda: not server.inflight))
        release.set()
        time.sleep(0.2)
        self.assertEqual((self.fake.calls, self.failures), ([], []))

    def test_peer_close_during_json_call(self):
        self.fake.gate("move")
        self.client.call_tool_and_disconnect("move", {}, accept_sse=False, wait_for="sent")
        self.assertTrue(wait_until(lambda: self.failures))
        self.assertEqual(len(self.failures), 1)
        call_id, reason = self.failures[0]
        self.assertEqual(reason, "tool_reply_delivery_uncertain")
        self.assertTrue(call_id.startswith("call-"))
        self.fake.gates["move"].set()

    def test_peer_close_found_at_final_write(self):
        server = self.make_server(peer_check_s=60, keepalive_s=60)  # only the pre-write check can see it
        client = FakeMcpClient(server.url, token=server.token)
        self.fake.gate("move")
        client.call_tool_and_disconnect("move", {}, progress_token=1)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        time.sleep(0.1)
        self.assertEqual(self.failures, [])
        self.fake.gates["move"].set()
        self.assertTrue(wait_until(lambda: self.failures))
        self.assertEqual(self.failures, [(self.fake.calls[0]["call_id"], "tool_reply_delivery_uncertain")])

    def test_write_error_on_final_write(self):
        server = self.make_server(peer_check_s=60, keepalive_s=60)
        server._peer_gone = lambda call: False  # force the write itself to discover the closed peer
        big = base64.b64encode(bytes(8 * 1024 * 1024)).decode()
        self.fake.results["move"] = {"success": True, "contentItems": [
            {"type": "inputImage", "imageUrl": "data:image/png;base64," + big}]}
        self.fake.gate("move")
        self.server = server
        sock = self.open_call(13, "move", meta={"claudecode/toolUseId": "toolu_w"})
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        self.hang_up(sock)
        time.sleep(0.05)
        self.fake.gates["move"].set()
        self.assertTrue(wait_until(lambda: self.failures))
        self.assertEqual(self.failures, [("toolu_w", "tool_reply_delivery_uncertain")])
        self.assertFalse(any(r["direction"] == "sent" and r["message"].get("id") == 13 for r in self.raw_records()))

    def test_close_abandons_in_flight_calls_and_is_idempotent(self):
        self.fake.gate("move")
        pending = self.client.start_call("move", {}, progress_token=1)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        self.server.close()
        self.server.close()
        self.assertEqual(self.failures, [(self.fake.calls[0]["call_id"], "tool_reply_delivery_uncertain")])
        self.assertIsNone(pending.wait(TIMEOUT).message)
        self.assertFalse(self.server.keepalive_thread.is_alive())
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", self.server.server.server_port), timeout=1).close()
        self.fake.gates["move"].set()


class CallbackFailureTest(ServerCase):
    def test_unrecorded_call_is_refused_before_dispatch(self):
        def raw_log(direction, message):
            if message.get("method") == "tools/call":
                raise OSError("disk full")
            self.store.raw_agent(direction, message, channel="mcp")

        server = self.make_server(raw_log=raw_log)
        client = FakeMcpClient(server.url, token=server.token)
        result = client.call_tool("move", {}, progress_token=1)
        self.assertEqual(result.error["code"], -32603)
        self.assertEqual((self.fake.calls, self.failures), ([], []))
        self.assertEqual(server.errors, [("raw_log", "OSError('disk full')")])
        self.assertEqual(client.ping(), {})

    def test_raising_delivery_callback_does_not_stop_the_server(self):
        def broken(call_id, reason):
            raise RuntimeError("boom")

        server = self.make_server(on_delivery_failure=broken)
        client = FakeMcpClient(server.url, token=server.token)
        self.fake.gate("move")
        pending = client.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_x"}, progress_token=1)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        client.cancel(pending.request_id)
        self.assertEqual(server.errors, [("toolu_x", "RuntimeError('boom')")])
        self.fake.gates["move"].set()
        self.assertFalse(client.call_tool("look", {}, progress_token=2).result["isError"])
        self.assertTrue(server.keepalive_thread.is_alive())


class AccessTest(ServerCase):
    def test_bearer_and_host(self):
        for token in ("wrong", None):
            response = FakeMcpClient(self.server.url, token=token).call_tool("look", {})
            self.assertEqual((response.status, response.message), (401, None))
        port = self.server.server.server_port
        for host in ("evil.example:%d" % port, "127.0.0.1:1", "10.0.0.1:%d" % port):
            client = FakeMcpClient(self.server.url, token=self.server.token, headers={"Host": host})
            self.assertEqual(client.call_tool("look", {}).status, 403)
            self.assertEqual(client.http("GET").status, 403)
        self.assertEqual(FakeMcpClient(self.server.url, token="wrong").http("GET").status, 401)
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.raw_records(), [])
        local = FakeMcpClient(self.server.url, token=self.server.token, headers={"Host": "localhost:%d" % port})
        self.assertFalse(local.call_tool("look", {}).result["isError"])
        other = FakeMcpClient(self.server.url.replace("/mcp", "/other"), token=self.server.token)
        self.assertEqual(other.request("ping").status, 404)

    def test_request_size_cap(self):
        server = self.make_server(max_request_bytes=512)
        client = FakeMcpClient(server.url, token=server.token)
        self.assertEqual(client.call_tool("look", {"x": "a" * 1024}).status, 413)
        self.assertEqual(client.call_tool("look", {"x": "a" * 100}).status, 200)
        chunked = client.post(b"", headers={"Transfer-Encoding": "chunked"})
        self.assertEqual(chunked.status, 411)
        self.assertEqual([call["arguments"]["x"] for call in self.fake.calls], ["a" * 100])

    def test_constructor_validation(self):
        base = dict(tools=TOOLS, presentation=self.presentation, call_tool=self.fake, raw_log=print,
                    on_delivery_failure=print)
        for bad in (dict(keepalive_s=0), dict(peer_check_s=-1), dict(io_timeout_s=True),
                    dict(max_request_bytes=0), dict(server_name="a b"), dict(result_size_hint_chars=0)):
            with self.assertRaises(ValueError):
                AurigaMcpServer(**base, **bad)
        with self.assertRaises(ValueError):
            self.server.start()


# ---------------------------------------------------------------------------------------------------------------------
class EncodeTest(unittest.TestCase):
    def test_encode_table_order_and_byte_identity(self):
        audio = base64.b64encode(b"RIFF....WAVE").decode()
        response = {"success": True, "contentItems": [
            {"type": "inputText", "text": "camera follows"},
            {"type": "inputImage", "imageUrl": "data:image/png;base64," + PNG_B64},
            {"type": "inputText", "text": "and audio"},
            {"type": "inputAudio", "audioUrl": "data:audio/wav;base64," + audio},
            {"type": "inputImage", "imageUrl": "data:image/webp;foo=bar;base64," + PNG_B64}]}
        encoded = encode_mcp_result(response)
        self.assertEqual(encoded, {"isError": False, "content": [
            {"type": "text", "text": "camera follows"},
            {"type": "image", "data": PNG_B64, "mimeType": "image/png"},
            {"type": "text", "text": "and audio"},
            {"type": "audio", "data": audio, "mimeType": "audio/wav"},
            {"type": "image", "data": PNG_B64, "mimeType": "image/webp"}]})
        self.assertEqual(base64.b64decode(encoded["content"][1]["data"]), PNG)
        self.assertEqual(response["contentItems"][1]["imageUrl"], "data:image/png;base64," + PNG_B64)  # not mutated

    def test_is_error_only_without_media(self):
        self.assertTrue(encode_mcp_result(text_result("failed", success=False))["isError"])
        self.assertFalse(encode_mcp_result(text_result("fine"))["isError"])
        with_image = {"success": False, "contentItems": [
            {"type": "inputText", "text": "failed, evidence follows"},
            {"type": "inputImage", "imageUrl": "data:image/jpeg;base64," + PNG_B64}]}
        encoded = encode_mcp_result(with_image)
        self.assertFalse(encoded["isError"])
        self.assertEqual([part["type"] for part in encoded["content"]], ["text", "image"])
        with_audio = {"success": False, "contentItems": [{"type": "inputAudio", "audioUrl": "data:audio/wav;base64,AA=="}]}
        self.assertFalse(encode_mcp_result(with_audio)["isError"])

    def test_rejections(self):
        cases = [
            ({"type": "inputImage", "imageUrl": "https://example.invalid/x.png"}, "invalid_tool_image_url"),
            ({"type": "inputImage", "imageUrl": "file:///tmp/x.png"}, "invalid_tool_image_url"),
            ({"type": "inputImage", "imageUrl": "data:image/png,rawbytes"}, "invalid_tool_image_url"),
            ({"type": "inputImage", "imageUrl": "data:;base64,AA=="}, "invalid_tool_image_url"),
            ({"type": "inputAudio", "audioUrl": "https://example.invalid/a.wav"}, "invalid_tool_audio_url"),
            ({"type": "inputVideo", "videoUrl": "data:video/mp4;base64,AA=="}, "invalid_native_content_item"),
            ({"type": "inputText"}, "invalid_native_content_item"),
            ("text", "invalid_native_content_item"),
        ]
        for item, code in cases:
            with self.subTest(item=item):
                with self.assertRaises(ValueError) as caught:
                    encode_mcp_result({"success": True, "contentItems": [item]})
                self.assertEqual(str(caught.exception), code)
        for envelope in (None, {"success": 1, "contentItems": []}, {"success": True}, {"success": True, "contentItems": {}}):
            with self.assertRaises(ValueError) as caught:
                encode_mcp_result(envelope)
            self.assertEqual(str(caught.exception), "invalid_tool_result")


# ---------------------------------------------------------------------------------------------------------------------
class RawLogRoundTripTest(ServerCase):
    def test_log_round_trips_through_agent_raw_evidence(self):
        self.fake.results["look"] = {"success": True, "contentItems": [
            {"type": "inputText", "text": "view"}, {"type": "inputImage", "imageUrl": "data:image/png;base64," + PNG_B64}]}
        self.client.handshake(discover_probe=True)
        self.client.list_tools()
        native = self.client.call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_img"}, progress_token=1)
        generated = self.client.call_tool_then_cancel("plain", {}, progress_token=2)
        self.fake.gate("move")
        pending = self.client.start_call("move", {}, progress_token=3)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        self.client.cancel(pending.request_id)
        self.fake.gates["move"].set()
        pending.wait(TIMEOUT)
        self.assertTrue(self.fake.finished.setdefault("move", threading.Event()).wait(TIMEOUT))
        records = self.raw_records()
        self.assertTrue(all(record["channel"] == "mcp" for record in records))
        calls = [r for r in records if r["message"].get("method") == "tools/call"]
        self.assertTrue(all(r["direction"] == "received" for r in calls))
        ids = {call["rpc_id"]: call["call_id"] for call in self.fake.calls}
        self.assertEqual([r["message"]["params"]["_meta"]["item_id"] for r in calls],
                         ["toolu_img", ids[generated.request_id], ids[pending.request_id]])
        self.assertEqual(calls[0]["message"]["params"]["_meta"]["claudecode/toolUseId"], "toolu_img")
        responses = [r for r in records if r["direction"] == "sent"]
        self.assertTrue(all("id" in r["message"] for r in responses))
        received_methods = [r["message"].get("method") for r in records if r["direction"] == "received"]
        self.assertEqual(received_methods[:4], ["server/discover", "initialize", "notifications/initialized",
                                                "tools/list"])
        self.assertEqual(received_methods.count("notifications/cancelled"), 2)
        self.assertFalse(any(r["message"].get("method") == "notifications/progress" for r in records))
        evidence = load_native_evidence(self.tmp.name, dict(thread_id="thr", agent_backend=dict(kind="claude_code")),
                                        self.store)
        self.assertEqual(evidence.tool_item_ids, {"toolu_img", ids[generated.request_id]})
        self.assertEqual(evidence.sent_image_sha256, {hashlib.sha256(PNG).hexdigest()})
        self.assertEqual(native.result["content"][1]["data"], PNG_B64)


class DispatcherOrderingTest(ServerCase):
    """Concurrent MCP calls through the session-style closure keep the OrderedBarrier semantics."""

    def setUp(self):
        super().setUp()
        self.events, self.lock = [], threading.Lock()
        self.submitted = {}
        self.gates = {name: threading.Event() for name in ("A", "D")}
        self.rendezvous = threading.Barrier(2, timeout=TIMEOUT)

        def handler(params):
            name = params["arguments"]["tag"]
            with self.lock:
                self.events.append(("start", name))
            if name in self.gates:
                self.gates[name].wait(TIMEOUT)
            if name in ("B", "C"):
                self.rendezvous.wait()  # completes only if B and C overlap
            with self.lock:
                self.events.append(("end", name))
            return text_result(name)

        def parallel_safe(tool, arguments):
            return tool == "look"

        self.dispatcher = ToolDispatcher(handler, parallel_safe, max_workers=8)
        self.addCleanup(self.dispatcher.shutdown, wait=False)

        def call_tool(name, arguments, *, call_id, native_call_id, rpc_id):
            future = self.dispatcher.submit(dict(tool=name, arguments=arguments, callId=call_id))
            self.submitted[arguments["tag"]].set()
            return future.result()

        self.server.call_tool = call_tool

    def start(self, tool, tag):
        self.submitted[tag] = threading.Event()
        pending = self.client.start_call(tool, {"tag": tag}, progress_token=tag)
        self.assertTrue(self.submitted[tag].wait(TIMEOUT))
        return pending

    def test_parallel_calls_overlap_and_ordered_calls_wait(self):
        a = self.start("move", "A")
        b = self.start("look", "B")
        c = self.start("look", "C")
        d = self.start("move", "D")
        time.sleep(0.1)
        self.assertEqual(self.events, [("start", "A")])  # B, C wait for the earlier ordered call
        self.gates["A"].set()
        for pending, tag in ((b, "B"), (c, "C")):
            self.assertEqual(pending.wait(TIMEOUT).result["content"][0]["text"], tag)
        self.assertTrue(wait_until(lambda: ("start", "D") in self.events))
        self.assertFalse(d.done())
        self.gates["D"].set()
        self.assertEqual(d.wait(TIMEOUT).result["content"][0]["text"], "D")
        self.assertEqual(a.wait(TIMEOUT).result["content"][0]["text"], "A")
        order = [event for event in self.events]
        self.assertLess(order.index(("end", "A")), order.index(("start", "B")))
        self.assertLess(order.index(("end", "A")), order.index(("start", "C")))
        self.assertLess(max(order.index(("end", "B")), order.index(("end", "C"))), order.index(("start", "D")))
        self.assertEqual(self.failures, [])


if __name__ == "__main__":
    unittest.main()
