"""ModelRequestProxy with the Anthropic Messages wire against FakeAnthropicServer (P1-2).

Covers the Claude Code client profile end to end through the real relay: forwarding and evidence, a tool round
trip with the egress scrub, single-attempt 403 denials, mid-stream failures, truncated upstream streams,
``ping`` keepalive spacing, deferred delivery (stub profile), ``count_tokens`` as an uncharged auxiliary
route, the budget, gzip request bodies, the client-retry runaway guard and "never close a stream without
a terminal".
"""
import gzip
import hashlib
import http.client
import json
from pathlib import Path
import re
import tempfile
import threading
import time
from types import MappingProxyType
import unittest
from unittest import mock

from auriga.core.presentation.actor_presentation import ActorPresentation
from auriga.core.storage.store import Store
from auriga.cortex.session import model_proxy
from auriga.cortex.session.model_proxy import ModelRequestProxy
from auriga.cortex.session.wire import AnthropicMessagesWire, ClientProfile, WireReply

from support.fake_model_servers import Failure, FakeAnthropicServer, Step, Usage, text_step, tool_step

SESSION = "9ff8e771-aa08-4c17-b00d-032e1cb2011f"
PREFIX = "mcp__auriga__"
REGISTERED = frozenset({"robot_observe"})
DIGEST = "0123456789abcdef" * 4
ARTIFACT = "artifact:" + "a" * 64 + "-" + "b" * 16
LONG_OBSERVATION = "observation-" + "c" * 32
PLANTED = f"sha256={DIGEST} see {ARTIFACT} and {LONG_OBSERVATION}"
IMAGE_SOURCE = ("[Image: source: " + "/private/run/claude-config/projects/-private-run-actor/" + SESSION
                + "/tool-results/mcp-auriga-blob-1794338642782-z5ip6n.png]")
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="
PING = b'event: ping\ndata: {"type": "ping"}\n\n'
CLIENT_HEADERS = (
    ("Content-Type", "application/json"), ("Accept", "application/json"), ("x-api-key", "auriga-local-relay"),
    ("anthropic-version", "2023-06-01"),
    ("anthropic-beta", "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27"),
    ("X-Claude-Code-Session-Id", SESSION), ("User-Agent", "claude-cli/2.1.321 (external, sdk-py, agent-sdk/0.2.180)"),
    ("Accept-Encoding", "gzip, deflate, br, zstd"), ("X-Auriga-Source", "private"),
)
NEUTRAL_400 = b'{"type":"error","error":{"type":"invalid_request_error","message":"auriga_model_request_failed"}}'


class FastPingWire(AnthropicMessagesWire):
    """Claude Code profile with a short ping spacing so spacing is measurable in a unit test."""
    keepalive_interval_s = 0.2


class DeferredStubWire(AnthropicMessagesWire):
    """Stub deferred-delivery profile (the real OpenCode profile is P3-4): nothing before validation, then
    the whole stream or one neutral HTTP 400 JSON."""
    client_profiles = MappingProxyType({"deferred_stub": ClientProfile(early_stream=False, client_loop_guard=True)})

    def relay_error(self, *, streaming, evidence, manual_retry_allowed):
        return WireReply(400 if not streaming else 599, headers={"Content-Type": "application/json"}, body=NEUTRAL_400)

    def denial(self, kind):
        return WireReply(400, headers={"Content-Type": "application/json"}, body=NEUTRAL_400)


class CountTokensFailure(FakeAnthropicServer):
    """count_tokens answers 529 with a provider message (which must never be retried by the relay)."""

    def _local(self, h, req):
        if req.method == "POST" and req.route == "/v1/messages/count_tokens":
            return self._send_json(h, req, 529, self.error_body(529, "fixture-private 9"))
        return super()._local(h, req)


def request_body(text="hello", **extra):
    return dict(dict(model="claude-opus-5-5", max_tokens=1024, stream=True,
                     messages=[dict(role="user", content=text)]), **extra)


def events(data):
    """Decoded SSE events ``[(type, value)]`` of a client-visible body, pings excluded."""
    found = []
    for block in data.replace(b"\r\n", b"\n").split(b"\n\n"):
        lines = [line for line in block.split(b"\n") if line.startswith(b"data:")]
        if not lines:
            continue
        value = json.loads(b"\n".join(line[5:].lstrip() for line in lines))
        if value.get("type") != "ping":
            found.append((value.get("type"), value))
    return found


class Reply:
    def __init__(self, status, reason, headers, chunks, head_s):
        self.status, self.reason, self.chunks, self.head_s = status, reason, chunks, head_s
        self.headers = {key.lower(): value for key, value in headers}
        self.body = b"".join(chunk for _, chunk in chunks)

    def json(self):
        return json.loads(self.body)


class AnthropicRelayTest(unittest.TestCase):
    maxDiff = 4000

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "run", dict(run_id="relay", episode_id="episode",
                                                              env_instance_id="environment", env_generation=1))
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)

    # ------------------------------------------------------------------ helpers
    def server(self, script=(), cls=FakeAnthropicServer, **kwargs):
        server = cls(script, **kwargs).start()
        self.addCleanup(server.close)
        return server

    def proxy(self, server, wire=None, *, project=False, **kwargs):
        wire = wire or AnthropicMessagesWire()
        if project:
            kwargs["request_projection"] = lambda request: wire.project_request(
                request, self.presentation, registered_tools=REGISTERED, tool_name_prefix=PREFIX)
        kwargs.setdefault("max_retries", 0)
        proxy = ModelRequestProxy(server.base_url, wire=wire, upstream_auth_headers={"x-api-key": "host-secret"},
                                  **kwargs).start()
        self.addCleanup(proxy.close)
        patch = mock.patch.object(proxy.policy, "_retry_delay_seconds", return_value=0)
        patch.start()
        self.addCleanup(patch.stop)
        return proxy

    @staticmethod
    def count(proxy, kind):
        with proxy.lock:
            return sum(event["kind"] == kind for event in proxy.events)

    @staticmethod
    def kinds(proxy, kind):
        with proxy.lock:
            return [dict(event) for event in proxy.events if event["kind"] == kind]

    def call(self, proxy, payload=None, *, path="/v1/messages?beta=true", headers=CLIENT_HEADERS, method="POST",
             data=None, bookkeeping="model_call_finished"):
        """One client request read to EOF with chunk arrival times; waits for the relay's bookkeeping event."""
        if data is None:
            data = b"" if payload is None else json.dumps(payload).encode()
        before = self.count(proxy, bookkeeping) if bookkeeping else 0
        port = int(proxy.base_url.rsplit(":", 1)[1])
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            connection.putrequest(method, path, skip_accept_encoding=True)
            for key, value in headers:
                connection.putheader(key, value)
            connection.putheader("Content-Length", str(len(data)))
            started = time.monotonic()
            connection.endheaders(data)
            response = connection.getresponse()
            head_s = time.monotonic() - started
            chunks = []
            while chunk := response.read1(65536):
                chunks.append((time.monotonic() - started, chunk))
            reply = Reply(response.status, response.reason, response.getheaders(), chunks, head_s)
        finally:
            connection.close()
        deadline = time.monotonic() + 10
        while bookkeeping and self.count(proxy, bookkeeping) <= before and time.monotonic() < deadline:
            time.sleep(0.005)
        return reply

    def assert_single_error_event(self, reply, message):
        self.assertEqual(reply.status, 200)
        self.assertEqual(reply.headers["content-type"], "text/event-stream")
        self.assertEqual([kind for kind, _ in events(reply.body)], ["error"])
        self.assertEqual(events(reply.body)[0][1], {"type": "error", "error": {"type": "invalid_request_error",
                                                                               "message": message}})
        self.assertTrue(reply.body.endswith(b"\n\n"))
        self.assertIsNone(re.search(rb"\d", reply.body.replace(PING, b"")))

    def assert_single_attempt_denial(self, reply, message):
        self.assertEqual(reply.status, 403)
        self.assertEqual(reply.headers["x-should-retry"], "false")
        self.assertEqual(reply.headers["content-type"], "application/json")
        self.assertEqual(reply.json(), {"type": "error", "error": {"type": "permission_error", "message": message}})
        self.assertIsNone(re.search(rb"\d", reply.body))

    # ------------------------------------------------------------------ tests
    def test_normal_completion_forwarding_attribution_and_usage(self):
        server = self.server([text_step("hello there", usage=Usage(input_tokens=11, output_tokens=5,
                                                                   cache_read_input_tokens=3,
                                                                   cache_creation_input_tokens=2))])
        proxy = self.proxy(server)
        proxy.register_thread(SESSION, "actor", "req-session")
        data = json.dumps(request_body("hi")).encode()
        reply = self.call(proxy, data=data)
        self.assertEqual(reply.status, 200)
        self.assertEqual(reply.headers["content-type"], "text/event-stream")
        kinds = [kind for kind, _ in events(reply.body)]
        self.assertEqual((kinds[0], kinds[-2], kinds[-1]), ("message_start", "message_delta", "message_stop"))
        self.assertEqual("".join(value["delta"]["text"] for kind, value in events(reply.body)
                                 if kind == "content_block_delta"), "hello there")
        upstream = server.generation_requests()
        self.assertEqual(len(upstream), 1)
        forwarded = upstream[0]
        self.assertEqual(forwarded.path, "/v1/messages?beta=true")
        self.assertEqual(forwarded.raw_body, data)
        self.assertEqual(forwarded.headers["x-api-key"], "host-secret")
        self.assertEqual(forwarded.headers.get_all("x-api-key"), ["host-secret"])
        self.assertEqual(forwarded.headers["anthropic-version"], "2023-06-01")
        self.assertEqual(forwarded.headers["anthropic-beta"], dict(CLIENT_HEADERS)["anthropic-beta"])
        self.assertEqual(forwarded.headers["Accept-Encoding"], "identity")
        self.assertIsNone(forwarded.headers.get("X-Auriga-Source"))
        self.assertEqual(proxy.request_count, 1)
        started = self.kinds(proxy, "model_call_started")[0]
        self.assertEqual((started["endpoint"], started["native_thread_id"], started["source"],
                          started["caller_request_id"]), ("/v1/messages", SESSION, "actor", "req-session"))
        finished = self.kinds(proxy, "model_request_finished")[0]
        self.assertEqual((finished["served_model"], finished["response_status"], finished["stop_reason"],
                          finished["response_id"]), ("claude-opus-5-5", "completed", "end_turn", "msg_fake_0"))
        self.assertEqual(proxy.wire.normalize_usage(finished["usage"]), dict(
            input_tokens=16, cached_input_tokens=3, cache_write_input_tokens=2, output_tokens=5,
            reasoning_output_tokens=None, total_tokens=21))

    def test_unregistered_session_falls_back_to_default_thread(self):
        server = self.server()
        proxy = self.proxy(server, default_thread_id="session-thread")
        proxy.register_thread("session-thread", "actor", "req-default")
        self.call(proxy, request_body())
        started = self.kinds(proxy, "model_call_started")[0]
        self.assertEqual((started["native_thread_id"], started["source"]), ("session-thread", "actor"))

    def test_tool_round_trip_scrubs_history_and_keeps_protocol_bytes(self):
        server = self.server([tool_step(PREFIX + "robot_observe", {"camera": "head"}, id="toolu_01A",
                                        thinking="look first", thinking_signature="sig-" + DIGEST[:20]),
                              text_step("done")])
        proxy = self.proxy(server, project=True)
        first = self.call(proxy, request_body("observe the scene " + PLANTED))
        found = events(first.body)
        start = next(value["content_block"] for kind, value in found if kind == "content_block_start"
                     and value["content_block"]["type"] == "tool_use")
        arguments = "".join(value["delta"]["partial_json"] for kind, value in found
                            if kind == "content_block_delta" and value["delta"]["type"] == "input_json_delta")
        self.assertEqual((start["id"], start["name"], json.loads(arguments)), ("toolu_01A", PREFIX + "robot_observe",
                                                                             {"camera": "head"}))
        signature = "".join(value["delta"].get("signature", "") for kind, value in found if kind == "content_block_delta")
        history = request_body()
        history["messages"] = [
            dict(role="user", content="observe the scene " + PLANTED),
            dict(role="system", content="environment " + PLANTED),
            dict(role="assistant", content=[dict(type="thinking", thinking="look first", signature=signature),
                                            dict(type="tool_use", id="toolu_01A", name=PREFIX + "robot_observe",
                                                 input={"camera": "head", "observation_id": LONG_OBSERVATION})]),
            dict(role="user", content=[dict(type="tool_result", tool_use_id="toolu_01A", content=[
                dict(type="text", text=json.dumps(dict(observation_id=LONG_OBSERVATION, frame_sha256=DIGEST,
                                                       note=PLANTED))),
                dict(type="image", source=dict(type="base64", media_type="image/png", data=PNG)),
                dict(type="text", text=IMAGE_SOURCE, cache_control=dict(type="ephemeral"))])]),
        ]
        second = self.call(proxy, history)
        self.assertEqual("".join(value["delta"].get("text", "") for kind, value in events(second.body)
                                 if kind == "content_block_delta"), "done")
        upstream = server.generation_requests()
        self.assertEqual(len(upstream), 2)
        for forwarded in upstream:
            for value in (DIGEST, ARTIFACT, LONG_OBSERVATION, "[Image: source:"):
                self.assertNotIn(value.encode(), forwarded.raw_body)
        body = upstream[1].json
        self.assertEqual([message["role"] for message in body["messages"]], ["user", "system", "assistant", "user"])
        self.assertEqual(body["messages"][2]["content"][0], history["messages"][2]["content"][0])
        self.assertEqual(body["messages"][2]["content"][1]["id"], "toolu_01A")
        result = server.find_tool_results(upstream[1])[0]
        self.assertEqual(result.call_id, "toolu_01A")
        self.assertEqual(result.raw["content"][2], dict(type="text", text="[Image]", cache_control=dict(type="ephemeral")))
        self.assertEqual([image.sha256 for image in result.images],
                         [hashlib.sha256(__import__("base64").b64decode(PNG)).hexdigest()])
        self.assertEqual(proxy.request_count, 2)

    def test_denials_are_single_attempt_403_with_no_retry_header(self):
        server = self.server([Step(failure=Failure(kind="http", status=401, error_type="authentication_error",
                                                   message="fixture-private 7"))])
        proxy = self.proxy(server, max_requests=3)
        failed = self.call(proxy, request_body("one"))
        self.assert_single_error_event(failed, "auriga_model_request_failed")
        self.assertNotIn(b"fixture-private", failed.body)
        self.assertTrue(proxy.transport_blocked)
        blocked = self.call(proxy, request_body("two"), bookkeeping=None)
        self.assert_single_attempt_denial(blocked, "auriga_model_transport_blocked")
        self.assertEqual(len(server.generation_requests()), 1)

        budget_server = self.server()
        budget = self.proxy(budget_server, max_requests=1)
        self.assertEqual(self.call(budget, request_body("one")).status, 200)
        self.assert_single_attempt_denial(self.call(budget, request_body("two"), bookkeeping=None),
                                          "auriga_model_budget_exhausted")
        self.assertEqual(len(budget_server.generation_requests()), 1)

        cancel_server = self.server()
        cancelled = self.proxy(cancel_server)
        cancelled.register_thread(SESSION, "actor", "req")
        cancelled.cancel_thread(SESSION)
        self.assert_single_attempt_denial(self.call(cancelled, request_body(), bookkeeping=None),
                                          "auriga_model_thread_cancelled")
        self.assertEqual(cancel_server.generation_requests(), [])

    def test_budget_counts_delivered_generations_only(self):
        limits = []
        server = self.server()
        proxy = self.proxy(server, max_requests=2, on_limit=limits.append)
        for text in ("one", "two"):
            self.assertEqual(self.call(proxy, request_body(text)).status, 200)
        self.assert_single_attempt_denial(self.call(proxy, request_body("three"), bookkeeping=None),
                                          "auriga_model_budget_exhausted")
        self.assertEqual((proxy.request_count, proxy.reserved_calls), (2, 0))
        self.assertEqual(len(server.generation_requests()), 2)
        deadline = time.monotonic() + 2
        while not limits and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(limits), 1)

    def test_mid_stream_error_delivers_one_error_event_and_no_partial_content(self):
        server = self.server([Step(text="partial answer that must not leak", chunk_size=4,
                                   failure=Failure(kind="stream_error", error_type="invalid_request_error",
                                                   message="fixture-private 123"))])
        proxy = self.proxy(server, max_retries=3)
        reply = self.call(proxy, request_body())
        self.assert_single_error_event(reply, "auriga_model_response_unusable")
        for leaked in (b"partial", b"fixture-private", b"message_start", b"text_delta"):
            self.assertNotIn(leaked, reply.body)
        self.assertEqual(len(server.generation_requests()), 1)  # configuration error: no retry
        self.assertTrue(proxy.transport_blocked)
        self.assertTrue(proxy.blocked_error["manual_retry_allowed"])
        self.assertEqual(proxy.request_count, 0)
        failed = self.kinds(proxy, "model_request_finished")[0]
        self.assertEqual((failed["response_status"], failed["response_error"]["classification"]),
                         ("failed", "configuration_error"))

        retry_server = self.server([Step(text="partial", failure=Failure(kind="stream_error"))] * 3)
        retrying = self.proxy(retry_server, max_retries=1)
        self.assert_single_error_event(self.call(retrying, request_body()), "auriga_model_response_unusable")
        self.assertEqual(len(retry_server.generation_requests()), 2)  # overloaded_error: retried, then failed
        self.assertEqual([event["error"]["classification"] for event in self.kinds(retrying, "model_request_retry")],
                         ["server_overloaded"])

    def test_truncated_upstream_is_resent_then_failed_with_a_terminal(self):
        server = self.server([Step(text="cut short", failure=Failure(kind="truncated"))] * 2)
        proxy = self.proxy(server, max_retries=1)
        reply = self.call(proxy, request_body())
        self.assert_single_error_event(reply, "auriga_model_request_failed")
        self.assertNotIn(b"cut short", reply.body)
        self.assertEqual(len(server.generation_requests()), 2)
        self.assertTrue(proxy.transport_blocked)
        recovered_server = self.server([Step(text="cut", failure=Failure(kind="truncated")), text_step("whole")])
        recovered = self.proxy(recovered_server, max_retries=1)
        reply = self.call(recovered, request_body())
        self.assertEqual(events(reply.body)[-1][0], "message_stop")
        self.assertEqual((recovered.request_count, len(recovered_server.generation_requests())), (1, 2))
        self.assertFalse(recovered.transport_blocked)

    def test_ping_keepalive_spacing_while_buffering(self):
        server = self.server([text_step("late answer", delay_s=1.1)])
        proxy = self.proxy(server, wire=FastPingWire())
        reply = self.call(proxy, request_body())
        self.assertLess(reply.head_s, 0.5)  # early 200: Claude Code cuts header waits at ~360 s
        head = reply.body[:reply.body.index(b"event: message_start")]
        self.assertGreaterEqual(head.count(PING), 3)
        self.assertEqual(head, PING * head.count(PING))
        times = [at for at, chunk in reply.chunks if chunk.startswith(PING)]
        gaps = [later - earlier for earlier, later in zip(times, times[1:])]
        self.assertTrue(gaps and all(0.1 <= gap <= 0.5 for gap in gaps), gaps)
        self.assertEqual(events(reply.body)[-1][0], "message_stop")

    def test_deferred_delivery_sends_no_byte_before_validation(self):
        server = self.server([text_step("late answer", delay_s=0.6),
                              Step(text="partial", failure=Failure(kind="stream_error", error_type="invalid_request_error")),
                              Step(text="cut", failure=Failure(kind="truncated")),
                              Step(text="cut", failure=Failure(kind="truncated"))])
        proxy = self.proxy(server, wire=DeferredStubWire(), max_retries=1)
        ok = self.call(proxy, request_body("one"))
        self.assertGreaterEqual(ok.head_s, 0.5)  # nothing (no status line, no keepalive) while buffering
        self.assertEqual(ok.status, 200)
        self.assertEqual(int(ok.headers["content-length"]), len(ok.body))
        self.assertNotIn(PING, ok.body)
        self.assertEqual(events(ok.body)[-1][0], "message_stop")
        failed = self.call(proxy, request_body("two"))
        self.assertEqual((failed.status, failed.body), (400, NEUTRAL_400))
        self.assertEqual(len(server.generation_requests()), 2)  # explicit failure: one attempt, no in-band error
        proxy.manual_retry()
        cut = self.call(proxy, request_body("three"))
        self.assertEqual((cut.status, cut.body), (400, NEUTRAL_400))
        self.assertEqual(len(server.generation_requests()), 4)  # truncated: 1 + max_retries attempts
        self.assertEqual(proxy.request_count, 1)

    def test_count_tokens_is_projected_forwarded_once_and_never_charged(self):
        server = self.server(count_tokens_value=42)
        proxy = self.proxy(server, project=True, max_requests=1)
        proxy.register_thread(SESSION, "actor", "req-session")
        body = dict(model="claude-opus-5-5", messages=[dict(role="user", content=[dict(type="text", text=PLANTED)])],
                    tools=[])
        reply = self.call(proxy, body, path="/v1/messages/count_tokens?beta=true", bookkeeping="model_aux_request")
        self.assertEqual((reply.status, reply.json()), (200, {"input_tokens": 42}))
        counted = [request for request in server.requests if request.route == "/v1/messages/count_tokens"]
        self.assertEqual(len(counted), 1)
        self.assertEqual(counted[0].path, "/v1/messages/count_tokens?beta=true")
        self.assertEqual(counted[0].headers["x-api-key"], "host-secret")
        for value in (DIGEST, ARTIFACT, LONG_OBSERVATION):
            self.assertNotIn(value.encode(), counted[0].raw_body)
        self.assertEqual((proxy.request_count, proxy.logical_call_count, proxy.reserved_calls, proxy.attempt_count),
                         (0, 0, 0, 0))
        event = self.kinds(proxy, "model_aux_request")[0]
        self.assertEqual((event["endpoint"], event["http_status"], event["native_thread_id"], event["source"],
                          event.get("delivered")), ("/v1/messages/count_tokens", 200, SESSION, "actor", True))
        self.assertEqual(self.count(proxy, "model_call_started"), 0)
        self.assertEqual(self.call(proxy, request_body()).status, 200)  # the one-call budget is still intact

        failing_server = self.server(cls=CountTokensFailure)
        # P1-2 review item 8: an auxiliary body is only forwarded projected, and an upstream error reaches the
        # client as the wire's sanitised relay error (403 on the Claude Code profile), never verbatim.
        failing = self.proxy(failing_server, project=True, max_retries=3)
        reply = self.call(failing, body, path="/v1/messages/count_tokens", bookkeeping="model_aux_request")
        self.assertEqual(reply.status, 403)
        self.assertEqual(len([r for r in failing_server.requests if r.route == "/v1/messages/count_tokens"]), 1)
        event = self.kinds(failing, "model_aux_request")[0]
        self.assertEqual(event["response_error"]["classification"], "server_overloaded")
        self.assertNotIn("fixture-private", json.dumps(event))
        self.assertFalse(failing.transport_blocked)

        cancelled = self.proxy(self.server())
        cancelled.register_thread(SESSION, "actor", "req")
        cancelled.cancel_thread(SESSION)
        self.assert_single_attempt_denial(self.call(cancelled, body, path="/v1/messages/count_tokens",
                                                    bookkeeping="model_aux_request"), "auriga_model_thread_cancelled")

    def test_auxiliary_projection_rejection_is_answered_without_forwarding(self):
        server = self.server()
        proxy = self.proxy(server, project=True)
        body = dict(model="m", messages=[dict(role="user", content="x")], tools=[dict(type="web_search_20260209",
                                                                                      name="web_search")])
        reply = self.call(proxy, body, path="/v1/messages/count_tokens", bookkeeping="model_aux_request")
        self.assertEqual(reply.status, 403)
        self.assertEqual(reply.json()["error"]["type"], "permission_error")
        self.assertEqual(server.requests, [])
        self.assertEqual(self.kinds(proxy, "model_aux_request")[0]["error_type"], "ValueError")

    def test_gzip_request_bodies_are_decoded_before_forwarding(self):
        server = self.server()
        proxy = self.proxy(server)
        raw = json.dumps(request_body("zipped")).encode()
        headers = CLIENT_HEADERS + (("Content-Encoding", "gzip"),)
        self.assertEqual(self.call(proxy, data=gzip.compress(raw), headers=headers).status, 200)
        forwarded = server.generation_requests()[0]
        self.assertEqual(forwarded.raw_body, raw)
        self.assertIsNone(forwarded.headers.get("Content-Encoding"))
        projected_server = self.server()
        projected = self.proxy(projected_server, project=True)
        planted = json.dumps(request_body("zipped " + PLANTED)).encode()
        self.assertEqual(self.call(projected, data=gzip.compress(planted), headers=headers).status, 200)
        self.assertNotIn(DIGEST.encode(), projected_server.generation_requests()[0].raw_body)
        rejected_server = self.server()
        rejected = self.proxy(rejected_server)
        reply = self.call(rejected, data=raw, headers=CLIENT_HEADERS + (("Content-Encoding", "br"),))
        self.assertEqual((reply.status, reply.json()["error"]["type"]), (403, "permission_error"))
        self.assertEqual(rejected_server.generation_requests(), [])

    def test_runaway_guard_blocks_a_tight_identical_retry_loop(self):
        server = self.server()
        proxy = self.proxy(server)
        body = request_body("same body")
        for _ in range(proxy.RUNAWAY_MAX_IDENTICAL_ADMISSIONS):
            self.assertEqual(self.call(proxy, body).status, 200)
        self.assert_single_attempt_denial(self.call(proxy, body), "auriga_client_retry_runaway")
        self.assertEqual(len(server.generation_requests()), proxy.RUNAWAY_MAX_IDENTICAL_ADMISSIONS)
        self.assertEqual(proxy.blocked_error["error_type"], "ClientRetryRunaway")
        self.assertTrue(proxy.blocked_error["manual_retry_allowed"])
        self.assertEqual(self.kinds(proxy, "model_client_retry_runaway")[0]["identical_admissions"],
                         proxy.RUNAWAY_MAX_IDENTICAL_ADMISSIONS + 1)
        self.assertEqual((proxy.request_count, proxy.reserved_calls), (proxy.RUNAWAY_MAX_IDENTICAL_ADMISSIONS, 0))
        self.assert_single_attempt_denial(self.call(proxy, request_body("other"), bookkeeping=None),
                                          "auriga_model_transport_blocked")

        distinct_server = self.server()
        distinct = self.proxy(distinct_server)
        for index in range(12):
            self.assertEqual(self.call(distinct, request_body(f"turn {index}")).status, 200)
        self.assertFalse(distinct.transport_blocked)

        spaced_server = self.server()
        spaced = self.proxy(spaced_server)
        spaced.RUNAWAY_WINDOW_S = 0.05
        for _ in range(spaced.RUNAWAY_MAX_IDENTICAL_ADMISSIONS + 3):
            self.assertEqual(self.call(spaced, body).status, 200)
            time.sleep(0.06)
        self.assertFalse(spaced.transport_blocked)

    def test_close_writes_a_terminal_into_every_open_stream(self):
        step = Step(text="never delivered", gate=True)
        server = self.server([step])
        proxy = self.proxy(server)
        replies = []
        client = threading.Thread(target=lambda: replies.append(self.call(proxy, request_body(), bookkeeping=None)))
        client.start()
        self.assertTrue(step.gate.wait_entered(5))
        proxy.close()
        client.join(10)
        self.assertEqual(len(replies), 1)
        self.assert_single_error_event(replies[0], "auriga_model_request_failed")
        self.assertNotIn(b"never delivered", replies[0].body)

    def test_local_and_unknown_routes_are_answered_by_the_relay(self):
        server = self.server()
        proxy = self.proxy(server)
        for method, path in (("GET", "/v1/models"), ("GET", "/v1/models/" + "claude-opus-5-5"), ("GET", "/v1/messages"),
                             ("POST", "/v1/complete"), ("POST", "/v1/messages/")):
            with self.subTest(method=method, path=path):
                reply = self.call(proxy, path=path, method=method, bookkeeping=None,
                                  data=b"" if method == "GET" else b"{}")
                self.assertEqual(reply.status, 404)
                self.assertEqual(reply.json()["error"]["type"], "not_found_error")
        self.assertEqual(server.requests, [])
        self.assertEqual(proxy.logical_call_count, 0)

    def test_worker_subprocess_receives_wire_and_client_profile(self):
        server = self.server([text_step("from worker")])
        proxy = self.proxy(server)
        original = model_proxy.subprocess.Popen
        with mock.patch.object(model_proxy.subprocess, "Popen", side_effect=lambda *a, **k: original(*a, **k)) as popen:
            reply = self.call(proxy, request_body())
        self.assertTrue(popen.called)
        self.assertEqual(events(reply.body)[-1][0], "message_stop")
        self.assertEqual(server.generation_requests()[0].path, "/v1/messages?beta=true")
        self.assertEqual(proxy.request_count, 1)


if __name__ == "__main__":
    unittest.main()
