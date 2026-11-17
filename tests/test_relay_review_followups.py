"""Regression tests for the P1-2 review follow-ups (review-p1-2.md items 1-9) and the T0 evidence on cut messages.

1. Refusals drain the declared request body first, so a client sending >= 1 MB reads the 403/409/429/404 instead of
   a broken pipe (Anthropic profile and Codex; Codex bytes are pinned by tests/test_wire_golden.py).
2. The generic walker for unknown block types fails closed (every string except protected keys is scrubbed).
3. The Claude Code image-source marker is rewritten wherever it appears in scrubbed prose; image bytes never change.
4./5. Stop reasons: closed usable set; a streamed tool_use cut before its first argument byte under max_tokens is
   incomplete.
6. ``manual_retry`` restarts the runaway count.
7. ``cancel_pending`` tombstones the cancelled threads on non-Codex profiles until ``reopen_thread``/re-registration.
8. Auxiliary route: refused without a projection hook, deadline + close() interruption, sanitised upstream errors.
9. ``tools[].input_examples``, legacy ``output_format.schema`` and non-assistant thinking blocks are scrubbed.
T0: an unvalidated message (truncated after a tool_use start, mid argument JSON, before message_stop) never reaches
   Claude Code: the client sees only keepalives and one error terminal, zero content bytes.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
from unittest import mock

from auriga.core.presentation.actor_presentation import ActorPresentation
from auriga.core.storage.store import Store
from auriga.cortex.session import model_proxy
from auriga.cortex.session.model_proxy import ModelRequestProxy
from auriga.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire, present_messages_request
from auriga.cortex.session.wire import anthropic_messages as am

from support.fake_model_servers import FakeAnthropicServer, FakeResponsesServer, Gate, Step

SESSION = "9ff8e771-aa08-4c17-b00d-032e1cb2011f"
PREFIX = "mcp__auriga__"
DIGEST = "0123456789abcdef" * 4
ARTIFACT = "artifact:" + "a" * 64 + "-" + "b" * 16
LONG_OBSERVATION = "observation-" + "c" * 32
PLANTED = f"sha256={DIGEST} see {ARTIFACT} and {LONG_OBSERVATION}"
PRIVATE_PATH = "/private/run/claude-config/projects/-private-run-actor/" + SESSION
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="
PING = b'event: ping\ndata: {"type": "ping"}\n\n'
CLAUDE_HEADERS = (("Content-Type", "application/json"), ("x-api-key", "auriga-local-relay"),
                  ("anthropic-version", "2023-06-01"), ("X-Claude-Code-Session-Id", SESSION))
CODEX_HEADERS = (("Content-Type", "application/json"), ("thread-id", SESSION))
ONE_MB = 1024 * 1024


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


MESSAGE_START = sse("message_start", {"type": "message_start", "message": {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "m", "content": [], "stop_reason": None,
    "usage": {"input_tokens": 5, "output_tokens": 1}}})
TEXT = (sse("content_block_start", {"type": "content_block_start", "index": 0,
                                    "content_block": {"type": "text", "text": ""}})
        + sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                      "delta": {"type": "text_delta", "text": "visible text"}})
        + sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
TOOL_START = sse("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {
    "type": "tool_use", "id": "toolu_cut", "name": PREFIX + "robot_move", "input": {}}})


def tool_delta(partial):
    return sse("content_block_delta", {"type": "content_block_delta", "index": 1,
                                       "delta": {"type": "input_json_delta", "partial_json": partial}})


TOOL_STOP = sse("content_block_stop", {"type": "content_block_stop", "index": 1})


def end(stop_reason):
    return (sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason},
                                  "usage": {"output_tokens": 3}})
            + sse("message_stop", {"type": "message_stop"}))


def generation_body(text="hello", size=0, **extra):
    """A Messages body; ``size`` pads the user text to roughly that many bytes."""
    padding = "x" * max(0, size)
    return dict(dict(model="fixture-model", max_tokens=64, stream=True,
                     messages=[dict(role="user", content=text + padding)]), **extra)


class RawUpstream:
    """Loopback upstream answering every POST with the next scripted (status, content type, bytes) and closing."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                upstream.requests.append((self.path, body))
                item = upstream.script[min(len(upstream.requests), len(upstream.script)) - 1]
                status, ctype, data = item if isinstance(item, tuple) else (200, "text/event-stream", item)
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
                self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class RelayCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "run", dict(run_id="relay", episode_id="episode",
                                                              env_instance_id="environment", env_generation=1))
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)

    def proxy(self, upstream_url, wire=None, *, project=True, **kwargs):
        wire = wire or AnthropicMessagesWire()
        if project and isinstance(wire, AnthropicMessagesWire):
            kwargs["request_projection"] = lambda request: wire.project_request(
                request, self.presentation, registered_tools=frozenset({"robot_move"}), tool_name_prefix=PREFIX)
        kwargs.setdefault("max_retries", 0)
        auth = {"x-api-key": "host-secret"} if isinstance(wire, AnthropicMessagesWire) else {"Authorization": "Bearer h"}
        proxy = ModelRequestProxy(upstream_url, wire=wire, upstream_auth_headers=auth, **kwargs).start()
        self.addCleanup(proxy.close)
        patch = mock.patch.object(proxy.policy, "_retry_delay_seconds", return_value=0)
        patch.start()
        self.addCleanup(patch.stop)
        return proxy

    @staticmethod
    def call(proxy, data, *, path="/v1/messages", headers=CLAUDE_HEADERS, bookkeeping="model_call_finished",
             timeout=30):
        """POST ``data`` (bytes) and read the reply to EOF; a transport error is returned as ('EXC', name)."""
        if bookkeeping:
            with proxy.lock:
                before = sum(event["kind"] == bookkeeping for event in proxy.events)
        port = int(proxy.base_url.rsplit(":", 1)[1])
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            connection.putrequest("POST", path, skip_accept_encoding=True)
            for key, value in headers:
                connection.putheader(key, value)
            connection.putheader("Content-Length", str(len(data)))
            connection.endheaders(data)
            response = connection.getresponse()
            result = (response.status, {k.lower(): v for k, v in response.getheaders()}, response.read())
        except (OSError, http.client.HTTPException) as error:
            return "EXC", type(error).__name__, b""
        finally:
            connection.close()
        deadline = time.monotonic() + 10
        while bookkeeping and time.monotonic() < deadline:
            with proxy.lock:
                if sum(event["kind"] == bookkeeping for event in proxy.events) > before:
                    break
            time.sleep(0.005)
        return result

    @staticmethod
    def kinds(proxy, kind):
        with proxy.lock:
            return [dict(event) for event in proxy.events if event["kind"] == kind]


# ------------------------------------------------------------------------------------------------- item 1
class LargeBodyRefusalTest(RelayCase):
    """Refusals are read only after the declared body was drained (a >= 1 MB body used to get BrokenPipe)."""

    def refusal_cases(self, wire, headers, path, *, expected):
        server = FakeAnthropicServer() if isinstance(wire, AnthropicMessagesWire) else FakeResponsesServer()
        server.start()
        self.addCleanup(server.close)
        if isinstance(wire, AnthropicMessagesWire):
            big = json.dumps(generation_body(size=ONE_MB + 1000)).encode()
        else:
            big = json.dumps(dict(model="gpt-fixture", stream=True, input=[dict(role="user", content="x" * (ONE_MB + 1000))])).encode()
        self.assertGreater(len(big), ONE_MB)
        results = {}
        cancelled = self.proxy(server.base_url, wire)
        cancelled.register_thread(SESSION, "actor", "req")
        cancelled.cancel_thread(SESSION)
        results["thread_cancelled"] = self.call(cancelled, big, path=path, headers=headers, bookkeeping=None)
        blocked = self.proxy(server.base_url, wire)
        with blocked.lock:
            blocked.blocked_error = dict(logical_call_id="fixture", error_type="Fixture", manual_retry_allowed=True,
                                         evidence=None)
        results["transport_blocked"] = self.call(blocked, big, path=path, headers=headers, bookkeeping=None)
        budget = self.proxy(server.base_url, wire, max_requests=1)
        with budget.lock:
            budget.request_count = 1
        results["budget"] = self.call(budget, big, path=path, headers=headers, bookkeeping="model_budget_exhausted")
        results["not_found"] = self.call(budget, big, path="/unknown/route", headers=headers, bookkeeping=None)
        for kind, status in expected.items():
            with self.subTest(kind=kind):
                self.assertEqual(results[kind][0], status, results[kind][:2])
        self.assertEqual(server.generation_requests(), [])
        return results

    def test_anthropic_profile_large_body_gets_the_single_attempt_403(self):
        results = self.refusal_cases(AnthropicMessagesWire(), CLAUDE_HEADERS, "/v1/messages",
                                     expected=dict(thread_cancelled=403, transport_blocked=403, budget=403,
                                                   not_found=404))
        for kind in ("thread_cancelled", "transport_blocked", "budget"):
            self.assertEqual(results[kind][1]["x-should-retry"], "false")
            self.assertEqual(json.loads(results[kind][2])["error"]["type"], "permission_error")

    def test_codex_large_body_gets_the_unchanged_409_429_pages(self):
        results = self.refusal_cases(OpenAIResponsesWire(), CODEX_HEADERS, "/responses",
                                     expected=dict(thread_cancelled=409, transport_blocked=409, budget=429,
                                                   not_found=404))
        self.assertIn(b"model thread explicitly cancelled", results["thread_cancelled"][2])

    def test_auxiliary_denial_with_large_body(self):
        server = FakeAnthropicServer().start()
        self.addCleanup(server.close)
        proxy = self.proxy(server.base_url)
        proxy.register_thread(SESSION, "actor", "req")
        proxy.cancel_thread(SESSION)
        body = json.dumps(generation_body(size=ONE_MB + 1000)).encode()
        status, headers, data = self.call(proxy, body, path="/v1/messages/count_tokens",
                                          bookkeeping="model_aux_request")
        self.assertEqual((status, json.loads(data)["error"]["message"]), (403, "auriga_model_thread_cancelled"))
        self.assertEqual(server.requests, [])


# ------------------------------------------------------------------------------------- items 2, 3 and 9
class WalkerFollowupsTest(RelayCase):
    def project(self, request, client="claude_code", unknown=None):
        return present_messages_request(request, self.presentation, registered_tools=frozenset({"robot_move"}),
                                        tool_name_prefix=PREFIX, client=client,
                                        on_unknown_part=None if unknown is None else unknown.append)

    def assert_clean(self, value):
        serialized = json.dumps(value)
        for planted in (DIGEST, ARTIFACT, LONG_OBSERVATION):
            self.assertNotIn(planted, serialized)

    def test_generic_walker_fails_closed_at_any_key_and_depth(self):
        unknown_block = dict(type="fixture_future_block", id="blk_" + "9" * 30, body=PLANTED, name=PLANTED,
                             uri="file:///" + PLANTED, thinking=PLANTED, input=dict(command=PLANTED),
                             items=[dict(note=PLANTED), [PLANTED]], data=PNG, signature="sig-fixture",
                             media_type="image/png", cache_control=dict(type="ephemeral"))
        resource = dict(type="resource", resource=dict(uri="file:///tmp/" + PLANTED, mimeType="text/plain",
                                                       text=PLANTED))
        resource_link = dict(type="resource_link", uri="file:///" + PLANTED, name=PLANTED, description=PLANTED)
        request = dict(messages=[dict(role="user", content=[
            unknown_block,
            dict(type="tool_result", tool_use_id="toolu_1", content=[resource, resource_link])])])
        unknown = []
        output = self.project(request, unknown=unknown)
        self.assert_clean(output)
        block = output["messages"][0]["content"][0]
        for key in ("id", "data", "signature", "media_type", "cache_control", "type"):
            self.assertEqual(block[key], unknown_block[key], key)
        self.assertEqual(unknown, ["fixture_future_block", "resource", "resource_link"])
        self.assertEqual(self.project(output), output)

    def test_image_source_marker_is_rewritten_wherever_it_appears(self):
        marker = "[Image: source: " + PRIVATE_PATH + "/tool-results/mcp-auriga-blob-1794338642782-z5ip6n.png]"
        bracket = "[Image: source: " + PRIVATE_PATH + "/odd]dir/blob.png]"
        image = dict(type="image", source=dict(type="base64", media_type="image/png", data=PNG))
        request = dict(system="System text " + marker, messages=[
            dict(role="user", content=[
                dict(type="tool_result", tool_use_id="t", content=[
                    image, dict(type="text", text=marker + "\n"), dict(type="text", text="result\n" + marker + "\nmore"),
                    dict(type="text", text=bracket)]),
                dict(type="tool_result", tool_use_id="u", content="text result " + marker),
                dict(type="text", text=marker),
                dict(type="text", text="[Image: source: " + PRIVATE_PATH + "/unterminated.png")]),
            dict(role="user", content="merged " + marker)])
        output = self.project(request)
        serialized = json.dumps(output)
        self.assertNotIn("Image: source:", serialized)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("odd", serialized)
        nested = output["messages"][0]["content"][0]["content"]
        self.assertEqual(nested[0], image)  # image bytes untouched
        self.assertEqual(nested[1]["text"], "[Image]\n")
        self.assertEqual(nested[2]["text"], "result\n[Image]\nmore")
        self.assertEqual(nested[3]["text"], "[Image]")
        self.assertEqual(output["messages"][0]["content"][2]["text"], "[Image]")
        self.assertEqual(output["messages"][1]["content"], "merged [Image]")
        self.assertEqual(self.project(output), output)
        # Other profiles never rewrite (OpenCode has no such marker).
        self.assertIn(PRIVATE_PATH, json.dumps(self.project(request, client="opencode")))

    def test_input_examples_legacy_output_format_and_non_assistant_thinking_are_scrubbed(self):
        assistant_thinking = dict(type="thinking", thinking="model reasoning " + PLANTED, signature="sig")
        redacted = dict(type="redacted_thinking", data="ciphertext-" + DIGEST)
        request = dict(
            tools=[dict(name=PREFIX + "robot_move", description="d", input_schema=dict(type="object"),
                        input_examples=[dict(target=PLANTED, nested=[PLANTED])])],
            output_format=dict(type="json_schema", schema=dict(type="object", description=PLANTED)),
            messages=[dict(role="user", content=[dict(type="thinking", thinking=PLANTED, signature="sig-user"),
                                                 dict(type="tool_result", tool_use_id="t", content=[
                                                     dict(type="thinking", thinking=PLANTED, signature="s")])]),
                      dict(role="assistant", content=[assistant_thinking, redacted])])
        output = self.project(request)
        self.assert_clean(output["tools"])
        self.assert_clean(output["output_format"])
        self.assertEqual(output["output_format"]["type"], "json_schema")
        self.assert_clean(output["messages"][0])
        self.assertEqual(output["messages"][0]["content"][0]["signature"], "sig-user")
        # Signed assistant output stays byte-identical.
        self.assertEqual(output["messages"][1]["content"], [assistant_thinking, redacted])


# ------------------------------------------------------------------------------------------- items 4, 5
class StopReasonFollowupsTest(unittest.TestCase):
    WIRE = AnthropicMessagesWire()

    def outcome(self, raw):
        complete, evidence, _ = self.WIRE.response_evidence(raw, self.WIRE.route("POST", "/v1/messages"))
        return self.WIRE.completion_outcome(complete, evidence)

    def test_blank_arguments_under_truncation_are_incomplete(self):
        for stop in ("max_tokens", "model_context_window_exceeded"):
            for partial in ("", "   "):
                with self.subTest(stop=stop, partial=partial):
                    raw = MESSAGE_START + TEXT + TOOL_START + tool_delta(partial) + TOOL_STOP + end(stop)
                    self.assertEqual(self.outcome(raw), "incomplete")
                    no_delta = MESSAGE_START + TEXT + TOOL_START + TOOL_STOP + end(stop)
                    self.assertEqual(self.outcome(no_delta), "incomplete")
        # A finished tool call without arguments is fine.
        self.assertEqual(self.outcome(MESSAGE_START + TOOL_START + TOOL_STOP + end("tool_use")), "usable")
        self.assertEqual(self.outcome(MESSAGE_START + TEXT + TOOL_START + tool_delta('{"a": 1}') + TOOL_STOP
                                      + end("max_tokens")), "usable")

    def test_unknown_stop_reason_is_incomplete(self):
        self.assertEqual(self.outcome(MESSAGE_START + TEXT + end("fixture_future_reason")), "incomplete")
        for stop in sorted(am.USABLE_STOP_REASONS):
            self.assertEqual(self.outcome(MESSAGE_START + TEXT + end(stop)), "usable", stop)


# -------------------------------------------------------------------------- T0: cut messages never released
class CutMessageNeverReleasedTest(RelayCase):
    # name -> (upstream bytes, resent by the relay's retry policy). A stream without its terminal is resent within
    # max_retries; a finished but structurally incomplete message is an "incomplete" answer (not retried).
    CUTS = {
        "after_tool_use_start": MESSAGE_START + TEXT + TOOL_START,
        "mid_input_json_delta": MESSAGE_START + TEXT + TOOL_START + tool_delta('{"target": [0.1, '),
        "before_message_stop": (MESSAGE_START + TEXT + TOOL_START + tool_delta('{"target": [0.1]}') + TOOL_STOP
                                + sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                                                        "usage": {"output_tokens": 3}})),
        "closed_but_cut_json_max_tokens": (MESSAGE_START + TEXT + TOOL_START + tool_delta('{"target": [0.1, ')
                                           + TOOL_STOP + end("max_tokens")),
        "blank_arguments_max_tokens": MESSAGE_START + TEXT + TOOL_START + TOOL_STOP + end("max_tokens"),
    }

    def test_client_sees_only_keepalives_and_one_error_terminal(self):
        for retries in (0, 2):
            for name, raw in self.CUTS.items():
                with self.subTest(cut=name, retries=retries):
                    upstream = RawUpstream([raw])
                    self.addCleanup(upstream.close)
                    proxy = self.proxy(upstream.base_url, max_retries=retries)
                    status, headers, body = self.call(proxy, json.dumps(generation_body()).encode())
                    self.assertEqual(status, 200)
                    frames = [frame for frame in body.split(b"\n\n") if frame]
                    errors = [frame for frame in frames if frame.startswith(b"event: error")]
                    self.assertEqual(len(errors), 1, body)
                    self.assertTrue(all(frame + b"\n\n" == PING for frame in frames if frame not in errors), body)
                    for forbidden in (b"tool_use", b"toolu_cut", b"content_block", b"message_start", b"visible text",
                                      b"input_json_delta", b"robot_move"):
                        self.assertNotIn(forbidden, body)
                    resent = name not in ("closed_but_cut_json_max_tokens", "blank_arguments_max_tokens")
                    self.assertEqual(len(upstream.requests), 1 + retries if resent else 1)
                    self.assertEqual(proxy.request_count, 0)


# ------------------------------------------------------------------------------------------- item 6
class RunawayManualRetryTest(RelayCase):
    def test_manual_retry_restarts_the_identical_body_count(self):
        server = FakeAnthropicServer().start()
        self.addCleanup(server.close)
        proxy = self.proxy(server.base_url)
        data = json.dumps(generation_body("loop")).encode()
        limit = proxy.RUNAWAY_MAX_IDENTICAL_ADMISSIONS
        statuses = [self.call(proxy, data)[0] for _ in range(limit + 1)]
        self.assertEqual(statuses, [200] * limit + [403])  # the run exceeded the limit: denied and blocked
        self.assertEqual(proxy.blocked_error["error_type"], "ClientRetryRunaway")
        proxy.manual_retry()
        status, _, body = self.call(proxy, data)
        self.assertEqual(status, 200)
        self.assertIn(b"message_stop", body)  # admitted and delivered, not denied as a runaway
        self.assertFalse(proxy.transport_blocked)


# ------------------------------------------------------------------------------------------- item 7
class CancelPendingTombstoneTest(RelayCase):
    def run_cancelled(self, proxy, server, headers, path, data):
        gate = server.script[0].gate
        replies = []
        worker = threading.Thread(target=lambda: replies.append(self.call(proxy, data, path=path, headers=headers)))
        worker.start()
        self.assertTrue(gate.wait_entered(5))
        proxy.cancel_pending("episode_stopped")
        worker.join(10)
        gate.release()
        return replies[0]

    def test_anthropic_profile_resend_after_cancel_pending_is_denied_until_reopened(self):
        server = FakeAnthropicServer([Step(text="gated", gate=Gate())]).start()
        self.addCleanup(server.close)
        proxy = self.proxy(server.base_url)
        proxy.register_thread(SESSION, "actor", SESSION)
        data = json.dumps(generation_body("same body")).encode()
        status, _, body = self.run_cancelled(proxy, server, CLAUDE_HEADERS, "/v1/messages", data)
        self.assertEqual(status, 200)
        self.assertIn(b"event: error", body)
        resent = self.call(proxy, data, bookkeeping="model_thread_request_rejected")
        self.assertEqual(resent[0], 403)
        self.assertEqual(json.loads(resent[2])["error"]["message"], "auriga_model_thread_cancelled")
        self.assertEqual(len(server.generation_requests()), 1)
        self.assertEqual(proxy.request_count, 0)
        # The session's next turn lifts the tombstone (re-registration or reopen_thread).
        proxy.register_thread(SESSION, "actor", SESSION)
        self.assertEqual(self.call(proxy, data)[0], 200)
        self.assertEqual(proxy.request_count, 1)
        self.assertEqual([e["native_thread_id"] for e in self.kinds(proxy, "model_thread_reopened")], [SESSION])

    def test_cancel_thread_tombstone_stays_permanent(self):
        proxy = ModelRequestProxy("http://127.0.0.1:1", wire=AnthropicMessagesWire())
        proxy.cancel_thread("helper-thread")
        self.assertFalse(proxy.reopen_thread("helper-thread"))
        with self.assertRaisesRegex(ValueError, "model_thread_already_cancelled"):
            proxy.register_thread("helper-thread", "helper", "req")

    def test_codex_cancel_pending_sets_no_tombstone(self):
        server = FakeResponsesServer([Step(text="gated", gate=Gate())]).start()
        self.addCleanup(server.close)
        proxy = self.proxy(server.base_url, OpenAIResponsesWire(), project=False)
        data = json.dumps(dict(model="gpt-fixture", stream=True, input="same body")).encode()
        self.run_cancelled(proxy, server, CODEX_HEADERS, "/responses", data)
        self.assertEqual(proxy.cancelled_threads, {})
        self.assertEqual(self.call(proxy, data, path="/responses", headers=CODEX_HEADERS)[0], 200)


# ------------------------------------------------------------------------------------------- item 8
class HangingCountTokens(FakeAnthropicServer):
    release = None

    def _local(self, h, req):
        if req.method == "POST" and req.route == "/v1/messages/count_tokens":
            self.release.wait(10)
        return super()._local(h, req)


class AuxiliaryFollowupsTest(RelayCase):
    BODY = json.dumps(dict(model="fixture-model", messages=[dict(role="user", content="count " + PLANTED)])).encode()

    def test_no_projection_hook_refuses_instead_of_forwarding_raw(self):
        server = FakeAnthropicServer().start()
        self.addCleanup(server.close)
        proxy = self.proxy(server.base_url, project=False)
        status, _, data = self.call(proxy, self.BODY, path="/v1/messages/count_tokens", bookkeeping="model_aux_request")
        self.assertEqual(status, 403)
        self.assertEqual(server.requests, [])
        self.assertEqual(self.kinds(proxy, "model_aux_request")[0]["denied"], "projection_unavailable")

    def test_upstream_error_is_sanitised(self):
        upstream = RawUpstream([(529, "application/json", json.dumps(
            {"type": "error", "error": {"type": "overloaded_error", "message": "provider-private 529 text"}}).encode())])
        self.addCleanup(upstream.close)
        proxy = self.proxy(upstream.base_url)
        status, headers, data = self.call(proxy, self.BODY, path="/v1/messages/count_tokens",
                                          bookkeeping="model_aux_request")
        self.assertEqual(status, 403)
        self.assertNotIn(b"provider-private", data)
        self.assertIsNone(re.search(rb"\d", data))
        event = self.kinds(proxy, "model_aux_request")[0]
        self.assertEqual((event["http_status"], event["response_error"]["classification"]), (529, "server_overloaded"))
        self.assertNotIn(DIGEST.encode(), upstream.requests[0][1])

    def test_forward_has_a_deadline(self):
        server = HangingCountTokens()
        server.release = threading.Event()
        server.start()
        self.addCleanup(server.close)
        self.addCleanup(server.release.set)
        proxy = self.proxy(server.base_url)
        with mock.patch.object(model_proxy, "AUX_REQUEST_TIMEOUT_S", 0.3):
            started = time.monotonic()
            status, _, _ = self.call(proxy, self.BODY, path="/v1/messages/count_tokens",
                                     bookkeeping="model_aux_request")
        self.assertEqual(status, 403)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(self.kinds(proxy, "model_aux_request")[0]["phase"], "forwarding")

    def test_close_interrupts_an_inflight_forward(self):
        server = HangingCountTokens()
        server.release = threading.Event()
        server.start()
        self.addCleanup(server.close)
        self.addCleanup(server.release.set)
        proxy = self.proxy(server.base_url)
        replies = []
        worker = threading.Thread(target=lambda: replies.append(self.call(
            proxy, self.BODY, path="/v1/messages/count_tokens", bookkeeping=None)))
        worker.start()
        deadline = time.monotonic() + 5
        while not proxy._aux_connections and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(proxy._aux_connections)
        started = time.monotonic()
        proxy.close()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(proxy._aux_connections, set())


if __name__ == "__main__":
    unittest.main()
