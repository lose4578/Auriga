"""Unit tests of the Anthropic Messages wire adapter (P1-2) and its egress information-boundary walker.

The egress tests feed (1) one synthetic body touching every row of the design §4.4 table with planted
digests, artifact references, long ids and a private image path, and (2) real Claude Code 2.1.321 /
opencode 1.19.5 request bodies captured by the spikes (``tests/fixtures/anthropic_egress_bodies.json``).
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
import unittest

from auriga.apps.reference.reference_api import upstream
from auriga.core.presentation.actor_presentation import ActorPresentation
from auriga.core.storage.store import Store
from auriga.cortex.session.usage import TOKEN_FIELDS
from auriga.cortex.session.wire import (AnthropicMessagesWire, CODEX_PROFILE, ClientProfile, OpenAIResponsesWire,
                                        Route, WireAdapter, WireReply, present_messages_request, wire_from_name)
from auriga.cortex.session.wire import anthropic_messages as am

WIRE = AnthropicMessagesWire()
POLICY = upstream().joint
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "anthropic_egress_bodies.json"
DIGEST = "0123456789abcdef" * 4
ARTIFACT = "artifact:" + "a" * 64 + "-" + "b" * 16
LONG_OBSERVATION = "observation-" + "c" * 32
PLANTED = f"sha256={DIGEST} see {ARTIFACT} and {LONG_OBSERVATION}"
PRIVATE_IMAGE_BLOCK = ("[Image: source: " + "/private/run/claude-config/projects/-private-run-actor/"
                       + "9ff8e771-aa08-4c17-b00d-032e1cb2011f/tool-results/mcp-auriga-blob-1794338642782-z5ip6n.png]")
MODELS_ID_PATH = "/v1/models/" + "claude-opus-5-5"
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="
PDF = "JVBERi0xLjQKJcfsj6IKMSAwIG9iago8PD4+CmVuZG9iagp0cmFpbGVyCjw8Pj4KJSVFT0YK"
PREFIX = "mcp__auriga__"
REGISTERED = frozenset({"robot_observe", "robot_move"})
FORBIDDEN_WORDS = ("rate limit", "overloaded", "timeout", "try again", "capacity", "server error", "connection",
                   "unavailable")


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def message_start(**usage):
    return sse("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-served", "content": [],
        "stop_reason": None, "usage": dict(dict(input_tokens=10, cache_read_input_tokens=4,
                                                cache_creation_input_tokens=2, output_tokens=1), **usage)}})


def block_start(index, block):
    return sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})


def delta(index, value):
    return sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": value})


def block_stop(index):
    return sse("content_block_stop", {"type": "content_block_stop", "index": index})


def message_end(stop_reason, output_tokens=7, **extra):
    return (sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                  "usage": {"output_tokens": output_tokens}, **extra})
            + sse("message_stop", {"type": "message_stop"}))


def tool_stream(stop_reason, arguments, *, closed=True):
    frames = [message_start(), block_start(0, {"type": "text", "text": ""}), delta(0, {"type": "text_delta", "text": "hi"}),
              block_stop(0), block_start(1, {"type": "tool_use", "id": "toolu_1", "name": "mcp__auriga__robot_observe",
                                             "input": {}})]
    frames += [delta(1, {"type": "input_json_delta", "partial_json": piece}) for piece in arguments]
    if closed:
        frames.append(block_stop(1))
    return b"".join(frames) + message_end(stop_reason)


def generation():
    return WIRE.route("POST", "/v1/messages")


class ProfileTest(unittest.TestCase):
    def test_client_profiles(self):
        self.assertEqual((WIRE.client, WIRE.early_stream, WIRE.client_loop_guard), ("claude_code", True, True))
        self.assertEqual(AnthropicMessagesWire(client="claude_code").client, "claude_code")
        for bad in ("codex", "gemini_cli", "", 3):  # "opencode" became a supported profile in P3-4
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "unsupported_wire_client:anthropic_messages"):
                AnthropicMessagesWire(client=bad)
        codex = OpenAIResponsesWire()
        self.assertEqual((codex.client, codex.early_stream, codex.client_loop_guard), ("codex", True, False))
        self.assertEqual(OpenAIResponsesWire(client="codex").client, "codex")
        with self.assertRaisesRegex(ValueError, "unsupported_wire_client:openai_responses:claude_code"):
            OpenAIResponsesWire(client="claude_code")
        base = WireAdapter()
        self.assertEqual((base.client, base.early_stream, base.client_loop_guard, base.keepalive_interval_s),
                         ("codex", True, False, 1.0))
        with self.assertRaisesRegex(ValueError, "unsupported_wire_client"):
            WireAdapter(client="claude_code")
        self.assertEqual(CODEX_PROFILE, ClientProfile(True, False))
        self.assertEqual(ClientProfile(), CODEX_PROFILE)

    def test_wire_from_name(self):
        self.assertIsInstance(wire_from_name("openai_responses"), OpenAIResponsesWire)
        restored = wire_from_name("anthropic_messages", client="claude_code")
        self.assertIsInstance(restored, AnthropicMessagesWire)
        self.assertEqual(restored.client, "claude_code")
        with self.assertRaisesRegex(ValueError, "unknown_model_wire:anthropic_messages:client_profile_required"):
            wire_from_name("anthropic_messages")
        with self.assertRaisesRegex(ValueError, "unsupported_wire_client:anthropic_messages:gemini_cli"):
            wire_from_name("anthropic_messages", client="gemini_cli")  # "opencode" is supported since P3-4
        with self.assertRaisesRegex(ValueError, "unknown_model_wire:openai_chat"):
            wire_from_name("openai_chat", client="opencode")

    def test_constants(self):
        self.assertEqual(WIRE.name, "anthropic_messages")
        self.assertEqual(dict(WIRE.auth_header_names), {"x-api-key": "x-api-key", "authorization": "Authorization"})
        self.assertEqual((WIRE.body_limit_bytes, WIRE.trim_target_bytes),
                         (OpenAIResponsesWire.body_limit_bytes, OpenAIResponsesWire.trim_target_bytes))
        self.assertFalse(WIRE.supports_service_tier)
        self.assertEqual(WIRE.methods, frozenset({"POST", "GET"}))
        self.assertEqual(OpenAIResponsesWire.methods, frozenset({"POST"}))
        # Claude Code resets its stream watchdog on pings spaced at most 10 s apart (spike S5).
        self.assertLessEqual(WIRE.keepalive_interval_s, 10.0)
        self.assertGreater(WIRE.keepalive_interval_s, 0)


class IngressTest(unittest.TestCase):
    def test_routes(self):
        self.assertEqual(generation(), Route("generation", "/v1/messages", "/v1/messages"))
        self.assertEqual(WIRE.route("POST", "/v1/messages?beta=true"),
                         Route("generation", "/v1/messages", "/v1/messages?beta=true"))
        self.assertEqual(WIRE.route("POST", "/v1/messages/count_tokens?beta=true"),
                         Route("auxiliary", "/v1/messages/count_tokens", "/v1/messages/count_tokens?beta=true"))
        self.assertEqual(WIRE.route("GET", "/v1/models"), Route("local", "/v1/models", ""))
        self.assertEqual(WIRE.route("GET", MODELS_ID_PATH).kind, "local")
        for method, path in (("GET", "/v1/messages"), ("POST", "/v1/messages/"), ("POST", "/messages"),
                             ("POST", "/v1/models"), ("PUT", "/v1/messages"), ("POST", "/v1/complete"),
                             ("POST", "/responses"), ("POST", None)):
            with self.subTest(method=method, path=path):
                self.assertIsNone(WIRE.route(method, path))
        self.assertEqual(WIRE.route_for_endpoint("/v1/messages"), generation())
        self.assertFalse(WIRE.route_for_endpoint("/v1/messages").compaction)

    def test_request_fields_and_thread_key(self):
        import http.client
        import email
        message = email.message_from_string("x-claude-code-session-id: sess-1\n\n", _class=http.client.HTTPMessage)
        self.assertEqual(WIRE.thread_key(message), "sess-1")
        self.assertEqual(WIRE.thread_key({"X-Claude-Code-Session-Id": "sess-2"}), "sess-2")
        self.assertEqual(WIRE.thread_key({"thread-id": "t"}), "")
        self.assertEqual(WIRE.thread_key({}), "")
        self.assertTrue(WIRE.is_streaming({"stream": True}))
        self.assertFalse(WIRE.is_streaming({"stream": "true"}))
        self.assertFalse(WIRE.is_streaming([]))
        self.assertEqual(WIRE.requested_model({"model": "claude-opus-5-5"}), "claude-opus-5-5")
        self.assertIsNone(WIRE.requested_model({"model": 3}))

    def test_gzip_decoding(self):
        raw = json.dumps({"model": "m", "messages": []}).encode()
        self.assertIs(WIRE.decode_body({}, raw), raw)
        self.assertIs(WIRE.decode_body({"Content-Encoding": "identity"}, raw), raw)
        self.assertEqual(WIRE.decode_body({"content-encoding": "gzip"}, gzip.compress(raw)), raw)
        with self.assertRaisesRegex(ValueError, "unsupported_request_content_encoding"):
            WIRE.decode_body({"Content-Encoding": "br"}, raw)
        with self.assertRaisesRegex(ValueError, "invalid_gzip_request_body"):
            WIRE.decode_body({"Content-Encoding": "gzip"}, b"not gzip")
        with self.assertRaisesRegex(ValueError, "invalid_gzip_request_body"):
            WIRE.decode_body({"Content-Encoding": "gzip"}, gzip.compress(raw)[:-12])
        small = AnthropicMessagesWire()
        small.max_decoded_body_bytes = 100
        with self.assertRaisesRegex(ValueError, "decoded_request_body_too_large"):
            small.decode_body({"Content-Encoding": "gzip"}, gzip.compress(b"x" * 10_000))


class StreamTest(unittest.TestCase):
    def test_terminal_detection_and_keepalive(self):
        self.assertEqual(WIRE.terminal_sse_type(b'event: message_stop\ndata: {"type":"message_stop"}'), "message_stop")
        self.assertEqual(WIRE.terminal_sse_type(b'event: error\ndata: {"type":"error","error":{}}'), "error")
        for block in (b'event: ping\ndata: {"type": "ping"}', b'event: message_delta\ndata: {"type":"message_delta"}',
                      b"data: [DONE]", b"event: message_stop", b"data: {not json", b": comment", b""):
            with self.subTest(block=block):
                self.assertIsNone(WIRE.terminal_sse_type(block))
        self.assertTrue(WIRE.terminal_allows_reuse("message_stop"))
        self.assertFalse(WIRE.terminal_allows_reuse("error"))
        self.assertEqual(WIRE.keepalive_frame(), b'event: ping\ndata: {"type": "ping"}\n\n')

    def test_usable_completion_and_evidence(self):
        raw = message_start() + block_start(0, {"type": "thinking", "thinking": ""}) + delta(0, {
            "type": "signature_delta", "signature": "sig"}) + block_stop(0) + sse("ping", {"type": "ping"}) + \
            block_start(1, {"type": "text", "text": ""}) + delta(1, {"type": "text_delta", "text": "ok"}) + \
            block_stop(1) + message_end("end_turn")
        complete, evidence, terminal = WIRE.response_evidence(raw, generation())
        self.assertTrue(complete)
        self.assertEqual(evidence, dict(response_id="msg_1", served_model="claude-served", response_status="completed",
                                        stop_reason="end_turn", usage=dict(input_tokens=10, cache_read_input_tokens=4,
                                        cache_creation_input_tokens=2, output_tokens=7), usage_known=True))
        self.assertEqual(terminal["stop_reason"], "end_turn")
        self.assertEqual(WIRE.completion_outcome(complete, evidence), "usable")
        self.assertFalse(WIRE.context_edit_applied(terminal))

    def test_stop_reasons_and_tool_json_integrity(self):
        cases = [
            ("tool_use", ['{"camera":', ' "head"}'], True, "usable"),
            ("end_turn", ['{}'], True, "usable"),
            ("stop_sequence", ['{"a": 1}'], True, "usable"),
            ("pause_turn", [], True, "usable"),
            ("refusal", [], True, "usable"),
            ("max_tokens", ['{"camera": "head"}'], True, "usable"),
            ("model_context_window_exceeded", ['{"a": [1, 2]}'], True, "usable"),
            ("max_tokens", ['{"camera": "he'], True, "incomplete"),
            ("model_context_window_exceeded", ['{"a": [1,'], True, "incomplete"),
            ("max_tokens", ['{"camera": "head"}'], False, "incomplete"),
            ("tool_use", ['["not", "object"]'], True, "incomplete"),
            ("tool_use", ['{"a": 1}'], False, "incomplete"),
            ("future_reason", ['{"a": 1}'], True, "incomplete"),  # P1-2 review item 5: closed usable set
            ("future_reason", ['{"a"'], True, "incomplete"),
        ]
        for stop_reason, arguments, closed, expected in cases:
            with self.subTest(stop_reason=stop_reason, arguments=arguments, closed=closed):
                complete, evidence, _ = WIRE.response_evidence(tool_stream(stop_reason, arguments, closed=closed),
                                                               generation())
                self.assertEqual(WIRE.completion_outcome(complete, evidence), expected)
                self.assertEqual(evidence["stop_reason"], stop_reason)

    def test_failed_missing_and_malformed_streams(self):
        error = sse("error", {"type": "error", "error": {"type": "overloaded_error", "message": "fixture-private"}})
        for raw in (message_start() + block_start(0, {"type": "text", "text": ""}) + error, error):
            complete, evidence, terminal = WIRE.response_evidence(raw, generation())
            self.assertTrue(complete)
            self.assertEqual(evidence["response_status"], "failed")
            self.assertEqual(evidence["response_error"]["classification"], "server_overloaded")
            self.assertTrue(evidence["response_error"]["retryable"])
            self.assertFalse(evidence["usage_known"])
            self.assertNotIn("fixture-private", json.dumps(evidence))
            self.assertEqual(WIRE.completion_outcome(complete, evidence), "failed")
            self.assertEqual(terminal["status"], "failed")
        with self.assertRaises(EOFError):
            WIRE.response_evidence(message_start() + block_start(0, {"type": "text", "text": ""}), generation())
        for raw in (sse("message_stop", {"type": "message_stop"}),
                    message_start() + message_start(),
                    message_start() + delta(5, {"type": "text_delta", "text": "x"}),
                    message_start() + block_stop(3),
                    message_start() + b"data: [1, 2]\n\n"):
            with self.subTest(raw=raw[:60]), self.assertRaises(ValueError):
                WIRE.response_evidence(raw, generation())
        # message_stop without any stop_reason: not a usable completion (resent like a cut stream)
        complete, evidence, _ = WIRE.response_evidence(message_start() + sse("message_stop", {"type": "message_stop"}),
                                                       generation())
        self.assertEqual(WIRE.completion_outcome(complete, evidence), "not_completed")

    def test_non_streaming_json(self):
        body = dict(id="msg_2", type="message", role="assistant", model="claude-served", stop_reason="tool_use",
                    content=[dict(type="tool_use", id="toolu_1", name="x", input={"a": 1})],
                    usage=dict(input_tokens=3, output_tokens=2),
                    context_management=dict(applied_edits=[dict(type="clear_thinking_20251015")]))
        complete, evidence, terminal = WIRE.response_evidence(json.dumps(body).encode(), generation())
        self.assertEqual(WIRE.completion_outcome(complete, evidence), "usable")
        self.assertEqual(evidence["usage"], dict(input_tokens=3, output_tokens=2))
        self.assertTrue(WIRE.context_edit_applied(terminal))
        error = dict(type="error", error=dict(type="invalid_request_error", message="prompt is too long: 9 > 8"))
        complete, evidence, _ = WIRE.response_evidence(json.dumps(error).encode(), generation())
        self.assertEqual(WIRE.completion_outcome(complete, evidence), "failed")
        self.assertEqual(evidence["response_error"]["classification"], "context_window_exceeded")
        for raw in (b"[]", b'{"type": "message"}', b"not json"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                WIRE.response_evidence(raw, generation())
        streamed_edit = message_start() + message_end("end_turn", context_management={"applied_edits": [{"type": "x"}]})
        self.assertTrue(WIRE.context_edit_applied(WIRE.response_evidence(streamed_edit, generation())[2]))


class ErrorTest(unittest.TestCase):
    def test_failed_response_error_table(self):
        table = {
            ("overloaded_error", "fixture-private"): ("server_overloaded", True),
            ("rate_limit_error", "fixture-private"): ("rate_limit", True),
            ("api_error", "fixture-private"): ("retryable", True),
            ("timeout_error", "fixture-private"): ("retryable", True),
            ("authentication_error", "fixture-private"): ("authentication_error", False),
            ("permission_error", "fixture-private"): ("permission_denied", False),
            ("not_found_error", "model: claude-x"): ("configuration_error", False),
            ("billing_error", "fixture-private"): ("quota_exceeded", False),
            ("request_too_large", "Request exceeds the maximum size"): ("image_rejected", True),
            ("invalid_request_error", "messages: roles must alternate"): ("configuration_error", False),
            ("invalid_request_error", "prompt is too long: 250000 tokens > 200000 maximum"): ("context_window_exceeded", False),
            ("invalid_request_error", "Your credit balance is too low to access the API"): ("quota_exceeded", False),
            ("invalid_request_error", "messages.0.content.1.image.source.base64.data: image exceeds 5 MB maximum"):
                ("image_rejected", True),
            ("invalid_request_error", "Too many images in request"): ("image_rejected", True),
            ("future_error", "fixture-private"): ("retryable", True),
        }
        for (kind, message), (classification, retryable) in table.items():
            with self.subTest(kind=kind, message=message):
                result = WIRE.failed_response_error({"type": "error", "error": {"type": kind, "message": message}})
                self.assertEqual((result["classification"], result["retryable"]), (classification, retryable))
                self.assertEqual(result["type"], kind)
                self.assertNotIn(message, json.dumps(result))
                self.assertEqual(set(result), {"code", "type", "code_state", "classification", "retryable", "retry_after_s"})
        for body in ({}, {"error": "text"}, {"error": {"message": "x"}}, None):
            with self.subTest(body=body):
                self.assertEqual(WIRE.failed_response_error(body)["classification"], "stream")

    def test_http_failure_classification(self):
        def failure(status, error_type=None, message="x", headers=None):
            body = {"type": "error", "error": {"type": error_type, "message": message}} if error_type else None
            return WIRE.classify_http_failure(status, headers or {}, body, policy=POLICY)
        cases = [
            (529, "overloaded_error", "x", True), (500, "api_error", "x", True), (503, None, "x", True),
            (429, "rate_limit_error", "x", True), (401, "authentication_error", "x", False),
            (403, "permission_error", "x", False), (404, "not_found_error", "x", False),
            (400, "invalid_request_error", "messages: bad", False),
            (400, "invalid_request_error", "prompt is too long", False),
            (400, "invalid_request_error", "Your credit balance is too low", False),
            (400, "invalid_request_error", "image exceeds 5 MB maximum", True),
            (413, "request_too_large", "x", True), (413, None, "x", True),
            (503, "not_found_error", "model: x", False),
        ]
        for status, error_type, message, retryable in cases:
            with self.subTest(status=status, error_type=error_type, message=message):
                result = failure(status, error_type, message)
                self.assertIsInstance(result, POLICY.InfrastructureFailure)
                self.assertEqual(result.retryable, retryable)
                self.assertEqual(result.status, status)
        self.assertEqual(failure(429, "rate_limit_error", headers={"Retry-After": "3"}).retry_after_s, 3.0)
        self.assertTrue(WIRE.image_rejected(dict(classification="image_rejected")))
        self.assertTrue(WIRE.image_rejected(dict(http_status=413, classification="stream")))
        self.assertFalse(WIRE.image_rejected(dict(http_status=400, classification="configuration_error")))

    def assert_neutral(self, data):
        text = data.decode()
        self.assertIsNone(re.search(r"\d", text), text)
        for word in FORBIDDEN_WORDS:
            self.assertNotIn(word, text.lower())

    def test_claude_code_denials_relay_errors_and_failed_payload(self):
        for kind in ("thread_cancelled", "transport_blocked", "budget", "client_retry_runaway"):
            with self.subTest(kind=kind):
                reply = WIRE.denial(kind)
                self.assertEqual(reply.status, 403)
                self.assertEqual(dict(reply.headers), {"Content-Type": "application/json", "x-should-retry": "false"})
                body = json.loads(reply.body)
                self.assertEqual(body["type"], "error")
                self.assertEqual(body["error"]["type"], "permission_error")
                self.assertTrue(body["error"]["message"].startswith("auriga_"))
                self.assert_neutral(reply.body)
        self.assertEqual(len({WIRE.denial(kind).body for kind in am.DENIAL_MESSAGES}), 4)
        with self.assertRaises(KeyError):
            WIRE.denial("unknown")
        evidence = dict(http_status=503, classification="retryable", code="abc123", detail="logical-call-9")
        streamed = WIRE.relay_error(streaming=True, evidence=evidence, manual_retry_allowed=True)
        self.assertEqual(streamed.body, b'event: error\ndata: {"type":"error","error":{"type":"invalid_request_error",'
                                        b'"message":"auriga_model_request_failed"}}\n\n')
        plain = WIRE.relay_error(streaming=False, evidence=evidence, manual_retry_allowed=False)
        self.assertEqual((plain.status, dict(plain.headers)), (403, dict(am.NO_RETRY_HEADERS)))
        self.assertEqual(json.loads(plain.body)["error"], {"type": "permission_error", "message": "auriga_model_request_failed"})
        failed = WIRE.failed_payload(dict(status="failed", error={"type": "overloaded_error", "message": "private 42"}),
                                     streaming=True)
        self.assertEqual(failed, b'event: error\ndata: {"type":"error","error":{"type":"invalid_request_error",'
                                 b'"message":"auriga_model_response_unusable"}}\n\n')
        self.assertIsNone(WIRE.failed_payload(dict(status="incomplete"), streaming=False))
        for data in (streamed.body, plain.body, failed):
            self.assert_neutral(data)
        found = WIRE.not_found()
        self.assertEqual(found.status, 404)
        self.assertEqual(json.loads(found.body)["error"]["type"], "not_found_error")
        self.assert_neutral(found.body)


class UsageTest(unittest.TestCase):
    def test_normalize_usage(self):
        full = dict(input_tokens=100, cache_read_input_tokens=40, cache_creation_input_tokens=10, output_tokens=7)
        self.assertEqual(WIRE.normalize_usage(full), dict(input_tokens=150, cached_input_tokens=40,
                         cache_write_input_tokens=10, output_tokens=7, reasoning_output_tokens=None, total_tokens=157))
        self.assertEqual(WIRE.normalize_usage(dict(input_tokens=5, output_tokens=2)),
                         dict(input_tokens=5, cached_input_tokens=None, cache_write_input_tokens=None, output_tokens=2,
                              reasoning_output_tokens=None, total_tokens=7))
        self.assertEqual(WIRE.normalize_usage(dict(input_tokens=5, cache_read_input_tokens=-1, output_tokens=2)),
                         dict(input_tokens=None, cached_input_tokens=None, cache_write_input_tokens=None,
                              output_tokens=2, reasoning_output_tokens=None, total_tokens=None))
        self.assertEqual(WIRE.normalize_usage(dict(cache_read_input_tokens=3, output_tokens="2"))["input_tokens"], None)
        for usage in (None, {}, "x"):
            with self.subTest(usage=usage):
                self.assertEqual(WIRE.normalize_usage(usage), {key: None for key in TOKEN_FIELDS})
        self.assertEqual(tuple(WIRE.normalize_usage(full)), TOKEN_FIELDS)


def observation_text(observation_id, cameras):
    return dict(type="text", text=json.dumps(dict(bundle=dict(
        observation_id=observation_id, cameras=[dict(camera_id=camera) for camera in cameras]))))


def image(seed):
    return dict(type="image", source=dict(type="base64", media_type="image/png", data=seed * 8))


class ImageTest(unittest.TestCase):
    def request(self):
        return dict(messages=[
            dict(role="user", content=[dict(type="text", text="task"), image("a")]),
            dict(role="assistant", content=[dict(type="tool_use", id="t1", name="mcp__auriga__robot_observe", input={})]),
            dict(role="user", content=[
                dict(type="tool_result", tool_use_id="t1", content=[observation_text("obs_1", ["head", "wrist"]),
                                                                     image("b"), dict(type="text", text=PRIVATE_IMAGE_BLOCK),
                                                                     image("c")]),
                dict(type="tool_result", tool_use_id="t2", content=[observation_text("obs_2", ["head"]), image("b")])]),
            dict(role="user", content="string content"),
            dict(role="system", content="env"),
            dict(role="user", content=[image("d"), dict(type="image", source=dict(type="url", url="https://x/y.png"))]),
            dict(role="assistant", content=[dict(type="text", text="ok")]),
        ])

    def test_locations_labels_and_boundaries(self):
        request = self.request()
        self.assertEqual(len(WIRE.image_locations(request)), 5)
        labels = [label for _, _, label in WIRE.image_locations(request, include_labels=True)]
        self.assertEqual(labels, [(None, None), ("obs_1", "head"), ("obs_1", "wrist"), ("obs_2", "head"), (None, None)])
        found, boundaries = WIRE.step_image_boundaries(request)
        self.assertEqual([parts[index] for parts, index in found],
                         [parts[index] for parts, index in WIRE.image_locations(request)])
        self.assertEqual(boundaries, [1, 4, 5])
        for value in (None, [], dict(messages="x"), dict(messages=[None, dict(content=3)])):
            self.assertEqual(WIRE.image_locations(value), [])
            self.assertEqual(WIRE.step_image_boundaries(value), ([], []))

    def test_trim_uses_the_generic_base_logic_with_text_placeholders(self):
        request = self.request()
        original = json.dumps(request)
        trimmed, count = WIRE.trim_image_range(request, 1, 3)
        self.assertEqual(count, 2)
        self.assertEqual(json.dumps(request), original)
        nested = trimmed["messages"][2]["content"][0]["content"]
        self.assertEqual(nested[1], dict(type="text", text=WIRE.omitted_image_text("obs_1", "head")))
        self.assertEqual(nested[3], dict(type="text", text=WIRE.omitted_image_text("obs_1", "wrist")))
        self.assertEqual(len(WIRE.image_locations(trimmed)), 3)
        self.assertEqual(WIRE.placeholder_part("x"), dict(type="text", text="x"))
        self.assertEqual(WIRE.image_part_size(image("a")), 8)
        self.assertEqual(WIRE.image_part_size(dict(type="image", source=dict(type="url"))), 0)
        body = json.dumps(request).encode()
        kept = WIRE.trim_oldest_images(body, keep_latest=2)
        self.assertEqual(len(WIRE.image_locations(json.loads(kept))), 2)
        self.assertIs(WIRE.trim_oldest_images(body, keep_latest=5), body)
        # Same algorithm and same thresholds as the Responses wire (user decision, design rev 3.1).
        self.assertIs(type(WIRE).trim_image_range, WireAdapter.trim_image_range)
        self.assertIs(type(WIRE).trim_oldest_images, WireAdapter.trim_oldest_images)

    def test_exact_reuse_uses_the_shared_helper(self):
        request = self.request()
        result, stats = WIRE.reuse_exact_images(request)
        self.assertEqual(stats, dict(images=5, replaced=1, saved_url_chars=8))
        replaced = result["messages"][2]["content"][1]["content"][1]
        self.assertEqual(replaced["type"], "text")
        self.assertIn("identical pixels to image 2", replaced["text"])
        self.assertEqual(request["messages"][2]["content"][1]["content"][1]["type"], "image")
        self.assertEqual(len(list(WIRE.image_slots(request))), 6)  # the url image is located but never deduplicated


# ======================================================================== egress field table
class Presentation:
    """A real ActorPresentation over a temporary Store."""

    def __init__(self, test):
        directory = tempfile.TemporaryDirectory()
        test.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "run", dict(run_id="egress", episode_id="episode",
                                                              env_instance_id="environment", env_generation=1))
        test.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)
        self.unknown = []

    def project(self, request, wire=WIRE):
        return wire.project_request(request, self.presentation, registered_tools=REGISTERED, tool_name_prefix=PREFIX,
                                    on_unknown_part=self.unknown.append)


def planted_request():
    """One body touching every row of the design §4.4 Anthropic table."""
    schema = {"type": "object", "properties": {"value_sha256": {"type": "string", "description": "digest " + DIGEST},
                                               "camera_id": {"type": "string", "description": "camera " + PLANTED}}}
    return {
        "model": "claude-opus-5-5", "max_tokens": 128000, "stream": True, "betas": ["context-management-2025-06-27"],
        "metadata": {"user_id": json.dumps({"device_id": DIGEST, "account_uuid": "", "session_id": "s"})},
        "thinking": {"type": "adaptive", "display": "updates"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": deepcopy(schema)}},
        "tool_choice": {"type": "auto"},
        "system": [{"type": "text", "text": "Identity " + PLANTED},
                   {"type": "text", "text": "System " + PLANTED, "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": "plain string " + PLANTED},
            {"role": "system", "content": "environment block " + PLANTED},
            {"role": "user", "content": [
                {"type": "text", "text": "user text " + PLANTED, "cache_control": {"type": "ephemeral"},
                 "citations": [{"type": "char_location", "cited_text": "cited " + DIGEST, "document_index": 0,
                                "start_char_index": 0, "end_char_index": 3}]},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
                {"type": "document", "title": "title " + PLANTED, "context": "context " + PLANTED,
                 "source": {"type": "text", "media_type": "text/plain", "data": "document " + PLANTED},
                 "citations": {"enabled": True}},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF}},
                {"type": "document", "title": "chunks", "source": {"type": "content", "content": [
                    {"type": "text", "text": "chunk " + PLANTED}]}},
                {"type": "search_result", "title": "result " + PLANTED, "source": "source " + PLANTED,
                 "content": [{"type": "text", "text": "snippet " + PLANTED}], "citations": {"enabled": True}}]},
            {"role": "system", "content": [{"type": "text", "text": "steer framing " + PLANTED}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "thinking " + PLANTED, "signature": "sig-" + DIGEST},
                {"type": "redacted_thinking", "data": "redacted-" + DIGEST},
                {"type": "text", "text": "assistant " + PLANTED},
                {"type": "tool_use", "id": "toolu_01" + DIGEST[:20], "name": PREFIX + "robot_observe",
                 "input": {"note": PLANTED, "observation_id": LONG_OBSERVATION, "digest_sha256": DIGEST},
                 "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01" + DIGEST[:20], "cache_control": {"type": "ephemeral"},
                 "content": [
                     {"type": "text", "text": json.dumps({"observation_id": LONG_OBSERVATION, "frame_sha256": DIGEST,
                                                          "note": PLANTED})},
                     {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
                     {"type": "text", "text": PRIVATE_IMAGE_BLOCK, "cache_control": {"type": "ephemeral"}},
                     {"type": "tool_reference", "tool_name": PREFIX + "robot_move"},
                     {"type": "text", "text": "caption " + PLANTED}]},
                {"type": "tool_result", "tool_use_id": "toolu_02", "content": "failed " + PLANTED, "is_error": True},
                {"type": "text", "text": "after results " + PLANTED}]},
            {"role": "user", "content": [{"type": "future_block", "text": "future " + PLANTED, "data": "keep-" + DIGEST,
                                          "content": [{"type": "image", "source": {
                                              "type": "base64", "media_type": "image/png", "data": PNG}}]}]},
        ],
        "tools": [
            {"name": "Bash", "description": "Run " + PLANTED, "input_schema": deepcopy(schema)},
            {"name": PREFIX + "robot_observe", "description": "Observe " + PLANTED, "input_schema": deepcopy(schema),
             "cache_control": {"type": "ephemeral"}, "eager_input_streaming": True},
            {"type": "custom", "name": "custom_tool", "description": "Custom " + PLANTED,
             "input_schema": {"type": "object"}, "strict": True},
        ],
    }


def paths_differing(before, after, path=""):
    """Leaf paths (dotted) where two JSON values differ; a key present on one side only is a difference."""
    if isinstance(before, dict) and isinstance(after, dict):
        found = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else str(key)
            if key not in before or key not in after:
                found.append(child)
            else:
                found.extend(paths_differing(before[key], after[key], child))
        return found
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        found = []
        for index, (left, right) in enumerate(zip(before, after)):
            found.extend(paths_differing(left, right, f"{path}.{index}" if path else str(index)))
        return found
    return [] if before == after else [path]


# Locations the walker may rewrite (design §4.4); every other byte of a body must stay identical.
SCRUBBABLE_PATHS = re.compile(r"""^(?:
      system\.\d+\.text | system
    | messages\.\d+\.content
    | messages\.\d+\.content\.\d+\.(?:text|title|context|input(?:\..+)?|content|source)
    | messages\.\d+\.content\.\d+\.source\.(?:data|content(?:\..+)?)
    | messages\.\d+\.content\.\d+\.content\.\d+\.(?:text|content(?:\..+)?)
    | tools\.\d+\.(?:description|input_schema(?:\..+)?)
    | output_config\.format\.schema(?:\..+)?
)$""", re.VERBOSE)


class EgressFieldTableTest(unittest.TestCase):
    def setUp(self):
        self.session = Presentation(self)

    def test_every_scrubbable_field_is_scrubbed(self):
        request = planted_request()
        output = self.session.project(request)
        m = output["messages"]
        tool_result = m[5]["content"][0]["content"]
        scrubbed = [output["system"][0]["text"], output["system"][1]["text"], m[0]["content"], m[1]["content"],
                    m[2]["content"][0]["text"], m[2]["content"][2]["title"], m[2]["content"][2]["context"],
                    m[2]["content"][2]["source"]["data"], m[2]["content"][4]["source"]["content"][0]["text"],
                    m[2]["content"][5]["title"], m[2]["content"][5]["source"], m[2]["content"][5]["content"][0]["text"],
                    m[3]["content"][0]["text"], m[4]["content"][2]["text"], json.dumps(m[4]["content"][3]["input"]),
                    tool_result[0]["text"], tool_result[4]["text"], m[5]["content"][1]["content"],
                    m[5]["content"][2]["text"], m[6]["content"][0]["text"],
                    output["tools"][0]["description"], output["tools"][1]["description"],
                    output["tools"][2]["description"], json.dumps(output["tools"][0]["input_schema"]),
                    json.dumps(output["output_config"]["format"]["schema"])]
        for index, value in enumerate(scrubbed):
            with self.subTest(index=index, value=value[:80]):
                self.assertNotIn(DIGEST, value)
                self.assertNotIn(ARTIFACT, value)
                self.assertNotIn(LONG_OBSERVATION, value)
        self.assertIn("artifact:ref_", output["system"][1]["text"])
        self.assertIn("obs_1", m[4]["content"][3]["input"]["observation_id"])
        self.assertNotIn("digest_sha256", m[4]["content"][3]["input"])
        self.assertNotIn("frame_sha256", tool_result[0]["text"])
        # Claude Code image-source block: exact constant, block and its other keys kept, indices unchanged
        self.assertEqual(tool_result[2], {"type": "text", "text": "[Image]", "cache_control": {"type": "ephemeral"}})
        self.assertEqual(len(tool_result), len(request["messages"][5]["content"][0]["content"]))
        # roles and positions of role:"system" messages unchanged
        self.assertEqual([message["role"] for message in m], [message["role"] for message in request["messages"]])
        self.assertEqual(self.session.unknown, ["future_block"])
        self.assertNotIn(DIGEST, m[6]["content"][0]["text"])
        self.assertEqual(m[6]["content"][0]["data"], "keep-" + DIGEST)

    def test_every_protected_field_is_byte_identical(self):
        request = planted_request()
        output = self.session.project(request)
        m, r = output["messages"], request["messages"]
        protected = [
            (output["model"], request["model"]), (output["max_tokens"], request["max_tokens"]),
            (output["stream"], request["stream"]), (output["betas"], request["betas"]),
            (output["metadata"], request["metadata"]), (output["thinking"], request["thinking"]),
            (output["context_management"], request["context_management"]),
            (output["tool_choice"], request["tool_choice"]),
            (output["output_config"]["effort"], request["output_config"]["effort"]),
            (output["output_config"]["format"]["type"], request["output_config"]["format"]["type"]),
            (output["system"][1]["cache_control"], request["system"][1]["cache_control"]),
            (m[2]["content"][0]["cache_control"], r[2]["content"][0]["cache_control"]),
            (m[2]["content"][0]["citations"], r[2]["content"][0]["citations"]),
            (m[2]["content"][1], r[2]["content"][1]),                        # image block, bytes and media type
            (m[2]["content"][2]["citations"], r[2]["content"][2]["citations"]),
            (m[2]["content"][2]["source"]["media_type"], r[2]["content"][2]["source"]["media_type"]),
            (m[2]["content"][3], r[2]["content"][3]),                        # base64 PDF document
            (m[2]["content"][5]["citations"], r[2]["content"][5]["citations"]),
            (m[4]["content"][0], r[4]["content"][0]),                        # thinking + signature
            (m[4]["content"][1], r[4]["content"][1]),                        # redacted_thinking
            (m[4]["content"][3]["id"], r[4]["content"][3]["id"]), (m[4]["content"][3]["name"], r[4]["content"][3]["name"]),
            (m[4]["content"][3]["cache_control"], r[4]["content"][3]["cache_control"]),
            (m[5]["content"][0]["tool_use_id"], r[5]["content"][0]["tool_use_id"]),
            (m[5]["content"][0]["cache_control"], r[5]["content"][0]["cache_control"]),
            (m[5]["content"][0]["content"][1], r[5]["content"][0]["content"][1]),   # image inside tool_result
            (m[5]["content"][0]["content"][3], r[5]["content"][0]["content"][3]),   # tool_reference
            (m[5]["content"][1]["is_error"], r[5]["content"][1]["is_error"]),
            (m[5]["content"][1]["tool_use_id"], r[5]["content"][1]["tool_use_id"]),
            (m[6]["content"][0]["content"][0], r[6]["content"][0]["content"][0]),   # image nested in unknown block
        ]
        for tool, original in zip(output["tools"], request["tools"]):
            protected += [(tool.get(key), original.get(key)) for key in ("name", "type", "cache_control",
                                                                          "eager_input_streaming", "strict")]
        for index, (value, original) in enumerate(protected):
            with self.subTest(index=index):
                self.assertEqual(json.dumps(value, sort_keys=False), json.dumps(original, sort_keys=False))
        for path in paths_differing(request, output):
            with self.subTest(path=path):
                self.assertRegex(path, SCRUBBABLE_PATHS)
        self.assertEqual(request, planted_request(), "the input body must not be mutated")

    def test_projection_is_deterministic_and_idempotent(self):
        first = self.session.project(planted_request())
        second = self.session.project(planted_request())
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual(json.dumps(self.session.project(first)), json.dumps(first))
        other = ActorPresentation(self.session.store)
        replay = WIRE.project_request(planted_request(), other, registered_tools=REGISTERED, tool_name_prefix=PREFIX)
        self.assertEqual(json.dumps(replay), json.dumps(first))

    def test_each_rejection_fires(self):
        def mutate(change):
            request = planted_request()
            change(request)
            return request
        user = lambda request: request["messages"][2]["content"]
        nested = lambda request: request["messages"][5]["content"][0]["content"]
        cases = [
            (lambda r: r["tools"].append({"type": "web_search_20260209", "name": "web_search"}), "sandbox_remote_model_tool_disabled"),
            (lambda r: r["tools"].append({"type": "code_execution_20260521", "name": "code_execution"}), "sandbox_remote_model_tool_disabled"),
            (lambda r: r["tools"].append({"type": "computer_toolset_20260801"}), "sandbox_remote_model_tool_disabled"),
            (lambda r: r["tools"].append({"type": "tool_search_tool_regex_20251119", "name": "s"}), "sandbox_remote_model_tool_disabled"),
            (lambda r: r["tools"].append({"type": "memory_20250818", "name": "memory"}), "sandbox_remote_model_tool_disabled"),
            (lambda r: r["tools"].append("Bash"), "sandbox_remote_model_tool_disabled"),
            (lambda r: r.update(mcp_servers=[{"type": "url", "url": "https://x", "name": "x"}]), "sandbox_remote_model_tool_disabled"),
            (lambda r: r.update(container="container_1"), "sandbox_remote_model_tool_disabled"),
            (lambda r: user(r).append({"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: user(r).append({"type": "image", "source": {"type": "file", "file_id": "file_1"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: nested(r).append({"type": "image", "source": {"type": "url", "url": "https://x"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: user(r).append({"type": "document", "source": {"type": "url", "url": "https://x/a.pdf"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: user(r).append({"type": "document", "source": {"type": "file", "file_id": "file_1"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: r["messages"][6]["content"][0]["content"].append({"type": "image", "source": {"type": "url", "url": "x"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: nested(r).append({"type": "tool_reference", "tool_name": PREFIX + "unregistered"}), "native_discovered_tool_not_registered"),
            (lambda r: nested(r).append({"type": "tool_reference", "tool_name": "robot_move"}), "native_discovered_tool_not_registered"),
        ]
        for block_type in ("server_tool_use", "web_search_tool_result", "web_fetch_tool_result", "code_execution_tool_result",
                           "mcp_tool_use", "mcp_tool_result", "container_upload", "bash_code_execution_tool_result",
                           "future_server_tool_result"):
            cases.append((lambda r, t=block_type: r["messages"][4]["content"].append({"type": t, "id": "srvtoolu_1"}),
                          "sandbox_remote_model_tool_disabled"))
        for index, (change, error) in enumerate(cases):
            with self.subTest(index=index, error=error), self.assertRaisesRegex(ValueError, error):
                self.session.project(mutate(change))
        with self.assertRaisesRegex(ValueError, "model_request_not_object"):
            self.session.project([])

    def test_image_source_rewrite_is_profile_and_position_specific(self):
        request = dict(messages=[dict(role="user", content=[
            dict(type="tool_result", tool_use_id="t", content=[dict(type="text", text=PRIVATE_IMAGE_BLOCK),
                                                               dict(type="text", text=PRIVATE_IMAGE_BLOCK + " extra"),
                                                               dict(type="text", text="[Image: source: a\nb]")]),
            dict(type="text", text=PRIVATE_IMAGE_BLOCK)])])
        output = self.session.project(request)
        nested = output["messages"][0]["content"][0]["content"]
        self.assertEqual(nested[0]["text"], "[Image]")
        self.assertNotEqual(nested[1]["text"], "[Image]")
        self.assertNotEqual(nested[2]["text"], "[Image]")
        # P1-2 review item 3: the marker is rewritten outside a tool_result too (claude_code profile).
        self.assertEqual(output["messages"][0]["content"][1]["text"], "[Image]")
        other = present_messages_request(request, self.session.presentation, registered_tools=REGISTERED,
                                         tool_name_prefix=PREFIX, client="opencode")
        self.assertNotEqual(other["messages"][0]["content"][0]["content"][0]["text"], "[Image]")
        self.assertEqual(self.session.project(output), output)

    def test_absent_and_minimal_fields(self):
        self.assertEqual(self.session.project({}), {})
        minimal = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}], "system": "plain"}
        self.assertEqual(self.session.project(minimal), minimal)
        stringy = {"system": "System " + PLANTED, "messages": [{"role": "user"}, "odd", {"role": "user", "content": 5}]}
        output = self.session.project(stringy)
        self.assertNotIn(DIGEST, output["system"])
        self.assertEqual(output["messages"], stringy["messages"])


class RealCapturedBodiesTest(unittest.TestCase):
    """Real request bodies of Claude Code 2.1.321 and opencode 1.19.5 (spike captures, sanitised)."""

    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURE.read_text())["cases"]

    def setUp(self):
        self.session = Presentation(self)

    @staticmethod
    def plant(body):
        """Plant digests/artifacts/long ids into the prose locations Claude Code fills with host data."""
        body = deepcopy(body)
        if isinstance(body.get("system"), list) and body["system"]:
            body["system"][-1]["text"] += "\n" + PLANTED
        for message in body.get("messages", []):
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content + " " + PLANTED
                continue
            for block in content:
                if block.get("type") == "text" and not am.IMAGE_SOURCE_BLOCK.fullmatch(block["text"]):
                    block["text"] += " " + PLANTED
                elif block.get("type") == "tool_result":
                    if isinstance(block.get("content"), str):
                        block["content"] += " " + PLANTED
                    else:
                        for part in block.get("content", []):
                            if part.get("type") == "text" and not am.IMAGE_SOURCE_BLOCK.fullmatch(part["text"]):
                                part["text"] += " " + PLANTED
        return body

    def test_fixture_covers_the_observed_shapes(self):
        self.assertEqual(set(self.cases), {"claude_code_tool_results", "claude_code_thinking_parallel",
                                           "claude_code_post_compaction", "claude_code_count_tokens",
                                           "opencode_anthropic_session"})
        text = json.dumps(self.cases)
        self.assertIn('"role": "system"', text)
        self.assertIn('"type": "thinking"', text)
        self.assertIn('"eager_input_streaming": true', text)
        self.assertGreaterEqual(len(am.IMAGE_SOURCE_BLOCK.findall(text)), 4)
        for name, case in self.cases.items():
            with self.subTest(name=name):
                route = WIRE.route(case["method"], case["path"])
                self.assertEqual(route.kind, "auxiliary" if "count_tokens" in name else "generation")

    def test_planted_values_never_survive_and_protocol_bytes_stay_identical(self):
        for name, case in self.cases.items():
            with self.subTest(name=name):
                original = case["body"]
                planted = self.plant(original)
                output = self.session.project(planted)
                serialized = json.dumps(output)
                for value in (DIGEST, ARTIFACT, LONG_OBSERVATION):
                    self.assertNotIn(value, serialized)
                for path in paths_differing(planted, output):
                    self.assertRegex(path, SCRUBBABLE_PATHS)
                self.assertNotIn("[Image: source:", serialized)
                self.assertEqual(serialized.count('"text": "[Image]"'),
                                 len(am.IMAGE_SOURCE_BLOCK.findall(json.dumps(original))))
                for key in ("model", "max_tokens", "stream", "metadata", "thinking", "context_management",
                            "tool_choice"):
                    self.assertEqual(output.get(key), original.get(key), key)
                self.assertEqual([t["name"] for t in output.get("tools", [])], [t["name"] for t in original.get("tools", [])])
                self.assertEqual(self._protected(output), self._protected(original))
                self.assertEqual(json.dumps(self.session.project(output)), serialized)
                self.assertEqual(self.session.unknown, [])

    def test_unplanted_bodies_change_only_scrubbable_prose(self):
        for name, case in self.cases.items():
            with self.subTest(name=name):
                output = self.session.project(case["body"])
                for path in paths_differing(case["body"], output):
                    self.assertRegex(path, SCRUBBABLE_PATHS)

    @staticmethod
    def _protected(body):
        """Every protocol-bearing block field of a body, in order."""
        found = []
        for message in body.get("messages", []):
            found.append(message.get("role"))
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                kind = block.get("type")
                found.append((kind, block.get("cache_control"), block.get("id"), block.get("name"),
                              block.get("tool_use_id"), block.get("is_error")))
                if kind in ("thinking", "redacted_thinking", "image"):
                    found.append(json.dumps(block))
                for part in block.get("content", []) if isinstance(block.get("content"), list) else []:
                    found.append((part.get("type"), part.get("cache_control")))
                    if part.get("type") == "image":
                        found.append(json.dumps(part))
        for tool in body.get("tools", []):
            found.append((tool.get("name"), tool.get("cache_control"), tool.get("eager_input_streaming")))
        for block in body.get("system", []) if isinstance(body.get("system"), list) else []:
            found.append(block.get("cache_control"))
        return found


if __name__ == "__main__":
    unittest.main()
