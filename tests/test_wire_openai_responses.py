"""Unit tests of the WireAdapter base and the OpenAI Responses wire (P0-2 extraction)."""
import base64
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

from auriga.apps.reference.reference_api import upstream
from auriga.core.presentation.image_reuse import reuse_exact_images, responses_image_slots
from auriga.cortex.session import model_proxy
from auriga.cortex.session.model_proxy import (AUTH_HEADERS, CLIENT_AUTH_HEADERS, ModelRequestProxy,
                                               upstream_auth_headers_copy)
from auriga.cortex.session.usage import TOKEN_FIELDS, account_responses_usage
from auriga.cortex.session.wire import (OpenAIResponsesWire, Route, WireAdapter, WireReply, wire_from_name)

import wire_golden_fixture as golden

WIRE = OpenAIResponsesWire()
GENERATION = WIRE.route("POST", "/responses")
COMPACT = WIRE.route("POST", "/responses/compact")


def sse_block(value):
    return b"data: " + json.dumps(value).encode()


def stream(*events):
    return b"".join(sse_block(event) + b"\n\n" for event in events)


def response(status, **fields):
    return dict(id="resp-1", model="served-model", status=status, output=[], **fields)


def image(seed):
    return dict(type="input_image", image_url="data:image/png;base64," + seed * 8)


def observation_text(observation_id, cameras):
    return dict(type="input_text", text=json.dumps(dict(bundle=dict(
        observation_id=observation_id, cameras=[dict(camera_id=camera) for camera in cameras]))))


class RouteTest(unittest.TestCase):
    def test_exact_codex_routes(self):
        self.assertEqual(GENERATION, Route("generation", "/responses", "/responses", False))
        self.assertEqual(COMPACT, Route("generation", "/responses/compact", "/responses/compact", True))
        for method, path in (("GET", "/responses"), ("POST", "/v1/responses"), ("POST", "/responses/"),
                             ("POST", "/responses?stream=1"), ("POST", "/responses/compact/"), ("POST", "/")):
            with self.subTest(method=method, path=path):
                self.assertIsNone(WIRE.route(method, path))

    def test_endpoint_lookup_falls_back_to_plain_generation(self):
        self.assertTrue(WIRE.route_for_endpoint("/responses/compact").compaction)
        self.assertEqual(WIRE.route_for_endpoint("/v1/other"), Route("generation", "/v1/other", "/v1/other"))
        self.assertFalse(WIRE.route_for_endpoint(None).compaction)

    def test_wire_constants_are_unchanged(self):
        self.assertEqual(WIRE.name, "openai_responses")
        self.assertEqual(dict(WIRE.auth_header_names), AUTH_HEADERS)
        self.assertEqual(AUTH_HEADERS, {"authorization": "Authorization", "chatgpt-account-id": "ChatGPT-Account-Id"})
        self.assertEqual(CLIENT_AUTH_HEADERS, {"authorization", "chatgpt-account-id", "cookie", "x-api-key", "api-key",
                                               "openai-organization", "openai-project"})
        self.assertEqual((WIRE.body_limit_bytes, WIRE.trim_target_bytes), (4_500_000, 2_400_000))
        self.assertTrue(WIRE.supports_service_tier)
        self.assertIs(wire_from_name("openai_responses").__class__, OpenAIResponsesWire)
        with self.assertRaisesRegex(ValueError, "unknown_model_wire:anthropic_messages"):
            wire_from_name("anthropic_messages")

    def test_request_fields(self):
        self.assertEqual(WIRE.thread_key({"thread-id": "t1"}), "t1")
        self.assertEqual(WIRE.thread_key({}), "")
        self.assertTrue(WIRE.is_streaming({"stream": True}))
        self.assertFalse(WIRE.is_streaming({"stream": "true"}))
        self.assertFalse(WIRE.is_streaming({}))
        self.assertEqual(WIRE.requested_model({"model": "m"}), "m")
        self.assertIsNone(WIRE.requested_model({}))
        self.assertEqual(WIRE.decode_body({}, b"raw"), b"raw")


class TerminalTest(unittest.TestCase):
    def test_terminal_types(self):
        for event_type in ("response.completed", "response.failed", "response.incomplete", "error"):
            with self.subTest(event_type=event_type):
                self.assertEqual(WIRE.terminal_sse_type(sse_block(dict(type=event_type))), event_type)
                self.assertTrue(model_proxy._terminal_sse_block(sse_block(dict(type=event_type))))
        for block in (sse_block(dict(type="response.created")), sse_block(dict(type="response.output_text.delta")),
                      b"data: [DONE]", b": comment", b"event: response.completed", b"data: {not json",
                      b'data: ["response.completed"]', b"data: \xff\xfe", b""):
            with self.subTest(block=block):
                self.assertIsNone(WIRE.terminal_sse_type(block))
                self.assertIsNone(model_proxy._terminal_sse_type(block))
                self.assertFalse(model_proxy._terminal_sse_block(block))

    def test_terminal_data_lines_join_and_event_lines_are_ignored(self):
        block = b'event: response.completed\ndata:{"type":\ndata:  "response.completed"}'
        self.assertEqual(WIRE.terminal_sse_type(block), "response.completed")
        self.assertEqual(WIRE.terminal_sse_type(b'event: x\r\ndata: {"type":"error"}'), "error")

    def test_keepalive_and_connection_reuse(self):
        self.assertEqual(WIRE.keepalive_frame(), b'event: auriga.keepalive\ndata: {"type":"auriga.keepalive"}\n\n')
        self.assertTrue(WIRE.terminal_allows_reuse("response.completed"))
        for terminal in ("response.failed", "response.incomplete", "error", None):
            self.assertFalse(WIRE.terminal_allows_reuse(terminal))
        self.assertTrue(WIRE.is_event_stream(b"  \ndata: x"))
        self.assertTrue(WIRE.is_event_stream(bytearray(b": ping")))
        self.assertFalse(WIRE.is_event_stream(b'{"status":"completed"}'))


class EvidenceTest(unittest.TestCase):
    def outcome(self, raw, route=GENERATION):
        complete, evidence, terminal = WIRE.response_evidence(raw, route)
        return WIRE.completion_outcome(complete, evidence), evidence, terminal

    def test_completed_stream_is_usable_with_served_model_and_raw_usage(self):
        usage = dict(input_tokens=5, output_tokens=2)
        outcome, evidence, terminal = self.outcome(stream(
            dict(type="response.output_text.delta", delta="x"),
            dict(type="response.completed", response=response("completed", usage=usage))))
        self.assertEqual(outcome, "usable")
        self.assertEqual(evidence, dict(response_id="resp-1", served_model="served-model", response_status="completed",
                                        usage=usage, usage_known=True))
        self.assertEqual(terminal["status"], "completed")

    def test_failed_and_incomplete_terminals(self):
        error = dict(code="context_length_exceeded", message="private provider text")
        outcome, evidence, _ = self.outcome(stream(dict(type="response.failed", response=response("failed", error=error))))
        self.assertEqual(outcome, "failed")
        self.assertEqual(evidence["response_error"]["classification"], "context_window_exceeded")
        self.assertNotIn("private", json.dumps(evidence))
        outcome, evidence, _ = self.outcome(stream(dict(type="response.incomplete", response=response("incomplete"))))
        self.assertEqual(outcome, "incomplete")
        self.assertNotIn("response_error", evidence)

    def test_error_event_is_a_failed_terminal_without_usage(self):
        outcome, evidence, terminal = self.outcome(stream(dict(type="error", code="server_error", message="private")))
        self.assertEqual(outcome, "failed")
        self.assertEqual(terminal, dict(status="failed", output=[], error=dict(type="error", code="server_error",
                                                                                message="private")))
        self.assertEqual(evidence["usage"], None)
        self.assertFalse(evidence["usage_known"])
        self.assertEqual(evidence["response_error"]["code"], "server_error")

    def test_missing_or_conflicting_terminal_raises(self):
        with self.assertRaises(EOFError):
            WIRE.response_evidence(stream(dict(type="response.created")), GENERATION)
        with self.assertRaisesRegex(ValueError, "conflict"):
            WIRE.response_evidence(stream(dict(type="response.completed", response=response("failed"))), GENERATION)
        with self.assertRaisesRegex(ValueError, "invalid response structure"):
            WIRE.response_evidence(stream(dict(type="response.completed", response=dict(status="completed"))), GENERATION)
        with self.assertRaises(ValueError):
            WIRE.response_evidence(b"data: {broken\n\n", GENERATION)

    def test_json_bodies_and_compaction_route(self):
        compacted = json.dumps(dict(object="response.compaction", output=[], usage={})).encode()
        outcome, evidence, _ = self.outcome(compacted, COMPACT)
        self.assertEqual(outcome, "usable")
        self.assertFalse(evidence["usage_known"])
        self.assertEqual(self.outcome(compacted, GENERATION)[0], "not_completed")
        self.assertEqual(self.outcome(json.dumps(response("in_progress")).encode())[0], "not_completed")
        self.assertEqual(self.outcome(json.dumps(response("completed")).encode())[0], "usable")
        self.assertEqual(ModelRequestProxy._response_evidence(compacted, "/responses/compact"),
                         WIRE.response_evidence(compacted, COMPACT))
        self.assertEqual(ModelRequestProxy._response_evidence(compacted, "/responses"),
                         WIRE.response_evidence(compacted, GENERATION))

    def test_failed_payload_drops_output_and_keeps_order(self):
        terminal = dict(id="r", status="failed", output=[dict(type="function_call")], error=dict(code="x"))
        self.assertEqual(WIRE.failed_payload(terminal, streaming=True),
                         b'event: response.failed\ndata: {"type": "response.failed", "response": '
                         b'{"id": "r", "status": "failed", "output": [], "error": {"code": "x"}}}\n\n')
        self.assertEqual(WIRE.failed_payload(dict(terminal, status="incomplete"), streaming=False),
                         b'{"id": "r", "status": "incomplete", "output": [], "error": {"code": "x"}}')
        self.assertEqual(terminal["output"], [dict(type="function_call")])


class ErrorTest(unittest.TestCase):
    def classify(self, code=None, type_=None, message=None):
        error = {key: value for key, value in (("code", code), ("type", type_), ("message", message)) if value is not None}
        return WIRE.failed_response_error(dict(error=error))

    def test_classification_table(self):
        table = {
            "context_length_exceeded": ("context_window_exceeded", False),
            "insufficient_quota": ("quota_exceeded", False), "credit_balance_exhausted": ("quota_exceeded", False),
            "invalid_api_key": ("authentication_error", False), "model_not_found": ("configuration_error", False),
            "unsupported_parameter": ("configuration_error", False), "invalid_prompt": ("invalid_prompt", False),
            "image_compression_failed": ("retryable", True), "server_is_overloaded": ("server_overloaded", True),
            "content_filter": ("refusal", False), "flex_unavailable": ("flex_unavailable", False),
            "rate_limit_exceeded": ("rate_limit", True), "slow_down": ("rate_limit", True),
            "server_error": ("retryable", True), "cyber_policy": ("cyber_policy", False),
        }
        for code, (classification, retryable) in table.items():
            with self.subTest(code=code):
                result = self.classify(code, message="secret provider message")
                self.assertEqual((result["classification"], result["retryable"]), (classification, retryable))
                self.assertEqual(result, ModelRequestProxy._failed_response_error(
                    dict(error=dict(code=code, message="secret provider message"))))
                self.assertNotIn("secret", json.dumps(result))
        self.assertEqual(self.classify(type_="invalid_request_error")["classification"], "configuration_error")
        self.assertEqual(self.classify(code="unknown", type_="overloaded")["classification"], "server_overloaded")

    def test_sanitising_and_retry_after(self):
        self.assertEqual(self.classify("rate_limit_exceeded", message="Please try again in 250ms.")["retry_after_s"], .25)
        self.assertEqual(self.classify("rate_limit_exceeded", message="try again in 3 seconds")["retry_after_s"], 3.0)
        self.assertIsNone(self.classify("rate_limit_exceeded", message="try again in 0s")["retry_after_s"])
        self.assertIsNone(self.classify("server_error", message="try again in 3s")["retry_after_s"])
        unsafe = self.classify("bad code with spaces")
        self.assertEqual((unsafe["code"], unsafe["code_state"]), (None, "redacted_or_invalid"))
        self.assertEqual(self.classify(type_="server_error")["code_state"], "unknown")
        for body in ({}, dict(error="text"), dict(error=dict(code=5)), dict(error=dict(resets_at="soon"))):
            with self.subTest(body=body):
                self.assertEqual(WIRE.failed_response_error(body)["classification"], "stream")
                self.assertTrue(WIRE.failed_response_error(body)["retryable"])

    def test_http_failure_classification_uses_pinned_policy(self):
        policy = upstream().joint
        def failure(status, body, headers=None):
            return WIRE.classify_http_failure(status, headers or {}, body, policy=policy)
        self.assertTrue(failure(503, dict(error=dict(code="server_error"))).retryable)
        self.assertFalse(failure(503, dict(error=dict(code="model_not_found", type="new_api_error"))).retryable)
        self.assertTrue(failure(400, dict(error=dict(code="image_compression_failed"))).retryable)
        self.assertFalse(failure(401, None).retryable)
        limited = failure(429, dict(error=dict(code="rate_limit_exceeded")), {"Retry-After": ".5"})
        self.assertTrue(limited.retryable)
        self.assertEqual(limited.retry_after_s, .5)
        self.assertIsInstance(failure(500, [1, 2]), policy.InfrastructureFailure)
        self.assertTrue(WIRE.image_rejected(dict(code="image_compression_failed")))
        self.assertFalse(WIRE.image_rejected(dict(classification="retryable")))

    def test_relay_errors_and_denials(self):
        evidence = dict(http_status=503, classification="retryable")
        streamed = WIRE.relay_error(streaming=True, evidence=evidence, manual_retry_allowed=True)
        self.assertEqual(streamed.body, (
            b'event: error\ndata: {"type": "error", "error": {"type": "upstream_error", "code": "model_request_failed", '
            b'"message": "Model request failed; provider-error retry policy exhausted or not applicable.", '
            b'"evidence": {"http_status": 503, "classification": "retryable"}, "manual_retry_allowed": true}}\n\n'))
        plain = WIRE.relay_error(streaming=False, evidence=evidence, manual_retry_allowed=False)
        self.assertEqual((plain.status, dict(plain.headers)), (503, {"Content-Type": "application/json"}))
        self.assertEqual(json.loads(plain.body), dict(error=dict(
            code="model_request_failed",
            message="Model request failed; provider-error retry policy exhausted or not applicable.",
            evidence=evidence, manual_retry_allowed=False)))
        self.assertEqual(WIRE.relay_error(streaming=False, evidence=None, manual_retry_allowed=False).status, 502)
        self.assertEqual(WIRE.denial("thread_cancelled"), WireReply(409, "model thread explicitly cancelled"))
        self.assertEqual(WIRE.denial("transport_blocked"), WireReply(409, "model transport blocked; manual review required"))
        self.assertEqual(WIRE.denial("budget"), WireReply(429, "episode model completion budget unavailable"))
        for kind in ("thread_cancelled", "transport_blocked", "budget"):
            self.assertIsNone(WIRE.denial(kind).body)
        with self.assertRaises(KeyError):
            WIRE.denial("unknown")
        self.assertEqual(WIRE.not_found(), WireReply(404))


class UsageTest(unittest.TestCase):
    def test_normalized_usage_matches_responses_accounting(self):
        samples = [
            dict(input_tokens=11, input_tokens_details=dict(cached_tokens=4, cache_write_tokens=2), output_tokens=3,
                 output_tokens_details=dict(reasoning_tokens=1), total_tokens=14),
            dict(input_tokens=5, output_tokens=2, cache_write_input_tokens=1),
            dict(input_tokens=-1, output_tokens="3", total_tokens=None),
            {}, None, "not usage",
        ]
        for usage in samples:
            with self.subTest(usage=usage):
                normalized = WIRE.normalize_usage(usage)
                self.assertEqual(tuple(normalized), TOKEN_FIELDS)
                accounted = account_responses_usage(usage)["tokens"]
                self.assertEqual(normalized, {key: accounted[key] for key in TOKEN_FIELDS})
        self.assertEqual(WIRE.normalize_usage(dict(input_tokens=5, output_tokens=2)),
                         dict(input_tokens=5, cached_input_tokens=None, cache_write_input_tokens=None,
                              output_tokens=2, reasoning_output_tokens=None, total_tokens=None))
        self.assertFalse(WIRE.context_edit_applied(dict(status="completed")))


class ImageTest(unittest.TestCase):
    def request(self):
        return dict(input=[
            dict(type="message", role="user", content=[dict(type="input_text", text="task"), image("a")]),
            dict(type="function_call", call_id="c1", name="robot_observe", arguments="{}"),
            dict(type="function_call_output", call_id="c1", output=[observation_text("obs_1", ["head", "wrist"]),
                                                                    image("b"), image("c")]),
            dict(type="function_call", call_id="c2", name="robot_observe", arguments="{}"),
            dict(type="custom_tool_call_output", call_id="c2", output=[observation_text("obs_2", ["head"]), image("b")]),
            dict(type="function_call_output", call_id="c3", output=[image("d"),
                 dict(type="input_image", image_url="https://example.org/x.png")]),
            dict(type="message", role="assistant", content=[dict(type="output_text", text="ok")]),
            dict(type="reasoning", content=[image("e")]),
        ])

    def test_locations_labels_and_step_boundaries(self):
        request = self.request()
        locations = WIRE.image_locations(request)
        self.assertEqual(len(locations), 5)
        labels = [label for _, _, label in WIRE.image_locations(request, include_labels=True)]
        self.assertEqual(labels, [(None, None), ("obs_1", "head"), ("obs_1", "wrist"), ("obs_2", "head"), (None, None)])
        found, boundaries = WIRE.step_image_boundaries(request)
        self.assertEqual(len(found), 5)
        self.assertEqual(boundaries, [1, 3, 5])
        self.assertEqual(ModelRequestProxy._step_image_boundaries(request), (found, boundaries))
        self.assertEqual(ModelRequestProxy._inline_image_locations(request, include_labels=True),
                         WIRE.image_locations(request, include_labels=True))
        self.assertEqual(ModelRequestProxy._part_observation_labels(request["input"][2]["output"]),
                         [(None, None), ("obs_1", "head"), ("obs_1", "wrist")])
        for value in (None, [], dict(input="text")):
            self.assertEqual(WIRE.image_locations(value), [])
            self.assertEqual(WIRE.step_image_boundaries(value), ([], []))

    def test_trim_placeholders_and_delegates(self):
        request = self.request()
        original = json.dumps(request)
        trimmed, count = WIRE.trim_image_range(request, 1, 4)
        self.assertEqual(count, 3)
        self.assertEqual(json.dumps(request), original)
        self.assertEqual(trimmed["input"][2]["output"][1], dict(type="input_text", text=(
            "[Earlier historical observation image omitted (observation_id=obs_1, camera=head) to stay within "
            "upstream gateway payload limit; call history_view_observation(observation_id='obs_1') if you need "
            "to inspect this frame.]")))
        self.assertEqual(len(WIRE.image_locations(trimmed)), 2)
        self.assertEqual(ModelRequestProxy._trim_image_range(request, 1, 4), (trimmed, 3))
        self.assertEqual(WIRE.trim_image_range(request, 3, 3), (request, 0))
        self.assertEqual(WIRE.trim_image_range(request, 0, 99)[1], 5)
        self.assertEqual(ModelRequestProxy._trim_prefix_images(request, 2), WIRE.trim_prefix_images(request, 2))
        self.assertEqual(WIRE.omitted_image_text("obs_9", None),
                         "[Earlier historical observation image omitted (observation_id=obs_9) to stay within upstream "
                         "gateway payload limit; call history_view_observation(observation_id='obs_9') if you need to "
                         "inspect this frame.]")
        self.assertEqual(WIRE.omitted_image_text(None, "head"),
                         "[Earlier historical observation image omitted to stay within upstream gateway payload limit.]")
        self.assertEqual(WIRE.placeholder_part("x"), dict(type="input_text", text="x"))
        self.assertEqual(WIRE.image_part_size(image("a")), len(image("a")["image_url"]))

    def test_trim_oldest_images_body(self):
        body = json.dumps(self.request(), ensure_ascii=False).encode()
        out = WIRE.trim_oldest_images(body, keep_latest=2)
        self.assertEqual(len(WIRE.image_locations(json.loads(out))), 2)
        self.assertNotIn(b", ", out.split(b'"text"')[0])
        self.assertEqual(ModelRequestProxy._trim_oldest_images(body, keep_latest=2), out)
        self.assertIs(WIRE.trim_oldest_images(body, keep_latest=5), body)
        self.assertEqual(WIRE.trim_oldest_images(b"not json", keep_latest=0), b"not json")

    def test_reuse_default_locator_and_custom_locator(self):
        request = self.request()
        default = reuse_exact_images(request)
        self.assertEqual(default[1], dict(images=5, replaced=1, saved_url_chars=len(image("b")["image_url"])))
        self.assertEqual(reuse_exact_images(request, locate=responses_image_slots), default)
        self.assertEqual(WIRE.reuse_exact_images(request), default)
        self.assertIn("identical pixels to image 2", default[0]["input"][4]["output"][1]["text"])

        def blocks(value):
            for message in value["messages"]:
                for index, part in enumerate(message["content"]):
                    if part.get("type") == "image":
                        source = part["source"]
                        yield message["content"], index, source["data"] if source.get("type") == "base64" else None
        messages = dict(messages=[dict(role="user", content=[
            dict(type="image", source=dict(type="url", url="https://example.org/x.png")),
            dict(type="image", source=dict(type="base64", media_type="image/png", data="AAAA")),
            dict(type="image", source=dict(type="base64", media_type="image/png", data="AAAA"))])])
        result, stats = reuse_exact_images(messages, locate=blocks, text_part=lambda text: dict(type="text", text=text))
        self.assertEqual(stats, dict(images=2, replaced=1, saved_url_chars=4))
        self.assertEqual(result["messages"][0]["content"][2]["type"], "text")
        self.assertIn("identical pixels to image 2", result["messages"][0]["content"][2]["text"])
        self.assertEqual(messages["messages"][0]["content"][2]["type"], "image")


class BaseAdapterTest(unittest.TestCase):
    def test_base_defaults_and_unimplemented_methods(self):
        base = WireAdapter()
        self.assertEqual(base.decode_body({}, b"x"), b"x")
        self.assertEqual(base.thread_key({"thread-id": "t"}), "")
        self.assertIsNone(base.keepalive_frame())
        self.assertFalse(base.terminal_allows_reuse("response.completed"))
        self.assertFalse(base.image_rejected(dict(code="image_compression_failed")))
        self.assertFalse(base.context_edit_applied({}))
        self.assertEqual(base.not_found(), WireReply(404))
        calls = [lambda: base.route("POST", "/"), lambda: base.is_streaming({}), lambda: base.requested_model({}),
                 lambda: base.conversation_key({}), lambda: base.history_fingerprints({}),
                 lambda: base.project_request({}, None, registered_tools=frozenset(), tool_name_prefix=""),
                 lambda: base.image_locations({}), lambda: base.step_image_boundaries({}),
                 lambda: base.image_part_size({}), lambda: base.placeholder_part("x"), lambda: base.reuse_exact_images({}),
                 lambda: base.terminal_sse_type(b""), lambda: base.response_evidence(b"", None),
                 lambda: base.completion_outcome(True, {}), lambda: base.failed_response_error({}),
                 lambda: base.classify_http_failure(500, {}, None, policy=None),
                 lambda: base.failed_payload({}, streaming=True),
                 lambda: base.relay_error(streaming=True, evidence=None, manual_retry_allowed=False),
                 lambda: base.denial("budget"), lambda: base.normalize_usage({}),
                 lambda: base.build_evidence_request(model="m", base_instructions="", developer_instructions="",
                                                     input_items=[], output_schema={}, effort="low"),
                 lambda: base.evidence_answer({})]
        for index, call in enumerate(calls):
            with self.subTest(index=index), self.assertRaises(NotImplementedError):
                call()


class ProxyWiringTest(unittest.TestCase):
    def setUp(self):
        self.upstream = golden.Upstream()
        self.addCleanup(self.upstream.close)

    def test_constructor_keywords(self):
        proxy = ModelRequestProxy("http://127.0.0.1:1")
        self.assertIsInstance(proxy.wire, OpenAIResponsesWire)
        self.assertIsNone(proxy.default_thread_id)
        self.assertIsNone(proxy.on_context_discontinuity)
        custom = OpenAIResponsesWire()
        self.assertIs(ModelRequestProxy("http://127.0.0.1:1", wire=custom).wire, custom)
        with self.assertRaisesRegex(ValueError, "model_proxy_wire_requires_adapter"):
            ModelRequestProxy("http://127.0.0.1:1", wire="openai_responses")
        for value in ("", 5, "a\nb", "x" * 257):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "model_proxy_invalid_default_thread_id"):
                ModelRequestProxy("http://127.0.0.1:1", default_thread_id=value)
        with self.assertRaisesRegex(ValueError, "model_proxy_context_discontinuity_requires_callable"):
            ModelRequestProxy("http://127.0.0.1:1", on_context_discontinuity="callback")
        self.assertEqual(ModelRequestProxy("http://127.0.0.1:1", default_thread_id="t",
                                           on_context_discontinuity=print).default_thread_id, "t")

    def test_auth_header_names_follow_the_wire(self):
        self.assertEqual(upstream_auth_headers_copy({"authorization": "Bearer a"}), {"Authorization": "Bearer a"})
        with self.assertRaisesRegex(ValueError, "invalid_upstream_auth_headers"):
            upstream_auth_headers_copy({"x-api-key": "k"})
        self.assertEqual(upstream_auth_headers_copy({"X-Api-Key": "k"}, {"x-api-key": "x-api-key"}), {"x-api-key": "k"})

        class KeyWire(OpenAIResponsesWire):
            auth_header_names = {"x-api-key": "x-api-key"}
        with ModelRequestProxy(self.upstream.url, wire=KeyWire(), upstream_auth_headers={"x-api-key": "host"}) as proxy:
            golden.call(proxy)
        headers = dict((key.lower(), value) for key, value in self.upstream.requests[0]["headers"])
        self.assertEqual(headers["x-api-key"], "host")
        self.assertNotIn("authorization", headers)
        self.assertNotIn("chatgpt-account-id", headers)

    def test_proxy_uses_wire_replies_for_denials(self):
        class StrictWire(OpenAIResponsesWire):
            def denial(self, kind):
                return WireReply(400, headers={"Content-Type": "application/json", "x-should-retry": "false"},
                                 body=json.dumps(dict(error=dict(type="invalid_request_error", denial=kind))).encode())
        with ModelRequestProxy(self.upstream.url, wire=StrictWire(), max_requests=1) as proxy:
            self.assertEqual(golden.call(proxy)["status"], 200)
            denied = golden.call(proxy, admitted=False)
        self.assertEqual(denied["status"], 400)
        self.assertIn(["x-should-retry", "false"], denied["headers"])
        self.assertEqual(json.loads(denied["body"]["text"]), dict(error=dict(type="invalid_request_error", denial="budget")))

    def test_default_thread_id_attributes_requests_without_a_registered_key(self):
        with ModelRequestProxy(self.upstream.url, default_thread_id="session-thread") as proxy:
            proxy.register_thread("session-thread", "actor", "req-session")
            proxy.register_thread("helper-thread", "helper", "req-helper")
            golden.call(proxy, headers=golden.CLIENT_HEADERS[:3])
            golden.call(proxy, extra_headers=(("thread-id", "unregistered"),))
            golden.call(proxy, extra_headers=(("thread-id", "helper-thread"),))
            proxy.cancel_thread("helper-thread")
            proxy.unregister_thread("helper-thread")
            cancelled = golden.call(proxy, extra_headers=(("thread-id", "helper-thread"),), admitted=False)
            started = [event for event in proxy.events if event["kind"] == "model_call_started"]
        self.assertEqual([(event["native_thread_id"], event["source"], event["caller_request_id"]) for event in started],
                         [("session-thread", "actor", "req-session"), ("session-thread", "actor", "req-session"),
                          ("helper-thread", "helper", "req-helper")])
        self.assertEqual(cancelled["status"], 409)

    def test_codex_default_keeps_header_attribution(self):
        with ModelRequestProxy(self.upstream.url) as proxy:
            golden.call(proxy, headers=golden.CLIENT_HEADERS[:3])
            started = next(event for event in proxy.events if event["kind"] == "model_call_started")
        self.assertEqual((started["native_thread_id"], started["source"], started["caller_request_id"]), ("", "actor", ""))

    def test_worker_payload_names_the_wire(self):
        payloads = []
        result = dict(status=200, headers={"Content-Type": "text/event-stream"},
                      body=base64.b64encode(golden.completed(1)[2]).decode(), terminal_seen=True)

        class FakeWorker:
            pid, returncode = 0, 0
            def __init__(self, *args, **kwargs):
                self.stdin = self
                self.stdout = iter([json.dumps(dict(result=result, incoming_bytes=1, first_upstream_byte_s=0.0)) + "\n"])
            def write(self, text):
                payloads.append(json.loads(text))
            def close(self): pass
            def poll(self): return 0
            def wait(self, timeout=None): return 0
            def communicate(self, *args, **kwargs): return "", ""
        with ModelRequestProxy(self.upstream.url) as proxy, \
                mock.patch("auriga.cortex.session.model_proxy.subprocess.Popen", FakeWorker):
            self.assertEqual(golden.call(proxy)["status"], 200)
        self.assertEqual(payloads[0]["wire"], "openai_responses")
        self.assertEqual(set(payloads[0]), {"url", "body", "headers", "wire"})

    def run_worker(self, payload):
        output = io.StringIO()
        with mock.patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), redirect_stdout(output):
            model_proxy._http_worker()
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_http_worker_reads_wire_name_with_responses_default(self):
        self.upstream.script = [golden.completed(1), golden.completed(2)]
        payload = dict(url=self.upstream.url + "/responses", body=base64.b64encode(b"{}").decode(),
                       headers={"Content-Type": "application/json"})
        for extra in ({}, dict(wire="openai_responses")):
            with self.subTest(extra=extra):
                lines = self.run_worker(dict(payload, **extra))
                result = lines[-1]["result"]
                self.assertTrue(result["terminal_seen"])
                self.assertEqual(base64.b64decode(result["body"]), golden.completed(len(self.upstream.requests))[2])
                self.assertIn("model_response_terminal_received", [line.get("kind") for line in lines])
        with self.assertRaisesRegex(ValueError, "unknown_model_wire"):
            self.run_worker(dict(payload, wire="unknown"))
        self.assertEqual(len(self.upstream.requests), 2)

    def test_worker_script_imports_adapters_from_its_own_checkout(self):
        import subprocess
        from pathlib import Path
        script = Path(model_proxy.__file__).resolve()
        probe = ("import runpy, sys; sys.argv=['model_proxy.py']; "
                 f"namespace = runpy.run_path({str(script)!r}); "
                 "import auriga.cortex.session.wire as wire; print(wire.__file__)")
        completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60,
                                   cwd="/")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(Path(completed.stdout.strip()).resolve(), script.parent / "wire" / "__init__.py")


if __name__ == "__main__":
    unittest.main()
