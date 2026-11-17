"""Relay upstream bindings and per-wire evidence request builders (design §4.1 "Upstream bindings", §9.2-§9.4; P1-7).

* ``build_evidence_request`` / ``evidence_answer`` on OpenAI Responses (``text.format`` json_schema) and Anthropic
  Messages (``output_config.format`` json_schema, never a forced ``tool_choice``): golden bodies, the Codex capture
  (``tests/fixtures/evidence/codex_output_schema_request.json``, what the pinned Codex app-server sends for
  ``outputSchema``), answer extraction and its failure strings.
* ``ModelRequestProxy.add_upstream``: ``/_auriga/u/<name>`` routing, ``X-Auriga-Thread`` attribution (never forwarded),
  ``identity.upstream``, shared budget and blocked-transport gate, credential isolation between bindings, egress
  projection with the binding's presentation, no continuity tracking / image trim on helper bindings, and the
  ``--http-worker`` subprocess path.
"""
import base64
import http.client
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from auriga.core.presentation.actor_presentation import ActorPresentation
from auriga.core.storage.store import Store
from auriga.cortex.session import model_proxy
from auriga.cortex.session.codex_evidence_agent import BASE_INSTRUCTIONS
from auriga.cortex.session.model_proxy import ModelRequestProxy, UpstreamBinding
from auriga.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire
from auriga.cortex.session.wire import anthropic_messages as am
from auriga.cortex.session.wire import base as wire_base
from auriga.cortex.session.wire import openai_responses as orw

from support.fake_model_servers import Failure, FakeAnthropicServer, FakeResponsesServer, Step, ToolCall, text_step

CAPTURE = Path(__file__).resolve().parent / "fixtures" / "evidence" / "codex_output_schema_request.json"
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="
DATA_URL = "data:image/png;base64," + PNG
DIGEST = "0123456789abcdef" * 4
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["assessment"],
          "properties": {"assessment": {"type": "string", "enum": ["supported", "uncertain"]}}}
ITEMS = [dict(type="text", text="evidence text"), dict(type="image", url=DATA_URL)]
DEVELOPER = "Use only the supplied evidence packet."
SESSION = "9ff8e771-aa08-4c17-b00d-032e1cb2011f"
CLAUDE_HEADERS = (("Content-Type", "application/json"), ("x-api-key", "auriga-local-relay"),
                  ("anthropic-version", "2023-06-01"), ("X-Claude-Code-Session-Id", SESSION))


def build(wire, **overrides):
    kwargs = dict(model="gpt-6.1-sol", base_instructions=BASE_INSTRUCTIONS, developer_instructions=DEVELOPER,
                  input_items=ITEMS, output_schema=SCHEMA, effort="high")
    kwargs.update(overrides)
    return wire.build_evidence_request(**kwargs)


class ResponsesEvidenceBuilderTest(unittest.TestCase):
    WIRE = OpenAIResponsesWire()

    def test_golden_body(self):
        body, headers = build(self.WIRE)
        self.assertEqual(headers, {})
        self.assertEqual(body, {
            "model": "gpt-6.1-sol", "instructions": BASE_INSTRUCTIONS,
            "input": [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": DEVELOPER}]},
                      {"type": "message", "role": "user", "content": [
                          {"type": "input_text", "text": "evidence text"},
                          {"type": "input_image", "image_url": DATA_URL}]}],
            "tools": [],
            "text": {"format": {"type": "json_schema", "name": "evidence_answer", "schema": SCHEMA, "strict": True}},
            "reasoning": {"effort": "high"}, "store": False, "stream": True})
        self.assertEqual(OpenAIResponsesWire(client="opencode").build_evidence_request(
            model="gpt-6.1-sol", base_instructions=BASE_INSTRUCTIONS, developer_instructions=DEVELOPER,
            input_items=ITEMS, output_schema=SCHEMA, effort="high"), (body, headers))

    def test_mirrors_the_pinned_codex_output_schema_request(self):
        """What the pinned Codex app-server (openai-codex 0.160.0) sent for one CodexEvidenceAgent turn with
        ``outputSchema`` against FakeResponsesServer (captured by fixtures/evidence/capture_codex_output_schema.py)."""
        captured = json.loads(CAPTURE.read_text())
        self.assertEqual(captured["codex_cli_version"], "0.160.0")
        body, _ = build(self.WIRE, output_schema=captured["output_schema"])
        codex_format, relay_format = captured["text"]["format"], body["text"]["format"]
        # The structured-output request the provider sees: same type, same strictness, same schema bytes.
        self.assertEqual(set(codex_format), set(relay_format))
        for key in ("type", "strict", "schema"):
            self.assertEqual(relay_format[key], codex_format[key], key)
        self.assertEqual(body["reasoning"]["effort"], captured["reasoning"]["effort"])
        self.assertEqual((body["store"], body["stream"]), (captured["store"], captured["stream"]))
        evidence_message = captured["input_items"][-1]
        self.assertEqual(evidence_message["role"], "user")
        self.assertEqual([part["type"] for part in body["input"][-1]["content"]],
                         [part["type"] for part in evidence_message["content"]])
        self.assertEqual(evidence_message["content"][1], {"type": "input_image", "image_url": "<data-url>"})
        # Recorded differences (Codex harness context and Codex-specific request fields, design §9.3/§9.6): the
        # relay request carries none of them.
        self.assertEqual(codex_format["name"], "codex_output_schema")
        self.assertEqual(relay_format["name"], "evidence_answer")
        self.assertEqual(captured["text"].get("verbosity"), "high")
        self.assertNotIn("verbosity", body["text"])
        self.assertEqual(captured["reasoning"].get("context"), "all_turns")
        self.assertEqual(set(captured["top_level_keys"]) - set(body),
                         {"client_metadata", "include", "parallel_tool_calls", "prompt_cache_key", "tool_choice"})
        self.assertEqual(set(body) - set(captured["top_level_keys"]), {"instructions", "tools"})
        self.assertFalse(captured["instructions_is_base_instructions"])  # Codex puts its own prompt there
        self.assertEqual(sum(item.get("role") == "developer" for item in captured["input_items"]), 5)

    def test_rejected_items(self):
        for items, error in (([], "evidence_input_item_unsupported"),
                             ([dict(type="audio", data="x")], "evidence_input_item_unsupported"),
                             ([dict(type="image", url="https://example.invalid/a.png")],
                              "evidence_image_requires_inline_data_url"),
                             ([dict(type="image", url="data:text/plain;base64,AAAA")],
                              "evidence_image_requires_inline_data_url")):
            with self.subTest(items=items), self.assertRaisesRegex(ValueError, error):
                build(self.WIRE, input_items=items)

    def test_answer_extraction(self):
        message = dict(type="message", role="assistant", content=[dict(type="output_text", text='{"assessment":'),
                                                                   dict(type="output_text", text='"uncertain"}')])
        reasoning = dict(type="reasoning", summary=[], encrypted_content="x")
        self.assertEqual(self.WIRE.evidence_answer(dict(status="completed", output=[reasoning, message])),
                         '{"assessment":"uncertain"}')
        cases = [(dict(status="incomplete", output=[message]), wire_base.EVIDENCE_UNVERIFIED),
                 (dict(status="failed", output=[]), wire_base.EVIDENCE_UNVERIFIED),
                 (dict(status="completed", output=[message, dict(type="function_call", name="x", arguments="{}")]),
                  wire_base.EVIDENCE_TOOL_ACTIVITY),
                 (dict(status="completed", output=[dict(type="web_search_call")]), wire_base.EVIDENCE_TOOL_ACTIVITY),
                 (dict(status="completed", output=[dict(message, content=[dict(type="refusal", refusal="no")])]),
                  wire_base.EVIDENCE_TOOL_ACTIVITY),
                 (dict(status="completed", output=[reasoning]), wire_base.EVIDENCE_ONE_ANSWER),
                 (dict(status="completed", output=[message, message]), wire_base.EVIDENCE_ONE_ANSWER)]
        for terminal, error in cases:
            with self.subTest(terminal=terminal), self.assertRaisesRegex(ValueError, error):
                self.WIRE.evidence_answer(terminal)


class AnthropicEvidenceBuilderTest(unittest.TestCase):
    WIRE = AnthropicMessagesWire()

    def test_golden_body(self):
        body, headers = build(self.WIRE, model="claude-opus-5-5")
        self.assertEqual(headers, {"anthropic-version": "2023-06-01"})
        self.assertEqual(body, {
            "model": "claude-opus-5-5", "max_tokens": 16000,
            "system": [{"type": "text", "text": BASE_INSTRUCTIONS + "\n\n" + DEVELOPER}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "evidence text"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}]}],
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "high"},
            "stream": True})
        for absent in ("tools", "tool_choice", "thinking", "temperature", "top_p", "top_k", "output_format"):
            self.assertNotIn(absent, body)
        self.assertEqual(build(AnthropicMessagesWire(client="opencode"), model="claude-opus-5-5"), (body, headers))

    def stream(self, blocks, stop_reason):
        frames = [am.KEEPALIVE_FRAME, f"event: message_start\ndata: {json.dumps(dict(type='message_start', message=dict(id='msg_e', model='claude-served', content=[], stop_reason=None, usage=dict(input_tokens=9, output_tokens=1))))}\n\n".encode()]
        for index, (start, deltas) in enumerate(blocks):
            frames.append(f"event: content_block_start\ndata: {json.dumps(dict(type='content_block_start', index=index, content_block=start))}\n\n".encode())
            for delta in deltas:
                frames.append(f"event: content_block_delta\ndata: {json.dumps(dict(type='content_block_delta', index=index, delta=delta))}\n\n".encode())
            frames.append(f"event: content_block_stop\ndata: {json.dumps(dict(type='content_block_stop', index=index))}\n\n".encode())
        frames.append(f"event: message_delta\ndata: {json.dumps(dict(type='message_delta', delta=dict(stop_reason=stop_reason), usage=dict(output_tokens=4)))}\n\n".encode())
        frames.append(b'event: message_stop\ndata: {"type": "message_stop"}\n\n')
        return b"".join(frames)

    def answer(self, raw):
        _, _, terminal = self.WIRE.response_evidence(raw, self.WIRE.route("POST", "/v1/messages"))
        return self.WIRE.evidence_answer(terminal)

    def test_answer_extraction_from_stream_and_json(self):
        thinking = ({"type": "thinking", "thinking": ""}, [{"type": "signature_delta", "signature": "s"}])
        text = ({"type": "text", "text": ""}, [{"type": "text_delta", "text": '{"assessment":'},
                                               {"type": "text_delta", "text": '"supported"}'}])
        self.assertEqual(self.answer(self.stream([thinking, text], "end_turn")), '{"assessment":"supported"}')
        body = dict(type="message", id="msg_j", model="m", role="assistant", stop_reason="end_turn",
                    content=[dict(type="thinking", thinking="t", signature="s"), dict(type="text", text='{"a":1}')],
                    usage=dict(input_tokens=1, output_tokens=1))
        self.assertEqual(self.answer(json.dumps(body).encode()), '{"a":1}')
        tool = ({"type": "tool_use", "id": "toolu_1", "name": "submit_answer", "input": {}},
                [{"type": "input_json_delta", "partial_json": "{}"}])
        cases = [(self.stream([text], "refusal"), wire_base.EVIDENCE_UNVERIFIED),
                 (self.stream([text], "max_tokens"), wire_base.EVIDENCE_UNVERIFIED),
                 (self.stream([text], "pause_turn"), wire_base.EVIDENCE_UNVERIFIED),
                 (self.stream([text, tool], "end_turn"), wire_base.EVIDENCE_TOOL_ACTIVITY),
                 (self.stream([({"type": "server_tool_use", "id": "s", "name": "web_search", "input": {}}, [])],
                              "end_turn"), wire_base.EVIDENCE_TOOL_ACTIVITY),
                 (self.stream([thinking], "end_turn"), wire_base.EVIDENCE_ONE_ANSWER)]
        for raw, error in cases:
            with self.subTest(error=error, raw=raw[-120:]), self.assertRaisesRegex(ValueError, error):
                self.answer(raw)
        with self.assertRaisesRegex(ValueError, wire_base.EVIDENCE_UNVERIFIED):
            self.WIRE.evidence_answer(dict(status="failed", error={}))


# ------------------------------------------------------------------------------------------------- bindings
class BindingsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "run", dict(run_id="relay", episode_id="episode",
                                                              env_instance_id="environment", env_generation=1))
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)
        self.actor_server = FakeAnthropicServer().start()
        self.evidence_server = FakeResponsesServer(default=text_step('{"assessment":"uncertain"}')).start()
        self.addCleanup(self.actor_server.close)
        self.addCleanup(self.evidence_server.close)

    def proxy(self, **kwargs):
        actor_wire = AnthropicMessagesWire()
        kwargs.setdefault("max_retries", 0)
        kwargs.setdefault("request_projection", lambda request: actor_wire.project_request(
            request, self.presentation, registered_tools=frozenset(), tool_name_prefix="mcp__auriga__"))
        proxy = ModelRequestProxy(self.actor_server.base_url, wire=actor_wire,
                                  upstream_auth_headers={"x-api-key": "actor-secret"},
                                  default_thread_id=SESSION, **kwargs).start()
        self.addCleanup(proxy.close)
        patch = mock.patch.object(proxy.policy, "_retry_delay_seconds", return_value=0)
        patch.start()
        self.addCleanup(patch.stop)
        proxy.register_thread(SESSION, "actor", SESSION)
        url = proxy.add_upstream("evidence", wire=OpenAIResponsesWire(), upstream_url=self.evidence_server.base_url,
                                 upstream_auth_headers={"Authorization": "Bearer evidence-secret"},
                                 presentation=self.presentation)
        self.assertEqual(url, proxy.base_url + "/_auriga/u/evidence")
        return proxy

    @staticmethod
    def post(base_url, path, body, headers):
        host, port = base_url.split("//", 1)[1].split("/", 1)[0].split(":")
        prefix = "/" + base_url.split("//", 1)[1].split("/", 1)[1] if base_url.count("/") > 2 else ""
        data = json.dumps(body).encode()
        connection = http.client.HTTPConnection(host, int(port), timeout=30)
        try:
            connection.putrequest("POST", prefix + path, skip_accept_encoding=True)
            for key, value in headers:
                connection.putheader(key, value)
            connection.putheader("Content-Length", str(len(data)))
            connection.endheaders(data)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def evidence_call(self, proxy, request_id="r1", items=ITEMS, headers=()):
        thread = "evidence-" + request_id
        proxy.register_thread(thread, "sol_visual_review", request_id)
        wire = proxy.upstream_binding("evidence").wire
        body, extra = build(wire, input_items=items)
        all_headers = (("Content-Type", "application/json"), ("X-Auriga-Thread", thread),
                       ("X-Auriga-Source", "sol_visual_review"), ("X-Auriga-Request-Id", request_id),
                       *extra.items(), *headers)
        status, raw = self.post(proxy.upstream_base_url("evidence"), "/responses", body, all_headers)
        return status, raw, thread

    def events(self, proxy, kind):
        with proxy.lock:
            return [dict(event) for event in proxy.events if event["kind"] == kind]

    def test_evidence_request_through_a_binding(self):
        proxy = self.proxy()
        planted = [dict(type="text", text="fingerprint sha256=" + DIGEST), ITEMS[1]]
        status, raw, thread = self.evidence_call(proxy, items=planted,
                                                 headers=(("Authorization", "Bearer client-copy"),))
        self.assertEqual(status, 200)
        wire = proxy.upstream_binding("evidence").wire
        complete, evidence, terminal = wire.response_evidence(raw.replace(orw.KEEPALIVE_FRAME, b""),
                                                              wire.route("POST", "/responses"))
        self.assertEqual(wire.completion_outcome(complete, evidence), "usable")
        self.assertEqual(json.loads(wire.evidence_answer(terminal)), {"assessment": "uncertain"})
        forwarded = self.evidence_server.generation_requests()
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(self.actor_server.requests, [])
        headers = {k.lower(): v for k, v in forwarded[0].headers.items()}
        self.assertEqual(headers["authorization"], "Bearer evidence-secret")  # client copy replaced
        for private in ("x-api-key", "x-auriga-thread", "x-auriga-source", "x-auriga-request-id"):
            self.assertNotIn(private, headers)
        self.assertNotIn(DIGEST, forwarded[0].body.decode())  # binding egress projection with its presentation
        self.assertEqual(forwarded[0].json["text"]["format"]["schema"], SCHEMA)
        self.assertEqual(forwarded[0].json["input"][1]["content"][1], {"type": "input_image", "image_url": DATA_URL})
        completed = self.events(proxy, "model_call_completed")[0]
        self.assertEqual((completed["upstream"], completed["native_thread_id"], completed["source"],
                          completed["caller_request_id"], completed["endpoint"]),
                         ("evidence", thread, "sol_visual_review", "r1", "/responses"))
        self.assertNotIn("call_purpose", completed)  # no continuity tracking on helper bindings
        self.assertEqual(proxy.request_count, 1)
        self.assertEqual(self.events(proxy, "model_upstream_bound")[0]["credentials"], ["Authorization"])
        self.assertNotIn("evidence-secret", json.dumps(proxy.events))

    def test_shared_budget_and_actor_isolation(self):
        proxy = self.proxy(max_requests=2)
        actor_body = dict(model="claude-opus-5-5", max_tokens=64, stream=True,
                          messages=[dict(role="user", content="hi")], tools=[dict(name="t", input_schema={})])
        status, _ = self.post(proxy.base_url, "/v1/messages", actor_body,
                              CLAUDE_HEADERS + (("Authorization", "Bearer evidence-secret"),))
        self.assertEqual(status, 200)
        actor_headers = {k.lower(): v for k, v in self.actor_server.generation_requests()[0].headers.items()}
        self.assertEqual(actor_headers["x-api-key"], "actor-secret")
        self.assertNotIn("authorization", actor_headers)  # never another binding's credential
        self.assertEqual(self.evidence_call(proxy)[0], 200)
        self.assertEqual(proxy.request_count, 2)
        status, raw, _ = self.evidence_call(proxy, request_id="r2")
        self.assertEqual(status, 429)  # the evidence binding's own (Codex) budget denial
        self.assertEqual(self.post(proxy.base_url, "/v1/messages", actor_body, CLAUDE_HEADERS)[0], 403)
        self.assertEqual(len(self.evidence_server.generation_requests()), 1)
        actor_events = [e for e in self.events(proxy, "model_call_completed") if "upstream" not in e]
        self.assertEqual([e["call_purpose"] for e in actor_events], ["main"])

    def test_blocked_transport_is_shared(self):
        self.evidence_server.default = Step(failure=Failure("http", 400, "provider-private rejection",
                                                            error_type="invalid_request_error"))
        proxy = self.proxy()
        status, raw, _ = self.evidence_call(proxy)
        self.assertEqual(status, 200)  # Codex profile: early stream, the failure is one error event
        self.assertIn(b"event: error", raw)
        self.assertTrue(proxy.transport_blocked)
        self.assertEqual(proxy.blocked_error["evidence"]["http_status"], 400)
        actor_body = dict(model="m", max_tokens=8, stream=True, messages=[dict(role="user", content="x")])
        self.assertEqual(self.post(proxy.base_url, "/v1/messages", actor_body, CLAUDE_HEADERS)[0], 403)
        self.assertEqual(self.actor_server.requests, [])

    def test_cancelled_evidence_thread_is_denied(self):
        proxy = self.proxy()
        proxy.register_thread("evidence-r9", "sol_visual_review", "r9")
        proxy.cancel_thread("evidence-r9")
        body, _ = build(OpenAIResponsesWire())
        status, _ = self.post(proxy.upstream_base_url("evidence"), "/responses", body,
                              (("Content-Type", "application/json"), ("X-Auriga-Thread", "evidence-r9")))
        self.assertEqual(status, 409)
        self.assertEqual(self.evidence_server.requests, [])

    def test_self_briefing_on_the_actor_binding_is_a_helper(self):
        """Self-briefing uses the actor binding (§9.4): no tools, so the tracker attributes it as a helper."""
        proxy = self.proxy()
        proxy.register_thread("evidence-brief", "self_briefing", "brief")
        body, extra = build(AnthropicMessagesWire(), model="claude-opus-5-5")
        self.actor_server.default = text_step('{"assessment":"supported"}')
        status, raw = self.post(proxy.base_url, "/v1/messages", body,
                                (("Content-Type", "application/json"), ("X-Auriga-Thread", "evidence-brief"),
                                 *extra.items()))
        self.assertEqual(status, 200)
        completed = self.events(proxy, "model_call_completed")[0]
        self.assertEqual((completed["native_thread_id"], completed["source"], completed["call_purpose"]),
                         ("evidence-brief", "self_briefing", "helper"))
        self.assertNotIn("upstream", completed)
        forwarded = self.actor_server.generation_requests()[0]
        self.assertNotIn("x-auriga-thread", {k.lower() for k in forwarded.headers.keys()})
        self.assertEqual(forwarded.json["output_config"]["format"]["type"], "json_schema")
        wire = AnthropicMessagesWire()
        _, _, terminal = wire.response_evidence(raw.replace(am.KEEPALIVE_FRAME, b""), wire.route("POST", "/v1/messages"))
        self.assertEqual(json.loads(wire.evidence_answer(terminal)), {"assessment": "supported"})

    def test_routing_validation_and_worker_subprocess(self):
        proxy = self.proxy()
        status, _ = self.post(proxy.base_url, "/_auriga/u/missing/responses", dict(model="m"), ())
        self.assertEqual(status, 404)
        status, _ = self.post(proxy.base_url, "/_auriga/u/evidence/v1/messages", dict(model="m"), ())
        self.assertEqual(status, 404)  # the binding's own routes only (Codex Responses: exact paths)
        for name in ("actor", "Evidence", "", "a" * 40, "with/slash"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "model_proxy_invalid_upstream_name"):
                proxy.add_upstream(name, wire=OpenAIResponsesWire(), upstream_url="http://127.0.0.1:1",
                                   upstream_auth_headers=None, presentation=self.presentation)
        with self.assertRaisesRegex(ValueError, "model_proxy_upstream_already_bound"):
            proxy.add_upstream("evidence", wire=OpenAIResponsesWire(), upstream_url="http://127.0.0.1:1",
                               upstream_auth_headers=None, presentation=self.presentation)
        with self.assertRaisesRegex(ValueError, "invalid_upstream_auth_headers"):
            proxy.add_upstream("other", wire=OpenAIResponsesWire(), upstream_url="http://127.0.0.1:1",
                               upstream_auth_headers={"x-api-key": "k"}, presentation=self.presentation)
        with self.assertRaisesRegex(ValueError, "model_proxy_requires_http_base_url"):
            proxy.add_upstream("other", wire=OpenAIResponsesWire(), upstream_url="http://u:p@127.0.0.1:1",
                               upstream_auth_headers=None, presentation=self.presentation)
        with self.assertRaisesRegex(ValueError, "model_proxy_upstream_requires_presentation"):
            proxy.add_upstream("other", wire=OpenAIResponsesWire(), upstream_url="http://127.0.0.1:1",
                               upstream_auth_headers=None, presentation=None)
        with self.assertRaises(KeyError):
            proxy.upstream_base_url("missing")
        self.assertEqual(proxy.upstream_base_url("actor"), proxy.base_url)
        self.assertIsInstance(proxy.upstream_binding("actor"), UpstreamBinding)
        # The killable --http-worker subprocess path rebuilds the binding's wire from its payload.
        with mock.patch.object(model_proxy.subprocess, "Popen", side_effect=subprocess.Popen) as spawned:
            status, raw, _ = self.evidence_call(proxy, request_id="worker")
        self.assertEqual(status, 200)
        self.assertTrue(spawned.called)
        self.assertIn(b"response.completed", raw)
        self.assertEqual(self.evidence_server.generation_requests()[-1].headers["Authorization"],
                         "Bearer evidence-secret")

    def test_helper_binding_skips_image_trim(self):
        proxy = self.proxy(max_inline_images=24)
        items = [dict(type="text", text="frames")] + [dict(type="image", url=DATA_URL)] * 40
        status, _, _ = self.evidence_call(proxy, items=items)
        self.assertEqual(status, 200)
        self.assertEqual(self.events(proxy, "model_images_trimmed"), [])
        forwarded = self.evidence_server.generation_requests()[0].json
        self.assertEqual(sum(part["type"] == "input_image" for part in forwarded["input"][1]["content"]), 40)


if __name__ == "__main__":
    unittest.main()
