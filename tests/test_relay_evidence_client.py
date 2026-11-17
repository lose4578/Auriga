"""RelayEvidenceClient (design 04 §9.2-§9.6, P1-8): one fresh, streamed, tool-free evidence request through the run's
relay, on every wire and client profile, against loopback fake model servers (no model, no network).

Covered: request bodies and headers per wire as the upstream receives them, attribution (``X-Auriga-Thread``,
source, caller request id), result shape and usage, the named ``"evidence"`` binding for visual review, the shared
budget across bindings, cancellation, the recovery gate, the egress scrub and alias restore, failure strings identical
to ``CodexEvidenceAgent``, the Codex evidence-client switch, Workbench wiring and manifest ``evidence_client``, a
non-Codex ``FakeAgentSession`` benchmark start with self-briefing through the relay, ``VisualReviewer`` through the
relay client, config/gating, and the queue's evidence-path warning.
"""
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

from auriga.apps.evaluation import queue
from auriga.apps.gateway.workbench import Workbench
from auriga.core.config import AgentConfig, Config, ConfigError, EvidenceConfig
from auriga.core.presentation.actor_presentation import ActorPresentation
from auriga.core.storage.store import Store
from auriga.cortex.agents import evidence as evidence_module
from auriga.cortex.agents import factory
from auriga.cortex.agents.base import AgentCapabilities
from auriga.cortex.agents.evidence import (BACKEND_LABEL, CONTEXT_POLICY, RelayEvidenceClient, bind_evidence_upstream,
                                           resolve_upstream)
from auriga.cortex.context import self_briefing as briefing
from auriga.cortex.session import visual_review
from auriga.cortex.session.codex import CodexSession
from auriga.cortex.session.codex_evidence_agent import BASE_INSTRUCTIONS, CodexEvidenceAgent
from auriga.cortex.session.model_proxy import ModelRequestProxy
from auriga.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire
from auriga.cortex.session.wire import base as wire_base

from support.fake_agent import FakeAgentSession
from support.fake_model_servers import (Failure, FakeAnthropicServer, FakeResponsesServer, Step, Usage, text_step,
                                        tool_step)

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="
DATA_URL = "data:image/png;base64," + PNG
DIGEST = "0123456789abcdef" * 4
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["assessment"],
          "properties": {"assessment": {"type": "string", "enum": ["supported", "uncertain"]}}}
ITEMS = [dict(type="text", text="evidence text"), dict(type="image", url=DATA_URL)]
DEVELOPER = "Use only the supplied evidence packet."
CLAUDE_SESSION = "9ff8e771-aa08-4c17-b00d-032e1cb2011f"
ANSWER = '{"assessment":"uncertain"}'
IDENTITY = dict(run_id="relay-evidence", episode_id="episode", env_instance_id="environment", env_generation=1)


def jpeg(color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


class RelayCase(unittest.TestCase):
    """A run Store plus helpers to start a relay in front of a fake server."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = Store(self.root / "run", dict(IDENTITY))
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)

    def server(self, cls, *script, default=None):
        server = cls(list(script), default=default or text_step(ANSWER)).start()
        self.addCleanup(server.close)
        return server

    def relay(self, server, wire, *, auth=None, **kwargs):
        kwargs.setdefault("max_retries", 0)
        kwargs.setdefault("request_projection", lambda request: wire.project_request(
            request, self.presentation, registered_tools=frozenset(), tool_name_prefix=""))
        if isinstance(wire, AnthropicMessagesWire) and wire.client == "claude_code":
            kwargs.setdefault("default_thread_id", CLAUDE_SESSION)
        proxy = ModelRequestProxy(server.base_url, wire=wire, upstream_auth_headers=auth, **kwargs).start()
        self.addCleanup(proxy.close)
        retry = mock.patch.object(proxy.policy, "_retry_delay_seconds", return_value=0)
        retry.start()
        self.addCleanup(retry.stop)
        return proxy

    def session(self, proxy):
        return SimpleNamespace(store=self.store, model_proxy=proxy)

    @staticmethod
    def events(proxy, kind):
        with proxy.lock:
            return [dict(event) for event in proxy.events if event["kind"] == kind]

    def store_events(self, kind):
        return [event["payload"] for event in self.store.events(limit=10000) if event["type"] == kind]


# ------------------------------------------------------------------------------------------- request bodies per wire
class RequestBodyTests(RelayCase):
    def test_responses_actor_binding(self):
        server = self.server(FakeResponsesServer, text_step(ANSWER, usage=Usage(
            input_tokens=30, output_tokens=7, cache_read_input_tokens=4, reasoning_tokens=2)))
        proxy = self.relay(server, OpenAIResponsesWire(), auth={"Authorization": "Bearer host-secret"})
        client = RelayEvidenceClient(self.session(proxy))
        planted = [dict(type="text", text="evidence text fingerprint sha256=" + DIGEST), ITEMS[1]]
        result = client.review_evidence(input_items=planted, output_schema=SCHEMA, request_id="r1")
        forwarded, = server.generation_requests()
        presented = ActorPresentation(self.store)
        expected, _ = OpenAIResponsesWire().build_evidence_request(
            model="gpt-6.1-sol", base_instructions=BASE_INSTRUCTIONS, developer_instructions=DEVELOPER,
            input_items=presented.present(planted), output_schema=presented.present_data(SCHEMA), effort="high")
        self.assertEqual(forwarded.json, expected)
        self.assertEqual(forwarded.route, "/responses")
        self.assertNotIn(DIGEST, forwarded.body.decode())  # presented before the request and re-scrubbed at egress
        self.assertEqual((forwarded.json["tools"], forwarded.json["stream"], forwarded.json["store"]), ([], True, False))
        self.assertEqual(forwarded.json["text"]["format"]["name"], "evidence_answer")
        headers = {key.lower(): value for key, value in forwarded.headers.items()}
        self.assertEqual(headers["authorization"], "Bearer host-secret")
        for private in ("x-auriga-thread", "x-auriga-source", "x-auriga-request-id"):
            self.assertNotIn(private, headers)
        tokens = dict(input_tokens=34, cached_input_tokens=4, cache_write_input_tokens=None, output_tokens=7,
                      reasoning_output_tokens=2, total_tokens=41)
        self.assertEqual(result, dict(
            model="gpt-6.1-sol", status="completed", answer={"assessment": "uncertain"},
            usage=dict(total=tokens, last=tokens, model_context_window=None), thread_id="evidence-r1",
            turn_id="resp_fake_0", served_model="gpt-6.1-sol", context_policy=CONTEXT_POLICY, backend=BACKEND_LABEL,
            wire_api="openai_responses", upstream="actor",
            model_verification="requested_model_recorded_served_model_reported",
            runtime_policy="single_stateless_request_no_session_no_memory_v1",
            tool_policy="no_tools_declared_reject_any_tool_output_v1"))
        completed, = self.events(proxy, "model_call_completed")
        self.assertEqual((completed["native_thread_id"], completed["source"], completed["caller_request_id"],
                          completed["endpoint"]), ("evidence-r1", "sol_visual_review", "r1", "/responses"))
        self.assertNotIn("upstream", completed)
        self.assertEqual(proxy.request_count, 1)
        self.assertEqual([event["native_thread_id"] for event in self.events(proxy, "model_thread_unregistered")],
                         ["evidence-r1"])
        self.assertNotIn("evidence-r1", proxy.thread_owners)

    def test_anthropic_actor_binding_self_briefing(self):
        server = self.server(FakeAnthropicServer, text_step('{"notes":"Grasp the cup first."}', usage=Usage(
            input_tokens=20, output_tokens=6, cache_read_input_tokens=3, cache_creation_input_tokens=2)))
        proxy = self.relay(server, AnthropicMessagesWire(), auth={"x-api-key": "host-key"})
        proxy.register_thread(CLAUDE_SESSION, "actor", CLAUDE_SESSION)
        client = RelayEvidenceClient(self.session(proxy), model="claude-opus-5-5",
                                     base_instructions=briefing.BASE_INSTRUCTIONS,
                                     developer_instructions=briefing.DEVELOPER_INSTRUCTIONS, source=briefing.SOURCE)
        result = client.review_evidence(input_items=ITEMS, output_schema=briefing.ANSWER_SCHEMA, request_id="brief")
        forwarded, = server.generation_requests()
        self.assertEqual(forwarded.route, "/v1/messages")
        self.assertEqual(forwarded.json, {
            "model": "claude-opus-5-5", "max_tokens": 16000,
            "system": [{"type": "text", "text": briefing.BASE_INSTRUCTIONS + "\n\n" + briefing.DEVELOPER_INSTRUCTIONS}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "evidence text"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}]}],
            "output_config": {"format": {"type": "json_schema", "schema": briefing.ANSWER_SCHEMA}, "effort": "high"},
            "stream": True})
        headers = {key.lower(): value for key, value in forwarded.headers.items()}
        self.assertEqual((headers["x-api-key"], headers["anthropic-version"]), ("host-key", "2023-06-01"))
        self.assertNotIn("x-auriga-thread", headers)
        self.assertEqual((result["model"], result["answer"], result["wire_api"], result["thread_id"]),
                         ("claude-opus-5-5", {"notes": "Grasp the cup first."}, "anthropic_messages", "evidence-brief"))
        self.assertEqual(result["usage"]["total"], dict(input_tokens=25, cached_input_tokens=3,
                                                         cache_write_input_tokens=2, output_tokens=6,
                                                         reasoning_output_tokens=None, total_tokens=31))
        completed, = self.events(proxy, "model_call_completed")
        # Not the actor's thread (the default thread key) and attributed as a no-tools helper by the tracker.
        self.assertEqual((completed["native_thread_id"], completed["source"], completed["call_purpose"]),
                         ("evidence-brief", "self_briefing", "helper"))
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])

    def test_opencode_profiles_deferred_delivery(self):
        for server_class, wire, path in ((FakeResponsesServer, OpenAIResponsesWire(client="opencode"), "/responses"),
                                         (FakeAnthropicServer, AnthropicMessagesWire(client="opencode"), "/v1/messages")):
            with self.subTest(wire=wire.name):
                server = self.server(server_class)
                proxy = self.relay(server, wire, max_requests=1)
                client = RelayEvidenceClient(self.session(proxy), model="m")
                result = client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="oc1")
                self.assertEqual(result["answer"], {"assessment": "uncertain"})
                self.assertEqual(server.generation_requests()[0].route, path)
                # Budget spent: the deferred profile's denial (HTTP 400) is an ordinary unverified failure.
                with self.assertRaisesRegex(ValueError, "^sol_agent_unverified_or_incomplete_turn$"):
                    client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="oc2")
                self.assertEqual(len(server.generation_requests()), 1)
                unverified = self.store_events("visual.agent_turn_unverified")[-1]
                self.assertEqual((unverified["request_id"], unverified["http_status"], unverified["wire_api"]),
                                 ("oc2", 400, wire.name))


# ------------------------------------------------------------------------------------------------- bindings and budget
class BindingTests(RelayCase):
    def config(self, server, **upstream):
        return Config.from_dict({"agent": {"backend": "claude_code", "evidence": {"upstream": {
            "wire_api": "openai_responses", "base_url": server.base_url, **upstream}}}})

    def test_visual_review_uses_the_evidence_binding_and_shares_the_budget(self):
        actor_server = self.server(FakeAnthropicServer, default=text_step("actor answer"))
        evidence_server = self.server(FakeResponsesServer)
        proxy = self.relay(actor_server, AnthropicMessagesWire(), auth={"x-api-key": "actor-secret"}, max_requests=2)
        proxy.register_thread(CLAUDE_SESSION, "actor", CLAUDE_SESSION)
        session = self.session(proxy)
        config = self.config(evidence_server, api_key_env="EVIDENCE_KEY")
        name = bind_evidence_upstream(session, config, environ={"EVIDENCE_KEY": "evidence-secret"})
        self.assertEqual(name, "evidence")
        self.assertEqual(bind_evidence_upstream(session, config, environ={}), "evidence")  # idempotent
        self.assertEqual(len(self.events(proxy, "model_upstream_bound")), 1)
        client = RelayEvidenceClient(session, upstream=name)
        result = client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="v1")
        self.assertEqual((result["upstream"], result["wire_api"], result["model"]),
                         ("evidence", "openai_responses", "gpt-6.1-sol"))
        self.assertEqual(actor_server.requests, [])
        forwarded, = evidence_server.generation_requests()
        self.assertEqual(forwarded.headers["Authorization"], "Bearer evidence-secret")
        self.assertNotIn("x-api-key", {key.lower() for key in forwarded.headers.keys()})
        completed, = self.events(proxy, "model_call_completed")
        self.assertEqual((completed["upstream"], completed["source"]), ("evidence", "sol_visual_review"))
        # One actor-binding self-briefing call spends the last unit of the shared 2-call budget.
        briefer = RelayEvidenceClient(session, model="claude-opus-5-5", source="self_briefing")
        actor_server.default = text_step('{"assessment":"supported"}')
        self.assertEqual(briefer.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="b1")["answer"],
                         {"assessment": "supported"})
        self.assertEqual(proxy.request_count, 2)
        with self.assertRaisesRegex(ValueError, "^sol_agent_unverified_or_incomplete_turn$"):
            client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="v2")
        self.assertEqual(len(evidence_server.generation_requests()), 1)
        unverified = self.store_events("visual.agent_turn_unverified")[-1]
        self.assertEqual((unverified["request_id"], unverified["http_status"], unverified["upstream"]),
                         ("v2", 429, "evidence"))
        self.assertTrue(self.events(proxy, "model_budget_exhausted"))
        self.assertNotIn("evidence-secret", json.dumps(proxy.events))

    def test_unbound_upstream_and_missing_relay(self):
        server = self.server(FakeResponsesServer)
        proxy = self.relay(server, OpenAIResponsesWire())
        with self.assertRaisesRegex(ValueError, "^evidence_upstream_not_bound:evidence$"):
            RelayEvidenceClient(self.session(proxy), upstream="evidence").review_evidence(
                input_items=ITEMS, output_schema=SCHEMA, request_id="x")
        with self.assertRaisesRegex(RuntimeError, "evidence_client_requires_started_model_relay"):
            RelayEvidenceClient(SimpleNamespace(store=self.store, model_proxy=None)).review_evidence(
                input_items=ITEMS, output_schema=SCHEMA, request_id="x")
        self.assertEqual(bind_evidence_upstream(self.session(proxy), Config()), "actor")
        self.assertEqual(server.requests, [])

    def test_resolve_upstream(self):
        upstream = Config.from_dict({"agent": {"backend": "opencode", "evidence": {"upstream": {
            "wire_api": "anthropic_messages", "base_url": "https://example.invalid", "api_key_env": "K",
            "auth_scheme": "x-api-key", "provider": "lab"}}}}).agent.evidence.upstream
        wire, url, headers, provider = resolve_upstream(upstream, environ={"K": "secret"})
        self.assertIsInstance(wire, AnthropicMessagesWire)
        self.assertEqual((url, headers, provider), ("https://example.invalid", {"x-api-key": "secret"}, "lab"))
        with self.assertRaisesRegex(ConfigError, "^evidence_upstream_api_key_missing:K"):
            resolve_upstream(upstream, environ={})
        home = self.root / "codex-home"
        home.mkdir()
        (home / "config.toml").write_text('model_provider = "openai"\n')
        (home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "sk-fixture"}))
        codex_home = Config.from_dict({"agent": {"backend": "opencode", "evidence": {"upstream": {
            "source": "codex_home"}}}}).agent.evidence.upstream
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_BASE_URL", None)
            wire, url, headers, provider = resolve_upstream(codex_home, codex_home=str(home))
        self.assertIsInstance(wire, OpenAIResponsesWire)
        self.assertEqual((url, headers, provider), ("https://api.openai.com/v1",
                                                    {"Authorization": "Bearer sk-fixture"}, "openai"))
        with self.assertRaisesRegex(ConfigError, "evidence_upstream_codex_home_unreadable"):
            resolve_upstream(codex_home, codex_home=str(self.root / "missing"))


# ------------------------------------------------------------------------------------------- cancellation and gates
class CancellationTests(RelayCase):
    def test_cancel_while_the_upstream_holds_the_answer(self):
        held = Step(text=ANSWER, gate=True)
        server = self.server(FakeResponsesServer, held, text_step(ANSWER))
        self.addCleanup(held.gate.release)
        proxy = self.relay(server, OpenAIResponsesWire())
        client = RelayEvidenceClient(self.session(proxy))
        cancel, outcome = threading.Event(), {}

        def call():
            try:
                client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="held",
                                       cancel_event=cancel)
            except BaseException as error:  # recorded for the assertion below
                outcome["error"] = error
        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        self.assertTrue(held.gate.wait_entered(10))
        started = time.monotonic()
        cancel.set()
        worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsInstance(outcome.get("error"), TimeoutError)
        self.assertEqual(str(outcome["error"]), "visual_review_program_cancelled")
        self.assertEqual(proxy.cancelled_threads.get("evidence-held"), "helper_turn_interrupted")
        self.assertFalse(proxy.transport_blocked)  # an explicit helper cancellation never blocks the actor
        self.assertEqual(proxy.request_count, 0)
        held.gate.release()
        # The relay keeps serving the next helper call.
        result = client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="after")
        self.assertEqual(result["answer"], {"assessment": "uncertain"})
        self.assertEqual(proxy.request_count, 1)

    def test_already_cancelled_sends_nothing(self):
        server = self.server(FakeResponsesServer)
        proxy = self.relay(server, OpenAIResponsesWire())
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(TimeoutError, "^visual_review_program_cancelled$"):
            RelayEvidenceClient(self.session(proxy)).review_evidence(
                input_items=ITEMS, output_schema=SCHEMA, request_id="early", cancel_event=cancel)
        self.assertEqual((server.requests, self.events(proxy, "model_thread_registered")), ([], []))

    def test_recovery_gate_refuses_the_cooldown_resend(self):
        server = self.server(FakeResponsesServer, Step(text=ANSWER, failure=Failure("truncated")))
        recovery = Config().model.transport_recovery.to_dict()
        proxy = self.relay(server, OpenAIResponsesWire(), transport_recovery=recovery,
                           recovery_gate=lambda: "physical_action_unconfirmed")
        with self.assertRaisesRegex(ValueError, "^sol_agent_unverified_or_incomplete_turn$"):
            RelayEvidenceClient(self.session(proxy)).review_evidence(
                input_items=ITEMS, output_schema=SCHEMA, request_id="gate")
        denied, = self.events(proxy, "model_transport_recovery_denied")
        self.assertEqual((denied["reason"], denied["native_thread_id"]), ("physical_action_unconfirmed", "evidence-gate"))
        self.assertEqual(len(server.generation_requests()), 1)


# ------------------------------------------------------------------------------- failure strings, egress and restore
class FailureStringTests(RelayCase):
    STRINGS = ("sol_agent_invalid_request_id", "visual_review_program_cancelled",
               "sol_agent_unverified_or_incomplete_turn", "sol_agent_unexpected_tool_or_context_item",
               "sol_agent_requires_one_structured_final_answer")

    def test_strings_are_the_codex_evidence_agent_strings(self):
        source = inspect.getsource(CodexEvidenceAgent)
        for text in self.STRINGS:
            self.assertIn('"' + text + '"', source, text)
        self.assertEqual((wire_base.EVIDENCE_UNVERIFIED, wire_base.EVIDENCE_TOOL_ACTIVITY, wire_base.EVIDENCE_ONE_ANSWER),
                         self.STRINGS[2:])
        self.assertEqual((evidence_module.INVALID_REQUEST_ID, evidence_module.CANCELLED), self.STRINGS[:2])

    def test_invalid_request_ids(self):
        client = RelayEvidenceClient(SimpleNamespace(store=self.store, model_proxy=None))
        for request_id in ("", "a/b", "x" * 129, None, 3):
            with self.subTest(request_id=request_id), self.assertRaisesRegex(ValueError, "^sol_agent_invalid_request_id$"):
                client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id=request_id)

    def test_failure_paths_per_wire(self):
        cases = (
            ("tool", lambda: tool_step("robot_move", {"dx": 1}), wire_base.EVIDENCE_TOOL_ACTIVITY),
            ("thinking_only", lambda: Step(thinking="only thoughts"), wire_base.EVIDENCE_ONE_ANSWER),
            ("max_tokens", lambda: Step(text=ANSWER, stop_reason="max_tokens"), wire_base.EVIDENCE_UNVERIFIED),
            ("refusal", lambda: Step(text=ANSWER, stop_reason="refusal"), wire_base.EVIDENCE_UNVERIFIED),
            ("provider_400", lambda: Step(failure=Failure("http", 400, "provider-private text",
                                                          error_type="invalid_request_error")),
             wire_base.EVIDENCE_UNVERIFIED),
            ("truncated", lambda: Step(text=ANSWER, failure=Failure("truncated")), wire_base.EVIDENCE_UNVERIFIED),
            ("not_json", lambda: text_step("not json"), "Expecting value"),
        )
        for server_class, make_wire in ((FakeResponsesServer, OpenAIResponsesWire),
                                        (FakeAnthropicServer, AnthropicMessagesWire)):
            for label, step, error in cases:
                if label == "refusal" and server_class is FakeResponsesServer:
                    continue  # a Responses refusal arrives as a message part (covered by the P1-7 extractor tests)
                with self.subTest(wire=make_wire.name, case=label):
                    server = self.server(server_class, step())
                    proxy = self.relay(server, make_wire())
                    with self.assertRaisesRegex(ValueError, error):
                        RelayEvidenceClient(self.session(proxy), model="m").review_evidence(
                            input_items=ITEMS, output_schema=SCHEMA, request_id="f-" + label)
                    self.assertEqual(proxy.thread_owners.get("f-" + label), None)
                    self.assertNotIn("provider-private text", json.dumps(self.store_events("visual.agent_turn_unverified")))

    def test_aliases_are_presented_and_restored(self):
        artifact = self.store.put_artifact(b"fixture", "rgb", "image/png")
        alias = ActorPresentation(self.store).present_text("see " + artifact).split()[-1]
        self.assertTrue(alias.startswith("artifact:ref_"))
        server = self.server(FakeResponsesServer, text_step(json.dumps({"assessment": "supported", "ref": alias})))
        proxy = self.relay(server, OpenAIResponsesWire())
        schema = {**SCHEMA, "properties": {**SCHEMA["properties"], "ref": {"type": "string"}}}
        result = RelayEvidenceClient(self.session(proxy)).review_evidence(
            input_items=[dict(type="text", text="frame " + artifact)], output_schema=schema, request_id="alias")
        body = server.generation_requests()[0].body.decode()
        self.assertNotIn(artifact, body)
        self.assertIn(alias, body)
        self.assertEqual(result["answer"], {"assessment": "supported", "ref": artifact})


# ------------------------------------------------------------------------------------------- Codex evidence client
class CodexSelectionTests(RelayCase):
    def codex(self, document):
        config = Config.from_dict({**document, "model": {"codex_home": str(self.root / "codex-home")}})
        session = CodexSession(self.root / "run", self.store, lambda params: {}, config=config)
        self.addCleanup(session.close)
        return session

    def test_native_by_default_relay_on_request(self):
        native = self.codex({})
        self.assertIsInstance(native.evidence_agent(), CodexEvidenceAgent)
        self.assertEqual(native.provenance()["evidence_client"], "codex_agent")
        relay = self.codex({"agent": {"evidence": {"client": "relay"}}})
        helper = relay.evidence_agent(model="gpt-6-astra", source="self_briefing")
        self.assertIsInstance(helper, RelayEvidenceClient)
        self.assertEqual((helper.session, helper.model, helper.source, helper.upstream), (relay, "gpt-6-astra",
                                                                                        "self_briefing", "actor"))
        self.assertEqual(relay.provenance()["evidence_client"], "relay")

    def test_codex_relay_through_the_codex_relay_shape(self):
        """Codex with client="relay": the request rides the actor (Responses, codex profile) binding."""
        server = self.server(FakeResponsesServer)
        proxy = self.relay(server, OpenAIResponsesWire())
        session = self.codex({"agent": {"evidence": {"client": "relay"}}})
        session.model_proxy = proxy
        result = session.evidence_agent().review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="c1")
        self.assertEqual((result["backend"], result["upstream"]), (BACKEND_LABEL, "actor"))
        session.model_proxy = None  # owned by the test


# ------------------------------------------------------------------------------------------ VisualReviewer via relay
class VisualReviewerTests(RelayCase):
    def test_review_through_the_relay_client_records_its_backend(self):
        answer = dict(visibility_after="visible", motion="displaced", after_center_px=[10, 10],
                      retained_with_gripper="uncertain", evidence="Cup moved right.")
        server = self.server(FakeResponsesServer, text_step(json.dumps(answer)))
        proxy = self.relay(server, OpenAIResponsesWire())
        reviewer = visual_review.VisualReviewer(RelayEvidenceClient(self.session(proxy)))
        self.assertEqual((reviewer.backend_label, reviewer.context_policy), (BACKEND_LABEL, CONTEXT_POLICY))
        rgb = self.store.put_artifact(jpeg(), "rgb", "image/jpeg")
        sources = [dict(rgb_ref=rgb, width=32, height=24), dict(rgb_ref=rgb, width=32, height=24)]
        with patch.object(visual_review, "comparison_inputs", return_value=(None, sources)), \
                patch.object(visual_review, "saved_observation_state", return_value={"sim_step": 1}):
            result = reviewer.review(self.store, self.store.identity, before_observation_id="b",
                                     after_observation_id="a", camera_id="head", roi=[1, 1, 4, 4], target="cup")
        self.assertEqual(result["answer"], answer)
        reviewed, = self.store_events("visual.model_reviewed")
        self.assertEqual((reviewed["backend"], reviewed["context_policy"], reviewed["agent_thread_id"]),
                         (BACKEND_LABEL, CONTEXT_POLICY, "evidence-" + reviewed["request_id"]))
        _, path = self.store.artifact(reviewed["request_ref"])
        request = json.loads(path.read_text())
        self.assertEqual((request["backend"], request["context_policy"]), (BACKEND_LABEL, CONTEXT_POLICY))
        forwarded, = server.generation_requests()
        self.assertEqual(sum(part["type"] == "input_image" for part in forwarded.json["input"][1]["content"]), 2)
        self.assertEqual(forwarded.json["text"]["format"]["schema"], visual_review.ANSWER_SCHEMA)

    def test_default_labels_for_clients_without_them(self):
        reviewer = visual_review.VisualReviewer(SimpleNamespace(model=visual_review.MODEL))
        self.assertEqual((reviewer.backend_label, reviewer.context_policy),
                         ("official_codex_app_server_sdk", "fresh_ephemeral_thread_selected_evidence_only_v1"))


# ------------------------------------------------------------------------------------------- gateway (Workbench)
class FakeBackendWorkbenchTests(unittest.IsolatedAsyncioTestCase):
    """Non-Codex FakeAgentSession run whose evidence helpers use the relay client (design §9, P2 acceptance item)."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def relay_for(self, server, wire, store):
        presentation = ActorPresentation(store)
        proxy = ModelRequestProxy(server.base_url, wire=wire, max_retries=0, default_thread_id=CLAUDE_SESSION,
                                  request_projection=lambda request: wire.project_request(
                                      request, presentation, registered_tools=frozenset(),
                                      tool_name_prefix="mcp__auriga__")).start()
        self.addCleanup(proxy.close)
        return proxy

    async def start(self, config, server, *, benchmark):
        created = []

        def create(config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
            session = FakeAgentSession(run_dir, store, tool_handler, capabilities=AgentCapabilities(),
                                       model="claude-opus-5-5", provider="anthropic", price_card=price_card)
            session.model_proxy = self.relay_for(server, AnthropicMessagesWire(), store)
            created.append(session)
            return session
        robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
        with patch("auriga.apps.gateway.workbench.create_session", side_effect=create):
            workbench = Workbench(self.root / "run", robot=robot, config=config)
        session, = created
        store = workbench.store
        cameras = [dict(camera_id="head", rgb_ref=store.put_artifact(jpeg("green"), "rgb", "image/jpeg"))]
        state = {"frame_id": "env_origin", "sim_step": 0, "step_domain": "adapter_control_step", "arms": {}}
        initial = {**store.identity, "schema_version": "0.1", "world_kind": "observed", "observation_id": "initial",
                   "sim_step": 0, "step_domain": "adapter_control_step",
                   "robot_state_ref": store.put_artifact(json.dumps(state).encode(), "robot_state", "application/json"),
                   "task_instruction": "Fixture native instruction.", "cameras": cameras,
                   "observation_profile": workbench.harness.observation_profile}
        harness = SimpleNamespace(
            accepting=True, active_action=None, states={}, action_count=0, latest=initial, home_check=None,
            capabilities={}, episode_done=threading.Event(), episode_terminal=None,
            action_budget=lambda: dict(native_steps_used=0, native_steps_limit=None, native_steps_remaining=None),
            native_action_count=0, started_at=time.monotonic(), last_native_progress_at=time.monotonic(),
            scene=SimpleNamespace(capabilities=lambda: {}), start=Mock())
        for field in ("tool_registry", "config", "disabled_tools", "observation_profile", "motion_review",
                      "enable_wait", "vision_compare", "visual_reviewer", "action_policy", "programs"):
            setattr(harness, field, getattr(workbench.harness, field))
        harness.request_stop = lambda: setattr(harness, "accepting", False)
        workbench.harness = harness
        with patch("auriga.apps.gateway.workbench.Terminal") as terminal:
            await workbench.start("Smoke instruction.", benchmark=benchmark)
        terminal.assert_not_called()
        self.addAsyncCleanup(workbench.close)
        return workbench, session

    async def test_self_briefing_through_the_relay(self):
        notes = "Grasp the green block first, lift slowly and confirm placement in the head view."
        server = FakeAnthropicServer([text_step(json.dumps({"notes": notes}))]).start()
        self.addCleanup(server.close)
        config = Config().with_overrides({"run": {"task": "pour_balls_into_vase"},
                                          "features": {"self_briefing": True, "visual_review_model": None}})
        workbench, session = await self.start(config, server, benchmark=True)
        forwarded, = server.generation_requests()  # exactly one extra tool-free call before the first actor turn
        self.assertNotIn("tools", forwarded.json)
        self.assertEqual(forwarded.json["model"], "claude-opus-5-5")
        self.assertTrue(forwarded.json["system"][0]["text"].startswith(briefing.BASE_INSTRUCTIONS))
        self.assertEqual(sum(block["type"] == "image" for block in forwarded.json["messages"][0]["content"]), 1)
        self.assertEqual([call[0] for call in session.calls[:2]], ["start", "run_turn"])
        prompt = session.calls[1][1]
        self.assertIn(briefing.SECTION_HEADING, prompt)
        self.assertIn(notes, prompt)
        manifest = json.loads((workbench.root / "manifest.json").read_text())
        self.assertEqual(manifest["evidence_client"], "relay")
        record = manifest["self_briefing"]
        self.assertEqual((record["status"], record["notes"], record["model"]), ("applied", notes, "claude-opus-5-5"))
        self.assertTrue(record["thread_id"].startswith("evidence-self-briefing"))
        self.assertEqual(record["usage"]["total"]["output_tokens"], 5)
        completed, = [event for event in session.model_proxy.events if event["kind"] == "model_call_completed"]
        self.assertEqual((completed["source"], completed["call_purpose"]), ("self_briefing", "helper"))

    async def test_visual_reviewer_uses_the_evidence_binding(self):
        actor = FakeAnthropicServer().start()
        reviewer_server = FakeResponsesServer([text_step('{"assessment":"supported"}')]).start()
        self.addCleanup(actor.close)
        self.addCleanup(reviewer_server.close)
        config = Config.from_dict({"features": {"self_briefing": False, "visual_review_model": "gpt-6.1-sol"},
                                   "agent": {"evidence": {"client": "relay", "upstream": {
                                       "wire_api": "openai_responses", "base_url": reviewer_server.base_url}}}})
        workbench, session = await self.start(config, actor, benchmark=False)
        client = workbench.harness.visual_reviewer.client
        self.assertIsInstance(client, RelayEvidenceClient)
        self.assertEqual((client.upstream, client.model, client.source), ("evidence", "gpt-6.1-sol", "sol_visual_review"))
        manifest = json.loads((workbench.root / "manifest.json").read_text())
        self.assertEqual(manifest["evidence_client"], "relay")
        result = client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="wb")
        self.assertEqual((result["answer"], result["upstream"]), ({"assessment": "supported"}, "evidence"))
        self.assertEqual((len(actor.generation_requests()), len(reviewer_server.generation_requests())), (0, 1))


class CodexWorkbenchManifestTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_visual_review_records_the_native_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
            workbench = Workbench(Path(temporary) / "run", robot=robot, visual_review_model="gpt-6.1-sol",
                                  config=Config.from_dict({"features": {"self_briefing": False}}))
            from test_agent_workbench import CODEX_MANIFEST_KEYS_BEFORE, fixture_harness, terminal_double
            workbench.harness = fixture_harness(workbench)
            workbench.session = SimpleNamespace(  # duck-typed Codex session without protocol members
                thread_id="fixture-thread", turn_id=None, state="completed", turn_done=threading.Event(),
                model="fixture", provider="fixture", effort="high", price_card=None, cli_executable=Path("/fixture/codex"),
                command_prefix=[], actor_dir=workbench.root, socket_path=workbench.root / "fixture.sock",
                env={}, process=SimpleNamespace(pid=1), start=Mock(return_value="fixture-thread"),
                run_turn=Mock(), close=Mock(), model_proxy=None, store=workbench.store)
            with patch("auriga.apps.gateway.workbench.Terminal", return_value=terminal_double()), \
                    patch("auriga.cortex.session.codex_evidence_agent.CodexEvidenceAgent") as native:
                native.return_value.model = "gpt-6.1-sol"
                await workbench.start("Smoke instruction.")
            try:
                native.assert_called_once_with(workbench.session)
                manifest = json.loads((workbench.root / "manifest.json").read_text())
                self.assertEqual(set(manifest) - CODEX_MANIFEST_KEYS_BEFORE, {"agent_backend", "evidence_client"})
                self.assertEqual((manifest["evidence_client"], manifest["agent_backend"]["evidence_client"]),
                                 ("codex_agent", "codex_agent"))
            finally:
                await workbench.close()


# --------------------------------------------------------------------------------------- config, gating, queue
class ConfigAndGatingTests(unittest.TestCase):
    def test_evidence_config_defaults_and_omission(self):
        self.assertEqual(AgentConfig().evidence, EvidenceConfig())
        self.assertEqual((EvidenceConfig().client, EvidenceConfig().upstream), ("native", None))
        self.assertNotIn("agent", Config().to_dict())
        exported = Config.from_dict({"agent": {"backend": "claude_code"}}).to_dict()
        self.assertNotIn("evidence", exported["agent"])  # agent snapshots of earlier checkouts stay identical
        relay = Config.from_dict({"agent": {"evidence": {"client": "relay"}}})
        self.assertEqual(relay.to_dict()["agent"]["evidence"], {"client": "relay", "upstream": None})
        self.assertEqual(Config.from_dict(relay.to_dict()), relay)
        bound = Config.from_dict({"agent": {"backend": "claude_code",
                                            "evidence": {"upstream": {"source": "codex_home"}}}})
        self.assertEqual(Config.from_dict(json.loads(json.dumps(bound.to_dict()))), bound)
        self.assertEqual(bound.agent.evidence.upstream.source, "codex_home")

    def test_evidence_config_validation(self):
        for document, message in (
                ({"evidence": {"client": "auto"}}, "invalid_config_choice:EvidenceConfig.client"),
                ({"evidence": {"upstream": {"source": "codex_home"}}}, "evidence_upstream_requires_relay_client"),
                ({"backend": "opencode", "evidence": {"upstream": {"wire_api": "openai_chat",
                                                                   "base_url": "http://127.0.0.1:1"}}},
                 "evidence_upstream_wire_not_implemented:openai_chat"),
                ({"backend": "opencode", "evidence": {"upstream": {"wire_api": "openai_responses"}}},
                 "evidence_upstream_requires_wire_api_and_base_url"),
                ({"backend": "opencode", "evidence": {"upstream": {"source": "codex_home",
                                                                   "wire_api": "anthropic_messages"}}},
                 "evidence_upstream_codex_home_requires_openai_responses"),
                ({"evidence": {"upstream": 3}}, "config_object_required:config.agent.evidence.upstream"),
                ({"evidence": {"upstream": {"api_key": "secret"}}}, "unknown_config_keys")):
            with self.subTest(document=document), self.assertRaisesRegex(ConfigError, message):
                Config.from_dict({"agent": document})

    def test_effective_client(self):
        self.assertEqual(factory.evidence_client(Config()), "codex_agent")
        self.assertEqual(factory.evidence_client(SimpleNamespace()), "codex_agent")  # no agent section: Codex native
        self.assertEqual(factory.evidence_client(Config.from_dict({"agent": {"evidence": {"client": "relay"}}})), "relay")
        for kind in ("claude_code", "opencode", "fake_agent"):
            self.assertEqual(factory.evidence_client(Config(), kind), "relay")
        for kind in ("claude_code", "opencode"):
            self.assertEqual(factory.evidence_client(Config.from_dict({"agent": {"backend": kind}})), "relay")

    def test_backend_requirements_evidence_rules(self):
        def code(document):
            with self.assertRaises(ConfigError) as raised:
                factory.backend_requirements(Config.from_dict(document))
            return str(raised.exception).split(":")[0]
        review = {"self_briefing": True, "visual_review_model": "gpt-6.1-sol"}
        # The interim refusal is gone: evidence features pass on to the backend's own checks.
        self.assertEqual(code({"agent": {"backend": "opencode"}, "features": review}), "opencode_upstream_incomplete")
        with mock.patch.object(factory, "claude_code_runtime_problem", return_value=None):
            self.assertIsNone(factory.backend_requirements(Config.from_dict(
                {"agent": {"backend": "claude_code"}, "features": {"self_briefing": True, "visual_review_model": None}})))
            self.assertEqual(code({"agent": {"backend": "claude_code"}, "features": review}),
                             "evidence_upstream_required_for_visual_review_model")
            self.assertIsNone(factory.backend_requirements(Config.from_dict(
                {"agent": {"backend": "claude_code", "evidence": {"upstream": {"source": "codex_home"}}},
                 "features": review})))
        self.assertEqual(code({"agent": {"backend": "opencode", "opencode": {"upstream": {
            "wire_api": "openai_responses", "base_url": "http://127.0.0.1:1", "api_key_env": "K"}}}, "features": review}),
            "agent_backend_unavailable")
        self.assertIsNone(factory.backend_requirements(Config.from_dict({"features": review})))

    def test_queue_projection_fails_closed_for_old_checkouts(self):
        source = (Path(__file__).parents[1] / "auriga/core/config.py").read_text()
        old = source.replace("    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)\n", "")
        self.assertNotEqual(old, source)
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            (checkout / "auriga/apps/gateway").mkdir(parents=True)
            (checkout / "auriga/core").mkdir(parents=True)
            (checkout / "auriga/apps/gateway/cli.py").write_text("parser.add_argument('--config')\nConfig.load()\n")
            (checkout / "auriga/core/config.py").write_text(old)
            schema = queue.runtime_config_schema(checkout)
        default_evidence = Config.from_dict({"agent": {"backend": "opencode"}}).to_dict()
        self.assertEqual(queue.project_config_for_runtimes(default_evidence, [schema]), default_evidence)
        relay = Config.from_dict({"agent": {"evidence": {"client": "relay"}}}).to_dict()
        with self.assertRaisesRegex(ValueError, "unified_config_incompatible:agent.evidence"):
            queue.project_config_for_runtimes(relay, [schema])
        current = queue.runtime_config_schema(Path(__file__).parents[1])
        self.assertEqual(queue.project_config_for_runtimes(relay, [current]), relay)

    def test_queue_warns_when_cases_use_different_evidence_clients(self):
        codex = Config.from_dict({"features": {"self_briefing": True}})
        claude = codex.with_overrides({"agent": {"backend": "claude_code"}})
        quiet_claude = claude.with_overrides({"features": {"self_briefing": False}})
        relay_codex = codex.with_overrides({"agent": {"evidence": {"client": "relay"}}})
        self.assertEqual(queue.evidence_path_warnings([codex, codex]), [])
        self.assertEqual(queue.evidence_path_warnings([codex, quiet_claude]), [])  # no evidence feature on one arm
        self.assertEqual(queue.evidence_path_warnings([relay_codex, claude]), [])  # the fairness rule followed
        warning, = queue.evidence_path_warnings([codex, claude, relay_codex])
        self.assertEqual((warning["kind"], warning["evidence_clients"], warning["agent_backends"]),
                         ("comparison_evidence_path_mismatch", {"codex_agent": [0], "relay": [1, 2]},
                          ["claude_code", "codex"]))


if __name__ == "__main__":
    unittest.main()
