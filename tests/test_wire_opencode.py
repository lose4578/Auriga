"""OpenCode client profiles of the Anthropic Messages and OpenAI Responses wires (design §4.2, plan T9 / P3-4).

Inputs are real opencode 1.19.5 requests captured by the S6 spike (``tests/fixtures/opencode/opencode_wire_bodies.json``)
and the S9 compaction sequences (``tests/fixtures/continuity/continuity_fixture.json``). The relay is exercised for
real against the fake model servers; OpenCode's client-side retry rule is simulated from the spike's observations
(report 06 S6.9) and opencode's ``session/retry.ts`` pattern list, and the simulator is first checked against the
spike's measured attempt counts. Then every relay denial and every failed delivery must cost exactly one client attempt.
"""
import base64
from email.message import Message
import json
from pathlib import Path
import re
import threading
import time
import unittest

from auriga.cortex.agents import context_signal as cs
from auriga.cortex.session.model_proxy import ModelRequestProxy
from auriga.cortex.session.wire import (AnthropicMessagesWire, OPENCODE_PROFILE, OpenAIResponsesWire, Route,
                                        present_messages_request, wire_from_name)
from auriga.cortex.session.wire import base as wire_base

from support.fake_model_servers import Failure, FakeAnthropicServer, FakeResponsesServer, Gate, Step, ToolCall
from test_model_proxy_continuity import RelayContinuityCase, fixture_body, load_fixture, post

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "opencode" / "opencode_wire_bodies.json"
DIGEST = "0123456789abcdef" * 4
ARTIFACT = "artifact:" + "a" * 64 + "-" + "b" * 16
LONG_OBSERVATION = "observation-" + "c" * 32
PLANTED = f"sha256={DIGEST} see {ARTIFACT} and {LONG_OBSERVATION}"
INSTRUCTIONS_FILE = "/Users/wangchengcheng/runs/run-7/private/opencode/instructions/AURIGA.md"
# opencode 1.19.5 packages/opencode/src/session/retry.ts RETRYABLE_MESSAGE_PATTERNS (checked on the error message and
# on the response body of an APIError; the generic branch for other errors also retries "exhausted"/"unavailable").
RETRYABLE_MESSAGE_PATTERNS = [re.compile(p, re.I) for p in (
    r"429|500|502|503|504|524",
    r"rate increased too quickly|rate limit|rate-limit|rate_limit|too many requests",
    r"overloaded|service unavailable|service_unavailable|service-unavailable|internal error|internal_error|"
    r"internal server error|server error|server_error|server-error|provider returned error|provider_returned_error|"
    r"provider-returned-error",
    r"terminated|fetch failed|failed to fetch|network[-_\s]error|upstream connect|connection error|"
    r"connection refused|connection lost|socket connection was closed|socket hang up|reset before headers|"
    r"getaddrinfo|enotfound|eai_again|econnrefused|econnreset|etimedout",
    r"^timeout$|\b(?:request|response|connection|network|stream|read) (?:timeout|timed out|time out)\b",
    r"try your request again|retry your request|resource exhausted|resource_exhausted",
    r"\btry again (?:later|in\b)|\b(?:currently|temporarily) at capacity\b")]
MAX_RETRIES = 5        # RETRY_MAX_RETRIES
RUNAWAY_CAP = 40       # the simulator stops a re-request loop here (the real client went to 45-970)


def matches_retry_pattern(value):
    return isinstance(value, str) and any(pattern.search(value) for pattern in RETRYABLE_MESSAGE_PATTERNS)


def sse_events(body):
    events = []
    for block in body.replace(b"\r\n", b"\n").split(b"\n\n"):
        data = b"\n".join(line[5:].lstrip() for line in block.split(b"\n") if line.startswith(b"data:"))
        if data and data != b"[DONE]":
            events.append(json.loads(data))
    return events


def opencode_reaction(wire_name, status, body):
    """What opencode 1.19.5 does with one answer (report 06 S6.9): ``success``, ``silent_success`` (an empty
    ``response.incomplete`` accepted as a finished turn), ``fail`` (one attempt, error surfaced), ``retry`` (backoff
    and re-send, at most five times) or ``loop`` (a stream without terminal: immediate re-request)."""
    if status != 200:
        try:
            error = json.loads(body).get("error")
        except ValueError:
            error = None
        message = error.get("message") if isinstance(error, dict) else None
        if status in (408, 409, 429) or status >= 500 or matches_retry_pattern(message) \
                or matches_retry_pattern(body.decode(errors="replace")):
            return "retry"
        return "fail"
    events = [event for event in sse_events(body) if event.get("type") != "ping"]
    kinds = [event.get("type") for event in events]
    if wire_name == "anthropic_messages":
        if "error" in kinds:
            if kinds[0] == "error":
                return "retry"  # a first-event stream error becomes APIError 500
            text = json.dumps(events[kinds.index("error")].get("error"))
            lower = text.lower()
            return "retry" if matches_retry_pattern(text) or "exhausted" in lower or "unavailable" in lower else "fail"
        return "success" if "message_stop" in kinds else "loop"
    if "response.failed" in kinds or "error" in kinds:
        return "retry"
    if "response.incomplete" in kinds:
        return "silent_success"
    return "success" if "response.completed" in kinds else "loop"


def run_opencode_client(send, wire_name):
    """Drive ``send() -> (status, body)`` with the simulated client rule; returns (attempts, final reaction)."""
    attempts, retries = 0, 0
    while True:
        status, body = send()
        attempts += 1
        reaction = opencode_reaction(wire_name, status, body)
        if reaction == "retry" and retries < MAX_RETRIES:
            retries += 1
            continue
        if reaction == "loop" and attempts < RUNAWAY_CAP:
            continue
        return attempts, reaction


def anthropic_error_body(message, error_type="invalid_request_error"):
    return json.dumps({"type": "error", "error": {"type": error_type, "message": message}}).encode()


def openai_error_body(message, error_type="invalid_request_error"):
    return json.dumps({"error": {"message": message, "type": error_type, "param": None, "code": None}}).encode()


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def rehydrate_images(node):
    """The continuity fixture stores media as digests (``<sha256:... len=N>``); the egress scrub rightly rejects a
    Responses ``input_image`` whose URL is not ``data:``. Map each digest to a deterministic data URL (equal digests,
    equal URLs), so every continuity relation of the captured sequence is preserved."""
    if isinstance(node, dict):
        out = {k: rehydrate_images(v) for k, v in node.items()}
        url = out.get("image_url")
        if out.get("type") == "input_image" and isinstance(url, str) and not url.startswith("data:"):
            out["image_url"] = "data:image/png;base64," + base64.b64encode(url.encode()).decode()
        return out
    if isinstance(node, list):
        return [rehydrate_images(v) for v in node]
    return node


class SimulatedClientMatchesSpikeTest(unittest.TestCase):
    """The simulator reproduces the measured attempt counts of report 06 S6.9 (s6-retry-*, s6-failform-*)."""

    def run_constant(self, wire_name, status, body):
        return run_opencode_client(lambda: (status, body), wire_name)

    def test_http_answers(self):
        for wire_name, error in (("anthropic_messages", anthropic_error_body), ("openai_responses", openai_error_body)):
            cases = [(400, "auriga relay denied this request", 1), (400, "auriga relay denied: rate limit reached", 6),
                     (400, "auriga relay denied call 503", 6), (400, "auriga relay denied: connection refused", 6),
                     (409, "auriga relay denied this request", 6), (429, "auriga relay denied this request", 6),
                     (500, "provider internal problem", 6), (529, "provider is busy", 6)]
            for status, message, expected in cases:
                with self.subTest(wire=wire_name, status=status, message=message):
                    self.assertEqual(self.run_constant(wire_name, status, error(message))[0], expected)

    def test_stream_answers(self):
        start = sse("message_start", {"type": "message_start", "message": {"id": "m", "content": []}})
        unusable = sse("error", {"type": "error", "error": {"type": "invalid_request_error",
                                                            "message": "auriga_model_response_unusable"}})
        overloaded = sse("error", {"type": "error", "error": {"type": "overloaded_error", "message": "mid stream failure"}})
        ping = b'event: ping\ndata: {"type": "ping"}\n\n'
        anthropic = [(start + unusable, 1, "fail"), (unusable, 6, "retry"), (ping + unusable, 6, "retry"),
                     (start + overloaded, 6, "retry"), (start, RUNAWAY_CAP, "loop")]
        failed = sse("response.failed", {"type": "response.failed", "response": {"status": "failed", "error": {
            "code": "auriga_model_response_unusable", "message": "auriga_model_response_unusable"}}})
        incomplete = sse("response.incomplete", {"type": "response.incomplete",
                                                 "response": {"status": "incomplete", "output": []}})
        created = sse("response.created", {"type": "response.created", "response": {"status": "in_progress"}})
        responses = [(created + failed, 6, "retry"), (incomplete, 1, "silent_success"), (created, RUNAWAY_CAP, "loop")]
        for wire_name, cases in (("anthropic_messages", anthropic), ("openai_responses", responses)):
            for body, attempts, reaction in cases:
                with self.subTest(wire=wire_name, body=body[:60]):
                    self.assertEqual(self.run_constant(wire_name, 200, body), (attempts, reaction))


# ------------------------------------------------------------------------------------------------- adapters
class OpenCodeProfileUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())["cases"]

    def wires(self):
        return (AnthropicMessagesWire(client="opencode"), OpenAIResponsesWire(client="opencode"))

    def test_profile_and_worker_round_trip(self):
        self.assertEqual(OPENCODE_PROFILE, wire_base.ClientProfile(early_stream=False, client_loop_guard=True))
        for wire in self.wires():
            with self.subTest(wire=wire.name):
                self.assertEqual((wire.client, wire.early_stream, wire.client_loop_guard), ("opencode", False, True))
                self.assertIsNone(wire.keepalive_frame())
                self.assertIsNone(wire.failed_payload(dict(status="failed", output=[]), streaming=True))
                restored = wire_from_name(wire.name, client="opencode")
                self.assertEqual((type(restored), restored.client), (type(wire), "opencode"))
        codex = OpenAIResponsesWire()
        self.assertEqual((codex.client, codex.early_stream), ("codex", True))
        self.assertIsNotNone(codex.keepalive_frame())

    def test_routes(self):
        responses = OpenAIResponsesWire(client="opencode")
        self.assertEqual(responses.route("POST", "/v1/responses"), Route("generation", "/responses", "/responses"))
        self.assertEqual(responses.route("POST", "/responses"), Route("generation", "/responses", "/responses"))
        self.assertEqual(responses.route("POST", "/v1/responses/compact"),
                         Route("generation", "/responses/compact", "/responses/compact", compaction=True))
        self.assertEqual(responses.route("POST", "/v1/responses?x=1").upstream_path, "/responses?x=1")
        self.assertEqual(responses.route("POST", "/v1/responses?x=1").endpoint, "/responses")
        for method, path in (("GET", "/v1/responses"), ("POST", "/v1/chat/completions"), ("POST", "/v1/models")):
            self.assertIsNone(responses.route(method, path))
        # Codex keeps exact routes only.
        self.assertIsNone(OpenAIResponsesWire().route("POST", "/v1/responses"))
        anthropic = AnthropicMessagesWire(client="opencode")
        self.assertEqual(anthropic.route("POST", self.fixture["anthropic_messages"]["path"]).kind, "generation")
        self.assertEqual(self.fixture["openai_responses"]["path"], "/v1/responses")
        self.assertEqual(responses.route("POST", self.fixture["openai_responses"]["path"]).kind, "generation")

    def test_thread_key_from_captured_headers(self):
        for wire in self.wires():
            with self.subTest(wire=wire.name):
                headers = self.fixture[wire.name]["headers"]
                self.assertTrue(headers["x-opencode-session-id"].startswith("ses_"))
                self.assertEqual(wire.thread_key(headers), headers["x-opencode-session-id"])
                message = Message()
                message["X-OpenCode-Session-Id"] = "ses_case"
                self.assertEqual(wire.thread_key(message), "ses_case")
                self.assertEqual(wire.thread_key({}), "")
        # The Claude Code / Codex keys are not read on the opencode profile and vice versa.
        self.assertEqual(AnthropicMessagesWire(client="opencode").thread_key({"X-Claude-Code-Session-Id": "x"}), "")
        self.assertEqual(AnthropicMessagesWire().thread_key({"x-opencode-session-id": "x"}), "")
        message = Message()
        message["thread-id"] = "codex-thread"
        self.assertEqual(OpenAIResponsesWire(client="opencode").thread_key(message), "")
        self.assertEqual(OpenAIResponsesWire().thread_key(message), "codex-thread")

    def test_relay_generated_answers_are_neutral_single_attempt_400s(self):
        forbidden = ("rate limit", "overloaded", "timeout", "try again", "capacity", "server error", "connection",
                     "unavailable", "exhausted", "logical", "evidence")
        for wire in self.wires():
            replies = {kind: wire.denial(kind) for kind in wire_base.OPENCODE_DENIAL_MESSAGES}
            replies["relay_error"] = wire.relay_error(streaming=False, evidence=dict(http_status=503, code="abc123",
                                                                                     detail="logical-call-9"),
                                                      manual_retry_allowed=True)
            replies["relay_error_streaming_flag"] = wire.relay_error(streaming=True, evidence=None,
                                                                     manual_retry_allowed=False)
            for name, reply in replies.items():
                with self.subTest(wire=wire.name, reply=name):
                    self.assertEqual(reply.status, 400)
                    self.assertEqual(dict(reply.headers)["Content-Type"], "application/json")
                    self.assertIsNone(re.search(rb"\d", reply.body))
                    for word in forbidden:
                        self.assertNotIn(word, reply.body.decode().lower())
                    self.assertEqual(opencode_reaction(wire.name, reply.status, reply.body), "fail")
                    run = run_opencode_client(lambda reply=reply: (reply.status, reply.body), wire.name)
                    self.assertEqual(run, (1, "fail"))
            self.assertEqual(len({reply.body for reply in replies.values()}), len(wire_base.OPENCODE_DENIAL_MESSAGES) + 1)
            missing = wire.not_found()
            self.assertEqual(missing.status, 404)
            self.assertIsNone(re.search(rb"\d", missing.body))
        self.assertEqual(json.loads(AnthropicMessagesWire(client="opencode").denial("budget").body)["error"]["type"],
                         "invalid_request_error")
        self.assertEqual(json.loads(OpenAIResponsesWire(client="opencode").denial("budget").body),
                         {"error": {"message": "auriga_model_budget_spent", "type": "invalid_request_error",
                                    "param": None, "code": None}})
        # Codex answers are unchanged.
        self.assertEqual(OpenAIResponsesWire().denial("budget").status, 429)
        self.assertEqual(OpenAIResponsesWire().not_found().body, None)


class _Presentation:
    def __init__(self, case):
        import tempfile
        from auriga.core.presentation.actor_presentation import ActorPresentation
        from auriga.core.storage.store import Store
        directory = tempfile.TemporaryDirectory()
        case.addCleanup(directory.cleanup)
        store = Store(Path(directory.name) / "run", dict(run_id="relay", episode_id="episode",
                                                         env_instance_id="environment", env_generation=1))
        case.addCleanup(store.close)
        self.presentation = ActorPresentation(store)


class OpenCodeEgressTest(unittest.TestCase):
    """Egress projection of real opencode bodies on both wires (planted host data, protected protocol fields)."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())["cases"]

    def setUp(self):
        self.presentation = _Presentation(self).presentation

    def project(self, wire, body):
        return wire.project_request(body, self.presentation, registered_tools=frozenset({"fast", "image"}),
                                    tool_name_prefix="auriga_")

    @staticmethod
    def plant_responses(body):
        body = json.loads(json.dumps(body))
        developer = body["input"][0]
        developer["content"] = developer["content"].replace(
            "<env>", "Instructions from: " + INSTRUCTIONS_FILE + "\nRUN " + PLANTED + "\n<env>", 1)
        for item in body["input"][1:]:
            if item.get("type") == "function_call_output":
                if isinstance(item["output"], str):
                    item["output"] += " " + PLANTED
                else:
                    item["output"][0]["text"] += " " + PLANTED
            elif item.get("type") == "function_call":
                item["arguments"] = json.dumps(dict(json.loads(item["arguments"]), note=PLANTED))
            elif isinstance(item.get("content"), list):
                item["content"][0]["text"] += " " + PLANTED
        return body

    def assert_clean(self, value):
        serialized = json.dumps(value)
        for planted in (DIGEST, ARTIFACT, LONG_OBSERVATION, INSTRUCTIONS_FILE):
            self.assertNotIn(planted, serialized)

    def test_responses_real_body(self):
        wire = OpenAIResponsesWire(client="opencode")
        original = self.fixture["openai_responses"]["last_body"]
        for key in ("prompt_cache_key", "include", "store", "max_output_tokens", "tool_choice", "stream"):
            self.assertIn(key, original)
        self.assertNotIn("instructions", original)
        planted = self.plant_responses(original)
        output = self.project(wire, planted)
        self.assert_clean(output)
        self.assertIn("\nInstructions:\nRUN ", output["input"][0]["content"])
        for key in ("model", "prompt_cache_key", "include", "store", "max_output_tokens", "tool_choice", "stream"):
            self.assertEqual(output[key], original[key], key)
        self.assertEqual([(t["type"], t["name"], t.get("strict")) for t in output["tools"]],
                         [(t["type"], t["name"], t.get("strict")) for t in original["tools"]])
        for before, after in zip(original["input"], output["input"]):
            self.assertEqual((after.get("type"), after.get("role"), after.get("call_id"), after.get("name")),
                             (before.get("type"), before.get("role"), before.get("call_id"), before.get("name")))
            if isinstance(before.get("output"), list):
                self.assertEqual([p for p in after["output"] if p["type"] == "input_image"],
                                 [p for p in before["output"] if p["type"] == "input_image"])
        self.assertEqual(self.project(wire, output), output)
        # The unplanted real body is unchanged apart from scrubbable prose.
        self.assertEqual(self.project(wire, original)["input"][1:], original["input"][1:])

    def test_anthropic_real_body_instruction_header(self):
        original = self.fixture["anthropic_messages"]["last_body"]
        body = json.loads(json.dumps(original))
        body["system"][0]["text"] = body["system"][0]["text"].replace(
            "<env>", "Instructions from: " + INSTRUCTIONS_FILE + "\nRUN " + PLANTED + "\n<env>", 1)
        output = self.project(AnthropicMessagesWire(client="opencode"), body)
        self.assert_clean(output)
        self.assertIn("\nInstructions:\nRUN ", output["system"][0]["text"])
        self.assertEqual(output["system"][0]["cache_control"], original["system"][0]["cache_control"])
        self.assertEqual([t.get("eager_input_streaming") for t in output["tools"]],
                         [t.get("eager_input_streaming") for t in original["tools"]])
        self.assertEqual(output["messages"], self.project(AnthropicMessagesWire(client="opencode"), original)["messages"])
        # Only the opencode profile rewrites the header; only the system text is touched.
        claude = present_messages_request(body, self.presentation, registered_tools=frozenset(),
                                          tool_name_prefix="mcp__auriga__", client="claude_code")
        self.assertIn(INSTRUCTIONS_FILE, claude["system"][0]["text"])
        user_header = dict(original, messages=[dict(role="user", content="Instructions from: " + INSTRUCTIONS_FILE)])
        self.assertIn(INSTRUCTIONS_FILE, json.dumps(self.project(AnthropicMessagesWire(client="opencode"), user_header)))

    def test_responses_continuity_inputs_separate_developer_text(self):
        body = self.fixture["openai_responses"]["first_body"]
        wire = OpenAIResponsesWire(client="opencode")
        view = cs.openai_responses_history(body)
        self.assertEqual(len(view.system), 1)
        self.assertEqual(view.system[0]["role"], "developer")
        inputs = wire.continuity_inputs(body)
        self.assertFalse(inputs["helper"])
        self.assertEqual(len(inputs["fingerprints"]), len(body["input"]) - 1)


# ------------------------------------------------------------------------------------- relay x fake servers
class OpenCodeRelayTest(RelayContinuityCase):
    """The real relay with the opencode profiles against the fake servers."""

    WIRES = (("anthropic_messages", "/v1/messages", FakeAnthropicServer, ""),
             ("openai_responses", "/v1/responses", FakeResponsesServer, ""))

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())["cases"]
        cls.continuity = {sequence["id"]: sequence for sequence in load_fixture()["sequences"]}

    def opencode_proxy(self, wire_name, server, **kwargs):
        wire = wire_from_name(wire_name, client="opencode")
        return self.proxy(server, wire, **kwargs)

    def headers(self, wire_name, session="ses_fixture"):
        headers = tuple((k, v) for k, v in self.fixture[wire_name]["headers"].items()
                        if k.lower() not in ("x-opencode-session-id", "x-session-affinity", "x-session-id"))
        return headers + (("x-opencode-session-id", session),)

    def body(self, wire_name, **extra):
        return json.dumps(dict(self.fixture[wire_name]["first_body"], **extra)).encode()

    def client(self, proxy, wire_name, data, path):
        return run_opencode_client(lambda: post(proxy, path, data, self.headers(wire_name))[::2], wire_name)

    def test_success_is_delivered_whole_and_nothing_is_sent_before_validation(self):
        for wire_name, path, server_cls, _ in self.WIRES:
            with self.subTest(wire=wire_name):
                gate = Gate()
                server = self.server(server_cls, [Step(text="final answer", gate=gate)])
                proxy = self.opencode_proxy(wire_name, server)
                received = []
                worker = threading.Thread(target=lambda: received.append(post(proxy, path, self.body(wire_name),
                                                                              self.headers(wire_name))))
                worker.start()
                self.assertTrue(gate.wait_entered(5))
                time.sleep(0.4)
                self.assertEqual(received, [])  # no status line, no keepalive while the upstream answer is pending
                gate.release()
                worker.join(10)
                status, headers, body = received[0]
                self.assertEqual(status, 200)
                self.assertEqual(headers["content-length"], str(len(body)))
                self.assertNotIn(b"keepalive", body)
                self.assertNotIn(b"ping", body)
                self.assertEqual(opencode_reaction(wire_name, status, body), "success")
                self.assertEqual(proxy.request_count, 1)
                forwarded = server.generation_requests()[0]
                self.assertEqual(forwarded.headers["x-opencode-session-id"], "ses_fixture")
                completed = self.events(proxy, "model_call_completed")[0]
                self.assertEqual((completed["native_thread_id"], completed["call_purpose"]), ("ses_fixture", "main"))
                if wire_name == "openai_responses":
                    self.assertEqual(forwarded.route, "/responses")

    def test_denials_cost_exactly_one_client_attempt(self):
        for wire_name, path, server_cls, _ in self.WIRES:
            server = self.server(server_cls)
            for kind in ("budget", "transport_blocked", "thread_cancelled"):
                with self.subTest(wire=wire_name, denial=kind):
                    proxy = self.opencode_proxy(wire_name, server, max_requests=1)
                    if kind == "budget":
                        with proxy.lock:
                            proxy.request_count = 1
                    elif kind == "transport_blocked":
                        with proxy.lock:
                            proxy.blocked_error = dict(logical_call_id="x", error_type="Fixture",
                                                       manual_retry_allowed=True, evidence=None)
                    else:
                        proxy.register_thread("ses_fixture", "actor", "ses_fixture")
                        proxy.cancel_thread("ses_fixture")
                    attempts, reaction = run_opencode_client(
                        lambda proxy=proxy: post(proxy, path, self.body(wire_name), self.headers(wire_name),
                                                 bookkeeping=None)[::2], wire_name)
                    self.assertEqual((attempts, reaction), (1, "fail"))
            self.assertEqual(server.generation_requests(), [])

    def test_failed_deliveries_cost_exactly_one_client_attempt(self):
        anthropic_failures = [
            Step(failure=Failure("stream_error", message="mid stream failure")),  # overloaded_error event
            Step(failure=Failure("stream_error", 400, "bad", error_type="invalid_request_error")),
            Step(text="cut", failure=Failure("truncated")),
            Step(tool_calls=[ToolCall("auriga_fast", {"note": "x"})], failure=Failure("malformed_tool_json"),
                 stop_reason="max_tokens"),
            Step(failure=Failure("http", 529, "provider is busy")),
            Step(failure=Failure("http", 400, "provider 400 text", error_type="invalid_request_error")),
        ]
        responses_failures = [
            Step(failure=Failure("failed", error_type="server_error")),
            Step(text="partial", failure=Failure("incomplete")),
            Step(failure=Failure("stream_error", message="mid stream failure")),
            Step(text="cut", failure=Failure("truncated")),
            Step(failure=Failure("http", 503, "provider unavailable")),
        ]
        for (wire_name, path, server_cls, _), failures in zip(self.WIRES, (anthropic_failures, responses_failures)):
            for retries in (0, 2):
                for step in failures:
                    with self.subTest(wire=wire_name, failure=step.failure.kind, status=step.failure.status,
                                      retries=retries):
                        server = self.server(server_cls, default=step)
                        proxy = self.opencode_proxy(wire_name, server, max_retries=retries)
                        attempts, reaction = run_opencode_client(
                            lambda proxy=proxy: post(proxy, path, self.body(wire_name), self.headers(wire_name))[::2],
                            wire_name)
                        self.assertEqual((attempts, reaction), (1, "fail"))
                        self.assertEqual(proxy.request_count, 0)
                        self.assertTrue(proxy.transport_blocked)

    def test_cut_messages_never_reach_the_client(self):
        """T0 for the deferred profile: an upstream answer cut after a tool_use start, mid argument JSON or before
        message_stop yields one neutral 400 and zero content bytes."""
        from test_relay_review_followups import CutMessageNeverReleasedTest, RawUpstream
        for name, raw in CutMessageNeverReleasedTest.CUTS.items():
            with self.subTest(cut=name):
                upstream = RawUpstream([raw])
                self.addCleanup(upstream.close)
                wire = AnthropicMessagesWire(client="opencode")
                proxy = ModelRequestProxy(upstream.base_url, wire=wire, max_retries=0,
                                          upstream_auth_headers={"x-api-key": "k"}).start()
                self.addCleanup(proxy.close)
                status, headers, body = post(proxy, "/v1/messages", self.body("anthropic_messages"),
                                             self.headers("anthropic_messages"))
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(body)["error"]["message"], "auriga_model_request_failed")
                for forbidden in (b"tool_use", b"toolu_cut", b"content_block", b"message_start", b"visible text"):
                    self.assertNotIn(forbidden, body)

    def test_compaction_replays_trigger_exactly_one_discontinuity(self):
        for wire_name, path, server_cls, _ in self.WIRES:
            sequence_id = "opencode/s9-anthropic-compact" if wire_name == "anthropic_messages" \
                else "opencode/s9-responses-compact"
            sequence = self.continuity[sequence_id]
            with self.subTest(wire=wire_name):
                self.reasons = []
                server = self.server(server_cls)
                proxy = self.opencode_proxy(wire_name, server)
                headers = tuple((k, v) for k, v in self.headers(wire_name) if k.lower() != "x-opencode-session-id")
                replayed = dict(sequence, values=rehydrate_images(sequence["values"]))
                statuses = self.replay(proxy, replayed, range(len(sequence["requests"])), path=path,
                                       thread_header="x-opencode-session-id", base_headers=headers)
                self.assertEqual(set(statuses), {200})
                events = self.events(proxy, "model_context_discontinuity")
                lost = [k + 1 for k, request in enumerate(sequence["requests"]) if request["label"] == "context_lost"]
                self.assertEqual([event["logical_call_index"] for event in events], lost)
                self.assertEqual(len(events), 1)
                self.assertEqual(self.reasons, [events[0]["reason"]])
                helper = [k + 1 for k, request in enumerate(sequence["requests"]) if request["label"] == "helper"]
                purposes = {e["logical_call_index"]: e["call_purpose"] for e in self.events(proxy, "model_call_completed")}
                self.assertEqual([index for index, purpose in purposes.items() if purpose == "helper"], helper)
                self.assertEqual(proxy.request_count, len(sequence["requests"]))  # helpers are charged too


if __name__ == "__main__":
    unittest.main()
