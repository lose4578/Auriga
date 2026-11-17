"""ContinuityTracker call sites in ModelRequestProxy (design §4.3, P1-3a integration).

Real spike request sequences (``tests/fixtures/continuity/continuity_fixture.json``) are replayed through the real relay
against ``FakeAnthropicServer``: a Claude Code auto-compaction produces exactly one discontinuity (with the
``/responses/compact`` effects: new image-context epoch, trim window reset, one ``model_context_discontinuity`` event,
``on_context_discontinuity(reason)``), normal multi-step sessions produce none, the relay's own image trim produces
none, and the baseline only advances on a delivered usable completion. The Codex profile installs no tracker.

Revision 4 R4.3 (work package R4-4): the probe R4-0 captures (real Claude Code subagents, foreground, background and
interrupted, on the session's own thread with their own system text) and subagent chains interleaved with captured
main chains (``test_context_signal.subagent_interleaving``: the captured R4-0 subagent requests, or SYNTHETIC ones built
from spike rows; an OpenCode child session with its own session id and the main system text is synthetic, none was
captured). Subagent requests are attributed ``call_purpose="subagent"`` and never start an image-context epoch; the
main chain's compaction still does, exactly once per compaction.
"""
import base64
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from auriga.core.presentation.actor_presentation import ActorPresentation
from auriga.core.storage.store import Store
from auriga.cortex.agents import context_signal as cs
from auriga.cortex.agents.context_signal import ContinuityTracker
from auriga.cortex.session.model_proxy import ModelRequestProxy, PROVIDER_CONTEXT_EDIT
from auriga.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire

from support.fake_model_servers import Failure, FakeAnthropicServer, Step, sse

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "continuity" / "continuity_fixture.json"
SESSION = "9ff8e771-aa08-4c17-b00d-032e1cb2011f"
PREFIX = "mcp__auriga__"
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="
CLAUDE_HEADERS = (("Content-Type", "application/json"), ("x-api-key", "auriga-local-relay"),
                  ("anthropic-version", "2023-06-01"))
COMPACTION_SEQUENCE = "claude_code/s5-compact/model-requests"


def load_fixture():
    return json.loads(FIXTURE.read_text())


def fixture_body(sequence, request, **extra):
    """The fixture request (system/messages/input/tools) as a complete generation body."""
    values = sequence["values"]
    body = {key: [values[i] for i in ref] if isinstance(ref, list) else values[ref]
            for key, ref in request["body"].items()}
    return dict(dict(model="fixture-model", stream=True), **body, **extra)


def post(proxy, path, data, headers, *, bookkeeping="model_call_finished"):
    """One client request read to EOF; waits for the relay's ``bookkeeping`` event (None: no wait).
    Returns (status, headers, body)."""
    with proxy.lock:
        before = sum(event["kind"] == bookkeeping for event in proxy.events)
    port = int(proxy.base_url.rsplit(":", 1)[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.putrequest("POST", path, skip_accept_encoding=True)
        for key, value in headers:
            connection.putheader(key, value)
        connection.putheader("Content-Length", str(len(data)))
        connection.endheaders(data)
        response = connection.getresponse()
        status, reply_headers, body = response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()
    deadline = time.monotonic() + 10
    while bookkeeping and time.monotonic() < deadline:
        with proxy.lock:
            if sum(event["kind"] == bookkeeping for event in proxy.events) > before:
                break
        time.sleep(0.005)
    return status, {k.lower(): v for k, v in reply_headers.items()}, body


class RelayContinuityCase(unittest.TestCase):
    """Shared relay/fake-server plumbing (no tests here)."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "run", dict(run_id="relay", episode_id="episode",
                                                              env_instance_id="environment", env_generation=1))
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)
        self.reasons = []

    def server(self, cls=FakeAnthropicServer, script=(), **kwargs):
        server = cls(script, **kwargs).start()
        self.addCleanup(server.close)
        return server

    def proxy(self, server, wire, *, upstream_suffix="", **kwargs):
        kwargs.setdefault("request_projection", lambda request: wire.project_request(
            request, self.presentation, registered_tools=frozenset(), tool_name_prefix=PREFIX))
        kwargs.setdefault("on_context_discontinuity", self.reasons.append)
        kwargs.setdefault("max_retries", 0)
        proxy = ModelRequestProxy(server.base_url + upstream_suffix, wire=wire,
                                  upstream_auth_headers={next(iter(wire.auth_header_names.values())): "host-secret"},
                                  **kwargs).start()
        self.addCleanup(proxy.close)
        patch = mock.patch.object(proxy.policy, "_retry_delay_seconds", return_value=0)
        patch.start()
        self.addCleanup(patch.stop)
        return proxy

    @staticmethod
    def events(proxy, kind):
        with proxy.lock:
            return [dict(event) for event in proxy.events if event["kind"] == kind]

    def replay(self, proxy, sequence, rows, *, path, thread_header, base_headers, session_prefix="s"):
        """Send fixture rows of one sequence in order; every answer is the fake's default text."""
        statuses = []
        for k in rows:
            request = sequence["requests"][k]
            headers = base_headers + ((thread_header, session_prefix + "-" + request["thread"]),)
            status, _, _ = post(proxy, path, json.dumps(fixture_body(sequence, request)).encode(), headers)
            statuses.append(status)
        return statuses


class ClaudeCodeContinuityTest(RelayContinuityCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture()
        cls.sequences = {sequence["id"]: sequence for sequence in cls.fixture["sequences"]}

    def replay_claude(self, proxy, sequence, rows, session_prefix="s"):
        return self.replay(proxy, sequence, rows, path="/v1/messages?beta=true",
                           thread_header="X-Claude-Code-Session-Id", base_headers=CLAUDE_HEADERS,
                           session_prefix=session_prefix)

    def test_replayed_compaction_triggers_exactly_one_discontinuity(self):
        sequence = self.sequences[COMPACTION_SEQUENCE]
        labels = [request["label"] for request in sequence["requests"]]
        first, second = [k for k, label in enumerate(labels) if label == "compaction_request"]
        lost = labels.index("context_lost")
        self.assertEqual(lost, first + 1)
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire())
        statuses = self.replay_claude(proxy, sequence, range(second))  # one compaction, stop before the next
        self.assertEqual(set(statuses), {200})
        events = self.events(proxy, "model_context_discontinuity")
        self.assertEqual(len(events), 1, events)
        event = events[0]
        self.assertEqual(event["reason"], "history_shortened")
        self.assertEqual(event["logical_call_index"], lost + 1)
        self.assertEqual(event["native_thread_id"], "s-t1")
        self.assertEqual(event["call_purpose"], "main")
        self.assertEqual(self.reasons, ["history_shortened"])
        self.assertEqual(proxy.image_context_epoch, event["logical_call_id"])
        # Claude Code's own summary request carries tools and only appends: continuous, main.
        started = {e["logical_call_index"]: e for e in self.events(proxy, "model_call_completed")}
        self.assertEqual(started[first + 1]["call_purpose"], "main")
        self.assertEqual(proxy.request_count, second)
        # Digests are internal comparison keys: they never reach events.
        recorded = json.dumps(proxy.events)
        for key in ("fingerprints", "system_key", "image_count"):
            self.assertNotIn('"%s"' % key, recorded)

    def test_full_compaction_sequence_reports_each_compaction_once(self):
        sequence = self.sequences[COMPACTION_SEQUENCE]
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire())
        self.replay_claude(proxy, sequence, range(len(sequence["requests"])))
        lost = [k + 1 for k, request in enumerate(sequence["requests"]) if request["label"] == "context_lost"]
        events = self.events(proxy, "model_context_discontinuity")
        self.assertEqual([event["logical_call_index"] for event in events], lost)
        self.assertEqual(self.reasons, ["history_shortened"] * 2)
        self.assertEqual(proxy.image_context_epoch, events[-1]["logical_call_id"])

    def test_normal_multi_step_sessions_trigger_none(self):
        """Every Claude Code spike session without compaction, prune or undelivered answer (tool rounds, images,
        parallel calls, steering merges, title helper calls, cache-marker moves) replays without a discontinuity."""
        clean = [sequence for sequence in self.fixture["sequences"] if sequence["client"] == "claude_code"
                 and all(r["delivered"] and r["status"] == 200 and r["label"] in ("continuous", "helper")
                         for r in sequence["requests"])]
        self.assertGreaterEqual(len(clean), 40)
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire(), max_requests=180 * 4)
        total = helpers = 0
        for index, sequence in enumerate(clean):
            self.replay_claude(proxy, sequence, range(len(sequence["requests"])), session_prefix="seq%d" % index)
            total += len(sequence["requests"])
            helpers += sum(request["label"] == "helper" for request in sequence["requests"])
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
        self.assertEqual(self.reasons, [])
        self.assertIsNone(proxy.image_context_epoch)
        completed = self.events(proxy, "model_call_completed")
        self.assertEqual(len(completed), total)
        self.assertEqual(sum(event["call_purpose"] == "helper" for event in completed), helpers)
        self.assertGreater(helpers, 0)

    def tool_session(self, steps):
        """A Claude-Code-shaped history after ``steps`` tool rounds, one tool-result image per round."""
        messages = [dict(role="user", content=[dict(type="text", text="observe the scene")])]
        for k in range(steps):
            call = "toolu_%02d" % k
            messages.append(dict(role="assistant", content=[dict(type="tool_use", id=call, name=PREFIX + "robot_observe",
                                                                input=dict(step=k))]))
            messages.append(dict(role="user", content=[dict(type="tool_result", tool_use_id=call, content=[
                dict(type="text", text=json.dumps(dict(observation_id="obs-%d" % k))),
                dict(type="image", source=dict(type="base64", media_type="image/png", data=PNG))])]))
        return dict(model="fixture-model", max_tokens=1024, stream=True, system="SYSTEM", messages=messages,
                    tools=[dict(name=PREFIX + "robot_observe", input_schema=dict(type="object"))])

    def test_relay_image_trim_does_not_trigger(self):
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire(), max_inline_images=24)
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        for steps in range(1, 33):
            status, _, _ = post(proxy, "/v1/messages", json.dumps(self.tool_session(steps)).encode(), headers)
            self.assertEqual(status, 200)
        self.assertTrue(self.events(proxy, "model_images_trimmed"))
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
        self.assertEqual(self.reasons, [])
        # Negative control: the bodies the relay forwarded (trimmed) would have been flagged had the tracker read them.
        wire, tracker, flagged = AnthropicMessagesWire(), ContinuityTracker(), []
        for recorded in server.generation_requests():
            inputs = wire.continuity_inputs(recorded.json)
            flagged.append(tracker.observe("t", **inputs))
            tracker.commit("t", **inputs)
        self.assertTrue(any(flagged), flagged)

    def test_baseline_advances_only_on_usable_delivery(self):
        def body(*texts):
            messages = []
            for k, text in enumerate(texts):
                messages.append(dict(role="user" if k % 2 == 0 else "assistant", content=text))
            return json.dumps(dict(model="fixture-model", max_tokens=64, stream=True, messages=messages,
                                   tools=[dict(name="t", input_schema=dict(type="object"))])).encode()
        rejected = Step(failure=Failure("http", 400, "fixture-private rejection", error_type="invalid_request_error"))
        server = self.server(script=[Step(text="a1"), rejected, Step(text="a2")])
        proxy = self.proxy(server, AnthropicMessagesWire())
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        self.assertEqual(post(proxy, "/v1/messages", body("u1"), headers)[0], 200)
        # A rewritten history is reported at admission, but its failed answer never becomes the baseline.
        status, _, failed = post(proxy, "/v1/messages", body("u0", "x", "u1b"), headers)
        self.assertEqual(status, 200)  # early stream: the failure is one error event in the open stream
        self.assertIn(b"event: error", failed)
        self.assertTrue(proxy.transport_blocked)
        self.assertEqual(self.reasons, ["history_rewritten"])
        proxy.manual_retry()
        # Continuing the delivered history is continuous (it would be a rewrite against the failed request).
        self.assertEqual(post(proxy, "/v1/messages", body("u1", "a1", "u2"), headers)[0], 200)
        self.assertEqual(self.reasons, ["history_rewritten"])
        self.assertEqual(len(self.events(proxy, "model_context_discontinuity")), 1)

    def test_provider_context_edit_on_a_usable_answer_starts_an_epoch(self):
        class EditingServer(FakeAnthropicServer):
            def stream_parts(self, turn):
                head, body, terminal, error = super().stream_parts(turn)
                delta = {"type": "message_delta", "delta": {"stop_reason": turn.stop, "stop_sequence": None},
                         "usage": {"output_tokens": 1},
                         "context_management": {"applied_edits": [{"type": "clear_tool_uses_20250919"}]}}
                return head, body, [sse("message_delta", delta), terminal[-1]], error
        server = self.server(cls=EditingServer)
        proxy = self.proxy(server, AnthropicMessagesWire())
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        status, _, _ = post(proxy, "/v1/messages", json.dumps(self.tool_session(1)).encode(), headers)
        self.assertEqual(status, 200)
        events = self.events(proxy, "model_context_discontinuity")
        self.assertEqual([event["reason"] for event in events], [PROVIDER_CONTEXT_EDIT])
        self.assertEqual(self.reasons, [PROVIDER_CONTEXT_EDIT])
        self.assertEqual(proxy.image_context_epoch, events[0]["logical_call_id"])

    def test_callback_failure_is_recorded_and_never_fails_the_call(self):
        def explode(reason):
            raise RuntimeError("session callback bug")
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire(), on_context_discontinuity=explode)
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        self.assertEqual(post(proxy, "/v1/messages", json.dumps(self.tool_session(2)).encode(), headers)[0], 200)
        self.assertEqual(post(proxy, "/v1/messages", json.dumps(self.tool_session(1)).encode(), headers)[0], 200)
        failures = self.events(proxy, "model_context_discontinuity_callback_failed")
        self.assertEqual([(event["reason"], event["error_type"]) for event in failures],
                         [("history_shortened", "RuntimeError")])
        self.assertEqual(proxy.request_count, 2)
        self.assertFalse(proxy.transport_blocked)

    def test_discontinuity_resets_the_thread_trim_window_before_trimming(self):
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire(), max_inline_images=24)
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        seen = []
        original = proxy._prepare_responses_body

        def spy(state, *args, **kwargs):
            with proxy.lock:
                seen.append(proxy._image_trim_watermarks.get(SESSION))
            return original(state, *args, **kwargs)
        with mock.patch.object(proxy, "_prepare_responses_body", side_effect=spy):
            for steps in (30, 31, 3):
                post(proxy, "/v1/messages", json.dumps(self.tool_session(steps)).encode(), headers)
        self.assertIsNone(seen[0])
        self.assertIsNotNone(seen[1])
        self.assertIsNone(seen[2])  # reset by the discontinuity before the relay looked at the shortened body
        self.assertEqual(self.reasons, ["history_shortened"])


class SubagentChainRelayTest(RelayContinuityCase):
    """Subagent chains through the real relay, wired as the sessions wire it (``default_thread_id`` + actor thread)."""

    @classmethod
    def setUpClass(cls):
        cls.sequences = {sequence["id"]: sequence for sequence in load_fixture()["sequences"]}

    def actor_proxy(self, wire, *, default=True, **kwargs):
        server = self.server()
        proxy = self.proxy(server, wire, default_thread_id="s-t1" if default else None, **kwargs)
        proxy.register_thread("s-t1", "actor", "s-t1")
        return proxy

    def purposes(self, proxy, kind="model_call_completed"):
        return {event["logical_call_index"]: event.get("call_purpose") for event in self.events(proxy, kind)}

    def expected(self, sequence):
        """``(main context_lost call indexes, {call index: expected call_purpose})``."""
        lost, purposes = [], {}
        for k, request in enumerate(sequence["requests"]):
            purposes[k + 1] = ("helper" if request["label"] == "helper" else
                               "subagent" if request["chain"] == "subagent" else "main")
            if request["chain"] == "main" and request["label"] == "context_lost":
                lost.append(k + 1)
        return lost, purposes

    def test_claude_task_subagent_interleaved_with_the_main_compaction(self):
        from test_context_signal import subagent_interleaving
        sequence = subagent_interleaving(self.sequences[COMPACTION_SEQUENCE],
                                         self.sequences["claude_code/s5-normal/model-requests"])
        proxy = self.actor_proxy(AnthropicMessagesWire())
        statuses = self.replay(proxy, sequence, range(len(sequence["requests"])), path="/v1/messages?beta=true",
                               thread_header="X-Claude-Code-Session-Id", base_headers=CLAUDE_HEADERS)
        self.assertEqual(set(statuses), {200})
        lost, purposes = self.expected(sequence)
        events = self.events(proxy, "model_context_discontinuity")
        self.assertEqual([event["logical_call_index"] for event in events], lost)
        self.assertEqual(len(lost), 2)
        self.assertEqual({event["call_purpose"] for event in events}, {"main"})
        self.assertEqual(self.reasons, ["history_shortened"] * 2)
        self.assertEqual(proxy.image_context_epoch, events[-1]["logical_call_id"])
        self.assertEqual(self.purposes(proxy), purposes)
        self.assertEqual(sum(purpose == "subagent" for purpose in purposes.values()), 24)
        self.assertEqual(proxy.request_count, len(sequence["requests"]))  # subagent calls are charged too
        self.assertEqual({event["native_thread_id"] for event in self.events(proxy, "model_call_completed")}, {"s-t1"})
        # Which events carry call_purpose: every event after the admission, none before the body is decoded.
        sub = next(index for index, purpose in purposes.items() if purpose == "subagent")
        for kind in ("model_call_started", "model_request_body_received"):
            self.assertNotIn("call_purpose", self.events(proxy, kind)[sub - 1])
        for kind in ("model_request_started", "model_request_finished", "model_call_completed", "model_call_finished"):
            self.assertEqual(self.purposes(proxy, kind)[sub], "subagent", kind)

    def test_captured_subagent_sessions(self):
        """Every R4-0 sequence (two subagents in one session, foreground and background, interrupted task trees)."""
        probe = [s for s in self.sequences.values() if s.get("source") == "r4-0"]
        self.assertEqual(len(probe), 4)
        for sequence in probe:
            with self.subTest(sequence=sequence["id"]):
                self.reasons = []
                proxy = self.actor_proxy(AnthropicMessagesWire())
                statuses = self.replay(proxy, sequence, range(len(sequence["requests"])), path="/v1/messages?beta=true",
                                       thread_header="X-Claude-Code-Session-Id", base_headers=CLAUDE_HEADERS)
                self.assertEqual(set(statuses), {200})
                self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
                self.assertEqual(self.reasons, [])
                expected = {k + 1: "subagent" if r["label"] == "subagent" else "main"
                            for k, r in enumerate(sequence["requests"])}
                self.assertEqual(self.purposes(proxy), expected)

    def test_claude_main_compaction_with_captured_subagent_requests(self):
        from test_context_signal import subagent_interleaving
        probe = self.sequences["claude_code/r4-0/cc-agent/model-requests"]
        sub = dict(probe, requests=[r for r in probe["requests"] if r["label"] == "subagent"])
        sequence = subagent_interleaving(self.sequences[COMPACTION_SEQUENCE], sub, sub_system=None,
                                         task_prompt=False, start=4, burst=1)
        proxy = self.actor_proxy(AnthropicMessagesWire())
        self.replay(proxy, sequence, range(len(sequence["requests"])), path="/v1/messages?beta=true",
                    thread_header="X-Claude-Code-Session-Id", base_headers=CLAUDE_HEADERS)
        lost, purposes = self.expected(sequence)
        self.assertEqual([e["logical_call_index"] for e in self.events(proxy, "model_context_discontinuity")], lost)
        self.assertEqual(self.reasons, ["history_shortened"] * 2)
        self.assertEqual(self.purposes(proxy), purposes)

    def test_claude_subagent_compaction_never_starts_an_epoch(self):
        from test_context_signal import subagent_interleaving
        sequence = subagent_interleaving(self.sequences["claude_code/s5-normal/model-requests"],
                                         self.sequences[COMPACTION_SEQUENCE], start=1, burst=1)
        proxy = self.actor_proxy(AnthropicMessagesWire())
        self.replay(proxy, sequence, range(len(sequence["requests"])), path="/v1/messages",
                    thread_header="X-Claude-Code-Session-Id", base_headers=CLAUDE_HEADERS)
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
        self.assertEqual(self.reasons, [])
        self.assertIsNone(proxy.image_context_epoch)
        _, purposes = self.expected(sequence)
        self.assertEqual(self.purposes(proxy), purposes)
        self.assertEqual(sum(purpose == "subagent" for purpose in purposes.values()), 26)

    def test_opencode_child_session_is_a_subagent(self):
        """A child session (own x-opencode-session-id, the main session's system text) prunes its history while the
        main session compacts. The main thread is ``default_thread_id`` or only a registered actor thread."""
        from test_context_signal import MAIN_SYSTEM, subagent_interleaving
        sequence = subagent_interleaving(self.sequences["opencode/s9-anthropic-compact"],
                                         self.sequences["opencode/s9-anthropic-prune"], sub_thread="t-child",
                                         sub_system=MAIN_SYSTEM)
        for default in (True, False):
            with self.subTest(default_thread_id=default):
                self.reasons = []
                proxy = self.actor_proxy(AnthropicMessagesWire(client="opencode"), default=default)
                statuses = self.replay(proxy, sequence, range(len(sequence["requests"])), path="/v1/messages",
                                       thread_header="x-opencode-session-id", base_headers=CLAUDE_HEADERS)
                self.assertEqual(set(statuses), {200})
                lost, purposes = self.expected(sequence)
                events = self.events(proxy, "model_context_discontinuity")
                self.assertEqual([event["logical_call_index"] for event in events], lost)
                self.assertEqual(len(lost), 1)
                self.assertEqual(self.reasons, ["history_shortened"])
                self.assertEqual(self.purposes(proxy), purposes)
                self.assertEqual(sum(purpose == "subagent" for purpose in purposes.values()), 15)
                # Attribution is unchanged: an unregistered child id falls back to default_thread_id when one is set.
                threads = {event["logical_call_index"]: event["native_thread_id"]
                           for event in self.events(proxy, "model_call_completed")}
                child = {threads[index] for index, purpose in purposes.items() if purpose == "subagent"}
                self.assertEqual(child, {"s-t1"} if default else {"s-t-child"})

    def test_provider_context_edit_on_a_subagent_answer_is_not_a_discontinuity(self):
        class EditingServer(FakeAnthropicServer):
            """Reports a server-side context edit on every answer to a request whose system text says EDIT."""
            def stream_parts(self, turn):
                head, body, terminal, error = super().stream_parts(turn)
                if "EDIT" not in json.dumps((turn.req.json or {}).get("system")):
                    return head, body, terminal, error
                delta = {"type": "message_delta", "delta": {"stop_reason": turn.stop, "stop_sequence": None},
                         "usage": {"output_tokens": 1},
                         "context_management": {"applied_edits": [{"type": "clear_tool_uses_20250919"}]}}
                return head, body, [sse("message_delta", delta), terminal[-1]], error
        server = self.server(cls=EditingServer)
        proxy = self.proxy(server, AnthropicMessagesWire(), default_thread_id=SESSION)
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        tools = [dict(name=PREFIX + "robot_observe", input_schema=dict(type="object"))]

        def body(system, *texts):
            messages = [dict(role="user" if k % 2 == 0 else "assistant", content=text) for k, text in enumerate(texts)]
            return json.dumps(dict(model="fixture-model", max_tokens=64, stream=True, system=system,
                                   messages=messages, tools=tools)).encode()
        self.assertEqual(post(proxy, "/v1/messages", body("MAIN", "u1"), headers)[0], 200)
        self.assertEqual(post(proxy, "/v1/messages", body("SUBAGENT EDIT", "task"), headers)[0], 200)
        self.assertEqual(post(proxy, "/v1/messages", body("MAIN", "u1", "done", "u2"), headers)[0], 200)
        self.assertEqual(list(self.purposes(proxy).values()), ["main", "subagent", "main"])
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
        self.assertEqual(self.reasons, [])
        # The same edit on a main answer is one (the P1-3a behaviour).
        self.assertEqual(post(proxy, "/v1/messages", body("MAIN EDIT", "u1", "done", "u2", "a", "u3"), headers)[0], 200)
        self.assertEqual([event["reason"] for event in self.events(proxy, "model_context_discontinuity")],
                         [cs.SYSTEM_CHANGED, PROVIDER_CONTEXT_EDIT])


class CodexProfileTest(unittest.TestCase):
    def test_codex_profile_installs_no_tracker(self):
        self.assertIsNone(ModelRequestProxy("http://127.0.0.1:1").continuity)
        self.assertIsNone(ModelRequestProxy("http://127.0.0.1:1", wire=OpenAIResponsesWire(client="codex")).continuity)
        self.assertIsInstance(ModelRequestProxy("http://127.0.0.1:1", wire=AnthropicMessagesWire()).continuity,
                              ContinuityTracker)

    def test_adapter_continuity_inputs_delegate_to_context_signal(self):
        from auriga.cortex.agents import context_signal as cs
        body = dict(system=[dict(type="text", text="S", cache_control=dict(type="ephemeral"))],
                    messages=[dict(role="user", content="hi"),
                              dict(role="user", content=[dict(type="image", source=dict(
                                  type="base64", media_type="image/png", data=base64.b64encode(b"x").decode()))])])
        wire = AnthropicMessagesWire()
        view = cs.anthropic_messages_history(body)
        self.assertEqual(wire.continuity_inputs(body), dict(
            helper=True, fingerprints=cs.history_fingerprints(view), image_count=1,
            system_key=cs.conversation_key(view)))
        self.assertEqual(wire.conversation_key(body), cs.conversation_key(view))
        self.assertEqual(wire.history_fingerprints(body), cs.history_fingerprints(view))
        self.assertEqual(wire.history_image_count(body), 1)
        self.assertFalse(wire.is_helper_request(dict(body, tools=[dict(name="t")])))
        responses = dict(instructions="I", input=[dict(role="developer", content="D"), dict(role="user", content="u")])
        rview = cs.openai_responses_history(responses)
        self.assertEqual(OpenAIResponsesWire().history_fingerprints(responses), cs.history_fingerprints(rview))
        self.assertEqual(OpenAIResponsesWire().conversation_key(responses), cs.conversation_key(rview))


if __name__ == "__main__":
    unittest.main()
