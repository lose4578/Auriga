"""Relay consume guard (``ModelRequestProxy``, class docstring "Consume guard"; work package CONSUME-GUARD).

Defect (T12, opencode 1.19.5, ``scratch/t12/probe400.py <wire> nullinput``): the provider's usage holds a value the
client's AI SDK cannot parse (``input_tokens: null``). The relay validates, delivers and charges every answer, but
OpenCode cannot finish the step and asks again: on the Responses wire at once, with the delivered text appended and no
new input (all 180 budgeted calls in about 4 s); on the Anthropic wire with the unchanged history after a backoff. Every
re-request has a different body, so the identical-body runaway guard never fires.

The tests drive the real relay against scripted fake providers that send exactly that usage and a client loop that
re-requests the way OpenCode did. The loop is blocked after ``CONSUME_GUARD_MAX_IDLE_COMPLETIONS`` idle completions, and
at most K + 1 completions are charged. Every captured Claude Code / OpenCode session of the continuity fixture, replayed
through the real relay with its own client profile, never produces an idle completion, and neither do synthetic tool
loops. The Codex profile has no guard, and its forwarded bytes stay pinned by ``test_wire_golden``.
"""
import json
import threading
import unittest
from unittest import mock

from auriga.cortex.session.model_proxy import ModelRequestProxy
from auriga.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire, wire_from_name

from support.fake_model_servers import FakeAnthropicServer, FakeResponsesServer, Gate, Step, text_step
from test_model_proxy_continuity import CLAUDE_HEADERS, RelayContinuityCase, load_fixture, post
from test_wire_opencode import FIXTURE as OPENCODE_FIXTURE, opencode_reaction, rehydrate_images, sse_events

K = ModelRequestProxy.CONSUME_GUARD_MAX_IDLE_COMPLETIONS
CONSUME_EVENT = "model_client_did_not_consume_completion"
OPENCODE_THREAD = "x-opencode-session-id"
CLAUDE_THREAD = "X-Claude-Code-Session-Id"
TOOLS = [dict(name="mcp__auriga__look", description="Read-only view of the scene.", input_schema=dict(type="object"))]
RESPONSES_TOOLS = [dict(type="function", name="auriga_look", description="Read-only view of the scene.",
                        parameters=dict(type="object"), strict=False)]


class NullUsageResponses(FakeResponsesServer):
    """Responses provider whose usage carries ``input_tokens: null`` (the T12 trigger)."""

    def _usage(self, u):
        usage = super()._usage(u)
        usage.update(input_tokens=None, total_tokens=None)
        return usage


class NullUsageAnthropic(FakeAnthropicServer):
    """Anthropic provider whose usage carries ``input_tokens: null`` (the T12 trigger)."""

    def _usage(self, u, output=None):
        usage = super()._usage(u, output)
        usage["input_tokens"] = None
        return usage


def delivered_responses_text(body):
    """Assistant text of a delivered Responses event stream (what OpenCode echoes back)."""
    for event in sse_events(body):
        if event.get("type") == "response.completed":
            return "".join(part.get("text", "") for item in event["response"]["output"] if item.get("type") == "message"
                           for part in item.get("content") or [])
    return ""


class GuardCase(RelayContinuityCase):
    """Relay plumbing shared by the consume-guard tests (no tests here)."""

    @classmethod
    def setUpClass(cls):
        cls.opencode = json.loads(OPENCODE_FIXTURE.read_text())["cases"]

    def opencode_headers(self, wire_name, session="ses_fixture"):
        headers = tuple((k, v) for k, v in self.opencode[wire_name]["headers"].items()
                        if k.lower() not in (OPENCODE_THREAD, "x-session-affinity", "x-session-id"))
        return headers + ((OPENCODE_THREAD, session),)

    def send(self, proxy, path, body, headers):
        status, _, data = post(proxy, path, json.dumps(body).encode(), headers)
        return status, data

    def assert_consume_block(self, proxy, server, statuses):
        """The loop was refused right after K idle delivered completions: K + 1 forwarded and charged."""
        self.assertEqual(statuses, [200] * (K + 1) + [statuses[-1]])
        self.assertNotEqual(statuses[-1], 200)
        self.assertEqual(len(server.generation_requests()), K + 1)
        self.assertEqual((proxy.request_count, proxy.reserved_calls), (K + 1, 0))
        evidence = dict(classification="client_did_not_consume_completion", idle_completions=K,
                        max_idle_completions=K)
        self.assertEqual(proxy.blocked_error["error_type"], "ClientDidNotConsumeCompletion")
        self.assertTrue(proxy.blocked_error["manual_retry_allowed"])
        self.assertEqual(proxy.blocked_error["evidence"], evidence)
        event, = self.events(proxy, CONSUME_EVENT)
        self.assertEqual((event["logical_call_index"], event["call_purpose"], event["idle_completions"]),
                         (K + 2, "main", K))
        self.assertEqual(len(self.events(proxy, "model_transport_blocked")), 1)
        self.assertEqual(self.events(proxy, "model_client_retry_runaway"), [])
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
        finished = {e["logical_call_index"]: e for e in self.events(proxy, "model_call_finished")}
        self.assertFalse(finished[K + 2]["delivered"])


class UnconsumedCompletionLoopTest(GuardCase):
    """The T12 repro: every answer is charged, the client re-requests without acting on it."""

    def test_opencode_responses_null_usage_loop_is_stopped(self):
        server = self.server(NullUsageResponses, default=text_step("NULL-USAGE"))
        proxy = self.proxy(server, wire_from_name("openai_responses", client="opencode"))
        headers = self.opencode_headers("openai_responses")
        body = dict(self.opencode["openai_responses"]["first_body"])
        statuses = []
        for _ in range(20):  # the real client went on until the budget (180) was spent
            status, data = self.send(proxy, "/v1/responses", body, headers)
            statuses.append(status)
            if status != 200:
                break
            # What opencode 1.19.5 sent next: the history plus the delivered text, no new input.
            echoed = dict(role="assistant", content=[dict(type="output_text", text=delivered_responses_text(data))])
            body = dict(body, input=body["input"] + [echoed])
        self.assert_consume_block(proxy, server, statuses)
        self.assertEqual(json.loads(data)["error"]["message"], "auriga_client_retry_runaway")
        self.assertEqual(opencode_reaction("openai_responses", status, data), "fail")  # one client attempt
        # Blocked: even a request with new input is refused before forwarding.
        follow_up = dict(body, input=body["input"] + [dict(role="user", content="next")])
        status, data = self.send(proxy, "/v1/responses", follow_up, headers)
        self.assertEqual((status, json.loads(data)["error"]["message"]), (400, "auriga_model_transport_blocked"))
        self.assertEqual(len(server.generation_requests()), K + 1)
        # The operator reviewed the loop: a manual retry starts a fresh count.
        proxy.manual_retry()
        self.assertEqual(proxy._idle_runs, {})
        self.assertEqual(self.send(proxy, "/v1/responses", follow_up, headers)[0], 200)
        self.assertEqual(proxy.request_count, K + 2)

    def test_opencode_anthropic_unchanged_history_loop_is_stopped(self):
        server = self.server(NullUsageAnthropic, default=text_step("NULL-USAGE"))
        proxy = self.proxy(server, wire_from_name("anthropic_messages", client="opencode"))
        headers = self.opencode_headers("anthropic_messages")
        body = dict(self.opencode["anthropic_messages"]["first_body"])
        statuses = []
        for _ in range(20):
            # What opencode 1.19.5 sent after each unparseable answer: the unchanged history (after a backoff).
            status, data = self.send(proxy, "/v1/messages", body, headers)
            statuses.append(status)
            if status != 200:
                break
        self.assert_consume_block(proxy, server, statuses)
        error = json.loads(data)["error"]
        self.assertEqual((error["type"], error["message"]), ("invalid_request_error", "auriga_client_retry_runaway"))
        self.assertEqual(opencode_reaction("anthropic_messages", status, data), "fail")

    def test_claude_code_profile_is_guarded_too(self):
        server = self.server(NullUsageAnthropic, default=text_step("NULL-USAGE"))
        proxy = self.proxy(server, AnthropicMessagesWire(client="claude_code"))
        headers = CLAUDE_HEADERS + ((CLAUDE_THREAD, "s-1"),)
        body = dict(model="fixture-model", max_tokens=1024, stream=True, system="SYSTEM", tools=TOOLS,
                    messages=[dict(role="user", content="hello")])
        statuses = []
        for _ in range(20):
            status, data = self.send(proxy, "/v1/messages?beta=true", body, headers)
            statuses.append(status)
            if status != 200:
                break
            echoed = dict(role="assistant", content=[dict(type="text", text="NULL-USAGE")])
            body = dict(body, messages=body["messages"] + [echoed])
        self.assert_consume_block(proxy, server, statuses)
        error = json.loads(data)["error"]
        self.assertEqual((status, error["type"], error["message"]),
                         (403, "permission_error", "auriga_client_retry_runaway"))

    def test_codex_profile_has_no_consume_guard(self):
        server = self.server(NullUsageResponses, default=text_step("NULL-USAGE"))
        proxy = ModelRequestProxy(server.base_url, max_retries=0).start()
        self.addCleanup(proxy.close)
        self.assertFalse(proxy.wire.client_loop_guard)
        self.assertIsNone(proxy.continuity)
        body = dict(model="fixture-model", stream=True, instructions="SYSTEM", tools=RESPONSES_TOOLS,
                    input=[dict(role="user", content="hello")])
        for _ in range(K + 4):
            status, data = self.send(proxy, "/responses", body, (("Content-Type", "application/json"),))
            self.assertEqual(status, 200)
            echoed = dict(role="assistant", content=[dict(type="output_text", text=delivered_responses_text(data))])
            body = dict(body, input=body["input"] + [echoed])
        self.assertEqual(proxy.request_count, K + 4)
        self.assertFalse(proxy.transport_blocked)
        self.assertEqual((proxy._idle_runs, self.events(proxy, CONSUME_EVENT)), ({}, []))


class ConsumeRuleTest(GuardCase):
    """What counts as idle, what ends a run, and what is never compared."""

    PATH = "/v1/messages"

    def setUp(self):
        super().setUp()
        self.fake = self.server(FakeAnthropicServer, default=text_step("answer"))

    def opencode_proxy(self, **kwargs):
        return self.proxy(self.fake, wire_from_name("anthropic_messages", client="opencode"), **kwargs)

    def call(self, proxy, messages, *, session="ses_main", tools=True, system="SYSTEM"):
        body = dict(model="fixture-model", max_tokens=1024, stream=True, system=system, messages=messages)
        if tools:
            body["tools"] = TOOLS
        return self.send(proxy, self.PATH, body, self.opencode_headers("anthropic_messages", session))[0]

    @staticmethod
    def answered(messages, text="answer"):
        return messages + [dict(role="assistant", content=[dict(type="text", text=text)])]

    def idle_run(self, proxy, messages, count, **kwargs):
        """``count`` idle re-requests (each echoes the previous answer, no new input); returns the history."""
        for _ in range(count):
            messages = self.answered(messages)
            self.assertEqual(self.call(proxy, messages, **kwargs), 200)
        return messages

    def assert_refused(self, proxy, messages, **kwargs):
        self.assertEqual(self.call(proxy, self.answered(messages), **kwargs), 400)
        self.assertEqual(proxy.blocked_error["error_type"], "ClientDidNotConsumeCompletion")

    def test_only_consecutive_idle_completions_count(self):
        proxy = self.opencode_proxy()
        messages = [dict(role="user", content="hello")]
        self.assertEqual(self.call(proxy, messages), 200)
        messages = self.idle_run(proxy, messages, K - 1)
        messages = self.answered(messages) + [dict(role="user", content="a new prompt")]
        self.assertEqual(self.call(proxy, messages), 200)          # client input: the run ends
        messages = self.idle_run(proxy, messages, K)               # K idle completions are tolerated
        self.assertFalse(proxy.transport_blocked)
        self.assert_refused(proxy, messages)                        # the next idle request is not
        self.assertEqual(proxy.request_count, 1 + (K - 1) + 1 + K)

    def test_client_input_of_every_kind_ends_a_run(self):
        hello = [dict(role="user", content=[dict(type="text", text="hello")])]
        call = dict(type="tool_use", id="toolu_1", name="mcp__auriga__look", input={})
        result = dict(type="tool_result", tool_use_id="toolu_1", content="scene: one cube")
        cases = {
            "user_message": lambda m: self.answered(m) + [dict(role="user", content="next")],
            "tool_result": lambda m: m + [dict(role="assistant", content=[call]),
                                          dict(role="user", content=[result])],
            # Claude Code merges a steer into the last user message (blocks appended to a client item).
            "merged_steer": lambda m: m[:-1] + [dict(m[-1], content=m[-1]["content"] + [
                dict(type="text", text="steer")])],
            "compaction": lambda m: [dict(role="user", content="summary of the conversation")],
        }
        for name, add_input in cases.items():
            with self.subTest(case=name):
                proxy = self.opencode_proxy()
                self.assertEqual(self.call(proxy, hello), 200)
                messages = self.idle_run(proxy, hello, K)
                if name == "merged_steer":
                    messages = messages + [dict(role="user", content=[dict(type="text", text="go on")])]
                    self.assertEqual(self.call(proxy, messages), 200)  # a client item: run ends, baseline here
                messages = add_input(messages)
                self.assertEqual(self.call(proxy, messages), 200)
                self.assertFalse(proxy.transport_blocked)
                messages = self.idle_run(proxy, messages, K)
                self.assert_refused(proxy, messages)

    def test_helper_and_subagent_requests_are_not_compared(self):
        proxy = self.opencode_proxy(default_thread_id="ses_main")
        hello = [dict(role="user", content="hello")]
        self.assertEqual(self.call(proxy, hello), 200)
        messages = self.idle_run(proxy, hello, K)
        for _ in range(K + 2):  # helper calls (no tools) with the same history: neither counted nor resetting
            self.assertEqual(self.call(proxy, self.answered(messages), tools=False), 200)
        child = [dict(role="user", content="child task")]  # an OpenCode child session: a subagent chain
        self.assertEqual(self.call(proxy, child, session="ses_child"), 200)
        self.idle_run(proxy, child, K + 2, session="ses_child")
        self.assertFalse(proxy.transport_blocked)
        self.assertEqual(set(proxy._idle_runs), {"ses_main"})
        self.assert_refused(proxy, messages)

    def test_an_undelivered_completion_never_counts(self):
        gate = Gate()
        self.fake = self.server(FakeAnthropicServer, [text_step("answer")] * K + [Step(text="never", gate=gate)],
                                default=text_step("answer"))
        self.addCleanup(gate.release)
        proxy = self.opencode_proxy()
        hello = [dict(role="user", content="hello")]
        self.assertEqual(self.call(proxy, hello), 200)
        messages = self.idle_run(proxy, hello, K - 1)
        held = []
        worker = threading.Thread(target=lambda: held.append(self.call(proxy, self.answered(messages))))
        worker.start()
        self.assertTrue(gate.wait_entered(10))
        proxy.cancel_pending("turn_interrupted")  # the idle request's answer is never delivered
        gate.release()
        worker.join(10)
        self.assertEqual(held, [400])
        self.assertFalse(proxy.transport_blocked)
        self.assertTrue(proxy.reopen_thread("ses_main"))
        self.assertEqual(proxy._idle_runs["ses_main"].idle, K - 1)
        messages = self.idle_run(proxy, messages, 1)  # the K-th idle completion is still tolerated
        self.assert_refused(proxy, messages)

    def test_responses_output_items(self):
        server = self.server(FakeResponsesServer, default=text_step("answer"))
        proxy = self.proxy(server, wire_from_name("openai_responses", client="opencode"))
        headers = self.opencode_headers("openai_responses")
        body = dict(model="fixture-model", stream=True, tools=RESPONSES_TOOLS,
                    input=[dict(role="developer", content="SYSTEM"), dict(role="user", content="hello")])

        def send(items):
            return self.send(proxy, "/v1/responses", dict(body, input=body["input"] + items), headers)[0]

        self.assertEqual(send([]), 200)
        reasoning = dict(type="reasoning", id="rs_1", summary=[], encrypted_content="opaque")
        call = dict(type="function_call", call_id="call_1", name="auriga_look", arguments="{}")
        output = dict(type="function_call_output", call_id="call_1", output="scene: one cube")
        tool_round = [reasoning, call, output]
        self.assertEqual(send(tool_round), 200)  # a tool result is client input
        idle = list(tool_round)
        for _ in range(K):  # model output only: reasoning, message and a call without its result
            idle += [reasoning, dict(role="assistant", content=[dict(type="output_text", text="answer")])]
            self.assertEqual(send(idle), 200)
        self.assertEqual(send(idle + [call]), 400)
        self.assertEqual(proxy.blocked_error["error_type"], "ClientDidNotConsumeCompletion")


class NormalSessionsNeverTripTest(GuardCase):
    """False-positive measurement: captured and synthetic normal sessions never produce an idle completion."""

    PROFILES = {
        ("claude_code", "anthropic_messages"): (FakeAnthropicServer, "/v1/messages?beta=true", CLAUDE_THREAD),
        ("opencode", "anthropic_messages"): (FakeAnthropicServer, "/v1/messages", OPENCODE_THREAD),
        ("opencode", "openai_responses"): (FakeResponsesServer, "/v1/responses", OPENCODE_THREAD),
    }

    def spy(self, proxy):
        """Record the consume-guard verdict of every delivered usable main completion."""
        verdicts = []
        original = proxy._commit_idle_run

        def commit(thread, call_purpose, fingerprints, idle):
            verdicts.append((call_purpose, idle))
            return original(thread, call_purpose, fingerprints, idle)

        patch = mock.patch.object(proxy, "_commit_idle_run", side_effect=commit)
        patch.start()
        self.addCleanup(patch.stop)
        return verdicts

    def test_captured_sessions_replay_without_an_idle_completion(self):
        """Every Claude Code and OpenCode sequence of the continuity fixture on a wire the relay serves: its delivered
        requests (the only ones that ever move the relay's state) in capture order, through the real relay with the
        client's own profile. Failed and aborted rows of the spike are left out, because their answers never reached
        the client; the relay answers every replayed row usably."""
        sequences = [s for s in load_fixture()["sequences"] if (s["client"], s["wire"]) in self.PROFILES]
        verdicts, replayed = [], 0
        for (client, wire_name), (server_cls, path, thread_header) in self.PROFILES.items():
            with self.subTest(client=client, wire=wire_name):
                server = self.server(server_cls)
                proxy = self.proxy(server, wire_from_name(wire_name, client=client), max_requests=4000)
                seen = self.spy(proxy)
                base = CLAUDE_HEADERS if client == "claude_code" else tuple(
                    (k, v) for k, v in self.opencode_headers(wire_name) if k.lower() != OPENCODE_THREAD)
                for index, sequence in enumerate(s for s in sequences if (s["client"], s["wire"]) == (client, wire_name)):
                    if wire_name == "openai_responses":
                        sequence = dict(sequence, values=rehydrate_images(sequence["values"]))
                    rows = [k for k, request in enumerate(sequence["requests"]) if request["delivered"]]
                    statuses = self.replay(proxy, sequence, rows, path=path, thread_header=thread_header,
                                           base_headers=base, session_prefix="seq%d" % index)
                    self.assertLessEqual(set(statuses), {200}, sequence["id"])  # some sequences deliver nothing
                    replayed += len(rows)
                self.assertFalse(proxy.transport_blocked)
                self.assertEqual(self.events(proxy, CONSUME_EVENT), [])
                verdicts += seen
        main = [idle for purpose, idle in verdicts if purpose == "main"]
        self.assertNotIn(True, main)
        # Measured on the checked-in fixture: 433 delivered rows, 420 main completions, 330 of them compared
        # (90 are the first main request of their thread), 0 idle.
        self.assertEqual((replayed, main.count(False), main.count(None)), (433, 330, 90))

    def test_synthetic_tool_loops_never_trip(self):
        """A conformance-style agent loop on every guarded profile: 3 x K tool rounds, a steer as a new user message,
        then a final answer and the next user turn."""
        for (client, wire_name), (server_cls, path, thread_header) in self.PROFILES.items():
            with self.subTest(client=client, wire=wire_name):
                server = self.server(server_cls, default=text_step("answer"))
                proxy = self.proxy(server, wire_from_name(wire_name, client=client))
                seen = self.spy(proxy)
                headers = (CLAUDE_HEADERS if client == "claude_code" else tuple(
                    (k, v) for k, v in self.opencode_headers(wire_name) if k.lower() != OPENCODE_THREAD)
                ) + ((thread_header, "loop"),)
                history = [dict(role="user", content="start")]
                for k in range(3 * K):
                    self.assertEqual(self.send(proxy, path, self.loop_body(wire_name, history), headers)[0], 200)
                    history = history + self.tool_round(wire_name, k)
                    if k == K:
                        history.append(dict(role="user", content="steer: go left"))
                for text in ("final", "next turn"):
                    self.assertEqual(self.send(proxy, path, self.loop_body(wire_name, history), headers)[0], 200)
                    history = history + self.answer(wire_name, text) + [dict(role="user", content=text)]
                self.assertEqual(self.send(proxy, path, self.loop_body(wire_name, history), headers)[0], 200)
                self.assertFalse(proxy.transport_blocked)
                self.assertEqual([idle for _, idle in seen], [None] + [False] * (3 * K + 2))

    @staticmethod
    def loop_body(wire_name, history):
        if wire_name == "openai_responses":
            return dict(model="fixture-model", stream=True, tools=RESPONSES_TOOLS,
                        input=[dict(role="developer", content="SYSTEM")] + history)
        return dict(model="fixture-model", max_tokens=1024, stream=True, system="SYSTEM", tools=TOOLS,
                    messages=history)

    @staticmethod
    def answer(wire_name, text):
        kind = "output_text" if wire_name == "openai_responses" else "text"
        return [dict(role="assistant", content=[dict(type=kind, text=text)])]

    @staticmethod
    def tool_round(wire_name, k):
        if wire_name == "openai_responses":
            return [dict(type="function_call", call_id="call_%d" % k, name="auriga_look", arguments="{}"),
                     dict(type="function_call_output", call_id="call_%d" % k, output="scene %d" % k)]
        call = "toolu_%02d" % k
        return [dict(role="assistant", content=[dict(type="tool_use", id=call, name="mcp__auriga__look", input={})]),
                dict(role="user", content=[dict(type="tool_result", tool_use_id=call, content="scene %d" % k)])]


class ModelOutputFlagsTest(unittest.TestCase):
    """``WireAdapter.history_model_output_flags``: one flag per conversation item, unknown items are client input."""

    def test_responses(self):
        request = dict(instructions="SYSTEM", input=[
            dict(role="developer", content="SYSTEM"), dict(role="user", content="hello"),
            dict(type="reasoning", summary=[]), dict(role="assistant", content="hi"),
            dict(type="function_call", call_id="c", name="n", arguments="{}"),
            dict(type="function_call_output", call_id="c", output="r"),
            dict(type="custom_tool_call", call_id="d", name="m", input="x"),
            dict(type="custom_tool_call_output", call_id="d", output="r"),
            dict(type="unknown_item"), dict(role="system", content="late system text")])
        for client in ("codex", "opencode"):
            wire = OpenAIResponsesWire(client=client)
            flags = wire.history_model_output_flags(request)
            self.assertEqual(flags, (False, True, True, True, False, True, False, False, False))
            self.assertEqual(len(flags), len(wire.history_fingerprints(request)))
        self.assertEqual(OpenAIResponsesWire().history_model_output_flags(dict(input="hello")), (False,))

    def test_anthropic_messages(self):
        request = dict(system="SYSTEM", messages=[
            dict(role="user", content="hello"), dict(role="assistant", content=[dict(type="text", text="hi")]),
            dict(role="system", content="environment"), "not a message"])
        for client in ("claude_code", "opencode"):
            wire = AnthropicMessagesWire(client=client)
            flags = wire.history_model_output_flags(request)
            self.assertEqual(flags, (False, True, False, False))
            self.assertEqual(len(flags), len(wire.history_fingerprints(request)))

    def test_documented_bound(self):
        # The constant's comment and the class docstring cite this value (an unparseable-answer loop costs K + 1 = 4
        # charged completions); change them together.
        self.assertEqual(K, 3)


if __name__ == "__main__":
    unittest.main()
