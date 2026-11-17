"""``turn_verdict`` / ``IncompleteTurnGuard`` against message snapshots captured from the real opencode 1.19.5 binary
(spikes S6-S9; fixtures under ``tests/fixtures/opencode/``, see each file's ``capture``/``evidence``/``sanitised``).

Covers plan 07 section 3.4 rules 1-7 and every verdict row of section 3.9: completion (plain, tool turn, two wires),
steer joining (during model call, during tool, during the final text, two steers), abort during a model call and
during an MCP call, relay denial / provider errors, auto-compaction, restart with a dangling tool, and the
"idle without a terminal answer twice" guard. Intermediate states are either prefixes of a captured snapshot (the
messages that existed at that point, as OpenCode stores them in creation order) or are folded from the captured
``/event`` frames.
"""
import copy
import json
import unittest
from pathlib import Path

from auriga.cortex.agents.opencode.verdict import (INCOMPLETE_ERROR, IncompleteTurnGuard, TurnVerdict, error_code,
                                                   final_text_of, status_type, turn_verdict)

FIXTURES = Path(__file__).parent / "fixtures" / "opencode"
IDLE, BUSY, RETRY = {}, {"type": "busy"}, {"type": "retry", "attempt": 1, "message": "x", "next": 1}


def load(name):
    return json.loads((FIXTURES / name).read_text())


def session(name, key):
    return load(name)["sessions"][key]


def user_id(messages, text):
    """Native id of the (last) user message whose text is ``text``."""
    ids = [m["info"]["id"] for m in messages if m["info"]["role"] == "user"
           and "".join(p.get("text", "") for p in m["parts"] if p["type"] == "text") == text]
    assert ids, text
    return ids[-1]


def until(messages, text):
    """Prefix of ``messages`` up to and including the message whose text parts contain ``text``."""
    for index, message in enumerate(messages):
        if any(text in (part.get("text") or "") for part in message["parts"]):
            return messages[:index + 1]
    raise AssertionError(text)


def fold(events, base=()):
    """Replay ``/event`` frames (``message.updated``, ``message.part.updated``, ``message.part.delta``,
    ``session.status``/``session.idle``) into a ``GET /message``-shaped snapshot and this session's status entry."""
    messages = {m["info"]["id"]: copy.deepcopy(m) for m in base}
    status = {}
    for frame in events:
        kind, props = frame["type"], frame["properties"]
        if kind == "message.updated":
            info = props["info"]
            messages.setdefault(info["id"], dict(info=info, parts=[]))["info"] = copy.deepcopy(info)
        elif kind == "message.part.updated":
            part = copy.deepcopy(props["part"])
            message = messages.setdefault(part["messageID"], dict(info=dict(id=part["messageID"]), parts=[]))
            message["parts"] = sorted([p for p in message["parts"] if p["id"] != part["id"]] + [part],
                                      key=lambda p: p["id"])
        elif kind == "message.part.delta":
            for part in messages.get(props["messageID"], dict(parts=[]))["parts"]:
                if part["id"] == props["partID"]:
                    part[props["field"]] = part.get(props["field"], "") + props["delta"]
        elif kind == "session.status":
            status = {} if props["status"]["type"] == "idle" else dict(props["status"])
        elif kind == "session.idle":
            status = {}
    ordered = sorted(messages.values(), key=lambda m: (m["info"].get("time", {}).get("created", 0), m["info"]["id"]))
    return [m for m in ordered if m["info"].get("role")], status


class Completion(unittest.TestCase):
    def test_tool_turn_and_second_turn_on_both_wires(self):
        for key in ("anthropic_two_turns", "responses_two_turns"):
            with self.subTest(wire=key):
                messages = session("messages_basic.json", key)["messages"]
                turn1 = until(messages, "FINAL-ANSWER-ONE")
                anchor1 = user_id(messages, "USER-TURN-ONE: please work.")
                verdict = turn_verdict(turn1, IDLE, last_user_message_id=anchor1)
                self.assertEqual((verdict.status, verdict.final_text, verdict.error, verdict.reason),
                                 ("completed", "FINAL-ANSWER-ONE", None, "answered"))
                self.assertEqual(verdict.message_id, turn1[-1]["info"]["id"])
                # "Calling fast." shares its message with a tool call: commentary, never the final text
                self.assertNotIn("Calling fast.", verdict.final_text)
                anchor2 = user_id(messages, "USER-TURN-TWO: continue.")
                verdict = turn_verdict(messages, IDLE, last_user_message_id=anchor2)
                self.assertEqual((verdict.status, verdict.final_text), ("completed", "FINAL-ANSWER-TWO"))

    def test_mid_tool_turn_is_running(self):
        messages = session("messages_basic.json", "anthropic_two_turns")["messages"]
        anchor = user_id(messages, "USER-TURN-ONE: please work.")
        prefix = until(messages, "Calling fast.")                     # finish = tool-calls
        self.assertEqual(turn_verdict(prefix, IDLE, last_user_message_id=anchor).reason, "answer_unfinished")
        for status in (BUSY, RETRY, "busy"):
            verdict = turn_verdict(messages, status, last_user_message_id=user_id(messages, "USER-TURN-TWO: continue."))
            self.assertEqual((verdict.status, verdict.reason, verdict.idle), ("running", "busy", False))
        # the user message exists but no answer yet
        self.assertEqual(turn_verdict(messages[:1], IDLE, last_user_message_id=anchor).reason, "awaiting_answer")

    def test_tool_calls_then_final_in_separate_messages(self):
        """S7 F: the /event stream was dropped and reconnected mid-turn (no replay); the verdict only needs the
        snapshot read after the gap."""
        messages = session("messages_steer_abort.json", "F_reconnect")["messages"]
        verdict = turn_verdict(messages, IDLE, last_user_message_id=messages[0]["info"]["id"])
        self.assertEqual((verdict.status, verdict.final_text), ("completed", "RECONNECT-FINAL"))

    def test_native_tool_turn(self):
        messages = session("messages_native_tools.json", "grep_glob_read_bash")["messages"]
        verdict = turn_verdict(messages, IDLE, last_user_message_id=messages[0]["info"]["id"])
        self.assertEqual((verdict.status, verdict.final_text), ("completed", "CONTROL-DONE"))

    def test_prompt_not_visible_yet(self):
        """prompt_async answers 204 before the user message exists: the previous turn's answer must not end the new
        turn (anchor None or unknown id -> user_message_pending)."""
        messages = session("messages_basic.json", "anthropic_two_turns")["messages"]
        turn1 = until(messages, "FINAL-ANSWER-ONE")
        for anchor in (None, "msg_not_in_snapshot"):
            verdict = turn_verdict(turn1, IDLE, last_user_message_id=anchor)
            self.assertEqual((verdict.status, verdict.reason, verdict.idle), ("running", "user_message_pending", True))

    def test_finish_length_with_empty_output_completes_without_text(self):
        messages = session("messages_errors.json", "responses_incomplete_empty")["messages"]
        verdict = turn_verdict(messages, IDLE, last_user_message_id=messages[0]["info"]["id"])
        self.assertEqual((verdict.status, verdict.final_text, verdict.error), ("completed", None, None))


class Steer(unittest.TestCase):
    def test_steer_during_model_and_during_tool(self):
        for key, steer, reply in (("A_steer_during_model", "STEER-DURING-MODEL", "REPLY-AFTER-TOOL"),
                                  ("B_steer_during_tool", "STEER-DURING-TOOL", "REPLY-AFTER-SLOW")):
            with self.subTest(key):
                messages = session("messages_steer_abort.json", key)["messages"]
                prompt, steer_id = messages[0]["info"]["id"], user_id(messages, steer)
                # the session learned the steer id, or still only knows the prompt: the later user message wins
                for anchor in (steer_id, prompt):
                    verdict = turn_verdict(messages, IDLE, last_user_message_id=anchor)
                    self.assertEqual((verdict.status, verdict.final_text), ("completed", reply))
                # tool step done, steer visible, its answer not yet written
                verdict = turn_verdict(until(messages, steer), IDLE, last_user_message_id=prompt)
                self.assertEqual((verdict.status, verdict.reason), ("running", "awaiting_answer"))

    def test_steer_during_final_text_supersedes_the_first_final(self):
        messages = session("messages_steer_abort.json", "C_steer_during_final")["messages"]
        prompt, steer = messages[0]["info"]["id"], user_id(messages, "STEER-DURING-FINAL")
        verdict = turn_verdict(messages, IDLE, last_user_message_id=steer)
        self.assertEqual((verdict.status, verdict.final_text), ("completed", "SECOND-FINAL"))
        # FIRST-FINAL is finished and the steer is visible: wait for the steer's answer
        verdict = turn_verdict(until(messages, "STEER-DURING-FINAL"), IDLE, last_user_message_id=prompt)
        self.assertEqual(verdict.reason, "awaiting_answer")
        # the steer was submitted but its message is not visible: the session passes None
        verdict = turn_verdict(until(messages, "FIRST-FINAL"), IDLE, last_user_message_id=None)
        self.assertEqual(verdict.reason, "user_message_pending")

    def test_two_steers_answered_once(self):
        messages = session("messages_steer_abort.json", "C2_two_steers")["messages"]
        for text in ("STEER-A", "STEER-B"):
            verdict = turn_verdict(messages, IDLE, last_user_message_id=user_id(messages, text))
            self.assertEqual((verdict.status, verdict.final_text), ("completed", "SECOND-FINAL"))

    def test_folded_event_timeline_never_completes_early(self):
        """Every intermediate snapshot folded from the real /event frames of S7 C: running until the final idle,
        then SECOND-FINAL; with the status forced to idle (status/message race) FIRST-FINAL is still never final."""
        capture = load("events_steer_abort.json")["sessions"]["C_steer_during_final"]
        events = capture["events"]
        verdicts = []
        for index in range(len(events) + 1):
            messages, status = fold(events[:index])
            users = [m for m in messages if m["info"]["role"] == "user"]
            anchor = users[-1]["info"]["id"] if users else None
            verdict = turn_verdict(messages, status, last_user_message_id=anchor)
            verdicts.append(verdict)
            forced = turn_verdict(messages, IDLE, last_user_message_id=anchor)
            self.assertNotEqual(forced.final_text, "FIRST-FINAL")
        idle = next(i for i, e in enumerate(events) if e["type"] == "session.status"
                    and e["properties"]["status"]["type"] == "idle")
        self.assertTrue(all(v.status == "running" for v in verdicts[:idle + 1]))     # verdicts[i] = events[:i]
        self.assertTrue(all((v.status, v.final_text) == ("completed", "SECOND-FINAL") for v in verdicts[idle + 1:]))
        self.assertGreaterEqual(len(verdicts) - idle - 1, 2)                         # status idle + session.idle


class Abort(unittest.TestCase):
    def test_abort_during_model_call(self):
        messages = session("messages_steer_abort.json", "D_abort_model+D2_after_abort_model")["messages"]
        aborted, anchor = messages[:2], messages[0]["info"]["id"]
        self.assertEqual(aborted[1]["parts"], [])                       # no parts, finish null, MessageAbortedError
        verdict = turn_verdict(aborted, IDLE, last_user_message_id=anchor)
        self.assertEqual((verdict.status, verdict.reason, verdict.error, verdict.final_text),
                         ("interrupted", "aborted", None, None))
        verdict = turn_verdict(aborted, IDLE, last_user_message_id=anchor, interrupt_requested=True)
        self.assertEqual((verdict.status, verdict.reason), ("interrupted", "interrupt_requested"))
        # the next turn on the same session completes normally
        verdict = turn_verdict(messages, IDLE, last_user_message_id=user_id(messages, "USER-AFTER-ABORT"))
        self.assertEqual((verdict.status, verdict.final_text), ("completed", "AFTER-ABORT-REPLY"))

    def test_abort_during_mcp_call(self):
        messages = session("messages_steer_abort.json", "E_abort_tool+E2_after_abort_tool")["messages"]
        tool = [p for p in messages[1]["parts"] if p["type"] == "tool"][0]
        self.assertEqual((tool["state"]["status"], tool["state"]["error"]), ("error", "Tool execution aborted"))
        verdict = turn_verdict(messages[:2], IDLE, last_user_message_id=messages[0]["info"]["id"])
        self.assertEqual((verdict.status, verdict.reason), ("interrupted", "aborted"))
        # a tool part aborted alone (message error not yet written) is already an interrupt
        partial = copy.deepcopy(messages[:2])
        partial[1]["info"].pop("error")
        self.assertEqual(turn_verdict(partial, IDLE, last_user_message_id=messages[0]["info"]["id"]).status,
                         "interrupted")
        verdict = turn_verdict(messages, IDLE, last_user_message_id=user_id(messages, "USER-AFTER-ABORT-TOOL"))
        self.assertEqual((verdict.status, verdict.final_text), ("completed", "AFTER-ABORT-TOOL-REPLY"))

    def test_folded_abort_timeline(self):
        """Real S7 E frames: the FIRST session.idle after POST /abort arrives BEFORE the tool part error and the
        message error are written, so a reconcile at that idle sees a dangling running tool (answer_unfinished)
        unless the host requested the interrupt; the second idle carries the settled MessageAbortedError."""
        capture = load("events_steer_abort.json")["sessions"]["E_abort_tool+E2_after_abort_tool"]
        events = capture["events"]
        idles = [i for i, e in enumerate(events) if e["type"] == "session.idle"]
        before = [self._verdict(events[:i]) for i in range(idles[0])]
        self.assertTrue(before and all(v.status == "running" for v in before))
        early = self._verdict(events[:idles[0] + 1])
        self.assertEqual((early.status, early.reason, early.idle), ("running", "answer_unfinished", True))
        self.assertEqual(self._verdict(events[:idles[0] + 1], interrupt_requested=True).status, "interrupted")
        settled = self._verdict(events[:idles[1] + 1])
        self.assertEqual((settled.status, settled.reason), ("interrupted", "aborted"))
        # the guard does not fire between the two idles (milliseconds apart) and resets on the terminal verdict
        guard = IncompleteTurnGuard()
        self.assertEqual(guard.observe(early, 5.000).status, "running")
        self.assertEqual(guard.observe(settled, 5.004), settled)

    def test_first_idle_precedes_the_terminal_write(self):
        """Same ordering on the real abort-during-model and provider stream-error timelines: session.error, idle,
        THEN the message error, then a second idle. A reconcile must run again after the second idle."""
        for fixture, key, expected in (
                ("events_steer_abort.json", "D_abort_model+D2_after_abort_model", ("interrupted", None)),
                ("events_errors.json", "stream_error", ("failed", "opencode_turn_failed:UnknownError"))):
            with self.subTest(key):
                events = load(fixture)["sessions"][key]["events"]
                idles = [i for i, e in enumerate(events) if e["type"] == "session.idle"]
                self.assertLess(idles[0] + 1, idles[1])
                early = self._verdict(events[:idles[0] + 1])
                self.assertEqual((early.status, early.reason), ("running", "answer_unfinished"))
                settled = self._verdict(events[:idles[1] + 1])
                self.assertEqual((settled.status, settled.error), expected)
                # retries keep the session non-idle ("retry" status) until the final failure
                retries = [i for i, e in enumerate(events) if e["type"] == "session.status"
                           and e["properties"]["status"]["type"] == "retry"]
                self.assertEqual(bool(retries), key == "stream_error")
                for index in retries:
                    self.assertEqual(self._verdict(events[:index + 1]).reason, "busy")

    @staticmethod
    def _verdict(events, *, interrupt_requested=False):
        messages, status = fold(events)
        users = [m for m in messages if m["info"]["role"] == "user"]
        return turn_verdict(messages, status, last_user_message_id=users[0]["info"]["id"] if users else None,
                            interrupt_requested=interrupt_requested)

    def test_interrupt_requested_wins_once_idle(self):
        """The host interrupt maps every idle outcome to interrupted (also an APIError the relay's cancel caused,
        or an answer that completed just before the abort), and never ends a busy session."""
        cases = [session("messages_errors.json", "deny400")["messages"],
                 session("messages_steer_abort.json", "F_reconnect")["messages"],
                 session("messages_basic.json", "anthropic_two_turns")["messages"][:1]]
        for messages in cases:
            anchor = messages[0]["info"]["id"]
            verdict = turn_verdict(messages, IDLE, last_user_message_id=anchor, interrupt_requested=True)
            self.assertEqual((verdict.status, verdict.final_text, verdict.error), ("interrupted", None, None))
            self.assertEqual(turn_verdict(messages, BUSY, last_user_message_id=anchor,
                                          interrupt_requested=True).status, "running")


class Failure(unittest.TestCase):
    def test_relay_denial_and_provider_errors(self):
        expected = {"deny400": "opencode_turn_failed:APIError:400", "e500_retried": "opencode_turn_failed:APIError:500",
                    "stream_error": "opencode_turn_failed:UnknownError"}
        for key, error in expected.items():
            with self.subTest(key):
                messages = session("messages_errors.json", key)["messages"]
                verdict = turn_verdict(messages, IDLE, last_user_message_id=messages[0]["info"]["id"])
                self.assertEqual((verdict.status, verdict.error, verdict.final_text, verdict.reason),
                                 ("failed", error, None, "error"))
                # no provider text in the error string
                self.assertNotIn("denied", verdict.error)

    def test_runaway_attempts_are_not_answers(self):
        messages = session("messages_errors.json", "truncated_runaway_aborted")["messages"]
        anchor = messages[0]["info"]["id"]
        # each truncated stream became an assistant message with finish "unknown" and the loop went on
        self.assertEqual(turn_verdict(messages[:3], IDLE, last_user_message_id=anchor).reason, "answer_unfinished")
        self.assertEqual(turn_verdict(messages, IDLE, last_user_message_id=anchor).status, "interrupted")

    def test_error_code_forms(self):
        self.assertEqual(error_code({"name": "APIError", "data": {"statusCode": 429}}),
                         "opencode_turn_failed:APIError:429")
        self.assertEqual(error_code({"name": "ContextOverflowError", "data": {"message": "x"}}),
                         "opencode_turn_failed:ContextOverflowError")
        self.assertEqual(error_code({}), "opencode_turn_failed:unknown")
        self.assertEqual(error_code({"name": "APIError", "data": {"statusCode": True}}),
                         "opencode_turn_failed:APIError")


class Compaction(unittest.TestCase):
    def setUp(self):
        self.messages = session("messages_compaction.json", "three_turns_compacted_in_turn_3")["messages"]
        self.anchor = user_id(self.messages, "USER-TURN-3")
        self.trigger = next(i for i, m in enumerate(self.messages)
                            if any(p["type"] == "compaction" for p in m["parts"]))

    def test_turn_completes_after_auto_compaction(self):
        verdict = turn_verdict(self.messages, IDLE, last_user_message_id=self.anchor)
        self.assertEqual((verdict.status, verdict.final_text), ("completed", "TURN-3-DONE"))
        # the answer replies to the synthetic "Continue ..." user message written after the summary
        continue_message = self.messages[self.trigger + 2]
        self.assertTrue(continue_message["parts"][0]["synthetic"])
        self.assertEqual(self.messages[-1]["info"]["parentID"], continue_message["info"]["id"])

    def test_intermediate_states_are_running(self):
        trigger, summary, cont = (self.messages[:self.trigger + 1], self.messages[:self.trigger + 2],
                                  self.messages[:self.trigger + 3])
        self.assertEqual(turn_verdict(trigger, IDLE, last_user_message_id=self.anchor).reason, "awaiting_answer")
        self.assertTrue(summary[-1]["info"]["summary"])
        verdict = turn_verdict(summary, IDLE, last_user_message_id=self.anchor)
        self.assertEqual((verdict.status, verdict.reason), ("running", "compaction_summary"))
        self.assertEqual(turn_verdict(cont, IDLE, last_user_message_id=self.anchor).reason, "awaiting_answer")
        self.assertEqual(turn_verdict(self.messages[:self.trigger + 4], IDLE,
                                      last_user_message_id=self.anchor).reason, "answer_unfinished")

    def test_earlier_turns(self):
        for prompt, final in (("USER-TURN-1", "TURN-1-DONE"), ("USER-TURN-2", "TURN-2-DONE")):
            verdict = turn_verdict(until(self.messages, final), IDLE,
                                   last_user_message_id=user_id(self.messages, prompt))
            self.assertEqual((verdict.status, verdict.final_text), ("completed", final))


class RestartAndGuard(unittest.TestCase):
    def test_restart_with_dangling_tool(self):
        capture = load("messages_restart.json")["sessions"]
        crashed, final = capture["after_crash"]["messages"], capture["final"]["messages"]
        self.assertEqual(capture["after_crash"]["status_after"], {})   # idle after the restart
        dangling = [p for p in crashed[-1]["parts"] if p["type"] == "tool"][0]
        self.assertEqual(dangling["state"]["status"], "running")
        verdict = turn_verdict(crashed, capture["after_crash"]["status_after"],
                               last_user_message_id=user_id(crashed, "TURN-2"))
        self.assertEqual((verdict.status, verdict.reason, verdict.idle), ("running", "answer_unfinished", True))
        guard = IncompleteTurnGuard()
        self.assertEqual(guard.observe(verdict, 100.0), verdict)
        self.assertEqual(guard.observe(verdict, 101.9), verdict)         # < 2 s apart: keep waiting
        failed = guard.observe(verdict, 102.0)
        self.assertEqual((failed.status, failed.error, failed.reason), ("failed", INCOMPLETE_ERROR, "incomplete"))
        # the next turn after the restart completes; the dangling message is superseded, not the answer
        verdict = turn_verdict(final, IDLE, last_user_message_id=user_id(final, "TURN-3-AFTER-CRASH"))
        self.assertEqual((verdict.status, verdict.final_text), ("completed", "AFTER-CRASH-REPLY"))

    def test_server_killed_during_native_bash(self):
        messages = session("messages_native_tools.json", "server_killed_during_bash")["messages"]
        verdict = turn_verdict(messages, IDLE, last_user_message_id=messages[0]["info"]["id"])
        self.assertEqual(verdict.reason, "answer_unfinished")
        aborted = session("messages_native_tools.json", "abort_during_bash")["messages"]
        self.assertEqual(turn_verdict(aborted, IDLE, last_user_message_id=aborted[0]["info"]["id"]).status,
                         "interrupted")

    def test_guard_prompt_never_materialised_busy_resets_and_interrupt(self):
        messages = session("messages_basic.json", "anthropic_two_turns")["messages"]
        pending = turn_verdict(messages, IDLE, last_user_message_id=None)
        guard = IncompleteTurnGuard(min_gap_s=2.0)
        guard.observe(pending, 0.0)
        busy = turn_verdict(messages, BUSY, last_user_message_id=None)
        self.assertEqual(guard.observe(busy, 1.0), busy)                 # busy resets the guard
        self.assertEqual(guard.observe(pending, 2.5).status, "running")  # first idle observation again
        self.assertEqual(guard.observe(pending, 4.5).error, INCOMPLETE_ERROR)
        guard.reset()
        guard.observe(pending, 10.0)
        interrupted = guard.observe(pending, 13.0, interrupt_requested=True)
        self.assertEqual((interrupted.status, interrupted.reason), ("interrupted", "interrupt_requested"))
        # a terminal verdict passes through unchanged and resets
        done = turn_verdict(messages, IDLE, last_user_message_id=user_id(messages, "USER-TURN-TWO: continue."))
        self.assertIs(guard.observe(done, 20.0), done)
        self.assertIsNone(guard.first_idle_at)
        with self.assertRaises(ValueError):
            IncompleteTurnGuard(min_gap_s=-1)


class Inputs(unittest.TestCase):
    def test_status_forms(self):
        sid = "ses_x"
        self.assertEqual(status_type(None), "idle")
        self.assertEqual(status_type({}), "idle")
        self.assertEqual(status_type({"type": "busy"}), "busy")
        self.assertEqual(status_type("retry"), "retry")
        self.assertEqual(status_type({sid: {"type": "busy"}}, sid), "busy")
        self.assertEqual(status_type({"ses_other": {"type": "busy"}}, sid), "idle")
        with self.assertRaises(TypeError):
            status_type(3)
        # the whole GET /session/status map works too (looked up by the messages' sessionID)
        messages = session("messages_basic.json", "anthropic_two_turns")["messages"]
        sid = messages[0]["info"]["sessionID"]
        anchor = user_id(messages, "USER-TURN-TWO: continue.")
        self.assertEqual(turn_verdict(messages, {sid: {"type": "busy"}}, last_user_message_id=anchor).reason, "busy")
        self.assertEqual(turn_verdict(messages, {"ses_other": {"type": "busy"}},
                                      last_user_message_id=anchor).status, "completed")

    def test_pure_and_tolerant(self):
        messages = session("messages_steer_abort.json", "C_steer_during_final")["messages"]
        before = copy.deepcopy(messages)
        turn_verdict(messages, IDLE, last_user_message_id=messages[0]["info"]["id"])
        self.assertEqual(messages, before)
        self.assertEqual(turn_verdict([], IDLE, last_user_message_id=None).reason, "user_message_pending")
        self.assertEqual(turn_verdict([None, {"x": 1}], IDLE, last_user_message_id="m").reason, "user_message_pending")
        self.assertIsInstance(turn_verdict(messages, IDLE, last_user_message_id=None), TurnVerdict)

    def test_final_text_excludes_synthetic_and_empty_parts(self):
        message = dict(info=dict(id="m"), parts=[dict(type="text", text="A"), dict(type="text", text="  "),
                                                 dict(type="text", text="S", synthetic=True),
                                                 dict(type="text", text="I", ignored=True),
                                                 dict(type="reasoning", text="R"), dict(type="text", text="B")])
        self.assertEqual(final_text_of(message), "A\nB")
        self.assertIsNone(final_text_of(dict(info={}, parts=[])))

    def test_pending_tool_in_stop_message_keeps_looping(self):
        """OpenCode continues when a 'stop' message still carries tool calls (prompt.ts hasToolCalls); an interrupted
        orphan (state error + metadata.interrupted) does not count."""
        messages = copy.deepcopy(session("messages_steer_abort.json", "F_reconnect")["messages"])
        tool = copy.deepcopy([p for p in messages[1]["parts"] if p["type"] == "tool"][0])
        messages[-1]["parts"].insert(1, tool)
        anchor = messages[0]["info"]["id"]
        self.assertEqual(turn_verdict(messages, IDLE, last_user_message_id=anchor).reason, "answer_unfinished")
        tool["state"] = dict(status="error", input={}, error="interrupted", metadata=dict(interrupted=True),
                             time=dict(start=1, end=2))
        self.assertEqual(turn_verdict(messages, IDLE, last_user_message_id=anchor).status, "completed")


if __name__ == "__main__":
    unittest.main()
