import tempfile
import unittest

from auriga.core.storage.store import Store
from auriga.cortex.agents.transcript import TranscriptWriter
from auriga.cortex.session.usage import COUNTING_EQUIVALENCE, TOKEN_FIELDS, account_usage

IDENTITY = dict(run_id="check", episode_id="ep", env_instance_id="env", env_generation=1)


def usage(**kw):
    return {key: kw.get(key, 1) for key in TOKEN_FIELDS}


class TranscriptWriterCheck(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = Store(self.dir.name, IDENTITY)
        self.writer = TranscriptWriter(self.store, backend="claude_code", thread_id="thr")

    def tearDown(self):
        self.store.close()
        self.dir.cleanup()

    def finals(self, turn):
        return self.store.completed_final_texts("thr", turn)

    def test_rows_round_trip_through_readers(self):
        w = self.writer
        turn = w.turn_started()
        w.user_message("do it", item_id="u1")
        w.reasoning("thinking", message_id="m1", item_id="r1")
        w.tool_started("call-1", "observe", {"a": 1}, message_id="m1")
        w.tool_completed("call-1", success=True, content_items=[dict(type="inputText", text="ok")], raw_ref="art:1")
        w.command("c1", "ls", aggregated_output="x", exit_code=0, message_id="m1")
        w.file_change("f1", [dict(path="a", kind="update", diff="d")])
        w.native_tool("n1", "Read", dict(path="a"), result="body")
        w.compaction(item_id="k1")
        w.error("boom", item_id="e1")
        w.assistant_text("all done", message_id="m2", item_id="a1", native_id="nat")
        w.turn_completed("completed")
        row = self.store.transcript_item("call-1")
        self.assertEqual((row["type"], row["tool"], row["success"], row["status"], row["backend"], row["turn_id"]),
                         ("dynamicToolCall", "observe", True, "completed", "claude_code", turn))
        self.assertEqual(row["arguments"], {"a": 1})
        self.assertEqual(row["raw_refs"], ["art:1"])
        self.assertIsInstance(row["durationMs"], int)
        self.assertEqual(self.store.transcript_item("a1")["native_id"], "nat")
        self.assertEqual(self.store.transcript_item("u1")["content"], [dict(type="text", text="do it")])
        self.assertEqual(self.store.transcript_item("r1")["summary"], ["thinking"])
        self.assertEqual(self.store.transcript_item("e1")["error"], dict(message="boom"))
        self.assertEqual(self.finals(turn), ["all done"])
        page = self.store.review_page()
        self.assertEqual([i["id"] for i in page["items"]],
                         ["u1", "r1", "call-1", "c1", "f1", "n1", "k1", "e1", "a1"])
        self.assertTrue(any(e["type"] == "agent.turn_completed" for e in self.store.events()))
        linked = [e for e in page["events"] if e["correlation"].get("turn_id") == turn]
        self.assertEqual(linked, [])  # review_page links events by item/tool-call id only

    def test_tool_failure_and_unknown_status(self):
        self.writer.turn_started()
        self.writer.tool_started("c", "t", {})
        self.writer.tool_completed("c", success=False)
        self.assertEqual(self.store.transcript_item("c")["status"], "failed")
        self.assertEqual(self.store.transcript_item("c")["tool"], "t")

    def test_final_answer_last_text_message_wins_commentary_before(self):
        w = self.writer
        turn = w.turn_started()
        w.assistant_text("first final?", message_id="m1", item_id="a1")
        w.assistant_text("plan", message_id="m2", item_id="a2")
        w.tool_started("t1", "observe", {}, message_id="m2")
        w.tool_completed("t1", success=True)
        w.assistant_text("part one", message_id="m3", item_id="a3")
        w.assistant_text("part two", message_id="m3", item_id="a4")
        self.assertEqual(self.finals(turn), [])  # nothing stamped before the terminal
        w.turn_completed("completed")
        phases = {i: self.store.transcript_item(i)["phase"] for i in ("a1", "a2", "a3", "a4")}
        self.assertEqual(phases, dict(a1="commentary", a2="commentary", a3="final_answer", a4="final_answer"))
        self.assertEqual(self.finals(turn), ["part one", "part two"])
        w.turn_completed("completed")  # idempotent
        self.assertEqual(self.finals(turn), ["part one", "part two"])

    def test_tool_only_last_message_does_not_become_final(self):
        w = self.writer
        turn = w.turn_started()
        w.assistant_text("summary", message_id="m1", item_id="a1")
        w.assistant_text("calling", message_id="m2", item_id="a2")
        w.tool_started("t1", "observe", {}, message_id="m2")
        w.tool_completed("t1", success=True)
        w.tool_started("t2", "observe", {}, message_id="m3")  # last message is tool-only
        w.turn_completed("completed")
        self.assertEqual(self.store.transcript_item("a2")["phase"], "commentary")
        self.assertEqual(self.finals(turn), ["summary"])

    def test_interrupted_and_failed_turns_have_no_final(self):
        for status in ("interrupted", "failed"):
            turn = self.writer.turn_started()
            self.writer.assistant_text("partial", message_id="m" + status, item_id="a" + status)
            self.writer.turn_completed(status, error="x")
            self.assertEqual(self.finals(turn), [])
            self.assertIsNone(self.store.transcript_item("a" + status)["phase"])
        with self.assertRaises(ValueError):
            self.writer.turn_completed("done")

    def test_turns_are_isolated_and_subagent_rows_excluded(self):
        w = self.writer
        t1 = w.turn_started()
        w.assistant_text("one", message_id="m1", item_id="a1")
        w.turn_completed("completed")
        t2 = w.turn_started()
        w.assistant_text("two", message_id="m2", item_id="a2")
        w.assistant_text("child says", message_id="cm", item_id="child", parent_item_id="task-1")
        w.turn_completed("completed")
        self.assertEqual(self.finals(t1), ["one"])
        self.assertEqual(self.finals(t2), ["two"])
        child = self.store.transcript_item("child")
        self.assertEqual(child["parent_item_id"], "task-1")
        self.assertIsNone(child["phase"])
        # a subagent-only tail must not displace the actor's final
        w2 = self.writer
        t3 = w2.turn_started()
        w2.assistant_text("actor", message_id="m3", item_id="a3")
        w2.assistant_text("later child", message_id="cm2", item_id="child2", parent_item_id="task-1")
        w2.turn_completed("completed")
        self.assertEqual(self.finals(t3), ["actor"])

    def test_streaming_update_keeps_stamped_phase_and_row_position(self):
        w = self.writer
        turn = w.turn_started()
        w.assistant_text("he", message_id="m", item_id="a", status="inProgress")
        w.assistant_text("hello", message_id="m", item_id="a")
        w.turn_completed("completed")
        w.assistant_text("hello", message_id="m", item_id="a")  # late replay must not drop the stamp
        self.assertEqual(self.finals(turn), ["hello"])

    def test_steer_row_event_and_permission_denial(self):
        w = self.writer
        turn = w.turn_started()
        w.steer("look left", item_id="s1")
        row = self.store.transcript_item("s1")
        self.assertEqual((row["type"], row["steer"], row["turn_id"]), ("userMessage", True, turn))
        w.permission_denied("Bash", tool_use_id="tu1", reason="not allowed")
        events = {e["type"]: e for e in self.store.events() if e["type"].startswith("agent.")}
        self.assertEqual(events["agent.turn_steered"]["correlation"]["item_id"], "s1")
        denial = events["agent.permission_denied"]
        self.assertEqual((denial["source"], denial["payload"]["tool"], denial["correlation"]["tool_call_id"],
                          denial["correlation"]["thread_id"]), ("agent_backend", "Bash", "tu1", "thr"))
        with self.assertRaises(ValueError):
            w.event("codex.turn")

    def test_usage_sum_and_unknown_is_sticky(self):
        w = self.writer
        w.turn_started()
        w.add_usage(usage(input_tokens=10, output_tokens=3, total_tokens=13), model_context_window=1000)
        record = w.add_usage(usage(input_tokens=5, output_tokens=2, total_tokens=7, cache_write_input_tokens=None,
                                   reasoning_output_tokens=None))
        self.assertEqual(record["total"]["input_tokens"], 15)
        self.assertEqual(record["total"]["total_tokens"], 20)
        self.assertIsNone(record["total"]["cache_write_input_tokens"])
        self.assertIsNone(record["total"]["reasoning_output_tokens"])
        self.assertEqual(record["unknown_calls"], 1)
        self.assertIsNone(record["raw_ref"])
        self.assertEqual(record["last"]["input_tokens"], 5)
        record = w.add_usage(usage())  # later known counters never resurrect an unknown total
        self.assertIsNone(record["total"]["cache_write_input_tokens"])
        self.assertEqual(self.store.read_state("runtime", "token_usage"), record)
        relay = self.store.read_state("runtime", "model_usage_relay")
        self.assertEqual((relay["calls"], relay["total"]), (3, record["total"]))
        accounting = w.write_accounting(source="relay.anthropic_messages.usage.sum", model="m", provider="p")
        self.assertEqual(accounting["source"], "relay.anthropic_messages.usage.sum")
        self.assertIsNone(accounting["tokens"]["cache_write_input_tokens"])
        self.assertEqual(accounting["tokens"]["input_tokens"], 16)

    def test_usage_resumes_from_store_for_same_thread_only(self):
        self.writer.add_usage(usage(input_tokens=4))
        again = TranscriptWriter(self.store, backend="claude_code", thread_id="thr")
        self.assertEqual(again.add_usage(usage(input_tokens=4))["total"]["input_tokens"], 8)
        other = TranscriptWriter(self.store, backend="claude_code", thread_id="other")
        self.assertEqual(other.add_usage(usage(input_tokens=4))["total"]["input_tokens"], 4)


class UsageAccountingCheck(unittest.TestCase):
    def test_account_usage_default_source_unchanged_and_override(self):
        snapshot = dict(total={key: 1 for key in TOKEN_FIELDS}, thread_id="t")
        self.assertEqual(account_usage(snapshot)["source"], "codex.thread/tokenUsage/updated.total")
        self.assertEqual(account_usage(snapshot, source="relay.x.usage.sum")["source"], "relay.x.usage.sum")

    def test_counting_equivalence(self):
        self.assertEqual(set(COUNTING_EQUIVALENCE.values()), {"all_delivered_usable_generation_completions"})
        self.assertIn("delivered_usable_actor_helper_compact_responses_v2", COUNTING_EQUIVALENCE)
        self.assertIn("delivered_usable_generation_completions_v1", COUNTING_EQUIVALENCE)


if __name__ == "__main__":
    unittest.main()
