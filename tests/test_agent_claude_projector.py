"""``ClaudeProjector``: Claude Agent SDK messages -> neutral transcript rows (real captured streams from spikes S1-S5)."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from auriga.core.storage.store import Store
from auriga.cortex.agents.claude_code.projector import ClaudeProjector, classify_result
from auriga.cortex.agents.transcript import TranscriptWriter
from support.fake_claude_sdk import (assistant_message, echo_message, init_message, load_capture, requires_sdk,
                                     result_message, state_message, system_message, tool_result_message, tool_use,
                                     usage_dict)

FIXTURES = Path(__file__).parent / "fixtures" / "claude_sdk"
IDENTITY = dict(run_id="check", episode_id="ep", env_instance_id="env", env_generation=1)


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = Store(self.dir.name, IDENTITY)
        self.writer = TranscriptWriter(self.store, backend="claude_code", thread_id="thr")
        self.projector = ClaudeProjector(self.writer)
        self.turn = self.writer.turn_started()

    def tearDown(self):
        self.store.close()
        self.dir.cleanup()

    def rows(self):
        return {row["id"]: row for row in self.store.states(self.store.TRANSCRIPT_KIND)}

    def of_type(self, kind):
        return [row for row in self.rows().values() if row["type"] == kind]

    def events(self, kind):
        return [event for event in self.store.events() if event["type"] == kind]

    def replay(self, name):
        return [self.projector.project(message) for message in load_capture(FIXTURES / name)]


@requires_sdk
class CapturedStreams(Base):
    def test_compaction_stream_native_rows_and_no_registry_rows(self):
        results = self.replay("s5-compact.jsonl")
        rows = self.rows()
        # registry tools (mcp__auriga__*) belong to the MCP transport: no row from the projector
        registry = {call_id for call_id, record in self.projector.tools.items() if record["registry"]}
        self.assertTrue(registry)
        self.assertEqual(registry & set(rows), set())
        # native Bash: commandExecution, one row per tool_use id, completed with output; exit code unknown (null)
        bash = [row for row in rows.values() if row["type"] == "commandExecution"]
        self.assertTrue(bash)
        self.assertEqual((bash[0]["command"], bash[0]["aggregatedOutput"], bash[0]["status"], bash[0]["exitCode"]),
                         ("echo step2", "step2", "completed", None))
        self.assertEqual(bash[0]["native_id"], bash[0]["id"])
        # thinking rows keep the text, never the signature
        thinking = self.of_type("reasoning")
        self.assertTrue(thinking)
        self.assertTrue(all(set(row["summary"]) and "signature" not in str(row) for row in thinking))
        self.assertEqual(thinking[0]["summary"], ["thinking about step 0"])
        # two real compactions -> two contextCompaction rows + two 'compaction' projections
        self.assertEqual(len(self.of_type("contextCompaction")), 2)
        compaction = [item for item in results if item.kind == "compaction"]
        self.assertEqual([(c.data["trigger"], c.data["pre_tokens"]) for c in compaction],
                         [("auto", 75276), ("auto", 73222)])
        self.assertEqual(sum(item.kind == "compacting" for item in results), 2)
        # usage: kept once per message id (two blocks of one API response share it)
        shared = [m for m in load_capture(FIXTURES / "s5-compact.jsonl") if getattr(m, "message_id", None) == "msg_fake_3"]
        self.assertEqual(len(shared), 3)
        self.assertEqual(len([k for k in self.projector.message_usage if k == "msg_fake_3"]), 1)
        total = self.projector.native_usage_total()
        self.assertEqual(total["input_tokens"], 2000 * len(self.projector.message_usage))
        # message ids on the rows let the final-answer rule see tool calls: text of msg_fake_3 has tool calls
        text = [row for row in rows.values() if row["type"] == "agentMessage" and row.get("message_id") == "msg_fake_3"]
        self.assertEqual(len(text), 1)
        # image items of tool results are not stored (native tool rows)
        self.assertNotIn("base64", str(rows))
        self.assertEqual(results[0].kind, "init")

    def test_final_answer_is_last_toolless_message_after_registry_row_linking(self):
        # the transport wrote the dynamicToolCall row first (with native id), the assistant message arrives later
        self.writer.tool_started("call_fake_0_0", "fast_text", {"text": "s0"}, native_id="call_fake_0_0")
        self.writer.tool_completed("call_fake_0_0", success=True, content_items=[dict(type="inputText", text="x")])
        for message in load_capture(FIXTURES / "s5-compact.jsonl")[:6]:
            self.projector.project(message)
        row = self.rows()["call_fake_0_0"]
        self.assertEqual((row["type"], row["status"], row["success"], row["native_id"]),
                         ("dynamicToolCall", "completed", True, "call_fake_0_0"))
        self.assertEqual(row["message_id"], "msg_fake_0")
        self.assertEqual(len([r for r in self.rows().values() if r["id"] == "call_fake_0_0"]), 1)
        self.writer.turn_completed("completed")
        # text rows of messages that also called tools are commentary
        finals = [r for r in self.of_type("agentMessage") if r.get("phase") == "final_answer"]
        self.assertTrue(all(not r["message_id"] or r["message_id"] != "msg_fake_3" for r in finals))

    def test_registry_row_written_after_assistant_message_is_linked_by_the_result(self):
        self.projector.project(assistant_message("plan", tool_use("c1", "mcp__auriga__robot_observe", k=1),
                                                 message_id="m1"))
        self.assertNotIn("c1", self.rows())             # no second row, nothing written for the registry call
        self.writer.tool_started("c1", "robot_observe", {"k": 1}, native_id="c1")
        self.writer.tool_completed("c1", success=True, content_items=[])
        self.projector.project(tool_result_message("c1", [dict(type="text", text="ok")]))
        self.assertEqual((self.rows()["c1"]["message_id"], self.rows()["c1"]["status"]), ("m1", "completed"))
        self.writer.turn_completed("completed")
        self.assertEqual(self.store.completed_final_texts("thr", self.turn), [])   # "plan" shares m1 with a tool call

    def test_refused_registry_call_becomes_one_failed_row(self):
        self.projector.project(assistant_message(tool_use("c9", "mcp__auriga__nope"), message_id="m9"))
        self.projector.project(tool_result_message("c9", "<tool_use_error>No such tool</tool_use_error>", is_error=True))
        row = self.rows()["c9"]
        self.assertEqual((row["type"], row["status"], row["tool"]), ("mcpToolCall", "failed", "mcp__auriga__nope"))
        self.assertIn("No such tool", row["error"]["message"])

    def test_permission_denials_are_recorded_once_per_tool_use(self):
        results = self.replay("s1-dontask-canuse.jsonl")
        denied = self.events("agent.permission_denied")
        self.assertEqual([(e["payload"]["tool"], e["payload"]["tool_use_id"]) for e in denied],
                         [("Bash", "call_fake_0_0"), ("Write", "call_fake_1_0")])   # system frames; result dedupes
        self.assertEqual(denied[0]["payload"]["reason"], "mode")
        rows = self.rows()
        self.assertEqual((rows["call_fake_0_0"]["type"], rows["call_fake_0_0"]["status"]), ("commandExecution", "failed"))
        self.assertEqual((rows["call_fake_1_0"]["type"], rows["call_fake_1_0"]["status"]), ("fileChange", "failed"))
        self.assertEqual(rows["call_fake_1_0"]["changes"][0]["kind"], "write")
        final = [item for item in results if item.kind == "result"][0]
        self.assertEqual((final.data["status"], final.data["denials"]), ("completed", []))

    def test_result_denials_without_system_frame_are_recorded(self):
        message = result_message(permission_denials=[dict(tool_name="Edit", tool_use_id="e1", tool_input={})])
        projected = self.projector.project(message)
        self.assertEqual(projected.data["denials"], ["e1"])
        self.assertEqual(len(self.events("agent.permission_denied")), 1)
        self.projector.project(message)
        self.assertEqual(len(self.events("agent.permission_denied")), 1)

    def test_steer_replay_marks_pending_rows_consumed(self):
        prompt = self.writer.user_message("Start the scripted task.", item_id="p1")
        steer = self.writer.user_message("STEER-MESSAGE-XYZ please also report the time", item_id="s1", steer=True)
        self.projector.expect_echo("Start the scripted task.", prompt)
        self.projector.expect_echo("STEER-MESSAGE-XYZ please also report the time", steer, steer=True)
        results = self.replay("s3-during-tool-state-replay.jsonl")
        self.assertEqual([item.kind for item in results],
                         ["state", "init", "prompt_echo", "assistant", "tool_result", "steer_echo", "assistant",
                          "result", "state"])
        self.assertEqual(results[0].data["state"], "running")
        self.assertEqual(results[-1].data["state"], "idle")
        rows = self.rows()
        self.assertTrue(rows["s1"]["steer"] and rows["s1"].get("native_id"))
        self.assertTrue(rows["p1"].get("native_id") and not rows["p1"].get("steer"))
        self.assertEqual(len(self.events("agent.turn_steered")), 1)
        self.assertEqual(len(self.of_type("userMessage")), 2)          # echoes did not create second rows
        self.assertEqual(self.projector._pending_echo, [])
        # the native registry tool_use of that stream (mcp__auriga__slow) produced no row
        self.assertNotIn("call_fake_0_0", rows)

    def test_unexpected_user_text_is_not_a_row(self):
        projected = self.projector.project(echo_message("something injected by the CLI"))
        self.assertEqual(projected.kind, "user")
        self.assertEqual(self.of_type("userMessage"), [])

    def test_steer_that_became_its_own_turn(self):
        self.projector.expect_echo("Start the scripted task.", "p1")
        self.writer.user_message("Start the scripted task.", item_id="p1")
        self.writer.user_message("STEER-MESSAGE-XYZ please also report the time", item_id="s1", steer=True)
        self.projector.expect_echo("STEER-MESSAGE-XYZ please also report the time", "s1", steer=True)
        results = self.replay("s3-final-gated-state-replay.jsonl")
        self.assertEqual([item.kind for item in results if item.kind in ("result", "state", "steer_echo")],
                         ["state", "result", "steer_echo", "result", "state"])
        self.assertEqual(len(self.events("agent.turn_steered")), 1)

    def test_interrupt_stream_classification_and_no_error_row(self):
        results = [item for item in self.replay("s3-interrupt-tool.jsonl") if item.kind == "result"]
        self.assertEqual([(r.data["status"], r.data["error"], r.data["terminal_reason"]) for r in results],
                         [("interrupted", None, "aborted_tools"), ("completed", None, "completed")])
        self.assertEqual(self.of_type("error"), [])

    def test_api_error_streams_single_error_row_and_failed_classification(self):
        for name, status in (("s5-retry-403.jsonl", 403), ("s5-retry-500-midstream.jsonl", 529)):
            with self.subTest(name):
                self.store.close()
                self.dir.cleanup()
                self.setUp()
                results = self.replay(name)
                final = [item for item in results if item.kind == "result"][-1]
                self.assertEqual((final.data["status"], final.data["error"]),
                                 ("failed", f"claude_code_turn_failed:api_error:{status}"))
                errors = self.of_type("error")
                self.assertEqual(len(errors), 1)                               # assistant error, not duplicated
                self.assertIn("API Error", errors[0]["error"]["message"])
                self.assertEqual([r for r in self.of_type("agentMessage")], [])  # never a final answer
                self.writer.turn_completed("failed", error=final.data["error"])

    def test_t0_turn_ends_one_result_then_idle(self):
        """Probe T0 captures: interrupts and API errors give exactly one ResultMessage per turn, then idle."""
        expected = {
            "t0-err-403.jsonl": ("failed", "claude_code_turn_failed:api_error:403"),
            "t0-err-529.jsonl": ("failed", "claude_code_turn_failed:api_error:529"),
            "t0-err-truncated.jsonl": ("failed", "claude_code_turn_failed:api_error"),
            "t0-int-model-gated.jsonl": ("interrupted", None),
            "t0-int-model-streaming.jsonl": ("interrupted", None),
            "t0-int-tool.jsonl": ("interrupted", None),
        }
        for name, (status, error) in expected.items():
            with self.subTest(name):
                self.store.close()
                self.dir.cleanup()
                self.setUp()
                kinds = [(item.kind, item.data) for item in self.replay(name)]
                turns, current = [], []
                for kind, data in kinds:
                    current.append((kind, data))
                    if kind == "state" and data["state"] == "idle":
                        turns.append(current)
                        current = []
                self.assertEqual(current, [])
                self.assertEqual(len(turns), 2)                       # the failing turn and the follow-up turn
                for turn in turns:
                    results = [data for kind, data in turn if kind == "result"]
                    self.assertEqual(len(results), 1)
                    self.assertEqual([kind for kind, _ in turn][-2:], ["result", "state"])
                first = [data for kind, data in turns[0] if kind == "result"][0]
                self.assertEqual((first["status"], first["error"]), (status, error))
                self.assertEqual([data for kind, data in turns[1] if kind == "result"][0]["status"], "completed")
                self.assertEqual(len(self.of_type("error")), 1 if status == "failed" else 0)


@requires_sdk
class SyntheticMessages(Base):
    def test_init_and_state_projection(self):
        projected = self.projector.project(init_message("sid", tools=["Bash", "mcp__auriga__x"], model="mm"))
        self.assertEqual((projected.kind, projected.data["session_id"], projected.data["model"],
                          projected.data["tools"], projected.data["permission_mode"], projected.data["cli_version"]),
                         ("init", "sid", "mm", ["Bash", "mcp__auriga__x"], "dontAsk", "2.1.321"))
        self.assertEqual(self.projector.init["model"], "mm")
        self.assertEqual(self.projector.project(state_message("requires_action")).data, {"state": "requires_action"})
        self.assertEqual(self.projector.project(system_message("api_retry", attempt=1)).kind, "other")
        self.assertEqual(self.rows(), {})

    def test_native_tools_map_to_row_types(self):
        calls = [("Bash", dict(command="ls -la")), ("Edit", dict(file_path="/a.txt", old_string="a\nb", new_string="c")),
                 ("Write", dict(file_path="/b.txt", content="x")), ("NotebookEdit", dict(notebook_path="/n.ipynb")),
                 ("Read", dict(file_path="/c.txt")), ("Grep", dict(pattern="x"))]
        for index, (name, arguments) in enumerate(calls):
            self.projector.project(assistant_message(tool_use(f"t{index}", name, **arguments), message_id=f"m{index}"))
        rows = self.rows()
        self.assertEqual([rows[f"t{i}"]["status"] for i in range(6)], ["inProgress"] * 6)
        self.assertEqual([rows[f"t{i}"]["type"] for i in range(6)],
                         ["commandExecution", "fileChange", "fileChange", "fileChange", "mcpToolCall", "mcpToolCall"])
        self.assertEqual(rows["t1"]["changes"], [dict(path="/a.txt", kind="update", diff="-a\n-b\n+c")])
        self.assertEqual(rows["t3"]["changes"][0]["path"], "/n.ipynb")
        image = dict(type="image", source=dict(type="base64", media_type="image/png", data="AAAA"))
        self.projector.project(tool_result_message("t4", [dict(type="text", text="line1"), image,
                                                          dict(type="text", text="line2")]))
        self.projector.project(tool_result_message("t5", "No matches", is_error=True))
        rows = self.rows()
        self.assertEqual((rows["t4"]["result"], rows["t4"]["status"]), ("line1\nline2", "completed"))
        self.assertEqual((rows["t5"]["status"], rows["t5"]["error"]), ("failed", dict(message="No matches")))
        self.assertEqual(rows["t4"]["message_id"], "m4")
        self.assertEqual(self.projector.abandon_open_items(), ["t0", "t1", "t2", "t3"])
        self.assertEqual(self.rows()["t0"]["status"], "failed")
        self.assertEqual(self.projector.abandon_open_items(), [])

    def test_subagent_output_is_linked_and_never_final(self):
        self.projector.project(assistant_message(tool_use("agent1", "Agent", prompt="go"), message_id="m1"))
        self.projector.project(assistant_message("sub says hi", message_id="sub1", parent_tool_use_id="agent1"))
        self.projector.project(assistant_message(tool_use("sub-bash", "Bash", command="pwd"), message_id="sub2",
                                                 parent_tool_use_id="agent1"))
        self.projector.project(tool_result_message("sub-bash", "/x", parent_tool_use_id="agent1"))
        self.projector.project(tool_result_message("agent1", "report"))
        self.projector.project(assistant_message("main answer", message_id="m2"))
        self.writer.turn_completed("completed")
        rows = self.rows()
        self.assertEqual(rows["sub-bash"]["parent_item_id"], "agent1")
        sub_text = [r for r in rows.values() if r["type"] == "agentMessage" and r.get("parent_item_id")]
        self.assertEqual([r["phase"] for r in sub_text], [None])
        self.assertEqual(self.store.completed_final_texts("thr", self.turn), ["main answer"])

    def test_unmatched_tool_result_and_unknown_message(self):
        projected = self.projector.project(tool_result_message("never-seen", "x"))
        self.assertEqual((projected.kind, projected.data["unmatched"]), ("tool_result", ["never-seen"]))
        self.assertEqual(self.projector.project(object()).kind, "other")
        self.assertEqual(self.rows(), {})

    def test_assistant_error_text_is_an_error_row_not_prose(self):
        projected = self.projector.project(assistant_message("API Error: 403 denied", error="authentication_failed",
                                                             message_id="e1"))
        self.assertTrue(projected.data["has_error"])
        self.assertEqual(self.of_type("agentMessage"), [])
        self.assertEqual(self.of_type("error")[0]["error"], dict(message="API Error: 403 denied"))

    def test_usage_is_per_message_id_latest_wins(self):
        self.projector.project(assistant_message("a", message_id="m", usage=usage_dict(input_tokens=5, output_tokens=1)))
        self.projector.project(assistant_message(tool_use("t", "Bash", command="x"), message_id="m",
                                                 usage=usage_dict(input_tokens=5, output_tokens=9)))
        self.projector.project(assistant_message("b", message_id="n", usage=usage_dict(input_tokens=7, output_tokens=2)))
        self.assertEqual(self.projector.native_usage_total(),
                         dict(input_tokens=12, cache_creation_input_tokens=0, cache_read_input_tokens=0,
                              output_tokens=11))

    def test_classify_result_table(self):
        cases = [
            (result_message(), False, ("completed", None)),
            (result_message(), True, ("interrupted", None)),
            (result_message(is_error=True, subtype="error_during_execution", terminal_reason="aborted_tools"), False,
             ("interrupted", None)),
            (result_message(is_error=True, terminal_reason="aborted_streaming"), False, ("interrupted", None)),
            (result_message(is_error=True, terminal_reason="api_error", api_error_status=529), False,
             ("failed", "claude_code_turn_failed:api_error:529")),
            (result_message(is_error=True, terminal_reason=None, subtype="error_max_turns"), False,
             ("failed", "claude_code_turn_failed:error_max_turns")),
        ]
        for message, interrupted, expected in cases:
            self.assertEqual(classify_result(message, interrupt_requested=interrupted), expected)


class Hygiene(unittest.TestCase):
    def test_projector_and_package_import_without_the_sdk(self):
        code = ("import sys; import auriga.cortex.agents.claude_code, auriga.cortex.agents.claude_code.projector, "
                "auriga.cortex.agents.claude_code.options; "
                "sys.exit(1 if 'claude_agent_sdk' in sys.modules else 0)")
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
