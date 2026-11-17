"""``OpenCodeProjector``: OpenCode 1.19.5 messages/parts/events -> neutral transcript rows.

Inputs are the snapshots and ``/event`` frames captured from the real binary (spikes S6-S9,
``tests/fixtures/opencode``).
``Host`` plays the two other writers of the same transcript exactly as the session and the MCP transport do: the
host ``userMessage`` row is written before ``prompt_async`` (``run_turn``/``steer``), and every registry tool call
that reached the MCP endpoint has its ``dynamicToolCall`` row under a host call id (OpenCode sends none).
Constructed parts (edit/write/apply_patch/reasoning/retry: not exercised by the spikes) follow the 1.19.5 schemas in
``packages/schema/src/v1/session.ts`` and the tool metadata of ``packages/opencode/src/tool``.
"""
import copy
import tempfile
import unittest

from auriga.core.contracts import new_id
from auriga.core.storage.store import Store
from auriga.cortex.agents.opencode.projector import INTERRUPTED_TOOL_ERROR, OpenCodeProjector, TOOL_NAME_PREFIX
from auriga.cortex.agents.opencode.verdict import turn_verdict
from auriga.cortex.agents.transcript import TranscriptWriter
from test_agent_opencode_verdict import fold, load, session, until, user_id

IDENTITY = dict(run_id="check", episode_id="ep", env_instance_id="env", env_generation=1)


def registry_parts(messages):
    return [part for message in messages for part in message["parts"]
            if part["type"] == "tool" and part["tool"].startswith(TOOL_NAME_PREFIX)]


class Host:
    """Session + MCP transport stand-in sharing one store with the projector under test."""

    def __init__(self, store, thread_id):
        self.store, self.thread_id = store, thread_id
        self.writer = TranscriptWriter(store, backend="opencode", thread_id=thread_id)
        self.projector = OpenCodeProjector(self.writer)
        self.called = {}                       # part id -> host call id

    def prompt(self, text):
        turn = self.writer.turn_started()
        return turn, self.writer.user_message(text)

    def steer(self, text):
        return self.writer.steer(text)

    def mcp(self, messages, *, skip=()):
        """Transport rows for every registry part of ``messages`` that reached MCP and has none yet (in order)."""
        for part in registry_parts(messages):
            state = part["state"]
            if part["id"] in self.called or part["callID"] in skip or state["status"] not in (
                    "running", "completed", "error"):
                continue
            call_id = new_id("call")
            self.called[part["id"]] = call_id
            self.writer.tool_started(call_id, part["tool"][len(TOOL_NAME_PREFIX):], state["input"])
            if state["status"] != "running":
                text = state.get("output") or state.get("error") or ""
                self.writer.tool_completed(call_id, success=state["status"] == "completed",
                                           content_items=[dict(type="inputText", text=text)])

    def rows(self):
        return {row["id"]: row for row in self.store.states(self.store.TRANSCRIPT_KIND)
                if row.get("thread_id") == self.thread_id}

    def of_type(self, kind):
        return [row for row in self.rows().values() if row["type"] == kind]


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = Store(self.dir.name, IDENTITY)

    def tearDown(self):
        self.store.close()
        self.dir.cleanup()

    def host(self, messages):
        return Host(self.store, messages[0]["info"]["sessionID"])

    def finals(self, host, turn):
        return self.store.completed_final_texts(host.thread_id, turn)


class ToolTurn(Base):
    def run_turn_one(self, key="anthropic_two_turns"):
        messages = session("messages_basic.json", key)["messages"]
        host = self.host(messages)
        turn, item = host.prompt("USER-TURN-ONE: please work.")
        turn1 = until(messages, "FINAL-ANSWER-ONE")
        host.mcp(turn1)
        result = host.projector.sync(turn1)
        return messages, turn1, host, turn, item, result

    def test_rows_final_answer_and_no_second_registry_rows(self):
        for key in ("anthropic_two_turns", "responses_two_turns"):         # two sessions (threads), one store
            with self.subTest(key):
                messages, turn1, host, turn, item, result = self.run_turn_one(key)
                rows = host.rows()
                # registry parts: no rows of their own; transport rows carry callID + message id
                for part in registry_parts(turn1):
                    self.assertNotIn(part["id"], rows)
                    row = rows[host.called[part["id"]]]
                    self.assertEqual((row["native_id"], row["message_id"], row["type"]),
                                     (part["callID"], part["messageID"], "dynamicToolCall"))
                self.assertEqual(len(host.of_type("dynamicToolCall")), 5)
                parallel = [p for p in registry_parts(turn1) if p["callID"].startswith("call_fake_3_")]
                self.assertEqual(len({rows[host.called[p["id"]]]["message_id"] for p in parallel}), 1)
                # native bash
                bash = host.of_type("commandExecution")
                self.assertEqual(len(bash), 1)
                self.assertEqual((bash[0]["command"], bash[0]["exitCode"], bash[0]["status"], bash[0]["native_id"]),
                                 ("env | sort", 0, "completed", "call_fake_4_0"))
                self.assertEqual(bash[0]["aggregatedOutput"], "AGENT=1\nOPENCODE=1\n")
                # the host prompt row got the native id; no native duplicate
                users = host.of_type("userMessage")
                self.assertEqual([(u["id"], u["native_id"]) for u in users], [(item, turn1[0]["info"]["id"])])
                self.assertEqual((result.kind, result.data["last_user_message_id"], result.data["pending_submissions"],
                                  result.data["unmatched_registry"]), ("snapshot", turn1[0]["info"]["id"], 0, []))
                # final answer: the transcript stamp and the verdict agree
                verdict = turn_verdict(turn1, {}, last_user_message_id=result.data["last_user_message_id"])
                host.writer.turn_completed("completed")
                self.assertEqual(self.finals(host, turn), [verdict.final_text])
                self.assertEqual(verdict.final_text, "FINAL-ANSWER-ONE")
                commentary = [r for r in host.of_type("agentMessage") if r["text"] == "Calling fast."]
                self.assertEqual(commentary[0]["phase"], "commentary")
                self.assertEqual(commentary[0]["message_id"], turn1[1]["info"]["id"])
                self.assertNotIn("iVBOR", str(host.rows()))

    def test_resync_is_idempotent(self):
        messages, turn1, host, *_ = self.run_turn_one()
        before = self.store.states(self.store.TRANSCRIPT_KIND)
        writes = []
        original = self.store.set_state
        self.store.set_state = lambda kind, ident, value: (writes.append(ident), original(kind, ident, value))[1]
        for _ in range(2):
            host.projector.sync(turn1)
            host.projector.sync(copy.deepcopy(turn1))
        self.store.set_state = original
        self.assertEqual(self.store.states(self.store.TRANSCRIPT_KIND), before)
        # a fresh projector (no caches) re-derives the same rows
        OpenCodeProjector(host.writer).sync(turn1)
        self.assertEqual(self.store.states(self.store.TRANSCRIPT_KIND), before)

    def test_second_turn_rows_belong_to_the_second_turn(self):
        messages, turn1, host, turn, item, result = self.run_turn_one()
        host.writer.turn_completed("completed")
        first_rows = host.rows()
        turn2, item2 = host.prompt("USER-TURN-TWO: continue.")
        result = host.projector.sync(messages)
        rows = host.rows()
        self.assertEqual({k: v["turn_id"] for k, v in rows.items() if k in first_rows},
                         {k: v["turn_id"] for k, v in first_rows.items()})
        new = {k: v for k, v in rows.items() if k not in first_rows}
        self.assertEqual({v["turn_id"] for v in new.values()}, {turn2})
        self.assertEqual(result.data["last_user_message_id"], user_id(messages, "USER-TURN-TWO: continue."))
        host.writer.turn_completed("completed")
        self.assertEqual(self.finals(host, turn2), ["FINAL-ANSWER-TWO"])
        self.assertEqual(self.finals(host, turn), ["FINAL-ANSWER-ONE"])

    def test_running_part_seen_before_its_mcp_request_is_matched_later(self):
        """A part turns ``running`` when OpenCode starts the call; the MCP request (and the transport row) may land
        after a sync. The captured completed part is rewound to its running state for the first sync."""
        messages = session("messages_steer_abort.json", "F_reconnect")["messages"]
        running = copy.deepcopy(messages[:2])
        tool = [p for p in running[1]["parts"] if p["type"] == "tool"][0]
        tool["state"] = dict(status="running", input=tool["state"]["input"], time=dict(start=1))
        running[1]["info"].pop("finish")
        host = self.host(messages)
        host.prompt("USER-ONE")
        result = host.projector.sync(running)
        self.assertEqual(result.data["unmatched_registry"], [tool["callID"]])
        self.assertEqual(host.of_type("mcpToolCall"), [])                  # never a second row for a registry call
        host.mcp(running)
        self.assertEqual(host.projector.sync(running).data["unmatched_registry"], [])
        host.projector.sync(messages)
        row = host.rows()[host.called[tool["id"]]]
        self.assertEqual((row["native_id"], row["message_id"], row["status"]),
                         (tool["callID"], tool["messageID"], "inProgress"))   # status stays the transport's

    def test_registry_call_refused_before_mcp_becomes_one_failed_row(self):
        messages = session("messages_basic.json", "anthropic_two_turns")["messages"]
        host = self.host(messages)
        host.prompt("USER-TURN-ONE: please work.")
        turn1 = until(messages, "FINAL-ANSWER-ONE")
        host.mcp(turn1, skip=("call_fake_2_0",))                           # auriga_fail never reached MCP
        for _ in range(2):
            host.projector.sync(turn1)
        failed = [p for p in registry_parts(turn1) if p["callID"] == "call_fake_2_0"][0]
        row = host.rows()[failed["id"]]
        self.assertEqual((row["type"], row["tool"], row["status"], row["native_id"]),
                         ("mcpToolCall", "auriga_fail", "failed", "call_fake_2_0"))
        self.assertEqual(row["error"], dict(message="tool failed: planted failure text"))
        self.assertEqual(len(host.of_type("mcpToolCall")), 1)
        self.assertEqual(len(host.of_type("dynamicToolCall")), 4)


class NativeTools(Base):
    def test_grep_glob_read_bash(self):
        messages = session("messages_native_tools.json", "grep_glob_read_bash")["messages"]
        host = self.host(messages)
        turn, _ = host.prompt("run tools")
        host.projector.sync(messages)
        generic = {row["tool"]: row for row in host.of_type("mcpToolCall")}
        self.assertEqual(set(generic), {"grep", "glob", "read"})
        self.assertEqual((generic["grep"]["arguments"], generic["grep"]["result"], generic["grep"]["status"]),
                         ({"pattern": "zzz", "path": "."}, "No files found", "completed"))
        self.assertEqual(generic["read"]["status"], "failed")
        self.assertEqual(generic["read"]["error"],
                         {"message": "File not found: /runs/s8-controls-C_hardened/actor/hello.txt"})
        bash = host.of_type("commandExecution")[0]
        self.assertEqual((bash["command"], bash["aggregatedOutput"], bash["exitCode"]),
                         ("echo hi > hello.txt; ls", "hello.txt\n", 0))
        self.assertTrue(all(row["turn_id"] == turn and row["backend"] == "opencode" for row in host.rows().values()))
        host.writer.turn_completed("completed")
        self.assertEqual(self.finals(host, turn), ["CONTROL-DONE"])

    def test_abort_during_bash(self):
        messages = session("messages_native_tools.json", "abort_during_bash")["messages"]
        host = self.host(messages)
        host.prompt("sleep")
        host.projector.sync(messages)
        bash = host.of_type("commandExecution")[0]
        self.assertEqual(bash["status"], "completed")
        self.assertIn("User aborted the command", bash["aggregatedOutput"])
        self.assertEqual(host.of_type("error"), [])                         # MessageAbortedError is not an error row

    def test_dangling_bash_after_server_kill(self):
        messages = session("messages_native_tools.json", "server_killed_during_bash")["messages"]
        host = self.host(messages)
        host.prompt("sleep")
        result = host.projector.sync(messages)
        part_id = [p for p in messages[1]["parts"] if p["type"] == "tool"][0]["id"]
        self.assertEqual(host.rows()[part_id]["status"], "inProgress")     # last assistant message: not superseded
        self.assertEqual(result.data["interrupted_tools"], [])
        self.assertEqual(host.projector.abandon_open_items(), [part_id])
        host.projector.sync(messages)                                      # native still says running: stays final
        row = host.rows()[part_id]
        self.assertEqual((row["status"], row["interrupted"], row["error"]),
                         ("failed", True, {"message": INTERRUPTED_TOOL_ERROR}))
        self.assertEqual(host.projector.abandon_open_items(), [])

    def test_dangling_native_tool_in_superseded_message(self):
        """S8 shape: after the restart the next prompt is answered, so the dead step is no longer the last
        assistant message. The follow-up messages are constructed (no native tool died in the S8 restart run)."""
        messages = copy.deepcopy(session("messages_native_tools.json", "server_killed_during_bash")["messages"])
        follow_user = dict(info=dict(messages[0]["info"], id="msg_zz_user_after_restart", time=dict(created=2)),
                           parts=[dict(type="text", text="again", id="prt_zz_1", messageID="msg_zz_user_after_restart",
                                       sessionID=messages[0]["info"]["sessionID"])])
        follow = dict(info=dict(messages[1]["info"], id="msg_zz_reply", parentID="msg_zz_user_after_restart",
                                finish="stop", time=dict(created=3, completed=4)),
                      parts=[dict(type="text", text="AGAIN-DONE", id="prt_zz_2", messageID="msg_zz_reply",
                                  sessionID=messages[0]["info"]["sessionID"], time=dict(start=3, end=4))])
        host = self.host(messages)
        host.prompt("sleep")
        result = host.projector.sync(messages + [follow_user, follow])
        part_id = [p for p in messages[1]["parts"] if p["type"] == "tool"][0]["id"]
        self.assertEqual(result.data["interrupted_tools"], [part_id])
        row = host.rows()[part_id]
        self.assertEqual((row["status"], row["interrupted"]), ("failed", True))
        # the unexpected native user message becomes its own (non-host) row
        native_user = host.rows()["msg_zz_user_after_restart"]
        self.assertEqual((native_user["source"], native_user["synthetic"]), ("native", False))

    def test_file_tools_reasoning_and_retry_parts(self):
        sid, mid = "ses_constructed", "msg_constructed_a"
        info = dict(id=mid, sessionID=sid, role="assistant", parentID="msg_constructed_u", finish="tool-calls",
                    time=dict(created=2, completed=3))
        base = dict(sessionID=sid, messageID=mid)
        edit = dict(base, id="prt_1", type="tool", tool="edit", callID="c1", state=dict(
            status="completed", input=dict(filePath="/runs/x/actor/a.py", oldString="a", newString="b"),
            output="Edit applied successfully.", title="a.py", metadata=dict(diff="-a\n+b", filediff={}),
            time=dict(start=1, end=2)))
        write = dict(base, id="prt_2", type="tool", tool="write", callID="c2", state=dict(
            status="completed", input=dict(filePath="/runs/x/actor/b.txt", content="hi"), output="Wrote file",
            title="b.txt", metadata=dict(filepath="/runs/x/actor/b.txt", exists=False), time=dict(start=1, end=2)))
        patch_text = "*** Begin Patch\n*** Add File: c.txt\n+x\n*** Delete File: d.txt\n*** End Patch"
        patch_running = dict(base, id="prt_3", type="tool", tool="apply_patch", callID="c3", state=dict(
            status="running", input=dict(patchText=patch_text), time=dict(start=1)))
        reasoning = dict(base, id="prt_4", type="reasoning", text="think first",
                         metadata=dict(anthropic=dict(signature="SIG-SECRET")), time=dict(start=1, end=2))
        retry = dict(base, id="prt_5", type="retry", attempt=2, time=dict(created=1),
                     error=dict(name="APIError", data=dict(message="provider text", statusCode=529)))
        user = dict(info=dict(id="msg_constructed_u", sessionID=sid, role="user", time=dict(created=1)),
                    parts=[dict(type="text", text="go", id="prt_0", sessionID=sid, messageID="msg_constructed_u")])
        host = Host(self.store, sid)
        host.prompt("go")
        host.projector.sync([user, dict(info=info, parts=[reasoning, edit, write, patch_running, retry])])
        rows = host.rows()
        self.assertEqual(rows["prt_1"]["changes"], [dict(path="/runs/x/actor/a.py", kind="update", diff="-a\n+b")])
        self.assertEqual(rows["prt_2"]["changes"], [dict(path="/runs/x/actor/b.txt", kind="write", diff=None)])
        self.assertEqual(rows["prt_3"]["changes"], [dict(path="c.txt", kind="add", diff=None),
                                                    dict(path="d.txt", kind="delete", diff=None)])
        self.assertEqual(rows["prt_3"]["status"], "inProgress")
        self.assertEqual((rows["prt_4"]["type"], rows["prt_4"]["summary"]), ("reasoning", ["think first"]))
        self.assertNotIn("SIG-SECRET", str(rows))
        self.assertEqual(rows["prt_5"]["error"], {"message": "opencode_retry:2:APIError:529"})
        self.assertNotIn("provider text", str(rows))
        # apply_patch finished: metadata.files wins
        done = dict(patch_running, state=dict(status="completed", input=dict(patchText=patch_text), output="ok",
                                              title="ok", time=dict(start=1, end=2), metadata=dict(files=[
                                                  dict(filePath="/runs/x/actor/c.txt", relativePath="c.txt",
                                                       type="add", patch="+x", additions=1, deletions=0)])))
        host.projector.sync([user, dict(info=info, parts=[reasoning, edit, write, done, retry])])
        self.assertEqual(host.rows()["prt_3"]["changes"], [dict(path="/runs/x/actor/c.txt", kind="add", diff="+x")])
        self.assertEqual(host.rows()["prt_3"]["status"], "completed")


class Steer(Base):
    def test_steer_during_final_text(self):
        messages = session("messages_steer_abort.json", "C_steer_during_final")["messages"]
        host = self.host(messages)
        turn, prompt = host.prompt("USER-ONE")
        steer = host.steer("STEER-DURING-FINAL")
        result = host.projector.sync(until(messages, "FIRST-FINAL"))
        self.assertEqual((result.data["last_user_message_id"], result.data["pending_submissions"]), (None, 1))
        result = host.projector.sync(messages)
        steer_id = user_id(messages, "STEER-DURING-FINAL")
        self.assertEqual((result.data["last_user_message_id"], result.data["pending_submissions"]), (steer_id, 0))
        rows = host.rows()
        self.assertEqual((rows[prompt]["native_id"], rows[steer]["native_id"], rows[steer]["steer"]),
                         (messages[0]["info"]["id"], steer_id, True))
        self.assertEqual(len(host.of_type("userMessage")), 2)
        verdict = turn_verdict(messages, {}, last_user_message_id=result.data["last_user_message_id"])
        host.writer.turn_completed(verdict.status)
        self.assertEqual(self.finals(host, turn), ["SECOND-FINAL"])
        phases = {row["text"]: row["phase"] for row in host.of_type("agentMessage")}
        self.assertEqual(phases, {"FIRST-FINAL": "commentary", "SECOND-FINAL": "final_answer"})

    def test_two_steers_match_in_order_and_withdrawn_steer(self):
        messages = session("messages_steer_abort.json", "C2_two_steers")["messages"]
        host = self.host(messages)
        host.prompt("USER-ONE")
        a, b = host.steer("STEER-A"), host.steer("STEER-B")
        rejected = host.steer("STEER-REJECTED")
        self.assertEqual(host.projector.submissions(), (None, 4))
        host.projector.withdraw_submission(rejected)
        result = host.projector.sync(messages)
        rows = host.rows()
        self.assertEqual((rows[a]["native_id"], rows[b]["native_id"]),
                         (user_id(messages, "STEER-A"), user_id(messages, "STEER-B")))
        self.assertEqual((result.data["last_user_message_id"], result.data["pending_submissions"]),
                         (user_id(messages, "STEER-B"), 0))
        self.assertFalse(host.projector.withdraw_submission("item-missing"))

    def test_steer_written_during_a_sync_keeps_the_turn_pending(self):
        """The session thread may write a steer row while the reconcile thread is inside ``sync``: the reported
        anchor must reflect it (fresh read), so the verdict cannot complete on the pre-steer answer."""
        messages = session("messages_steer_abort.json", "C_steer_during_final")["messages"]
        host = self.host(messages)
        host.prompt("USER-ONE")
        original = host.projector._part
        written = []

        def part_then_steer(*args, **kwargs):
            original(*args, **kwargs)
            if not written:
                written.append(host.steer("STEER-DURING-FINAL"))
        host.projector._part = part_then_steer
        result = host.projector.sync(until(messages, "FIRST-FINAL"))
        self.assertEqual((result.data["last_user_message_id"], result.data["pending_submissions"]), (None, 1))
        self.assertEqual(turn_verdict(until(messages, "FIRST-FINAL"), {}, last_user_message_id=None).status,
                         "running")

    def test_steer_text_from_another_turn_is_not_matched(self):
        messages = session("messages_steer_abort.json", "A_steer_during_model")["messages"]
        host = self.host(messages)
        host.prompt("STEER-DURING-MODEL")                                  # stale host row of an older turn
        stale = host.of_type("userMessage")[0]["id"]
        host.prompt("USER-ONE")
        steer = host.steer("STEER-DURING-MODEL")
        host.projector.sync(messages)
        rows = host.rows()
        self.assertNotIn("native_id", rows[stale])
        self.assertEqual(rows[steer]["native_id"], user_id(messages, "STEER-DURING-MODEL"))


class AbortAndErrors(Base):
    def test_abort_during_mcp_call(self):
        messages = session("messages_steer_abort.json", "E_abort_tool+E2_after_abort_tool")["messages"]
        host = self.host(messages)
        host.prompt("USER-ONE")
        host.mcp(messages[:2])                                             # the MCP request was in flight
        host.projector.sync(messages[:2])
        call = host.of_type("dynamicToolCall")
        self.assertEqual(len(call), 1)
        self.assertEqual((call[0]["native_id"], call[0]["tool"]), ("call_fake_10_0", "slow"))
        self.assertEqual(host.of_type("mcpToolCall"), [])
        self.assertEqual(host.of_type("error"), [])
        host.writer.turn_completed("interrupted")
        turn2, _ = host.prompt("USER-AFTER-ABORT-TOOL")
        host.projector.sync(messages)
        host.writer.turn_completed("completed")
        self.assertEqual(self.finals(host, turn2), ["AFTER-ABORT-TOOL-REPLY"])

    def test_abort_during_model_call_writes_no_rows(self):
        messages = session("messages_steer_abort.json", "D_abort_model+D2_after_abort_model")["messages"]
        host = self.host(messages)
        host.prompt("USER-ONE")
        host.projector.sync(messages[:2])
        self.assertEqual([row["type"] for row in host.rows().values()], ["userMessage"])

    def test_error_rows_carry_no_provider_text(self):
        for key, code in (("deny400", "opencode_turn_failed:APIError:400"),
                          ("e500_retried", "opencode_turn_failed:APIError:500"),
                          ("stream_error", "opencode_turn_failed:UnknownError")):
            with self.subTest(key):
                messages = session("messages_errors.json", key)["messages"]
                host = self.host(messages)
                host.prompt(messages[0]["parts"][0]["text"])
                host.projector.sync(messages)
                errors = host.of_type("error")
                self.assertEqual([(e["id"], e["error"]) for e in errors],
                                 [(messages[1]["info"]["id"] + ":error", {"message": code})])
                self.assertEqual(host.of_type("agentMessage"), [])          # the empty retry text parts
                for word in ("denied", "provider", "overloaded"):
                    self.assertNotIn(word, str(host.rows()))


class Compaction(Base):
    def test_three_turns_with_auto_compaction(self):
        messages = session("messages_compaction.json", "three_turns_compacted_in_turn_3")["messages"]
        host = self.host(messages)
        turns = []
        for prompt, final in (("USER-TURN-1", "TURN-1-DONE"), ("USER-TURN-2", "TURN-2-DONE"),
                              ("USER-TURN-3", "TURN-3-DONE")):
            turn, _ = host.prompt(prompt)
            prefix = until(messages, final)
            host.mcp(prefix)
            result = host.projector.sync(prefix)
            verdict = turn_verdict(prefix, {}, last_user_message_id=result.data["last_user_message_id"])
            self.assertEqual((verdict.status, verdict.final_text), ("completed", final))
            host.writer.turn_completed("completed")
            self.assertEqual(self.finals(host, turn), [final])
            turns.append((turn, result))
        compaction_part = [p for m in messages for p in m["parts"] if p["type"] == "compaction"][0]
        self.assertEqual([r.data["compactions"] for _, r in turns], [[], [], [compaction_part["id"]]])
        self.assertEqual(host.projector.sync(messages).data["compactions"], [])        # reported once
        row = host.rows()[compaction_part["id"]]
        self.assertEqual((row["type"], row["auto"], row["overflow"], row["turn_id"]),
                         ("contextCompaction", True, False, turns[2][0]))
        summary = [r for r in host.of_type("agentMessage") if r["text"].startswith("SUMMARY:")][0]
        self.assertEqual((summary["parent_item_id"], summary["phase"]), (compaction_part["id"], None))
        synthetic = [r for r in host.of_type("userMessage") if r.get("source") == "native"]
        self.assertEqual(len(synthetic), 1)
        self.assertTrue(synthetic[0]["synthetic"])
        self.assertTrue(synthetic[0]["content"][0]["text"].startswith("Continue if you have next steps"))
        # identical-argument calls (auriga_big {"kb": 3}) are paired with transport rows in FIFO order
        for part in registry_parts(messages):
            self.assertEqual(host.rows()[host.called[part["id"]]]["native_id"], part["callID"])
        self.assertEqual(len(host.of_type("dynamicToolCall")), len(registry_parts(messages)))


class Restart(Base):
    def test_resume_adopts_history_then_runs_a_new_turn(self):
        capture = load("messages_restart.json")["sessions"]
        crashed, final = capture["after_crash"]["messages"], capture["final"]["messages"]
        first = self.host(crashed)
        first.prompt("TURN-1")
        turn1_part = until(crashed, "FIRST-TURN-DONE")
        first.mcp(turn1_part)
        first.projector.sync(turn1_part)
        first.writer.turn_completed("completed")
        first.prompt("TURN-2")
        first.mcp(crashed)                                                # auriga_slow reached MCP, then SIGKILL
        first.projector.sync(crashed)
        before = self.store.states(self.store.TRANSCRIPT_KIND)
        # new process: new writer + projector on the same run store; start() syncs before the first turn
        second = Host(self.store, first.thread_id)
        result = second.projector.sync(crashed)
        self.assertEqual(self.store.states(self.store.TRANSCRIPT_KIND), before)
        self.assertEqual((result.data["last_user_message_id"], result.data["pending_submissions"]), (None, 0))
        turn3, item3 = second.prompt("TURN-3-AFTER-CRASH")
        result = second.projector.sync(final)
        self.assertEqual(result.data["last_user_message_id"], user_id(final, "TURN-3-AFTER-CRASH"))
        verdict = turn_verdict(final, {}, last_user_message_id=result.data["last_user_message_id"])
        second.writer.turn_completed(verdict.status)
        self.assertEqual(self.finals(second, turn3), ["AFTER-CRASH-REPLY"])
        rows = second.rows()
        self.assertEqual({row["turn_id"] for k, row in rows.items() if k not in {r["id"] for r in before}}, {turn3})
        slow = [r for r in second.of_type("dynamicToolCall") if r["tool"] == "slow"][0]
        self.assertEqual((slow["status"], slow["native_id"]), ("inProgress", "call_fake_2_0"))   # transport-owned


class Events(Base):
    def test_event_replay_matches_snapshot_projection(self):
        """S7 A frames (``events_turn_with_tool.json``) through project_event, then a sync of the folded snapshot,
        give the same rows as a sync alone on a fresh store."""
        events = load("events_turn_with_tool.json")
        messages, status = fold(events)
        self.assertEqual(status, {})
        host = self.host(messages)
        host.prompt("USER-ONE")
        host.mcp(messages)
        host.steer("STEER-DURING-MODEL")
        kinds = [host.projector.project_event(frame).kind for frame in events]
        self.assertIn("delta", kinds)
        self.assertIn("ignored", [host.projector.project_event(dict(frame, properties=dict(
            frame["properties"], sessionID="ses_other"))).kind for frame in events[-2:]])
        live = {row["text"]: row["status"] for row in host.of_type("agentMessage")}
        self.assertEqual(live, {"REPLY-AFTER-TOOL": "completed"})
        result = host.projector.sync(messages)
        self.assertEqual(result.data["pending_submissions"], 0)
        projected = self._normal(host.rows())
        other = tempfile.TemporaryDirectory()
        store = Store(other.name, IDENTITY)
        try:
            fresh = Host(store, host.thread_id)
            fresh.prompt("USER-ONE")
            fresh.mcp(messages)
            fresh.steer("STEER-DURING-MODEL")
            fresh.projector.sync(messages)
            self.assertEqual(projected, self._normal(fresh.rows()))
        finally:
            store.close()
            other.cleanup()

    def test_event_gap_then_snapshot_reconciles(self):
        """/event drops after half of the frames (no replay); the reconcile sync of the snapshot read after the
        reconnect yields the same rows as an uninterrupted projection."""
        events = load("events_turn_with_tool.json")
        messages, _ = fold(events)
        host = self.host(messages)
        host.prompt("USER-ONE")
        host.mcp(messages)
        host.steer("STEER-DURING-MODEL")
        for frame in events[:len(events) // 2]:
            host.projector.project_event(frame)
        self.assertNotIn("REPLY-AFTER-TOOL", [r["text"] for r in host.of_type("agentMessage")])
        result = host.projector.sync(messages)
        self.assertEqual((result.data["pending_submissions"], result.data["unmatched_registry"]), (0, []))
        self.assertEqual([r["text"] for r in host.of_type("agentMessage")], ["REPLY-AFTER-TOOL"])
        self.assertEqual(turn_verdict(messages, {}, last_user_message_id=result.data["last_user_message_id"])
                         .final_text, "REPLY-AFTER-TOOL")

    @staticmethod
    def _normal(rows):
        keep = ("type", "text", "status", "native_id", "message_id", "tool", "arguments", "content", "steer")
        return sorted((tuple((k, str(row.get(k))) for k in keep)) for row in rows.values())

    def test_streamed_deltas_and_unknown_frames(self):
        capture = load("events_steer_abort.json")["sessions"]["C_steer_during_final"]
        events = capture["events"]
        host = Host(self.store, capture["session_id"])
        host.prompt("USER-ONE")
        host.steer("STEER-DURING-FINAL")
        seen = {}
        for frame in events:
            projected = host.projector.project_event(frame)
            if frame["type"] == "message.part.delta":
                seen[frame["properties"]["delta"]] = (projected.data["applied"], host.rows()[
                    frame["properties"]["partID"]]["text"])
        self.assertEqual(seen, {"FIRST-FINAL": (True, "FIRST-FINAL"), "SECOND-FINAL": (True, "SECOND-FINAL")})
        self.assertEqual(host.projector.submissions()[1], 0)
        self.assertEqual(host.projector.project_event(dict(type="message.part.updated", properties=dict(
            part=dict(id="prt_x", messageID="msg_unknown", sessionID=capture["session_id"], type="text",
                      text="x")))).kind, "deferred")
        self.assertFalse(host.projector.project_event(dict(type="message.part.delta", properties=dict(
            sessionID=capture["session_id"], messageID="msg_unknown", partID="prt_y", field="text",
            delta="z"))).data["applied"])
        self.assertEqual(host.projector.project_event(dict(type="server.heartbeat", properties={})).kind, "other")
        self.assertEqual(host.projector.project_event("not a frame").kind, "other")
        self.assertEqual(host.projector.project_event(dict(type="session.compacted", properties=dict(
            sessionID=capture["session_id"]))).kind, "compacted")

    def test_compaction_part_event_is_the_early_signal(self):
        messages = session("messages_compaction.json", "three_turns_compacted_in_turn_3")["messages"]
        window = load("events_compaction.json")["sessions"]["compaction_window"]
        trigger = next(i for i, m in enumerate(messages) if any(p["type"] == "compaction" for p in m["parts"]))
        host = self.host(messages)
        host.prompt("USER-TURN-3")
        host.projector.sync(messages[:trigger])
        kinds = [host.projector.project_event(frame) for frame in window["events"]]
        compactions = [p for p in kinds if p.kind == "compaction"]
        self.assertEqual([c.data["new"] for c in compactions], [True] + [False] * (len(compactions) - 1))
        self.assertTrue(compactions[0].data["auto"])
        self.assertEqual(len({c.data["part_id"] for c in compactions}), 1)
        self.assertEqual(kinds[-1].kind, "compacted")
        summary = [r for r in host.of_type("agentMessage") if r["text"].startswith("SUMMARY:")][0]
        self.assertEqual(summary["parent_item_id"], compactions[0].data["part_id"])
        self.assertEqual(len([r for r in host.of_type("userMessage") if r.get("synthetic")]), 1)
        # the folded window equals the captured messages
        folded, _ = fold(window["events"], base=messages[:trigger])
        self.assertEqual([m["info"]["id"] for m in folded], [m["info"]["id"] for m in messages[:trigger + 3]])


if __name__ == "__main__":
    unittest.main()
