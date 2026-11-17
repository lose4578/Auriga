import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from auriga.apps.evaluation.native_evidence import (
    AgentRawEvidence, CodexRawEvidence, NativeEvidence, load_native_evidence)
from auriga.core.storage.store import Store

IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
T0, T1, T2 = "2026-01-01T00:00:01+00:00", "2026-01-01T00:00:02+00:00", "2026-01-01T00:00:03+00:00"


def write_lines(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def image_b64(payload):
    return base64.b64encode(payload).decode()


class StoreHelpersCheck(unittest.TestCase):
    def test_raw_agent_record_shape_and_channel_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, IDENTITY)
            store.raw_agent("sent", {"command": "interrupt"}, channel="control")
            store.raw_agent("received", {"jsonrpc": "2.0"}, channel="mcp")
            store.raw_agent("received", {"type": "assistant"}, channel="native")
            for channel in ("codex", "", None, "MCP"):
                with self.assertRaisesRegex(ValueError, "invalid_agent_raw_channel"):
                    store.raw_agent("sent", {}, channel=channel)
            with self.assertRaises(TypeError):
                store.raw_agent("sent", {})
            lines = (Path(directory) / "agent-raw.jsonl").read_text().splitlines()
            records = [json.loads(line) for line in lines]
            self.assertEqual([r["channel"] for r in records], ["control", "mcp", "native"])
            self.assertEqual(set(records[0]), {"received_at", "direction", "channel", "message"})
            self.assertEqual(records[0]["message"], {"command": "interrupt"})
            self.assertFalse((Path(directory) / "codex-raw.jsonl").exists())
            store.close()

    def test_raw_agent_requires_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            Store(directory, IDENTITY).close()
            reader = Store(directory, IDENTITY, read_only=True)
            with self.assertRaisesRegex(ValueError, "store_is_read_only"):
                reader.raw_agent("sent", {}, channel="control")
            reader.close()

    def test_transcript_helpers_share_the_codex_item_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, IDENTITY)
            self.assertEqual(Store.TRANSCRIPT_KIND, "codex_item")
            self.assertIsNone(store.transcript_item("a"))
            self.assertEqual(store.transcript_item("a", {}), {})
            row = dict(id="a", type="agentMessage", text="hi")
            self.assertEqual(store.set_transcript_item("a", row), row)
            self.assertEqual(store.transcript_item("a"), row)
            self.assertEqual(store.read_state("codex_item", "a"), row)
            store.set_state("codex_item", "b", dict(id="b"))
            self.assertEqual(store.transcript_item("b"), dict(id="b"))
            store.close()


class CodexEvidenceCheck(unittest.TestCase):
    def codex_log(self, root):
        png = b"\x89PNGcodex"
        write_lines(root / "codex-raw.jsonl", [
            dict(received_at=T0, direction="sent", message=dict(method="turn/start", params=dict(
                threadId="thr", input=[dict(type="text", text="go"), dict(type="image", url="x")]))),
            dict(received_at=T0, direction="sent", message=dict(method="turn/start", params=dict(
                threadId="other", input=[dict(type="text", text="ignored")]))),
            dict(received_at=T0, direction="received", message=dict(method="turn/started", params=dict(threadId="thr", turn=dict(id="t1")))),
            dict(received_at=T0, direction="received", message=dict(method="item/started", params=dict(item=dict(id="item-1")))),
            dict(received_at=T0, direction="received", message=dict(id=1, result=dict(contentItems=[
                dict(type="inputImage", imageUrl="data:image/png;base64," + image_b64(png)), dict(type="inputText")]))),
            dict(received_at=T1, direction="sent", message=dict(method="turn/interrupt", params=dict(threadId="thr", turnId="t1"))),
            dict(received_at=T2, direction="received", message=dict(method="turn/completed", params=dict(threadId="thr", turn=dict(
                id="t1", status="interrupted", items=[dict(type="agentMessage", text="AURIGA_CONTINUITY_OK"), dict(type="reasoning")])))),
            dict(received_at=T2, direction="received", message=dict(method="turn/completed", params=dict(threadId="other", turn=dict(id="x", status="completed")))),
        ])
        return hashlib.sha256(png).hexdigest()

    def test_audit_profile_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = self.codex_log(root)
            evidence = load_native_evidence(root, dict(thread_id="thr"), require_texts=True)
            self.assertEqual(evidence.tool_item_ids, {"item-1"})
            self.assertEqual(evidence.sent_image_sha256, {digest})
            self.assertEqual(evidence.turn_terminals, [dict(id="t1", status="interrupted")])
            self.assertEqual(evidence.completed_turn_texts, ["AURIGA_CONTINUITY_OK"])
            self.assertEqual([r["turn_id"] for r in evidence.interrupt_requests], ["t1"])
            self.assertIsNotNone(evidence.turn_started_at["t1"])
            self.assertEqual(evidence.turn_completed["t1"][0], "interrupted")
            self.assertEqual(evidence.submitted_prompts, [])
            self.assertEqual(load_native_evidence(root, dict(thread_id="thr")).completed_turn_texts, [])

    def test_report_profile_prompts_and_missing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = load_native_evidence(root, dict(thread_id="thr"), profile="report")
            self.assertFalse(missing.log_present)
            self.assertEqual(missing.submitted_prompts, [])
            self.codex_log(root)
            evidence = load_native_evidence(root, dict(thread_id="thr"), profile="report")
            self.assertTrue(evidence.log_present)
            self.assertEqual(evidence.submitted_prompts, [["go"]])
            self.assertEqual(load_native_evidence(root, dict(), profile="report").submitted_prompts, [])
            with self.assertRaises(FileNotFoundError):
                load_native_evidence(root / "nowhere", dict(thread_id="thr"))

    def test_malformed_records_keep_historical_failures(self):
        cases = [
            ("audit", [dict(direction="sent", message=[1])], "invalid_native_rpc_record"),
            ("audit", [dict(direction="sent", message=dict(params=[]))], "invalid_native_rpc_record"),
            ("audit", [[1]], "invalid_native_rpc_record"),
            ("report", [dict(direction="bogus", message={})], "invalid_native_rpc_record"),
            ("report", [dict(direction="sent", message=dict(params=3))], "invalid_native_rpc_record"),
            ("report", [dict(direction="sent", message=dict(method="turn/start", params=dict(threadId="", input=[])))], "invalid_native_turn_start"),
            ("report", [dict(direction="sent", message=dict(method="turn/start", params=dict(threadId="t", input=[dict(type="text")])))], "invalid_native_turn_start"),
        ]
        for profile, records, message in cases:
            with self.subTest(profile=profile, records=records), tempfile.TemporaryDirectory() as directory:
                write_lines(Path(directory) / "codex-raw.jsonl", records)
                with self.assertRaisesRegex(ValueError, message):
                    load_native_evidence(directory, dict(thread_id="t"), profile=profile)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "codex-raw.jsonl").write_text("{not json\n")
            with self.assertRaises(json.JSONDecodeError):
                load_native_evidence(directory, dict(thread_id="t"), profile="report")

    def test_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "codex-raw.jsonl").write_text("")
            (root / "agent-raw.jsonl").write_text("")
            for manifest in (dict(), dict(agent_backend=None), dict(agent_backend=dict(kind="codex"))):
                self.assertIsInstance(load_native_evidence(root, manifest), NativeEvidence)
            self.assertTrue(load_native_evidence(root, dict(agent_backend=dict(kind="claude_code"))).log_present)
            with self.assertRaisesRegex(ValueError, "unsupported_agent_backend"):
                load_native_evidence(root, dict(agent_backend=dict(kind="mystery")))
            with self.assertRaisesRegex(ValueError, "unsupported_native_evidence_profile"):
                load_native_evidence(root, dict(), profile="x")
            self.assertTrue(issubclass(CodexRawEvidence, object) and issubclass(AgentRawEvidence, object))


class AgentEvidenceCheck(unittest.TestCase):
    def build(self, directory):
        store = Store(directory, IDENTITY)
        png = b"\x89PNGagent"
        store.raw_agent("sent", dict(command="submit_prompt", thread_id="thr", turn_id="t1",
                                     input=[dict(type="text", text="first"), dict(type="image")]), channel="control")
        store.raw_agent("sent", dict(command="submit_prompt", thread_id="other", turn_id="z",
                                     input=[dict(type="text", text="ignored")]), channel="control")
        store.raw_agent("received", dict(jsonrpc="2.0", id=7, method="tools/call", params=dict(name="observe", _meta=dict(item_id="item-7"))), channel="mcp")
        store.raw_agent("received", dict(jsonrpc="2.0", id=8, method="tools/call", params=dict(name="plan")), channel="mcp")
        store.raw_agent("sent", dict(jsonrpc="2.0", id=7, result=dict(content=[
            dict(type="text", text="ok"), dict(type="image", data=image_b64(png), mimeType="image/png")])), channel="mcp")
        store.raw_agent("sent", dict(jsonrpc="2.0", id=8, result=dict(content=[dict(type="text", text="ok")])), channel="mcp")
        store.raw_agent("sent", dict(jsonrpc="2.0", id=99, result=dict(content=[])), channel="mcp")  # unmatched
        store.raw_agent("received", dict(type="assistant"), channel="native")
        store.raw_agent("sent", dict(command="interrupt", thread_id="thr", turn_id="t1"), channel="control")
        store.event("agent.turn_started", {}, dict(thread_id="thr", turn_id="t1"), source="agent_backend")
        store.event("agent.turn_completed", dict(status="interrupted"), dict(thread_id="thr", turn_id="t1"), source="agent_backend")
        store.event("agent.turn_completed", dict(status="completed"), dict(thread_id="thr", turn_id="t2"), source="agent_backend")
        store.event("agent.turn_completed", dict(status="completed"), dict(thread_id="other", turn_id="o1"), source="agent_backend")
        store.set_transcript_item("m1", dict(id="m1", thread_id="thr", turn_id="t2", type="agentMessage", text="AURIGA_CONTINUITY_OK"))
        store.set_transcript_item("m2", dict(id="m2", thread_id="thr", turn_id="t9", type="agentMessage", text="not terminal"))
        store.set_transcript_item("m3", dict(id="m3", thread_id="other", turn_id="o1", type="agentMessage", text="other thread"))
        store.set_transcript_item("c1", dict(id="c1", thread_id="thr", turn_id="t2", type="commandExecution"))
        return store, hashlib.sha256(png).hexdigest()

    def test_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            store, digest = self.build(directory)
            manifest = dict(thread_id="thr", agent_backend=dict(kind="claude_code"))
            evidence = load_native_evidence(directory, manifest, store, require_texts=True)
            self.assertEqual(evidence.tool_item_ids, {"item-7", "8"})
            self.assertEqual(evidence.sent_image_sha256, {digest})
            self.assertEqual(evidence.submitted_prompts, [["first"]])
            self.assertEqual([(r["turn_id"]) for r in evidence.interrupt_requests], ["t1"])
            self.assertIsNotNone(evidence.interrupt_requests[0]["sent_at"])
            self.assertEqual(evidence.turn_terminals, [dict(id="t1", status="interrupted"), dict(id="t2", status="completed")])
            self.assertEqual(set(evidence.interrupted_completion_seq), {"t1"})
            self.assertIsNotNone(evidence.turn_started_at["t1"])
            self.assertEqual(evidence.turn_completed["t2"][0], "completed")
            self.assertEqual(evidence.completed_turn_texts, ["AURIGA_CONTINUITY_OK"])
            self.assertEqual(load_native_evidence(directory, manifest, store).completed_turn_texts, [])
            without_store = load_native_evidence(directory, manifest)
            self.assertEqual(without_store.turn_terminals, [])
            self.assertEqual(without_store.submitted_prompts, [["first"]])
            store.close()

    def test_report_profile_tolerates_missing_log_audit_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = dict(thread_id="thr", agent_backend=dict(kind="opencode"))
            self.assertFalse(load_native_evidence(directory, manifest, profile="report").log_present)
            with self.assertRaises(FileNotFoundError):
                load_native_evidence(directory, manifest)

    def test_malformed_lines_are_rejected(self):
        bad = [
            dict(direction="sent", channel="mcp", message=[]),
            dict(direction="sideways", channel="mcp", message={}),
            dict(direction="sent", channel="codex", message={}),
            dict(direction="sent", message={}),
            [1],
        ]
        manifest = dict(thread_id="thr", agent_backend=dict(kind="claude_code"))
        for record in bad:
            with self.subTest(record=record), tempfile.TemporaryDirectory() as directory:
                write_lines(Path(directory) / "agent-raw.jsonl", [record])
                with self.assertRaisesRegex(ValueError, "invalid_agent_raw_record"):
                    load_native_evidence(directory, manifest)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "agent-raw.jsonl").write_text("{broken\n")
            with self.assertRaises(json.JSONDecodeError):
                load_native_evidence(directory, manifest)
        with tempfile.TemporaryDirectory() as directory:
            write_lines(Path(directory) / "agent-raw.jsonl", [dict(direction="sent", channel="control",
                message=dict(command="submit_prompt", thread_id="thr", input="oops"))])
            with self.assertRaisesRegex(ValueError, "invalid_agent_raw_record"):
                load_native_evidence(directory, manifest)


if __name__ == "__main__":
    unittest.main()
