"""Offline replay safety and cache tests; the fixture is not production display data."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from auriga.cortex.session.native_terminal_replay import CACHE_VERSION, NativeReplayError, NativeTerminalReplay
from support.host_capabilities import native_replay_sandbox, require


THREAD = "01a100bb-1592-7820-a7bc-5ed3e48aaa50"
DISPLAY = "› original archived task\n• Called robot_observe · 12ms\n• Failed robot_move_ee · 34ms"


class NativeReplayTests(unittest.TestCase):
    def test_unknown_archive_schema_fails_before_native_capture(self):
        with sqlite3.connect(self.home / 'state_5.sqlite') as db:
            db.execute('ALTER TABLE threads RENAME COLUMN reasoning_effort TO incompatible_effort')
        with patch.object(self.replay, '_capture', side_effect=AssertionError('unsupported schema launched native CLI')):
            with self.assertRaisesRegex(NativeReplayError, 'unsupported native archive schema'):
                self.replay.get(self.evidence, THREAD)
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = self.root / "archive"
        self.home = self.evidence / "private" / "codex-home"
        sessions = self.home / "sessions" / "2026" / "10" / "03"
        sessions.mkdir(parents=True)
        self.rollout = sessions / f"rollout-2026-10-03T15-47-04-{THREAD}.jsonl"
        self.rollout.write_text(json.dumps({"type": "session_meta", "payload": {"id": THREAD}}) + "\n")
        self.auth = self.home / "auth.json"
        self.auth.write_text('{"secret":"do-not-copy"}')
        (self.home / "config.toml").write_text('untrusted_field = "do-not-copy"\n')
        self.original_terminal = self.evidence / "terminal.bin"
        self.original_terminal.write_bytes(b"Error: account/read failed during TUI bootstrap: archived failure")
        with sqlite3.connect(self.home / "state_5.sqlite") as db:
            db.execute("CREATE TABLE threads(id TEXT, model_provider TEXT, model TEXT, reasoning_effort TEXT, rollout_path TEXT, cwd TEXT)")
            db.execute("INSERT INTO threads VALUES(?,?,?,?,?,?)",
                       (THREAD, "auriga_model_proxy", "gpt-6-astra", "high", str(self.rollout), str(self.evidence)))
        with sqlite3.connect(self.home / "thread_history_1.sqlite") as db:
            db.execute("CREATE TABLE thread_turns(thread_id TEXT, turn_id TEXT, status TEXT)")
            db.execute("CREATE TABLE thread_items(thread_id TEXT, item_id TEXT, item_json TEXT, rollout_ordinal INTEGER)")
            db.execute("INSERT INTO thread_turns VALUES(?,?,?)", (THREAD, "turn-original", "interrupted"))
            for ordinal, item in enumerate([
                {"type": "userMessage", "content": [{"type": "text", "text": "original archived task"}]},
                {"type": "dynamicToolCall", "tool": "robot_observe", "status": "completed", "success": True, "durationMs": 12},
                {"type": "dynamicToolCall", "tool": "robot_move_ee", "status": "failed", "success": False, "durationMs": 34},
            ]):
                db.execute("INSERT INTO thread_items VALUES(?,?,?,?)",
                           (THREAD, "item-" + str(ordinal), json.dumps(item), ordinal))
        self.fixture = self.root / "native_fixture.py"
        self.fixture.write_text('''import json,os,socket,sys,time
from pathlib import Path
assert os.isatty(0) and os.isatty(1)
home = Path(os.environ["CODEX_HOME"])
assert not (home / "auth.json").exists()
assert "do-not-copy" not in (home / "config.toml").read_text()
assert not any("OPENAI" in key or "API_KEY" in key for key in os.environ)
for family in (socket.AF_INET, socket.AF_INET6):
    try: socket.socket(family)
    except PermissionError: pass
    else: raise AssertionError("network access was not blocked")
try: Path(sys.argv[1]).write_text("forbidden")
except PermissionError: pass
else: raise AssertionError("outside-cache write was not blocked")
(home / "fixture-command.json").write_text(json.dumps(sys.argv[2:]))
os.write(1,b"\\x1b[6n")
time.sleep(.1)
print("OpenAI Codex fixture: recorded history",flush=True)
print("''' + DISPLAY.replace("\n", "\\n") + '''",flush=True)
print("Ask Codex to do anything",flush=True)
time.sleep(30)
''')
        self.replay = NativeTerminalReplay(
            self.root / "cache", command_prefix=(sys.executable, str(self.fixture), str(self.original_terminal)),
            cli_version="test-fixture", timeout=3, quiet_period=.15)

    def fingerprint(self):
        return {str(path.relative_to(self.evidence)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.evidence.rglob("*") if path.is_file()}

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_native_bytes_are_cached_without_auth_archive_writes_or_new_turns(self):
        before = self.fingerprint()
        self.assertTrue(self.replay.available(self.evidence, THREAD))
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))
        result = self.replay.get(self.evidence, THREAD)
        self.assertEqual(before, self.fingerprint())
        self.assertEqual(self.replay.peek(self.evidence, THREAD), result)
        self.assertEqual(self.replay.get(self.evidence, THREAD), result)
        status = result["status"]
        self.assertEqual((status["cols"], status["rows"]), (140, 40))
        self.assertEqual(status["exit_code"], 0)
        self.assertFalse(status["bootstrap_failed"])
        self.assertEqual(status["replay"]["kind"], "recovered_native_tui")
        self.assertFalse(status["replay"]["original_recording"])
        self.assertEqual(status["replay"]["compatibility"]["adapter"], "codex_archive_v1")
        self.assertTrue(status["replay"]["original_bootstrap_failed"])
        self.assertTrue(status["replay"]["history_unchanged"])
        self.assertEqual(status["replay"]["recorded_turns"], 1)
        self.assertEqual(status["replay"]["recorded_items"], 3)
        self.assertEqual(status["replay"]["display_validation"]["expected_tools"], 2)
        self.assertTrue(status["replay"]["display_validation"]["complete"])
        self.assertEqual(status["replay"]["capture_exit_code"], -15)
        self.assertIn(b"recorded history", result["path"].read_bytes())
        copied = result["path"].parent / "codex-home"
        self.assertFalse((copied / "auth.json").exists())
        args = json.loads((copied / "fixture-command.json").read_text())
        self.assertEqual(args, ["resume", "--no-daemon", "--no-alt-screen", "-C",
                               str(result["path"].parent / "workspace"), "-s", "read-only",
                               "-a", "never", THREAD])

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_concurrent_requests_share_one_complete_capture(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.replay.get, self.evidence, THREAD) for _ in range(2)]
            first, second = [future.result(timeout=8) for future in futures]
        self.assertEqual(first, second)
        self.assertEqual(len(list((self.root / "cache").glob("*/metadata.json"))), 1)

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_source_change_and_corrupt_cache_are_not_reused(self):
        first = self.replay.get(self.evidence, THREAD)
        metadata_path = first["path"].parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["cache_version"] = CACHE_VERSION - 1
        metadata_path.write_text(json.dumps(metadata))
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))
        first["path"].write_bytes(b"corrupt")
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))
        repaired = self.replay.get(self.evidence, THREAD)
        self.assertIn(b"recorded history", repaired["path"].read_bytes())
        self.rollout.write_text(self.rollout.read_text() + '{"type":"event_msg","payload":{}}\n')
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))
        changed = self.replay.get(self.evidence, THREAD)
        self.assertNotEqual(first["path"], changed["path"])

    def test_invalid_identity_symlink_and_nested_cache_are_rejected(self):
        self.assertFalse(self.replay.available(self.evidence, "../../escape"))
        with self.assertRaises((ValueError, NativeReplayError)):
            self.replay.get(self.evidence, "../../escape")
        original = self.rollout.read_bytes()
        target = self.root / "external.jsonl"
        target.write_bytes(original)
        self.rollout.unlink(); self.rollout.symlink_to(target)
        self.assertFalse(self.replay.available(self.evidence, THREAD))
        with self.assertRaises(NativeReplayError):
            self.replay.get(self.evidence, THREAD)
        self.rollout.unlink(); self.rollout.write_bytes(original)
        nested = NativeTerminalReplay(self.evidence / "bad-cache", command_prefix=(sys.executable,), cli_version="fixture")
        self.assertFalse(nested.available(self.evidence, THREAD))
        link = self.root / "cache-link"
        link.symlink_to(self.root / "cache", target_is_directory=True)
        with self.assertRaises(NativeReplayError):
            NativeTerminalReplay(link, command_prefix=(sys.executable,), cli_version="fixture")

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_failure_and_deadline_never_publish_a_ready_cache(self):
        for program in ("print('Error: account/read failed during TUI bootstrap: unavailable')\n",
                        "import time\nprint('OpenAI Codex loading', flush=True)\ntime.sleep(30)\n"):
            with self.subTest(program=program):
                self.fixture.write_text(program)
                self.replay.timeout = .3
                with self.assertRaises(NativeReplayError):
                    self.replay.get(self.evidence, THREAD)
                self.assertIsNone(self.replay.peek(self.evidence, THREAD))
                source = self.replay._source(self.evidence, THREAD)
                directory = self.root / "cache" / source["key"]
                self.assertTrue((directory / "failure.json").is_file())
                self.assertFalse((directory / "metadata.json").exists())

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_archive_history_mutation_rejects_capture(self):
        self.fixture.write_text('''import os,sqlite3,time
from pathlib import Path
with sqlite3.connect(Path(os.environ["CODEX_HOME"])/"thread_history_1.sqlite") as db:
    db.execute("INSERT INTO thread_turns VALUES(?,?,?)", ("''' + THREAD + '''", "new-turn", "completed"))
print("OpenAI Codex fixture: recorded history",flush=True)
print("''' + DISPLAY.replace("\n", "\\n") + '''",flush=True)
print("Ask Codex to do anything",flush=True)
time.sleep(30)
''')
        before = self.fingerprint()
        with self.assertRaisesRegex(NativeReplayError, "changed archived turns or items"):
            self.replay.get(self.evidence, THREAD)
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))
        self.assertEqual(before, self.fingerprint())

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_loading_composer_and_partial_history_do_not_finish_capture(self):
        self.fixture.write_text('''import time
print("OpenAI Codex fixture: loading Resuming session…",flush=True)
print("Ask Codex to do anything",flush=True)
time.sleep(.35)
print("› original archived task",flush=True)
print("• Called robot_observe · 12ms",flush=True)
time.sleep(.35)
print("• Failed robot_move_ee · 34ms",flush=True)
time.sleep(30)
''')
        started = time.monotonic()
        result = self.replay.get(self.evidence, THREAD)
        self.assertGreater(time.monotonic() - started, .7)
        self.assertIn("Failed robot_move_ee".encode(), result["path"].read_bytes())
        self.assertEqual(result["status"]["replay"]["display_validation"]["observed_tools"], 2)

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_missing_messages_tools_or_original_tool_duration_reject_publication(self):
        outputs = ["loading Resuming session…",
                   "› original archived task\n• Called robot_observe · 12ms\n• Called robot_observe · 12ms",
                   "› original archived task\n• Called robot_observe · 12ms\n• Failed robot_move_ee · 35ms",
                   "• Called robot_observe · 12ms\n• Failed robot_move_ee · 34ms"]
        self.replay.timeout = .35
        for output in outputs:
            with self.subTest(output=output):
                self.fixture.write_text("import time\nprint('OpenAI Codex fixture',flush=True)\n"
                                       + f"print({output!r},flush=True)\n"
                                       + "print('Ask Codex to do anything',flush=True)\ntime.sleep(30)\n")
                with self.assertRaises(NativeReplayError):
                    self.replay.get(self.evidence, THREAD)
                self.assertIsNone(self.replay.peek(self.evidence, THREAD))


if __name__ == "__main__":
    unittest.main()
