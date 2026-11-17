"""No model/GPU calls: verify main-thread delivery, deduplication and kernel wait."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from support.host_capabilities import linux_proc, require

spec = importlib.util.spec_from_file_location("experiment_hook", Path(__file__).parents[1] / "scripts/experiment_hook.py")
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)
THREAD = "01900000-0000-7000-8000-000000000001"


def fake_codex(root):
    calls = root / "calls.jsonl"
    codex = root / "fake-codex"
    codex.write_text(f'''#!{sys.executable}
import json,sys
from pathlib import Path
with Path({str(calls)!r}).open('a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')
print('notification accepted')
print('fixture stderr', file=sys.stderr)
raise SystemExit(9 if Path({str(root / 'fail')!r}).exists() else 0)
''')
    codex.chmod(0o700)
    return codex, calls


class CompletionHookTests(unittest.TestCase):
    @require(linux_proc, "binds processes by /proc/<pid>/stat start ticks")
    def test_pidfd_adoption_accepts_packaged_entrypoints_and_rejects_unrelated_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "queue"
            output.mkdir()
            package = root / "auriga/apps/evaluation"
            package.mkdir(parents=True)
            (root / "auriga/__init__.py").write_text("")
            (root / "auriga/apps/__init__.py").write_text("")
            (package / "__init__.py").write_text("")
            source = "import sys,time\nprint('ready', flush=True)\nsys.stdin.readline()\ntime.sleep(.1)\n"
            (package / "queue.py").write_text(source)
            for name in ("auriga-queue", "unrelated.py"):
                (root / name).write_text(source)
            commands = ((["-u", "-m", "auriga.apps.evaluation.queue"], True),
                        ([str(root / "auriga-queue")], True),
                        (["-m", "unrelated", "auriga.apps.evaluation.queue"], False),
                        ([str(root / "unrelated.py"), "benchmark_queue.py"], False))
            for command, accepted in commands:
                with self.subTest(command=command):
                    process = subprocess.Popen([sys.executable, *command, "--output", str(output)], cwd=root,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                    try:
                        self.assertEqual(process.stdout.readline().strip(), "ready")
                        ticks = hook.process_identity(process.pid)
                        if accepted:
                            with self.assertRaisesRegex(RuntimeError, "reused"):
                                hook.wait_queue(output, process.pid, ticks + 1)
                            with self.assertRaisesRegex(RuntimeError, "output differs"):
                                hook.wait_queue(root / "other", process.pid, ticks)
                            process.stdin.write("exit\n")
                            process.stdin.flush()
                            hook.wait_queue(output, process.pid, ticks)
                            self.assertEqual(process.wait(timeout=5), 0)
                        else:
                            with self.assertRaisesRegex(RuntimeError, "not an Auriga benchmark queue"):
                                hook.wait_queue(output, process.pid, ticks)
                    finally:
                        if process.poll() is None:
                            process.stdin.write("exit\n")
                            process.stdin.flush()
                        process.wait(timeout=5)
                        process.stdin.close()
                        process.stdout.close()

    def test_stopped_queue_can_report_unconfirmed_cleanup_without_claiming_process_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codex, calls = fake_codex(root)
            for status in ("stop_unconfirmed", "exited"):
                output = root / status
                output.mkdir()
                hook.save_json(output / "state.json", {"status": "failed", "cleanup_stop_unconfirmed": True,
                    "cases": [{"status": status, "queue_cleanup": {"native_close_confirmed": False}}]})
                receipt = hook.notify_completion(output, THREAD, codex, queue_stopped=True)
                self.assertEqual(receipt["status"], "sent")
                prompt = (output / "completion-hook/prompt.txt").read_text()
                self.assertIn("实验调度已结束", prompt)
                self.assertIn("正常关闭未确认或发生强制回收", prompt)
                self.assertNotIn("进程已退出", prompt)
            self.assertEqual(len(calls.read_text().splitlines()), 2)
            active = root / "active"
            active.mkdir()
            hook.save_json(active / "state.json", {"status": "running", "cases": [{"status": "running"}]})
            with self.assertRaisesRegex(RuntimeError, "no terminal"):
                hook.notify_completion(active, THREAD, codex, queue_stopped=True)

    def test_existing_thread_notification_and_uncertain_delivery_never_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codex, calls = fake_codex(root)
            output = root / "queue"
            output.mkdir()
            hook.save_json(output / "state.json", {"status": "failed", "cases": [
                {"status": "exited", "returncode": 0, "summary": {"status": "completed"}},
                {"status": "exited", "returncode": 1, "summary": None},
                {"status": "exited", "returncode": 0},
                {"status": "not_started"}]})
            receipt = hook.notify_completion(output, THREAD, codex, remote="unix:///tmp/main-server.sock")
            self.assertEqual(receipt["status"], "sent")
            argv = json.loads(calls.read_text().splitlines()[0])
            self.assertEqual(argv[:4], ["queue", "--thread", THREAD, "--message"])
            self.assertEqual(argv[5:], ["--remote", "unix:///tmp/main-server.sock"])
            prompt = argv[4]
            self.assertIn(str(output.resolve()), prompt)
            self.assertEqual(prompt, f"实验已结束，结果目录：{output}")
            self.assertEqual(len(prompt.splitlines()), 1)
            artifacts = output / "completion-hook"
            self.assertEqual((artifacts / "prompt.txt").read_text(), prompt)
            self.assertEqual(hook.read_json(artifacts / "command.json")["argv"], [str(codex), *argv])
            self.assertEqual((artifacts / "stdout.log").read_text().strip(), "notification accepted")
            self.assertEqual((artifacts / "stderr.log").read_text().strip(), "fixture stderr")
            self.assertTrue(hook.notify_completion(output, THREAD, codex)["already_claimed"])
            with self.assertRaisesRegex(ValueError, "different thread"):
                hook.notify_completion(output, "01900000-0000-7000-8000-000000000002", codex)
            self.assertEqual(len(calls.read_text().splitlines()), 1)

            interrupted = root / "interrupted"
            interrupted.mkdir()
            hook.save_json(interrupted / ".main-session-wake.json", {"thread_id": THREAD, "status": "delivery_unknown"})
            self.assertEqual(hook.notify_completion(interrupted, THREAD, codex)["status"], "delivery_unknown")
            self.assertEqual(len(calls.read_text().splitlines()), 1)

            failure = root / "failure"
            failure.mkdir()
            hook.save_json(failure / "state.json", {"status": "gpu_wait_timeout", "cases": [{"status": "not_started"}]})
            (root / "fail").touch()
            receipt = hook.notify_completion(failure, THREAD, codex)
            self.assertEqual((receipt["status"], receipt["returncode"]), ("delivery_unknown", 9))
            self.assertTrue(hook.notify_completion(failure, THREAD, codex)["already_claimed"])
            self.assertEqual(len(calls.read_text().splitlines()), 2)
            (root / "fail").unlink()

            missing = root / "missing"
            missing.mkdir()
            receipt = hook.notify_completion(missing, THREAD, root / "absent-codex", process_exited=True)
            self.assertEqual((receipt["status"], receipt["returncode"]), ("delivery_unknown", None))
            self.assertIn("实验队列进程已退出", (missing / "completion-hook/prompt.txt").read_text())
            self.assertIn("state_error", hook.read_json(missing / "completion-hook/queue-state.json"))
            self.assertTrue(hook.notify_completion(missing, THREAD, codex)["already_claimed"])
            self.assertEqual(len(calls.read_text().splitlines()), 2)

    @require(linux_proc, "binds processes by /proc/<pid>/stat start ticks")
    def test_pidfd_identity_wait_and_active_state_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "queue"
            output.mkdir()
            codex, calls = fake_codex(root)
            hook.save_json(output / "state.json", {"status": "finished", "cases": [{"status": "running"}]})
            with self.assertRaisesRegex(RuntimeError, "possibly active"):
                hook.notify_completion(output, THREAD, codex)
            hook.save_json(output / "state.json", {"status": "running", "cases": [{"status": "not_started"}]})
            with self.assertRaisesRegex(RuntimeError, "no terminal"):
                hook.notify_completion(output, THREAD, codex)
            self.assertFalse(calls.exists())
            self.assertFalse((output / ".main-session-wake.json").exists())

            script = root / "benchmark_queue.py"
            script.write_text("import sys,time\nprint('ready', flush=True)\nsys.stdin.readline()\ntime.sleep(.1)\n")
            process = subprocess.Popen([sys.executable, str(script), "--output", str(output)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), "ready")
                ticks = hook.process_identity(process.pid)
                with self.assertRaisesRegex(RuntimeError, "reused"):
                    hook.wait_queue(output, process.pid, ticks + 1)
                with self.assertRaisesRegex(RuntimeError, "output differs"):
                    hook.wait_queue(root / "other", process.pid, ticks)
                process.stdin.write("exit\n")
                process.stdin.flush()
                hook.wait_queue(output, process.pid, ticks)
                self.assertEqual(process.wait(timeout=5), 0)
                # Queue exited without updating its record: notify for diagnosis, not success.
                receipt = hook.notify_completion(output, THREAD, codex, process_exited=True)
                self.assertEqual(receipt["status"], "sent")
                prompt = (output / "completion-hook/prompt.txt").read_text()
                self.assertIn("记录不完整、部分实验停止未确认", prompt)
                with self.assertRaisesRegex(RuntimeError, "no terminal"):
                    hook.wait_queue(output, process.pid, ticks)
                hook.save_json(output / "state.json", {"status": "finished", "cases": [{"status": "exited"}]})
                hook.wait_queue(output, process.pid, ticks)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                process.stdin.close()
                process.stdout.close()


if __name__ == "__main__":
    unittest.main()
