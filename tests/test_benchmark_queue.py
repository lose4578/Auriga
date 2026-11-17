"""Queue policy checks use synthetic GPU samples and owned fake subprocesses."""
import json
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from scripts import benchmark_queue as queue
from scripts import queue_retry
from support.host_capabilities import linux_proc, require


def observation(utilization=0, free=30000, index=0, uuid="GPU-a"):
    return {"gpus": [dict(index=index, uuid=uuid, utilization=utilization, free_mib=free)],
            "error": None, "sample_started_monotonic": queue.time.monotonic()}


class FastEvent(threading.Event):
    waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.is_set()


class QueueCheck(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        patch = mock.patch.object(queue, "GPU_LOCK_DIR", self.root / "locks")
        patch.start()
        self.addCleanup(patch.stop)
        settings = mock.patch.object(queue, "EXPERIMENT_SETTINGS", self.root / "experiments.json")
        settings.start()
        self.addCleanup(settings.stop)
        binding = mock.patch.object(queue, "WAKE_THREAD_FILE", self.root / "completion-thread")
        binding.start()
        self.addCleanup(binding.stop)
        self.cwd = self.root / "checkout"
        (self.cwd / "auriga/apps/gateway").mkdir(parents=True)
        (self.cwd / "auriga/apps/gateway/cli.py").write_text("parser.add_argument('--config')\nConfig.load()\n")
        self.case = dict(cwd=str(self.cwd), run_dir=str(self.root / "run-1"), task="general_pickup",
                         seed=0, layout_id=2, title="paired baseline", tag=["paired", "baseline"])

    def test_new_queue_freezes_parallel_default_and_manifest_precedence(self):
        for configured, manifest_value, expected in (({}, {}, 60), ({'max_parallel': None}, {}, 60),
                ({'max_parallel': 1}, {}, 1), ({'max_parallel': 2}, {'max_parallel': 3}, 3)):
            with self.subTest(configured=configured, manifest=manifest_value):
                config = queue.Config.from_dict({'queue': configured})
                manifest = {'cases': [self.case], **manifest_value}
                before = json.dumps(manifest)
                effective = queue.configured_manifest(manifest, config)
                self.assertEqual(effective['max_parallel'], expected)
                self.assertEqual(effective['auriga_config']['queue']['max_parallel'], expected)
                self.assertEqual(json.dumps(manifest), before)
                self.assertEqual(config.queue.max_parallel, configured.get('max_parallel', 60))

    def test_experiment_configuration_is_scoped_and_reloaded_without_logging_secrets(self):
        homes = [self.root / name for name in ("profile-one", "profile-two")]
        for home in homes:
            home.mkdir()
            (home / "config.toml").write_text('model_provider = "custom"\n')
            (home / "auth.json").write_text('{"OPENAI_API_KEY":"synthetic-private-key"}')
        settings = queue.EXPERIMENT_SETTINGS
        original = dict(os.environ)
        with mock.patch.dict(os.environ, {"AURIGA_EXPERIMENT_CONFIG": str(settings)}):
            with self.assertRaises(FileNotFoundError): queue.experiment_environment()
            for home in homes:
                settings.write_text(json.dumps({"codex_home": str(home)}))
                environment, selected = queue.experiment_environment()
                self.assertEqual(environment["CODEX_HOME"], str(home))
                self.assertEqual(selected["codex_home"], str(home))
                self.assertNotIn("synthetic-private-key", json.dumps(selected))
                self.assertEqual(os.environ.get("CODEX_HOME"), original.get("CODEX_HOME"))
            settings.write_text(json.dumps({"codex_home": "relative"}))
            with self.assertRaises(ValueError): queue.experiment_environment()
            settings.write_text(json.dumps({"api_key": "synthetic-private-key"}))
            with self.assertRaises(ValueError): queue.experiment_environment()
        self.assertEqual(dict(os.environ), original)

    def test_cli_notifies_frozen_main_thread_after_success_or_failure(self):
        from scripts import experiment_hook
        main_thread = "01a0f6c4-19c6-7811-98f6-2e32bc4d16e5"
        another_thread = "01a0f748-ff2f-7330-980a-0bce7aafae16"
        manifest, output = self.root / "manifest.json", self.root / "queue"
        manifest.write_text(json.dumps({"cases": [self.case]}))
        binding = queue.WAKE_THREAD_FILE
        argv = ["benchmark_queue.py", str(manifest), "--output", str(output)]
        for result in (0, 1):
            for override in (False, True):
                binding.write_text(main_thread)

                def run(*args):
                    binding.write_text(another_thread)
                    return result

                with mock.patch.object(queue.sys, "argv", argv + (["--wake-thread", another_thread] if override else [])), \
                     mock.patch.object(queue, "run_queue", side_effect=run), \
                     mock.patch.object(experiment_hook, "notify_completion", return_value={"status": "sent"}) as notify:
                    self.assertEqual(queue.main(), result)
                    notify.assert_called_once_with(output, another_thread if override else main_thread, codex="codex", remote=None, queue_stopped=True)
        binding.unlink()
        with mock.patch.object(queue.sys, "argv", argv), mock.patch.object(queue, "run_queue", return_value=0), \
             mock.patch.object(experiment_hook, "notify_completion") as notify:
            self.assertEqual(queue.main(), 0)
            notify.assert_not_called()
        for invalid in ("", "123", "not-a-uuid"):
            binding.write_text(invalid)
            with mock.patch.object(queue.sys, "argv", argv), mock.patch.object(queue, "run_queue") as run:
                with self.assertRaises(ValueError): queue.main()
                run.assert_not_called()

    def test_cli_completion_uses_explicit_binary_and_endpoint_without_changing_actor_environment(self):
        from scripts import experiment_hook
        thread = "01a0f6c4-19c6-7811-98f6-2e32bc4d16e5"
        manifest, output = self.root / "manifest.json", self.root / "queue"
        manifest.write_text(json.dumps({"cases": [self.case]}))
        codex, remote = "/native/bin/codex", "unix:///tmp/verified-main-server.sock"
        argv = ["benchmark_queue.py", str(manifest), "--output", str(output),
                "--wake-thread", thread, "--wake-codex", codex, "--wake-remote", remote]
        original_environment = dict(os.environ)
        for result, delivery, expected in ((0, "sent", 0), (1, "sent", 1), (0, "delivery_unknown", 1)):
            with self.subTest(result=result, delivery=delivery), \
                 mock.patch.object(queue.sys, "argv", argv), \
                 mock.patch.object(queue, "run_queue", return_value=result) as run, \
                 mock.patch.object(experiment_hook, "notify_completion", return_value={"status": delivery}) as notify:
                self.assertEqual(queue.main(), expected)
                run.assert_called_once()
                supplied, actual_output, timeout = run.call_args.args
                self.assertEqual(supplied["cases"], [self.case])
                self.assertEqual((actual_output, timeout), (output, 1800))
                self.assertEqual(supplied["auriga_config"]["queue"]["wake_codex"], codex)
                self.assertEqual(supplied["auriga_config"]["queue"]["wake_remote"], remote)
                notify.assert_called_once_with(output, thread, codex=codex, remote=remote, queue_stopped=True)
                self.assertEqual(dict(os.environ), original_environment)

    def test_gpu_selection_and_missing_data(self):
        previous = {"gpus": observation()["gpus"] + observation(free=40000, index=1, uuid="GPU-b")["gpus"]}
        self.assertEqual(queue.select_gpu(previous, previous)["index"], 1)
        busy = observation(utilization=80, free=50000, index=2, uuid="GPU-c")
        mixed = {"gpus": previous["gpus"] + busy["gpus"]}
        self.assertEqual(queue.select_gpu(mixed, mixed)["index"], 1)  # Idle beats more free memory.
        self.assertEqual(queue.select_gpu(busy, busy)["utilization"], 80)  # Authorized fallback.
        lower = observation(utilization=20, free=25000, index=3, uuid="GPU-d")
        both_busy = {"gpus": busy["gpus"] + lower["gpus"]}
        self.assertEqual(queue.select_gpu(both_busy, both_busy)["index"], 3)
        minimum = queue.CASE_MEMORY_MIB + queue.GPU_HEADROOM_MIB
        self.assertEqual(minimum, 11264)
        self.assertIsNotNone(queue.select_gpu(observation(free=minimum), observation(free=minimum)))
        for current in (observation(free=minimum - 1), observation(uuid="GPU-replaced"), {"gpus": []}):
            self.assertIsNone(queue.select_gpu(observation(), current))
        self.assertIsNone(queue.select_gpu(observation(free=minimum - 1), observation()))
        raw = "0, GPU-a, 0, 30000\n1, GPU-b, [N/A], 40000\n2, GPU-c, 0, [N/A]\n"
        with mock.patch.object(queue.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, raw)):
            measured = queue.sample_gpus()
        self.assertEqual([gpu["index"] for gpu in measured["gpus"]], [0])
        self.assertIsNotNone(measured["error"])
        with mock.patch.object(queue.subprocess, "run", side_effect=FileNotFoundError("nvidia-smi")):
            self.assertEqual(queue.sample_gpus()["gpus"], [])
        with mock.patch.object(queue.subprocess, "run", side_effect=[
                subprocess.CompletedProcess([], 0, "GPU-a, 123, 9675\n"),
                subprocess.CompletedProcess([], 0, "0, GPU-a, 0, 35000\n")]) as probe:
            measured = queue.sample_gpus()
        self.assertIn("--query-compute-apps", probe.call_args_list[0].args[0][1])
        self.assertEqual(measured["processes"], [{"uuid": "GPU-a", "pid": 123, "used_mib": 9675}])

    def test_frozen_checkout_from_before_the_package_move_launches_its_own_gateway(self):
        frozen = self.root / "frozen-checkout"
        (frozen / "daedalus").mkdir(parents=True)
        (frozen / "daedalus/gateway.py").write_text("parser.add_argument('--config')\nConfig.load()\n")
        command = queue.command({**self.case, "cwd": str(frozen)}, 3)
        self.assertEqual(command[:5], [queue.sys.executable, "-u", "-m", "daedalus.gateway", "--benchmark"])
        queue.require_config_gateway(frozen)

    def test_command_budgets_and_reuse_rejection(self):
        case = {**self.case, "task_notes": "Release grippers.", "task_notes_source": "https://example.org/public-rules"}
        command = queue.command(case, 3)
        self.assertEqual(command[:5], [queue.sys.executable, "-u", "-m", "auriga.apps.gateway", "--benchmark"])
        for flag, value in (("--gpu", "3"), ("--layout-id", "2")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--max-actions", command)
        self.assertNotIn("--wall-budget-s", command)
        self.assertEqual(command.count("--tag"), 2)
        for update in ({"max_actions": True}, {"wall_budget_s": float("nan")}, {"seed": -1},
                       {"task_notes": "Missing source"}, {"run_dir": "relative"}, {"extra_argument": "--resume"}):
            with self.subTest(update=update), self.assertRaises(ValueError): queue.command({**self.case, **update}, 0)
        output = self.root / "already-used"
        output.mkdir()
        with self.assertRaises(FileExistsError): queue.run_queue({"cases": [self.case]}, output)
        with self.assertRaises(ValueError): queue.run_queue({"cases": [self.case, self.case]}, self.root / "duplicate")
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): queue.run_queue({"cases": [self.case], "max_parallel": value}, self.root / "bad-parallel")
            for name in ("memory_per_case_mib", "gpu_headroom_mib"):
                if name == "gpu_headroom_mib" and value == 0: continue
                with self.assertRaises(ValueError): queue.run_queue({"cases": [self.case], name: value}, self.root / "bad-memory")
        run = Path(self.case["run_dir"])
        run.symlink_to(self.root / "missing-target")
        with self.assertRaisesRegex(ValueError, "already exists"): queue.command(self.case, 0)
        run.unlink()
        run.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"): queue.command(self.case, 0)

    def test_calibrated_default_and_explicit_motion_and_legacy_controls(self):
        default = queue.command(self.case, 0)
        self.assertEqual(default[default.index("--observation-profile") + 1], "rgb_proprio_calibrated_v1")
        self.assertIn("--motion-review", default)
        self.assertEqual(default.count("--enable-wait"), 1)
        disabled_wait = queue.command({**self.case, "enable_wait": False}, 0)
        self.assertEqual(disabled_wait.count("--no-enable-wait"), 1)
        self.assertNotIn("--enable-wait", disabled_wait)
        control = queue.command({**self.case, "motion_review": False}, 0)
        self.assertIn("--no-motion-review", control)
        self.assertEqual(control[control.index("--observation-profile") + 1], "rgb_proprio_calibrated_v1")
        legacy = queue.command({**self.case, "observation_profile": "rgb_proprio_v1"}, 0)
        self.assertIn("--no-motion-review", legacy)
        for update in ({"motion_review": "false"}, {"motion_review": 0}, {"observation_profile": "hidden"}, {"grasp_memory_dir": "relative"}, {"grasp_memory_dir": None}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                queue.command({**self.case, **update}, 0)

    def test_experiment_condition_stays_queue_metadata_without_changing_gateway_command(self):
        expected = queue.command(self.case, 3)
        case = {**self.case, "experiment_condition": "baseline"}
        before = json.dumps(case)
        with mock.patch.object(queue, "cli_overrides", wraps=queue.cli_overrides) as overrides:
            command = queue.command(case, 3)
        self.assertEqual(command, expected)
        self.assertNotIn("--experiment-condition", command)
        self.assertNotIn("experiment_condition", overrides.call_args.args[0])
        effective = queue.configured_manifest({"cases": [case]}, queue.Config())
        self.assertEqual(effective["cases"][0]["experiment_condition"], "baseline")
        self.assertEqual(json.dumps(case), before)
        for invalid in (None, True, 0, [], {}, "", " \t\n"):
            with self.subTest(condition=invalid), self.assertRaisesRegex(ValueError, "experiment_condition"):
                queue.command({**self.case, "experiment_condition": invalid}, 0)

    def test_completion_guidance_modes_are_forwarded_without_changing_budgets(self):
        self.assertNotIn("--completion-guidance", queue.command(self.case, 0))
        for mode in ("none", "initial", "hook", "both"):
            with self.subTest(mode=mode):
                command = queue.command({**self.case, "completion_guidance": mode}, 0)
                self.assertEqual(command[command.index("--completion-guidance") + 1], mode)
                self.assertNotIn("--max-actions", command)
                self.assertNotIn("--wall-budget-s", command)
        for mode in (None, True, "home", ["hook"]):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "completion_guidance"):
                queue.command({**self.case, "completion_guidance": mode}, 0)

    def test_handoff_policy_case_flag_preserves_existing_dual_arm_queue_and_cost_features(self):
        for airborne in (True, False):
            with self.subTest(airborne=airborne):
                case = {**self.case, "airborne_handoff": airborne,
                        "arm_queue": True, "motion_cost_preview": True}
                command = queue.command(case, 0)
                expected = "--airborne-handoff" if airborne else "--no-airborne-handoff"
                excluded = "--no-airborne-handoff" if airborne else "--airborne-handoff"
                self.assertEqual(command.count(expected), 1)
                self.assertNotIn(excluded, command)
                self.assertIn("--arm-queue", command)
                self.assertIn("--motion-cost-preview", command)
                chosen = queue.Config().with_overrides(queue.cli_overrides(case)).validate_launch()
                self.assertIs(chosen.features.airborne_handoff, airborne)
                self.assertTrue(chosen.features.arm_queue)
                self.assertTrue(chosen.features.motion_cost_preview)

    def test_handoff_case_field_is_strict_boolean_and_omission_inherits_config(self):
        command = queue.command(self.case, 0)
        self.assertNotIn("--airborne-handoff", command)
        self.assertNotIn("--no-airborne-handoff", command)
        for invalid in (0, 1, "false", None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "airborne_handoff"):
                queue.command({**self.case, "airborne_handoff": invalid}, 0)

    def test_handoff_preference_case_roundtrip_and_validation(self):
        for preference in ('surface','airborne'):
            case={**self.case,'airborne_handoff':True,'handoff_preference':preference}
            command=queue.command(case,0)
            self.assertEqual(command[command.index('--handoff-preference')+1],preference)
            chosen=queue.Config().with_overrides(queue.cli_overrides(case))
            self.assertTrue(chosen.features.airborne_handoff)
            self.assertEqual(chosen.features.handoff_preference,preference)
        for value in (None,False,'invalid',[]):
            with self.assertRaisesRegex(ValueError,'handoff_preference'):
                queue.command({**self.case,'handoff_preference':value},0)
        with self.assertRaisesRegex(ValueError,'requires_airborne'):
            queue.command({**self.case,'airborne_handoff':False,'handoff_preference':'airborne'},0)

    def test_default_task_notes_can_be_disabled_without_dropping_custom_notes(self):
        for enabled, flag in ((True, "--default-task-notes"), (False, "--no-default-task-notes")):
            with self.subTest(enabled=enabled):
                command = queue.command({**self.case, "default_task_notes": enabled,
                    "task_notes": "Public extra rule.", "task_notes_source": "public-fixture"}, 0)
                self.assertIn(flag, command)
                self.assertEqual(command[command.index("--task-notes") + 1], "Public extra rule.")
                self.assertEqual(command[command.index("--task-notes-source") + 1], "public-fixture")
        self.assertNotIn("--no-default-task-notes", queue.command(self.case, 0))
        for invalid in (None, 0, "false"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "default_task_notes"):
                queue.command({**self.case, "default_task_notes": invalid}, 0)

    def test_launch_recheck_reranks_and_rejects_insufficient_memory(self):
        initial = {"gpus": observation()["gpus"] + observation(index=1, uuid="GPU-b", utilization=40)["gpus"]}
        changed = {"gpus": observation(utilization=90)["gpus"] + observation(index=1, uuid="GPU-b", utilization=0)["gpus"]}
        samples = iter([initial, changed, changed])
        output = self.root / "rerank-queue"
        process = mock.Mock(pid=123, returncode=0)
        process.poll.return_value = 0
        run = Path(self.case["run_dir"])

        memory = str(self.root / "isolated-grasp-memory")
        case_with_memory = {**self.case, "grasp_memory_dir": memory}

        def launch(args, **kwargs):
            self.assertEqual(args[args.index("--gpu") + 1], "1")
            self.assertEqual(kwargs["env"]["AURIGA_GRASP_MEMORY_DIR"], memory)
            run.mkdir()
            (run / "summary.json").write_text('{"status":"completed"}')
            return process

        with mock.patch.object(queue, "sample_gpus", side_effect=lambda: next(samples)), \
             mock.patch.object(queue, "SAMPLE_INTERVAL_S", 0), mock.patch.object(queue.threading, "Event", FastEvent), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
            self.assertEqual(queue.run_queue({"cases": [case_with_memory], "gpu_headroom_mib": 0}, output), 0)
        entry = json.loads((output / "state.json").read_text())["cases"][0]
        self.assertEqual(entry["admission"]["classification"], "idle")
        self.assertEqual(entry["admission"]["gpu"]["index"], 1)

        insufficient = observation(free=queue.CASE_MEMORY_MIB - 1)
        samples = iter([observation(), observation(), insufficient])
        case = {**self.case, "run_dir": str(self.root / "run-no-memory")}
        with mock.patch.object(queue, "sample_gpus", side_effect=lambda: next(samples, insufficient)), \
             mock.patch.object(queue, "SAMPLE_INTERVAL_S", .001), mock.patch.object(queue.subprocess, "Popen") as launch:
            self.assertEqual(queue.run_queue({"cases": [case]}, self.root / "no-memory", wait_timeout_s=.03), 1)
        launch.assert_not_called()
        self.assertEqual(json.loads((self.root / "no-memory/state.json").read_text())["status"], "gpu_wait_timeout")

    def test_parallel_children_share_one_gpu_with_fewer_cases_than_default_capacity(self):
        script = self.root / "child.py"
        script.write_text("""import json, sys, time
from pathlib import Path
run, shared, code = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
run.mkdir()
(shared / run.name).touch()
deadline = time.monotonic() + 5
while len(list(shared.iterdir())) < 2 and time.monotonic() < deadline:
    time.sleep(.01)
time.sleep(.08)
(run / 'summary.json').write_text(json.dumps({'status':'completed', 'overlap':len(list(shared.iterdir())) >= 2}))
sys.exit(code)
""")
        shared = self.root / "started"
        shared.mkdir()
        real_popen, children, allocated = subprocess.Popen, [], []
        samples = observation(free=100000)

        def launch(args, **kwargs):
            run = args[args.index("--run-dir") + 1]
            gpu = int(args[args.index("--gpu") + 1])
            allocated.append(gpu)
            self.assertEqual(len(kwargs["pass_fds"]), 2)
            process = real_popen([queue.sys.executable, str(script), run, str(shared), "7" if not children else "0"], **kwargs)
            children.append(process)
            return process

        cases = [{**self.case, "run_dir": str(self.root / f"run-{i}")} for i in range(3)]
        output = self.root / "parallel"
        with mock.patch.object(queue, "sample_gpus", return_value=samples), \
             mock.patch.object(queue, "SAMPLE_INTERVAL_S", .001), mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
            self.assertEqual(queue.run_queue({"cases": cases}, output, wait_timeout_s=.03), 1)
        state = json.loads((output / "state.json").read_text())
        self.assertEqual(state["status"], "finished")
        self.assertEqual([entry["returncode"] for entry in state["cases"]], [7, 0, 0])
        self.assertTrue(all(entry["summary"]["overlap"] for entry in state["cases"]))
        self.assertEqual(allocated, [0, 0, 0])
        self.assertEqual(state["cases"][1]["admission"]["shadow_reserved_mib"], queue.CASE_MEMORY_MIB)
        events = list(map(json.loads, (output / "events.jsonl").read_text().splitlines()))
        started = [i for i, row in enumerate(events) if row["kind"] == "case_started"]
        exited = [i for i, row in enumerate(events) if row["kind"] == "case_exited"]
        self.assertLess(started[1], exited[0])
        self.assertLess(started[2], exited[0])
        self.assertEqual(state["scheduling"]["max_parallel"], 60)
        self.assertEqual(state["scheduling"]["memory_per_case_mib"], 11264)
        self.assertEqual(state["scheduling"]["gpu_headroom_mib"], 0)

    def test_cross_queue_memory_inherited_lease_and_legacy_exclusion(self):
        capacity = 2 * queue.CASE_MEMORY_MIB + queue.GPU_HEADROOM_MIB
        gpu = observation(free=capacity)["gpus"][0]
        queue.GPU_LOCK_DIR.mkdir()
        legacy = (queue.GPU_LOCK_DIR / "GPU-a.lock").open("a")
        fcntl.flock(legacy, fcntl.LOCK_EX)
        self.addCleanup(legacy.close)
        self.assertEqual(queue.reserve_gpu(gpu)[1]["reason"], "legacy_exclusive_reservation")
        legacy.close()

        with mock.patch.object(queue, "sample_gpus", return_value=observation(free=capacity)):
            first, _ = queue.reserve_gpu(gpu)
            self.addCleanup(first.close)
            second, _ = queue.reserve_gpu(gpu)
            self.addCleanup(second.close)
            refused, reason = queue.reserve_gpu(gpu)
            self.assertIsNone(refused)
            self.assertEqual(reason["shadow_reserved_mib"], 2 * queue.CASE_MEMORY_MIB)
            self.assertEqual(reason["reason"], "insufficient_unreserved_memory")

            child = subprocess.Popen([queue.sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE,
                                     pass_fds=first.filenos())
            self.addCleanup(lambda: child.poll() is None and child.kill())
            first.bind(child.pid)
            first.close()
            self.assertIsNone(queue.reserve_gpu(gpu)[0])
            with (queue.GPU_LOCK_DIR / "GPU-a.lock").open("a") as legacy:
                with self.assertRaises(BlockingIOError): fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child.communicate(timeout=5)
            replacement, _ = queue.reserve_gpu(gpu)
            self.assertIsNotNone(replacement)
            replacement.close()

    @require(linux_proc, "binds processes by /proc/<pid>/stat start ticks")
    def test_memory_attribution_and_utilization_limit_are_cross_queue_and_temporary(self):
        gpu = observation(free=500000)["gpus"][0]
        leases = []
        self.addCleanup(lambda: [lease.close() for lease in leases])
        with mock.patch.object(queue, "sample_gpus", return_value=observation(free=500000)), \
             mock.patch.object(queue.time, "monotonic", return_value=1000):
            for index in range(30):
                lease, admission = queue.reserve_gpu(gpu)
                self.assertIsNotNone(lease)
                leases.append(lease)
                self.assertEqual(admission["recent_launches"], index)
            self.assertEqual(queue.reserve_gpu(gpu)[1]["reason"], "startup_rate_limit")
        with mock.patch.object(queue, "sample_gpus", return_value=observation(free=500000)), \
             mock.patch.object(queue.time, "monotonic", return_value=1091):
            next_lease, admission = queue.reserve_gpu(gpu)
            self.assertIsNotNone(next_lease)
            leases.append(next_lease)
            self.assertEqual(admission["recent_launches"], 0)
        for lease in leases: lease.close()

        capacity = queue.CASE_MEMORY_MIB + queue.GPU_HEADROOM_MIB + 5000
        with mock.patch.object(queue, "sample_gpus", return_value=observation(free=capacity)):
            first, _ = queue.reserve_gpu(gpu)
            leases.append(first)
            first.bind(os.getpid())
            self.assertEqual(queue.reserve_gpu(gpu)[1]["reason"], "insufficient_unreserved_memory")
            measured = {**observation(free=capacity), "processes": [{"uuid": "GPU-a", "pid": os.getpid(), "used_mib": 9675}]}
            with mock.patch.object(queue, "sample_gpus", return_value=measured):
                second, admission = queue.reserve_gpu(gpu)
                leases.append(second)
                self.assertEqual(admission["shadow_reserved_mib"], queue.CASE_MEMORY_MIB - 9675)
                self.assertIsNotNone(second)
        for lease in leases: lease.close()

        for utilization, limit in ((0, 30), (30, 30), (95, 30)):
            with mock.patch.object(queue, "sample_gpus", return_value=observation(free=500000, utilization=utilization)):
                for _ in range(limit):
                    lease, _ = queue.reserve_gpu(gpu)
                    leases.append(lease)
                self.assertEqual(queue.reserve_gpu(gpu)[1]["reason"], "startup_rate_limit")
            for lease in leases: lease.close()
        with mock.patch.object(queue, "sample_gpus", return_value=observation(free=500000)):
            first, admission = queue.reserve_gpu({**gpu, "utilization": 95})
            leases.append(first)
            self.assertEqual(admission["launches_per_interval"], 30)

    def test_simultaneous_queue_processes_cannot_oversubscribe_startup_memory(self):
        script = """import json, sys, time
from pathlib import Path
from scripts import benchmark_queue as q
q.GPU_LOCK_DIR = Path(sys.argv[1])
gpu = dict(index=0, uuid='GPU-a', utilization=0, free_mib=q.CASE_MEMORY_MIB + q.GPU_HEADROOM_MIB)
def sample():
    time.sleep(.03)
    return {'gpus': [gpu], 'processes': None}
q.sample_gpus = sample
print('ready', flush=True)
sys.stdin.readline()
lease, detail = q.reserve_gpu(gpu)
print('held' if lease else detail['reason'], flush=True)
sys.stdin.readline()
if lease: lease.close()
"""
        children = [subprocess.Popen([queue.sys.executable, "-c", script, str(queue.GPU_LOCK_DIR)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) for _ in range(3)]
        try:
            for child in children: self.assertEqual(child.stdout.readline().strip(), "ready")
            for child in children:
                child.stdin.write("go\n")
                child.stdin.flush()
            decisions = [child.stdout.readline().strip() for child in children]
            self.assertEqual(decisions.count("held"), 1)
            self.assertTrue(set(decisions) <= {"held", "concurrent_admission", "insufficient_unreserved_memory"})
        finally:
            for child in children: child.communicate("release\n", timeout=5)

    def test_recorded_free_memory_burst_cannot_expand_periodic_capacity(self):
        # iteration004 GPU 5/6: an unrelated process briefly frees ~60 GiB,
        # then reallocates it. One existing worker has 9958 MiB resident.
        for gpu_index, baseline_free, rising_free, falling_free in (
                (5, 16157, 74009, 13662), (6, 19117, 61097, 15153)):
            with self.subTest(gpu=gpu_index):
                leases = []
                gpu_uuid = f"GPU-recorded-{gpu_index}"

                def observed(free, utilization, timestamp, resident):
                    return {**observation(free=free, utilization=utilization, index=gpu_index, uuid=gpu_uuid),
                            "sample_started_monotonic": timestamp,
                            "processes": [{"uuid": gpu_uuid, "pid": pid, "used_mib": used} for pid, used in resident]}

                try:
                    identity = lambda pid: {"ppid": 1, "start_ticks": pid} if pid in (101, 102) else None
                    initial = observed(100000, 0, 99, [])
                    with mock.patch.object(queue, "process_identity", side_effect=identity), \
                         mock.patch.object(queue.time, "monotonic", return_value=100), \
                         mock.patch.object(queue, "sample_gpus", return_value=initial):
                        old, _ = queue.reserve_gpu(initial["gpus"][0])
                        leases.append(old)
                        old.bind(101)

                    baseline = observed(baseline_free, 99, 1000, [(101, 9958)])
                    with mock.patch.object(queue, "process_identity", side_effect=identity), \
                         mock.patch.object(queue.time, "monotonic", return_value=1001), \
                         mock.patch.object(queue, "sample_gpus", return_value=baseline):
                        new, admission = queue.reserve_gpu(baseline["gpus"][0], baseline=baseline)
                        self.assertIsNotNone(new)
                        leases.append(new)
                        new.bind(102)
                        self.assertEqual(admission["baseline_shadow_reserved_mib"], queue.CASE_MEMORY_MIB - 9958)

                    for free, utilization in ((rising_free, 2), (falling_free, 99)):
                        latest = observed(free, utilization, 1002, [(101, 9958), (102, 670)])
                        with mock.patch.object(queue, "process_identity", side_effect=identity), \
                             mock.patch.object(queue.time, "monotonic", return_value=1003), \
                             mock.patch.object(queue, "sample_gpus", return_value=latest):
                            refused, admission = queue.reserve_gpu(latest["gpus"][0], baseline=baseline)
                        self.assertIsNone(refused)
                        self.assertEqual(admission["reason"], "insufficient_unreserved_memory")
                        self.assertEqual(admission["baseline_effective_free_mib"], baseline_free - (queue.CASE_MEMORY_MIB - 9958) - queue.CASE_MEMORY_MIB)
                        self.assertEqual(admission["shadow_reserved_mib"], (queue.CASE_MEMORY_MIB - 9958) + (queue.CASE_MEMORY_MIB - 670))
                        self.assertEqual(admission["launches_per_interval"], 30)
                finally:
                    for lease in leases: lease.close()

    def test_baseline_accounts_each_snapshot_without_double_counting_or_pid_credit(self):
        leases = []
        self.addCleanup(lambda: [lease.close() for lease in leases])
        gpu_uuid = "GPU-a"

        def observed(free, timestamp, resident):
            return {**observation(free=free), "sample_started_monotonic": timestamp,
                    "processes": [{"uuid": gpu_uuid, "pid": pid, "used_mib": used} for pid, used in resident]}

        identity = lambda pid: {"ppid": 1, "start_ticks": pid} if pid in (101, 102, 103) else None
        with mock.patch.object(queue, "process_identity", side_effect=identity):
            with mock.patch.object(queue.time, "monotonic", return_value=100), \
                 mock.patch.object(queue, "sample_gpus", return_value=observed(100000, 99, [])):
                old, _ = queue.reserve_gpu(observation()["gpus"][0])
                leases.append(old)
                old.bind(101)
            # Same GPU, one old worker grows 6 -> 10 GiB. Free decreases by
            # exactly that growth; both samples have the same admission capacity.
            baseline = observed(32000, 1000, [(101, 6000), (102, 11000)])
            latest = observed(28000, 1001, [(101, 10000)])
            with mock.patch.object(queue.time, "monotonic", return_value=1002), \
                 mock.patch.object(queue, "sample_gpus", return_value=latest):
                new, admission = queue.reserve_gpu(latest["gpus"][0], baseline=baseline)
                leases.append(new)
                new.bind(102)
            self.assertEqual(admission["baseline_effective_free_mib"], 26736)
            self.assertEqual(admission["latest_effective_free_mib"], 26736)

            # New worker is now resident. Its current bytes cannot be credited
            # against baseline free memory, including an old snapshot's reused PID.
            latest = observed(17000, 1003, [(101, 10000), (102, 11000)])
            with mock.patch.object(queue.time, "monotonic", return_value=1004), \
                 mock.patch.object(queue, "sample_gpus", return_value=latest):
                third, admission = queue.reserve_gpu(latest["gpus"][0], baseline=baseline)
                self.assertIsNotNone(third)  # Stable capacity still permits three concurrent cases.
                leases.append(third)
                third.bind(103)
            self.assertEqual(admission["baseline_effective_free_mib"], 15472)
            self.assertEqual(admission["latest_effective_free_mib"], 15472)

            # A different queue sees all live leases through the same files;
            # a transient jump in free memory cannot extend this burst further.
            latest = observed(80000, 1005, [(101, 10000), (102, 11000), (103, 11000)])
            with mock.patch.object(queue.time, "monotonic", return_value=1006), \
                 mock.patch.object(queue, "sample_gpus", return_value=latest):
                refused, admission = queue.reserve_gpu(latest["gpus"][0], baseline=baseline)
            self.assertIsNone(refused)
            self.assertEqual(admission["baseline_effective_free_mib"], 4208)

            # Newly freed stable capacity is reconsidered next period, without
            # a permanent same-card cap or a new memory safety margin.
            latest = observed(80000, 1306, [(101, 10000), (102, 11000), (103, 11000)])
            with mock.patch.object(queue.time, "monotonic", return_value=1307), \
                 mock.patch.object(queue, "sample_gpus", return_value=latest):
                fourth, admission = queue.reserve_gpu(latest["gpus"][0], baseline=latest)
            self.assertIsNotNone(fourth)
            leases.append(fourth)
            self.assertEqual(admission["gpu_headroom_mib"], 0)

    def test_periodic_resampling_while_full_or_without_pending_work(self):
        for pending in (False, True):
            with self.subTest(pending=pending):
                children, clock, samples_at = [], [0.0], []

                class SimulatedTimeEvent(threading.Event):
                    def wait(self, timeout=None):
                        clock[0] += timeout
                        if clock[0] >= 181: self.set()
                        return self.is_set()

                def sample():
                    samples_at.append(clock[0])
                    return observation(free=100000, index=2, uuid="GPU-c")

                def launch(args, **kwargs):
                    self.assertEqual(args[args.index("--gpu") + 1], "2")
                    child = mock.Mock(pid=12345678, returncode=None)
                    child.poll.side_effect = lambda: child.returncode
                    child.wait.side_effect = lambda *args, **kwargs: setattr(child, "returncode", 0)
                    children.append(child)
                    return child

                cases = [{**self.case, "run_dir": str(self.root / f"periodic-{pending}-{index}")} for index in range(2 if pending else 1)]
                output = self.root / f"periodic-{pending}"
                with mock.patch.object(queue, "sample_gpus", side_effect=sample), \
                     mock.patch.object(queue.time, "monotonic", side_effect=lambda: clock[0]), \
                     mock.patch.object(queue.threading, "Event", SimulatedTimeEvent), \
                     mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
                    self.assertEqual(queue.run_queue({"cases": cases, "max_parallel": 1}, output), 1)
                self.assertEqual(samples_at, [0, 0, 0, 90, 180])
                self.assertEqual(len(children), 1)
                self.assertEqual(clock[0], 181)
                children[0].terminate.assert_called_once()

    def test_live_parallelism_increase_decrease_and_invalid_control(self):
        clock, children, launches = [0.0], [], []
        output = self.root / "live-control"

        class ClockEvent(threading.Event):
            def wait(event, timeout=None):
                clock[0] += timeout
                if clock[0] == .5:
                    queue.set_max_parallel(output, 3)
                elif clock[0] == 1:
                    (output / "control.json").write_text('{"max_parallel":true}')
                elif clock[0] == 2:
                    (output / "control.json").write_text('{')
                elif clock[0] == 3:
                    queue.set_max_parallel(output, 1)
                if clock[0] > 12: event.set()
                return event.is_set()

        def launch(args, **kwargs):
            number = len(children)
            launches.append(clock[0])
            child = mock.Mock(pid=12345678 + number, returncode=None)

            def poll():
                if clock[0] >= (4, 6, 8, 9)[number]: child.returncode = 0
                return child.returncode

            child.poll.side_effect = poll
            children.append(child)
            run = Path(args[args.index("--run-dir") + 1])
            run.mkdir()
            (run / "summary.json").write_text('{"status":"completed"}')
            if len(children) == 3:
                # A change during the admission burst must stop its next launch.
                queue.set_max_parallel(output, 1)
            return child

        cases = [{**self.case, "run_dir": str(self.root / f"controlled-{index}")} for index in range(4)]
        with mock.patch.object(queue, "sample_gpus", side_effect=lambda: observation(free=100000)), \
             mock.patch.object(queue.time, "monotonic", side_effect=lambda: clock[0]), \
             mock.patch.object(queue.threading, "Event", ClockEvent), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
            self.assertEqual(queue.run_queue({"cases": cases, "max_parallel": 1}, output), 0)
        self.assertEqual(launches, [0, .5, .5, 8])
        for child in children:
            child.terminate.assert_not_called()
            child.kill.assert_not_called()
        state = json.loads((output / "state.json").read_text())
        self.assertEqual((state["status"], state["scheduling"]["max_parallel"]), ("finished", 1))
        events = list(map(json.loads, (output / "events.jsonl").read_text().splitlines()))
        changed = [row for row in events if row["kind"] == "max_parallel_changed"]
        self.assertEqual([(row["old_max_parallel"], row["new_max_parallel"], row["active_cases"])
                          for row in changed], [(1, 3, 1), (3, 1, 3)])
        invalid = [row for row in events if row["kind"] == "queue_control_invalid"]
        self.assertEqual(len(invalid), 2)
        self.assertTrue(all(row["retained_max_parallel"] == 1 for row in invalid))
        # Experimental conditions stay frozen; state/events are the effective schedule.
        self.assertEqual(json.loads((output / "effective_config.json").read_text())["queue"]["max_parallel"], 1)

    @require(linux_proc, "binds processes by /proc/<pid>/stat start ticks")
    def test_set_max_parallel_cli_requires_live_supported_queue_without_loading_config(self):
        output = self.root / "set-control"
        output.mkdir()
        control = output / "control.json"
        control.write_text('{"max_parallel":1}')
        scheduling = {"control_path": str(control), "queue_pid": os.getpid(),
                      "queue_start_ticks": queue.process_identity(os.getpid())["start_ticks"]}
        state_path = output / "state.json"
        state_path.write_text(json.dumps({"status": "running", "scheduling": scheduling}))
        argv = ["benchmark_queue.py", "--output", str(output), "--set-max-parallel", "4"]
        with mock.patch.object(queue.sys, "argv", argv), \
             mock.patch.object(queue, "queue_config", side_effect=AssertionError("ambient config read")), \
             mock.patch.object(queue, "run_queue", side_effect=AssertionError("queue launch")), \
             mock.patch("builtins.print"):
            self.assertEqual(queue.main(), 0)
        self.assertEqual(json.loads(control.read_text()), {"max_parallel": 4})
        for state in ([], {"status": "running", "scheduling": []},
                      {"status": "running", "scheduling": {}},
                      {"status": "finished", "scheduling": scheduling},
                      {"status": "running", "scheduling": {**scheduling, "queue_start_ticks": -1}}):
            state_path.write_text(json.dumps(state))
            with self.subTest(state=state), self.assertRaises(ValueError):
                queue.set_max_parallel(output, 2)
            self.assertEqual(json.loads(control.read_text()), {"max_parallel": 4})
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                queue.set_max_parallel(output, invalid)

    def test_next_interval_admits_newly_available_gpu_zero(self):
        clock, allocated = [0.0], []

        class SimulatedTimeEvent(threading.Event):
            def wait(self, timeout=None):
                if self.is_set(): return True
                clock[0] += timeout
                if clock[0] > 600: self.set()
                return self.is_set()

        def sample():
            minimum = queue.CASE_MEMORY_MIB + queue.GPU_HEADROOM_MIB
            zero = observation(free=100000 if clock[0] >= 90 else minimum - 1)
            other = observation(index=6, uuid="GPU-g", free=minimum)
            return {"gpus": zero["gpus"] + other["gpus"]}

        def launch(args, **kwargs):
            allocated.append(int(args[args.index("--gpu") + 1]))
            child = mock.Mock(pid=12345678 + len(allocated), returncode=None)
            child.poll.side_effect = lambda: child.returncode
            child.wait.side_effect = lambda *args, **kwargs: setattr(child, "returncode", 0)
            if len(allocated) == 2: os.kill(os.getpid(), signal.SIGTERM)
            return child

        cases = [{**self.case, "run_dir": str(self.root / f"newly-free-{index}")} for index in range(2)]
        with mock.patch.object(queue, "sample_gpus", side_effect=sample), \
             mock.patch.object(queue.time, "monotonic", side_effect=lambda: clock[0]), \
             mock.patch.object(queue.threading, "Event", SimulatedTimeEvent), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
            self.assertEqual(queue.run_queue({"cases": cases}, self.root / "newly-free", wait_timeout_s=1), 1)
        self.assertEqual(allocated, [6, 0])
        self.assertEqual(clock[0], 90)

    def test_completed_case_refills_immediately_without_waiting_for_sibling_or_telemetry(self):
        self.assertEqual(queue.SAMPLE_INTERVAL_S, 90)
        clock, children, launches = [0.0], [], []
        budget = 11264
        samples = {"gpus": observation(free=budget)["gpus"] + observation(free=budget, index=1, uuid="GPU-b")["gpus"]}

        class ClockEvent(threading.Event):
            def wait(self, timeout=None):
                clock[0] += timeout
                if clock[0] > 10: self.set()
                return self.is_set()

        def launch(args, **kwargs):
            number = len(children)
            launches.append(clock[0])
            child = mock.Mock(pid=12345678 + number, returncode=None)
            end = (2, 6, 3)[number]

            def poll():
                if clock[0] >= end: child.returncode = 0
                return child.returncode

            child.poll.side_effect = poll
            children.append(child)
            run = Path(args[args.index("--run-dir") + 1])
            run.mkdir()
            (run / "summary.json").write_text('{"status":"completed"}')
            return child

        cases = [{**self.case, "run_dir": str(self.root / f"refill-{index}")} for index in range(3)]
        output = self.root / "completion-refill"
        with mock.patch.object(queue, "sample_gpus", return_value=samples), \
             mock.patch.object(queue.time, "monotonic", side_effect=lambda: clock[0]), \
             mock.patch.object(queue.threading, "Event", ClockEvent), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
            self.assertEqual(queue.run_queue({"cases": cases, "max_parallel": 2, "memory_per_case_mib": budget}, output), 0)
        self.assertEqual(launches, [0, 0, 2])
        state = json.loads((output / "state.json").read_text())
        self.assertEqual(state["cases"][2]["admission"]["trigger"], "case_completed")
        self.assertEqual(state["cases"][2]["admission"]["case_memory_mib"], budget)
        self.assertEqual(state["scheduling"]["sample_interval_s"], 90)
        self.assertEqual(state["scheduling"]["admission_retry_interval_s"], 5)
        self.assertEqual(clock[0], 6)  # The second initial case was still alive at refill.

    def test_completed_case_refills_even_at_window_launch_limit_and_respects_custom_sample_interval(self):
        clock, children, launches = [0.0], [], []
        budget = 11264
        samples = {"gpus": observation(free=budget * 4)["gpus"]}

        class ClockEvent(threading.Event):
            def wait(self, timeout=None):
                clock[0] += timeout
                if clock[0] > 20: self.set()
                return self.is_set()

        def launch(args, **kwargs):
            number = len(children)
            launches.append(clock[0])
            child = mock.Mock(pid=12345678 + number, returncode=None)
            end = (1.0, 5.0, 2.0)[number]

            def poll():
                if clock[0] >= end: child.returncode = 0
                return child.returncode

            child.poll.side_effect = poll
            children.append(child)
            run = Path(args[args.index("--run-dir") + 1])
            run.mkdir()
            (run / "summary.json").write_text('{"status":"completed"}')
            return child

        cases = [{**self.case, "run_dir": str(self.root / f"window-refill-{index}")} for index in range(3)]
        output = self.root / "window-completion-refill"
        with mock.patch.object(queue, "INITIAL_LAUNCH_LIMIT", 2), \
             mock.patch.object(queue, "sample_gpus", return_value=samples), \
             mock.patch.object(queue.time, "monotonic", side_effect=lambda: clock[0]), \
             mock.patch.object(queue.threading, "Event", ClockEvent), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
            self.assertEqual(queue.run_queue(
                {"cases": cases, "max_parallel": 2, "memory_per_case_mib": budget, "sample_interval_s": 30},
                output,
            ), 0)
        self.assertEqual(launches, [0, 0, 1.0])
        state = json.loads((output / "state.json").read_text())
        self.assertEqual(state["scheduling"]["sample_interval_s"], 30)
        self.assertEqual(state["cases"][2]["admission"]["trigger"], "case_completed")

    def test_external_memory_release_retries_with_fresh_baseline_within_five_seconds(self):
        clock, launches, samples_at = [0.0], [], []
        budget = 11264

        class ClockEvent(threading.Event):
            def wait(self, timeout=None):
                clock[0] += timeout
                if clock[0] > 20: self.set()
                return self.is_set()

        def sample():
            samples_at.append(clock[0])
            return observation(free=budget if clock[0] >= 4 else budget - 1)

        def launch(args, **kwargs):
            launches.append(clock[0])
            run = Path(self.case["run_dir"])
            run.mkdir()
            (run / "summary.json").write_text('{"status":"completed"}')
            return mock.Mock(pid=12345678, returncode=0, poll=mock.Mock(return_value=0))

        output = self.root / "external-release"
        with mock.patch.object(queue, "sample_gpus", side_effect=sample), \
             mock.patch.object(queue.time, "monotonic", side_effect=lambda: clock[0]), \
             mock.patch.object(queue.threading, "Event", ClockEvent), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
            self.assertEqual(queue.run_queue({"cases": [self.case], "memory_per_case_mib": budget}, output, wait_timeout_s=20), 0)
        self.assertEqual(launches, [5])
        self.assertEqual(samples_at, [0, 0, 5, 5, 5])
        state = json.loads((output / "state.json").read_text())
        self.assertEqual(state["cases"][0]["admission"]["trigger"], "capacity_retry")

    def test_completion_refill_rechecks_memory_and_obeys_stop_before_launch(self):
        for stop_during_refill in (False, True):
            with self.subTest(stop=stop_during_refill):
                clock, launches, refill_samples = [0.0], [], [0]
                budget = 11264

                class ClockEvent(threading.Event):
                    def wait(self, timeout=None):
                        if not self.is_set(): clock[0] += timeout
                        if clock[0] > 10: self.set()
                        return self.is_set()

                def sample():
                    free = budget
                    if clock[0] == .5:
                        refill_samples[0] += 1
                        if stop_during_refill: os.kill(os.getpid(), signal.SIGTERM)
                        if refill_samples[0] == 2: free -= 1  # Fresh second sample vetoes admission.
                    return observation(free=free)

                def launch(args, **kwargs):
                    launches.append(clock[0])
                    run = Path(args[args.index("--run-dir") + 1])
                    run.mkdir()
                    (run / "summary.json").write_text('{"status":"completed"}')
                    return mock.Mock(pid=12345678 + len(launches), returncode=0, poll=mock.Mock(return_value=0))

                cases = [{**self.case, "run_dir": str(self.root / f"recheck-{stop_during_refill}-{index}")} for index in range(2)]
                output = self.root / f"recheck-refill-{stop_during_refill}"
                with mock.patch.object(queue, "sample_gpus", side_effect=sample), \
                     mock.patch.object(queue.time, "monotonic", side_effect=lambda: clock[0]), \
                     mock.patch.object(queue.threading, "Event", ClockEvent), \
                     mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
                    result = queue.run_queue({"cases": cases, "max_parallel": 1, "memory_per_case_mib": budget}, output)
                self.assertEqual(result, 1 if stop_during_refill else 0)
                self.assertEqual(launches, [0] if stop_during_refill else [0, 5.5])

    def test_source_change_after_first_case_blocks_the_next_launch(self):
        cases = [{**self.case, "run_dir": str(self.root / f"source-{index}")} for index in range(2)]
        output = self.root / "source-queue"

        def launch(args, **kwargs):
            run = Path(args[args.index("--run-dir") + 1])
            run.mkdir()
            (run / "summary.json").write_text('{"status":"completed"}')
            (self.cwd / "auriga/new_runtime.py").write_text("changed = True\n")
            return mock.Mock(pid=12345678, returncode=0, poll=mock.Mock(return_value=0))

        with mock.patch.object(queue, "sample_gpus", return_value=observation()), \
             mock.patch.object(queue, "SAMPLE_INTERVAL_S", 0), \
             mock.patch.object(queue.threading, "Event", FastEvent), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch) as popen:
            self.assertEqual(queue.run_queue({"cases": cases, "max_parallel": 1}, output), 1)
        self.assertEqual(popen.call_count, 1)
        state = json.loads((output / "state.json").read_text())
        manifest = json.loads((output / "config.json").read_text())["manifest"]
        self.assertIn("queue_source_changed", state["error"])
        self.assertEqual(state["cases"][0]["source_sha256"], manifest["source_bindings"][str(self.cwd)]["sha256"])
        self.assertEqual(state["cases"][1]["status"], "not_started")

    def test_default_parallelism_caps_at_fifteen_and_refills(self):
        allocated, children = [], []
        capacity = 10 * queue.CASE_MEMORY_MIB + queue.GPU_HEADROOM_MIB
        samples = {"gpus": observation(free=capacity)["gpus"] + observation(free=capacity, index=1, uuid="GPU-b")["gpus"]}

        owner = self
        class CompleteThenStop(FastEvent):
            def wait(self, timeout=None):
                if children[0].returncode is None:
                    owner.assertEqual(len(children), 15)
                    run = Path(cases[0]['run_dir'])
                    run.mkdir()
                    (run / 'summary.json').write_text('{"status":"completed"}')
                    children[0].returncode = 0
                else:
                    owner.assertEqual(len(children), 16)
                    self.set()
                return self.is_set()

        def launch(args, **kwargs):
            allocated.append(int(args[args.index("--gpu") + 1]))
            child = mock.Mock(pid=12345678 + len(allocated), returncode=None)
            child.poll.side_effect = lambda: child.returncode
            child.wait.side_effect = lambda *args, **kwargs: setattr(child, "returncode", 0)
            children.append(child)
            return child

        cases = [{**self.case, "run_dir": str(self.root / f"parallel-{index}")} for index in range(16)]
        with mock.patch.object(queue, "sample_gpus", return_value=samples), \
             mock.patch.object(queue, "SAMPLE_INTERVAL_S", 0), \
             mock.patch.object(queue.threading, "Event", CompleteThenStop), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
            self.assertEqual(queue.run_queue({"cases": cases, "max_parallel": 15}, self.root / "parallel"), 1)
        self.assertEqual(allocated, [0] * 10 + [1] * 5 + [0])
        state = json.loads((self.root / "parallel/state.json").read_text())
        self.assertEqual(state["scheduling"]["max_parallel"], 15)
        events = list(map(json.loads, (self.root / 'parallel/events.jsonl').read_text().splitlines()))
        starts = [i for i, row in enumerate(events) if row['kind'] == 'case_started']
        first_exit = next(i for i, row in enumerate(events) if row['kind'] == 'case_exited')
        self.assertLess(starts[14], first_exit)
        self.assertLess(first_exit, starts[15])

    def test_persistent_evidence_failure_still_cleans_all_children(self):
        children, lock_fds = [], []
        samples = {"gpus": observation()["gpus"] + observation(index=1, uuid="GPU-b")["gpus"]}
        original_open = Path.open
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

        def open_file(path, *args, **kwargs):
            if path.name == "events.jsonl" and len(children) == 2:
                raise OSError(28, "No space left on device")
            return original_open(path, *args, **kwargs)

        def launch(args, **kwargs):
            child = mock.Mock(pid=300 + len(children), returncode=None)
            child.poll.side_effect = lambda: child.returncode
            child.wait.side_effect = lambda *args, **kwargs: setattr(child, "returncode", 0)
            children.append(child)
            lock_fds.extend(kwargs["pass_fds"])
            return child

        cases = [{**self.case, "run_dir": str(self.root / f"disk-{i}")} for i in range(3)]
        output = self.root / "disk-failure"
        with mock.patch.object(queue, "sample_gpus", return_value=samples), \
             mock.patch.object(queue, "SAMPLE_INTERVAL_S", 0), mock.patch.object(queue.threading, "Event", FastEvent), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch), mock.patch.object(Path, "open", open_file):
            self.assertEqual(queue.run_queue({"cases": cases}, output), 1)
        self.assertEqual(len(children), 2)
        for child in children:
            child.terminate.assert_called_once()
            child.wait.assert_called_once()
        for fd in lock_fds:
            with self.assertRaises(OSError): os.fstat(fd)
        for sig, handler in handlers.items(): self.assertEqual(signal.getsignal(sig), handler)
        state = json.loads((output / "state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertIn("No space left on device", state["error"])
        self.assertTrue(state["cleanup_errors"])
        self.assertEqual([entry["status"] for entry in state["cases"]], ["exited", "exited", "not_started"])

    def test_signal_stops_all_owned_children_and_preserves_serial_mode(self):
        children = []
        samples = {"gpus": observation()["gpus"] + observation(index=1, uuid="GPU-b")["gpus"]}

        def launch(args, **kwargs):
            process = mock.Mock(pid=100 + len(children), returncode=None)
            process.poll.side_effect = lambda: process.returncode
            process.wait.side_effect = lambda *args, **kwargs: setattr(process, "returncode", 0)
            children.append(process)
            if len(children) == 2: os.kill(os.getpid(), signal.SIGTERM)
            return process

        cases = [{**self.case, "run_dir": str(self.root / f"signal-{i}")} for i in range(3)]
        output = self.root / "signal"
        with mock.patch.object(queue, "sample_gpus", return_value=samples), \
             mock.patch.object(queue, "SAMPLE_INTERVAL_S", 0), mock.patch.object(queue.threading, "Event", FastEvent), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch), mock.patch.object(os, "killpg") as group_kill:
            self.assertEqual(queue.run_queue({"cases": cases}, output), 1)
        self.assertEqual(len(children), 2)
        for process in children:
            process.terminate.assert_called_once()
            process.wait.assert_called_once()
        group_kill.assert_not_called()
        state = json.loads((output / "state.json").read_text())
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual([item["status"] for item in state["cases"]], ["exited", "exited", "not_started"])
        self.assertEqual(state["scheduling"]["max_parallel"], 60)

        def serial_launch(args, **kwargs):
            self.assertTrue(all(child.poll() is not None for child in children))
            run = Path(args[args.index("--run-dir") + 1])
            run.mkdir()
            (run / "summary.json").write_text('{"status":"completed"}')
            child = mock.Mock(pid=200 + len(children), returncode=0)
            child.poll.return_value = 0
            children.append(child)
            return child

        children.clear()
        with mock.patch.object(queue, "sample_gpus", return_value=samples), \
             mock.patch.object(queue, "SAMPLE_INTERVAL_S", 0), mock.patch.object(queue.threading, "Event", FastEvent), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=serial_launch):
            self.assertEqual(queue.run_queue({"cases": cases, "max_parallel": 1}, self.root / "serial"), 0)

    def test_cleanup_escalation_is_bounded_and_never_claims_native_close(self):
        child = mock.Mock(pid=12345678, returncode=None)
        child.poll.side_effect = lambda: child.returncode
        tracker = mock.Mock(errors=[])

        def wait(deadline):
            self.assertIsInstance(deadline, (int, float))
            if tracker.wait.call_count == 3:
                child.returncode = -signal.SIGKILL
                return True
            return False

        tracker.wait.side_effect = wait
        tracker.remaining.return_value = []

        def launch(*args, **kwargs):
            os.kill(os.getpid(), signal.SIGTERM)
            return child

        output = self.root / "forced-cleanup"
        with mock.patch.object(queue, "sample_gpus", return_value=observation()), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch), \
             mock.patch.object(queue, "GatewayCleanup", return_value=tracker):
            self.assertEqual(queue.run_queue({"cases": [self.case]}, output), 1)
        self.assertEqual(tracker.signal.call_args_list, [mock.call(signal.SIGTERM), mock.call(signal.SIGKILL)])
        self.assertEqual(tracker.wait.call_count, 3)
        child.wait.assert_not_called()  # All waits go through the bounded helper.
        state = json.loads((output / "state.json").read_text())
        entry = state["cases"][0]
        self.assertTrue(state["cleanup_stop_unconfirmed"])
        self.assertEqual(entry["status"], "stop_unconfirmed")
        self.assertEqual(entry["queue_cleanup"]["forced_termination"], "SIGKILL")
        self.assertFalse(entry["queue_cleanup"]["native_close_confirmed"])
        self.assertFalse(entry["queue_cleanup"]["process_exit_confirmed"])
        self.assertTrue(entry["queue_cleanup"]["gateway_exit_confirmed"])

    def test_unconfirmed_cleanup_quarantines_memory_until_identity_exit_is_confirmed(self):
        child = mock.Mock(pid=12345678, returncode=None)
        child.poll.return_value = None
        remaining = {"pid": child.pid, "start_ticks": 1234, "fd": None}
        tracker = mock.Mock(errors=[], wait=mock.Mock(return_value=False),
                            remaining=mock.Mock(return_value=[remaining]))

        def launch(*args, **kwargs):
            os.kill(os.getpid(), signal.SIGTERM)
            return child

        sample = observation(free=queue.CASE_MEMORY_MIB)
        output = self.root / "unconfirmed-cleanup"
        with mock.patch.object(queue, "sample_gpus", return_value=sample), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch), \
             mock.patch.object(queue, "GatewayCleanup", return_value=tracker):
            self.assertEqual(queue.run_queue({"cases": [self.case]}, output), 1)
        state = json.loads((output / "state.json").read_text())
        self.assertEqual(state["cases"][0]["status"], "stop_unconfirmed")
        self.assertFalse(state["cases"][0]["queue_cleanup"]["process_exit_confirmed"])
        lease_path, = queue.GPU_LOCK_DIR.glob("*.lease-*.json")
        self.assertTrue(json.loads(lease_path.read_text())["cleanup_stop_unconfirmed"])
        # The queue has closed its descriptors, but subsequent queues must keep
        # this budget reserved. Unreadable /proc is not proof of an exited worker.
        live = {**remaining, "ppid": 1, "state": "D"}
        for observed in (live, PermissionError("synthetic unreadable /proc")):
            patch_snapshot = (mock.patch.object(queue, "process_snapshot", side_effect=observed)
                              if isinstance(observed, Exception)
                              else mock.patch.object(queue, "process_snapshot", return_value=observed))
            with mock.patch.object(queue, "sample_gpus", return_value=sample), patch_snapshot:
                reservation, decision = queue.reserve_gpu(sample["gpus"][0])
            self.assertIsNone(reservation)
            self.assertEqual(decision["reason"], "insufficient_unreserved_memory")
            self.assertTrue(lease_path.exists())
        with mock.patch.object(queue, "sample_gpus", return_value=sample), \
             mock.patch.object(queue, "process_snapshot", return_value=None):
            reservation, _ = queue.reserve_gpu(sample["gpus"][0])
        self.assertIsNotNone(reservation)
        self.assertFalse(lease_path.exists())
        reservation.close()
        for state in ("Z", "X"):
            with mock.patch.object(queue, "process_snapshot", return_value={**live, "state": state}):
                self.assertTrue(queue.identity_exited(remaining))
        with mock.patch.object(queue, "process_snapshot", return_value={**live, "start_ticks": 1235}):
            self.assertTrue(queue.identity_exited(remaining))
        self.assertFalse(queue.quarantine_exited({"remaining_processes": []}))

    def test_natural_gateway_exit_preserves_worker_quarantine_before_releasing_lease(self):
        sample = observation(free=queue.CASE_MEMORY_MIB)
        worker = {"pid": 12345679, "start_ticks": 4321, "process_exit_confirmed": False,
                  "native_close_acknowledged": False}
        run = Path(self.case["run_dir"])
        summary = {"status": "failed", "cleanup_status": "failed", "cleanup": {"robot": worker}}

        def launch(*args, **kwargs):
            run.mkdir()
            (run / "summary.json").write_text(json.dumps(summary))
            return mock.Mock(pid=12345678, returncode=1, poll=mock.Mock(return_value=1))

        output = self.root / "exited-gateway-live-worker"
        cases = [self.case, {**self.case, "run_dir": str(self.root / "must-not-refill")}]
        with mock.patch.object(queue, "sample_gpus", return_value=sample), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch) as popen, \
             mock.patch.object(queue, "process_snapshot", return_value={**worker, "ppid": 1, "state": "D"}):
            self.assertEqual(queue.run_queue({"cases": cases, "max_parallel": 1}, output), 1)
        popen.assert_called_once()
        state = json.loads((output / "state.json").read_text())
        self.assertEqual(state["cases"][1]["status"], "not_started")
        entry = state["cases"][0]
        self.assertEqual(entry["status"], "stop_unconfirmed")
        self.assertTrue(entry["queue_cleanup"]["gateway_exit_confirmed"])
        self.assertFalse(entry["queue_cleanup"]["worker_exit_confirmed"])
        self.assertEqual(entry["queue_cleanup"]["remaining_processes"], [{"pid": worker["pid"], "start_ticks": 4321}])
        lease_path, = queue.GPU_LOCK_DIR.glob("*.lease-*.json")
        self.assertTrue(json.loads(lease_path.read_text())["cleanup_stop_unconfirmed"])
        with mock.patch.object(queue, "sample_gpus", return_value=sample), \
             mock.patch.object(queue, "process_snapshot", return_value=None):
            reservation, _ = queue.reserve_gpu(sample["gpus"][0])
        self.assertIsNotNone(reservation)
        reservation.close()

    def test_abrupt_gateway_exit_without_summary_keeps_unknown_budget_reserved(self):
        sample = observation(free=queue.CASE_MEMORY_MIB)
        child = mock.Mock(pid=12345678, returncode=-signal.SIGKILL,
                          poll=mock.Mock(return_value=-signal.SIGKILL))
        output = self.root / "abrupt-gateway-exit"
        with mock.patch.object(queue, "sample_gpus", return_value=sample), \
             mock.patch.object(queue.subprocess, "Popen", return_value=child):
            self.assertEqual(queue.run_queue({"cases": [self.case]}, output), 1)
        entry = json.loads((output / "state.json").read_text())["cases"][0]
        self.assertEqual(entry["status"], "stop_unconfirmed")
        self.assertIsNone(entry["queue_cleanup"]["worker_exit_confirmed"])
        self.assertIsNone(entry["queue_cleanup"]["native_close_confirmed"])
        with mock.patch.object(queue, "sample_gpus", return_value=sample):
            reservation, decision = queue.reserve_gpu(sample["gpus"][0])
        self.assertIsNone(reservation)
        self.assertEqual(decision["reason"], "insufficient_unreserved_memory")

    @require(linux_proc, "binds processes by /proc/<pid>/stat start ticks")
    def test_pidfd_cleanup_covers_detached_descendants_and_leaves_other_jobs_alive(self):
        # Owned CPU-only fixtures; no model, simulator, GPU, or process-group kill.
        child_code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
        parent_code = ("import subprocess,sys,signal,time; "
                       "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                       f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True); "
                       "print(p.pid,flush=True); time.sleep(30)")
        parent = subprocess.Popen([queue.sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True)
        unrelated = subprocess.Popen([queue.sys.executable, "-c", "import time; time.sleep(30)"])
        tracker = None
        try:
            child_pid = int(parent.stdout.readline())
            tracker = queue.GatewayCleanup(parent)
            self.assertIn(parent.pid, tracker.targets)
            self.assertIn(child_pid, tracker.targets)
            self.assertNotIn(unrelated.pid, tracker.targets)
            self.assertFalse(tracker.wait(queue.time.monotonic()))
            tracker.signal(signal.SIGKILL)
            self.assertTrue(tracker.wait(queue.time.monotonic() + 3))
            self.assertEqual(tracker.remaining(), [])
            self.assertIsNone(unrelated.poll())
        finally:
            if tracker: tracker.close()
            if parent.poll() is None: parent.kill()
            parent.wait(timeout=3)
            parent.stdout.close()
            unrelated.terminate()
            unrelated.wait(timeout=3)

    def test_capture_rejects_pid_reuse_before_signalling(self):
        root = {"pid": 10000001, "ppid": os.getpid(), "start_ticks": 100, "state": "S"}
        child = {"pid": 10000002, "ppid": root["pid"], "start_ticks": 101, "state": "S"}
        calls = {child["pid"]: 0}

        def snapshot(pid):
            if pid == root["pid"]: return root
            calls[pid] += 1
            return child if calls[pid] == 1 else {**child, "start_ticks": 999}

        process = mock.Mock(pid=root["pid"], returncode=None)
        with mock.patch.object(queue, "process_identity", return_value=root), \
             mock.patch.object(queue, "process_snapshot", side_effect=snapshot), \
             mock.patch.object(Path, "glob", return_value=[Path(f"/proc/{child['pid']}/stat")]), \
             mock.patch.object(queue, "pidfd_open", side_effect=lambda pid: pid + 1), \
             mock.patch.object(os, "close") as close:
            tracker = queue.GatewayCleanup(process)
            self.assertNotIn(child["pid"], tracker.targets)
            self.assertIn(root["pid"], tracker.targets)
            close.assert_called_once_with(child["pid"] + 1)
            tracker.close()

    def oom_summary(self, **changes):
        return {"status": "failed", "primary_outcome": "unscored_infrastructure",
                "native_episode": {"ended": False}, "task_success": None, "normalized_score": None,
                "infrastructure_error": "robot_gpu_out_of_memory: native GPU allocation failed; robot status unknown; do not replay action",
                "cleanup_status": "forced_termination", "cleanup": {"robot": {
                    "process_exit_confirmed": True, "native_close_acknowledged": False,
                    "forced_termination": "SIGTERM", "returncode": -15}}, **changes}

    def run_retry_scenario(self, summaries, *, name="retry", policy=None, sample=None, stop_at=None,
                           stop_on_launch=False, manifest=None, recovery=None, wait_timeout_s=1800):
        clock, launches, snapshots = [0.0], [], []
        output = self.root / name
        case = {**self.case, "run_dir": str(self.root / (name + "-run")), "completion_guidance": "both"}
        manifest = manifest or {"cases": [case], "oom_retry": policy or {}}

        class ClockEvent(threading.Event):
            def wait(event, timeout=None):
                snapshots.append(json.loads((output / "state.json").read_text()))
                clock[0] += max(.001, timeout)
                if stop_at is not None and clock[0] >= stop_at: event.set()
                if clock[0] > 1000: raise AssertionError("retry scheduler did not terminate")
                return event.is_set()

        def launch(args, **kwargs):
            run = Path(args[args.index("--run-dir") + 1])
            summary, code = summaries[len(launches)]
            launches.append({"at": clock[0], "args": args, "run": run})
            run.mkdir()
            if summary is not None: (run / "summary.json").write_text(json.dumps(summary))
            if stop_on_launch: os.kill(os.getpid(), signal.SIGTERM)
            return mock.Mock(pid=12345670 + len(launches), returncode=code, poll=mock.Mock(return_value=code))

        with mock.patch.object(queue, "sample_gpus", side_effect=lambda: sample(clock[0], len(launches)) if sample else observation()), \
             mock.patch.object(queue.time, "monotonic", side_effect=lambda: clock[0]), \
             mock.patch.object(queue.time, "time", side_effect=lambda: 1000 + clock[0]), \
             mock.patch.object(queue.threading, "Event", ClockEvent), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
            result = queue.run_queue(manifest, output, wait_timeout_s=wait_timeout_s, recovery=recovery)
        return result, json.loads((output / "state.json").read_text()), launches, snapshots, output

    def test_oom_detection_uses_native_evidence_and_excludes_terminal_or_actor_text(self):
        run = self.root / "evidence-run"
        (run / "private" / "robot").mkdir(parents=True)
        log = run / "private" / "robot" / "robot-worker.log"
        entry = {"case": {**self.case, "run_dir": str(run)}, "summary": self.oom_summary()}
        evidence = queue.oom_evidence(entry)
        self.assertEqual(evidence["field"], "infrastructure_error")
        self.assertEqual(len(evidence["sha256"]), 64)
        for signature in ("ERROR_OUT_OF_DEVICE_MEMORY", "Out of GPU memory allocating resource '/private/scene.usd'",
                          "CUDA error: out of memory", "PxgCudaDeviceMemoryAllocator failed to allocate memory 268435456 bytes! Result = 2",
                          "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 256 MiB"):
            with self.subTest(signature=signature):
                log.write_text("native startup\n" + signature + "\n")
                entry["summary"] = self.oom_summary(infrastructure_error="bootstrap_failed: downstream NoneType")
                evidence = queue.oom_evidence(entry)
                self.assertEqual(evidence["source"], str(log))
                self.assertEqual(evidence["byte_offset"], len("native startup\n"))
        for change in ({"native_episode": {"ended": True}}, {"primary_outcome": "scored_failure"},
                       {"primary_outcome": "scored_success"}, {"termination_reason": "native_step_limit"},
                       {"termination_reason": "model_request_limit"}):
            entry["summary"] = self.oom_summary(**change)
            self.assertIsNone(queue.oom_evidence(entry))
        log.write_text('AURIGA_JSON {"text":"CUDA error: out of memory"}\n')
        entry["summary"] = self.oom_summary(infrastructure_error="codex_turn_failed: model said CUDA error: out of memory")
        self.assertIsNone(queue.oom_evidence(entry))
        log.write_text("room for GPU memory; failed to open display\n")
        self.assertIsNone(queue.oom_evidence(entry))
        # Detection bounds I/O, and never scans historical log prefixes repeatedly.
        log.write_bytes(b"CUDA error: out of memory\n" + b"x" * queue_retry.LOG_TAIL_BYTES)
        self.assertIsNone(queue.oom_evidence(entry))

    def test_oom_retry_has_new_episode_directory_delay_identity_and_fresh_gpu_admission(self):
        success = {"status": "completed", "primary_outcome": "scored_success", "native_episode": {"ended": True}}
        def sample(now, count):
            return observation(index=0 if not count else 1, uuid="GPU-a" if not count else "GPU-b")
        result, state, launches, snapshots, output = self.run_retry_scenario([(self.oom_summary(), 1), (success, 0)], sample=sample)
        self.assertEqual(result, 0)
        first, second = state["cases"]
        self.assertEqual(first["retry_status"], "scheduled")
        self.assertEqual(second["attempt_number"], 2)
        self.assertEqual(first["logical_task_id"], second["logical_task_id"])
        self.assertEqual(second["previous_attempt_run_dir"], first["case"]["run_dir"])
        self.assertEqual(second["case"]["run_dir"], first["case"]["run_dir"] + ".oom-retry-01")
        self.assertGreaterEqual(launches[1]["at"] - launches[0]["at"], 60)
        self.assertEqual([entry["gpu"]["index"] for entry in state["cases"]], [0, 1])
        for entry in state["cases"]:
            self.assertIn("--completion-guidance", entry["command"])
            self.assertNotIn("--resume", entry["command"])
            self.assertNotIn("--max-actions", entry["command"])
        self.assertTrue(any(len(item["cases"]) == 2 and item["cases"][1]["retry_status"] == "pending" for item in snapshots))
        task, = state["logical_tasks"].values()
        self.assertEqual(task["first_attempt_outcome"], "unscored_infrastructure")
        self.assertEqual(task["final_attempt_outcome"], "scored_success")
        self.assertTrue(task["had_oom"])
        self.assertEqual(task["retries_started"], 1)
        self.assertEqual(json.loads((launches[0]["run"] / "summary.json").read_text()), self.oom_summary())
        self.assertEqual(len(list(output.glob("case-*.log"))), 2)

    def test_oom_retry_exhaustion_reports_all_attempts_and_exponential_backoff(self):
        result, state, launches, _, _ = self.run_retry_scenario([(self.oom_summary(), 1)] * 3)
        self.assertEqual(result, 1)
        self.assertEqual(len(launches), 3)
        self.assertGreaterEqual(launches[1]["at"] - launches[0]["at"], 60)
        self.assertGreaterEqual(launches[2]["at"] - launches[1]["at"], 120)
        self.assertEqual(state["cases"][-1]["retry_status"], "exhausted")
        task, = state["logical_tasks"].values()
        self.assertEqual(task["status"], "exhausted")
        self.assertEqual(task["retries_started"], 2)

    def test_non_oom_failures_unknown_kills_and_unconfirmed_cleanup_do_not_retry(self):
        samples = [(self.oom_summary(infrastructure_error="robot worker disconnected (exit=137)"), 137),
                   (self.oom_summary(infrastructure_error="codex_turn_failed: CUDA error: out of memory"), 1),
                   (self.oom_summary(primary_outcome="scored_failure", termination_reason="native_step_limit"), 1),
                   (self.oom_summary(native_episode={"ended": True}), 1), (None, 137), (None, -9),
                   (self.oom_summary(cleanup={}, cleanup_status="unknown"), 1)]
        for number, summary in enumerate(samples):
            with self.subTest(number=number):
                result, state, launches, _, _ = self.run_retry_scenario([summary], name=f"no-retry-{number}")
                self.assertEqual(result, 1)
                self.assertEqual(len(launches), 1)
                if number >= 4:
                    self.assertEqual(state["cases"][0]["status"], "stop_unconfirmed")
                if number == 6:
                    self.assertEqual(state["cases"][0]["retry_status"], "blocked_cleanup")
                # This test runs multiple isolated queues; unknown leases must
                # remain quarantined, so isolate the next synthetic GPU too.
                for lease in queue.GPU_LOCK_DIR.glob("*.lease-*.json"): lease.unlink()

    def test_stop_cancels_pending_or_just_finished_oom_retry(self):
        for stop_on_launch in (False, True):
            with self.subTest(stop_on_launch=stop_on_launch):
                result, state, launches, _, _ = self.run_retry_scenario([(self.oom_summary(), 1)],
                    name=f"stop-{stop_on_launch}", stop_at=1, stop_on_launch=stop_on_launch)
                self.assertEqual(result, 1)
                self.assertEqual(len(launches), 1)
                self.assertEqual(state["status"], "interrupted")
                self.assertEqual(state["cases"][-1]["retry_status"], "cancelled")

    def test_retry_backoff_does_not_consume_gpu_wait_timeout_and_memory_is_rechecked(self):
        success = {"status": "completed", "primary_outcome": "scored_success", "native_episode": {"ended": True}}
        sampled = []
        def sample(now, count):
            sampled.append(now)
            return observation(free=0 if count and now < 71 else 30000)
        result, state, launches, _, _ = self.run_retry_scenario([(self.oom_summary(), 1), (success, 0)],
            sample=sample, wait_timeout_s=20)
        self.assertEqual(result, 0)
        self.assertGreaterEqual(launches[1]["at"], 71)
        self.assertTrue(any(60 <= now < 71 for now in sampled))
        self.assertEqual(state["cases"][1]["admission"]["trigger"], "capacity_retry")

    def test_terminal_queue_oom_recovery_preserves_history_deadline_and_excludes_other_cases(self):
        _, original, _, _, source = self.run_retry_scenario([(self.oom_summary(), 1)], stop_at=1)
        raw = (source / "state.json").read_bytes()
        ordinary = {"case": {**self.case, "run_dir": str(self.root / "ordinary")}, "status": "exited",
                    "returncode": 0, "summary": {"status": "completed", "primary_outcome": "scored_failure"},
                    "queue_cleanup": {"gateway_exit_confirmed": True, "worker_exit_confirmed": True}}
        Path(ordinary["case"]["run_dir"]).mkdir()
        (Path(ordinary["case"]["run_dir"]) / "summary.json").write_text(json.dumps(ordinary["summary"]))
        original["cases"].append(ordinary)
        (source / "state.json").write_text(json.dumps(original))
        raw = (source / "state.json").read_bytes()
        manifest, recovery = queue.recover_oom(source)
        self.assertEqual(len(manifest["cases"]), 1)
        self.assertEqual(recovery["entries"][0]["retry_not_before"], original["cases"][0]["retry_not_before"])
        self.assertEqual(recovery["entries"][0]["logical_task_id"], original["cases"][0]["logical_task_id"])
        self.assertEqual(len(recovery["prior_attempts"][0]["source_summary_sha256"]), 64)
        success = {"status": "completed", "primary_outcome": "scored_success", "native_episode": {"ended": True}}
        result, state, launches, _, destination = self.run_retry_scenario([(success, 0)], name="recovered",
            manifest=manifest, recovery=recovery)
        self.assertEqual(result, 0)
        self.assertEqual(len(launches), 1)
        self.assertEqual(len(state["prior_attempts"]), 1)
        task, = state["logical_tasks"].values()
        self.assertEqual(len(task["attempts"]), 2)
        self.assertEqual(task["first_attempt_outcome"], "unscored_infrastructure")
        self.assertEqual((source / "state.json").read_bytes(), raw)
        with self.assertRaisesRegex(ValueError, "No cleanup-confirmed OOM"):
            queue.recover_oom(destination)
        with self.assertRaisesRegex(ValueError, "already exists"):
            queue.run_queue(manifest, self.root / "duplicate-recovery", recovery=recovery)
        # In-place/crashed restart is unsupported, and cannot silently take ownership.
        original["status"] = "running"
        (source / "state.json").write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "terminal queue"):
            queue.recover_oom(source)

    def test_recovery_replays_frozen_command_through_subsequent_retries(self):
        _, state, _, _, source = self.run_retry_scenario([(self.oom_summary(), 1)], name="legacy-source", stop_at=1)
        archived = json.loads((source / 'config.json').read_text())
        archived['manifest'].pop('max_parallel')
        archived['manifest']['auriga_config']['queue']['max_parallel'] = None
        (source / 'config.json').write_text(json.dumps(archived))
        snapshot = json.dumps(archived['manifest']['auriga_config'])
        (source / 'effective_config.json').write_text(snapshot)
        original = state["cases"][0]
        original["case"]["enable_wait"] = False
        args = original["command"]
        args[args.index("--observation-profile") + 1] = "rgb_proprio_v1"
        args[:] = [item for item in args if item not in {"--motion-review", "--no-motion-review", "--no-enable-wait"}]
        (source / "state.json").write_text(json.dumps(state))
        manifest, recovery = queue.recover_oom(source)
        success = {"status": "completed", "primary_outcome": "scored_success", "native_episode": {"ended": True}}
        with mock.patch.object(queue, 'queue_config', side_effect=AssertionError('ambient config read')), \
             mock.patch.object(queue, 'configured_manifest', side_effect=AssertionError('new defaults applied')):
            result, recovered, launches, _, _ = self.run_retry_scenario([(self.oom_summary(), 1), (success, 0)],
                name="legacy-recovered", manifest=manifest, recovery=recovery)
        self.assertEqual(result, 0)
        self.assertEqual(recovered['scheduling']['max_parallel'], 1)
        self.assertEqual((source / 'effective_config.json').read_text(), snapshot)
        self.assertEqual(len(launches), 2)
        for entry in recovered["cases"]:
            expected = list(args)
            expected[expected.index("--run-dir") + 1] = entry["case"]["run_dir"]
            expected[expected.index("--gpu") + 1] = str(entry["gpu"]["index"])
            self.assertEqual(entry["command"], expected)
            self.assertEqual(entry["case"]["cwd"], original["case"]["cwd"])
        del original["command"]
        (source / "state.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "original gateway command"):
            queue.recover_oom(source)

    def test_recovery_revalidates_cached_oom_and_summary_bytes(self):
        _, state, launches, _, source = self.run_retry_scenario([(self.oom_summary(), 1)], stop_at=1)
        summary_path = launches[0]["run"] / "summary.json"
        entry = state["cases"][0]
        entry["summary"]["native_episode"] = {"ended": True}
        self.assertIsNotNone(entry["oom_evidence"])
        (source / "state.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "differs from terminal queue state"):
            queue.recover_oom(source)
        summary_path.write_text(json.dumps(entry["summary"]))
        with self.assertRaisesRegex(ValueError, "No cleanup-confirmed OOM"):
            queue.recover_oom(source)

    def test_recovery_does_not_use_incoming_retry_deadline_for_cancelled_new_failure(self):
        _, state, _, _, source = self.run_retry_scenario([(self.oom_summary(), 1)] * 2, stop_at=61)
        entry = state["cases"][1]
        entry.update(retry_status="cancelled", retry_not_before=1001)
        state["cases"] = state["cases"][:2]
        (source / "state.json").write_text(json.dumps(state))
        with mock.patch.object(queue_retry.time, "time", return_value=2000):
            _, recovery = queue.recover_oom(source)
        self.assertEqual(recovery["entries"][0]["retry_not_before"], 2120)
        self.assertEqual(recovery["entries"][0]["attempt_number"], 3)

    def test_retry_binds_original_experiment_settings_without_exposing_credentials(self):
        homes = [self.root / name for name in ("original-home", "new-home")]
        for home in homes:
            home.mkdir()
            (home / "config.toml").write_text("model_provider = 'fixture'\n")
            (home / "auth.json").write_text('{"OPENAI_API_KEY":"must-not-log-fixture"}')
        original_settings = self.root / "original-settings.json"
        current_settings = self.root / "current-settings.json"
        original_settings.write_text(json.dumps({"codex_home": str(homes[0])}))
        current_settings.write_text(json.dumps({"codex_home": str(homes[1])}))
        recorded = {"configured": True, "settings_file": str(original_settings), "codex_home": str(homes[0])}
        with mock.patch.dict(os.environ, {"AURIGA_EXPERIMENT_CONFIG": str(current_settings)}):
            environment, selected = queue.experiment_environment(recorded)
            self.assertEqual(environment["CODEX_HOME"], str(homes[0]))
            self.assertEqual(environment["AURIGA_EXPERIMENT_CONFIG"], str(original_settings))
            self.assertEqual(selected["codex_home"], str(homes[0]))
            self.assertNotIn("must-not-log-fixture", json.dumps(selected))
            self.assertEqual(os.environ["AURIGA_EXPERIMENT_CONFIG"], str(current_settings))
            original_settings.write_text(json.dumps({"codex_home": str(homes[1])}))
            with self.assertRaisesRegex(ValueError, "settings differ"):
                queue.experiment_environment(recorded)
            _, selected = queue.experiment_environment({"configured": False})
            self.assertEqual(selected["recovery_binding"], "inherited_environment_unrecorded")

    def test_oom_retry_configuration_and_reserved_directories_are_validated(self):
        for policy in ({"max_retries": True}, {"max_retries": -1}, {"max_retries": 21},
                       {"initial_delay_s": 0}, {"max_delay_s": float("nan")}, {"unexpected": 2}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                queue.run_queue({"cases": [self.case], "oom_retry": policy}, self.root / "bad-policy")
        (self.root / "run-1.oom-retry-01").mkdir()
        with self.assertRaisesRegex(ValueError, "retry run_dir already exists"):
            queue.run_queue({"cases": [self.case]}, self.root / "collision")
        result, state, launches, _, _ = self.run_retry_scenario([(self.oom_summary(), 1)], policy={"max_retries": 0})
        self.assertEqual(result, 1)
        self.assertEqual(len(launches), 1)
        self.assertEqual(state["cases"][0]["retry_status"], "exhausted")


if __name__ == "__main__": unittest.main()
