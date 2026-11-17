"""Whole-subtask WALL-OSS delegation through Harness: loop, stop conditions, receipt and boundary.

CPU only: a deterministic robot fixture executes EEChunkActions and writes policy
captures; a fake policy bridge returns WallX-shaped chunks. No simulator, no
policy server, no model.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid

import numpy as np

from test_harness import FixtureRobot
from auriga.core.config import Config
from auriga.core.storage.store import Store
from auriga.runtime.harness import Harness
from auriga.runtime.policy.delegate_policy import PolicyBridgeError
from auriga.runtime.tools.definitions import public_definitions

IDENTITY = dict(run_id="delegation-fixture", episode_id="ep", env_instance_id="env", env_generation=1)
INSTRUCTION = "Stack the bowls on the plate."
HASH = "sha256: " + "a" * 64


def policy_row(step):
    return dict(left_ee_pose=[0.3 + 0.001 * step, 0.2, 0.9, 1.0, 0.0, 0.0, 0.0],
                right_ee_pose=[0.3, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0], left_ee_joint_state=[0.25], right_ee_joint_state=[1.0])


class DelegateRobot(FixtureRobot):
    """Fixture backend: EE chunks advance a native counter and return policy captures."""

    def __init__(self, root, *, end_at=None, chunk_status="completed", chunk_reason=None, step_limit=1000):
        super().__init__(root)
        self.root, self.sim_step, self.native = Path(root), 0, 0
        self.end_at, self.chunk_status, self.chunk_reason, self.step_limit = end_at, chunk_status, chunk_reason, step_limit
        self.received, self.captures = [], []

    def start(self):
        return dict(super().start(), action_budget=dict(native_step_limit=self.step_limit))

    def observe(self):
        value = super().observe()
        value.update(sim_step=self.sim_step, instruction=f"{INSTRUCTION} {HASH}")
        return value

    def _capture(self, frames):
        path = self.root / f"capture-{uuid.uuid4().hex}.npz"
        np.savez(path, head=np.zeros((frames, 2, 2, 3), np.uint8))
        self.captures.append(path)
        return dict(path=str(path), frames=frames, sim_steps=list(range(frames)), final_sim_step=self.sim_step,
                    instruction=INSTRUCTION)

    def policy_observation(self):
        return self._capture(1)

    def execute(self, action, progress_callback=None, cancel_flag=None):
        self.received.append(action)
        rows = len(action["actions"])
        executed = rows if self.end_at is None else max(0, min(rows, self.end_at - self.native))
        self.sim_step += executed
        self.native += executed
        ended = self.end_at is not None and self.native >= self.end_at
        status, reason = ("stopped", "episode_ended") if ended else (self.chunk_status, self.chunk_reason)
        receipt = dict(status=status, reason=reason, executed_steps=executed, executed_native_steps=executed,
                       completed_native_steps=executed, terminal_observation=self.observe(),
                       episode_terminal=dict(ended=ended, success=True if ended else None,
                                             reason="native_success" if ended else None,
                                             native_action_count=self.native, native_step_limit=self.step_limit))
        if action["type"] == "EEChunkAction":
            receipt["ee_chunk_ik"] = dict(left=dict(failed_steps=1, unknown_steps=0, max_consecutive_failed_steps=1),
                                          right=dict(failed_steps=0, unknown_steps=0, max_consecutive_failed_steps=0))
            if executed:
                receipt["policy_capture"] = self._capture(executed)
        return receipt


class FakePolicy:
    def __init__(self, fail_on=None, timeout=True, fail_reset=False):
        self.resets, self.calls, self.fail_on, self.timeout, self.fail_reset = [], [], fail_on, timeout, fail_reset
        self.metadata = dict(status="ready", server="fake")

    def reset(self, *, task_id, context_key):
        if self.fail_reset:
            raise PolicyBridgeError("policy_server_error:reset_failed", timeout=False)
        self.resets.append((task_id, context_key))

    def infer(self, capture_paths, instruction):
        assert all(Path(path).exists() for path in capture_paths), "capture consumed before inference"
        self.calls.append((list(capture_paths), instruction))
        if self.fail_on == len(self.calls):
            raise PolicyBridgeError("policy_client_response_timeout" if self.timeout else "boom", timeout=self.timeout)
        return dict(status="inference_completed", actions=[policy_row(step) for step in range(32)], chunk_length=32,
                    sim_frame=16 * (len(self.calls) - 1), inference_seconds=0.01, memory_frame_ids=[])


def temporary(case):
    """Removed after the Store closes (cleanups run last-in first-out); Windows locks open files."""
    directory = tempfile.mkdtemp()
    case.addCleanup(shutil.rmtree, directory, ignore_errors=True)
    return directory


class DelegationFixture(unittest.TestCase):
    def harness(self, directory, policy=None, wall_oss=None, robot_class=None, **robot_options):
        store = Store(Path(directory) / "store", dict(IDENTITY))
        self.addCleanup(store.close)
        robot = (robot_class or DelegateRobot)(directory, **robot_options)
        harness = Harness(store, robot)
        if wall_oss:
            harness.config = harness.config.with_overrides(dict(wall_oss=wall_oss))
        harness.start()
        harness.delegate_policy = policy
        return harness, robot

    def delegate(self, harness, call="call-1", **args):
        args.setdefault("observation_id", harness.latest["observation_id"])
        return harness.invoke("robot_policy_delegate", args, dict(tool_call_id=call), role="actor")


class DelegationCheck(DelegationFixture):

    def test_tool_is_gated_by_wall_oss_and_is_actor_only(self):
        directory = temporary(self)
        if True:
            harness, _ = self.harness(directory)
            registry = harness.tool_registry
            self.assertNotIn("robot_policy_delegate", {item["name"] for item in registry.catalog(harness)})
            with self.assertRaisesRegex(ValueError, "wall_oss_not_enabled_for_run"):
                self.delegate(harness)
            harness.delegate_policy = FakePolicy()
            self.assertIn("robot_policy_delegate", {item["name"] for item in registry.catalog(harness)})
            with self.assertRaises(ValueError):
                registry.spec("robot_policy_delegate", role="program")
            self.assertNotIn("robot_policy_delegate", {item["name"] for item in registry.catalog(harness, role="program")})

    def test_closed_loop_runs_to_budget_and_returns_compact_receipt(self):
        directory = temporary(self)
        if True:
            policy = FakePolicy()
            harness, robot = self.harness(directory, policy)
            start = harness.latest["observation_id"]
            response = self.delegate(harness, max_policy_steps=40, subtask_note="Place the second bowl.")
            self.assertIsNone(response.contract_error)
            self.assertEqual(response.kind, "delegation")
            receipt = response.data["receipt"]
            self.assertEqual((receipt["status"], receipt["stop_reason"]), ("finished", "max_policy_steps"))
            self.assertEqual((receipt["policy_steps_used"], receipt["inference_calls"], receipt["replan_steps"]), (40, 3, 16))
            self.assertEqual([len(action["actions"]) for action in robot.received], [16, 16, 8])
            self.assertEqual(receipt["chunk_action_ids"], ["delegation-call-1-chunk-000", "delegation-call-1-chunk-001",
                                                           "delegation-call-1-chunk-002"])
            self.assertEqual(receipt["final_observation_id"], harness.latest["observation_id"])
            self.assertTrue(response.data["result_observation_is_current"])
            self.assertNotEqual(receipt["final_observation_id"], start)
            self.assertEqual(receipt["ik_failed_steps"], dict(left=3, right=0))
            self.assertEqual(receipt["last_policy_action"]["left"]["position_m"], policy_row(7)["left_ee_pose"][:3])
            self.assertEqual(receipt["last_policy_action"]["left"]["gripper_command"], 0.25)
            self.assertEqual(receipt["subtask_note"], "Place the second bowl.")
            self.assertEqual(receipt["action_budget"]["native_steps_used"], 40)
            # Every chunk went through Harness.execute as an ordinary recorded action.
            for identifier in receipt["chunk_action_ids"]:
                request = harness.store.read("action_request", identifier)
                self.assertEqual(request["type"], "EEChunkAction")
                self.assertEqual(request["policy_provenance"]["generator"], "wall-oss-1.0-lite-robodojo")
                self.assertEqual(harness.store.read("action_receipt", identifier)["status"], "completed")
            self.assertEqual(policy.resets[0][0], "stack_bowls")
            # The capture of one chunk is the input of the next inference; all are deleted afterwards.
            self.assertEqual([len(paths) for paths, _ in policy.calls], [1, 1, 1])
            self.assertFalse(any(path.exists() for path in robot.captures))

    def test_information_boundary_of_the_actor_reply(self):
        directory = temporary(self)
        if True:
            policy = FakePolicy()
            harness, robot = self.harness(directory, policy, end_at=20)
            response = self.delegate(harness, max_policy_steps=64, subtask_note="secret plan note")
            receipt = response.data["receipt"]
            self.assertEqual((receipt["status"], receipt["stop_reason"]), ("finished", "episode_ended"))
            self.assertEqual(receipt["policy_steps_used"], 20)
            # What the actor model receives after the shared presentation boundary.
            text = json.dumps(harness.actor_output.present(response, tool="robot_policy_delegate"))
            for forbidden in ("native_success", "episode_terminal", ".npz", "policy_capture", "kfmem", "memory_frame_ids",
                              "prediction_ref", "server", HASH, str(Path(directory))):
                self.assertNotIn(forbidden, text)
            self.assertIn("episode_ended", text)
            self.assertNotIn("success", json.dumps(response.data))
            # The policy receives the public native instruction with digests scrubbed, never the note.
            self.assertEqual({instruction for _, instruction in policy.calls}, {INSTRUCTION})
            self.assertEqual(receipt["instruction_source"], "native_task")

    def test_stop_conditions(self):
        cases = [
            (dict(policy=FakePolicy(fail_on=2)), "failed", "policy_timeout", 16),
            (dict(policy=FakePolicy(fail_on=1, timeout=False)), "rejected", "policy_error", 0),
            (dict(policy=FakePolicy(fail_reset=True)), "rejected", "policy_error", 0),
            (dict(policy=FakePolicy(), chunk_status="stopped", chunk_reason="ik_failure_stop"), "stopped", "safety_stop", 16),
            (dict(policy=FakePolicy(), chunk_status="failed", chunk_reason="backend_error"), "failed", "chunk_failed", 16),
            (dict(policy=FakePolicy(), step_limit=16), "finished", "native_budget_exhausted", 16),
            (dict(policy=FakePolicy(), end_at=24), "finished", "episode_ended", 24),
        ]
        for options, status, reason, steps in cases:
            with self.subTest(reason=reason):
                directory = temporary(self)
                harness, robot = self.harness(directory, **options)
                receipt = self.delegate(harness, max_policy_steps=200).data["receipt"]
                self.assertEqual((receipt["status"], receipt["stop_reason"], receipt["policy_steps_used"]),
                                 (status, reason, steps))
                self.assertFalse(any(path.exists() for path in robot.captures))

    def test_idempotent_replay_conflict_and_admission(self):
        directory = temporary(self)
        if True:
            policy = FakePolicy()
            harness, robot = self.harness(directory, policy)
            stale = harness.latest["observation_id"]
            first = self.delegate(harness, delegation_id="d-1", max_policy_steps=16, instruction="Stack the bowls.")
            again = harness.invoke("robot_policy_delegate", dict(observation_id=stale, delegation_id="d-1",
                max_policy_steps=16, instruction="Stack the bowls."), dict(tool_call_id="other"), role="actor")
            self.assertEqual(first.data["receipt"], again.data["receipt"])
            self.assertEqual((len(policy.calls), len(robot.received)), (1, 1))
            self.assertEqual(first.data["receipt"]["instruction_source"], "actor")
            with self.assertRaisesRegex(ValueError, "delegation_id_content_conflict"):
                harness.invoke("robot_policy_delegate", dict(observation_id=stale, delegation_id="d-1", max_policy_steps=8),
                               dict(tool_call_id="x"), role="actor")
            with self.assertRaisesRegex(ValueError, "stale_observation"):
                self.delegate(harness, call="stale", observation_id=stale)
            # The published schema bound (400) rejects it before the run's own bound does.
            with self.assertRaisesRegex(ValueError, "invalid_tool_arguments|invalid_max_policy_steps"):
                self.delegate(harness, call="too-many", max_policy_steps=401)
            with self.assertRaises(KeyError):
                harness.store.read("policy_delegation_request", "delegation-stale")
            self.assertEqual(len(policy.calls), 1)

    def test_every_decision_is_logged_for_gate_analysis(self):
        directory = temporary(self)
        if True:
            harness, _ = self.harness(directory, FakePolicy())
            harness.store.set_state("runtime", "token_usage", dict(total=dict(total_tokens=1234)))
            self.delegate(harness, call="logged", max_policy_steps=16, subtask_note="Try the insertion.")
            events = {event["type"]: event for event in harness.store.events(limit=1000)
                      if event["type"].startswith("policy_delegation.")}
            started, finished = events["policy_delegation.started"], events["policy_delegation.finished"]
            self.assertEqual(started["correlation"]["tool_call_id"], "logged")
            self.assertEqual(started["payload"]["arguments"]["subtask_note"], "Try the insertion.")
            self.assertEqual(started["payload"]["token_usage"]["total"]["total_tokens"], 1234)
            self.assertEqual(finished["payload"]["receipt"]["policy_steps_used"], 16)
            self.assertIn("token_usage", finished["payload"])
            self.assertEqual(finished["payload"]["inferences"][0]["executed_rows"], 16)
            record = harness.store.read("policy_delegation_request", "delegation-logged")
            self.assertEqual((record["task_id"], record["gate"]), ("stack_bowls", "none"))




class NoGateCheck(DelegationFixture):
    """gate none (default): neutral description, optional instruction, actor-chosen budget, no supervisor."""

    def test_published_contract_and_guidance(self):
        from auriga.cortex.context import delegation_gate
        item = {entry["name"]: entry for entry in public_definitions()}["robot_policy_delegate"]
        self.assertEqual(item["description"].split(" Examples")[0], delegation_gate.MECHANISM)
        self.assertEqual(item["inputSchema"]["required"], ["observation_id"])
        self.assertEqual(item["inputSchema"]["properties"]["max_policy_steps"]["maximum"], 400)
        self.assertEqual(public_definitions(), public_definitions(wall_oss_gate="none"))
        config = Config.from_dict(dict(features=dict(wall_oss=True), wall_oss=dict(server_url="ws://h:1")))
        self.assertEqual(config.wall_oss.gate, "none")
        self.assertEqual(delegation_gate.developer_instructions(config), "")
        self.assertIsNone(delegation_gate.call_budget(config.wall_oss, 500))

    def test_default_budget_and_no_supervisor(self):
        harness, robot = self.harness(temporary(self), FakePolicy(), robot_class=CameraRobot)
        harness.delegate_monitor = FakeMonitor(["done"])
        receipt = self.delegate(harness).data["receipt"]
        self.assertEqual((receipt["max_policy_steps"], receipt["policy_steps_used"], receipt["stop_reason"]),
                         (200, 200, "max_policy_steps"))
        self.assertNotIn("monitor", receipt)
        self.assertEqual(harness.delegate_monitor.calls, [])
        self.assertEqual(harness.store.read_state("runtime", "policy_delegation")["monitor"], "none")


class CameraRobot(DelegateRobot):
    """DelegateRobot whose observations carry the three named cameras the supervisor reads."""

    def observe(self):
        value = super().observe()
        value["cameras"] = [dict(camera_id=camera, png_path=str(self.path), width=1, height=1)
                            for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")]
        return value


class FakeMonitor:
    """Returns scripted verdicts (then 'normal'); records what each check was given."""

    def __init__(self, verdicts=(), raise_on=()):
        self.verdicts, self.raise_on, self.calls = list(verdicts), set(raise_on), []

    def check(self, context, images):
        self.calls.append((dict(context), sorted(images)))
        if len(self.calls) in self.raise_on:
            raise RuntimeError("monitor down")
        verdict = self.verdicts[len(self.calls) - 1] if len(self.calls) <= len(self.verdicts) else "normal"
        if verdict == "error":
            return dict(verdict=None, error="provider 503")
        return dict(verdict=verdict, confidence=0.9, reason=f"scripted {verdict}", usage=dict(total_tokens=1500),
                    latency_s=0.01, attempts=1, model="fake")


class PickPlaceV9Check(DelegationFixture):
    """pickplace_v9: actor-written instruction, budget = 80 % of the remaining native budget, sequence supervisor every 64
    policy steps that hands over after one takeover verdict (the config defaults)."""

    GATE = dict(gate="pickplace_v9", monitor_base_url="http://monitor.invalid/v1", monitor_key_file="/nonexistent/key")
    INSTRUCTION = "Do the task."
    # Object nouns of the benchmark task instructions: none may appear in the gate text, guidance, field help or examples.
    NOUNS = ("egg", "doll", "toast", "block", "xylophone", "tower", "clothes", "conveyor", "charger", "bottle", "dustbin",
             "equation", "bowl", "box", "coin", "pen", "toolbox", "tube", "mug", "ball", "vase", "kong", "screw", "liquid",
             "cup", "laptop", "headphone", "rack", "shelf", "slot", "plate", "drawer", "stand", "peg", "hole", "key")

    def run_with(self, monitor, call="v9", **wall_oss):
        harness, robot = self.harness(temporary(self), FakePolicy(), wall_oss=dict(self.GATE, **wall_oss),
                                      robot_class=CameraRobot)
        harness.delegate_monitor = monitor
        return harness, self.delegate(harness, call=call, instruction=self.INSTRUCTION).data["receipt"]

    def test_contract_and_guidance(self):
        import re
        from auriga.cortex.context import delegation_gate
        item = {entry["name"]: entry for entry in public_definitions(wall_oss_gate="pickplace_v9")}["robot_policy_delegate"]
        schema = item["inputSchema"]
        self.assertEqual(schema["required"], ["observation_id", "instruction"])
        self.assertNotIn("max_policy_steps", schema["properties"])
        self.assertTrue(all("max_policy_steps" not in example for example in schema["examples"]))
        self.assertTrue(schema["properties"]["instruction"]["description"].startswith("Required."))
        text = item["description"]
        for phrase in ("trained on whole manipulation tasks", "does not read text, numbers or symbols",
                       "memory is empty at every call", "native task instruction", "never single one out by its position",
                       "Never send a fragment", "fails the episode at once", "re-issue the same step once",
                       "camera images since the call started and the budget used",
                       "1 consecutive check(s) find that progress is not as expected",
                       "at most 80% of the episode's remaining", "(and at most 2000 steps)", "every 64 policy steps",
                       "monitor_done", "monitor_takeover", "you do not choose a budget", "geometry_query"):
            self.assertIn(phrase, text)
        other = {entry["name"]: entry for entry in public_definitions(wall_oss_gate="pickplace_v9", wall_oss_call_budget=1500,
                                                                      wall_oss_monitor=(32, 3, 0.6))}["robot_policy_delegate"]
        for phrase in ("every 32 policy steps", "3 consecutive check(s)", "at most 60% of", "(and at most 1500 steps)"):
            self.assertIn(phrase, other["description"])
        config = Config.from_dict(dict(features=dict(wall_oss=True), wall_oss=dict(server_url="ws://h:1", **self.GATE)))
        guidance = delegation_gate.developer_instructions(config)
        for phrase in ("Object-level delegation", "never send fragments or corrections", "instant-fail task rule",
                       "1 consecutive check(s) with progress not as expected", "at most 80% of the remaining",
                       "(at most 2000 steps)"):
            self.assertIn(phrase, guidance)
        blob = " ".join((text, guidance, json.dumps(delegation_gate.field_help("pickplace_v9")),
                         json.dumps(delegation_gate.examples("pickplace_v9"))))
        for noun in self.NOUNS:
            self.assertIsNone(re.search(rf"\b{noun}(e?s)?\b", blob, re.I), noun)
        for name in ("RoboDojo", "stack_bowls", "subtask_released", "max_policy_steps"):
            self.assertNotIn(name, text)
        self.assertEqual([delegation_gate.call_budget(config.wall_oss, r) for r in (1000, 200, 5000, 0, None)],
                         [800, 160, 2000, 1, 2000])
        self.assertEqual(delegation_gate.monitor_steps(config.wall_oss), 64)
        # Startup guidance is off when the tool is disabled.
        self.assertEqual(delegation_gate.developer_instructions(config.with_overrides(
            dict(features=dict(disabled_tools=["robot_policy_delegate"])))), "")

    def test_config(self):
        settings = Config().wall_oss
        self.assertEqual((settings.call_budget_fraction, settings.monitor_every_chunks, settings.monitor_takeover_consecutive,
                          settings.call_budget_max_steps), (0.8, 4, 1, 2000))
        base = dict(server_url="ws://h:1", gate="pickplace_v9")
        with self.assertRaisesRegex(ValueError, "monitor_requires_http_base_url"):
            Config.from_dict(dict(features=dict(wall_oss=True), wall_oss=dict(base, monitor_key_file="/k")))
        with self.assertRaisesRegex(ValueError, "monitor_requires_key_file"):
            Config.from_dict(dict(features=dict(wall_oss=True), wall_oss=dict(base, monitor_base_url="http://h:2/v1")))
        # The supervisor settings are only required when delegation is enabled.
        Config.from_dict(dict(wall_oss=base))
        for bad in (dict(call_budget_max_steps=20001), dict(call_budget_fraction=0.0), dict(gate="v0"),
                    dict(monitor_every_chunks=17)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Config.from_dict(dict(wall_oss=bad))
        # The 400-step max_policy_steps bound of the none gate does not limit a v9 call.
        Config.from_dict(dict(wall_oss=dict(base, call_budget_max_steps=1500)))

    def test_instruction_required_and_policy_reset(self):
        harness, robot = self.harness(temporary(self), FakePolicy(), wall_oss=self.GATE, robot_class=CameraRobot)
        policy = harness.delegate_policy
        harness.delegate_monitor = FakeMonitor(["done", "done"])
        with self.assertRaisesRegex(ValueError, "instruction_required|invalid_tool_arguments"):
            self.delegate(harness, call="no-instruction")
        from auriga.runtime.policy.delegation import _request
        with self.assertRaisesRegex(ValueError, "instruction_required"):
            _request(dict(observation_id="o"), harness.config.wall_oss, 1000)
        self.assertEqual((policy.calls, robot.received), ([], []))
        self.delegate(harness, call="a", instruction="Move the first object.")
        self.delegate(harness, call="b", instruction="Move the second object.")
        self.assertEqual(len(policy.resets), 2)
        self.assertNotEqual(policy.resets[0][1], policy.resets[1][1])
        self.assertEqual([instruction for _, instruction in policy.calls][-1], "Move the second object.")

    def test_done_stops_the_call_and_is_logged(self):
        monitor = FakeMonitor(["normal", "done"])
        harness, receipt = self.run_with(monitor)
        self.assertEqual((receipt["status"], receipt["stop_reason"], receipt["policy_steps_used"], receipt["max_policy_steps"]),
                         ("finished", "monitor_done", 128, 800))
        self.assertIn("supervisor at policy step 128: scripted done", receipt["stop_detail"])
        self.assertEqual((receipt["monitor"]["checks"], receipt["monitor"]["verdicts"], receipt["monitor"]["total_tokens"]),
                         (2, dict(normal=1, done=1, takeover=0), 3000))
        # Each check sees the delegated instruction, the steps and budget, the head history and the three cameras.
        context, images = monitor.calls[0]
        self.assertEqual((context["instruction"], context["steps"], context["budget"], context["history_steps"]),
                         (self.INSTRUCTION, 64, 800, [0]))
        self.assertTrue(context["task_instruction"].startswith(INSTRUCTION))
        self.assertEqual(images, ["cam_head", "cam_left_wrist", "cam_right_wrist", "history"])
        events = [e for e in harness.store.events(limit=1000) if e["type"] == "policy_delegation.monitor"]
        self.assertEqual([(e["payload"]["policy_steps"], e["payload"]["verdict"]) for e in events], [(64, "normal"), (128, "done")])
        summary = harness.store.read_state("runtime", "policy_delegation")
        self.assertEqual((summary["gate"], summary["monitor"], summary["monitor_checks"], summary["monitor_stops"],
                          summary["monitor_tokens"], summary["stop_reasons"], summary["policy_native_steps"]),
                         ("pickplace_v9", "gpt", 2, 1, 3000, dict(monitor_done=1), 128))
        finished = [e["payload"]["gate_log"] for e in harness.store.events(limit=1000) if e["type"] == "policy_delegation.finished"]
        self.assertEqual((finished[0]["gate"], finished[0]["monitor"], finished[0]["instruction"]),
                         ("pickplace_v9", "gpt", self.INSTRUCTION))
        # The actor sees the stop reason and the supervisor's reason, never the gate analysis records.
        harness.delegate_monitor = FakeMonitor(["done"])
        text = json.dumps(harness.actor_output.present(self.delegate(harness, call="again", instruction=self.INSTRUCTION),
                                                       tool="robot_policy_delegate"))
        self.assertIn("monitor_done", text)
        self.assertIn("scripted done", text)
        for forbidden in ("gate_log", "total_tokens_before", "token_usage"):
            self.assertNotIn(forbidden, text)

    def test_takeover_streak(self):
        # Default: one takeover verdict hands over.
        _, receipt = self.run_with(FakeMonitor(["normal", "takeover"]))
        self.assertEqual((receipt["stop_reason"], receipt["policy_steps_used"]), ("monitor_takeover", 128))
        self.assertIn("1 checks in a row", receipt["stop_detail"])
        # A configured streak of 3 needs consecutive verdicts; checks every 2 chunks.
        _, receipt = self.run_with(FakeMonitor(["takeover", "normal", "takeover", "takeover", "takeover"]),
                                   monitor_takeover_consecutive=3, monitor_every_chunks=2)
        self.assertEqual((receipt["stop_reason"], receipt["policy_steps_used"]), ("monitor_takeover", 160))
        self.assertEqual(receipt["monitor"]["verdicts"], dict(normal=1, done=0, takeover=4))

    def test_monitor_failures_never_stop_the_call(self):
        _, receipt = self.run_with(FakeMonitor(["error"] * 3, raise_on={4, 5}))
        self.assertEqual((receipt["stop_reason"], receipt["policy_steps_used"]), ("max_policy_steps", 800))
        self.assertEqual((receipt["monitor"]["checks"], receipt["monitor"]["errors"]), (12, 5))
        self.assertEqual(receipt["monitor"]["verdicts"]["normal"], 7)
        # An error between takeover verdicts resets the streak.
        _, receipt = self.run_with(FakeMonitor(["takeover", "takeover", "error", "takeover", "takeover", "takeover"]),
                                   monitor_takeover_consecutive=3)
        self.assertEqual((receipt["stop_reason"], receipt["policy_steps_used"]), ("monitor_takeover", 384))

    def test_budget_is_a_share_of_the_rest_of_the_episode(self):
        # Fixture episode budget 1000 steps; a monitor that never stops the call: the call uses 80 % of it.
        harness, receipt = self.run_with(FakeMonitor())
        self.assertEqual((receipt["max_policy_steps"], receipt["policy_steps_used"], receipt["stop_reason"]),
                         (800, 800, "max_policy_steps"))
        receipt = self.delegate(harness, call="second", instruction=self.INSTRUCTION).data["receipt"]
        self.assertEqual((receipt["max_policy_steps"], receipt["policy_steps_used"]), (160, 160))
        _, receipt = self.run_with(FakeMonitor(), call_budget_max_steps=300)
        self.assertEqual((receipt["max_policy_steps"], receipt["policy_steps_used"]), (300, 300))
        _, receipt = self.run_with(FakeMonitor(), call_budget_fraction=0.5)
        self.assertEqual(receipt["max_policy_steps"], 500)
        # The actor's max_policy_steps is ignored.
        harness, _ = self.harness(temporary(self), FakePolicy(), wall_oss=self.GATE, robot_class=CameraRobot)
        harness.delegate_monitor = FakeMonitor(["done"])
        receipt = self.delegate(harness, call="x", instruction=self.INSTRUCTION, max_policy_steps=16).data["receipt"]
        self.assertEqual((receipt["max_policy_steps"], receipt["policy_steps_used"], receipt["stop_reason"]), (800, 64, "monitor_done"))

    def test_sequence_history(self):
        from auriga.runtime.policy import progress_monitor as pm
        monitor = FakeMonitor(verdicts=("normal",) * 5 + ("takeover",))
        self.run_with(monitor)
        contexts = [context for context, _ in monitor.calls]
        self.assertEqual([c["steps"] for c in contexts], [64, 128, 192, 256, 320, 384])
        # The start image plus at most SEQUENCE_HISTORY earlier checks.
        self.assertEqual(contexts[3]["history_steps"], [0, 64, 128, 192])
        self.assertEqual(contexts[5]["history_steps"], [0, 128, 192, 256, 320])
        self.assertEqual(pm.SEQUENCE_HISTORY, 4)
        self.assertEqual({c["budget"] for c in contexts}, {800})
        text = pm.sequence_prompt(dict(task_instruction="T.", instruction="I.", steps=192, budget=800,
                                       history_steps=[0, 64, 128], cameras=("cam_head", "cam_left_wrist")))
        for phrase in ("used 192 of the 800 policy steps of this delegation (24% used, 608 left", "(step 0), at step 64, at step 128",
                       "current head camera (step 192), left wrist camera", "same attempt is repeated"):
            self.assertIn(phrase, text)
        self.assertEqual(pm.parse_verdict('{"verdict": "normal", "completed": "two of three"}')["completed"], "two of three")

    def test_gpt_monitor_request_and_parsing(self):
        from unittest import mock
        from auriga.runtime.policy import progress_monitor as pm
        self.assertEqual(pm.parse_verdict('ok {"verdict": "Done", "confidence": 0.8, "reason": "placed"} '),
                         dict(verdict="done", confidence=0.8, reason="placed"))
        for bad in ("no json", '{"verdict": "maybe"}'):
            with self.assertRaises(ValueError):
                pm.parse_verdict(bad)
        directory = Path(temporary(self))
        (directory / "auth.json").write_text(json.dumps(dict(OPENAI_API_KEY="sk-test-1")))
        (directory / "env.txt").write_text("# comment\nBASE_URL=https://x/v1\nAPI_KEY=sk-test-2\n")
        (directory / "empty.json").write_text("{}")
        self.assertEqual((pm.read_key(directory / "auth.json"), pm.read_key(directory / "env.txt")), ("sk-test-1", "sk-test-2"))
        with self.assertRaises(ValueError):
            pm.read_key(directory / "empty.json")
        sent = []

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return json.dumps(self.payload).encode()

        def answer(request, timeout):
            sent.append((request.full_url, dict(request.header_items()), json.loads(request.data)))
            return Response(dict(choices=[dict(message=dict(content='{"verdict": "takeover", "confidence": 0.7, '
                                                                    '"reason": "wrong object"}'))],
                                 usage=dict(total_tokens=1600)))

        monitor = pm.GptProgressMonitor("http://h:2/v1/", "sk-test-1", "gpt-test", timeout_s=5)
        png = b"\x89PNG\r\n\x1a\n" + b"0" * 8
        with mock.patch("urllib.request.urlopen", answer):
            result = monitor.check(dict(task_instruction="Tidy the desk.", instruction="Move the first object.",
                                        note="stage 1", steps=64, budget=800, history_steps=[0]),
                                   dict(history=[png], cam_head=png, cam_left_wrist=png))
        self.assertEqual((result["verdict"], result["reason"], result["usage"]["total_tokens"]), ("takeover", "wrong object", 1600))
        url, headers, body = sent[0]
        self.assertEqual((url, headers["Authorization"], body["model"]), ("http://h:2/v1/chat/completions", "Bearer sk-test-1", "gpt-test"))
        self.assertEqual(body["messages"][0], dict(role="system", content=pm.SYSTEM))
        user = body["messages"][1]["content"]
        self.assertEqual([part["type"] for part in user], ["text", "image_url", "image_url", "image_url"])
        self.assertTrue(user[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        for phrase in ("Overall task: Tidy the desk.", "Sub-instruction given to the policy: Move the first object.",
                       "Agent's note for this sub-instruction: stage 1", "used 64 of the 800 policy steps",
                       "head camera at the start of this delegation (step 0); then the current head camera (step 64), "
                       "left wrist camera."):
            self.assertIn(phrase, user[0]["text"])
        self.assertNotIn("right wrist camera", user[0]["text"])
        # Provider failures are retried, then reported (never raised).
        with mock.patch("urllib.request.urlopen", side_effect=OSError("down")), mock.patch("time.sleep"):
            failed = monitor.check(dict(instruction="x", steps=0), dict(cam_head=png))
        self.assertEqual((failed["verdict"], failed["attempts"]), (None, 3))
        self.assertIn("down", failed["error"])


class Pi05UnchangedCheck(unittest.TestCase):
    def test_policy_step_contract_and_execution_path_are_unaffected(self):
        neutral = {item["name"]: item for item in public_definitions()}
        gated = {item["name"]: item for item in public_definitions(wall_oss_gate="pickplace_v9")}
        self.assertEqual(neutral["robot_policy_step"], gated["robot_policy_step"])
        self.assertNotEqual(neutral["robot_policy_delegate"]["description"], gated["robot_policy_delegate"]["description"])
        self.assertTrue(neutral["robot_policy_step"]["description"].startswith("Delegate one local intent to the matched ARX X5 pi0.5"))

        class Pi05:
            calls = 0
            def infer(self, *args, **kwargs):
                Pi05.calls += 1
                return dict(actions=[[0.0] * 14 for _ in range(50)])

        directory = temporary(self)

        if True:
            store = Store(Path(directory) / "store", dict(IDENTITY))
            self.addCleanup(store.close)
            robot = DelegateRobot(directory)
            harness = Harness(store, robot)
            harness.start()
            harness.action_policy = Pi05()
            harness.dispatch("robot_policy_step", dict(observation_id=harness.latest["observation_id"], prefix=3,
                prompt="Close the gripper."), dict(tool_call_id="pi"))
            self.assertEqual(robot.received[0]["type"], "JointPrefixAction")
            self.assertEqual(len(robot.received[0]["actions"]), 3)
            self.assertNotIn("robot_policy_delegate", {item["name"] for item in harness.tool_registry.catalog(harness)})


class ConfigCheck(unittest.TestCase):
    def test_wall_oss_keys_defaults_and_validation(self):
        settings = Config().wall_oss
        self.assertEqual((settings.server_url, settings.default_policy_steps, settings.max_policy_steps, settings.replan_steps,
                          settings.timeout_s, settings.gate), (None, 200, 400, 16, 60.0, "none"))
        with self.assertRaises(ValueError):
            Config.from_dict(dict(wall_oss=dict(max_policy_steps=401)))
        with self.assertRaisesRegex(ValueError, "default_policy_steps"):
            Config.from_dict(dict(wall_oss=dict(default_policy_steps=300, max_policy_steps=250)))
        with self.assertRaisesRegex(ValueError, "websocket"):
            Config.from_dict(dict(wall_oss=dict(server_url="ws://h:1", server_urls=["http://h:2"])))
        venv = Config.from_dict(dict(wall_oss=dict(python="/opt/venvs/client/bin/python"))).wall_oss.python
        self.assertEqual(venv, "/opt/venvs/client/bin/python")
        with self.assertRaisesRegex(ValueError, "absolute_path"):
            Config.from_dict(dict(wall_oss=dict(python="venv/bin/python")))
        with self.assertRaisesRegex(ValueError, "server_urls_require_server_url"):
            Config.from_dict(dict(wall_oss=dict(server_urls=["ws://h:2"])))
        gated = Config.from_dict(dict(features=dict(wall_oss=True), wall_oss=dict(
            server_url="ws://h:1", server_urls=["ws://h:2"], gate="pickplace_v9", monitor_base_url="http://h:3/v1",
            monitor_key_file="/k")))
        self.assertEqual(gated.wall_oss.server_urls, ("ws://h:2",))
        from auriga.cortex.context.delegation_gate import developer_instructions
        self.assertIn("robot_policy_delegate", developer_instructions(gated))
        self.assertFalse(Config().features.wall_oss)
        with self.assertRaisesRegex(ValueError, "wall_oss_requires_server_url"):
            Config.from_dict(dict(features=dict(wall_oss=True)))
        with self.assertRaisesRegex(ValueError, "websocket"):
            Config.from_dict(dict(wall_oss=dict(server_url="http://host:1")))
        with self.assertRaises(ValueError):
            Config.from_dict(dict(wall_oss=dict(replan_steps=49)))
        enabled = Config.from_dict(dict(features=dict(wall_oss=True), wall_oss=dict(server_url="ws://10.0.0.2:19000")))
        self.assertTrue(enabled.features.wall_oss)
        self.assertEqual(developer_instructions(enabled), "")
        self.assertEqual(developer_instructions(gated.with_overrides(dict(features=dict(wall_oss=False)))), "")
        self.assertEqual(developer_instructions(Config()), "")


if __name__ == "__main__":
    unittest.main()
