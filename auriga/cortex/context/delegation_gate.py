"""Delegation gate for whole-subtask policy delegation: when the actor should hand over to WALL-OSS.

The gate is prompt-level only. The actor decides from the tool description and
the startup guidance; the Harness never routes on its behalf. ``wall_oss.gate``
selects the text:

* ``none`` (default): a neutral description of the mechanism and no guidance.
  The actor may omit ``instruction`` (the native task instruction is sent) and
  chooses ``max_policy_steps`` (default ``wall_oss.default_policy_steps``, at
  most ``wall_oss.max_policy_steps``). No supervisor runs.
* ``pickplace_v9``: the actor writes the policy instruction itself: the native
  task instruction for the whole rest of the task when no decision is needed in
  between, otherwise the complete action on one object (or on the objects the
  task instruction handles together); never a fragment or a correction. The
  description keeps only facts about the policy (closed loop, trained on whole
  tasks, no reading or reasoning, empty memory), the instruction format and the
  instant-fail-rule step, with no concrete examples. The actor does not choose a
  budget: a call may use at most ``wall_oss.call_budget_fraction`` of the
  episode's remaining native budget (at most ``wall_oss.call_budget_max_steps``),
  so a failed call leaves the actor a reserve, and the same step may be
  re-issued once. A camera supervisor (runtime/policy/progress_monitor.py)
  checks the call every ``wall_oss.monitor_every_chunks`` chunks with the head
  camera at the call start and at the last few checks, the current three
  cameras and the budget used, and ends the call when the subtask is done
  (``monitor_done``) or after ``wall_oss.monitor_takeover_consecutive``
  checks in a row find that progress is not as expected (``monitor_takeover``).

Every delegation and its outcome are logged by runtime/policy/delegation.py so
the quality of the gate can be measured offline.
"""

MECHANISM = (
    "Delegate a whole subtask to the WALL-OSS-1.0-lite vision-language-action policy (dual-arm, native "
    "end-effector pose chunks). It runs closed-loop from current three-camera RGB and end-effector state: "
    "infer a chunk, execute up to replan_steps native steps, repeat, until max_policy_steps, the episode ends, "
    "a policy error or timeout, a rejected/failed chunk or a safety stop. Control then returns to you with a "
    "compact receipt and the final observation. Every executed row is one official native step from the "
    "episode budget; the receipt never reports task success.")

PICKPLACE_V9 = (
    "Hand both arms to a low-level visuomotor policy (WALL-OSS). It runs closed-loop from the current camera images and "
    "gripper poses. It was trained on whole manipulation tasks, each given as one plain instruction like the native task "
    "instructions. It does not read text, numbers or symbols and does no reasoning, and its memory is empty at every "
    "call. "
    "Instruction: one plain command in the style of this episode's native task instruction: that instruction itself for "
    "the whole rest of the task when no decision is needed in between, or otherwise the complete action on one object "
    "(or on the objects the task instruction handles together), naming the object and its destination. Decide the object "
    "and the destination yourself and name them by plain appearance and position; the policy cannot tell identical "
    "objects apart, so never single one out by its position. Never send a fragment (only a grasp, a single movement or a "
    "gripper command) or a correction of an earlier result: the policy does not stop after a fragment. Do a step yourself "
    "when breaking a task rule there fails the episode at once (read the task rules, e.g. a height limit), and delegate "
    "at most its coarse object moves. After a call that failed (monitor_takeover, a safety stop, or a result you find "
    "wrong), you may re-issue the same step once in the same wording; otherwise finish it yourself. "
    "Stopping: every {every} policy steps a supervisor looks at the camera images since the call started and the "
    "budget used. The call ends when it judges the subtask done (stop_reason monitor_done), or when "
    "{streak} consecutive check(s) find that progress is not as expected (the same attempt repeated, no "
    "object getting closer to its end state, an object lost while nothing else progresses, or too slow for "
    "the budget left; stop_reason monitor_takeover: take over yourself); "
    "otherwise it may use at most {fraction}% of the episode's remaining native step budget (and at most {steps} "
    "steps), so that you keep a reserve to finish or correct the step yourself; you do not choose a budget. "
    "Control then returns to you with a receipt (steps used, stop reason with the supervisor's reason, IK failures) "
    "and the final observation; every step is spent from the episode's native step budget. The supervisor can be "
    "wrong, especially for near misses (an object slightly off its target or tilted): after every call verify the "
    "result yourself (images, geometry_query) before the next step, and correct it yourself.")

DEVELOPER_PICKPLACE_V9 = """
## Object-level delegation (WALL-OSS)
robot_policy_delegate gives both arms to a low-level visuomotor policy trained on whole manipulation tasks; it does not read text, numbers or symbols, does no reasoning and remembers nothing between calls. Send one plain command in the style of the native task instruction: that instruction for the whole rest of the task when no decision is needed in between, otherwise the complete action on one object (or on the objects the instruction handles together). Decide the object and the destination yourself and name them by appearance and position; never pick one of several identical objects by position, and never send fragments or corrections. Do a step under an instant-fail task rule yourself. After a failed call you may re-issue the same step once; otherwise finish it yourself. A camera supervisor that sees the images since the call started and the budget used ends the call when the subtask looks done (monitor_done) or after {streak} consecutive check(s) with progress not as expected (monitor_takeover), otherwise after at most {fraction}% of the remaining native step budget (at most {steps} steps), keeping you a reserve; you do not choose a budget. Verify every result yourself before the next step.
"""

PICKPLACE_V9_FIELD_HELP = {"instruction": (
    "Required. One plain command in the style of the native task instruction, sent verbatim to the policy, 1..2000 "
    "characters: the native instruction for the whole rest of the task, or the complete action on one object, naming "
    "the object and its destination by plain appearance and position; no fragments. Arm names or distances in the text "
    "are not constraints.")}

PICKPLACE_V9_EXAMPLES = [dict(observation_id="obs_1",
                              instruction="<one plain command in the style of the native task instruction>",
                              subtask_note="<which step this is and the expected end state>")]

GATES = ("none", "pickplace_v9")
# Gates with the camera supervisor (runtime/policy/progress_monitor.py), an actor-written instruction and a call budget
# fixed by the run (max_policy_steps not published, a supplied value ignored).
SUPERVISED_GATES = ("pickplace_v9",)


def _render(template, steps, every, streak, fraction):
    return template.replace("{fraction}", str(int(round(fraction * 100)))).format(steps=steps, every=every,
                                                                                 streak=streak)


def pickplace_v9(steps=2000, every=64, streak=1, fraction=0.8):
    return _render(PICKPLACE_V9, steps, every, streak, fraction)


def developer_pickplace_v9(steps=2000, every=64, streak=1, fraction=0.8):
    return _render(DEVELOPER_PICKPLACE_V9, steps, every, streak, fraction)


def supervised(gate):
    """Whether calls at this gate run under the camera supervisor."""
    return gate in SUPERVISED_GATES


def tool_description(gate, call_budget=2000, monitor_every=64, monitor_streak=1, call_fraction=0.8):
    if gate == "pickplace_v9":
        return pickplace_v9(call_budget, monitor_every, monitor_streak, call_fraction)
    return MECHANISM


def fixed_budget(gate):
    """Whether the run fixes the per-call budget (max_policy_steps is then not published and is ignored)."""
    return supervised(gate)


def call_budget(settings, native_remaining=None):
    """The per-call budget the run fixes, or None when the actor chooses max_policy_steps. pickplace_v9:
    wall_oss.call_budget_fraction of the episode's remaining native budget, at most wall_oss.call_budget_max_steps."""
    if not fixed_budget(settings.gate):
        return None
    cap = settings.call_budget_max_steps
    if not isinstance(native_remaining, int):
        return cap
    return max(1, min(cap, int(native_remaining * settings.call_budget_fraction)))


def published_budget(settings):
    """The step number quoted in the tool text: the upper bound of a fixed call budget, else wall_oss.max_policy_steps."""
    return settings.call_budget_max_steps if fixed_budget(settings.gate) else settings.max_policy_steps


def requires_instruction(gate):
    """Gates at which the actor must write the policy instruction itself."""
    return supervised(gate)


def field_help(gate):
    """Gate-specific overrides of robot_policy_delegate field help (empty for none)."""
    return dict(PICKPLACE_V9_FIELD_HELP) if gate == "pickplace_v9" else {}


def examples(gate):
    return [dict(item) for item in PICKPLACE_V9_EXAMPLES] if gate == "pickplace_v9" else None


def monitor_steps(settings):
    """Policy steps between two supervisor checks."""
    return settings.monitor_every_chunks * settings.replan_steps


def developer_instructions(config):
    """Startup guidance appended to the actor instructions, or empty."""
    if not config.features.wall_oss or config.wall_oss.gate != "pickplace_v9":
        return ""
    if "robot_policy_delegate" in config.features.disabled_tools:
        return ""
    settings = config.wall_oss
    return developer_pickplace_v9(published_budget(settings), monitor_steps(settings),
                                  settings.monitor_takeover_consecutive, settings.call_budget_fraction)
