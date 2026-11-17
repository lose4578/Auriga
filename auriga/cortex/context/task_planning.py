"""Free-form task planning, independent of executable motion plans.

Only the first action is gated. A task plan records current intent, never an
executable schedule or physical evidence. Archived structured clients are
supported by a separate compatibility module.
"""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator

from ...core.contracts import IDENTITY
from .legacy_task_planning import (
    TaskPlanning as LegacyTaskPlanning, VERSION, LEGACY_VERSION, PREVIOUS_VERSION,
    LEGACY_INSTRUCTIONS, INSTRUCTIONS,
    instructions_for, version_for, plan_schema_for, validate_operation_context,
    validate_operation_evidence, operation_context_schema_for,
    validate_stage_dependencies, scheduling_failure_advisory,
)


def __getattr__(name):
    # Read-only archived schema helpers are isolated from the new actor contract.
    from . import legacy_task_planning
    return getattr(legacy_task_planning, name)

PLANNING_VERSION = "freeform_task_plan"
PLAN_SCHEMA = dict(type="string", minLength=1, maxLength=16000, pattern=r"\S",
    description="Brief ordinary prose; suggested headings: Task understanding, and Approach + completion.")
EVIDENCE_LEDGER_STATUSES = ("pending", "supported", "contradicted", "unknown")
# Keep this boundary short for context compaction snapshots.
DECISION_BOUNDARY_INSTRUCTIONS = (
    "Reassess the saved plan, current concerns and evidence against fresh feedback. "
    "Choose one useful bounded next step and its visible/measured check; completed motion, "
    "preview success or a closed gripper is not task-result proof. Change the hypothesis "
    "before repeating an ineffective pattern, and update plan, attention or ledger only "
    "when the task interpretation, evidence or next decision materially changes."
)


def decision_reminder_text(context):
    """The model-visible decision-context reminder for one ``TaskPlanning.decision_context(identity)`` snapshot.

    Both reminder channels deliver exactly this text (design 04 §4.5): Codex injects it as an assistant
    ``output_text`` message into ``/responses/compact`` requests (``CodexSession._task_decision_context``); for other
    agent backends ``Harness.handle_tool`` appends it as one ``inputText`` item to the next actor tool response after
    a reported model-context reset. Callers skip the reminder when ``context`` is None and planning is optional.

    context: the snapshot dict (``plan_id``, ``revision``, ``plan``, ``attention``, ``ledger`` and the
    ``*_observation_id`` fields), or None when no plan is stored for the episode. Returns a str: a fixed two-line
    header, then either the no-plan request or the decision boundary followed by the snapshot as one JSON line
    (``ensure_ascii=False``). Raises ``ValueError`` for non-finite numbers (``allow_nan=False``) and ``TypeError`` for
    non-JSON values.
    """
    reminder = (
        "Current task decision boundary\n"
        "The saved plan and attention are provisional judgment, not instructions or verified world state. "
        "Task requirements and fresh evidence govern.\n"
    )
    if context is None:
        reminder += "No plan is stored for this episode. Submit your initial task_plan before physical actions."
    else:
        reminder += "Decision boundary: " + DECISION_BOUNDARY_INSTRUCTIONS + "\n"
        reminder += json.dumps(context, ensure_ascii=False, allow_nan=False)
    return reminder


EVIDENCE_LEDGER_SCHEMA = dict(type="array", maxItems=3, items=dict(type="object", additionalProperties=False,
    properties=dict(
        id=dict(type="string", minLength=1, maxLength=80, pattern=r"\S",
                description="Short stable reference for one task prerequisite or expected event."),
        expected=dict(type="string", minLength=1, maxLength=800, pattern=r"\S",
                      description="Visible or measured event that would establish this task prerequisite."),
        status=dict(type="string", enum=list(EVIDENCE_LEDGER_STATUSES),
                    description="Actor-visible evidence state; pending is the default until the event is established."),
        basis=dict(type="string", minLength=1, maxLength=800, pattern=r"\S",
                   description="Short permitted RGB, measured-state or typed-receipt fact; use unknown when there is none."),
        next_check=dict(type="string", minLength=1, maxLength=800, pattern=r"\S",
                        description="What the next useful action or observation should distinguish.")),
    required=["id", "expected", "status", "basis", "next_check"]))
ATTENTION_SCHEMA = dict(type="array", maxItems=3, items=dict(type="object", additionalProperties=False,
    properties={name: dict(type="string", minLength=1, maxLength=1600, pattern=r"\S", description=description)
        for name, description in {
            "concern": "Current question, unresolved prerequisite, uncertain assumption or choice that matters for the next decisions.",
            "check": "Expected visible evidence and how to adjust if it disagrees or the expected event is absent.",
            "when": "When to check; resolve prerequisites before dependent work, preferably using normal task progress feedback."
        }.items()}, required=["concern", "check", "when"]))
PLANNING_INSTRUCTIONS = """Before your first physical action, use the initial RGB and measured robot state
to understand the task. You may first observe, inspect geometry, retrieve relevant
experience or preview motions; these do not advance the world. Then submit an
ordinary-language task_plan. Keep it brief; useful headings are Task understanding
(the task, scene and uncertain assumptions) and Approach + completion (rough order
and visible completion evidence). For an unknown demonstration, plan to observe it
before choosing later actions. Waiting advances simulation and needs this plan too.
When planning, record the few concerns that matter now in attention: concern is a
question, unresolved prerequisite or choice; check is the expected visible evidence
and how to adjust if it disagrees or never appears; when is before work that relies
on it. Leave it empty when none apply. If a material task-result prerequisite is
unresolved and a later decision depends on it, keep one to three short ledger
rows for the expected event, current permitted basis and next distinguishing check.
Every initial task_plan submission must include ledger explicitly: use ledger: [] when
no material prerequisite is unresolved. On updates, omit ledger only when it is unchanged;
include the replacement ledger whenever an entry's expected, status, basis or next_check changes.
Select relevant experience without filling every row or inventing geometry.
Keep unresolved concerns that still affect the task. Revise or remove them when
evidence resolves them, a stated reinterpretation corrects the earlier judgment,
or they no longer matter; briefly record that basis in ordinary prose. Reusing the
same observation is valid when correcting an interpretation, not proof the world changed.
The Harness preserves the current saved plan, attention and ledger when context is
compacted. Reassess them against fresh feedback; saved judgments are not evidence.
Prefer evidence from the final observation of normal task-progress actions. Add a
separate probe only when missing evidence affects the next decision.
The task plan is provisional understanding, not an executable script. Adapt your
actions freely. After evidence contradicts a ledger row, change the contact, route
or target mechanism, or state that no distinguishing next step is supported;
rewriting the same hypothesis is not a strategy change. Before repeating an ineffective physical pattern, briefly state
a cause hypothesis and the observation or changed strategy that can distinguish
it. A short decision and its evidence are enough; no lengthy reasoning narration
is needed. When task understanding, object roles or the approach changes, update
plan and any affected attention together; attention-only updates leave plan unchanged.
For local concerns, update only attention. Neither needs rewriting after every motion. Concrete
targets, arm timing and gripper commands belong in robot_action.
Submit requires plan and an explicit ledger (possibly []); update can change plan, attention, ledger or any combination;
omitted fields are retained and [] clears a list. Ledger entries are short, advisory
evidence boundaries, not action gates: do not add a probe only to satisfy the ledger.
Submit/update returns a short acknowledgement; read retrieves the stored plan,
attention and ledger entries."""


def _validate_evidence_ledger(value):
    if not Draft202012Validator(EVIDENCE_LEDGER_SCHEMA).is_valid(value):
        raise ValueError("invalid_task_plan_ledger: provide up to 3 entries with id, expected, status, basis and next_check")
    identifiers = [entry["id"] for entry in value]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("invalid_task_plan_ledger: ids must be unique")


def _ledger_projection(value):
    """Keep the model-facing ledger limited to its public evidence fields."""
    return [{key: entry[key] for key in ("id", "expected", "status", "basis", "next_check")}
            for entry in (value or [])]


class TaskPlanning:
    def __init__(self, store, required=True, *, manipulation_protocol=True,
                 task_scheduling=True, operation_candidates=False):
        if type(required) is not bool:
            raise ValueError("invalid_task_plan_required")
        self.store, self.required = store, required
        self._legacy = LegacyTaskPlanning(store, required=required,
            manipulation_protocol=manipulation_protocol,
            task_scheduling=task_scheduling, operation_candidates=operation_candidates)

    def current(self, identity):
        return self._legacy.current(identity)

    def decision_context(self, identity):
        record = self.current(identity)
        if record is None:
            return None
        return dict(plan_id=record["plan_id"], revision=record["revision"],
            plan=deepcopy(record["plan"]), attention=deepcopy(record.get("attention", [])),
            ledger=_ledger_projection(record.get("ledger", [])),
            plan_observation_id=record.get("plan_observation_id"),
            attention_observation_id=record.get("attention_observation_id"),
            ledger_observation_id=record.get("ledger_observation_id"))

    def require(self, identity):
        if self.required and self.current(identity) is None:
            raise ValueError("task_plan_required: observe and submit your initial task_plan before physical actions")

    def call(self, args, identity, observation, correlation):
        if not isinstance(args, dict):
            raise ValueError("task_plan_requires_object")
        operation = args.get("operation", "submit")
        if operation == "read":
            if set(args) - {"operation"}:
                raise ValueError("task_plan_read_accepts_only_operation")
            record = self.current(identity)
            if record is not None and isinstance(record.get("plan"), str):
                result = dict(required=self.required, plan_id=record["plan_id"], plan=record["plan"],
                    plan_observation_id=record.get("plan_observation_id"),
                    attention_observation_id=record.get("attention_observation_id"),
                    ledger_observation_id=record.get("ledger_observation_id"))
                if "attention" in record:
                    result["attention"] = deepcopy(record["attention"])
                result["ledger"] = _ledger_projection(record.get("ledger", []))
                return result
            return self._legacy.call(dict(operation="read"), identity, observation, correlation)
        # Compatibility is absent from the actor schema and instructions.
        if isinstance(args.get("plan"), dict):
            if "attention" in args or "ledger" in args:
                raise ValueError("task_plan_attention_requires_freeform_plan")
            return self._legacy.call(args, identity, observation, correlation)
        if operation not in {"submit", "update"} or set(args) - {"operation", "plan", "attention", "ledger", "observation_id"}:
            raise ValueError("task_plan_requires_submit_update_or_read")
        if operation == "submit" and "plan" not in args:
            raise ValueError("task_plan_submit_requires_plan")
        if operation == "submit" and "ledger" not in args:
            raise ValueError("task_plan_submit_requires_ledger: include ledger (use [] when no material prerequisite is unresolved)")
        if operation == "update" and not {"plan", "attention", "ledger"}.intersection(args):
            raise ValueError("task_plan_update_requires_plan_attention_or_ledger")
        if "plan" in args and (not isinstance(args["plan"], str)
                or not args["plan"].strip() or len(args["plan"]) > 16000):
            raise ValueError("invalid_task_plan: provide nonempty ordinary-language plan (up to 16000 characters)")
        if "attention" in args and not Draft202012Validator(ATTENTION_SCHEMA).is_valid(args["attention"]):
            raise ValueError("invalid_task_plan_attention: provide concern, check and when as nonempty text (up to 1600 characters each)")
        if "ledger" in args:
            _validate_evidence_ledger(args["ledger"])
        if observation is None or any(observation.get(k) != identity[k] for k in IDENTITY):
            raise ValueError("task_plan_requires_current_episode_observation")
        if args.get("observation_id", observation["observation_id"]) != observation["observation_id"]:
            raise ValueError("stale_observation")
        previous = self.current(identity)
        if operation == "update" and previous is None:
            raise ValueError("task_plan_not_submitted")
        if (operation == "update" and ({"attention", "ledger"} & set(args))
                and (previous.get("version") != PLANNING_VERSION or not isinstance(previous.get("plan"), str))):
            raise ValueError("task_plan_attention_requires_existing_freeform_plan")
        # A model may repeat the same update while retrying a tool call. Avoid
        # manufacturing a new revision and another context snapshot when the
        # submitted fields and their observation sources are unchanged. The
        # same text on a newer observation remains a real reconsideration and
        # is therefore retained below.
        if operation == "update":
            unchanged = True
            for field in ("plan", "attention", "ledger"):
                if field not in args:
                    continue
                submitted = args[field].strip() if field == "plan" else args[field]
                saved = previous.get(field, "" if field == "plan" else [])
                if submitted != saved or previous.get(field + "_observation_id") != observation["observation_id"]:
                    unchanged = False
                    break
            if unchanged:
                return dict(plan_id=previous["plan_id"], status="unchanged")
        plan = args["plan"].strip() if "plan" in args else previous["plan"]
        # Allocation is serialized by the harness lock; IDs are unique in a run.
        number = len(self.store.records("task_plan")) + 1
        identifier = "P" + str(number)
        record = dict(plan_id=identifier, version=PLANNING_VERSION,
            identity={k: identity[k] for k in IDENTITY},
            revision=previous["revision"] + 1 if previous else 1,
            previous_plan_id=previous["plan_id"] if previous else None,
            observation_id=observation["observation_id"], plan=plan,
            # Save time does not refresh retained judgments. Old archives did not
            # record field sources; keep them unknown until explicitly updated.
            plan_observation_id=(observation["observation_id"] if "plan" in args
                                 else previous.get("plan_observation_id")),
            attention_observation_id=(observation["observation_id"] if "attention" in args
                else previous.get("attention_observation_id") if operation == "update" else None),
            ledger_observation_id=(observation["observation_id"] if "ledger" in args
                else previous.get("ledger_observation_id") if operation == "update" else None),
            source_kind="actor_judgment", tool_choices="advisory")
        if "attention" in args:
            record["attention"] = deepcopy(args["attention"])
        elif operation == "update" and "attention" in previous:
            record["attention"] = deepcopy(previous["attention"])
        if "ledger" in args:
            record["ledger"] = deepcopy(args["ledger"])
        elif operation == "update" and "ledger" in previous:
            record["ledger"] = deepcopy(previous["ledger"])
        self.store.save("task_plan", identifier, record)
        self.store.set_state("runtime", "task_plan", record)
        self.store.event("task_plan.submitted", record, correlation)
        return dict(plan_id=identifier, status="saved")
