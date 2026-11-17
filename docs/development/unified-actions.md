# Unified Action Interface and Program Review

Model tool entry points for new runs are managed by `runtime/tools/definitions.py` and `builtins.py` through `ToolRegistry` (the legacy `tool_catalog.py` has been removed). Task planning, action compilation, physical execution, model presentation, and operational memory are decoupled, while robot state and raw execution evidence remain owned by `Harness` and `Store`.

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `actor_instructions.py` / `definitions.py`, `builtins.py` (formerly `tool_catalog.py`, removed) | Centralize launch instructions, public tool declarations, and concise descriptions; legacy tools are excluded from the model catalog. |
| `task_planning.py` | Plain-text task plan and replaceable attention checklist; gates the first physical action; updating the plan never mutates already compiled trajectories. |
| `codex.py` / `model_proxy.py` | Keep standard `/responses` requests as strict prefix extensions of conversation history (applying only public-boundary filtering and deterministic exact-image deduplication); strip prior Harness snapshots and inject the full current plan snapshot only during primary-actor `/responses/compact` compaction requests. Auxiliary visual review models never receive the episode plan. |
| `unified_actions.py` | `robot_action` argument validation, short plan IDs, preview/execute/inspect modes, and result/image delivery. |
| `batch_contract.py` / `batch_motion.py` | Native batch protocol, dependency graph, whole-batch compilation, shared timeline, cached trajectory execution, and observation checkpoint resumption. |
| `batch_geometry.py` | Clearance estimation using the public robot model, caller-supplied obstacles, and held-object bounding boxes; never reads hidden task state. |
| `programs.py` / `program_review.py` | Isolated Python processes, primary-model action review, cancellation and active-work timing, and trusted action result aggregation. |
| `stability.py` | Opt-in bounded visible-motion settling detection that consumes real native steps via standard wait actions. |
| `actor_output.py` / `actor_presentation.py` | Model-facing field compaction, reversible short IDs, and artifact references; raw archived evidence is never mutated. |
| `operation_memory.py` | Separate storage and retrieval for current-task vs. cross-task operational experience, extracting action facts from verified history. |

## From Task Understanding to Execution

Robot capabilities are delivered automatically at launch. The model may observe the scene, query geometry, or preview actions before submitting a natural-language task plan:

```json
{
  "plan": "Task goal: transfer the target object as instructed. Approach: select a reachable grasp, transport, and place it. Completion criteria: the target is supported at the destination and remains stable after gripper withdrawal.",
  "attention": [
    {
      "concern": "Whether closing the gripper actually lifts the intended object remains uncertain.",
      "check": "Use feedback from the normal lift stage to verify the object leaves its support and moves with the hand; if not, adjust contact position or grasp orientation.",
      "when": "After lifting and before relying on held-object state for transport."
    }
  ]
}
```

The default operation of `task_plan` is `submit`, which returns a compact confirmation; `update` revises the overall plan or current attention items, and `read` retrieves the full active version. A plan must be submitted before the first physical action (including `robot_wait`). Plans do not need to be updated before every action and are not an instruction table for the motion executor.

We recommend structuring `plan` as a short summary of task goal, approach, and completion criteria. `attention` is an optional list where each item specifies `concern` (current question, hypothesis, or candidate option), `check` (observable verification criteria and how to adjust if violated), and `when` (the action or event that depends on this conclusion). Keep only currently relevant items; items can be added, revised, removed, or paired with operational memories as new evidence arrives. `update` requires at least `plan` or `attention`; omitted fields retain their previous values, and `attention: []` explicitly clears the list. Resubmitting identical fields under the same observation returns `status=unchanged` without creating a new revision or duplicate snapshot, whereas resubmitting the same content under a newer observation records the updated observation provenance. Updating replaces the active checklist while keeping older revisions auditable; retained plan text never becomes verified fact simply by being updated. Each text field is capped at 1,600 characters (excess is rejected rather than truncated), with no mandatory item count.

Standard non-compacting `/responses` requests neither append nor mutate decision reminders so each request remains a strict prefix extension of the previous one to maximize prompt-cache hits. Only when context length triggers a primary-thread actor `/responses/compact` request does the proxy re-read the episode's latest plan, revision, full attention list, evidence ledger, and source observations to inject a complete snapshot. `plan_observation_id`, `attention_observation_id`, and `ledger_observation_id` record the observation active when each respective section was last updated (partial updates preserve unchanged provenance, and legacy archives without provenance report `null`); they do not prove the model's judgment is correct. The snapshot is formatted as an assistant message representing the model's own tentative notes, passed through shared metadata filtering, never elevated to `developer` instructions, never sent to auxiliary visual reviewers, and never used to mutate historical tool receipts. At the compaction boundary, prior Harness snapshots are removed and replaced with a single fresh snapshot while preserving ordinary assistant messages; request retries reuse the exact projected payload without re-invoking context hooks. Agent backends other than Codex get no request-body injection: when the session reports that its model may have lost earlier context (native or relay-detected compaction, uncertain tool-reply delivery, resume after an unclean exit) through `AgentSession.on_context_reset`, `Harness.model_context_reset` marks the reminder due and `Harness.handle_tool` appends the same text and snapshot (`task_planning.decision_reminder_text`, after the same metadata filtering) as one extra `inputText` item to the next actor tool response, then clears the flag. A response whose reminder cannot be built is returned unchanged, with its receipt, and the reminder waits for the next response. Codex runs never receive this sideband (`Harness.decision_reminder_channel`).

Verification should piggyback on normal manipulation feedback whenever possible: a purposeful approach, lift, or withdrawal can simultaneously test a hypothesis. Add dedicated probing steps only when existing evidence is insufficient and the answer would change the next action; in multi-stage batches, insert observation checkpoints only when later stages depend on uncertain earlier outcomes. When a program pauses for default action review, control returns to the primary model along with the latest checklist; programs launched with `allow_direct_actions=true` execute without step-by-step model reasoning and must either evaluate computable evidence conditions themselves or return control to the primary model when judgment is required. The checklist never triggers automatic observations, auto-checks items, or dispatches recovery actions on its own.

Although the pinned Codex SDK/CLI `0.160.0` includes a model tool `update_plan` and `turn/plan/updated` progress notifications, its generated client RPC schema provides no plan read/write API, and `thread/read` / `thread/resume` do not expose a recoverable structured checklist. Auriga therefore uses Harness's `task_plan` as the single source of truth rather than maintaining a duplicate native to-do mirror. See the [App Server reference](https://www.codex-docs.com/docs/app-server) and [Hooks reference](https://www.codex-docs.com/docs/hooks) (verified against our pinned local version). Regressions live in `test_task_planning`, `test_codex_requests`, and `test_model_proxy`.

Coordinates in the example below must be derived from the current observation. A `move` stage accepts either `grasp_position_m + angles_deg` or `link6` `position_m + quaternion_xyzw` (the two representations cannot be mixed in the same stage).

Stage parameters are declared per `kind` and share field definitions with the compiler. A `move` stage requires `arm` and a complete pose specification; contact reference fields apply only to grasp poses, and setting gripper speed or synchronized timing requires `gripper_goal_normalized`. A `gripper` stage requires `arm` and `goal_normalized`. The `reason` field is valid only on `observe` checkpoints and must be a non-blank string. Fields from different stage kinds cannot be mixed, and both actor and program calls validate against the same schema before entering the handler. Quaternion normalization, dependency graphs, backend capabilities, cache validity, and execution budgets remain enforced at their respective domain boundaries.

```python
preview = robot_action(
    observation_id=current_observation,
    stages=[
        dict(id="left_approach", kind="move", arm="left",
             position_m=left_target, quaternion_xyzw=left_orientation),
        dict(id="right_prepare", kind="move", arm="right",
             position_m=right_target, quaternion_xyzw=right_orientation),
        dict(id="check", kind="observe", reason="Verify dual-arm arrival and object alignment"),
        dict(id="close", kind="gripper", arm="left", goal_normalized=0),
    ],
)
# After inspecting cost, trajectory, overlay images, and unknowns in the preview:
result = robot_action(mode="execute", plan_id=preview_plan_id)
# After pausing at the observation checkpoint and verifying live feedback, resume the same plan if still valid:
result = robot_action(mode="execute", plan_id=preview_plan_id)
```

The snippet above illustrates arguments passed by the primary model to the native dynamic tool. Inside Python programs, `AurigaClient.call` returns `success`, `kind`, `data`, and `contentItems`; read `data` keyed by `kind` (with action receipts at `data.receipt`) rather than parsing status strings from display text. The primary model receives images directly as independent image items without needing a JavaScript wrapper or manual base64 decoding. Preview and historical images from program sub-calls are forwarded automatically by `program_run` and `program_status`—never print base64 image strings to `stdout`.

Stages on the same arm are ordered implicitly, while `depends_on` adds cross-arm dependencies; independent stages on opposite arms execute concurrently. An `observe` stage without `arm` waits for all preceding stages on both arms to complete, whereas specifying `arm` pauses both arms as soon as that arm finishes its preceding stage. Each stage can specify its own speed, and `gripper_motion` on a `move` stage schedules gripper actuation across a fractional window of that move (if omitted, the gripper command runs after arm arrival).

To move both arms simultaneously, place left and right stages into the **same** `robot_action` `stages` list—even if all left-arm stages are listed before all right-arm stages, the two arms still advance in parallel on the shared timeline. The launch instruction to "await tool calls sequentially" governs separate tool invocations, not the two arms within a single batch. Splitting left and right moves into two separate `execute` calls runs them serially; similarly, adding a cross-arm `depends_on` or inserting an `observe` stage without `arm` between them intentionally serializes execution. Add cross-arm dependencies only when a genuine physical precondition exists.

`preview` is the default mode and does not advance the simulation. An executable plan is returned only if the entire batch compiles; if any later stage fails planning, no earlier stage is executed. `execute` dispatches the exact cached trajectory after verifying environment identity, measured starting state, remaining budgets, and first-step joint jumps—never silently replanning. Resuming from an observation checkpoint additionally verifies the execution cursor and checkpoint state. Replaying an identical request is idempotent and does not re-execute motion. Changing targets, routes, or invalidating the plan requires a fresh preview. Explicit `direct` mode plans and executes in a single call.

In addition to explicit `observe` stages, `robot_action` (API version `1.1`) accepts an optional boolean `checkpoint_after_gripper` (default `false`) on `move`/`gripper` stages containing a gripper command or at the batch top level. When explicitly enabled—and when that gripper command window changes the gripper goal and is followed by a dependent `move` stage (via same-arm ordering or `depends_on`)—the compiler automatically inserts an observation checkpoint at the existing native cursor at the end of that window (setting `false` on the window's final stage overrides a batch-level `true`, and an existing explicit checkpoint at the same cursor is merged). Preview `checkpoints` and execution `checkpoint_stage_ids` report the triggering stage ID. Execution pauses with `stopped` / `observation_checkpoint`, a formal terminal observation, and a typed receipt, allowing the model to resume the same `plan_id` with zero replanning. When omitted, the batch executes continuously without interrupting routine gripper openings or post-release withdrawals.

Because a checkpoint pauses both arms, an independent stage running on the opposite arm also pauses at the shared cursor and resumes along its cached trajectory afterward; within a single stage, any motion remaining after an in-motion synchronized gripper ramp finishes to the end of that stage before pausing. Terminal gripper stages, commands that do not change the latched goal, and gripper stages without downstream physical dependents do not add checkpoints; stages already separated by an explicit `observe` continue to follow explicit observation boundaries.

Batch execution currently runs under the `resampled` native mode, supporting up to 32 stages total (up to 16 per arm) and at most 250 native steps across the entire batch (including segments resumed across observation checkpoints). The plan cache holds up to 16 plans; evicted plans or plans predating a native worker rebuild cannot be executed. Executed trajectory evidence remains inspectable via `mode="inspect", action_id=...` or a saved `trajectory_ref` (paginated by default, or complete via `detail="full"`).

Joints, end-effector poses, and gripper values in a preview represent planned commands, whereas execution returns actual measurements, failure or partial-execution statuses, and synchronized frame images. Obstacle and held-object bounding boxes are caller estimates, and held-object clearance assumes rigid attachment. Unavailable geometry is reported as `unknown` rather than certifying collision-free motion. Contact declarations only annotate geometric clearance estimates and never prove physical contact, grasp retention, or object support.

Default action feedback preserves per-joint first-step jump magnitudes and allowed limits (`first_joint_step_rad`, `max_joint_step_rad`) as well as segment-start tracking deviations (`max_joint_start_error_rad`) on failure. Trusted action summaries in the program broker likewise retain these diagnostics alongside per-stage endpoint errors, contact-reference position feedback, full stage diagnostics, and unknown extension fields; program summaries cover all actions executed by the job so early failures are never evicted by a sliding window.

For failed previews, self-collision model diagnostics pass through the information boundary check and deliver all sampled witnesses in full rather than truncating to the deepest three. The existing `planner_collision_witnesses` JSON artifact and `collision_model_witnesses_ref` remain available for inspection. Joint position, velocity, acceleration, and jerk constraint checks likewise retain the worst violation per joint so none are dropped outside a top-three cutoff. These witnesses derive solely from candidate trajectories and the public robot model—not from executed collisions or ground-truth environment contact. If artifact archiving fails, the boundary-validated witnesses are still delivered inline alongside an archive error diagnostic; any data failing permission or serialization checks is withheld. This behavior does not backfill dropped witnesses in historical archives or recover uncaptured solver seeds and intermediate optimizer paths.

Failed previews are also recorded as `action.batch_preview` events, preserving the public failure reason, safety witnesses, source observation binding, and budget snapshot. The recovery ledger links actor tool requests and public `program.call_started` broker requests to their corresponding previews within the same episode, thread, turn, and call identity (previews inside programs undergo the same evidence validation and are never fabricated from `stdout` or evaluator state). Entries are recorded only when zero physical steps occurred and the observation/budget bindings match; a candidate is marked as a budget rejection only when its known cost exceeds the remaining budget with a matching budget-rejection witness. Unknown costs or IK failures are never misclassified as insufficient budget. Duplicate results for the exact same candidate are deduplicated while changes in status, cost, or evidence are retained. No single candidate rejection proves the overall task is infeasible. See `test_unified_actions` and `test_recovery_evidence` for regressions.

## Program Review and Active-Work Timing

By default, `program_run(operation="start", source=...)` does not grant direct physical action authority. When a program calls `client.action(...)` or `client.call("robot_action", ...)` to execute motion, the broker first compiles a preview, freezes the program's process group, and returns `awaiting_review`, a `review_id`, the preview data, and overlay images to the primary model:

```json
{"operation": "resume", "job_id": "<returned job_id>", "review_id": "<returned review_id>", "decision": "approve"}
```

Approving executes the reviewed cached plan; rejecting returns a rejection payload to the Python program so it can handle the decision. If another call invalidates the cached plan while review is pending, the stale review cannot proceed. A program cannot call `resume` to approve itself or use legacy tool names to bypass review. Because `robot_wait` advances physical time, it also requires review.

The primary model may pass `allow_direct_actions=true` at `start` time to authorize direct execution for that specific program run; permissions, source code, and action arguments cannot be modified during `resume`. Review logic is centralized in the broker and currently approved autonomously by the primary model without waiting for a human operator.

Program active-work timeout defaults to 180 seconds (maximum 300 seconds) with no cap on tool call count. Active-work timing excludes wall-clock time spent waiting for auxiliary models or primary-model review. Cancellation remains available while paused for review and cleanly thaws and reaps the process group. Process exit codes and `action_summary` are tracked separately: action outcomes are collected exclusively from trusted broker replies, never inferred from `stdout` or exit code `0`. Terminal frame images are delivered via native dynamic image items.

A `direct` action can also fail during the planning phase. Only when the typed `plan` result explicitly records `status=failed`, `physics_advanced=false`, and an integer step count of `0` does the program's `action_summary` record `not_executed` / `phase=planning` while preserving the failure reason and prior observation (without generating an action receipt). If those proofs are absent or the execution channel is interrupted, the outcome remains `unknown` with an unknown step count—never backfilled as zero based on display text or outer `success`.

## Optional Settling Wait and Operational Memory

Actions return immediately upon completion by default. Passing an explicit `settle={"roi": [x, y, w, h], "max_native_steps": 25}` runs bounded visible-motion settling detection (capped at 50 native steps) using standard `robot_wait` calls that count against the native action budget. Consecutive trackable low-displacement frames only indicate low visible motion within the ROI; camera motion or insufficient texture returns `unknown`. Settling results are reported separately from the primary action receipt, and the current observation ID advances to the final post-wait frame. Standalone `robot_wait` remains available.

`operation_memory` stores and retrieves operational experience across any manipulation phase. Queries for `scope="task"` and `scope="shared"` are kept strictly separate, and cross-task memories must state their applicability conditions. Recording a memory verifies action and observation ownership and extracts execution facts directly from run history. Historical coordinates and past task-success judgments never substitute for live observations.

## Compatibility Boundaries

Legacy motion tools, standalone preview/cost tools, queues, manipulation templates, fixed skills, grasp tables, and `scene_*` tools are no longer exposed to models in new runs. Optional visual manipulation review is consolidated into `vision_review(evidence_mode="operation")`. Legacy native execution and history-parsing adapters remain in the codebase for reading historical archives and testing low-level mechanisms, without granting model or program execution entry points. Legacy program-template tests run through test-only adapters; no production flag re-enables them.

The presentation layer strips versions, hashes, source file paths, and duplicated request payloads, retaining only actionable/inspectable short IDs and essential measurements while preserving full raw evidence and trajectories in the archive. Image selection policies, high-frequency feedback decisions, and auxiliary-model review policies are never altered by presentation formatting.

## Registration and Internal Boundaries

The `ToolSpec` declaration for `robot_action` drives the model catalog, actor calls, and program calls alike. Running under the `feedback` backend mode, `unified_action.available=false`, or an uncalibrated observation profile marks `robot_action` unavailable and rejects direct calls before entering the handler. The legacy `motion_review` toggle governs historical single-arm previews and does not disable the unified `resampled` batch interface. Default effects for `preview`, `execute`, and `inspect` are `plan`, `physical`, and `read`, respectively.

Batch compilation calls `arx_actions.compile_action(kind, arguments, identity, capabilities)` directly rather than routing through legacy tool names or private `Harness` conversion methods. `Harness` retains ownership of identity, current observation tracking, action admission, single-writer physical dispatch, and immutable receipts, while the backend enforces native constraints through its `validate_action` port. New actions cannot be planned when the current observation is unavailable, whereas replaying an already committed `action_id` returns its original persisted record.

Review and execution return internal `ToolResponse(kind="plan"|"action", data=...)` objects consumed directly by both the program broker and generated programs; client-facing `kind/data` copies undergo the same provenance filtering and short-reference resolution while retaining `contentItems` for wire compatibility. Primary-model native tool replies continue to use `{success, contentItems}`. For adding new tools, see [Tool Registration and Extensions](tool-plugins.md).

`grasp_position_m` specifies the center point between the two fingers on the selected contact cross-section—not the surface of a single finger—and `contact_region` selects the axial cross-section along the finger without guaranteeing physical object contact. `position_tolerance_m` checks measured `link6` position and `orientation_tolerance_rad` checks measured orientation (defaulting to `10 mm` and `0.15 rad` under `tolerance_preset="normal"` when omitted). Tracking failure halts subsequent stages, including not-yet-dispatched gripper commands; relaxing tolerances never implies physical contact occurred. Completing a gripper command or measuring a change in jaw aperture likewise does not prove an object is held; verify retention using task-appropriate observable evidence.

## Explicit Motion and Held-Evidence Contract

Grasp-angle planning preserves the requested orientation. Infeasible poses return a planning failure so the actor can explicitly choose new angles and preview again; joint-space replanning remains available without altering the Cartesian target. Vertical compliance never relaxes the declared orientation tolerance, and downward error remains bounded by the declared position tolerance. Contact-stop modifies only stopping behavior and the documented upward compliance window, never declared horizontal or angular tolerances.

Held-object metadata reports `retention_status="unverified"` (no retention assertion), `attachment_assumption="rigid_since_grasp"` (conditional kinematic model), `geometry_evidence="current_mask"|"prediction_only"` (availability of usable current segmentation), and `current_mask_views` (head/wrist views with usable masks). These categorical fields carry no physical units and appear on both carried and fallback rows; a current mask does not establish co-motion. Prediction-only rows remain available during occlusion but cannot establish a successful grasp or continued retention. No additional observation or model call is incurred.
