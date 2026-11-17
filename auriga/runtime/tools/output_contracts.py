"""ToolResponse.data contracts for built-in actor tools.

Fields and types come from archived typed results and their producers. A type
also admits every value a producer can legitimately emit (often null for an
unknown count or missing evidence). Every object stays open: owners may add
diagnostics without a contract change. These schemas are not shown to models.
"""

S, I, N, B, O, A, NULL = 'string', 'integer', 'number', 'boolean', 'object', 'array', 'null'


def _value(types, description):
    return dict(type=types, description=description)


def _object(description, properties, required=(), types=O):
    result = dict(type=types, description=description, properties=properties, additionalProperties=True)
    if required:
        result['required'] = list(required)
    return result


def _list(description, items):
    return dict(type=A, description=description, items=items)


ENVELOPE = dict(type='object', additionalProperties=True,
    description='Arbitrary JSON object; this tool has no field-level contract.')

ACTION_BUDGET = _object('Native-step accounting snapshot; null counts are unknown, never zero.', dict(
    native_steps_used=_value([I, NULL], 'Official native steps consumed in this episode.'),
    native_steps_limit=_value([I, NULL], 'Native-step limit of this episode.'),
    native_steps_remaining=_value([I, NULL], 'Native steps left: limit minus used, floored at 0.'),
    per_request_native_limit=_value([I, NULL], 'Maximum native steps one action request may consume.'),
    source=_value(S, 'Accounting owner, e.g. native_execution_accounting.'),
    unknown_values_are_null=_value(B, 'True: a null count means unknown.'),
    semantics=_value(S, 'Accounting scope prose.')))

RECEIPT = _object('Owner action receipt. Only status is guaranteed; an unrecorded or in-progress action can carry nothing else.', dict(
    status=_value(S, 'Execution status, e.g. completed, failed, stopped, rejected, unknown; accepted/running while in progress.'),
    reason=_value([S, NULL], 'Failure or stop reason; null when none.'),
    result_observation_id=_value([S, NULL], 'Observation captured after execution; null when no terminal capture exists.'),
    execution_start_observation_id=_value(S, 'Historical observation captured at the execution start.'),
    observation_unavailable_reason=_value(S, 'Why result_observation_id is null.'),
    executed_native_steps=_value([I, NULL], 'Official native actions executed by this request; null when unknown.'),
    completed_native_steps=_value([I, NULL], 'Native take_action calls that returned successfully; null when unknown.'),
    executed_steps=_value([I, NULL], 'Official native action count delta, including a partially executed faulting call.'),
    executed_control_steps=_value([I, NULL], 'Physics steps within the official action interpolation.'),
    step_semantics=_value(O, 'Prose meaning of each step counter.'),
    stop_confirmation=_value(O, 'Evidence that motion stopped after a stop/failure.'),
    action_budget=ACTION_BUDGET,
    batch_status=_value(S, 'Cached-batch status, e.g. completed, failed, awaiting_observation.'),
    batch_plan_id=_value(S, 'Native batch plan executed.'),
    batch_cursor=_value(I, 'Native rows of the cached batch already executed (resume cursor).'),
    batch_revision=_value(I, 'Batch revision after this request; an observation checkpoint increments it.'),
    complete_batch_native_steps=_value(I, 'Native rows of the complete cached batch.'),
    stage_results=_value(O, 'Per-stage outcome keyed by stage id.'),
    completed_stage_ids=_list('Stages completed by this request.', _value(S, 'Stage id.')),
    pending_stage_ids=_list('Stages not yet executed.', _value(S, 'Stage id.')),
    checkpoint_stage_ids=_list('Stages that ended at an observation checkpoint.', _value(S, 'Stage id.')),
    failed_stage_id=_value(S, 'Stage that failed.'),
    failed_arm=_value(S, 'Arm (left/right) whose stage failed.'),
    waypoint_results=_value(A, 'Per-waypoint endpoint verification.'),
    motion_safety_checks=_value(A, 'Joint-travel admission checks performed before dispatch.'),
    endpoint_residual_m=_value(O, 'Measured endpoint residual: dx_m/dy_m/dz_m/position_error_m in metres, orientation_error_rad in radians.'),
    speed_scale=_value(N, 'Resolved arm speed scale (dimensionless; 1 = nominal).'),
    gripper_speed_scale=_value(N, 'Resolved gripper speed scale (dimensionless; 1 = nominal).'),
    wait_observation_ids=_list('Observations sampled during a bounded wait.', _value(S, 'Observation id.')),
    planned_trace_ref=_value([S, NULL], 'Artifact of the planned joint trace; null when not recorded.'),
    dispatched_trace_ref=_value([S, NULL], 'Artifact of the dispatched command trace; null when not recorded.'),
    actual_trace_ref=_value([S, NULL], 'Artifact of the measured trace; null when not recorded.'),
    before_state_ref=_value([S, NULL], 'Artifact of the measured start state; null when not recorded.'),
    planned_trace_available=_value(B, 'Projected receipts: whether planned_trace_ref exists.'),
    dispatched_trace_available=_value(B, 'Projected receipts: whether dispatched_trace_ref exists.'),
    actual_trace_available=_value(B, 'Projected receipts: whether actual_trace_ref exists.'),
    before_state_available=_value(B, 'Projected receipts: whether before_state_ref exists.')), required=['status'])

REQUEST = _object('Recorded action request after compilation (the executed command).', dict(
    action_id=_value(S, 'Action id.'),
    based_on_observation_id=_value(S, 'Observation on which the request was decided.'),
    type=_value(S, 'Compiled action type, e.g. BatchAction.'),
    mode=_value(S, 'Pose mode, e.g. absolute.'),
    frame_id=_value(S, 'Reference frame of poses.'),
    tcp_config_id=_value(S, 'Tool-centre-point configuration of pose targets.'),
    timeout_s=_value(N, 'Request timeout, seconds.'),
    stages=_value(A, 'BatchAction stages in execution order.'),
    batch_plan_id=_value(S, 'Native cached batch plan this request executes.'),
    batch_expected_cursor=_value(I, 'Batch cursor the request expected before executing.'),
    batch_expected_revision=_value(I, 'Batch revision the request expected before executing.'),
    hold_native_steps=_value(I, 'Bounded wait: native steps to hold the measured pose.')))

FULL_RESULT = _object('Arguments of the tool call that reads the complete record.', dict(
    tool=_value(S, 'Tool to call.'), detail=_value(S, 'Detail level to request.')))

ACTION = _object('Original action evidence; the receipt is the execution authority.', dict(
    receipt=RECEIPT, request=REQUEST,
    current_observation_id=_value([S, NULL], 'Current observation after the request; null when none is current.'),
    result_observation_is_current=_value(B, 'Whether receipt.result_observation_id is the current observation.')),
    required=['receipt'])

DELEGATION_RECEIPT = _object('Whole-subtask policy delegation receipt. It reports execution facts only, never task success.', dict(
    delegation_id=_value(S, 'Delegation id; reuse with identical input rereads this receipt.'),
    status=_value(S, 'finished (policy ran until its budget, the episode ended, or the camera supervisor judged the subtask done or took over), stopped (safety stop or cancellation), failed (policy error/timeout or a chunk that was rejected or failed), rejected (nothing executed) or unknown (a chunk execution is unconfirmed).'),
    stop_reason=_value(S, 'max_policy_steps, monitor_done (the supervisor judged the subtask done; not proof of success), monitor_takeover (the supervisor judged progress not as expected), episode_ended, native_budget_exhausted, policy_error, policy_timeout, policy_input_unavailable, safety_stop, cancelled, chunk_rejected, chunk_failed, chunk_stopped or chunk_unknown.'),
    stop_detail=_value([S, NULL], 'Owner reason behind stop_reason; null when none.'),
    policy_steps_used=_value([I, NULL], 'Official native steps executed by the policy in this delegation; null when a chunk count is unknown.'),
    max_policy_steps=_value(I, 'Native-step budget of this delegation.'),
    replan_steps=_value(I, 'Native steps executed per policy inference at most.'),
    inference_calls=_value(I, 'Policy inferences made.'),
    chunk_action_ids=_list('Recorded EEChunkAction ids in execution order.', _value(S, 'Action id.')),
    instruction=_value(S, 'Language instruction sent to the policy.'),
    instruction_source=_value(S, 'native_task (current observation instruction) or actor (caller-supplied).'),
    subtask_note=_value([S, NULL], 'Caller statement of the delegated subtask; not sent to the policy.'),
    final_observation_id=_value([S, NULL], 'Observation after the last chunk; null when none was captured.'),
    ik_failed_steps=_value(O, 'Per-arm native steps whose native IK failed (that arm kept its previous joint target).'),
    last_policy_action=_value([O, NULL], 'Last executed policy row per arm: commanded position_m, quaternion_wxyz (env_origin) and gripper opening command; null when nothing executed.'),
    action_budget=ACTION_BUDGET), required=['delegation_id', 'status', 'stop_reason'])

DELEGATION = _object('robot_policy_delegate result; the receipt is the execution authority.', dict(
    request=_value(O, 'Recorded delegation request.'),
    receipt=DELEGATION_RECEIPT,
    current_observation_id=_value([S, NULL], 'Current observation after the delegation; null when none is current.'),
    result_observation_is_current=_value(B, 'Whether receipt.final_observation_id is the current observation.')),
    required=['receipt'])

UNIFIED_ACTION = _object('robot_action execute/direct result: original action evidence for a cached plan.', dict(
    plan_id=_value(S, 'Short plan id that was executed.'),
    action_id=_value(S, 'Action id of this execution.'),
    receipt=RECEIPT, request=REQUEST,
    settling=_value(O, 'Bounded settling requested by the plan, after a completed batch.'),
    postprocess=_value(O, 'Settling postprocess failure; the receipt remains authoritative.'),
    current_observation_id=_value([S, NULL], 'Current observation after settling.'),
    result_observation_is_current=_value(B, 'Whether receipt.result_observation_id is the current observation.')),
    required=['receipt', 'request', 'plan_id', 'action_id'])

PLAN = _object('robot_action preview/inspect plan summary. A failed preview carries no plan_id; inspect returns the updated summary.', dict(
    status=_value(S, 'Plan status: planned, failed or rejected; after execution the latest batch status.'),
    reason=_value([S, NULL], 'Failure reason, e.g. path_infeasible or motion_safety_rejected.'),
    plan_id=_value(S, 'Short plan id for execute/inspect.'),
    observation_id=_value(S, 'Observation the plan is bound to.'),
    physics_advanced=_value(B, 'Whether planning advanced physics (expected false).'),
    executed_native_steps=_value(I, 'Native steps executed while planning (expected 0).'),
    planned_native_steps=_value(I, 'Native rows the plan would execute.'),
    nominal_sim_time_s=_value(N, 'Nominal simulated duration of the plan, seconds.'),
    executed_batch_native_steps=_value(I, 'Native rows of this plan already executed.'),
    plan_revision=_value(I, 'Plan revision; observation checkpoints increment it.'),
    next_action=_value(O, 'Arguments of the next robot_action call for this plan.'),
    last_action_id=_value(S, 'Latest action that executed this plan.'),
    checkpoint_observation_id=_value([S, NULL], 'Observation captured at the pending checkpoint.'),
    failed_stage_id=_value([S, NULL], 'Stage whose planning failed.'),
    failed_arm=_value(S, 'Arm (left/right) whose planning failed.'),
    failure_feedback=_value(O, 'Planner failure diagnostics.'),
    motion_safety_checks=_value(A, 'Joint-travel admission checks of a rejected preview.'),
    motion_safety=_value(O, 'Admission summary of a planned preview; details_ref holds the checks.'),
    trajectory_ref=_value(S, 'Artifact of the candidate trajectory.'),
    trajectory_overlay=_value(O, 'Trajectory overlay image metadata.'),
    trajectory_evidence=_value(O, 'Candidate trajectory diagnostics.'),
    stage_costs=_value(O, 'Native-step cost per stage.'),
    arm_costs=_value(O, 'Native-step cost per arm.'),
    checkpoints=_value(A, 'Observation checkpoints inside the batch.'),
    geometry_review=_value(O, 'Geometry review of the candidate.'),
    execution_semantics=_value(S, 'Execution prose for the cached rows.'),
    action_budget=ACTION_BUDGET,
    settle=_value(O, 'Requested post-batch settling options.'),
    additional_settle_budget=_value(O, 'Native-step budget for settling, separate from the plan cost.')),
    required=['status'])

ACTION_INSPECTION = _object('robot_action inspect by action/trajectory: null when the inspection is display-only.', dict(
    action_id=_value(S, 'Action whose trajectory feedback is unavailable.'),
    available=_value(B, 'False: no measured trajectory feedback was recorded.'),
    trajectory_feedback=_value(O, 'Diagnostic of the missing feedback.')), types=[O, NULL])

CAMERA = _object('Camera frame of an ObservationBundle.', dict(
    camera_id=_value(S, 'Camera id, e.g. cam_head.'),
    frame_id=_value(S, 'Capture frame id.'),
    rgb_ref=_value(S, 'Artifact of the original RGB image.'),
    width=_value(I, 'Image width, pixels.'),
    height=_value(I, 'Image height, pixels.'),
    pixel_transform=_value(S, 'Pixel transform from the original image (identity).'),
    capture_time=_value([S, NULL], 'Capture time, ISO 8601.'),
    clock_domain=_value([S, NULL], 'Capture clock domain.'),
    sim_step=_value([I, NULL], 'Simulation step of the capture.'),
    step_domain=_value([S, NULL], 'Domain of sim_step.'),
    synchronization=_value([S, NULL], 'Capture alignment with physics.'),
    control_dt_s=_value([N, NULL], 'Control period, seconds.'),
    image_encoding=_value(O, 'Stored image encoding.'),
    calibration_ref=_value([S, NULL], 'Artifact of the camera calibration (calibrated profiles).'),
    pose_sample_ref=_value([S, NULL], 'Artifact of the camera pose sample (calibrated profiles).')),
    required=['camera_id', 'rgb_ref', 'width', 'height'])

OBSERVATION = _object('ObservationBundle (contracts.validate_observation already checked it).', dict(
    observation_id=_value(S, 'Observation id.'),
    robot_state_ref=_value(S, 'Artifact of the measured robot state.'),
    cameras=_list('Camera frames captured together.', CAMERA),
    world_kind=_value(S, 'Always observed.'),
    sim_step=_value([I, NULL], 'Simulation step of the capture.'),
    step_domain=_value([S, NULL], 'Domain of sim_step, e.g. adapter_control_step.'),
    sim_time_s=_value([N, NULL], 'Simulation time of the capture, seconds.'),
    clock_domain=_value(S, 'Environment instance and generation of the clock.'),
    observation_profile=_value(S, 'Observation permission profile.'),
    robot_model_id=_value(S, 'Robot model id.'),
    robot_model_ref=_value([S, NULL], 'Artifact of the robot kinematic model (calibrated profiles).'),
    calibration_id=_value([S, NULL], 'Calibration identity; null without calibration.'),
    last_action_id=_value([S, NULL], 'Action that produced this observation; null for an explicit capture.'),
    task_instruction=_value(S, 'Native public task instruction.'),
    data_origin=_value(S, 'Data source, e.g. robodojo_runtime.'),
    action_budget=ACTION_BUDGET),
    required=['observation_id', 'robot_state_ref', 'cameras'])

OBSERVATION_METADATA = _object(
    'Observation metadata as delivered: the first display part. Compact profiles replace artifact refs '
    'with *_available flags and unchanged static fields with metadata_delivery references.', dict(
    bundle=_object('Saved ObservationBundle projection.', dict(
        observation_id=_value(S, 'Observation id.'),
        cameras=_list('Camera frames.', _object('Camera frame projection.', dict(
            camera_id=_value(S, 'Camera id.'),
            frame_id=_value(S, 'Capture frame id.'),
            width=_value(I, 'Image width, pixels.'),
            height=_value(I, 'Image height, pixels.'),
            sim_step=_value([I, NULL], 'Simulation step of the capture.')), required=['camera_id'])),
        sim_step=_value([I, NULL], 'Simulation step of the capture.'),
        sim_time_s=_value([N, NULL], 'Simulation time of the capture, seconds.'),
        last_action_id=_value([S, NULL], 'Action that produced this observation.'),
        task_instruction=_value(S, 'Native public task instruction.'),
        world_kind=_value(S, 'Always observed.'),
        action_budget=ACTION_BUDGET), required=['observation_id']),
    robot_state=_object('Measured robot state at capture time.', dict(
        arms=_value(O, 'Per-arm measured state keyed by arm.'),
        sim_step=_value([I, NULL], 'Simulation step of the measurement.'))),
    camera_calibration=_value(O, 'Disclosed camera calibration (calibrated profiles).'),
    camera_pose_sample=_value(O, 'Camera pose sample of this observation (calibrated profiles).'),
    image_view=_object('Cropped/scaled view metadata when roi or scale was requested.', dict(
        camera_id=_value(S, 'Camera of the view.'),
        source_roi=_value(A, 'Crop [x, y, width, height] in original pixels.'),
        display_scale=_value(I, 'Integer upscale factor.'),
        width=_value(I, 'View width, pixels.'),
        height=_value(I, 'View height, pixels.'),
        pixel_transform=_value(A, '3x3 transform from original to view pixels.'),
        preprocessing=_value(S, 'Image preprocessing, e.g. crop_resize_lanczos.'))),
    metadata_delivery=_object('References to static fields delivered earlier in this model context.', dict(
        mode=_value(S, 'Delivery mode.'),
        static_refs=_value(O, 'Earlier observations holding each omitted static group.'),
        read_full=_value(O, 'Arguments to read this observation in full.'))),
    geometry_delivery=_object('Which reconstructive geometry was omitted inline.', dict(
        mode=_value(S, 'Delivery mode.'),
        omitted=_value(S, 'What was omitted.'),
        read_full=_value(O, 'Arguments to read the full geometry.'))),
    current_observation_id=_value([S, NULL], 'Current observation of the run; null when none is current.'),
    observation_is_current=_value(B, 'Whether this observation is the current one.'),
    current_action_budget=ACTION_BUDGET,
    observation_budget_semantics=_value(S, 'bundle.action_budget is the capture-time snapshot.')),
    required=['bundle', 'robot_state', 'current_observation_id', 'observation_is_current'])

GEOMETRY = _object('Calibrated RGB geometry result or the geometry help catalog.', dict(
    status=_value(S, 'supported, conditional, unknown or invalid_input.'),
    reason=_value(S, 'Why status is unknown/invalid_input.'),
    query_type=_value(S, 'Requested geometry kind.'),
    world_kind=_value(S, 'internal: a derived estimate, not an observation.'),
    frame_id=_value(S, 'Frame of metric values.'),
    unit=_value(S, 'Length unit of metric values (m).'),
    observation_ids=_value(A, 'Observations used.'),
    frame_ids=_value(A, 'Capture frame ids used.'),
    observation_states=_value(A, 'Measured robot state of each observation used.'),
    calibration_id=_value([S, NULL], 'Calibration identity used.'),
    camera_pose_source=_value(S, 'Camera pose source, e.g. simulator_camera_transform or robot_kinematics.'),
    camera_pose_diagnostics=_value(O, 'Camera pose consistency diagnostics.'),
    value=_value([O, NULL], 'Kind-specific result; null when unavailable.'),
    assumptions=_value(A, 'Assumptions the result depends on.'),
    unknown_regions=_value(A, 'Regions the result cannot describe.'),
    diagnostics=_value(O, 'Failure diagnostics and input guidance.'),
    duration_s=_value(N, 'Processing wall time, seconds.'),
    common=_value(S, 'Help: fields shared by every request.'),
    requests=_value(O, 'Help: request kinds and their fields.'),
    limitation=_value(S, 'Help: evidence limitation.')),
    required=['status', 'world_kind', 'frame_id', 'unit'])

DEPTH = _object('Depth-derived result (depth-enabled runs only).', dict(
    depth_derived=_value(B, 'Always true for depth results.'),
    measurement_id=_value(S, 'Depth measurement id.'),
    kind=_value(S, 'Measured depth query kind.'),
    status=_value(S, 'Measurement or target status.'),
    reason=_value(S, 'Why the measurement is not fully supported.'),
    observation_id=_value(S, 'Observation measured.'),
    camera_id=_value(S, 'Camera measured.'),
    target_id=_value(S, 'Registered depth target.'),
    target_status=_value(S, 'Target status after registration: tracked or uncertain.')),
    required=['depth_derived'])

MEMORY_RECORD = _object('Public operation-memory record.', dict(
    memory_id=_value(S, 'Memory id.'),
    scope=_value(S, 'task or shared.'),
    operation=_value(S, 'Action category.'),
    summary=_value(S, 'Actor lesson.'),
    conditions=_value(S, 'When the lesson applies (empty for task scope without conditions).'),
    facts=_value(A, 'Recorded execution facts of the evidence actions.'),
    observation_ids=_list('Evidence observations.', _value(S, 'Observation id.')),
    source=_value(O, 'Source run and episode.'),
    source_kind=_value(S, 'actor_lesson_with_recorded_execution_facts.'),
    physical_outcome=_value(S, 'not_verified: a lesson is not a verified object outcome.'),
    reuse=_value(S, 'Shared records: how to revalidate before reuse.')),
    required=['memory_id', 'scope'])

PROGRAM_JOB = dict(
    job_id=_value(S, 'Program job id.'),
    status=_value(S, 'Process status: running, awaiting_review, succeeded, failed, cancelled or unknown.'),
    reason=_value(S, 'Failure/stop reason.'),
    identity=_value(O, 'Run/episode identity of the job.'),
    created_at=_value(S, 'Start time, ISO 8601 UTC.'),
    finished_at=_value(S, 'End time, ISO 8601 UTC; absent while running or awaiting review.'),
    timeout_s=_value(N, 'Work wall-time limit, seconds; excludes model and review waits.'),
    call_limit=_value([I, NULL], 'Tool-call limit; null when unlimited.'),
    allow_direct_actions=_value(B, 'Whether the program could execute actions without main-actor review.'),
    action_authorization_scope=_value(S, 'Scope of the action authorization.'),
    calls=_value(I, 'Tool calls made by the program.'),
    model_wait_s=_value(N, 'Wall time waiting for the program\'s own model calls, seconds.'),
    review_wait_s=_value(N, 'Wall time waiting for main-actor review, seconds.'),
    returncode=_value([I, NULL], 'Process exit code; negative for a signal; null when unknown.'),
    stop_confirmed=_value(B, 'Whether the process is confirmed stopped.'),
    action_summary=_object('Summary of broker action replies; not task success.', dict(
        action_calls=_value(I, 'Action tool calls.'),
        status_counts=_value(O, 'Count of action replies per receipt status.'),
        executed_native_steps_known=_value(I, 'Known native steps of actions and settling.'),
        action_native_steps_known=_value(I, 'Known native steps of actions.'),
        actions_without_step_count=_value(I, 'Actions whose native-step count is unknown.'),
        review_rejections=_value(I, 'Previews rejected by the main actor.'),
        recent_actions=_value(A, 'Recent action evidence.'),
        task_success=_value(S, 'not_assessed.'))),
    call_summary=_object('Summary of broker tool calls.', dict(
        calls=_value(I, 'Tool calls.'),
        failed_calls=_value(I, 'Failed tool calls.'),
        recent_errors=_value(A, 'Up to 8 latest call errors.'))),
    pending_review=_value(O, 'Preview awaiting main-actor review.'),
    visual_feedback=_value(A, 'Images the program requested for the main actor.'),
    final_observation_id=_value([S, NULL], 'Latest observation returned to the program; null when none.'),
    final_observation=_value([O, NULL], 'Final observation forwarding record; null when none.'),
    source_ref=_value(S, 'Artifact of the program source.'),
    stdout_ref=_value(S, 'Artifact of the complete stdout.'),
    stderr_ref=_value(S, 'Artifact of the complete stderr.'),
    stdout_text=_value(S, 'stdout text, possibly truncated; program-authored, not action evidence.'),
    stderr_text=_value(S, 'stderr text, possibly truncated.'),
    stdout_truncated=_value(B, 'Whether stdout_text omits output.'),
    stderr_truncated=_value(B, 'Whether stderr_text omits output.'))

PROGRAM_STATUS = dict(PROGRAM_JOB,
    stdout_excerpt=_value(O, 'Summary: byte ranges of the stdout excerpt.'),
    stderr_excerpt=_value(O, 'Summary: byte ranges of the stderr excerpt.'),
    skill_output=_value(O, 'Structured output of a fixed built-in skill.'),
    full_result=FULL_RESULT,
    resume=_value(O, 'Arguments to resume a program awaiting review.'),
    result_semantics=_value(S, 'Meaning of program status.'))

# history_read_trace returns saved trace JSON or a selected JSON value, which
# can be an array, scalar or null; only JSON serialization is required.
UNCONSTRAINED = dict(history_read_trace='saved trace JSON or a JSON-pointer selection of any type')

CONTRACTS = dict(
    tool_guide=dict(data=_object('Complete input field guide of one published tool (layered catalog).', dict(
        tool=_value(S, 'Tool whose guide is returned.'),
        topic=_value([S, NULL], 'Requested topic; null returns the whole guide.'),
        entries=_list('Guide entries, one field path each: meaning, units, defaults, bounds and branch rules.',
                      _value(S, 'One field path with its guide text.')),
        examples=_list('Published input examples of the tool.', _value(O, 'One example input object.')),
        topic_matched=_value(B, 'False when no entry named the topic and the whole guide is returned.')),
        required=('tool', 'topic', 'entries', 'examples'))),
    robot_action=dict(action=UNIFIED_ACTION, plan=PLAN, data=ACTION_INSPECTION),
    robot_wait=dict(action=ACTION),
    robot_policy_step=dict(action=ACTION),
    robot_policy_delegate=dict(delegation=DELEGATION),
    robot_observe=dict(observation=OBSERVATION),
    history_view_observation=dict(data=OBSERVATION_METADATA),
    robot_contact_overlay=dict(data=_object('Contact-reference overlay metadata; derived projections, not contact evidence.', dict(
        observation_id=_value(S, 'Observation drawn on.'),
        camera_id=_value(S, 'Camera drawn on.'),
        overlay_ref=_value(S, 'Artifact of the overlay image.'),
        original_rgb_ref=_value([S, NULL], 'Artifact of the original RGB image.'),
        current=_value(B, 'Whether the observation is the current one.'),
        world_kind=_value(S, 'Always observed.'),
        frame_id=_value(S, 'Reference frame of position_m.'),
        image_frame_id=_value([S, NULL], 'Capture frame id of the image.'),
        sim_step=_value([I, NULL], 'Simulation step of the image.'),
        step_domain=_value([S, NULL], 'Domain of sim_step.'),
        width=_value(I, 'Image width, pixels.'),
        height=_value(I, 'Image height, pixels.'),
        original_image_coordinates=_value(B, 'Pixel coordinates refer to the original image.'),
        image_resized=_value(B, 'Whether the overlay image was resized.'),
        camera_pose_source=_value([S, NULL], 'Camera pose source.'),
        source_kind=_value(S, 'Derivation of the drawn references.'),
        arms=_value(O, 'Per-arm selected contact references and point indices.'),
        points=_list('Projected reference points.', _object('Projected point.', dict(
            arm=_value(S, 'Arm (left/right).'),
            reference=_value(S, 'Contact reference, e.g. pad_center.'),
            position_m=_value(A, 'Position [x, y, z] in frame_id, metres.'),
            pixel_xy=_value([A, NULL], 'Projected [u, v] in original pixels; null when not finite.'),
            depth_m=_value(N, 'Depth along the camera optical axis, metres; negative behind the camera.'),
            in_front_of_camera=_value(B, 'Whether depth_m is positive.'),
            in_image=_value(B, 'Whether the projection lies inside the image (frustum only, no occlusion check).'),
            source_kind=_value(S, 'derived_from_measured_link6 or caller_request.')))),
        arrows=_list('Drawn approach arrows between point indices.', _object('Approach arrow.', dict(
            arm=_value(S, 'Arm (left/right).'),
            kind=_value(S, 'actual_approach or requested_approach.'),
            start=_value(I, 'Index into points.'),
            end=_value(I, 'Index into points.'),
            drawn=_value(B, 'Whether both ends are in the image.')))),
        approach_arrow_length_m=_value(N, 'Approach arrow length, metres.'),
        observation_state=_value(O, 'Saved robot state at the image capture time.'),
        physics_advanced=_value(B, 'Always false: drawing does not advance physics.'),
        executed_native_steps=_value(I, 'Always 0.'),
        executed_control_steps=_value(I, 'Always 0.')),
        required=['observation_id', 'camera_id', 'overlay_ref', 'points', 'arms'])),
    program_run=dict(program=_object('Original program job record (process facts, not action outcomes).',
        PROGRAM_JOB, required=['job_id', 'status'])),
    program_status=dict(data=_object('Program job record; summary detail adds excerpts and omits provenance.',
        PROGRAM_STATUS, required=['job_id', 'status'])),
    task_plan=dict(data=_object('Task plan save/read result.', dict(
        plan_id=_value(S, 'Plan id, e.g. P3.'),
        status=_value(S, 'saved or unchanged.'),
        required=_value(B, 'Read: whether a plan is required before physical actions.'),
        plan=_value([S, O, NULL], 'Read: free-form plan text; legacy plans are objects; null before any plan.'),
        attention=_value(A, 'Read: attention entries.'),
        ledger=_value(A, 'Read: evidence ledger entries.'),
        plan_observation_id=_value([S, NULL], 'Read: observation on which the plan text was written.'),
        attention_observation_id=_value([S, NULL], 'Read: observation on which attention was written.'),
        ledger_observation_id=_value([S, NULL], 'Read: observation on which the ledger was written.')))),
    robot_capabilities=dict(data=_object(
        'Backend capability document; a repeated summary carries only capabilities_delivery and current_action_budget.', dict(
        backend=_value(S, 'Robot backend.'),
        robot_model_id=_value(S, 'Robot model id.'),
        tcp_config_id=_value(S, 'Tool-centre-point configuration of pose targets.'),
        frame_id=_value(S, 'World frame of poses.'),
        world_kind=_value(S, 'Always observed.'),
        limits_profile=_value(S, 'Joint/step limit profile.'),
        observation_profile=_value(S, 'Observation permission profile of the run.'),
        motion_execution_mode=_value(S, 'resampled or feedback.'),
        native_control_hz=_value(N, 'Native control rate, Hz.'),
        physics_dt_s=_value(N, 'Physics tick, seconds.'),
        native_physics_steps_per_action=_value(I, 'Physics ticks per native action.'),
        motion_duration_limit_s=_value(N, 'Maximum motion duration of one request, seconds.'),
        max_waypoints=_value(I, 'Maximum waypoints per path.'),
        max_simultaneous_arms=_value(I, 'Arms that can move in one request.'),
        dual_arm=_value(B, 'Whether both arms can move together.'),
        arm_joint_names=_value(O, 'Joint names per arm.'),
        arm_joint_limits_rad=_value(O, 'Per-arm [lower, upper] joint limits, radians.'),
        arm_max_step_rad=_value(O, 'Per-arm maximum joint change per native action, radians.'),
        action_budget=_value(O, 'Native task step limit (native_step_limit, null when unknown).'),
        current_action_budget=ACTION_BUDGET,
        capabilities_delivery=_value(O, 'Summary delivery: inline or unchanged in this model context.'),
        unified_action=_value(O, 'robot_action availability, modes and limits.'),
        action_speed=_value(O, 'Speed-scale ranges and presets.'),
        bounded_wait=_value(O, 'robot_wait limits.'),
        versions=_value(O, 'Backend component versions.')))),
    robot_information=dict(data=_object('Static robot reference, resource listing or calibration disclosure.', dict(
        robot_id=_value([S, NULL], 'Robot model id; null when the backend reports none.'),
        section=_value(S, 'Requested section.'),
        available=_value(B, 'Whether the requested information exists.'),
        briefing=_value(S, 'Default section: physical briefing text.'),
        urdf=_value(S, 'Default section: verbatim URDF.'),
        reference=_value(O, 'Physical reference records.'),
        calibration=_value(O, 'Disclosed calibration information.'),
        resources=_value(A, 'Downloadable resources.'),
        resource=_value(O, 'Selected resource.'),
        resource_ref=_value(S, 'Artifact of the selected resource.'),
        description=_value(S, 'Geometry image description.'),
        error=_value(S, 'Why the information is unavailable.'),
        message=_value(S, 'Unavailable explanation.')), required=['robot_id', 'available'])),
    robot_action_status=dict(data=_object('Recorded request and receipt; summary detail projects the receipt.', dict(
        request=REQUEST, receipt=RECEIPT, full_result=FULL_RESULT), required=['request', 'receipt'])),
    history_list_actions=dict(data=dict(anyOf=[
        _object('Summary page: action index, or interaction blocks when anchor was given.', dict(
            detail=_value(S, 'Always summary.'),
            actions=_list('Selected actions in request order.', _object('Index entry or interaction block.', dict(
                action_id=_value(S, 'Index: action id.'),
                status=_value(S, 'Index: receipt status.'),
                request=_value(O, 'Block: action identity of the request.'),
                receipt=_value(O, 'Block: projected receipt.'),
                episode_action_index=_value(I, 'Block: zero-based index in this episode.'),
                anchor_reasons=_value(A, 'Block: why the action was selected.'),
                observation_ids=_value(O, 'Block: decision, execution_start and result observation ids.'),
                evidence_gaps=_value(A, 'Block: missing observation boundaries.'),
                measured_delta=_value([O, NULL], 'Block: measured state change from execution start to result.'),
                full_result=FULL_RESULT), required=['full_result'])),
            total=_value(I, 'Matching actions.'),
            start=_value(I, 'Index of the first returned action.'),
            limit=_value(I, 'Page size.'),
            next_start=_value([I, NULL], 'start of the next page; null on the last page.'),
            order=_value(S, 'recorded_action_request_order.'),
            anchor=_value(S, 'Blocks: events, failures, gripper_commands or all.'),
            scope=_value(S, 'Blocks: current_episode.'),
            overview=_value(O, 'Blocks: counts of the returned page.'),
            observations=_value(O, 'Blocks: observation states keyed by observation id.'),
            episode_action_total=_value(I, 'Blocks: actions in this episode.'),
            omitted_actions_after=_value(I, 'Blocks: requests after the last returned block.'),
            full_result=FULL_RESULT),
            required=['detail', 'actions', 'total', 'start', 'limit', 'next_start', 'order']),
        _list('Full detail: every recorded request and receipt.', _object('Recorded action.', dict(
            request=REQUEST, receipt=RECEIPT), required=['request', 'receipt']))],
        description='Paged summary object, or the full list of recorded actions.')),
    history_annotate_action=dict(data=_object('Saved action note.', dict(
        action_id=_value(S, 'Annotated action.'),
        note=_value(S, 'Note text.'),
        source_kind=_value(S, 'actor_judgment.'),
        correlation=_value(O, 'Tool call that wrote the note.')), required=['action_id', 'note', 'source_kind'])),
    geometry_query=dict(data=dict(anyOf=[GEOMETRY, DEPTH],
        description='Geometry result; depth kinds return a depth-derived result.')),
    depth_query=dict(data=DEPTH),
    vision_compare=dict(data=_object('RGB feature-motion estimate between two observations.', dict(
        source_kind=_value(S, 'rgb_feature_estimate.'),
        sources=_value(A, 'Before and after images used.'),
        target=_object('Tracked target region.', dict(
            status=_value(S, 'Tracking status, e.g. tracked_features or lost.'),
            reason=_value(S, 'Why tracking failed.'),
            displacement_px=_value([A, NULL], 'Median [du, dv] image displacement, pixels; null when lost.'),
            accepted_points=_value(I, 'Consistent feature tracks.'))),
        reference=_value([O, NULL], 'Tracked reference region; null without reference_roi.'),
        relative_displacement_px=_value([A, NULL], 'Target minus reference displacement, pixels.'),
        limitations=_value(A, 'Evidence limitations.')))),
    vision_review=dict(data=_object('Fallible model visual judgment over selected evidence.', dict(
        model=_value(S, 'Reviewer model.'),
        answer=_value(O, 'Schema-validated reviewer answer.'),
        sources=_value(A, 'Images reviewed.'),
        source_kind=_value(S, 'model_visual_judgment or model_operation_judgment.'),
        limitations=_value(S, 'Evidence limitations.'),
        phase=_value(S, 'Operation mode: reviewed phase.'),
        request_id=_value(S, 'Operation mode: review request id.'),
        assessment_constraints=_value(A, 'Operation mode: constraints applied to the answer.'),
        task_completion=_value(S, 'Operation mode: not_assessed.')),
        required=['model', 'answer', 'sources', 'source_kind'])),
    operation_memory=dict(data=_object('Operation-memory help, query or record result.', dict(
        scope=_value(S, 'Query/record: task or shared.'),
        records=_list('Query: matching records.', MEMORY_RECORD),
        total_matches=_value(I, 'Query: matches before the limit.'),
        record=_value(O, 'Record: the saved public record; help: a request template.'),
        semantics=_value(S, 'Meaning of lessons.')))))
