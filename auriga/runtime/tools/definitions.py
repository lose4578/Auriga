"""Existing wire definitions, retained for trusted legacy integrations."""
from ...cortex.context.task_planning import ATTENTION_SCHEMA, EVIDENCE_LEDGER_SCHEMA, PLAN_SCHEMA
from ...cortex.context.operation_verification import TOOL_PROPERTIES, TOOL_REQUIRED
from ...core.tool_output import DETAIL_SCHEMA
from ...cortex.context import delegation_gate
from ...cortex.context.delegation_gate import tool_description

def tool(name, description, properties=None, required=(), *, defer_loading=False, **schema_rules):
    return dict(type="function", name=name, description=description,
                **({"deferLoading": True} if defer_loading else {}),
                inputSchema=dict(type="object", properties=properties or {}, required=list(required),
                                 additionalProperties=False, **schema_rules))


TOOLS = [
    tool("task_plan", "Submit the required task-level plan before the first robot action, revise it when useful, or read this episode's current plan. Stages describe goals and observable advance_when evidence. suggested_tools is optional and advisory; execution may use other tools or adapt the plan. No motion or evaluator access. Submit requires the current observation_id and plan; read requires only operation.",
         dict(operation=dict(type="string", enum=["read", "submit"]),
              observation_id=dict(type="string"), plan=PLAN_SCHEMA), ("operation",)),
    tool("robot_observe", "Get permitted RGB, measured robot state and budgets; no hidden object state or score. Summary references previously delivered unchanged calibration/conventions; static_refs.bundle spans shared fields across metadata sections. Current camera poses and measurements stay inline. detail='full' repeats full geometry. The world advances only during gateway actions.",
         dict(camera_id=dict(type="string"),
              roi=dict(type="array", items=dict(type="integer"), minItems=4, maxItems=4),
              scale=dict(type="integer", minimum=1, maximum=4, default=1),
              detail=DETAIL_SCHEMA)),
    tool("robot_capabilities", "Read exact frame, TCP, supported paths, limits and gripper semantics. Static capabilities are sent once per model context; later reads report unchanged status and current budget. Use detail=full to repeat them.", dict(detail=DETAIL_SCHEMA), defer_loading=True),
    tool("robot_information", "Read robot information without moving or advancing time. Initialization supplies the full briefing, original URDF text, supplemental parameters, a readable native mesh ZIP and current camera/base calibration. Default repeats them. calibration rereads the latest published observation's intrinsics, distortion, hand-eye and camera/base transforms; it does not refresh sensors. kinematics, geometry, dynamics and control return detailed records, hardware returns optional comparisons, all returns the complete numerical reference plus hardware. geometry_image is an optional static illustration. resources lists native geometry and CAD files; resource plus resource_id returns a file reference readable with client.artifact. Read geometry_model as ZIP, parse its assembly and native meshes for exact geometry; no images or prose shapes substitute for them. Omit robot_id for the current robot. Calibration for another robot is unavailable.",
         dict(section=dict(type="string", enum=["default", "kinematics", "geometry", "dynamics", "control", "hardware", "all", "calibration", "geometry_image", "resources", "resource"]),
              robot_id=dict(type="string"), resource_id=dict(type="string")), defer_loading=True),
    tool("robot_contact_overlay", "Draw measured pad_center/finger_middle/fingertip and selected contact references on an existing calibrated RGB image. Optional selections maps arms to contact_region OR contact_depth_m. Optional targets maps arms to requested selected-contact position_m and optional approach_axis_world. Labels distinguish measured and requested positions; robot geometry only, no object/contact/occlusion inference. No simulation advancement. Uses original image coordinates; useful when depth/selected reference is unclear.",
         dict(observation_id=dict(type="string"), camera_id=dict(type="string"),
              arms=dict(type="array", minItems=1, maxItems=2, uniqueItems=True, items=dict(enum=["left", "right"])),
              selections=dict(type="object"), targets=dict(type="object")), ("observation_id", "camera_id")),
    tool("robot_action_status", "Read action status without re-execution. Summary omits repeated provenance and passing guard diagnostics; errors, partial execution and original request remain. detail='full' returns the exact stored request/receipt.", dict(action_id=dict(type="string"), detail=DETAIL_SCHEMA), ("action_id",), defer_loading=True),
    tool("robot_wait", "Advance 1–50 native simulation steps while holding both arms at their measured joints and retaining gripper commands. At 25 Hz, 25 steps is 1 simulated second. Use short waits to observe autonomous motion, a demonstration, or settling; robot_observe and real-time sleep do not advance simulation. Every step consumes the official action budget. Returns chronological head-camera samples followed by final RGB and measured state. The reply identifies whether that final observation is still current; replaying an old action does not advance time. No automatic phase detection, order inference, grasp judgment or retry.",
         dict(observation_id=dict(type="string"), native_steps=dict(type="integer", minimum=1, maximum=50),
              action_id=dict(type="string"), timeout_s=dict(type="number", minimum=.1, maximum=300)),
         ("observation_id", "native_steps")),
    tool("history_list_actions", "Read a paged action status index, including failures and partial execution. Default start=0, limit=25 (max100); follow next_start. Each item gives a robot_action_status detail='full' lookup. detail='full' without pagination returns the original complete list. No execution.",
         dict(detail=DETAIL_SCHEMA, start=dict(type="integer", minimum=0), limit=dict(type="integer", minimum=1, maximum=100),
              anchor=dict(type="string", enum=["events", "failures", "gripper_commands", "all"]))),
    tool("history_view_observation", "Read immutable saved RGB at original resolution. observation_is_current distinguishes historical frames; use current_action_budget rather than the saved capture-time budget. Optional camera_id + roi:[x,y,width,height] + scale (1..4) returns an enlarged crop with exact source and pixel transform. Summary references previously delivered unchanged intrinsics/conventions while keeping the saved camera world poses inline; detail='full' returns full calibration/kinematic metadata. No world advancement. Geometry pixels use original-image coordinates; convert crops with the recorded transform.", dict(observation_id=dict(type="string"), camera_id=dict(type="string"), roi=dict(type="array", items=dict(type="integer"), minItems=4, maxItems=4), scale=dict(type="integer", minimum=1, maximum=4), detail=DETAIL_SCHEMA), ("observation_id",)),
    tool("history_read_trace", "Read saved planned/dispatched/measured traces. Summary keeps small values and windows long sequences (default25, max100); windows report exact JSON Pointer path, total and next_start. Use path/start/limit for a selected sequence, or detail='full' without selectors for the exact original. Values retain precision; planned values are not measurements. Evaluator/privileged artifacts remain inaccessible.",
         dict(action_id=dict(type="string"), kind=dict(type="string", enum=["planned_trace", "dispatched_trace", "actual_trace", "before_state"]),
              detail=DETAIL_SCHEMA, path=dict(type="string"), start=dict(type="integer", minimum=0),
              limit=dict(type="integer", minimum=1, maximum=100)), ("action_id", "kind"), defer_loading=True),
    tool("history_annotate_action", "Append an actor judgment to an action; never alters its measurement or execution status.", dict(action_id=dict(type="string"), note=dict(type="string")), ("action_id", "note"), defer_loading=True),
    tool("geometry_query", "Calibrated geometry with provenance and degeneracy checks. First use request={kind:'help'} for fields. Example {kind:'pixel_ray',observation_id:'...',camera_id:'cam_head',pixel:[u,v]}; triangulate uses observation_ids, camera_id/camera_id_b and pixels_a/pixels_b. No scene required for rays/triangulation. Projection queries also return overlay images: forward their image URLs separately, never print raw base64.", dict(request=dict(type="object")), ("request",)),
    tool("program_run", "Run bounded Python with AurigaClient; 1 GiB memory, numerical libraries use one thread. Write relative files only in your per-job directory. Use from auriga_client import AurigaClient; client=AurigaClient(); r=client.call(tool_name,arguments). r contains success, kind, data and display contentItems. Outer success only describes tool dispatch, not action or task success. Read typed data directly: for kind='action', r['data']['receipt'] holds status, reason/error, partial-execution evidence and result_observation_id; kind='plan' carries the preview or planning rejection in data. Missing/untyped data remains unknown; never infer execution from display text. Check all feedback/image items. Return control on non-completed/unknown status or unavailable fresh observation; do not blindly retry. completed does not prove grasp/contact or native task success. Program succeeded means process exit 0 only. Omitted robot_action preview/direct and robot_wait timeouts are capped to remaining program work time minus the 5-second action reserve and 1-second dispatch margin before planning. Explicit timeouts and cached plan deadlines are unchanged; admission still requires 5 seconds beyond the action deadline, including after expensive planning. Read-only preview remains available without execution reserve. client.artifact(ref) returns permitted bytes. All real actions use the same gateway.",
         dict(source=dict(type="string"), source_ref=dict(type="string"), timeout_s=dict(type="number", minimum=1, maximum=300), max_calls=dict(type="integer", minimum=1, maximum=100), detail=DETAIL_SCHEMA), ()),
    tool("program_status", "Read a prior program job without execution. Summary keeps status and both ends of stdout/stderr with byte ranges and full artifact references; logs are program-authored, not verified task outcomes. detail='full' reads complete original stdout/stderr.", dict(job_id=dict(type="string"), detail=DETAIL_SCHEMA), ("job_id",), defer_loading=True),
    tool("vision_compare", "Optional read-only local RGB texture tracking. Use when measuring visible patch motion could change your next decision, e.g. suspected slip during a short lift/transfer or whether a placed object followed the withdrawing gripper. Skip when RGB and measured feedback already answer the question; it is not a required manipulation checkpoint or a substitute for operation_verify. Reuse saved action observations from the same episode and camera; no new capture, robot motion or model request is made. Choose a textured patch on the target in the BEFORE original image, excluding fingers/background. Prefer a clear wrist close-up for a held object; tiny, textureless, occluded or strongly rotated patches often provide no useful correspondence. Returns target.status, pixel displacement and match-quality counts; an optional reference patch gives target-minus-reference displacement only when both tracks succeed. tracked_features supports texture correspondence, not semantic identity, retained grasp, release, contact, metric motion or task success. lost/ambiguous means insufficient correspondence, not a dropped object or failed action; null displacement supplies no motion estimate. Combine with RGB and measured feedback; choose another view or a changed ROI only if it can resolve remaining uncertainty. Do not repeat an unchanged image pair and ROI. Summarize status, displacement and quality rather than echoing point arrays.",
         dict(before_observation_id=dict(type="string", description="Saved BEFORE actor observation from the current episode; select the ROI in this image."),
              after_observation_id=dict(type="string", description="Saved AFTER actor observation from the same episode, at the same or a later simulation step; use a changed frame for motion evidence."),
              camera_id=dict(type="string", description="The same camera in both observations, e.g. cam_right_wrist or cam_head; image dimensions must match."),
              roi=dict(type="array", items=dict(type="integer"), minItems=4, maxItems=4,
                       description="BEFORE original-image pixels [x,y,width,height], fully in bounds, width/height >=4. Select target texture only. Convert enlarged-crop coordinates back to the original image."),
              reference_roi=dict(type="array", items=dict(type="integer"), minItems=4, maxItems=4,
                                 description="Optional identifiable landmark texture in the BEFORE original image, with the same bounds as roi. relative_displacement_px is target minus reference, or null if either track is unavailable. This is image-relative motion, not a world-motion or depth estimate."),
              max_motion_px=dict(type="number", minimum=1, maximum=320, default=160,
                                 description="Maximum accepted target/reference pixel translation magnitude; larger apparent motion is ambiguous. This does not change the matching search radius or recover lost features.")),
         ("before_observation_id", "after_observation_id", "camera_id", "roi"), defer_loading=True),
    tool("robot_policy_step", "Delegate one atomic local operation to the matched RoboDojo ARX X5 pi0.5 policy using current three-camera RGB and proprioception. Astra owns the overall task, target selection, approach and handoff. Supply prompt describing this local intent (for example grasp closure, short lift, placement adjustment or release); task context may accompany it. Execute only prefix rows (1..50), then return fresh feedback to Astra. Default prefix=5 is 0.20 seconds of simulated control. Do not delegate an entire task or loop this tool without reviewing feedback and deciding the next local goal. Text such as '1 cm' or an arm name is NOT a movement/arm constraint. Available only when this run enables pi0.5. Native success stops execution immediately.",
         dict(observation_id=dict(type="string"), prompt=dict(type="string"), prefix=dict(type="integer", minimum=1, maximum=50),
              timeout_s=dict(type="number", minimum=.1, maximum=300), action_id=dict(type="string")), ("observation_id", "prompt"), defer_loading=True),
    tool("robot_policy_delegate", tool_description("none"),
         dict(observation_id=dict(type="string"), instruction=dict(type="string", minLength=1, maxLength=2000),
              subtask_note=dict(type="string", minLength=1, maxLength=1000),
              max_policy_steps=dict(type="integer", minimum=1, maximum=400),
              delegation_id=dict(type="string", minLength=1, maxLength=128)), ("observation_id",)),
    tool("vision_review", "Get a GPT-6.1 Sol second opinion on two saved RGB images: visible target motion and possible retention. Supply the target's BEFORE roi=[x,y,width,height] and a short target description. This is fallible image interpretation, not metric measurement, contact or task scoring. Await the result directly.",
         dict(before_observation_id=dict(type="string"),after_observation_id=dict(type="string"),camera_id=dict(type="string"),
              roi=dict(type="array",items=dict(type="integer"),minItems=4,maxItems=4),target=dict(type="string"),
              condition=dict(type="string",enum=["retained_with_gripper","released_from_gripper"])),
         ("before_observation_id","after_observation_id","camera_id","roi","target"), defer_loading=True),
]

TOOLS += [
    tool("depth_query", "Explicit RGB-D numerical queries. Start with request={kind:'help'}. "
         "Query depth_points, region_geometry, grasp_candidates, place_candidates or compare_regions. "
         "register_target binds current observation_id,camera_id,roi and optional arm to target_id; "
         "subsequent geometry queries may use target_id. Numeric summaries only, no point clouds. "
         "Candidate poses are hypotheses; unknown surfaces/contact remain unknown.",
         dict(request=dict(type="object")), ("request",), defer_loading=True),
]


from copy import deepcopy


_PROGRAM_OUTPUT_HELP = (
    " In summary, stdout and stderr retain all available saved log text; no head/tail byte cap removes middle diagnostics. "
    "*_truncated and *_excerpt.source_complete report incomplete sources or media omission. "
    "Inline image data URLs are replaced by explicit placeholders in summary only; the excerpt reports original byte size, removed-media ranges, and whether its ranges address presentation text. "
    "Repeated unchanged log text over 256 bytes from the same job in this model context is omitted with *_delivery.status='unchanged'; process/action/review status is still returned. "
    "Read program_status with the returned job_id and detail='full' to repeat logs or include inline media. "
    "full always reads all available saved log text, including on repeated calls; displayed references may use aliases. "
    "Read stdout_ref/stderr_ref with client.artifact for exact saved bytes."
)
_PROGRAM_VISUAL_HELP = (
    " program_run/program_status automatically forward images from read-only or preview program tools "
    "(including history views and geometry overlays), alongside pending-review/final-observation images. "
    "Images are separate inputImage items; do not print base64 to expose them. "
    "Forwarding preserves the returned imageUrl, ROI and labels without re-encoding. "
    "visual_feedback links supplementary images to their source tool/call, arguments and saved reply_ref; historical images remain historical. "
    "Repeated unchanged supplementary images in the same model context may be omitted in summary with previous_tool_call_id "
    "and a full-result lookup; full or a reset model context delivers them again."
)

DESCRIPTIONS = {
    "task_plan": "Save or read this episode's brief ordinary-language task plan, dynamic attention list and short evidence ledger. Submit plan before the first physical action, including a wait, and always include ledger explicitly (use [] when no material prerequisite is unresolved). Suggested plan headings: Task understanding; Approach + completion. Keep only currently relevant concerns, evidence/adjustments and check times in attention. When a material task-result prerequisite is unresolved and a later decision depends on it, keep only one to three short ledger rows. The ledger has at most 3 task prerequisites or expected events with status=pending|supported|contradicted|unknown, a short permitted evidence basis and the next distinguishing check. After contradicted evidence, change the contact, route or target mechanism, or state that no distinguishing next step is supported. It is an advisory decision boundary: completed motion, preview success and closed gripper do not prove an object result, and pending rows do not block useful actions. Add, replace or remove rows as evidence changes, without updating after every motion. Prefer normal task-progress actions' final observations; add a probe only when missing evidence affects the next decision. Plans, attention and ledger are advisory and do not execute or constrain later actions. update accepts plan, attention, ledger or any combination; omitted fields are retained and [] clears a list; when a ledger value changes, include the replacement ledger. Repeating the same field values at the same observation returns status=unchanged without a new revision; the same values at a newer observation record a new judgment source. submit/update return a plan ID and acknowledgement; read returns the current plan, attention, ledger and field-source observation IDs, and accepts only operation.",
    "robot_observe": "Capture current permitted RGB, same-frame measured joint/EE/gripper state and budgets. Returns observation_id, state/calibration metadata and images. Optional camera_id with roi/scale inspects a zoomed single-view crop of the current observation on demand without advancing simulation or replacing the current observation_id. No hidden object/score access. summary avoids repeating unchanged calibration; full repeats it.",
    "robot_capabilities": "Read coordinate/TCP conventions, supported action modes, native limits, speed/gripper semantics and current budget without moving. Startup supplies capabilities; later summary reads may report unchanged static values. full repeats the complete capability record.",
    "robot_information": "Read static robot knowledge or the latest published calibration without motion. default returns briefing, original URDF, supplemental physical parameters and native geometry reference; named sections return their detailed records. calibration does not capture a new frame. resources lists file IDs; resource returns an actor-readable file reference for client.artifact. geometry_image returns an illustrative image, not a measured scene. Unknown robots/resources return unavailable.",
    "robot_contact_overlay": "Draw measured contact-section references and caller-selected targets on saved calibrated RGB. Returns labeled robot-geometry metadata and an overlay image; measured and requested positions remain distinct. Uses original-image coordinates and no simulation steps. Image overlap does not establish object contact or clearance.",
    "robot_action_status": "Read a saved action request and execution receipt without executing it again. summary keeps execution/partial/failure evidence and a full-result lookup; full reads original stored evidence. Inspect receipt.status and result_observation_id; tool dispatch success does not establish physical or task success.",
    "robot_wait": "Advance 1..50 native simulation steps while holding measured arm joints and retaining gripper commands. At 25 Hz, 25 steps is one simulated second; every step consumes the native budget. Returns execution receipt, chronological camera samples, final RGB/state and whether that observation remains current. Use for external motion or settling; observations and wall-clock sleep do not advance simulation. Requires the initial task plan; no automatic phase detection or retry.",
    "history_list_actions": "Read action history in recorded request order without execution or new observation. Optional anchor ('events', 'failures', 'gripper_commands', 'all') returns a top-level episode overview and compact observation-action-receipt-observation blocks with typed execution/partial/failure evidence, observation IDs, measured_delta across execution_start to result, gripper command locations and explicit evidence gaps. The latest action receipt is already returned inline by robot_action; use anchor='failures', 'gripper_commands' or 'events' on demand during multi-step recovery or long-horizon tasks when you need to compare earlier attempts, check cumulative measured_delta, or audit prior gripper commands before planning the next adjustment. Use robot_action_status detail='full' for an action's full stored request/receipt and history_read_trace for saved traces. Without anchor, summary returns a paged status index with total/next_start; full returns the original complete list and cannot be combined with anchor/start/limit.",
    "history_view_observation": "Read saved permitted RGB and capture-time measured state/calibration without advancing simulation. Returns images, observation_is_current and separately identified current budget. camera_id selects one view; roi/scale crop or enlarge it while recording the pixel transform. Historical frames remain historical; full repeats saved geometry metadata.",
    "history_read_trace": "Read a saved action's planned, dispatched, measured or before-state JSON evidence without execution. summary preserves exact values and pages long sequences, reporting JSON Pointer paths and next_start. full returns original evidence and cannot be combined with path/start/limit. Missing trace returns unavailable; planned commands are not measurements.",
    "history_annotate_action": "Append your textual judgment to an existing action and return the saved annotation. Does not alter measured state, receipts or execution status. Keep interpretations and uncertainty separate from physical evidence.",
    "geometry_query": "Compute calibrated RGB/proprioception geometry from saved permitted observations. request.kind selects camera/FK, rays, triangulation, pixel_plane intersections with a caller-assumed plane, point/path projection, tracking, batch (1..32 geometry queries in one call via queries=[...]) or explicitly reconstructed-scene queries. Prefer triangulate across two calibrated views (such as head and wrist) when a target feature is visible and matchable in both views, since it measures 3D position without assuming a plane height. Use pixel_plane on demand when a target or destination lies on a known support/table/container plane or when single-view occlusion/weak texture prevents reliable two-view matching: it returns a conditional point (not measured depth), d_point_d_normal_offset_m and cross_camera_projections in other calibrated views, plus optional plane_origin_alt_m sensitivity (assumption_delta_m) and optional target_camera_id overlay image; verify the plane hypothesis via cross_camera_projections or target_camera_id before choosing robot_action targets. Returns numerical value, source/quality/degeneracy evidence and projection images when applicable. build_scene returns a proposed patch without applying it or creating a saved scene_version; scene queries require an existing saved version. No native object truth, movement or physics rollout. Projection images are returned as independent image items. Depth kinds use the depth_query contract only when RGB-D is enabled.",
    "program_run": "Start bounded Python or resume a job paused for your action review. Start supplies source/source_ref; resume supplies job_id, review_id and approve/reject, without replacing the invocation. Defaults: 180 seconds work time, 1 GiB memory, numerical libraries use one thread; own helper-model/review waits do not consume work time. New robot actions pause for actor review unless allow_direct_actions=true at start. Approve via this tool, never separately execute the program's cached plan. Returns job/process status, broker action summary, authored stdout/stderr, pending review and final observation. Process exit 0 or tool success does not prove robot/task success. In source use from auriga_client import AurigaClient, ActionIncompleteError; client.action(..., require_completed=True) raises ActionIncompleteError (with .receipt and .response) whenever receipt.status != 'completed' so loops stop immediately on failure or pause; client.call(name,args) returns success/kind/data and display contentItems. For kind='action', read r['data']['receipt'] (one receipt level); for kind='plan', data is the preview or planning rejection. Missing/untyped data remains unknown; do not infer execution from display text. client.artifact(ref) returns permitted bytes; print the fields needed for the next decision (IDs, status/reason, measured errors and selected diagnostics), retaining full artifact references for deeper reads. Leave at least 5 seconds between each inner action timeout and remaining program time; inspect partial/unknown execution and missing observations before any further action. Tool calls are not capped; program tools cannot be called recursively from a program." + _PROGRAM_OUTPUT_HELP + _PROGRAM_VISUAL_HELP,
    "program_status": "Read an existing Python job without starting or resuming it. Returns process status, broker action facts, review state and program-authored stdout/stderr. Once the process is terminal, repeated summary reads add no process progress; use full when saved detail or supplementary images are needed. awaiting_review requires program_run resume rather than polling for completion. Authored output and process success are not verified robot/task outcomes." + _PROGRAM_OUTPUT_HELP + _PROGRAM_VISUAL_HELP,
    "vision_compare": "Track a caller-selected textured patch between saved RGB frames without a new capture, action or model request. Returns texture-correspondence status, pixel displacement and match quality; optional reference tracking gives target-minus-reference displacement. Select target texture in BEFORE original-image pixels, excluding fingers/background. lost/ambiguous means insufficient correspondence, not a dropped object. Correspondence does not prove semantic identity, grasp, contact, metric motion or task success; use optional vision_review when available for uncertain visual interpretation. Do not repeat an unchanged frame pair and ROI.",
    "vision_review": "Request an optional helper-model assessment using saved permitted RGB. evidence_mode=images (default) judges the described target's retention/release from a before/after pair; operation also checks ordered action receipts and measured state for a named manipulation phase/expected_result. Returns a fallible assessment, evidence gaps and follow-up advice; never executes, authorizes motion or scores the task. Calls consume the shared model budget. Await the answer and combine it with actual execution feedback.",
    "robot_policy_step": "Delegate one local intent to the matched ARX X5 pi0.5 policy when enabled. Uses current three-camera RGB/proprioception, executes only the selected prefix and returns an action receipt plus fresh feedback. Default prefix=5 is 0.20 simulated seconds at 25 Hz; each row costs a native step and native success stops execution. The actor still selects targets and the next goal; prompt text naming an arm or distance is not a movement constraint. Requires the initial task plan. Review feedback before another prefix; do not delegate an entire task.",
    "depth_query": "Query saved permitted RGB-D numerically when enabled. request.kind selects measured depth/region geometry, bounded grasp/place hypotheses, before/after comparison or episode-local target-handle management. Returns small numerical summaries with quality/uncertainty, target/candidate IDs and source observations, not raw point clouds. Candidate IDs require the same current observation. Feed chosen poses to robot_action; a fitted visible surface does not establish full-object shape, contact, retention or task success.",
}

_DETAIL_HELP = "Optional summary/full. Omission uses the actor's configured display default (normally summary) or full for program calls; full repeats original evidence/metadata."
_OBSERVATION_HELP = "ID of a saved permitted observation. Physical actions require the latest current observation in this episode; copy the returned observation_id, never invent one."
_ACTION_HELP = "ID of an existing recorded action; copy its returned action_id. This read does not replay it."
_CAMERA_HELP = "Camera ID present in the selected observation, such as cam_head or cam_right_wrist."
_ROI_HELP = "Original-image integer pixels [x,y,width,height], fully in bounds; use the recorded crop transform to convert enlarged-image coordinates."
_NEW_ACTION_HELP = "Optional caller-chosen idempotency ID; omission derives it from this tool call. Reusing an ID requires identical original input and rereads that action rather than executing again."

PARAMETER_DESCRIPTIONS = {
    "task_plan": {
        "operation": "Optional submit (default), update or read. submit requires plan and ledger (use [] when none is unresolved). update requires an existing plan and at least one of plan/attention/ledger; repeating the same submitted values against the same observation returns status=unchanged. attention updates require an existing freeform plan. read accepts only operation.",
        "observation_id": "Optional on submit/update; defaults to the current episode observation and, when supplied, must match it. Omit on read.",
        "plan": "Required on submit, optional on update: nonempty ordinary prose up to 16000 characters. Suggested short headings: Task understanding (task, scene, assumptions); Approach + completion (rough order and visible completion evidence). Omission on update preserves it. Omit on read; not an executable stage list.",
        "attention": "Optional array of at most three concern/check/when entries; keep only currently relevant questions or choices, evidence and adjustments, and check times. Each text is nonempty and at most 1600 characters; excess is rejected, never truncated. Replace the list to reselect concerns; omission on update preserves it, [] explicitly clears it. Omit on read.",
        "ledger": "Required on initial submit, including [] when no material prerequisite is unresolved; optional replacement array on update with at most 3 short evidence-boundary entries. When a material task-result prerequisite is unresolved and a later decision depends on it, prefer one to three rows. Each entry requires a unique id, expected visible/measured event, status=pending|supported|contradicted|unknown, short permitted RGB/measured/typed-receipt basis (use unknown when absent), and next_check. Keep pending until the expected event is established; do not claim hidden object truth. After contradicted evidence, change the contact, route or target mechanism, or state that no distinguishing next step is supported. Omission on update preserves it; [] clears it. Pending entries are advisory and do not block actions. Omit on read.",
    },
    "robot_observe": {
        "camera_id": _CAMERA_HELP + " Optional unless roi is supplied or scale differs from 1; omission returns the current views.",
        "roi": _ROI_HELP + " Optional crop; requires camera_id and positive width/height.",
        "scale": "Optional integer enlargement 1..4, default 1. A value other than 1 requires camera_id; changes displayed pixels, not source geometry.",
        "detail": _DETAIL_HELP,
    },
    "robot_capabilities": {"detail": _DETAIL_HELP},
    "robot_information": {
        "section": "Optional default (default), kinematics, geometry, dynamics, control, hardware, all, calibration, geometry_image, resources or resource. all returns numerical sections and hardware; resource selects a file, resources lists IDs.",
        "robot_id": "Optional robot ID; omission selects this run's robot. Another robot's current-episode calibration is unavailable.",
        "resource_id": "Required only for section=resource: copy an ID from section=resources. Omit for every other section.",
    },
    "robot_contact_overlay": {
        "observation_id": _OBSERVATION_HELP,
        "camera_id": _CAMERA_HELP,
        "arms": "Optional unique left/right arms to draw; defaults to both. selections/targets may name only these arms.",
        "selections": "Optional arm-to-contact-selection map. Each arm chooses contact_region=pad_center (default), finger_middle or fingertip OR contact_depth_m=-0.0075..0.0625 metres toward the palm from pad_center; never both.",
        "targets": "Optional arm-to-requested-target map. Each target requires position_m=[x,y,z] in env_origin metres for the selected contact center, not link6; optional approach_axis_world=[x,y,z] is a nonzero world direction. No target is a measured object position.",
    },
    "robot_action_status": {"action_id": _ACTION_HELP, "detail": "Optional summary (actor default) or full (program default). full reads the original stored action evidence; the display may summarize repeated diagnostics."},
    "robot_wait": {
        "observation_id": _OBSERVATION_HELP,
        "native_steps": "Required integer 1..50 native control steps; at 25 Hz, 25 advances one simulated second. Budget counts every step.",
        "action_id": _NEW_ACTION_HELP,
        "timeout_s": "Optional wall-clock execution deadline in seconds, 0.1..300, default 30. Not simulated duration. In programs, omission is capped to remaining work time minus 5 seconds of action reserve and 1 second of dispatch margin; explicit values remain unchanged.",
    },
    "history_list_actions": {
        "detail": "Optional summary (actor default) or full (program default without selectors). anchor selects summary for both callers; full forbids anchor/start/limit.",
        "start": "Optional zero-based offset in the selected results, default 0. Results preserve recorded request order; use returned next_start for another page. Forbidden with full.",
        "limit": "Optional page size 1..100, default 5 with anchor or 25 for the plain index; forbidden with full. Anchor pages contain compact complete summaries; use the full-result lookup for original request/receipt fields.",
        "anchor": "Optional current-episode interaction selection: events keeps first/latest actions, execution issues and any requested gripper command; failures keeps non-completed or uncertain execution and incomplete stage receipts; gripper_commands keeps requested gripper events even when unexecuted; all keeps every current-episode request. No images are automatically replayed. Omission preserves the original status-index/full-list behavior.",
    },
    "history_view_observation": {
        "observation_id": _OBSERVATION_HELP,
        "camera_id": _CAMERA_HELP + " Optional unless roi is supplied or scale differs from 1; omission returns the saved views.",
        "roi": _ROI_HELP + " Optional crop; requires camera_id and positive width/height.",
        "scale": "Optional integer enlargement 1..4, default 1. A value other than 1 requires camera_id; changes displayed pixels, not source geometry.",
        "detail": _DETAIL_HELP,
    },
    "history_read_trace": {
        "action_id": _ACTION_HELP,
        "kind": "Required evidence source: planned_trace (planned commands), dispatched_trace (sent commands), actual_trace (measured execution) or before_state (recorded pre-action robot state).",
        "detail": "Optional summary (actor default) or full (program default without selectors). full forbids path/start/limit; selectors select summary when detail is omitted.",
        "path": "Optional JSON Pointer to the selected value, default root (''). Escape ~ as ~0 and / as ~1. Use a returned window path for an exact sequence; forbidden with full.",
        "start": "Optional zero-based sequence offset, default 0. Requires the selected path to resolve to a list; forbidden with full.",
        "limit": "Optional sequence window size 1..100, default 25. Samples/vectors retain complete values; forbidden with full.",
    },
    "history_annotate_action": {
        "action_id": _ACTION_HELP,
        "note": "Required actor-authored judgment, at most 16000 characters. Describe evidence/uncertainty; this annotation is not a physical receipt.",
    },
    "geometry_query": {"request": "Required query object with kind and the applicable fields in the tool description's input field guide. Distances use env_origin metres; pixels use original saved images. The gateway supplies episode identity; do not supply hidden state."},
    "program_run": {
        "operation": "Optional start (default) or resume. resume accepts operation/job_id/review_id/decision plus optional display detail; source/time/authorization cannot change.",
        "source": "Python source for start, nonempty and at most 128 KiB UTF-8. Use source or source_ref; if both are supplied this literal source takes precedence. Files are relative to this job's isolated workspace.",
        "source_ref": "Optional actor-readable program_source artifact reference for start when source is omitted; the referenced bytes are UTF-8 Python.",
        "timeout_s": "Optional start work-time budget, 1..300 seconds, default 180; excludes this job's own helper-model/action-review waits. Omit on resume.",
        "allow_direct_actions": "Optional start-only authorization, default false. false pauses new physical requests for actor review; true delegates execution decisions to this invocation only. Omit on resume.",
        "job_id": "Required on resume: existing job returned by program_run/program_status. Omit on start.",
        "review_id": "Required on resume: copy pending_review.review_id for this job's current paused action. Omit on start.",
        "decision": "Required on resume: approve executes the exact reviewed request through the broker, reject returns the rejection to the program. Omit on start.",
        "detail": "Optional summary/full output display, default summary. summary retains all saved stdout/stderr text after replacing inline image data URLs with placeholders; unchanged logs/supplementary images may be omitted within the current context. full always reads all available saved job/log text and delivers supplementary images again. Displayed references may use aliases; artifact references retain exact saved bytes.",
    },
    "program_status": {"job_id": "Required ID of an existing program job; no execution or resume.", "detail": "Optional summary/full output display, default summary. summary retains all saved stdout/stderr text after replacing inline image data URLs with placeholders; unchanged logs/supplementary images may be omitted within the current context. full always reads all available saved job/log text and delivers supplementary images again. Use full to repeat previously delivered logs or client.artifact on stdout_ref/stderr_ref for exact saved bytes."},
    "vision_compare": {
        "before_observation_id": "Required saved BEFORE observation from the current episode; roi/reference_roi refer to this original image.",
        "after_observation_id": "Required saved AFTER observation from the same episode at the same or later simulation step. A changed frame is needed for new motion evidence.",
        "camera_id": _CAMERA_HELP + " Required same camera and image dimensions in both observations.",
        "roi": _ROI_HELP + " Required BEFORE target-texture patch, width/height at least 4; exclude gripper/background.",
        "reference_roi": _ROI_HELP + " Optional BEFORE reference-texture patch, width/height at least 4. Relative displacement is target minus reference, or null if either is unavailable.",
        "max_motion_px": "Optional maximum accepted translation magnitude in pixels, 1..320, default 160; larger motion is ambiguous. Does not enlarge the matching search or recover lost features.",
    },
    "vision_review": {
        "evidence_mode": "Optional images (default) or operation. operation requires phase, expected_result and ordered action_ids in addition to paired images/target.",
        "before_observation_id": "Required saved BEFORE observation from the current episode; roi selects its target texture/appearance.",
        "after_observation_id": "Required saved AFTER observation in the same episode at the same or later simulation step; for operation use the final listed action's result observation.",
        "camera_id": _CAMERA_HELP + " Required same camera in both observations.",
        "roi": _ROI_HELP + " Required BEFORE target region, width/height at least 4.",
        "target": "Required nonempty description of the same visible target to assess, at most 1000 characters; not a hidden-state assertion.",
        "condition": "Optional images-mode condition retained_with_gripper or released_from_gripper. Omit for a general visible target-motion/retention comparison; omit in operation mode.",
        "phase": "Required only for operation: approach, grasp, short_lift, transport, place, release or retreat. A descriptive checkpoint, not a required execution sequence.",
        "expected_result": "Required only for operation: nonempty caller-expected visible result, at most 1000 characters.",
        "destination": "Optional operation-mode visible destination description, at most 1000 characters; omit in images mode.",
        "action_ids": "Required only for operation: 1..8 unique existing action IDs in execution order, linking BEFORE to final AFTER; copy actual receipt IDs. Omit in images mode.",
    },
    "robot_policy_step": {
        "observation_id": _OBSERVATION_HELP,
        "prompt": "Required nonempty local natural-language intent, at most 2000 characters; naming an arm/distance does not enforce that constraint. Use a bounded subgoal and review the returned actual state.",
        "prefix": "Optional integer 1..50 policy action rows, default 5; at 25 Hz, 5 is 0.20 simulated seconds. Consumes native steps.",
        "timeout_s": "Optional wall-clock execution deadline in seconds, 0.1..300, default 120; not simulated duration.",
        "action_id": _NEW_ACTION_HELP,
    },
    "robot_policy_delegate": {
        "observation_id": _OBSERVATION_HELP + " The policy starts from this unmoved state.",
        "instruction": "Optional language instruction sent to the policy, 1..2000 characters. Default: the native task instruction of the current observation, the only kind of language the policy was trained on. A free-form sub-goal is passed verbatim but is outside its training distribution; text naming an arm or distance is not a constraint.",
        "subtask_note": "Optional 1..1000 character statement of the subtask you are delegating and the state you expect when it returns. Recorded in the receipt and delegation log for later review; never sent to the policy.",
        "max_policy_steps": "Optional native-step budget for this delegation, integer 1..400; default wall_oss.default_policy_steps (200 unless configured), upper bound wall_oss.max_policy_steps (400 unless configured). At 25 Hz, 25 steps is 1 simulated second; each step consumes the official episode budget. The policy replans every replan_steps (default 16) steps.",
        "delegation_id": "Optional caller-chosen idempotency ID, 1..128 characters; omission derives it from this tool call. Reusing an ID with identical input rereads that delegation's receipt instead of running the policy again; different input is rejected.",
    },
    "depth_query": {"request": "Required RGB-D query/target-management object with kind and the applicable fields in the tool description's input field guide. Coordinates are env_origin metres; pixels/ROI use original images. Target-based measurements/candidate use require their recorded episode and current observation; target_status/forget_target can inspect/remove stale handles."},
}

GEOMETRY_REQUEST_DESCRIPTIONS = {
    "kind": "Required: help, camera_pose, fk, pixel_ray, pixel_plane, project_points, preview_ee, triangulate, sparse_track, build_scene, ray_surface, project_geometry, alignment or batch. RGB-D kinds follow depth_query when enabled.",
    "observation_id": "Required for single-view kinds: saved calibrated observation ID.",
    "observation_ids": "For tracking/triangulation/build_scene: [before_id,after_id] sharing calibration and reference frame; same ID with two cameras is allowed. For alignment: held-out observation IDs, excluding scene training frames.",
    "camera_id": "Required except help: camera in the first/selected observation.",
    "camera_id_b": "Optional second-view camera for tracking/triangulation, default camera_id.",
    "frame_id": "Optional reference frame; defaults to the observation calibration frame (normally env_origin) and must match it.",
    "pixel": "Required for pixel_ray/pixel_plane/ray_surface: [u,v] original-image pixels, inside the image; ray depth alone remains unknown.",
    "plane_origin_m": "Required only for pixel_plane: an explicit [x,y,z] point on your hypothesized infinite plane, in calibration-frame metres (normally env_origin). No default surface height or object depth is supplied; an intersection is not an observed object surface.",
    "plane_normal": "Required only for pixel_plane: finite nonzero [nx,ny,nz] plane normal in the calibration frame; normalized by the tool. The selected pixel is assumed to lie on this plane. Absolute normal dot ray below 0.01, or intersection at/behind the camera, returns unknown/null. Result ray_plane_condition_factor is 1/abs(dot) and d_point_d_normal_offset_m is the [dx,dy,dz] shift of point_m per +1 m plane shift along this normal.",
    "plane_origin_alt_m": "Optional only for pixel_plane: a second [x,y,z] point for the same caller-supplied plane normal. The tool returns alternate_point_m, alternate_ray_distance_m and assumption_delta_m so you can see how the plane hypothesis changes the result; it does not select a default or measure the surface. If either intersection is near-parallel, nonfinite or behind the camera, the query returns unknown/null.",
    "target_camera_id": "Optional only for pixel_plane: saved calibrated camera in the same observation (such as a wrist view) on which to project point_m (and optional alternate_point_m) and return a geometry overlay image in the same call.",
    "points_m": "Required for project_points: 1..2000 caller-selected [x,y,z] points in the calibration frame, in metres. Optional explicit points for project_geometry/alignment.",
    "pixels_a": "Optional triangulation/tracking correspondences [[u,v],...], 1..2000 points from the first original image; supply pixels_b too with equal count. Omission uses sparse RGB tracking.",
    "pixels_b": "Second-image correspondences in the same order/count as pixels_a; both sets must be in bounds.",
    "roi": "Optional first-image tracking region [x,y,width,height] in original pixels; omission uses the whole first image.",
    "tracking_method": "Optional lk or sift feature correspondence method; default lk for the same camera, sift for different cameras.",
    "static_scene_assumption": "Must be true for temporal triangulate/build_scene; explicitly asserts the matched scene stayed static between observations. Not needed for simultaneous same-observation views.",
    "min_angle_deg": "Optional triangulation minimum parallax angle in degrees, >0 and <90, default 0.5.",
    "max_error_px": "Optional triangulation maximum reprojection error in pixels, >0 and <=100, default 2.",
    "connect_points": "Optional true joins consecutive projected points in the overlay; default false, no geometric constraint.",
    "action": "Required only for preview_ee: projection-only absolute EEAction with frame_id matching calibration and path.waypoints (1..100); each waypoint maps left/right to position_m and normalized quaternion_xyzw. No IK, motion or collision check.",
    "axis_length_m": "Optional preview_ee overlay axis length in metres, default 0.05; affects drawing only.",
    "scene_version": "Required for ray_surface/alignment and project_geometry without explicit points_m: existing explicitly reconstructed local scene version. Never a hidden native scene.",
    "entity_id": "Required for ray_surface: selected reconstructed entity. For build_scene: optional stable entity name. For project_geometry: optional one entity when entity_ids is omitted.",
    "entity_ids": "Optional project_geometry entity list; defaults to all tracked entities in scene_version.",
    "base_scene_version": "Required build_scene base version; use 0 for an initial proposed patch, otherwise an existing saved local version. This query does not apply the proposal.",
    "primitive": "Optional build_scene fitted primitive plane (default) or box; requires at least 6 accepted triangulation points and nondegenerate support. Only measured visible support is reconstructed.",
    "plane_threshold_m": "Optional build_scene plane-fit inlier distance in metres, default 0.01.",
    "rebuild": "Optional build_scene boolean, default false; true rebuilds instead of associating with prior local entities.",
    "max_displacement_m": "Optional build_scene association displacement limit in metres, default 0.3; larger changes leave identity uncertain.",
    "pixels": "For alignment with explicit points_m: corresponding held-out original-image [u,v] annotations.",
    "correspondence_source": "Optional alignment annotation-source label, default caller_annotation; quality labels do not turn annotation into ground truth.",
    "artifact": "Saved point-cloud artifact handle (env_origin m).",
    "queries": "1..32 query objects executed in order.",
}

DEPTH_REQUEST_DESCRIPTIONS = {
    "kind": "Required: help, depth_points, region_geometry, grasp_candidates, place_candidates, compare_regions, register_target, refresh_target, target_status, forget_target or skill_help. skill_help only reads the optional fixed skill catalog.",
    "observation_id": "Required for explicit single-view queries and register_target/refresh_target; target management requires the latest current observation.",
    "camera_id": "Required for explicit measurement/register_target: saved calibrated RGB-D camera. Target-based queries inherit it; refresh_target retains the registered camera.",
    "frame_id": "Optional measurement frame; only env_origin is supported and is the default.",
    "target_id": "Existing current target handle replaces observation_id/camera_id/roi/seed_pixel for region_geometry/grasp_candidates/place_candidates; do not also supply those fields. Required for refresh/forget; optional target_status filters one target. depth_points/compare_regions require explicit observation fields.",
    "roi": "Required for region/candidate queries and register/refresh: original-image integer [x,y,width,height], width/height >=1, fully in bounds and area <=1048576 pixels. compare_regions uses the BEFORE region.",
    "seed_pixel": "Optional [u,v] original-image region seed inside roi; defaults to its center. Refreshing a target discards its old seed when omitted.",
    "pixels": "Required depth_points: 1..16 [u,v] original-image positions; optical depth is z in metres, not ray distance.",
    "radius_px": "Optional depth_points integer neighborhood radius 0..4 pixels, default 1.",
    "depth_tolerance_m": "Optional region/candidate/comparison seed-depth neighborhood tolerance 0.0005..0.15 metres, default 0.015.",
    "rgb_tolerance": "Optional region/candidate/comparison color distance threshold 1..255, default 60; not semantic segmentation confidence.",
    "count": "Optional grasp/place maximum candidates 1..3, default 3; fewer can be returned when evidence is insufficient.",
    "max_opening_m": "Optional grasp jaw-opening bound 0.005..0.088807166 metres, default 0.088807166; a geometry constraint, not grasp verification.",
    "clearance_m": "Optional grasp finger clearance 0.001..0.03 metres, default 0.005.",
    "grasp_depth_m": "Optional grasp descent from the visible surface, 0..0.04 metres, default 0.01.",
    "approach_height_m": "Optional grasp approach height 0.02..0.20 metres, default 0.06.",
    "held_dimensions_m": "Required place dimensions [x,y,z], each 0.002..0.75 metres; caller estimates for the whole held object.",
    "grasp_offset_m": "Required place [x,y,z] offset from object center to grasp point in metres, each -0.75..0.75; caller estimate, not a measured center of mass.",
    "margin_m": "Optional place support-edge margin 0.002..0.10 metres, default 0.01.",
    "yaw_deg": "Optional place yaw -180..180 degrees, default 0, using the public grasp-angle convention.",
    "support_gap_m": "Optional place gap above the fitted support 0.001..0.02 metres, default 0.003.",
    "before_observation_id": "Required compare_regions: saved BEFORE RGB-D observation.",
    "after_observation_id": "Required compare_regions: saved AFTER RGB-D observation from the same episode.",
    "arm": "Optional left/right for register_target, or compare_regions to report motion relative to that measured gripper.",
    "label": "Optional register_target nonempty actor label up to 200 characters. At most four active target handles; forget unused targets.",
}

PUBLIC_EXAMPLES = {
    "task_plan": [dict(operation="submit", observation_id="obs_1", plan="Task understanding: move the visible target to its support. Approach + completion: choose a clear grasp and route, release, then verify stable placement.", ledger=[]),
        dict(operation="update", attention=[dict(concern="The object may have shifted in the gripper.",
            check="Use visible object-to-finger alignment; if it shifted, adjust the placement approach.",
            when="From the normal transfer action's final observation, before placement.")]),
        dict(operation="update", attention=[]), dict(operation="read")],
    "robot_observe": [{}],
    "robot_capabilities": [dict(detail="full")],
    "robot_information": [dict(section="resources"), dict(section="resource", resource_id="geometry_model")],
    "robot_contact_overlay": [dict(observation_id="obs_1", camera_id="cam_head", arms=["left"], selections=dict(left=dict(contact_region="fingertip")))],
    "robot_action_status": [dict(action_id="act_1", detail="full")],
    "robot_wait": [dict(observation_id="obs_1", native_steps=10)],
    "history_list_actions": [dict(anchor="events", start=0, limit=5), dict(start=0, limit=25)],
    "history_view_observation": [dict(observation_id="obs_1", camera_id="cam_head", roi=[100,100,80,60], scale=2)],
    "history_read_trace": [dict(action_id="act_1", kind="actual_trace", detail="summary", path="/samples", start=0, limit=25)],
    "history_annotate_action": [dict(action_id="act_1", note="The object is released; landing remains uncertain behind the rim.")],
    "geometry_query": [dict(request=dict(kind="pixel_ray", observation_id="obs_1", camera_id="cam_head", pixel=[320,240]))],
    "program_run": [dict(operation="start", source="from auriga_client import AurigaClient\nprint(AurigaClient().call('robot_observe', {})['data']['observation_id'])"), dict(operation="resume", job_id="job_1", review_id="review_1", decision="approve")],
    "program_status": [dict(job_id="job_1", detail="full")],
    "vision_compare": [dict(before_observation_id="obs_1", after_observation_id="obs_2", camera_id="cam_head", roi=[100,100,40,40])],
    "vision_review": [dict(before_observation_id="obs_1", after_observation_id="obs_2", camera_id="cam_head", roi=[100,100,40,40], target="the selected cup", condition="retained_with_gripper"), dict(evidence_mode="operation", phase="release", target="the selected cup", expected_result="The cup remains supported after the gripper withdraws.", action_ids=["act_1"], before_observation_id="obs_1", after_observation_id="obs_2", camera_id="cam_head", roi=[100,100,40,40])],
    "robot_policy_step": [dict(observation_id="obs_1", prompt="Adjust the local grasp, then return feedback.", prefix=5)],
    "robot_policy_delegate": [dict(observation_id="obs_1", subtask_note="Finish the contact-rich placement from the current grasp.", max_policy_steps=160)],
    "depth_query": [dict(request=dict(kind="depth_points", observation_id="obs_1", camera_id="cam_head", pixels=[[320,240]]))],
    "operation_memory": [dict(request=dict(kind="help"))],
}


def _geometry_request_schema():
    """Reuse geometry-owner types without imposing HTTP-only envelope rules."""
    from ...core.contracts import IDENTITY
    from ...perception.scene.scene_contracts import GEOMETRY_KINDS
    branches = []
    for kind, source in GEOMETRY_KINDS.items():
        branch = deepcopy(source)
        properties = branch['properties']
        injected = {*IDENTITY, 'schema_version', 'world_kind', 'budget'}
        for key in injected:
            properties.pop(key, None)
        branch['required'] = [key for key in branch['required'] if key not in injected]
        # Direct GeometryTools accepts both references, choosing observation_ids.
        if 'oneOf' in branch:
            branch['anyOf'] = branch.pop('oneOf')
        branch['additionalProperties'] = kind != 'pixel_plane'
        if kind == 'project_geometry':
            branch['required'].remove('scene_version')
            branch.setdefault('allOf', []).append({'anyOf': [
                {'required': ['points_m']}, {'required': ['scene_version']}]})
        if kind == 'preview_ee':
            properties['axis_length_m'] = dict(type='number', default=.05)
        if 'roi' in properties:
            # The direct RGB tracker converts numeric coordinates with int();
            # its HTTP interface alone requires already-integral coordinates.
            properties['roi'] = dict(type='array', items=dict(type='number'), minItems=4, maxItems=4)
        branches.append(branch)
    branches.append(dict(
        type='object',
        properties=dict(
            kind=dict(type='string', const='batch'),
            observation_id=dict(type='string', minLength=1),
            camera_id=dict(type='string', minLength=1),
            artifact=dict(type='string', minLength=1),
            queries=dict(type='array', minItems=1, maxItems=32, items=dict(type='object')),
        ),
        required=['queries'],
        additionalProperties=False,
    ))
    return dict(type='object', oneOf=branches)


def _describe_request(schema, descriptions):
    """Annotate copied owner fields; branch selection and numeric types stay real."""
    for branch in schema['oneOf']:
        for field, original in branch['properties'].items():
            # Owner schemas intentionally reuse vector/string dictionaries.
            # Each field has its own meaning even when its shape is shared.
            definition = branch['properties'][field] = deepcopy(original)
            if 'type' not in definition and ('enum' in definition or 'const' in definition):
                definition['type'] = 'string'
            if field in descriptions:
                definition['description'] = descriptions[field]


def _with_input_guide(item):
    """Keep field meaning/defaults/bounds in the tool itself on every transport."""
    from ...core.contracts import canonical
    fields = {}

    def logic(node):
        """Render structural alternatives as logic, never flattened requirements."""
        terms = []
        for field, value in node.get('properties', {}).items():
            if 'const' in value or 'enum' in value:
                choice = ('equals ' + canonical(value['const']) if 'const' in value
                          else 'is one of ' + canonical(value['enum']))
                terms.append(field + ' ' + choice + ('' if field in node.get('required', []) else ' when supplied'))
        if node.get('required'):
            terms.append('requires ' + ', '.join(node['required']))
        if node.get('additionalProperties') is False:
            terms.append('only declared fields')
        for key, phrase in (('minProperties', 'at least'), ('maxProperties', 'at most')):
            if key in node:
                terms.append(phrase + ' ' + str(node[key]) + ' fields')
        for key, required in node.get('dependentRequired', {}).items():
            terms.append('if ' + key + ' is present, requires ' + ', '.join(required))
        for key, phrase in (('oneOf', 'exactly one of'), ('anyOf', 'at least one of'), ('allOf', 'all of')):
            choices = [logic(part) for part in node.get(key, [])]
            if choices:
                terms.append(phrase + ' (' + ' | '.join(choice or 'no additional restriction' for choice in choices) + ')')
        if 'not' in node:
            terms.append('must not satisfy (' + logic(node['not']) + ')')
        if 'if' in node:
            rule = 'if (' + logic(node['if']) + '), then (' + logic(node.get('then', {})) + ')'
            if 'else' in node:
                rule += ', otherwise (' + logic(node['else']) + ')'
            terms.append(rule)
        return '; '.join(terms)

    def condition(node, path):
        choices = []
        for field, value in node.get('properties', {}).items():
            field = path + '.' + field if path else field
            if 'const' in value:
                choices.append(field + '=' + canonical(value['const']))
            elif len(value.get('enum', [])) == 1:
                choices.append(field + '=' + canonical(value['enum'][0]))
        return ', '.join(choices)

    def selected_context(context, selected):
        return ' and '.join(dict.fromkeys(part for part in (context, selected) if part))

    def rules(node):
        values = []
        if 'type' in node:
            kind = node['type']
            values.append('Type: ' + (' or '.join(kind) if isinstance(kind, list) else kind))
        for key, label in (('enum', 'Allowed'), ('const', 'Value'), ('default', 'Default'),
                ('minimum', 'Minimum (inclusive)'), ('maximum', 'Maximum (inclusive)'),
                ('exclusiveMinimum', 'Minimum (exclusive)'), ('exclusiveMaximum', 'Maximum (exclusive)'),
                ('minItems', 'Minimum items'), ('maxItems', 'Maximum items'),
                ('minLength', 'Minimum characters'), ('maxLength', 'Maximum characters'),
                ('pattern', 'Pattern'), ('uniqueItems', 'Unique items')):
            if key in node:
                values.append(label + ': ' + canonical(node[key]))
        return '; '.join(values)

    def visit(node, path='', context='', structural=True):
        if not isinstance(node, dict):
            return
        text = node.get('description', '')
        limits = rules(node)
        structure = logic(node) if structural else ''
        if structure:
            limits += ('. ' if limits else '') + 'Object rules: ' + structure
        if text or limits:
            entry = fields.setdefault(path or 'input', {'help': {}, 'rules': {}})
            if text:
                entry['help'].setdefault(text, set()).add(context)
            if limits:
                entry['rules'].setdefault(limits, set()).add(context)
        for field, child in node.get('properties', {}).items():
            visit(child, path + '.' + field if path else field, context)
        if isinstance(node.get('items'), dict):
            visit(node['items'], path + '[]', context)
        for index, child in enumerate(node.get('prefixItems', [])):
            visit(child, path + '[' + str(index) + ']', context)
        for key in ('oneOf', 'anyOf', 'allOf'):
            for branch in node.get(key, []):
                selected = condition(branch, path)
                visit(branch, path, selected_context(context, selected), structural=False)
        if 'then' in node:
            selected = condition(node.get('if', {}), path)
            visit(node['then'], path, selected_context(context, selected), structural=False)
        if 'else' in node:
            selected = condition(node.get('if', {}), path)
            visit(node['else'], path, selected_context(context, 'not (' + selected + ')' if selected else ''), structural=False)

    visit(item['inputSchema'])
    guide = []
    for path, entry in fields.items():
        parts = []
        for category in ('help', 'rules'):
            for text, contexts in entry[category].items():
                # Repeated meanings are printed once; different branch rules
                # retain their selector instead of becoming simultaneous rules.
                qualifier = '' if '' in contexts or category == 'help' and len(entry[category]) == 1 else '[' + ' or '.join(sorted(contexts)) + '] '
                parts.append(qualifier + text + ('.' if category == 'rules' else ''))
        guide.append(path + ': ' + ' '.join(parts))
    if guide:
        item['description'] += '\nInput field guide (use only fields applicable to the selected mode/kind):\n' + '\n'.join(guide)
    return item


def _describe_public(item, vggt_height=False, object_table=False, scene3d=False):
    from ...core.contracts import canonical
    name, schema = item["name"], item["inputSchema"]
    overrides, examples = item.pop("_field_help", None) or {}, item.pop("_examples", None)
    schemas = schema.get("oneOf", [schema])
    for branch in schemas:
        for field, field_schema in branch.get("properties", {}).items():
            description = overrides.get(field) or PARAMETER_DESCRIPTIONS.get(name, {}).get(field)
            if description:
                field_schema["description"] = description
        request = branch.get("properties", {}).get("request")
        if name in {"geometry_query", "depth_query"}:
            if name == 'geometry_query':
                from ...perception.depth.depth_runtime import DEPTH_KINDS
                geometry = _geometry_request_schema()
                descriptions = GEOMETRY_REQUEST_DESCRIPTIONS
                if vggt_height:
                    from ...perception.external.vggt_height import DEFAULT_DESCRIPTION_SENTENCE, DESCRIPTION_SENTENCE, extend_definition
                    selected = vggt_height if isinstance(vggt_height, str) else "optional"
                    descriptions = extend_definition(geometry, descriptions, selected)
                    item["description"] += DEFAULT_DESCRIPTION_SENTENCE if selected == "default" else DESCRIPTION_SENTENCE
                if object_table:
                    from ...perception.external.object_table import extend_definition as extend_object_table
                    descriptions = extend_object_table(geometry, descriptions)
                if scene3d:
                    from ...perception.external.scene3d import extend_definition as extend_scene3d
                    descriptions = extend_scene3d(geometry, descriptions)
                _describe_request(geometry, descriptions)
                if object_table:
                    from ...perception.external.object_table import describe_branch
                    describe_branch(geometry)
                if scene3d:
                    from ...perception.external.scene3d import describe_branch as describe_scene3d
                    describe_scene3d(geometry)
                # Keep the existing RGB-D compatibility route; its complete
                # field contract is published once on the canonical tool.
                geometry['oneOf'].append(dict(type='object', properties=dict(kind=dict(type='string',
                    enum=sorted(DEPTH_KINDS), description='RGB-D compatibility kinds, only when enabled. Use depth_query for the complete field contract, units and bounds; these requests are forwarded to the same depth owner for validation.')),
                    required=['kind'], additionalProperties=True))
                request.update(geometry)
            else:
                from ...perception.depth.depth_runtime import request_schema
                depth = request_schema()
                _describe_request(depth, DEPTH_REQUEST_DESCRIPTIONS)
                request.update(depth)
    if examples or name in PUBLIC_EXAMPLES:
        schema["examples"] = examples or deepcopy(PUBLIC_EXAMPLES[name])
        item["description"] += " Examples (replace IDs/pixels with actual saved evidence): " + " ; ".join(canonical(example) for example in schema["examples"])
    return _with_input_guide(item)


def public_definitions(*, vggt_height=False, object_table=False, scene3d=False, observe3d=(False, False), pose_skill=False,
                       wall_oss_gate="none", wall_oss_call_budget=2000, wall_oss_monitor=(64, 1, 0.8)):
    """Publish the same documented input contract to actor and program callers."""
    from ..actions.robot_action_schema import ROBOT_ACTION_TOOL
    from ...cortex.context.operation_memory import MEMORY_DESCRIPTION
    selected = []
    for original in TOOLS:
        name = original["name"]
        item = deepcopy(original)
        if name in DESCRIPTIONS:
            item["description"] = DESCRIPTIONS[name]
        if name == "robot_policy_delegate":
            # wall_oss_monitor = (policy steps between supervisor checks, consecutive takeover verdicts, call budget
            # fraction); used by pickplace_v9 only.
            item["description"] = tool_description(wall_oss_gate, wall_oss_call_budget, *wall_oss_monitor)
            if delegation_gate.requires_instruction(wall_oss_gate):
                item["inputSchema"]["required"] = ["observation_id", "instruction"]
            if delegation_gate.fixed_budget(wall_oss_gate):
                # The run fixes the call budget (wall_oss.call_budget_*); the actor does not choose one.
                item["inputSchema"]["properties"].pop("max_policy_steps", None)
            item["_field_help"] = delegation_gate.field_help(wall_oss_gate)
            item["_examples"] = delegation_gate.examples(wall_oss_gate)
        if name == "robot_observe" and any(observe3d[:2]):
            from ...perception.external.obs3d import catalogue_sentence
            item["description"] += catalogue_sentence(*observe3d)
        if name == "task_plan":
            item["inputSchema"] = dict(type="object", additionalProperties=False,
                properties=dict(operation=dict(type="string", enum=["submit", "update", "read"]),
                    plan=deepcopy(PLAN_SCHEMA), attention=deepcopy(ATTENTION_SCHEMA),
                    ledger=deepcopy(EVIDENCE_LEDGER_SCHEMA),
                    observation_id=dict(type="string")), required=[], allOf=[{
                        "if": {"properties": {"operation": {"const": "read"}}, "required": ["operation"]},
                        "then": {"maxProperties": 1},
                        "else": {
                            "if": {"properties": {"operation": {"const": "update"}}, "required": ["operation"]},
                            "then": {"anyOf": [{"required": ["plan"]}, {"required": ["attention"]},
                                                  {"required": ["ledger"]}]},
                            "else": {"required": ["plan", "ledger"]}}}])
        if name == "program_run":
            item["inputSchema"] = dict(type="object", additionalProperties=False,
                properties=dict(operation=dict(type="string", enum=["start", "resume"]),
                    source=dict(type="string"), source_ref=dict(type="string"), timeout_s=dict(type="number", minimum=1, maximum=300),
                    allow_direct_actions=dict(type="boolean", default=False),
                    job_id=dict(type="string"), review_id=dict(type="string"),
                    decision=dict(type="string", enum=["approve", "reject"]),
                    detail=dict(type="string", enum=["summary", "full"])), required=[])
        if name == "vision_review":
            image_schema = item["inputSchema"]
            image_schema["properties"]["evidence_mode"] = dict(type="string", enum=["images"])
            operation_schema = dict(type="object", properties=deepcopy(TOOL_PROPERTIES), required=list(TOOL_REQUIRED),
                                    additionalProperties=False)
            operation_schema["properties"]["evidence_mode"] = dict(type="string", enum=["operation"])
            operation_schema["required"].append("evidence_mode")
            item["inputSchema"] = dict(type="object", oneOf=[image_schema, operation_schema])
        selected.append(_describe_public(item, vggt_height, object_table, scene3d) if (vggt_height or object_table or scene3d)
                        else _describe_public(item))
    selected.append(_with_input_guide(deepcopy(ROBOT_ACTION_TOOL)))
    selected.append(_describe_public(dict(type="function", name="operation_memory", description=MEMORY_DESCRIPTION,
            inputSchema=dict(type="object", properties=dict(request=dict(type="object",
                description="Required operation-memory request. kind selects help/query/record; scope selects current-task or shared experience. Read the operation_memory description for kind-specific fields and evidence requirements.")),
                required=["request"], additionalProperties=False))))
    if pose_skill:
        from ...perception.external.pose_skill import definition as pose_plan_definition
        selected.append(_with_input_guide(pose_plan_definition()))
    return selected
