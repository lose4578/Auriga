"""The only observed-world action writer, shared by dynamic tools and programs."""

import json
import hashlib
import threading
import time
import io
from copy import deepcopy
from pathlib import Path

from ..core.contracts import IDENTITY, canonical, digest, new_id, validate_action, validate_observation, validate_identity
from ..robot.model import roboprobe_pose
from ..robot.model.grasp_contact import contact_target_feedback
from .actions.home_check import HomePoseCheck
from ..robot.motion.motion_speed import SPEED_INPUT_FIELDS, SPEED_EFFECTIVE_FIELDS
from ..robot.motion.action_compilation import validate_backend_action
from ..cortex.context.task_planning import TaskPlanning, decision_reminder_text
from ..cortex.agents.base import read_agent_identity
from ..perception.scene.store import SceneStore
from ..robot.model.geometry import GeometryTools
from ..core.observation_policy import LEGACY_ACTOR_PROFILE, require_actor_observation, actor_robot_state, reject_privileged, has_calibration, allows_depth
from ..perception.depth.depth_runtime import DepthRuntime
from ..core.storage.images import camera_path, encode_image, image_url, share_capture_file, store_image_settings
from ..core.presentation.actor_payload import ACTOR_PAYLOAD_PROFILES, lean_observation
from ..core.tool_output import (StructuredText, ToolOutput, ToolResponse, output_contract_error,
    resolve_detail, validate_display, wire_content, wire_response)


_UNSET = object()

# Decision-context reminder channels (design 04 §4.5); recorded on model_context.* events.
# Codex: CodexSession re-injects the reminder as an assistant message into its own /responses/compact request.
REMINDER_IN_COMPACT_REQUEST = "compact_request_assistant_message"
# Other agent backends: Harness.handle_tool appends it to the next actor tool response after a context reset.
REMINDER_IN_TOOL_RESULT = "post_compaction_tool_result"

from ..core.presentation.feedback_delivery import execution_unconfirmed


def _execution_acknowledgement(raw):
    """Keep only validated execution facts if the remaining worker payload is unsafe."""
    status = raw.get("status")
    receipt = dict(status=status if isinstance(status, str) and status in {
        "completed", "partial", "failed", "stopped", "cancelled", "rejected", "unknown", "status_unknown"} else "unknown")
    for key in ("executed_steps", "executed_native_steps", "completed_native_steps", "executed_control_steps"):
        value = raw.get(key)
        if key in raw and (value is None or type(value) is int and value >= 0):
            receipt[key] = value
    if isinstance(raw.get("reason"), str):
        receipt["reason"] = raw["reason"]
    for key in ("execution_uncertain", "stop_confirmed"):
        if type(raw.get(key)) is bool:
            receipt[key] = raw[key]
    stop = raw.get("stop_confirmation")
    if stop is False:
        receipt["stop_confirmation"] = False
    elif isinstance(stop, dict) and type(stop.get("confirmed")) is bool:
        receipt["stop_confirmation"] = dict(confirmed=stop["confirmed"])
    elif (isinstance(stop, dict) and set(stop) <= {"clock", "actuator_target", "settling_steps"}
            and stop.get("clock") == "paused"
            and stop.get("actuator_target") == "last_native_command_retained"
            and type(stop.get("settling_steps")) is int and stop["settling_steps"] >= 0):
        receipt["stop_confirmation"] = deepcopy(stop)
    # Keep nested negative acknowledgements without copying unsafe arm/stage bodies.
    for field in ("arm_results", "gripper_execution", "stage_results"):
        if isinstance(raw.get(field), dict) and any(execution_unconfirmed(value) for value in raw[field].values()):
            receipt["execution_uncertain"] = True
    return receipt


class Harness:
    # Channel recorded on actor deliveries ("delivery" records): "codex_tool" for the Codex backend (unchanged),
    # "agent_tool" for other agent backends. Set by Workbench from the session's backend_kind.
    delivery_channel = "codex_tool"
    # Forced full delivery (design 04 §4.3). False (default; Codex): an image, static observation metadata field,
    # capability summary or program-log excerpt already delivered in the current model-context epoch may be left out
    # of a later actor tool result. True: _model_context_epoch() carries a fresh marker on every call, so every actor
    # tool result is complete and nothing delivered earlier is assumed to be in the model's context. Workbench sets it
    # right after session.start() from AgentSession.context_delivery ("full" -> True).
    context_epoch_volatile = False
    # Where the decision-context reminder goes after a model-context reset: REMINDER_IN_COMPACT_REQUEST (default;
    # Codex, the Harness never adds it) or REMINDER_IN_TOOL_RESULT (other backends: handle_tool sideband). Set by
    # Workbench from the session's backend_kind.
    decision_reminder_channel = REMINDER_IN_COMPACT_REQUEST
    # True while a reported model-context reset waits for its reminder: the next actor tool response returned by
    # handle_tool gains one inputText item with decision_reminder_text(...), then the flag clears. Set only by
    # model_context_reset() and only on the REMINDER_IN_TOOL_RESULT channel.
    decision_reminder_due = False

    def __init__(self, store, robot, max_actions=None, actor_payload_profile=_UNSET, enable_wait=_UNSET,
                 observation_profile=None, motion_review=None, grasp_memory_store=None, vision_compare=_UNSET, config=None,
                 task_plan_required=False, tool_specs=()):
        from ..core.config import Config
        self.config = config if config is not None else Config.load()
        actor_payload_profile = self.config.features.actor_payload_profile if actor_payload_profile is _UNSET else actor_payload_profile
        enable_wait = self.config.features.enable_wait if enable_wait is _UNSET else enable_wait
        vision_compare = self.config.features.vision_compare if vision_compare is _UNSET else vision_compare
        self.disabled_tools = frozenset(self.config.features.disabled_tools)
        from .tools.builtins import create_registry
        self.tool_registry = create_registry(tool_specs, runtime=self)
        from .tools.builtins import CONFIG_TOOL_NAMES
        unknown = self.disabled_tools - self.tool_registry.names - CONFIG_TOOL_NAMES
        if unknown: raise ValueError('unknown_disabled_tools:' + ','.join(sorted(unknown)))
        if actor_payload_profile not in ACTOR_PAYLOAD_PROFILES:
            raise ValueError("invalid_actor_payload_profile")
        if type(enable_wait) is not bool:
            raise ValueError("invalid_enable_wait")
        if type(vision_compare) is not bool:
            raise ValueError("invalid_vision_compare")
        self.vision_compare = vision_compare
        self.store, self.robot, self.max_actions = store, robot, max_actions
        from ..core.presentation.actor_presentation import ActorPresentation
        self.actor_presentation = ActorPresentation(store)
        from ..core.presentation.actor_output import ActorOutput
        self.actor_output = ActorOutput(store, self.actor_presentation, actor_payload_profile)
        self._lean_compact = None
        if getattr(self.config.features, "lean_compact", False):
            from ..core.presentation.lean_compact import LeanCompact
            projection = getattr(self.config.features, "lean_projection", False)
            self._lean_compact = LeanCompact(anchor_period=(
                getattr(self.config.features, "lean_projection_block", 8) if projection else None))
        # Actor sessions require a plan; direct backend clients can retain their
        # existing execution contract without pretending to be task actors.
        self.task_planning = TaskPlanning(store, required=task_plan_required,
                                          manipulation_protocol=self.config.features.manipulation_protocol,
                                          task_scheduling=self.config.features.task_scheduling,
                                          operation_candidates=self.config.features.operation_candidates)
        if task_plan_required and "task_plan" in self.disabled_tools:
            raise ValueError("required_task_plan_tool_cannot_be_disabled")
        # Workbench passes its explicit profile; unprofiled custom backends keep
        # their existing legacy interface. Native RobotBackend defaults calibrated.
        self.observation_profile = observation_profile or (config.features.observation_profile if config is not None else getattr(robot, "observation_profile", LEGACY_ACTOR_PROFILE))
        self.motion_review = (self.config.features.motion_review and hasattr(robot, "preview_motion") and has_calibration(self.observation_profile)) if motion_review is None else motion_review
        if type(self.motion_review) is not bool: raise ValueError("invalid_motion_review")
        if self.motion_review and not has_calibration(self.observation_profile):
            raise ValueError("motion_review_requires_calibration")
        self.depth_skills = self.config.features.depth_skills
        self.depth_auto_feedback = self.config.features.depth_auto_feedback
        self.depth_event_payload = self.config.features.depth_event_payload
        if any((self.depth_skills, self.depth_auto_feedback, self.depth_event_payload)) and not allows_depth(self.observation_profile):
            raise ValueError("depth_features_require_explicit_rgbd_profile")
        if self.depth_event_payload and not self.depth_auto_feedback:
            raise ValueError("depth_event_payload_requires_auto_feedback")
        from ..cortex.context.operation_memory import OperationMemory
        self.operation_memory = OperationMemory(store, shared_store=grasp_memory_store)
        self.actor_payload_profile, self.enable_wait = actor_payload_profile, enable_wait
        self.image_settings = store_image_settings(store, self.config.images.to_dict())
        # Only full RGB images actually emitted to Astra advance these baselines.
        # Program-internal observations/crops must not hide unseen scene changes.
        self._last_delivered_views = {}
        self._view_context_epoch = None
        from ..core.presentation.capability_output import CapabilityDelivery
        self._capability_delivery = CapabilityDelivery()
        from ..core.presentation.program_output_delivery import ProgramOutputDelivery
        self._program_output_delivery = ProgramOutputDelivery()
        from ..core.presentation.observation_output import MetadataDelivery
        self._metadata_delivery = MetadataDelivery()
        if hasattr(robot, "image_settings"):
            robot.image_settings = dict(self.image_settings)
        self.identity = store.identity
        self.scene = SceneStore(store.root / "scene", store.resolve_artifact, store.put_artifact)
        self.geometry = GeometryTools(self.scene, settings=self.image_settings)
        self.depth = DepthRuntime(self)
        self.scene_artifacts = store
        self.programs = None
        self.action_policy = None
        self.visual_reviewer = None
        self.latest = None
        self.home_check = (HomePoseCheck(store)
                           if self.config.features.completion_guidance in {"hook", "both"} else None)
        self.capabilities = None
        self.lock = threading.RLock()
        # Guards decision_reminder_due: the session reports resets on its relay/reader thread while tool workers
        # (possibly several in parallel) consume the flag in handle_tool.
        self._decision_reminder_lock = threading.Lock()
        self.robot_lock = threading.Lock()
        self.policy_lock = threading.Lock()
        self.cancel = threading.Event()
        self.accepting = True
        self.stop_revision = 0
        self.active_action = None
        self.active_context = None
        self.wall_deadline = None
        self.episode_done = threading.Event()
        self.episode_terminal = None
        self.on_episode_end = None
        self.started_at = self.last_native_progress_at = time.monotonic()
        self.model_wait_at_start = self.model_wait_at_native_progress = 0.0
        self.native_action_count = 0
        self.native_budget_limit = None
        self.native_budget_known = False
        self.failed_paths = set()
        self.action_count = store.accepted_action_count()
        self.states = {}
        self.thread_id = None
        self.call_items = {}
        # Lost worker acknowledgements are never automatically retried on recovery.
        for request in store.records("action_request"):
            key = request["action_id"]
            try:
                self.states[key] = store.read("action_receipt", key)
            except KeyError:
                self.states[key] = dict(status="unknown", reason="gateway_restarted_without_terminal_receipt")
                self.accepting = False
                store.event("action.status_unknown", self.states[key], {"action_id": key})
            if (all(request.get(name) == self.identity[name] for name in IDENTITY)
                    and execution_unconfirmed(self.states[key])):
                self.accepting = False

    def start(self):
        from ..robot.motion.motion_speed import SPEED_PRESETS, GRIPPER_SPEED_PRESETS
        self.capabilities = deepcopy(self.robot.start())
        from ..cortex.context.task_notes import handoff_policy
        self.capabilities["handoff_policy"] = {
            key: value for key, value in handoff_policy(self.config.features.airborne_handoff,
                handoff_preference=self.config.features.handoff_preference).items() if key != "text"}
        limit = self.capabilities.get("action_budget", {}).get("native_step_limit")
        if type(limit) is int and limit > 0:
            self.native_budget_limit, self.native_budget_known = limit, True
        for key in ("arm_queue", "motion_cost_preview"):
            if key in self.capabilities:
                self.capabilities[key]["available"] = bool(self.capabilities[key].get("available")
                    and getattr(self.config.features, key))
        speed = self.capabilities.get("action_speed")
        if speed is not None:
            speed.setdefault("speed_scale", {})["default"] = self.config.features.default_motion_speed_scale
            speed.update(speed_presets=dict(SPEED_PRESETS), gripper_speed_presets=dict(GRIPPER_SPEED_PRESETS),
                         default_scope="harness_motion_request")
            if speed.get("waypoint_speed_scales", {}).get("available"):
                speed["waypoint_speed_presets"] = dict(available=True, presets=dict(SPEED_PRESETS),
                    order="via waypoints then final target; each controls its incoming segment",
                    null_semantics="inherit resolved request speed; omission of request speed uses run default",
                    mutually_exclusive_with="waypoint_speed_scales")
        if not self.enable_wait:
            self.capabilities.pop("bounded_wait", None)
        require_actor_observation(self.capabilities, self.observation_profile)
        if self.motion_review and not hasattr(self.robot, "preview_motion"):
            raise ValueError("motion_review_backend_unavailable")
        self.capabilities["motion_review_enabled"] = self.motion_review
        self.capabilities["vision_compare_enabled"] = self.vision_compare
        if self.capabilities.get("observer_video"): self.robot.record(enabled=self.config.features.record_video)
        self.store.save("configuration", "robot_capabilities-"+digest(self.capabilities), self.capabilities)
        self.store.set_state("runtime","capabilities",self.capabilities)
        observation = self.observe()
        self.capture_home_reference(observation)
        return observation

    def capture_home_reference(self, observation):
        if self.home_check is not None:
            self.home_check.capture_initial(observation)
            self.store.event("completion_guidance.home_reference", self.home_check.reference_record())

    def action_budget(self):
        """Public execution accounting only; never score or evaluator success."""
        used = self.native_action_count if self.native_budget_known else None
        limit = self.native_budget_limit
        return dict(native_steps_used=used, native_steps_limit=limit,
                    native_steps_remaining=max(0, limit-used) if limit is not None and used is not None else None,
                    source="native_execution_accounting", unknown_values_are_null=True,
                    per_request_native_limit=250 if (self.capabilities or {}).get("backend") == "robodojo_curobo" else None,
                    semantics="official native steps; separate from tool requests and model-call budget; reserves are not included")

    def publish_observation(self, raw, correlation=None, last_action_id=None, make_current=True):
        observation_id = new_id("obs")
        cameras = []
        require_actor_observation(raw, self.observation_profile)
        state = actor_robot_state(raw["robot_state"], self.observation_profile)
        calibrated = has_calibration(self.observation_profile)
        calibration_ref = pose_ref = model_ref = None
        if calibrated:
            if not all(raw.get(key) for key in ("camera_calibration", "camera_pose_sample", "robot_model")):
                raise ValueError("calibrated_observation_missing_geometry")
            calibration_ref = self.store.put_artifact(canonical(raw["camera_calibration"]).encode(), "camera_calibration", "application/json")
            pose_ref = self.store.put_artifact(canonical(raw["camera_pose_sample"]).encode(), "camera_pose_sample", "application/json")
            model_ref = self.store.put_artifact(canonical(raw["robot_model"]).encode(), "robot_model", "application/json")
        for camera in raw["cameras"]:
            data = camera_path(camera).read_bytes()
            encoding = camera.get("image_encoding")
            if not (encoding and encoding.get("format") == "jpeg"
                    and encoding.get("quality") == self.image_settings["storage_jpeg_quality"]
                    and encoding.get("sha256") == hashlib.sha256(data).hexdigest()):
                data, encoding = encode_image(data, self.image_settings, storage=True)
            ref = self.store.put_artifact(data, "rgb", "image/jpeg")
            if camera.get("image_encoding") == encoding:
                share_capture_file(self.store, camera_path(camera), ref)
            cameras.append(dict(camera_id=camera["camera_id"], frame_id=camera.get("frame_id") or new_id("frame"), rgb_ref=ref,
                                width=camera["width"], height=camera["height"], pixel_transform="identity",
                                capture_time=raw.get("captured_at"), clock_domain=raw.get("clock_domain"),
                                sim_step=raw.get("sim_step"), step_domain=raw.get("step_domain"),
                                synchronization=camera.get("capture_alignment"),
                                control_dt_s=raw.get("control_dt_s"), image_encoding=encoding))
            if calibrated: cameras[-1].update(calibration_ref=calibration_ref, pose_sample_ref=pose_ref)
            if allows_depth(self.observation_profile):
                from ..perception.depth.depth_capture import validate_depth_archive
                depth_data = Path(camera["depth_path"]).read_bytes()
                validate_depth_archive(depth_data, camera)
                depth_ref = self.store.put_artifact(depth_data, "depth", "application/x-npz")
                cameras[-1].update(depth_ref=depth_ref, **{key: camera[key] for key in
                    ("depth_kind", "depth_unit", "depth_clip_m", "depth_shape")})
        state_ref = self.store.put_artifact(canonical(state).encode(), "robot_state", "application/json")
        bundle = dict(schema_version="0.1", **self.identity, world_kind="observed", observation_id=observation_id,
                      sim_step=raw.get("sim_step"), step_domain=raw.get("step_domain"), sim_time_s=raw.get("sim_time_s"),
                      clock_domain=f'{self.identity["env_instance_id"]}/generation-{self.identity["env_generation"]}',
                      observation_profile=self.observation_profile, robot_model_id=(self.capabilities or {}).get("robot_model_id", "arx-x5"),
                      calibration_id=digest(raw["camera_calibration"]) if calibrated else None, cameras=cameras, robot_state_ref=state_ref, last_action_id=last_action_id,
                      data_origin=raw.get("data_origin", "robodojo_runtime"))
        bundle["action_budget"] = self.action_budget()
        if calibrated: bundle["robot_model_ref"] = model_ref
        # Native public language instruction, never object poses or evaluator state.
        if raw.get("instruction") is not None:
            bundle["task_instruction"] = raw["instruction"]
        validate_observation(bundle)
        self.store.save("observation", observation_id, bundle)
        if self.scene_artifacts is not self.store:
            for reference in [state_ref, calibration_ref, pose_ref, model_ref] + [camera[key] for camera in cameras for key in ("rgb_ref", "depth_ref") if key in camera]:
                if reference:
                    info, content = self.store.resolve_artifact(reference)
                    copied = self.scene_artifacts.put_artifact(content, info["kind"], info["mime_type"], info["access_scope"])
                    if copied != reference: raise ValueError("scene_artifact_copy_identity_mismatch")
        self.scene.import_observation(bundle)
        if make_current:
            self.latest = bundle
            # During motion a progress frame can precede the final robot state.
            # Only an explicit terminal capture below makes it usable again.
            if self.home_check is not None and not self.active_action:
                self.home_check.update_current(bundle)
        self.store.event("observation.available", {"observation_id": observation_id, "historical": not make_current},
                         {**(correlation or {}), "observation_id": observation_id},
                         sim_step=bundle["sim_step"], step_domain=bundle["step_domain"], sim_time_s=bundle["sim_time_s"])
        return bundle

    def observe(self, correlation=None):
        with self.robot_lock:
            return self.publish_observation(self.robot.observe(), correlation)

    def current_observation(self):
        if self.latest is None:
            raise ValueError('current_observation_unavailable')
        return self.latest

    def action(self, action_id):
        request = self.store.read("action_request", action_id)
        return dict(request=request, receipt=self.states.get(action_id, {"status": "unknown"}))

    def reset(self, seed=0, layout_id=0):
        with self.lock:
            if self.active_action: raise ValueError("reset_requires_confirmed_motion_stop")
            if not self.capabilities.get("reset"): raise ValueError("unsupported_capability:reset")
            if hasattr(self.robot,"validate_reset"): self.robot.validate_reset(seed,layout_id)
            self.accepting=False
            stop_revision=self.stop_revision
            self.cancel.clear()
        with self.robot_lock:
            previous=dict(self.identity)
            self.identity.update(episode_id=new_id("ep"),env_generation=self.identity["env_generation"]+1)
            self.latest=None
            if self.home_check is not None:
                self.home_check.invalidate_current("environment_reset")
            self.episode_done.clear()
            self.episode_terminal = None
            self.native_action_count = 0
            self.native_budget_known = self.native_budget_limit is not None
            self.started_at = self.last_native_progress_at = time.monotonic()
            self.model_wait_at_start = self.model_wait_at_native_progress = getattr(getattr(self, "model_proxy", None), "model_wait_seconds", 0.0)
            self.failed_paths.clear()
            self.store.set_state("runtime","identity",dict(self.identity))
            self.store.set_state("runtime","reset",dict(**self.identity,status="starting"))
            self.store.event("environment.reset_requested",{"previous_identity":previous,"seed":seed,"layout_id":layout_id})
            try:
                result=self.robot.reset(seed=seed,layout_id=layout_id)
                if getattr(self.robot,"process",None):
                    self.store.set_state("runtime","robot",{"pid":self.robot.process.pid,"run_dir":str(self.robot.run_dir)})
                if self.capabilities.get("observer_video"): self.robot.record(enabled=self.config.features.record_video)
                observation=self.publish_observation(result["observation"])
                self.capture_home_reference(observation)
            except Exception as error:
                self.store.set_state("runtime","reset",dict(**self.identity,status="unknown",reason=str(error)))
                self.store.event("environment.reset_unknown",{"reason":str(error),"previous_identity":previous})
                raise RuntimeError("reset_acknowledgement_unknown:"+str(error)) from error
            with self.lock:
                self.accepting=self.stop_revision==stop_revision
                if self.accepting: self.cancel.clear()
            self.store.set_state("runtime","reset",dict(**self.identity,status="completed"))
            self.store.event("environment.reset",{"previous_identity":previous,"seed":seed,"layout_id":layout_id,"observation_id":observation["observation_id"]})
        return observation

    def execute(self, action, correlation=None):
        from ..robot.motion.motion_speed import resolve_action_speed
        original_speed = {key: action[key] for key in SPEED_INPUT_FIELDS if key in action}
        action = resolve_action_speed(action, self.config.features.default_motion_speed_scale)
        action_id = action.get("action_id")
        if not action_id:
            raise ValueError("missing_action_id")
        correlation = {**(correlation or {}), "action_id": action_id, "observation_id": action.get("based_on_observation_id")}
        with self.lock:
            try:
                old = self.store.read("action_request", action_id)
            except KeyError:
                old = None
            if old is not None:
                if digest(resolve_action_speed(old, self.config.features.default_motion_speed_scale)) != digest(action):
                    raise ValueError("action_id_content_conflict")
                return self.action(action_id)
            self.store.save("action_request", action_id, action)
            self.store.event("action.requested", {"request": action}, correlation)
            self.store.event("action.speed_resolved", dict(action_id=action_id, input=original_speed,
                preview_plan_id=action.get("preview_plan_id"),
                default_motion_speed_scale=self.config.features.default_motion_speed_scale,
                effective={key: action[key] for key in SPEED_EFFECTIVE_FIELDS if key in action}), correlation)
            try:
                self.current_observation()
                if self.programs and self.programs.unresolved_jobs():
                    raise ValueError("program_stop_unconfirmed")
                if action.get("type") == "ArmQueueAction" and not self.config.features.arm_queue:
                    raise ValueError("arm_queue_not_enabled_for_run")
                self.task_planning.require(self.identity)
                if action.get("type") == "JointPrefixAction":
                    validate_identity(action)
                    if any(action.get(key) != self.identity[key] for key in IDENTITY): raise ValueError("stale_environment_identity")
                    if action.get("world_kind") != "observed": raise ValueError("internal_action_cannot_execute_observed_world")
                    if action.get("based_on_observation_id") != self.latest["observation_id"]: raise ValueError("stale_observation")
                else:
                    validate_action(action, self.identity, self.latest["observation_id"], self.capabilities)
                validate_backend_action(self.robot, action, self.capabilities)
                if action.get("preview_plan_id"):
                    raise ValueError("motion_preview_unavailable")
                if not self.accepting:
                    raise ValueError("run_stopping_or_reconciliation_required")
                if correlation.get('program_id'):
                    if self.programs is None:
                        raise ValueError('program_channel_expired')
                    self.programs.admit_action(correlation['program_id'], action['timeout_s'])
                if self.active_action:
                    raise ValueError("action_writer_busy")
                if self.max_actions is not None and self.action_count >= self.max_actions:
                    raise ValueError("action_budget_exhausted")
                if action.get("path") and digest(action["path"]) in self.failed_paths:
                    self.store.event("action.prior_planner_failure", dict(action_id=action_id,
                        assessment_role="advisory", message="The same path previously failed planning; the actor chose to try again."), correlation)
                self.invalidate_scene(action_id)
            except ValueError as error:
                receipt = dict(status="rejected", reason=str(error), result_observation_id=None,
                               observation_unavailable_reason="action_not_executed")
                self.states[action_id] = receipt
                self.store.save("action_receipt", action_id, receipt)
                self.store.event("action.rejected", receipt, correlation)
                return self.action(action_id)
            self.cancel.clear()
            self.active_action = action_id
            if self.home_check is not None:
                self.home_check.invalidate_current("action_in_progress")
            self.active_context = correlation
            self.action_count += 1
            self.states[action_id] = dict(status="accepted")
            task_plan = self.task_planning.current(self.identity)
            self.store.event("action.accepted", {"task_plan_id": task_plan["plan_id"] if task_plan else None}, correlation)

        wait_observations = []

        def progress(value):
            value = dict(value)
            self._native_progress(value.pop("episode_terminal", None))
            raw_observation = value.pop("observation", None)
            if raw_observation:
                value["observation_id"] = self.publish_observation(raw_observation, correlation, action_id)["observation_id"]
                steps = value.get("executed_steps", 0)
                if ("hold_native_steps" in action and steps and steps % 10 == 0
                        and steps < action["hold_native_steps"]):
                    wait_observations.append(value["observation_id"])
            stage = value.get("phase", "progress")
            self.states[action_id] = dict(status="running", progress=value)
            self.store.event("action.started" if stage == "started" else "action.progress", value, correlation)

        acknowledged_receipt = None
        try:
            with self.robot_lock:
                raw_receipt = self.robot.execute(action, progress_callback=progress, cancel_flag=self.cancel)
                episode_terminal = raw_receipt.pop("episode_terminal", None)
                terminal_raw = raw_receipt.pop("terminal_observation", None)
                before_raw = raw_receipt.pop("before_observation", None)
                acknowledged_receipt = _execution_acknowledgement(raw_receipt)
                self._native_progress(episode_terminal)
                from ..robot.motion.motion_failure import archive_collision_witnesses
                archive_collision_witnesses(self.store, raw_receipt, self.observation_profile)
                reject_privileged(raw_receipt, self.observation_profile,
                    stage_ids=[stage["id"] for stage in action["stages"]] if action.get("type") == "BatchAction" else None)
                validate_display(raw_receipt)
                receipt = dict(raw_receipt)
                # Execution is already acknowledged. Keep the public receipt
                # apart from evidence which can still fail to publish, and
                # never leave unprojected state/trace bodies in a fallback reply.
                details = {name: receipt.pop(name, None) for name in
                           ("planned_trace", "dispatched_trace", "actual_trace", "before_state")}
                acknowledged_receipt = receipt
                if "hold_native_steps" in action:
                    receipt["wait_observation_ids"] = wait_observations
                if details.get("before_state"):
                    details["before_state"] = actor_robot_state(details["before_state"], self.observation_profile)
                for sample in details.get("actual_trace") or []:
                    if "state" in sample: sample["state"] = actor_robot_state(sample["state"], self.observation_profile)
                if before_raw:
                    # Start-frame capture predates progress; keep it as historical evidence.
                    start_bundle = self.publish_observation(before_raw, correlation, make_current=False)
                    receipt["execution_start_observation_id"] = start_bundle["observation_id"]
                for name in ["planned_trace", "dispatched_trace", "actual_trace", "before_state"]:
                    value = details[name]
                    receipt[name + "_ref"] = self.store.put_artifact(canonical(value).encode(), name, "application/json") if value is not None else None
                if terminal_raw:
                    terminal = self.publish_observation(terminal_raw, correlation, action_id)
                    receipt["result_observation_id"] = terminal["observation_id"]
                    if action.get("input_grasp_pose"):
                        try:
                            feedback = contact_target_feedback(action,
                                actor_robot_state(terminal_raw["robot_state"], self.observation_profile))
                            # Keep a diagnostic serialization failure from changing an executed action's status.
                            canonical(feedback)
                            if feedback:
                                receipt["grasp_contact_feedback"] = feedback
                        except Exception as error:
                            receipt["grasp_contact_feedback"] = dict(available=False,
                                reason="diagnostic_unavailable", error_type=type(error).__name__)
                    if self.home_check is not None:
                        self.home_check.update_current(terminal)
                else:
                    receipt.update(result_observation_id=None, observation_unavailable_reason="backend_terminal_capture_unavailable")
                    if receipt.get('status') != 'rejected' and receipt.get('physics_advanced') is not False:
                        self.latest = None
                    if self.home_check is not None:
                        self.home_check.invalidate_current("backend_terminal_capture_unavailable")
        except Exception as error:
            self.latest = None
            if self.home_check is not None:
                self.home_check.invalidate_current("post_execution_publication_failed"
                    if acknowledged_receipt is not None else "worker_acknowledgement_or_capture_failed")
            if acknowledged_receipt is not None:
                receipt = acknowledged_receipt
                receipt.update(result_observation_id=None,
                    observation_unavailable_reason="post_execution_observation_or_evidence_publication_failed",
                    post_execution_error=dict(phase="after_execution_acknowledgement",
                        error_type=type(error).__name__, message=str(error), action_replay_permitted=False))
                self.store.event("action.evidence_publication_failed", receipt["post_execution_error"], correlation)
                # Missing current observation closes new-action admission until
                # fresh observe succeeds. It does not erase confirmed movement
                # or reopen a stop/unknown gate which was already closed.
            else:
                receipt = dict(status="unknown", reason=str(error), result_observation_id=None,
                               observation_unavailable_reason="worker_acknowledgement_or_capture_failed")
                self.accepting = False
        with self.lock:
            receipt["action_budget"] = self.action_budget()
            self.states[action_id] = receipt
            self.store.save("action_receipt", action_id, receipt)
            status = receipt.get("status", "unknown")
            if execution_unconfirmed(receipt):
                self.accepting = False
            event_type = "action.status_unknown" if status in ("unknown", "status_unknown") else "action." + status
            self.store.event(event_type, receipt, correlation)
            self.active_action = None
            self.active_context = None
            if action.get("path"):
                if receipt.get("reason") == "path_infeasible":
                    self.failed_paths.add(digest(action["path"]))
                elif receipt.get("status") == "completed":
                    self.failed_paths.clear()
        return self.action(action_id)

    def _native_progress(self, terminal):
        """Keep evaluator state private; publish only filtered execution budget."""
        if terminal is None:
            return
        with self.lock:
            count = terminal.get("native_action_count", 0)
            if type(count) is int and count >= 0:
                self.native_budget_known = True
            limit = terminal.get("native_step_limit")
            if type(limit) is int and limit > 0:
                self.native_budget_limit = limit
            if count > self.native_action_count:
                self.native_action_count = count
                self.last_native_progress_at = time.monotonic()
                self.model_wait_at_native_progress = getattr(getattr(self, "model_proxy", None), "model_wait_seconds", 0.0)
                self.failed_paths.clear()
            self.store.set_state("runtime", "native_episode", terminal)
            if terminal.get("ended") and not self.episode_done.is_set():
                self.episode_terminal = terminal
                self.accepting = False
                reference = self.store.put_artifact(canonical(terminal).encode(), "native_terminal", "application/json", "observer")
                self.store.event("environment.ended", {"result_ref": reference}, source="private_evaluator")
                self.episode_done.set()
                if self.on_episode_end:
                    self.on_episode_end()

    def invalidate_scene(self, action_id):
        head=self.scene.head(self.identity)
        if not head["scene_version"]: return
        current=self.scene.version(self.identity,head["scene_version"])
        operations=[dict(op="mark_dirty",entity_id=entity,source_kind="observation_inference",reason="Observed motion may change geometry; no contact extent is known.") for entity,value in current["entities"].items() if value["state"] not in ("retired","dirty")]
        if not operations: return
        patch=dict(schema_version="0.1",**self.identity,world_kind="internal",base_scene_version=head["scene_version"],observation_ids=[self.latest["observation_id"]],builder_digest=digest("observed-action-invalidation-v1"),operations=operations)
        try:
            result=self.scene.apply_patch(patch)
            self.store.event("scene.invalidated",{"scene_version":result["scene_version"],"reason":"observed_action_admitted"},{"action_id":action_id},world_kind="internal")
        except ValueError as error:
            # Admission fails closed if a concurrent scene commit prevents invalidation.
            raise ValueError("scene_invalidation_conflict:"+str(error)) from error

    def _review_controls(self, correlation):
        identifier = correlation.get('program_id')
        if identifier is None:
            return dict(cancel_event=self.cancel)
        if self.programs is None:
            raise ValueError('program_channel_expired')
        return self.programs.review_controls(identifier)

    def _model_context_epoch(self):
        proxy = getattr(self, "model_proxy", None)
        # Codex: the identity's instance_id is runtime/codex app_server_instance_id (same value as before).
        epoch = (dict(self.identity), self.thread_id,
                 self.store.read_state("runtime", "image_context_epoch"),
                 (read_agent_identity(self.store) or {}).get("instance_id"),
                 getattr(proxy, "transport_id", None), getattr(proxy, "image_context_epoch", None))
        if self.context_epoch_volatile:
            # Forced full delivery: no two calls share an epoch, so every delivery baseline (views, static
            # metadata, capability summary, program logs) is discarded and each tool result is complete.
            epoch += (new_id("volatile-context"),)
        return epoch

    def model_context_reset(self, reason):
        """The agent session reports that its model may have lost earlier context (``AgentSession.on_context_reset``).

        reason: short non-empty str naming the cause, e.g. ``"native_context_compaction"``,
        ``"relay_context_discontinuity"``, ``"tool_reply_delivery_uncertain"`` or ``"agent_resume_after_unclean_exit"``;
        a non-str or empty value is recorded as ``"unspecified"``. On the ``REMINDER_IN_TOOL_RESULT`` channel it sets
        ``decision_reminder_due`` so the next actor tool response carries the decision-context reminder; on the Codex
        channel the flag is left alone (Codex reminds through its compaction request). Always records the observer
        event ``model_context.reset`` ``{reason, decision_reminder_channel, decision_reminder_due}``. The delivery
        epoch is not changed here: the session has already written ``runtime/image_context_epoch``. Safe from any
        thread; never raises for an unwritable Store (the flag is the control effect, the event is audit only).
        Returns None.
        """
        reason = reason if isinstance(reason, str) and reason else "unspecified"
        sideband = self.decision_reminder_channel == REMINDER_IN_TOOL_RESULT
        if sideband:
            with self._decision_reminder_lock:
                self.decision_reminder_due = True
        try:
            self.store.event("model_context.reset", dict(reason=reason,
                decision_reminder_channel=self.decision_reminder_channel, decision_reminder_due=sideband))
        except Exception:
            pass

    def _deliver_decision_reminder(self, response, correlation):
        """Append the due decision-context reminder to one actor tool ``response`` (design 04 §4.5).

        The appended ``inputText`` item carries ``decision_reminder_text(task_planning.decision_context(...))`` after
        ``ActorPresentation.present_text`` -- the same text Codex's compaction request carries after its egress scrub.
        Exactly one response consumes the flag. Nothing is added (and the flag clears) when no plan is stored and
        planning is optional, as on Codex. If the reminder cannot be built, ``response`` is returned unchanged with
        every executed fact and receipt intact, the flag stays due for the next response and the observer event
        ``model_context.decision_reminder_failed {error_type}`` is recorded. A response that is not a
        ``{"contentItems": [...]}`` envelope is returned unchanged and leaves the flag due.
        """
        if not self.decision_reminder_due:
            return response
        if not isinstance(response, dict) or not isinstance(response.get("contentItems"), list):
            return response
        failure = None
        with self._decision_reminder_lock:
            if not self.decision_reminder_due:
                return response
            try:
                context = self.task_planning.decision_context(self.store.identity)
                text = (None if context is None and not self.task_planning.required
                        else self.actor_presentation.present_text(decision_reminder_text(context)))
            except Exception as error:
                failure = type(error).__name__
            else:
                self.decision_reminder_due = False
        if failure is not None:
            try:
                self.store.event("model_context.decision_reminder_failed", dict(error_type=failure), correlation)
            except Exception:
                pass
            return response
        if text is not None:
            response = dict(response, contentItems=[*response["contentItems"], dict(type="inputText", text=text)])
        try:
            self.store.event("model_context.decision_reminder_delivered", dict(
                included=text is not None, plan_revision=(context or {}).get("revision")), correlation)
        except Exception:
            pass
        return response

    def observation_content(self, bundle, correlation, camera_id=None, roi=None, scale=1, numeric_only=False,
                            *, feedback_action=None, feedback_receipt=None, detail=None,
                            feedback_context=None, full_view_reason=None, metadata_delivery=True, structured=False):
        compact_delivery = resolve_detail(detail, program=bool(correlation.get("program_id")),
            default="full" if self.actor_payload_profile == "full_v1" else "summary") == "summary"
        if (roi is not None or scale != 1) and camera_id is None: raise ValueError("image_view_requires_camera_id")
        if type(scale) is not int or not 1 <= scale <= 4: raise ValueError("invalid_image_scale")
        if camera_id is not None and camera_id not in [camera["camera_id"] for camera in bundle["cameras"]]: raise ValueError("unknown_camera")
        require_actor_observation(bundle, self.observation_profile)
        _, state_bytes = self.store.resolve_artifact(bundle["robot_state_ref"], allowed_scopes=("actor",))
        metadata = dict(bundle=bundle, robot_state=actor_robot_state(json.loads(state_bytes), self.observation_profile))
        if has_calibration(self.observation_profile) and not numeric_only:
            metadata["camera_calibration"] = json.loads(self.store.resolve_artifact(bundle["cameras"][0]["calibration_ref"])[1])
            metadata["camera_pose_sample"] = json.loads(self.store.resolve_artifact(bundle["cameras"][0]["pose_sample_ref"])[1])
        for state in metadata["robot_state"].get("arms", {}).values():
            pose = state.get("ee_pose", {})
            if pose.get("tcp_config_id") != "arx-x5-link6-v1" or pose.get("source_kind") != "measurement": continue
            x, y, z, w = pose["quaternion_xyzw"]
            values = roboprobe_pose.pose_to_values([*pose["position_m"], w, x, y, z])
            state["grasp_pose"] = dict(position_m=[values[key] for key in ("x", "y", "z")],
                angles_deg=[values[key] for key in ("pitch_deg", "roll_deg", "yaw_deg")],
                frame_id=pose["frame_id"], source_kind="derived_from_measured_link6", convention="robot_move_grasp",
                contact_region="pad_center", contact_depth_m=0.,
                contact_point_meaning="center_between_fingers_at_distal_15mm_pad_center")
        content = [StructuredText(metadata, projection="observation")]
        delivered = []
        resolved_images = {}
        selection = None
        is_model = not bool(correlation.get("program_id"))
        if is_model:
            epoch = self._model_context_epoch()
            if epoch != self._view_context_epoch:
                self._last_delivered_views.clear()
                self._view_context_epoch = epoch
        if (self.image_settings.get("feedback_policy", "all") == "adaptive" and is_model
                and not numeric_only and camera_id is None and roi is None and scale == 1):
            from ..perception.view_selection import select_observation_views
            for camera in bundle["cameras"]:
                resolved_images[camera["camera_id"]] = self.store.resolve_image(camera["rgb_ref"], allowed_scopes=("actor",))
            active = None
            action = feedback_action or {}
            if action.get("type") == "ArmQueueAction":
                active = [arm for arm, stages in action.get("arm_queues", {}).items()
                          if any(stage.get("kind") in ("move", "gripper") for stage in stages)]
            elif action.get("path") and not action.get("hold_native_steps"):
                active = sorted({arm for point in action["path"].get("waypoints", [])
                                 for arm in ("left", "right") if arm in point}
                                | {event["arm"] for event in action.get("gripper_events", [])
                                   if event.get("arm") in ("left", "right")})
            receipt = feedback_receipt or {}
            from ..core.presentation.feedback_delivery import execution_exception
            if not full_view_reason and execution_exception(receipt):
                full_view_reason = "execution_exception"
            calibration_valid = {}
            for camera in bundle["cameras"]:
                try:
                    self.geometry.camera(self.identity, bundle["observation_id"], camera["camera_id"])
                    calibration_valid[camera["camera_id"]] = True
                except Exception:
                    calibration_valid[camera["camera_id"]] = False
            try:
                selection = select_observation_views(bundle,
                    {key: value[1] for key, value in resolved_images.items()}, self._last_delivered_views,
                    active_arms=active, context=feedback_context or ("post_action" if feedback_action is not None else "explicit_observe"), receipt_status=receipt.get("status"),
                    receipt_reason=receipt.get("reason"),
                    force_full=bool(full_view_reason or receipt.get("execution_uncertain")),
                    full_reason=full_view_reason or ("execution_uncertain" if receipt.get("execution_uncertain") else None),
                    calibration_valid=calibration_valid)
            except Exception as error:
                # Image optimization must never obscure an already executed action.
                selection = dict(selected_camera_ids=[camera["camera_id"] for camera in bundle["cameras"]],
                    skipped_camera_ids=[], fallback_reason="selection_failed:" + type(error).__name__)
            metadata["image_feedback"] = dict(policy="adaptive", observation_id=bundle["observation_id"],
                included_camera_ids=selection["selected_camera_ids"], omitted_camera_ids=selection["skipped_camera_ids"],
                retrieval=dict(tool="history_view_observation", observation_id=bundle["observation_id"],
                    camera_id="choose an omitted camera ID; saved image; no native steps"),
                full_view_reason=selection.get("force_full_reason", selection.get("fallback_reason")),
                omission_meaning="Original view is below all change thresholds versus its last actual model delivery; not proof of unchanged task state. Omitted views remain retrievable.")
            metadata["image_feedback"]["omitted_views"] = [
                {key: decision[key] for key in ("camera_id", "reason", "baseline_observation_id", "retrieval")
                 if key in decision} for decision in selection.get("decisions", []) if not decision["include"]]
        baselines, baseline_errors = {}, []
        for camera in bundle["cameras"]:
            if numeric_only:
                continue
            if camera_id is not None and camera["camera_id"] != camera_id: continue
            if selection is not None and camera["camera_id"] not in selection["selected_camera_ids"]: continue
            artifact, data = (resolved_images[camera["camera_id"]] if camera["camera_id"] in resolved_images
                              else self.store.resolve_image(camera["rgb_ref"], allowed_scopes=("actor",)))
            if (is_model and roi is None and scale == 1
                    and self.image_settings.get("feedback_policy", "all") == "adaptive"):
                from ..perception.view_selection import baseline_record
                try:
                    baselines[camera["camera_id"]] = baseline_record(bundle, camera["camera_id"], data)
                except Exception as error:
                    # This optional bookkeeping must not obscure an executed
                    # action. With no baseline the next view is included.
                    self._last_delivered_views.pop(camera["camera_id"], None)
                    baseline_errors.append(dict(camera_id=camera["camera_id"], reason=type(error).__name__))
            reference = camera["rgb_ref"]
            delivery = dict(camera_id=camera["camera_id"], rgb_ref=reference, sha256=artifact["sha256"], preprocessing="identity")
            if artifact.get("reconstruction"):
                delivery["reconstruction"] = artifact["reconstruction"]
                metadata["bundle"] = self.store.observation_view(bundle)
            if roi is not None or scale != 1:
                from PIL import Image
                bounds = roi if roi is not None else [0, 0, camera["width"], camera["height"]]
                if not isinstance(bounds, list) or len(bounds) != 4 or any(type(value) is not int for value in bounds): raise ValueError("invalid_image_roi")
                x, y, width, height = bounds
                if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > camera["width"] or y + height > camera["height"]: raise ValueError("image_roi_out_of_bounds")
                with Image.open(io.BytesIO(data)) as original:
                    cropped_image = original.crop((x, y, x+width, y+height)).resize((width*scale, height*scale), Image.Resampling.LANCZOS)
                    data, _ = encode_image(cropped_image, self.image_settings, storage=True)
                reference = self.store.put_artifact(data, "rgb_crop", "image/jpeg")
                cropped, _ = self.store.artifact(reference)
                delivery.update(rgb_ref=reference, original_rgb_ref=camera["rgb_ref"], sha256=cropped["sha256"], preprocessing="crop_resize_lanczos",
                    source_roi=bounds, display_scale=scale, width=width*scale, height=height*scale,
                    pixel_transform=[[scale, 0, scale*(.5-x)-.5], [0, scale, scale*(.5-y)-.5], [0, 0, 1]])
                metadata["image_view"] = delivery
            url, encoding = image_url(data, self.image_settings)
            delivery.update(source_sha256=delivery["sha256"], sha256=encoding["sha256"], encoding=encoding)
            content.append(dict(type="inputImage", imageUrl=url))
            delivered.append(delivery)
        if numeric_only:
            metadata = dict(observation_id=bundle["observation_id"], robot_state=metadata["robot_state"],
                            action_budget=bundle.get("action_budget"),
                            images="saved; use history_view_observation when needed")
        elif compact_delivery and self.actor_payload_profile == "compact_v1":
            from ..core.presentation.actor_payload import compact_observation
            metadata = compact_observation(metadata)
        elif compact_delivery:
            metadata = lean_observation(metadata)
        from ..core.presentation.observation_output import MetadataDelivery, observation_metadata
        compact_geometry = compact_delivery and self.actor_payload_profile not in ("full_v1", "lean_v1")
        # Use the unprojected refs as the source of exact detail addresses: lean
        # delivery may replace expanded artifact handles with availability flags.
        if compact_geometry and "camera_calibration" in metadata:
            metadata["camera_calibration"]["full_ref"] = bundle["cameras"][0].get("calibration_ref")
            metadata["camera_pose_sample"]["full_ref"] = bundle["cameras"][0].get("pose_sample_ref")
        metadata = observation_metadata(metadata,
            current_observation_id=self.latest["observation_id"] if self.latest else None,
            current_action_budget=self.action_budget(), compact_geometry=compact_geometry)
        inline_metadata = metadata
        content[0] = StructuredText(inline_metadata, projection="observation")
        projection = None
        # The lean request projection summarizes earlier results in place, so
        # every observation must stay self-contained (no static-field refs).
        if compact_delivery and is_model and metadata_delivery and not getattr(self.config.features, "lean_projection", False):
            try:
                projection = self._metadata_delivery.prepare(metadata, epoch=epoch)
                validate_display(projection.metadata)
                content[0] = StructuredText(projection.metadata, projection="observation")
            except Exception:
                # Optional transport optimization cannot hide an already
                # executed action or teach undelivered static conventions.
                self._metadata_delivery = MetadataDelivery()
                projection = None
                content[0] = StructuredText(inline_metadata, projection="observation")
        # Counterfactual transport size uses the same encoder and is telemetry only.
        omitted_bytes_error = None
        try:
            omitted_transport_bytes = sum(image_url(resolved_images[key][1], self.image_settings)[1]["byte_size"]
                                          for key in (selection or {}).get("skipped_camera_ids", []))
        except Exception as error:
            omitted_transport_bytes, omitted_bytes_error = None, type(error).__name__
        metrics = dict(image_count=len(delivered), transport_bytes=sum(item["encoding"]["byte_size"] for item in delivered),
                       omitted_image_count=len((selection or {}).get("skipped_camera_ids", [])),
                       omitted_transport_bytes=omitted_transport_bytes)
        if baseline_errors:
            metrics["baseline_errors"] = baseline_errors
        if omitted_bytes_error:
            metrics["omitted_transport_bytes_unavailable"] = omitted_bytes_error
        self.store.save("delivery", new_id("delivery"), dict(observation_id=bundle["observation_id"], correlation=correlation,
                                                             image_selection=selection, image_metrics=metrics,
                                                             images=delivered, delivered_to_actor=not bool(correlation.get("program_id")), audience="actor_program" if correlation.get("program_id") else "astra_model", delivery_channel="program_fd" if correlation.get("program_id") else self.delivery_channel))
        self.store.event("observation.delivered", {"images": delivered, "image_selection": selection, "image_metrics": metrics,
            "audience":"actor_program" if correlation.get("program_id") else "astra_model","delivered_to_model":is_model},
            {**correlation, "observation_id": bundle["observation_id"]})
        self._last_delivered_views.update(baselines)
        if projection is not None:
            try:
                committed = self._model_context_epoch() == epoch and self._metadata_delivery.commit(projection)
            except Exception:
                self._metadata_delivery = MetadataDelivery()
                committed = False
            if not committed:
                # A compaction/reset during assembly can invalidate referenced
                # static fields. Deliver the complete inline projection instead.
                content[0] = StructuredText(inline_metadata, projection="observation")
        return content if structured else wire_content(content)

    def handle_tool(self, params):
        if params.get("threadId") != self.thread_id:
            raise ValueError("tool_call_thread_mismatch")
        name, args = params["tool"], params["arguments"]
        if not isinstance(args, dict):
            raise ValueError("tool_arguments_must_be_object")
        correlation = dict(thread_id=params["threadId"], turn_id=params["turnId"], tool_call_id=params["callId"],
                           item_id=self.call_items.get(params["callId"], params["callId"]))
        if name in {'program_run', 'program_status'}:
            correlation['actor_log_delivery'] = True
        self.store.save("tool_call", params["callId"], dict(params=params, correlation=correlation,
                        tool_call_id_source=params.get("callIdSource", "native_callId")))
        try:
            result = self.invoke(name, self.actor_output.restore(args), correlation, role="actor")
            response = self.actor_output.present(result, tool=name)
            if getattr(self, "_lean_compact", None) is not None:
                try:
                    epoch = self._model_context_epoch()
                except Exception:
                    epoch = None
                response = self._lean_compact.apply(name, args, response, epoch=epoch, thread=self.thread_id,
                                                    call_id=params["callId"])
        except Exception:
            self._invalidate_actor_delivery(params['callId'])
            raise
        if not response.get('success', False):
            self._invalidate_actor_delivery(params['callId'])
        return self._deliver_decision_reminder(response, correlation)

    def _invalidate_actor_delivery(self, call_id):
        # A producer may already have prepared RGB/program baselines when the
        # final projection fails. Nothing from that failed reply is reusable.
        self._program_output_delivery.invalidate()
        self.store.set_state('runtime', 'image_context_epoch', dict(
            source='tool_output_failed', tool_call_id=call_id,
            marker=new_id('tool-output-failure')))

    def robot_information(self, section="default", robot_id=None):
        """One interface for the native geometry, static reference and calibration."""
        from ..robot.model.robot_information import robot_information, robot_resource_path
        from ..robot.model.robot_calibration_information import robot_calibration_information
        current = (self.capabilities or {}).get("robot_model_id") or "unknown"
        selected = robot_id if robot_id is not None else current
        if section in {"default", "calibration"}:
            calibration = (robot_calibration_information(self) if selected == current else
                           dict(available=False, reason="calibration_robot_mismatch"))
        if section == "calibration":
            result = dict(robot_id=selected, section=section,
                available=calibration["available"], calibration=calibration)
        else:
            result = robot_information(selected, section)
            if section == "default":
                result["calibration"] = calibration
                if result.get("available"):
                    geometry = result["reference"]["geometry"]
                    path = robot_resource_path(selected, geometry["resource_id"])
                    if path is None:
                        raise ValueError("robot_geometry_archive_unavailable")
                    geometry["resource_ref"] = self.store.put_artifact(
                        path.read_bytes(), "robot_geometry_model", "application/zip")
        # This API also serves startup directly, before the registry boundary.
        archived = self._archive_tool_result("robot_information", ToolOutput(result).response(),
            dict(robot_id=selected, section=section), role="internal")
        if archived.contract_error is not None:
            result = dict(result, output_contract_error=archived.contract_error)
        return self.actor_output.present(result, projection="robot_information")

    def _scheduling_advisory(self, receipt, correlation):
        if (correlation.get("program_id") or not self.config.features.task_scheduling
                or not self.config.features.manipulation_protocol):
            return None
        from ..cortex.context.task_planning import scheduling_failure_advisory
        # Do not bind an immutable action replay to a newer mutable plan. The
        # concise advice asks the actor to reassess its current commitments.
        return scheduling_failure_advisory(receipt)

    def unified_action_postprocess(self, result, correlation, *, settle=None):
        if settle is None or result["receipt"].get("status") != "completed":
            return result
        from .actions.stability import bounded_settle
        result = dict(result)
        result["settling"] = bounded_settle(self, settle, correlation)
        result["current_observation_id"] = self.latest["observation_id"] if self.latest else None
        result["result_observation_is_current"] = (
            result["receipt"].get("result_observation_id") == result["current_observation_id"])
        return result

    def invoke(self, name, args, correlation, *, role):
        result = self.tool_registry.invoke(name, args, correlation, runtime=self, role=role)
        return self._archive_tool_result(name, result, correlation, role=role)

    def _archive_tool_result(self, name, result, correlation, *, role):
        """Keep the unprojected reply locally before any actor presentation.

        Large native media stay in an immutable observer artifact. Each return
        has its own record, even when an idempotent call ID is read repeatedly.
        A storage defect after execution must never become a replayable input
        error or erase the business receipt.
        """
        record = dict(tool=name, role=role, correlation=correlation)
        payload, errors = {}, []
        original = getattr(result, "raw_response", result)
        fields = (dict(kind=original.kind, data=original.data, success=original.success)
            if isinstance(original, ToolResponse) else {})
        if isinstance(result, ToolResponse):
            fields["contract_error"] = result.contract_error
        for key, value in fields.items():
            try:
                validate_display(value)
                payload[key] = value
            except Exception as error:
                errors.append(dict(field_path="/" + key, reason="not_json_serializable",
                    error_type=type(error).__name__))
                if key == "data" and isinstance(value, dict):
                    # Keep valid top-level business fields, especially receipts,
                    # without inventing a representation for arbitrary objects.
                    payload[key] = {}
                    for field, part in value.items():
                        try:
                            validate_display({field: part})
                            payload[key][field] = part
                        except Exception:
                            errors.append(dict(field_path="/data/" + (field if isinstance(field, str) else "<non_string_key>"),
                                reason="not_json_serializable"))
        try:
            wire = wire_response(original)
            validate_display(wire)
            payload["wire_response"] = wire
        except Exception as error:
            errors.append(dict(field_path="/wire_response", reason="not_json_serializable",
                error_type=type(error).__name__))
        if errors:
            payload["archival_errors"] = errors
        record["archival_status"] = "partial" if errors else "complete"
        identifier = new_id("tool-result")
        try:
            record["raw_ref"] = self.store.put_artifact(canonical(payload).encode(),
                "tool_result", "application/json", "observer")
            self.store.save("tool_result", identifier, record)
        except Exception as error:
            record.update(archival_status="failed", error_type=type(error).__name__,
                reason="raw_result_archival_failed")
            try:
                self.store.save("tool_result", identifier, record)
            except Exception:
                pass  # The actor warning is still available when the Store is unwritable.
            return output_contract_error(result, name, "raw_result_archival_failed", "/archive")
        return result

    def dispatch(self, name, args, correlation):
        """Trusted in-process compatibility entry; actors/programs use invoke."""
        return wire_response(self.invoke(name, args, correlation, role="trusted_legacy"))

    def program_services(self):
        """Bind program permissions and atomic action ownership on the server."""
        from .programs.runner import ProgramServices
        from ..perception.external.scene3d import workspace_modules as scene3d_workspace_modules

        def validate_tool(name):
            self.tool_registry.spec(name, role='program')

        def validate_arguments(name, arguments):
            self.tool_registry.validate(name, arguments, role='program', runtime=self)

        def observation():
            with self.lock:
                return dict(observation_id=self.latest['observation_id'] if self.latest else None,
                    observation_profile=self.observation_profile)

        def active(identifier, *, cancel=False):
            with self.lock:
                owns_action = (self.active_context or {}).get('program_id') == identifier
                if owns_action and cancel:
                    self.cancel.set()
                return owns_action

        return ProgramServices(store=self.store, validate_tool=validate_tool,
            validate_arguments=validate_arguments,
            invoke=lambda name, arguments, correlation: self.invoke(name, arguments, correlation, role='program'),
            effect=lambda name, arguments: self.tool_registry.spec(name, role='program').effect_for(arguments),
            observation=observation, action_active=active,
            cancel_action=lambda identifier: active(identifier, cancel=True),
            sandbox_enabled=self.config.features.sandbox_enabled,
            workspace_modules=scene3d_workspace_modules(self))

    def policy_step(self, args, correlation):
        if "dual_arm_progression" in args:
            raise ValueError("dual_arm_progression_requires_dual_arm_motion")
        request = dict(observation_id=args["observation_id"], prefix=args.get("prefix", 5),
                       prompt=args.get("prompt"), timeout_s=args.get("timeout_s", 120))
        with self.policy_lock:
            try:
                old = self.store.read("action_request", args["action_id"])
            except KeyError:
                old = None
            if old is not None:
                if old.get("type") != "JointPrefixAction" or old.get("policy_input") != request:
                    raise ValueError("action_id_content_conflict")
                return self.action(args["action_id"])
            self.task_planning.require(self.identity)
            if self.action_policy is None: raise ValueError("pi05_policy_not_enabled_for_run")
            if request["observation_id"] != self.latest["observation_id"]: raise ValueError("stale_observation")
            prefix, prompt = request["prefix"], request["prompt"]
            if type(prefix) is not int or not 1 <= prefix <= 50: raise ValueError("invalid_policy_prefix")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
                raise ValueError("pi05_local_operation_prompt_required")
            with self.lock:
                if not self.accepting: raise ValueError("run_stopping_or_reconciliation_required")
            prediction = self.action_policy.infer(self.store.root, request["observation_id"], prompt=prompt)
            provenance = self.store.put_artifact(canonical(prediction).encode(), "policy_prediction", "application/json", "observer")
            action = dict(schema_version="0.1", **self.identity, world_kind="observed", type="JointPrefixAction",
                action_id=args["action_id"], based_on_observation_id=request["observation_id"],
                actions=prediction["actions"][:prefix], timeout_s=request["timeout_s"], policy_input=request,
                policy_provenance=dict(prediction_ref=provenance, prefix=prefix))
            return self.execute(action, correlation)

    def _action_from_args(self, name, args):
        from ..robot.motion.action_compilation import compile_action
        kind = {"robot_wait": "wait"}[name]
        return compile_action(getattr(self, "robot", None), kind, args, self.identity, self.capabilities)

    def request_stop(self):
        with self.lock:
            self.accepting = False
            self.stop_revision += 1
            self.cancel.set()
            if self.active_action:
                self.store.event("action.stop_requested", {}, {"action_id": self.active_action})
