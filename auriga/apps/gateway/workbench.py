"""Loopback workbench, persistent SSE, read-only PTY, and run supervision."""

import asyncio
import json
import secrets
import time
import fcntl
import hashlib
from pathlib import Path


from ...cortex.session.codex import CodexSession, CODEX_SDK_VERSION, CODEX_CLI_VERSION
from ...cortex.agents.base import AgentCapabilityError, read_agent_identity, session_backend_kind, session_capabilities
from ...cortex.agents.factory import EVIDENCE_CLIENT_RELAY, backend_requirements, create_session, evidence_client
from ...cortex.session.model_proxy import DEFAULT_MAX_RETRIES, RETRY_POLICY
from ...core.contracts import canonical, new_id, digest, IDENTITY
from ...core.presentation.actor_payload import ACTOR_PAYLOAD_PROFILES
from ...runtime.harness import Harness, REMINDER_IN_COMPACT_REQUEST, REMINDER_IN_TOOL_RESULT
from ...runtime.tools.builtins import create_registry, CONFIG_TOOL_NAMES
from ...robot.backends.factory import create_backend
from ...core.storage.store import Store
from ...cortex.session.terminal import Terminal
from ...runtime.programs.runner import ProgramRunner
from ...perception.scene.scene_service import SceneService, TERMINAL as JOB_TERMINAL
from ...cortex.session.usage import load_price_card
from ...core.observation_policy import ACTOR_PROFILES, LEGACY_ACTOR_PROFILE, actor_policy, has_calibration, allows_depth
from ...perception.depth.depth_runtime import INSTRUCTIONS as DEPTH_INSTRUCTIONS
from ...cortex.supervision.completion_guidance import (COMPLETION_GUIDANCE_MODES, COMPLETION_GUIDANCE_SOURCE,
                                  COMPLETION_GUIDANCE_TRIGGER, INITIAL_COMPLETION_GUIDANCE,
                                  hook_completion_guidance)
from ...cortex.context.task_notes import resolve_task_notes, handoff_policy
from ...runtime.actions.home_check import HOME_CHECK_POLICY
from ...cortex.supervision.actor_stall import ACTOR_STALL_POLICY, ACTOR_PROGRESS_POLICY, ACTOR_EVIDENCE_POLICY
from ..library.scoring import SCORING_POLICY
from ...runtime.supervision import INACTIVITY_TIMEOUT_S
from ..evaluation.episode import BENCHMARK_CONTINUATION, record_evaluation, record_video

from ...core.config import Config, cli_overrides, namespace_values
from ...core.storage.images import store_image_settings

from ...cortex.context.task_planning import PLANNING_VERSION
from ...cortex.context.public_task_rules import public_task_rules
from ...cortex.context.task_instruction_overrides import apply_task_instruction_override, task_instruction_override
from ...cortex.context.actor_instructions import CALIBRATED_OBSERVATION_INSTRUCTIONS, LEGACY_OBSERVATION_INSTRUCTIONS
from .routes import WorkbenchRoutes

_UNSET = object()

DEFAULT_PROMPT = """Development acceptance: inspect robot_capabilities and robot_observe. Using only those RGB images and measured robot state, choose one arm and move its EE a small safe distance of at most 10 mm in free space, preserving orientation and gripper. Set timeout_s=60 for this one movement. Execute at most one movement. Inspect the actual final feedback and RGB. Report request versus measurement, errors, and any uncertainty. This is an interface smoke test, not completion of the RoboDojo task. Do not inspect simulator task files or evaluation state."""


def _configured_tool_specs(config, tool_specs=(), owner=None):
    """Load optional tools only when enabled; bind reads to the owning workbench."""
    if config.features.robotic_view:
        from ...runtime.tools.plugins.robotic_view import make_robotic_view_tool, read_robotic_view

        def read_view(arguments, correlation):
            if owner is None:
                raise RuntimeError('robotic_view_owner_not_bound')
            return read_robotic_view(owner.harness, arguments, correlation)

        tool_specs = (*tool_specs, make_robotic_view_tool(read_view))
    if config.features.stereo_guide:
        from ...runtime.tools.plugins.stereo_guide import make_stereo_guide_tool, read_stereo_guide

        def read_stereo(arguments, correlation):
            if owner is None:
                raise RuntimeError('stereo_guide_owner_not_bound')
            return read_stereo_guide(owner.harness, arguments, correlation)

        tool_specs = (*tool_specs, make_stereo_guide_tool(read_stereo))
    return tool_specs


class Workbench(WorkbenchRoutes):
    def __init__(self, root, task=_UNSET, gpu=_UNSET, max_actions=_UNSET, robot=None,
                 resume=_UNSET, replay=_UNSET, observation_profile=_UNSET, seed=_UNSET, layout_id=_UNSET,
                 price_card=None, pi05=_UNSET, visual_review_model=_UNSET, actor_payload_profile=_UNSET,
                 enable_wait=_UNSET, tracking_guard=_UNSET, joint_replan=_UNSET, completion_guidance=_UNSET,
                 motion_review=_UNSET, default_task_notes=_UNSET, vision_compare=_UNSET,
                 robotic_view=_UNSET, stereo_guide=_UNSET,
                 depth_skills=_UNSET, depth_auto_feedback=_UNSET, depth_event_payload=_UNSET, config=None, tool_specs=()):
        if observation_profile is not _UNSET and replay is not True:
            actor_policy(observation_profile)
        supplied = {key: value for key, value in locals().copy().items()
                    if value is not _UNSET and key in namespace_values(Config()) and key != 'price_card'}
        for name in ('enable_wait', 'tracking_guard', 'joint_replan', 'motion_review', 'pi05', 'default_task_notes', 'vision_compare',
                     'robotic_view', 'stereo_guide', 'depth_skills', 'depth_auto_feedback', 'depth_event_payload'):
            if name in supplied and type(supplied[name]) is not bool:
                raise ValueError('invalid_control_feature_flag')
        if actor_payload_profile is not _UNSET and actor_payload_profile not in ACTOR_PAYLOAD_PROFILES:
            raise ValueError('invalid_actor_payload_profile')
        if completion_guidance is not _UNSET and completion_guidance not in COMPLETION_GUIDANCE_MODES:
            raise ValueError('invalid_completion_guidance')
        selected = (config if config is not None else Config.load()).with_overrides(cli_overrides(supplied))
        requested_backend = selected.agent.backend
        resume, replay = selected.run.resume, selected.run.replay
        invocation_port = selected.run.port
        self.root = Path(root).resolve()
        occupied = self.root.exists() and any(path.name not in {'experiment.json', 'runtime.json'} for path in self.root.iterdir())
        if occupied and not (resume or replay):
            raise ValueError('run_directory_already_used: choose a fresh --run-dir or --resume')
        if (resume or replay) and not occupied:
            raise ValueError('resume_requires_existing_run')
        manifest = json.loads((self.root / 'manifest.json').read_text()) if (resume or replay) else {}
        if resume or replay:
            if manifest.get('effective_config') is not None:
                archived = dict(manifest['effective_config'])
                archived_features = dict(archived.get('features', {}))
                archived_features.setdefault('default_task_notes', False)
                archived_features.setdefault('vision_compare', True)
                archived_features.setdefault('actor_payload_profile', 'full_v1')
                archived_features.setdefault('completion_guidance', manifest.get('completion_guidance', {}).get('mode', 'none'))
                archived_features.setdefault('airborne_handoff', True)
                archived_features.setdefault('joint_replan', manifest.get('joint_replan', {}).get('enabled', False))
                for flag in ('bounded_continuation', 'motion_safety_guard', 'trajectory_preview', 'manipulation_protocol'):
                    archived_features.setdefault(flag, False)
                for flag in ('task_scheduling', 'operation_candidates', 'preview_intent', 'recovery_evidence', 'robotic_view', 'stereo_guide', 'self_briefing'):
                    archived_features.setdefault(flag, False)
                archived_features.setdefault('action_trajectory_delivery', 'automatic')
                archived['features'] = archived_features
                selected = Config.from_dict(archived)
            else:
                # Translate the actual historical fields without consulting a new
                # config file for features absent from old experiments.
                saved = {'run': {key: manifest[key] for key in ('task', 'seed', 'layout_id', 'max_actions', 'benchmark', 'task_notes', 'task_notes_source') if key in manifest},
                         'features': dict(observation_profile=manifest.get('observation_profile') if manifest.get('observation_profile') in ACTOR_PROFILES else LEGACY_ACTOR_PROFILE,
                            motion_review=manifest.get('motion_review', False),
                            vision_compare=manifest.get('vision_compare', True),
                            default_task_notes=manifest.get('task_notes_config', {}).get('default_enabled', False),
                            actor_payload_profile=manifest.get('actor_payload_profile', 'full_v1'),
                            enable_wait=manifest.get('enable_wait', False),
                            tracking_guard=manifest.get('tracking_guard', {}).get('enabled', False),
                            joint_replan=manifest.get('joint_replan', {}).get('enabled', False),
                            airborne_handoff=True,
                            bounded_continuation=False, motion_safety_guard=False,
                            trajectory_preview=False, manipulation_protocol=False,
                            task_scheduling=False, operation_candidates=False,
                            preview_intent=False, recovery_evidence=False,
                            robotic_view=False, stereo_guide=False, self_briefing=False,
                            completion_guidance=manifest.get('completion_guidance', {}).get('mode', 'none'),
                            pi05=manifest.get('pi05_enabled', False), visual_review_model=manifest.get('visual_review_model'))}
                if manifest.get('grasp_memory_dir'):
                    saved['features']['grasp_memory_dir'] = manifest['grasp_memory_dir']
                if 'task_notes_config' in manifest:
                    saved['run']['task_notes'] = manifest['task_notes_config'].get('custom_text')
                    saved['run']['task_notes_source'] = manifest['task_notes_config'].get('custom_source')
                saved['run']['wall_budget_s'] = manifest.get('actor_wait_budget_s')
                saved['model'] = {key: manifest[old] for key, old in (('name', 'model'), ('effort', 'effort')) if manifest.get(old)}
                if manifest.get('image_settings'):
                    saved['images'] = dict(manifest['image_settings'])
                    saved['images'].setdefault('reuse_exact_images', False)
                # Host paths can be selected on legacy runs that never recorded
                # deployment settings. Their experiment semantics stay frozen.
                saved.update(backend=selected.backend.to_dict(), policy=selected.policy.to_dict())
                saved['model']['codex_home'] = selected.model.codex_home
                selected = Config.from_dict(saved)
        selected = selected.with_overrides({'run': dict(run_dir=str(self.root), resume=resume, replay=replay, port=invocation_port)})
        if not has_calibration(selected.features.observation_profile):
            selected = selected.with_overrides({'features': {'motion_review': False, 'robotic_view': False, 'stereo_guide': False}})
        configured_specs = _configured_tool_specs(selected, tool_specs, self)
        unknown_tools = set(selected.features.disabled_tools) - create_registry(configured_specs).names - CONFIG_TOOL_NAMES
        if unknown_tools:
            raise ValueError('unknown_disabled_tools:' + ','.join(sorted(unknown_tools)))
        if not replay and "task_plan" in selected.features.disabled_tools:
            raise ValueError("required_task_plan_tool_cannot_be_disabled")
        if not replay:
            # Refuse a backend/feature combination before any run state, robot or native process exists.
            backend_requirements(selected)
        self.config = selected
        run, features = selected.run, selected.features
        task, gpu, seed, layout_id, max_actions = run.task, run.gpu, run.seed, run.layout_id, run.max_actions
        observation_profile = features.observation_profile
        actor_payload_profile, enable_wait = features.actor_payload_profile, features.enable_wait
        tracking_guard, joint_replan, motion_review = features.tracking_guard, features.joint_replan, features.motion_review
        pi05, visual_review_model, completion_guidance = features.pi05, features.visual_review_model, features.completion_guidance
        if not replay:
            actor_policy(observation_profile)
        if price_card is None and selected.model.price_card and not (resume or replay):
            price_card = load_price_card(selected.model.price_card)
        default_task_notes, vision_compare = features.default_task_notes, features.vision_compare
        self.default_task_notes = default_task_notes
        self.archived_task_notes = ({
            "text": manifest.get("task_notes"), "source": manifest.get("task_notes_source"),
            "config": manifest.get("task_notes_config", {
                "default_enabled": False, "default_version": None,
                "custom_text": manifest.get("task_notes"), "custom_source": manifest.get("task_notes_source"),
            }),
        } if resume or replay else None)
        self.archived_public_task_rules = manifest.get("public_task_rules") if resume or replay else None
        self.archived_instruction_override = manifest.get("task_instruction_override") if resume or replay else None
        self.completion_guidance = completion_guidance
        self.initial_observation_id = None
        self.actor_payload_profile, self.enable_wait, self.tracking_guard = actor_payload_profile, enable_wait, tracking_guard
        self.joint_replan = joint_replan
        self.vision_compare = vision_compare
        self.observation_profile = observation_profile
        self.motion_review = motion_review
        if resume and (manifest.get("observation_profile") not in ACTOR_PROFILES or
                       manifest.get("actor_information_policy") != actor_policy(observation_profile)):
            raise ValueError("resume_requires_clean_actor_history: create a fresh run and thread")
        if resume:
            previous_card = manifest.get("price_card")
            if price_card is not None and price_card != previous_card:
                raise ValueError("resume_price_card_conflict: use offline report for a different schedule")
            price_card = previous_card
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_lock = (self.root/"gateway.lock").open("a")
        try: fcntl.flock(self.run_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("run_gateway_already_owned")
        self.resume, self.replay = resume, replay
        self.identity = {key:manifest[key] for key in IDENTITY} if (resume or replay) else dict(run_id=self.root.name, episode_id=new_id("ep"), env_instance_id=new_id("env"), env_generation=1)
        self.store = Store(self.root, self.identity)
        self.identity.update(self.store.read_state("runtime", "identity", self.identity))
        # Identity of the agent session that recorded this run (runtime/agent; runtime/codex for older archives).
        self.previous_agent = read_agent_identity(self.store) if resume else None
        recorded_backend = (self.previous_agent or {}).get("backend")
        # Both the backend this invocation asked for and the archived configuration must match the recorded session.
        if recorded_backend is not None and {requested_backend, self.config.agent.backend} != {recorded_backend}:
            self.store.close()
            self.run_lock.close()
            raise ValueError(f"resume_agent_backend_changed: the run was recorded with agent backend {recorded_backend}; "
                             f"resume with --agent-backend {recorded_backend} or start a fresh run")
        if resume:
            worker = self.store.read_state("runtime", "robot") or {}
            pid = worker.get("pid")
            if pid and Path(f"/proc/{pid}/cmdline").exists():
                command = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
                if "auriga.robot.backends.robodojo.robot_worker" in command and str(self.root/"private"/"robot") in command:
                    raise ValueError("resume_requires_previous_owned_robot_worker_exit")
            previous = dict(self.identity)
            self.identity.update(episode_id=new_id("ep"), env_generation=self.identity["env_generation"]+1)
            self.store.set_state("runtime", "identity", dict(self.identity))
            self.store.event("environment.recovery_reset", {"previous_identity":previous,"physics_restore":False})
        self.task, self.seed, self.layout_id = task, seed, layout_id
        self.pi05_enabled, self.gpu = pi05, gpu
        self.visual_review_model = visual_review_model
        # Snapshot every local implementation module, including optional adapters,
        # so adding a module cannot silently omit it from run provenance.
        source_paths = sorted(Path(__file__).parents[2].rglob("*.py"))
        source_paths += [Path(__file__).parents[2] / "cortex/context/robodojo_task_rules.json",
                         Path(__file__).parents[2] / "cortex/context/task_instruction_overrides.json",
                         Path(__file__).parents[2] / "cortex/session/codex_responses.json"]
        self.source_hashes = {str(path.relative_to(Path(__file__).parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in source_paths}
        self.source_ref = None
        if not replay:
            sources = {name: (Path(__file__).parents[2] / name).read_text() for name in self.source_hashes}
            self.source_ref = self.store.put_artifact(canonical(sources).encode(), "runtime_source", "application/json", "observer")
        self.robot = robot if robot is not None else create_backend(task=task, gpu_id=gpu, seed=seed, layout_id=layout_id, run_dir=self.root / "private" / "robot", observation_profile=observation_profile, tracking_guard=tracking_guard, joint_replan=joint_replan, config=self.config)
        actual_images = store_image_settings(self.store, self.config.images.to_dict())
        self.config = self.config.with_overrides({"images": actual_images})
        self.grasp_memory_store = None
        if not replay:
            memory_path = Path(self.config.features.grasp_memory_dir).expanduser().resolve()
            self.grasp_memory_store = Store(memory_path, self.identity)
        self.harness = Harness(self.store, self.robot, max_actions, actor_payload_profile=actor_payload_profile, enable_wait=enable_wait,
                               observation_profile=observation_profile, motion_review=self.motion_review,
                               grasp_memory_store=self.grasp_memory_store, vision_compare=vision_compare, config=self.config,
                               task_plan_required=True, tool_specs=configured_specs)
        if resume: self.harness.accepting=True  # Fresh environment generation; old unknown receipts remain immutable.
        self.session = create_session(self.config, self.root, self.store, self.harness.handle_tool,
                                      price_card=price_card, task_planning=self.harness.task_planning)
        codex_backend = self._backend_kind() == "codex"
        self.harness.delivery_channel = "codex_tool" if codex_backend else "agent_tool"
        # Codex re-injects the decision context into its own compaction request; every other backend gets it from
        # the Harness on the next actor tool response after the session reports a context reset (design §4.5).
        self.harness.decision_reminder_channel = (REMINDER_IN_COMPACT_REQUEST if codex_backend
                                                  else REMINDER_IN_TOOL_RESULT)
        if not (resume or replay):
            (self.root / 'effective_config.json').write_text(json.dumps(self.config.to_dict(), indent=2) + '\n')
        self.session.on_disconnect=self.harness.request_stop
        self.session.on_context_reset = self.harness.model_context_reset
        self.session.transport_recovery_gate = self.transport_recovery_block
        self.session.tool_parallel_safe = self.harness.tool_registry.parallel_safe
        # Published ToolSpec.parallel per actor tool (True | False | "request_dependent"), for MCP readOnlyHint.
        self.session.tool_parallel_hints = {tool["name"]: tool["parallel"]
                                            for tool in self.harness.tool_registry.manifest(self.harness)["tools"]}
        self.terminal = None
        self.status = "starting"
        self.starting = True
        self.control_responses = {}
        self.control_lock = asyncio.Lock()
        self.scene_service = None
        self.scene_forwarder = None
        self.controller = None
        self.loop = None
        self.closed = False

    async def start(self, prompt, benchmark=False, task_notes=None, task_notes_source=None, wall_budget_s=None):
        if self.resume and hasattr(self, 'config'):
            benchmark = self.config.run.benchmark
            task_notes, task_notes_source = self.config.run.task_notes, self.config.run.task_notes_source
            wall_budget_s = self.config.run.wall_budget_s
        if not self.replay and hasattr(self, 'config'):
            self.config = self.config.with_overrides({'run': dict(prompt=None if benchmark else prompt, benchmark=benchmark,
                task_notes=task_notes, task_notes_source=task_notes_source, wall_budget_s=wall_budget_s)})
            (self.root / 'effective_config.json').write_text(json.dumps(self.config.to_dict(), indent=2) + '\n')
        if self.replay:
            self.starting=False
            self.status="replay"
            self.harness.accepting=False
            self.scene_service=SceneService(self.root/"twin-service", image_settings=getattr(self.harness, "image_settings", self.config.images.to_dict()), **{key: value for key, value in getattr(self, "config", Config()).scene.to_dict().items() if key not in {"root", "port"}})
            self.harness.scene=self.scene_service.scene
            observations=self.store.records("observation")
            self.harness.latest=observations[-1] if observations else None
            self.harness.capabilities=self.store.read_state("runtime","capabilities") or self.store.read("configuration","robot_capabilities")
            info=read_agent_identity(self.store) or {}
            enter_replay=getattr(self.session, "enter_replay", None)
            if callable(enter_replay):
                enter_replay(info.get("thread_id"))
            else:
                self.session.thread_id=info.get("thread_id")
                self.session.state="replay"
            self.terminal=Terminal(self.root,self.root)
            saved_terminal=json.loads((self.root/"terminal.json").read_text()) if (self.root/"terminal.json").exists() else {}
            self.terminal.session_id=saved_terminal.get("session_id",self.terminal.session_id)
            self.terminal.thread_id=self.session.thread_id
            self.terminal._closed=True
            return
        notes = self.archived_task_notes if self.resume else resolve_task_notes(
            default_enabled=benchmark and self.default_task_notes,
            custom_text=task_notes, custom_source=task_notes_source,
            manipulation_protocol=self.config.features.manipulation_protocol,
            task_scheduling=self.config.features.task_scheduling, operation_candidates=self.config.features.operation_candidates)
        from ...cortex.context.actor_instructions import task_notes as current_task_notes
        if not self.resume:
            notes = current_task_notes(notes)
        task_notes, task_notes_source = notes["text"], notes["source"]
        handoff = handoff_policy(self.config.features.airborne_handoff,
                                 handoff_preference=self.config.features.handoff_preference)
        rules = (self.archived_public_task_rules if self.resume
                 else public_task_rules(self.task) if benchmark else None)
        instruction_override = (self.archived_instruction_override if self.resume
                                else task_instruction_override(self.task, self.config.features.task_instruction_overrides)
                                if benchmark else None)
        self.loop = asyncio.get_running_loop()
        self_briefing_at = self_briefing_task_text = None
        self.self_briefing = None
        try:
            if isinstance(self.harness, Harness):
                # Scene reconstruction is not implemented for actor tasks. Keep the
                # local geometry store, but do not start an unused scene service.
                self.harness.programs = ProgramRunner(self.harness.program_services())
            if not self.harness.accepting:
                return
            await asyncio.to_thread(self.harness.start)
            self.initial_observation_id = (getattr(self.harness, "latest", None) or {}).get("observation_id")
            if self.pi05_enabled:
                from ...runtime.policy.action_policy import OpenPIProcessPolicy
                self.harness.action_policy = await asyncio.to_thread(OpenPIProcessPolicy,
                    Path(self.config.policy.python),
                    Path(self.config.policy.checkpoint),
                    gpu=self.gpu, log_path=self.root / "private/pi05.log", timeout=120,
                    server_url=self.config.policy.server_url)
            if benchmark:
                instruction = self.harness.latest.get("task_instruction")
                if not instruction: raise ValueError("native_task_instruction_missing")
                instruction, rules_text = apply_task_instruction_override(instruction, rules, instruction_override)
                prompt = (f"Complete the following native RoboDojo task: {instruction}\n"
                          "Use permitted RGB, measured state and tools; inspect actual feedback. "
                          "The environment ends execution automatically on its terminal condition. "
                          "Do not read simulator files or evaluator state.")
                if rules_text:
                    prompt += "\n" + rules_text
                if getattr(self.config.features, "self_briefing", False) and not self.resume:
                    # Notes, when generated, are inserted here: after the task text and public rules.
                    self_briefing_at = len(prompt)
                    self_briefing_task_text = instruction + ("\n" + rules_text if rules_text else "")
                prompt += "\nCurrent native execution budget: " + canonical(self.harness.action_budget())
            initial_guidance = (INITIAL_COMPLETION_GUIDANCE
                                if benchmark and not self.resume and self.completion_guidance in {"initial", "both"}
                                else None)
            if initial_guidance:
                prompt += f"\nPublic completion guidance: {initial_guidance}"
            if task_notes:
                prompt += "\nTask notes:\n" + task_notes
            home_check = getattr(self.harness, "home_check", None)
            state_aware_hook = self.completion_guidance in {"hook", "both"}
            guidance_record = {
                "mode": self.completion_guidance, "source": COMPLETION_GUIDANCE_SOURCE,
                "initial_trigger": "initial_actor_prompt" if initial_guidance else None,
                "initial_prompt": initial_guidance,
                "initial_observation_id": self.initial_observation_id,
                "continuation_trigger": COMPLETION_GUIDANCE_TRIGGER if benchmark else None,
                "continuation_condition": "actor_turn_completed_before_native_terminal" if benchmark else None,
                "continuation_policy": HOME_CHECK_POLICY if benchmark and state_aware_hook else "generic",
                "home_reference": home_check.reference_record() if home_check is not None else None,
                # The selected prompt depends on the current measured state;
                # actual decisions and delivery attempts are recorded per turn.
                "continuation_prompt": BENCHMARK_CONTINUATION if benchmark and not state_aware_hook else None,
                "continuation_prompts": {
                    "away": hook_completion_guidance(self.initial_observation_id),
                    "home": BENCHMARK_CONTINUATION, "unknown": BENCHMARK_CONTINUATION,
                } if benchmark and state_aware_hook else None}
            if benchmark and self.config.features.bounded_continuation:
                from ...cortex.supervision import recovery_guidance as recovery_module
                recovery_version = (recovery_module.EVIDENCE_VERSION if self.config.features.recovery_evidence
                                    else recovery_module.VERSION)
                recovery_instructions = (recovery_module.EVIDENCE_INSTRUCTIONS if self.config.features.recovery_evidence
                                         else recovery_module.INSTRUCTIONS)
                guidance_record["recovery_guidance"] = dict(version=recovery_version,
                    instructions=recovery_instructions, context="current actor observation and native budget; measured home status")
            prompt_record = {"prompt": prompt, "native_instruction": (getattr(self.harness, "latest", None) or {}).get("task_instruction"), "benchmark": benchmark,
                             "public_task_rules": rules,
                             "task_instruction_override": instruction_override,
                             "task_planning": {"version": PLANNING_VERSION, "required_before_robot_actions": True,
                                               "tool_choices": "advisory", "revisions": "actor_discretion"},
                             "task_notes": task_notes, "task_notes_source": task_notes_source,
                             "task_notes_config": notes["config"],
                             "handoff_policy": handoff,
                             "completion_guidance": guidance_record}
            self.store.save("configuration", "task_prompt-" + digest(prompt_record), prompt_record)
            if not self.harness.accepting:
                return
            if getattr(self.robot, "process", None):
                self.store.set_state("runtime", "robot", {"pid":self.robot.process.pid,"run_dir":str(self.root/"private"/"robot")})
            # Published Harness source for every backend (design R4.4): mounted at /harness by the sandbox, named by
            # the instructions below otherwise. CodexSession._prepare refreshes the same copy.
            from ...cortex.session.actor_source import snapshot_actor_source
            (self.root / "private").mkdir(mode=0o700, exist_ok=True)
            snapshot_actor_source(self.root / "private" / "actor-source")
            from ...cortex.context.actor_instructions import instructions as actor_instructions
            observation_contract = (DEPTH_INSTRUCTIONS if allows_depth(self.observation_profile) else
                CALIBRATED_OBSERVATION_INSTRUCTIONS if has_calibration(self.observation_profile)
                else LEGACY_OBSERVATION_INSTRUCTIONS)
            instructions = actor_instructions(observation_contract, handoff["text"],
                source_root=None if self.config.features.sandbox_enabled else self.root / "private" / "actor-source",
                operation_memory="operation_memory" not in self.config.features.disabled_tools)
            if self.visual_review_model and not {"vision_review", "operation_verify"}.intersection(self.config.features.disabled_tools):
                from ...cortex.context.operation_verification import INSTRUCTIONS as verification_instructions
                instructions += "\n" + verification_instructions.replace("operation_verify", "vision_review(evidence_mode='operation')")
            from ...perception.external.obs3d import developer_instructions as obs3d_instructions
            instructions += obs3d_instructions(self.config.features)
            if getattr(self.config.features, "pose_skill", False):
                from ...perception.external.pose_skill import INSTRUCTIONS as pose_skill_instructions
                instructions += pose_skill_instructions
            if getattr(self.config.features, "held_perception", False):  # features.held_perception: one sentence
                from ...perception.external.held_perception import INSTRUCTION as held_perception_instruction
                instructions += "\n" + held_perception_instruction
            if self.config.images.feedback_policy == "adaptive":
                instructions += "\nAutomatic RGB may omit unchanged views; this does not prove unchanged object state. Read a needed saved view with history_view_observation."
            # Automatic initialization uses the same capability delivery path as a read.
            if isinstance(self.harness, Harness):
                capability_result = self.harness.invoke("robot_capabilities", {"detail": "summary"},
                    {"tool_call_id": "initialize-capabilities"}, role="actor")
                capability_result = self.harness.actor_output.present(capability_result)
                instructions += "\nRobot runtime capabilities\n" + capability_result["contentItems"][0]["text"]
                self.store.event("robot_capabilities.initialized", {"delivered": True})
            # Read the same robot-only API exposed to the actor. This costs no
            # actor turn, advances no simulation time and opens no task assets.
            if callable(getattr(self.harness, "robot_information", None)):
                robot_information = self.harness.robot_information()
            else:
                from ...robot.model.robot_information import robot_information as read_robot_information
                robot_information = read_robot_information(
                    (self.harness.capabilities or {}).get("robot_model_id") or "unknown")
            self.store.set_state("runtime", "robot_information", robot_information)
            self.store.event("robot_information.initialized", dict(
                robot_id=robot_information.get("robot_id"), available=robot_information.get("available"),
                section="default", order=["robot_information", "harness_instructions", "task", "task_notes"]))
            if isinstance(self.harness, Harness):
                robot_information = self.harness.actor_output.present(robot_information)
            from ...robot.model.robot_information import robot_information_text
            geometry_guidance = ("The original URDF is supplied above. Read its STL/GLB meshes from the native ZIP with client.artifact using resource_ref when you need surface geometry. "
                                 if "urdf" in robot_information else "")
            instructions = (robot_information_text(robot_information)
                + "\n\nHarness instructions\n" + instructions
                + "\n" + geometry_guidance
                + "Calibration, when available, is anchored to its observation, not a substitute for current measured motion. robot_information can repeat these inputs, calibration rereads the latest published camera/base calibration, and section=all returns the detailed physical reference. "
                  "geometry_image is an optional illustrative drawing, not live visual evidence.")
            # A recorded identity without a thread (empty native record) starts a fresh thread, as before.
            if self.resume and (self.previous_agent or {}).get("thread_id"):
                self.harness.thread_id = await asyncio.to_thread(self.session.start, self.actor_tools(), instructions, self.previous_agent["thread_id"])
            else:
                self.harness.thread_id = await asyncio.to_thread(self.session.start, self.actor_tools(), instructions)
            self.harness.model_proxy = getattr(self.session, "model_proxy", None)
            # Forced full delivery (design §4.3) for a backend that cannot yet prove what its model still holds;
            # read after start() so a session may settle its delivery mode while starting. Codex: "delta".
            self.harness.context_epoch_volatile = getattr(self.session, "context_delivery", "delta") == "full"
            if isinstance(self.harness, Harness):
                # The capabilities already delivered in startup belong to the
                # newly created model context, not the pre-start empty identity.
                self.harness._capability_delivery.project(self.harness.capabilities,
                    epoch=self.harness._model_context_epoch(), current_action_budget=self.harness.action_budget())
            if not self.harness.accepting:
                return
            if self.visual_review_model:
                from ...cortex.session.visual_review import VisualReviewer
                options = {}
                if self._evidence_client() == EVIDENCE_CLIENT_RELAY:
                    # agent.evidence.upstream (when set) becomes the relay's "evidence" binding (design §9.4).
                    from ...cortex.agents.evidence import bind_evidence_upstream
                    upstream = bind_evidence_upstream(self.session, self.config)
                    if upstream != "actor":
                        options["upstream"] = upstream
                self.harness.visual_reviewer = VisualReviewer(self._evidence_agent(**options))
            if self_briefing_at is not None:
                # features.self_briefing: one fresh helper turn of the actor model before the first actor turn.
                from ...cortex.context import self_briefing as briefing
                agent = self._evidence_agent(model=self.session.model,
                    base_instructions=briefing.BASE_INSTRUCTIONS,
                    developer_instructions=briefing.DEVELOPER_INSTRUCTIONS, source=briefing.SOURCE)
                record = await asyncio.to_thread(briefing.generate, agent, self.store,
                    getattr(self.harness, "latest", None), self_briefing_task_text)
                if record["status"] == "applied":
                    prompt = prompt[:self_briefing_at] + briefing.section(record["notes"]) + prompt[self_briefing_at:]
                record["actor_prompt"] = prompt
                self.self_briefing = record
                self.store.save("configuration", "self_briefing-" + digest(record), record)
                if not self.harness.accepting:
                    return
            # An empty thread has no rollout; start the task once before attaching TUI.
            if not self.resume:
                await asyncio.to_thread(self.session.run_turn, prompt)
            if not self.harness.accepting and not getattr(self.harness, "episode_terminal", None):
                return
            attach = self._terminal_attach_spec() if getattr(self, 'config', Config()).features.terminal else None
            if attach is not None:
                self.terminal = Terminal(self.root, attach["cwd"], command_prefix=attach["command_prefix"],
                                         env=attach["env"])
                await self.terminal.start(attach["attach_uri"], attach["thread_id"])
            if not self.harness.accepting and not getattr(self.harness, "episode_terminal", None):
                return
            self.status = "running"
            codex_backend = self._backend_kind() == "codex"
            manifest = dict(schema_version="0.1", **self.identity, data_origin="robodojo_runtime",
                            actor_information_policy=actor_policy(self.observation_profile),
                            motion_review=self.motion_review, vision_compare=self.vision_compare,
                            task_planning=dict(version=PLANNING_VERSION, required_before_robot_actions=True,
                                               tool_choices="advisory", revisions="actor_discretion"),
                            grasp_memory_dir=str(self.grasp_memory_store.root) if self.grasp_memory_store else None,
                            observation_profile=(getattr(self.harness,"latest",None) or {}).get("observation_profile", "synthetic_fixture"), clock_policy="paused_between_gateway_operations",
                            model=self.session.model, provider=self.session.provider, effort=self.session.effort,
                            price_card=self.session.price_card, codex_sdk=CODEX_SDK_VERSION if codex_backend else None,
                            codex_cli=CODEX_CLI_VERSION if codex_backend else None,
                            codex_cli_executable=str(self.session.cli_executable) if codex_backend else None,
                            capabilities=self.harness.capabilities,
                            image_settings=self.store.read_state("runtime", "image_settings"),
                            model_image_policy=self.store.read_state("runtime", "model_image_policy"),
                            actor_payload_profile=self.actor_payload_profile, enable_wait=self.enable_wait,
                            tracking_guard={"enabled": self.tracking_guard, "threshold_rad": .5,
                                            "consecutive_native_steps": 5, "scope": "active_arm_ee_motion"},
                            joint_replan={"enabled": self.joint_replan, "scope": "backend_free_ee",
                                          "candidate": "nearest_limit_valid_periodic_endpoint",
                                          "selection": "native_success_and_shorter_resampled_path",
                                          "minimum_remaining_wall_s": 15},
                            task=self.task, seed=self.seed, layout_id=self.layout_id,
                            max_actions=self.harness.max_actions if isinstance(self.harness, Harness) else None,
                            actor_wait_budget_s=wall_budget_s,
                            runtime_protocol="robodojo-native-v1", model_request_limit=180,
                            model_request_count_basis="delivered usable completed actor, visual-helper and compaction responses; failed/incomplete attempts excluded",
                            model_attempt_max_retries=DEFAULT_MAX_RETRIES,
                            model_request_timeout_s=None, model_auto_retries=DEFAULT_MAX_RETRIES, model_retry_policy=RETRY_POLICY,
                            no_progress_timeout_s=INACTIVITY_TIMEOUT_S, startup_timeout_s=900,
                            progress_clock="last_model_code_robot_or_simulator_activity",
                            read_only_queries_between_actions=None,
                            pi05_enabled=self.pi05_enabled, visual_review_model=self.visual_review_model,
                            source_sha256=self.source_hashes, benchmark=benchmark,
                            benchmark_terminal_policy="native_terminal_and_task_limits_v3" if benchmark else None,
                            actor_stall_policy=ACTOR_STALL_POLICY if benchmark else None,
                            actor_progress_policy=(ACTOR_EVIDENCE_POLICY if self.config.features.recovery_evidence else ACTOR_PROGRESS_POLICY)
                                if benchmark and self.config.features.bounded_continuation else None,
                            scoring_policy=SCORING_POLICY,
                            completion_guidance=guidance_record,
                            source_ref=self.source_ref,
                            task_notes=task_notes, task_notes_source=task_notes_source,
                            task_notes_config=notes["config"],
                            handoff_policy=handoff,
                            public_task_rules=rules,
                            task_instruction_override=instruction_override,
                            thread_id=self.session.thread_id,
                            app_server_pid=self.session.process.pid if codex_backend else None,
                            terminal_session_id=self.terminal.session_id if self.terminal else None,
                            terminal_pid=self.terminal.process.pid if self.terminal else None,
                            effective_config=getattr(self, "config", Config()).to_dict(),
                            tool_catalog=self.harness.tool_registry.manifest(self.harness),
                            local_scene=self.harness.scene.capabilities())
            if not codex_backend:
                # Codex-only provenance keys are absent for other backends; agent_backend carries theirs.
                for key in ("codex_sdk", "codex_cli", "codex_cli_executable", "app_server_pid"):
                    manifest.pop(key)
            manifest["agent_backend"] = self._agent_provenance()
            if self.visual_review_model or (benchmark and self.config.features.self_briefing):
                # Effective evidence helper client (design §9.6); absent without an evidence feature, so
                # default Codex manifests keep their key set.
                manifest["evidence_client"] = self._evidence_client()
            if self.self_briefing is not None:
                manifest["self_briefing"] = self.self_briefing
            self.store.set_state("runtime", "identity", dict(self.identity))
            (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))
            self.store.event("run.started", {"manifest_ref": "manifest.json"}, {"thread_id": self.session.thread_id})
        finally:
            try:
                if not self.harness.accepting and not getattr(self.harness, "episode_terminal", None):
                    self.status = "unknown"
                    try:
                        await asyncio.to_thread(self.robot.close)
                    finally:
                        try:
                            if self.terminal:
                                await self.terminal.close()
                        finally:
                            await asyncio.to_thread(self.session.close)
                    unresolved = self.current_unknown_actions()
                    self.status = "unknown" if self.harness.active_action or unresolved else "stopped"
                    self.store.event("run." + self.status, {"reason": "stop_during_startup"})
            finally:
                self.starting = False

    # ---- agent session protocol, read tolerantly -------------------------------------------------------------
    # Sessions implement auriga.cortex.agents.base.AgentSession. Duck-typed, Codex-shaped sessions (test fakes,
    # pre-protocol objects) lack the protocol members; they are served by the Codex implementation of the member,
    # i.e. exactly the attribute probing the gateway used before the protocol existed.

    def _backend_kind(self):
        """The session's backend kind; duck-typed sessions without ``backend_kind`` are Codex-shaped."""
        return session_backend_kind(self.session)

    def _session_capability(self, name):
        """``session.capabilities.<name>``; duck-typed sessions without capabilities get the Codex value."""
        return bool(getattr(session_capabilities(self.session, CodexSession.capabilities), name, False))

    def _terminal_attach_spec(self):
        """Live native TUI attachment, or None when the backend has no terminal (skipped, not an error)."""
        if not self._session_capability("terminal_attach"):
            return None
        spec = getattr(self.session, "terminal_attach_spec", None)
        return spec() if callable(spec) else CodexSession.terminal_attach_spec(self.session)

    def _owned_session_processes(self):
        owned = getattr(self.session, "owned_processes", None)
        return list(owned()) if callable(owned) else CodexSession.owned_processes(self.session)

    def _evidence_client(self):
        """Effective evidence helper client of this run: "codex_agent" (Codex default) or "relay"."""
        return evidence_client(self.config, self._backend_kind())

    def _evidence_agent(self, **options):
        """Evidence helper for visual review / self-briefing from ``session.evidence_agent(**options)``.

        Codex: ``CodexEvidenceAgent`` unless ``agent.evidence.client="relay"``; other backends: ``RelayEvidenceClient``.
        Duck-typed, Codex-shaped sessions without the protocol member get the Codex selection for this run's config.
        """
        factory = getattr(self.session, "evidence_agent", None)
        if callable(factory):
            return factory(**options)
        if self._evidence_client() == EVIDENCE_CLIENT_RELAY:
            from ...cortex.agents.evidence import RelayEvidenceClient
            return RelayEvidenceClient(self.session, **options)
        from ...cortex.session.codex_evidence_agent import CodexEvidenceAgent
        return CodexEvidenceAgent(self.session, **options)

    def _agent_provenance(self):
        provenance = getattr(self.session, "provenance", None)
        return provenance() if callable(provenance) else CodexSession.provenance(self.session)

    def actor_tools(self):
        if hasattr(self, 'harness'):
            return self.harness.tool_registry.catalog(self.harness, role='actor')
        from types import SimpleNamespace
        config = getattr(self, 'config', Config())
        runtime = SimpleNamespace(config=config, disabled_tools=config.features.disabled_tools,
            observation_profile=getattr(self, 'observation_profile', config.features.observation_profile),
            motion_review=getattr(self, 'motion_review', config.features.motion_review),
            enable_wait=getattr(self, 'enable_wait', False),
            vision_compare=getattr(self, 'vision_compare', config.features.vision_compare),
            visual_reviewer=getattr(self, 'visual_review_model', None),
            action_policy=True if getattr(self, 'pi05_enabled', False) else None,
            capabilities=None, programs=True)
        return create_registry(runtime=runtime).catalog(runtime, role='actor')

    def drain_scene_events(self):
        if not self.scene_service: return
        cursor=self.store.read_state("runtime","scene_cursor",0)
        end = self.scene_service.store.event_cursor()
        while cursor < end:
            page = self.scene_service.event_page(cursor)
            if not page["events"]:
                break
            for event in page["events"]:
                self.store.ingest_event(event)
            cursor = page["next_cursor"]
            self.store.set_state("runtime", "scene_cursor", cursor)

    async def forward_scene_events(self):
        while True:
            self.drain_scene_events()
            await asyncio.sleep(.05)

    def submit_scene_thread(self, request, correlation):
        async def submit():
            job = self.scene_service.submit(request, "build_scene", identifier="job-"+digest(correlation)[:32])
            self.store.event("scene.job_linked", {"job_id":job["job_id"]}, correlation, world_kind="internal")
            return job
        return asyncio.run_coroutine_threadsafe(submit(),self.loop).result(15)

    def snapshot(self):
        if self.controller and self.controller["expires_at"] < time.time(): self.controller=None
        return dict(**self.identity, status=self.status, thread_id=self.session.thread_id, turn_id=self.session.turn_id,
                    codex_status=self.session.state, agent_backend=self._backend_kind(), agent_status=self.session.state,
                    action_count=self.harness.action_count, accepting_actions=self.harness.accepting,
                    active_action=self.harness.active_action, motion_status=self.harness.states.get(self.harness.active_action,{}).get("status","idle"),
                    controller={"owner_client_id":self.controller["client_id"] if self.controller else None,"expires_at":self.controller["expires_at"] if self.controller else None,"can_claim":self.controller is None},
                    pending_requests=[r for r in self.store.states("native_request") if r["status"]=="pending"],
                    token_accounting=self.store.read_state("runtime", "token_accounting"),
                    terminal=self.terminal.status() if self.terminal else None, capabilities=self.harness.capabilities)

    def authorize_writer(self, value, origin):
        self.snapshot()
        if not self.controller:
            if origin: raise ValueError("input_owner_claim_required")
            return
        if value.get("client_id") != self.controller["client_id"] or not secrets.compare_digest(str(value.get("writer_token","")),self.controller["token"]):
            raise ValueError("input_owner_conflict")
        self.controller["expires_at"]=time.time()+30

    async def execute_control(self, value, origin=None):
        command, identifier = value.get("command"), value["request_id"]
        if self.replay: raise ValueError("replay_is_read_only")
        if command in ("claim_input","renew_input","release_input"):
            client_id=value.get("client_id")
            if not isinstance(client_id,str) or not 0<len(client_id)<=128: raise ValueError("invalid_client_id")
            self.snapshot()
            if command=="claim_input":
                if self.controller and self.controller["client_id"]!=client_id: raise ValueError("input_owner_conflict")
                if self.controller: self.authorize_writer(value,origin)
                if not self.controller: self.controller={"client_id":client_id,"token":secrets.token_urlsafe(32),"expires_at":time.time()+30}
            else:
                self.authorize_writer(value,origin)
                if command=="release_input": self.controller=None
            return {"request_id":identifier,"status":"released" if not self.controller else "owned","writer_token":self.controller["token"] if self.controller else None,"expires_at":self.controller["expires_at"] if self.controller else None}
        if command=="stop": return await self.stop(identifier)
        self.authorize_writer(value,origin)
        if command=="reply":
            result=await asyncio.to_thread(self.session.reply,value["native_request_id"],value["response"])
            return {**result,"control_request_id":identifier}
        if command=="reset":
            if self.starting or self.session.state=="running" or self.harness.active_action: raise ValueError("reset_requires_idle_turn_and_motion")
            if self.harness.programs:
                programs = self.harness.programs.snapshot()
                if programs['active_jobs'] or programs['unresolved_jobs']:
                    raise ValueError("reset_requires_program_stop")
            if self.scene_service:
                for job in self.scene_service.jobs():
                    if job["status"] not in JOB_TERMINAL: self.scene_service.transition(job,job["status"],cancel_requested=True,cancel_reason="environment_reset")
                deadline=time.monotonic()+10
                while self.scene_service.processes and time.monotonic()<deadline: await asyncio.sleep(.05)
                if self.scene_service.processes: raise ValueError("scene_workers_stop_unconfirmed")
            observation=await asyncio.to_thread(self.harness.reset,value.get("seed",0),value.get("layout_id",0))
            if self.scene_service: self.scene_service.closing=False
            self.status="running" if self.harness.accepting else "stopped"
            return {"request_id":identifier,"status":"reset","observation_id":observation["observation_id"],**self.identity}
        if command=="steer":
            # Mid-turn operator input: joins the running turn; stop/hold remain the way to halt motion.
            text=value.get("prompt")
            if not isinstance(text,str) or not text.strip() or len(text)>16000: raise ValueError("invalid_task_prompt")
            if not self._session_capability("steer"): raise AgentCapabilityError("steer_unsupported_by_backend")
            if self.starting or self.session.state!="running": raise ValueError("no_active_turn_to_steer")
            turn_id=await asyncio.to_thread(self.session.steer,text)
            return dict(request_id=identifier,thread_id=self.session.thread_id,turn_id=turn_id,status="steered")
        if command=="prompt":
            if self.starting or not self.harness.accepting or self.session.state=="running" or self.harness.active_action: raise ValueError("run_not_ready_for_new_task")
            prompt=value.get("prompt")
            if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>16000: raise ValueError("invalid_task_prompt")
            turn_id=await asyncio.to_thread(self.session.run_turn,prompt)
            return dict(request_id=identifier,thread_id=self.session.thread_id,turn_id=turn_id,status="running")
        raise ValueError("unsupported_control_command")

    def current_unknown_actions(self):
        from ...core.presentation.feedback_delivery import execution_unconfirmed
        programs = getattr(self.harness, 'programs', None)
        if programs and programs.unresolved_jobs():
            return True
        reset=self.store.read_state("runtime","reset")
        if reset and reset["status"]=="unknown" and hasattr(self,"identity") and all(reset.get(key)==self.identity[key] for key in IDENTITY): return True
        for identifier,state in self.harness.states.items():
            if not execution_unconfirmed(state): continue
            if not hasattr(self,"identity"): return True
            try: request=self.store.read("action_request",identifier)
            except KeyError: return True
            if all(request.get(key)==self.identity[key] for key in IDENTITY): return True
        return False

    def transport_recovery_block(self):
        """A model resend is safe only while the paused world has no pending action."""
        harness = self.harness
        if self.closed or not harness.accepting:
            return "harness_not_accepting"
        if harness.episode_done.is_set():
            return "native_episode_ended"
        if harness.active_action:
            return "action_in_progress"
        if self.current_unknown_actions():
            return "action_status_unknown"
        if self.has_active_programs():
            return "program_active"
        return None

    def has_active_programs(self):
        programs = getattr(self.harness, "programs", None)
        return bool(programs and programs.snapshot()['active_jobs'])

    def program_continuation_state(self):
        """Only an acknowledged review pause permits another benchmark turn."""
        programs = getattr(self.harness, "programs", None)
        return programs.continuation_state() if programs else dict(blocked=False, pending_reviews=[])

    async def abort_startup(self, starting):
        """Unblock the original startup owner without leaving late-created children."""
        signalled = {}
        while not starting.done():
            processes = (("robot", getattr(self.robot, "process", None)),
                         *Workbench._owned_session_processes(self),
                         ("terminal", getattr(self.terminal, "process", None)))
            for role, process in processes:
                if process is None or process.poll() is not None:
                    continue
                previous = signalled.get(process)
                signal_name = "SIGTERM" if previous is None else "SIGKILL"
                if previous is not None and (previous[2] == "SIGKILL" or time.monotonic() - previous[1] < 5):
                    continue
                try:
                    process.terminate() if previous is None else process.kill()
                except ProcessLookupError:
                    continue
                signalled[process] = (role, time.monotonic(), signal_name)
                if role == "robot":
                    self.robot.startup_forced_termination = signal_name
                self.store.event("run.startup_process_signalled", dict(role=role, pid=process.pid, signal=signal_name))
            await asyncio.sleep(.05)
        for process, (role, _, signal_name) in signalled.items():
            self.store.event("run.startup_process_exit", dict(role=role, pid=process.pid,
                signal=signal_name, process_exit_confirmed=process.poll() is not None, returncode=process.returncode))

    async def stop(self, request_id):
        if request_id in self.control_responses:
            return self.control_responses[request_id]
        self.status = "stop_requested"
        self.harness.request_stop()  # Close motion admission before cancelling any owned program.
        if getattr(self.harness,"programs",None):
            self.harness.programs.cancel_all()
        if getattr(self,"scene_service",None):
            await self.scene_service.stop_jobs()
            self.drain_scene_events()
        if self.starting:
            response = dict(request_id=request_id, status="stop_requested", reason="waiting_for_startup_to_release_owned_resources")
            self.control_responses[request_id] = response
            self.store.event("run.stop_requested", response)
            return response
        try:
            await asyncio.wait_for(asyncio.to_thread(self.session.interrupt), timeout=45)
            deadline = asyncio.get_running_loop().time() + 45
            while (self.harness.active_action or self.session.state == "running" or self.has_active_programs()) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
            unresolved = self.current_unknown_actions()
            self.status = "stopped" if self.harness.active_action is None and self.session.state != "running" and not self.has_active_programs() and not unresolved else "unknown"
        except Exception as error:
            self.status = "unknown"
            self.store.event("run.stop_unknown", {"reason": str(error)})
            deadline = asyncio.get_running_loop().time() + 45
            while (self.harness.active_action or self.has_active_programs()) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(.05)
        if (getattr(self.harness, "episode_terminal", None) and self.harness.active_action is None
                and not self.has_active_programs() and not self.current_unknown_actions()):
            # Native completion is already latched and motion admission closed.
            # An actor interrupt failure must not discard the robot's result.
            if self.session.state == "running":
                try:
                    await asyncio.wait_for(asyncio.to_thread(self.session.close), timeout=45)
                except Exception as error:
                    self.store.event("codex.close_after_native_terminal_failed", {"error_type":type(error).__name__})
            self.status = "stopped"
        response = dict(request_id=request_id, status=self.status)
        self.control_responses[request_id] = response
        self.store.event("run." + self.status, response)
        return response

    async def close(self):
        if self.closed: return getattr(self, "cleanup_report", None)
        self.closed=True
        report = self.cleanup_report = {"steps": {}, "errors": [], "robot": None}

        async def cleanup(name, operation):
            try:
                result = await operation()
                report["steps"][name] = "completed"
                return result
            except Exception as error:
                report["steps"][name] = "failed"
                report["errors"].append({"step": name, "error": f"{type(error).__name__}: {error}"})

        await cleanup("motion_admission", lambda: asyncio.to_thread(self.harness.request_stop))
        if self.session.state == "running" or self.harness.active_action or self.has_active_programs():
            await cleanup("stop", lambda: self.stop(new_id("shutdown")))
        # Release the worker before session.close waits for tool-handler threads.
        # Every cleanup step is independent: a broken PTY/policy/model must not
        # bypass the owned simulator's native close and bounded process fallback.
        report["robot"] = await cleanup("robot", lambda: asyncio.to_thread(self.robot.close))
        if self.terminal:
            await cleanup("terminal", self.terminal.close)
        if getattr(self.harness, "action_policy", None) is not None:
            await cleanup("policy", lambda: asyncio.to_thread(self.harness.action_policy.close))
        await cleanup("session", lambda: asyncio.to_thread(self.session.close))
        if getattr(self.harness,"programs",None):
            await cleanup("programs", lambda: asyncio.to_thread(self.harness.programs.close))
        if self.scene_service:
            await cleanup("scene_jobs", self.scene_service.stop_jobs)
            await cleanup("scene_events", lambda: asyncio.to_thread(self.drain_scene_events))
        if self.scene_forwarder:
            self.scene_forwarder.cancel()
            await asyncio.gather(self.scene_forwarder,return_exceptions=True)
        if self.scene_service:
            await cleanup("scene_service", self.scene_service.close)
        if getattr(self, "grasp_memory_store", None) is not None:
            await cleanup("grasp_memory_store", lambda: asyncio.to_thread(self.grasp_memory_store.close))
        await cleanup("store", lambda: asyncio.to_thread(self.store.close))
        await cleanup("run_lock", lambda: asyncio.to_thread(self.run_lock.close))
        return report

    def require_observer_idle(self):
        if (self.replay or self.starting or self.harness.active_action or (self.session.state == "running" and not getattr(self.harness, "episode_terminal", None))
                or self.current_unknown_actions() or self.has_active_programs()):
            raise ValueError("observer_operation_requires_idle_run")

    def observer_evaluate(self):
        with self.harness.robot_lock:
            self.require_observer_idle()
            return record_evaluation(self.store, self.identity, self.robot.evaluate(final_check=True))

    def observer_video(self):
        with self.harness.robot_lock:
            self.require_observer_idle()
            return record_video(self.store, self.identity, self.robot.save_video())
