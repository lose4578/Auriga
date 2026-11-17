"""Validated, immutable operator configuration shared by launchers and services.

Resolution: built-in defaults < legacy environment < public/local JSON < explicit overrides.
Snapshots use from_dict(), never load(), so archived behavior cannot drift with the host.
Credentials and fixed benchmark/security protocol limits are deliberately not settings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
import json
import math
import os
import re
from urllib.parse import urlsplit
from pathlib import Path
import types
from typing import Union, get_args, get_origin, get_type_hints


LOCAL_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'configs/auriga.local.json'


class ConfigError(ValueError):
    pass


def option(default=None, *, choices=None, minimum=None, maximum=None, path=False):
    return field(default=default, metadata=dict(choices=choices, minimum=minimum, maximum=maximum, path=path))


def _type_ok(value, annotation):
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return any(_type_ok(value, item) for item in get_args(annotation))
    if origin is tuple:
        return isinstance(value, tuple) and all(_type_ok(item, get_args(annotation)[0]) for item in value)
    if annotation is float:
        return type(value) in (int, float) and math.isfinite(value)
    if annotation in (int, bool, str):
        return type(value) is annotation
    if annotation is type(None):
        return value is None
    return isinstance(value, annotation)


class Section:
    def __post_init__(self):
        hints = get_type_hints(type(self))
        for item in fields(self):
            value = getattr(self, item.name)
            key = type(self).__name__ + '.' + item.name
            if not _type_ok(value, hints[item.name]):
                raise ConfigError(f'invalid_config_type:{key}')
            if value is None:
                continue
            metadata = item.metadata
            if metadata.get('choices') is not None and value not in metadata['choices']:
                raise ConfigError(f'invalid_config_choice:{key}: {value!r}')
            if metadata.get('minimum') is not None and value < metadata['minimum']:
                raise ConfigError(f'config_below_minimum:{key}')
            if metadata.get('maximum') is not None and value > metadata['maximum']:
                raise ConfigError(f'config_above_maximum:{key}')
            if isinstance(value, str) and (not value.strip() or '\x00' in value):
                raise ConfigError(f'invalid_config_string:{key}')
            if isinstance(value, tuple) and any(not item.strip() for item in value):
                raise ConfigError(f'invalid_config_list:{key}')

    def to_dict(self):
        # JSON-compatible arrays rather than Python tuples in exported snapshots.
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True)
class RunConfig(Section):
    run_dir: str | None = option(path=True)
    experiment_root: str = option('~/exp/auriga', path=True)
    category: str = 'main'
    title: str | None = None
    tag: tuple[str, ...] = ()
    replay: bool = False
    resume: bool = False
    task: str = 'stack_bowls'
    seed: int = option(0, minimum=0)
    layout_id: int = option(0, minimum=0)
    gpu: int = option(4, minimum=0)
    port: int = option(0, minimum=0, maximum=65535)
    max_actions: int | None = option(minimum=1)
    wall_budget_s: float | None = option(minimum=0.001)
    prompt: str | None = None
    exit_after_turn: bool = False
    benchmark: bool = False
    task_notes: str | None = None
    task_notes_source: str | None = None


@dataclass(frozen=True)
class FeaturesConfig(Section):
    sandbox_enabled: bool = False
    observation_profile: str = option('rgb_proprio_calibrated_v1', choices=('rgb_proprio_v1', 'rgb_proprio_calibrated_v1', 'rgbd_proprio_calibrated_v1'))
    depth_skills: bool = False
    depth_auto_feedback: bool = False
    depth_event_payload: bool = False
    actor_payload_profile: str = option('lean_v1', choices=('full_v1', 'compact_v1', 'lean_v1', 'lean_v2'))
    tool_catalog_profile: str = option('full_v1', choices=('full_v1', 'layered_v1'))
    default_task_notes: bool = True
    # Per-task corrections of the actor-facing task text (auriga/task_instruction_overrides.json).
    task_instruction_overrides: bool = True
    # One task-independent pre-action call to the actor model that writes short
    # operating notes appended to the initial task text (auriga/cortex/context/self_briefing.py).
    self_briefing: bool = True
    vision_compare: bool = True
    robotic_view: bool = False
    stereo_guide: bool = False
    enable_wait: bool = True
    motion_review: bool = True
    tracking_guard: bool = False
    joint_replan: bool = True
    motion_execution_mode: str = option('resampled', choices=('resampled', 'feedback'))
    default_motion_speed_scale: float = option(1.0, minimum=.25, maximum=2.0)
    tight_position_tolerance_m: float = option(0.003, minimum=0.001, maximum=0.05)
    default_position_tolerance_m: float = option(0.010, minimum=0.001, maximum=0.05)
    soft_position_tolerance_m: float = option(0.010, minimum=0.001, maximum=0.05)
    relaxed_position_tolerance_m: float = option(0.015, minimum=0.001, maximum=0.05)
    compliant_position_tolerance_m: float = option(0.015, minimum=0.001, maximum=0.05)
    coarse_position_tolerance_m: float = option(0.025, minimum=0.001, maximum=0.05)
    max_position_tolerance_m: float = option(0.10, minimum=0.005, maximum=0.2)
    tight_orientation_tolerance_rad: float = option(0.05, minimum=0.005, maximum=3.14159265)
    orientation_tolerance_rad: float = option(0.15, minimum=0.005, maximum=3.14159265)
    relaxed_orientation_tolerance_rad: float = option(0.20, minimum=0.005, maximum=3.14159265)
    coarse_orientation_tolerance_rad: float = option(0.35, minimum=0.005, maximum=3.14159265)
    max_orientation_tolerance_rad: float = option(3.14159265, minimum=0.05, maximum=3.14159265)
    contact_divergence_threshold_m: float = option(0.004, minimum=0.0005, maximum=0.05)
    stall_guard_strict_divergence_m: float = option(0.010, minimum=0.002, maximum=0.15)
    stall_guard_contact_stop_divergence_m: float = option(0.020, minimum=0.002, maximum=0.15)
    stall_guard_compliant_divergence_m: float = option(0.020, minimum=0.002, maximum=0.15)
    stall_guard_min_divergence_m: float = option(0.005, minimum=0.001, maximum=0.05)
    stall_guard_max_divergence_m: float = option(0.15, minimum=0.01, maximum=0.30)
    stall_guard_step_velocity_threshold_m: float = option(0.0015, minimum=0.0001, maximum=0.02)
    stall_guard_consecutive_steps: int = option(2, minimum=1, maximum=20)
    max_gripper_aperture_m: float = option(0.088, minimum=0.01, maximum=0.20)
    grip_empty_threshold_mm: float = option(0.5, minimum=0.0, maximum=40.0)
    grip_thin_threshold_mm: float = option(1.5, minimum=0.0, maximum=40.0)
    grip_holding_threshold_mm: float = option(3.0, minimum=0.1, maximum=40.0)
    grip_solid_threshold_mm: float = option(3.5, minimum=0.1, maximum=40.0)
    grip_non_empty_min_aperture_m: float = option(0.0015, minimum=0.0, maximum=0.088)
    grip_thin_min_aperture_m: float = option(0.0005, minimum=0.0, maximum=0.088)
    grip_thin_max_aperture_m: float = option(0.065, minimum=0.001, maximum=0.088)
    grip_rigid_min_aperture_m: float = option(0.002, minimum=0.0, maximum=0.088)
    grip_rigid_max_aperture_m: float = option(0.087, minimum=0.005, maximum=0.088)
    grip_wide_min_aperture_m: float = option(0.003, minimum=0.0, maximum=0.088)
    grip_wide_max_aperture_m: float = option(0.0875, minimum=0.01, maximum=0.088)
    max_min_grip_m: float = option(0.088, minimum=0.001, maximum=0.20)
    closed_jaw_goal_max: float = option(0.05, minimum=0.0, maximum=1.0)
    checkpoint_empty_jaw_measured_max: float = option(0.03, minimum=0.0, maximum=1.0)
    joint_travel_standard_max_rad: float = option(math.pi, minimum=0.5, maximum=20.0)
    joint_travel_standard_backtracking_rad: float = option(math.pi / 2, minimum=0.2, maximum=20.0)
    joint_travel_extended_max_rad: float = option(1.5 * math.pi, minimum=0.5, maximum=20.0)
    joint_travel_extended_backtracking_rad: float = option(math.pi, minimum=0.2, maximum=20.0)
    segment_start_joint_drift_rad: float = option(0.10, minimum=0.001, maximum=3.14159265)
    resume_max_joint_drift_rad: float = option(0.10, minimum=0.0001, maximum=1.0)
    resume_max_ee_drift_m: float = option(0.015, minimum=0.0001, maximum=0.1)
    resume_max_orientation_drift_rad: float = option(0.10, minimum=0.0001, maximum=1.0)
    resume_max_jaw_drift_normalized: float = option(0.10, minimum=0.0001, maximum=1.0)
    batch_plan_cache_size: int = option(16, minimum=1, maximum=256)
    hotspot_xy_radius_m: float = option(0.025, minimum=0.005, maximum=0.20)
    hotspot_z_radius_m: float = option(0.045, minimum=0.005, maximum=0.20)
    ledger_max_recent_outcomes: int = option(3, minimum=1, maximum=20)
    ledger_max_hotspots: int = option(6, minimum=1, maximum=32)
    target_exemption_radius_m: float = option(0.055, minimum=0.0, maximum=0.15)
    clearance_height_m: float = option(0.13, minimum=0.0, maximum=0.25)
    arm_queue: bool = True
    motion_cost_preview: bool = True
    airborne_handoff: bool = True
    handoff_preference: str = option('surface', choices=('surface', 'airborne'))
    bounded_continuation: bool = True
    motion_safety_guard: bool = True
    trajectory_preview: bool = True
    manipulation_protocol: bool = True
    task_scheduling: bool = True
    operation_candidates: bool = False
    preview_intent: bool = False
    recovery_evidence: bool = True
    action_trajectory_delivery: str = option('automatic', choices=('automatic', 'diagnostic'))
    pi05: bool = False
    visual_review_model: str | None = option(None, choices=('gpt-6.1-sol',))
    completion_guidance: str = option('hook', choices=('none', 'initial', 'hook', 'both'))
    disabled_tools: tuple[str, ...] = ()
    # Cache-preserving upstream request history projection (all off by default;
    # baseline behavior unchanged). Enabling lean_projection replaces closed-block
    # non-anchor images with placeholders and condenses high-volume historical tool
    # replies into deterministic summaries. NOTE: Compressing raw historical tool
    # outputs and images discards fine-grained historical trajectory samples, full
    # program stdout, and visual details unless the model explicitly re-reads them
    # via history_view_observation, which may affect long-horizon spatial reasoning
    # or multi-step recovery performance.
    lean_projection: bool = False
    lean_projection_k: int = option(1, minimum=1, maximum=8)
    # Block mode (default): no trailing message; tool steps in the open block of
    # B steps and the first step of every block are sent unchanged, older steps
    # as summaries/placeholders, rewritten once per B steps. 0 = trailing mode
    # (the upstream cache does not reuse a request after its trailing message
    # is dropped, see auriga/core/presentation/lean_projection.py).
    lean_projection_block: int = option(8, minimum=0, maximum=64)
    # Lossless compact encoding of model-facing tool replies (auriga/core/presentation/lean_compact.py).
    lean_compact: bool = False
    # Diagnostics only: write the first N upstream request bodies (image data
    # replaced by digests) to run_dir/private/request-dump/. 0 disables.
    lean_request_dump: int = option(0, minimum=0, maximum=500)
    # external_perception (on by default): one-click master switch for external
    # vision models (VGGT-Omega + SAM 3.1). When True, enables the external 3D
    # perception stack (vggt_height in optional mode, object_table, scene3d,
    # object_table_observe, scene3d_observe, held_perception, and 3D
    # preview/obstacle/ledger checks) unless individual sub-switches are
    # explicitly overridden. When False, preserves native standalone behavior
    # with zero external model calls.
    external_perception: bool = True
    vggt_height: bool = True
    vggt_height_mode: str = option('optional', choices=('optional', 'default'))
    vggt_endpoint: str = 'http://127.0.0.1:8331'
    vggt_support_z_m: float = option(0.765, minimum=0.0, maximum=3.0)
    vggt_timeout_s: float = option(60.0, minimum=1.0, maximum=300.0)
    object_table: bool = True
    object_table_max_objects: int = option(20, minimum=1, maximum=40)
    object_table_max_height_m: float = option(0.30, minimum=0.05, maximum=1.0)
    object_table_head_exclude_px: str = '200,405,440,480'
    scene3d: bool = True
    scene3d_voxel_m: float = option(0.01, minimum=0.005, maximum=0.03)
    scene3d_z_above_m: float = option(0.60, minimum=0.05, maximum=2.0)
    scene3d_z_below_m: float = option(0.05, minimum=0.0, maximum=1.0)
    object_table_observe: bool = True
    scene3d_observe: bool = True
    obs3d_timeout_s: float = option(60.0, minimum=1.0, maximum=120.0)
    obs3d_max_objects: int = option(10, minimum=1, maximum=40)
    obs3d_action_overlay: bool = False
    pose_skill: bool = False
    # held_perception (on by default with external_perception): object tables (geometry_query kind object_table and the
    # obs3d attached table) exclude robot pixels with an FK render of the robot meshes (dilated 3 px) for held objects,
    # and an arm whose jaws are closed on something gets a held-object row: extents and long axis from the object's last
    # table row before the grasp carried with link6, centre from the FK grasp centre (moved along the long axis by the
    # carried offset for elongated objects); without a pre-grasp row, extents from the VGGT points of the SAM mask
    # rescaled to the FK grasp depth; never the floating -> table rescale. Needs the service variant object-table-v5-hp1
    # (robot_mask_b64, held, POST /held; services/vggt_height) at vggt_endpoint. Adds one developer-instruction sentence.
    # See auriga/perception/external/held_perception.py and docs/held_perception.md.
    held_perception: bool = True
    grasp_memory_dir: str = option('~/.local/share/auriga/grasp-memory', path=True)
    record_video: bool = True
    export_video: bool = True
    terminal: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.external_perception and not any(getattr(self, flag) for flag in _EXTERNAL_PERCEPTION_SWITCHES):
            for flag in _EXTERNAL_PERCEPTION_SWITCHES:
                object.__setattr__(self, flag, True)
        elif not self.external_perception and all(getattr(self, flag) for flag in _EXTERNAL_PERCEPTION_SWITCHES):
            for flag in _EXTERNAL_PERCEPTION_SWITCHES:
                object.__setattr__(self, flag, False)


@dataclass(frozen=True)
class AnalysisConfig(Section):
    scene_truth_log: bool = False


@dataclass(frozen=True)
class TransportRecoveryConfig(Section):
    # After the pinned provider-error retries are exhausted, resend the same
    # request after a cooldown instead of ending the episode. max_rounds=0 disables.
    max_rounds: int = option(10, minimum=0, maximum=20)
    initial_delay_s: float = option(60, minimum=1)
    backoff_multiplier: float = option(2, minimum=1)
    max_delay_s: float = option(600, minimum=1)
    max_episode_delay_s: float = option(3600, minimum=0)
    transport_incomplete: bool = True


@dataclass(frozen=True)
class ModelConfig(Section):
    name: str = 'gpt-6-astra'
    effort: str = option('high', choices=('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'))
    codex_home: str = option('~/.codex', path=True)
    price_card: str | None = option(path=True)
    native_memories: bool = True
    transport_recovery: TransportRecoveryConfig = field(default_factory=TransportRecoveryConfig)


@dataclass(frozen=True)
class BackendConfig(Section):
    environment: str = 'robodojo'
    robot_model_id: str = 'arx-x5'
    root: str = option('~/code/robodojo', path=True)
    python: str = option('~/miniforge3/envs/RoboDojo/bin/python', path=True)


@dataclass(frozen=True)
class PolicyConfig(Section):
    python: str = option('~/.local/share/auriga-openpi-arx/bin/python', path=True)
    checkpoint: str = option('/home/ckpt/robodojo_pi05_official/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999', path=True)
    server_url: str | None = None


@dataclass(frozen=True)
class ImagesConfig(Section):
    feedback_policy: str = option('all', choices=('all', 'adaptive'))
    transport_format: str = option('webp', choices=('webp', 'jpeg'))
    webp_quality: int = option(85, minimum=1, maximum=100)
    webp_method: int = option(4, minimum=0, maximum=6)
    jpeg_quality: int = option(95, minimum=1, maximum=100)
    storage_jpeg_quality: int = option(95, minimum=1, maximum=100)
    reuse_exact_images: bool = True
    # Model proxy: a /responses body over 4.5 MB keeps the opening and recent
    # images and replaces the middle ones with observation_id placeholders, once
    # per compaction cycle; an image_compression_failed retry trims further.
    # False forwards the native history unchanged.
    trim_oversized_requests: bool = True
    # Same trim when inline images exceed this count (provider limit is 64); 0 disables.
    trim_max_inline_images: int = option(60, minimum=0, maximum=1000)


@dataclass(frozen=True)
class LibraryConfig(Section):
    root: str = option('~/exp/auriga', path=True)
    port: int = option(8780, minimum=0, maximum=65535)
    scan_interval: float = option(1, minimum=0.1, maximum=60)
    public_origin: str | None = None
    task_source_root: str | None = option(path=True)


@dataclass(frozen=True)
class OomRetryConfig(Section):
    max_retries: int = option(2, minimum=0, maximum=20)
    initial_delay_s: float = option(60, minimum=1)
    backoff_multiplier: float = option(2, minimum=1)
    max_delay_s: float = option(300, minimum=1)


@dataclass(frozen=True)
class SceneConfig(Section):
    root: str = option('runs/local_scene', path=True)
    port: int = option(8770, minimum=0, maximum=65535)
    max_workers: int = option(2, minimum=1)
    max_queue: int = option(4, minimum=1)
    max_blob_store_bytes: int = option(536870912, minimum=1)


@dataclass(frozen=True)
class QueueConfig(Section):
    max_parallel: int | None = option(60, minimum=1)
    memory_per_case_mib: int = option(11264, minimum=1)
    gpu_headroom_mib: int = option(0, minimum=0)
    wait_timeout_s: float = option(1800, minimum=0.001)
    oom_retry: OomRetryConfig = field(default_factory=OomRetryConfig)
    wake_enabled: bool = True
    wake_thread: str | None = None
    wake_codex: str = 'codex'
    wake_remote: str | None = None


ENV_NAME_PATTERN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
ANTHROPIC_BASE_URL = 'https://api.anthropic.com'


@dataclass(frozen=True)
class UpstreamConfig(Section):
    """Where an agent backend's model requests are forwarded by the local accounting relay.

    Real credentials never live in the config: only the NAME of the environment variable that holds the key is stored,
    and the value is read by the gateway process at launch. The agent process itself only sees a local placeholder.
    """
    # Credential source. "explicit" = use wire_api/base_url/api_key_env below; "codex_home" = reuse the Codex credential
    # source (model.codex_home config.toml + auth.json) on the OpenAI Responses wire; "claude_settings" = read the
    # Anthropic Messages endpoint and key from a Claude Code settings file (`settings` below), claude_code only.
    source: str = option('explicit', choices=('explicit', 'codex_home', 'claude_settings'))
    # Claude Code settings.json in the official format, used when source is "claude_settings": `env.ANTHROPIC_BASE_URL`
    # (default https://api.anthropic.com) and `env.ANTHROPIC_AUTH_TOKEN` (sent as Authorization: Bearer) or
    # `env.ANTHROPIC_API_KEY` (sent as x-api-key). Read by the gateway only; the key is never copied into the run
    # archive or the agent. A relative path is resolved next to the config file that sets it. null = not set.
    settings: str | None = option(None, path=True)
    # Wire protocol of the upstream endpoint; null = the backend's own default (claude_code: anthropic_messages).
    wire_api: str | None = option(None, choices=('anthropic_messages', 'openai_responses', 'openai_chat'))
    # http(s) base URL of the upstream API, without credentials, query or fragment (e.g. https://api.anthropic.com).
    # null = not set (a backend that needs it rejects the launch).
    base_url: str | None = None
    # NAME (not value) of the environment variable holding the API key, e.g. ANTHROPIC_API_KEY. null = not set.
    api_key_env: str | None = None
    # Header carrying the key: "bearer" = Authorization: Bearer <key>; "x-api-key" = x-api-key: <key>.
    auth_scheme: str = option('bearer', choices=('bearer', 'x-api-key'))
    # Free label of the serving provider; keys price cards and resume identity, never sent upstream. null = unlabelled.
    provider: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.base_url is not None:
            try:
                parts = urlsplit(self.base_url)
                parts.port  # Raises ValueError for a malformed port.
            except ValueError as error:
                raise ConfigError('invalid_upstream_base_url') from error
            if (parts.scheme not in ('http', 'https') or not parts.hostname or parts.username is not None
                    or parts.password is not None or parts.query or parts.fragment or '?' in self.base_url
                    or '#' in self.base_url):
                raise ConfigError('invalid_upstream_base_url: http(s) URL without credentials, query or fragment')
        if self.api_key_env is not None and not ENV_NAME_PATTERN.match(self.api_key_env):
            raise ConfigError('invalid_upstream_api_key_env: expected an environment variable NAME, never the key')
        if self.source == 'claude_settings' and not self.settings:
            raise ConfigError('upstream_claude_settings_requires_settings: set upstream.settings to a Claude Code '
                              'settings.json path')


def _anthropic_upstream():
    return UpstreamConfig(wire_api='anthropic_messages', base_url=ANTHROPIC_BASE_URL, api_key_env='ANTHROPIC_API_KEY',
                          auth_scheme='x-api-key', provider='anthropic')


@dataclass(frozen=True)
class ClaudeCodeConfig(Section):
    """Settings of the Claude Code agent backend (agent.backend = "claude_code"; optional extra auriga-harness[claude])."""
    # Absolute path of the Claude Code CLI binary; null = the CLI bundled with the pinned claude-agent-sdk.
    cli_path: str | None = option(path=True)
    # Upstream model endpoint. Defaults: anthropic_messages wire, https://api.anthropic.com, ANTHROPIC_API_KEY, x-api-key, "anthropic".
    upstream: UpstreamConfig = field(default_factory=_anthropic_upstream)
    # How task context reaches the model each turn: "full" = complete state every turn; "delta" = only what changed
    # since the previous delivery.
    context_delivery: str = option('full', choices=('full', 'delta'))
    # true (default) = the native subagent tool (Agent/Task) stays available; false = the actor works alone.
    subagents: bool = True
    # Token window (tokens, 100000..1000000) at which native auto-compaction triggers; null = the CLI default.
    auto_compact_window_tokens: int | None = option(None, minimum=100000, maximum=1000000)


@dataclass(frozen=True)
class OpenCodeConfig(Section):
    """Settings of the OpenCode agent backend (agent.backend = "opencode"; binary installed outside the Python env)."""
    # Absolute path of the platform binary of the pinned opencode-ai release (not the node wrapper); null = not set.
    executable: str | None = option(path=True)
    # Upstream model endpoint; wire_api, base_url and api_key_env must be supplied (or source "codex_home").
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    # Model context window in tokens (model limit.context); drives native auto-compaction.
    context_limit_tokens: int = option(200000, minimum=1)
    # Maximum tokens generated per model call (model limit.output).
    output_limit_tokens: int = option(32000, minimum=1)
    # How task context reaches the model each turn: "full" or "delta" (as claude_code.context_delivery).
    context_delivery: str = option('full', choices=('full', 'delta'))
    # true (default) = the native subagent (task) tool is available; false = the actor works alone.
    subagents: bool = True


EVIDENCE_UPSTREAM_WIRES = ('openai_responses', 'anthropic_messages')


@dataclass(frozen=True)
class EvidenceConfig(Section):
    """Client of the evidence helpers (features.visual_review_model and features.self_briefing; design 04 section 9).

    Omitted from snapshots while unchanged, so Codex snapshots and frozen-checkout projections stay byte-identical.
    """
    # Which client answers visual review and self-briefing. "native" (default) = the backend's own evidence helper where
    # it has one (Codex: CodexEvidenceAgent, a fresh ephemeral Codex app-server thread); backends without one always use
    # the relay client. "relay" = RelayEvidenceClient on every backend: one fresh, streamed, tool-free model request per
    # helper call through the run's own model relay (same counted 180-call budget, retries, recovery gate and egress
    # scrub). Arms of a harness comparison that enable an evidence feature must all set "relay".
    client: str = option('native', choices=('native', 'relay'))
    # Upstream the relay client sends features.visual_review_model requests to, bound on the run's relay as the
    # "evidence" upstream (its own wire and credentials; the shared budget). Needed when the actor's upstream cannot
    # serve the review model, e.g. {"source": "codex_home"} for Sol review next to an anthropic_messages actor.
    # null (default) = the actor's own upstream. Self-briefing always uses the actor's model and upstream. Only the
    # relay client uses it: with client "native" on Codex it is refused. wire_api must be openai_responses or
    # anthropic_messages; "explicit" needs wire_api and base_url, "codex_home" implies openai_responses.
    upstream: UpstreamConfig | None = None

    def __post_init__(self):
        super().__post_init__()
        upstream = self.upstream
        if upstream is None:
            return
        if upstream.wire_api == 'openai_chat':
            raise ConfigError('evidence_upstream_wire_not_implemented:openai_chat: choose openai_responses or '
                              'anthropic_messages for agent.evidence.upstream.wire_api')
        if upstream.source == 'codex_home':
            if upstream.wire_api not in (None, 'openai_responses'):
                raise ConfigError('evidence_upstream_codex_home_requires_openai_responses')
        elif upstream.wire_api not in EVIDENCE_UPSTREAM_WIRES or upstream.base_url is None:
            raise ConfigError('evidence_upstream_requires_wire_api_and_base_url: set agent.evidence.upstream.wire_api '
                              'and base_url, or source "codex_home"')


@dataclass(frozen=True)
class AgentConfig(Section):
    """Which agent runtime drives the actor. Default codex; the section is omitted from snapshots while unchanged.

    model.name, model.effort, model.price_card and model.transport_recovery apply to every backend; model.codex_home
    and model.native_memories are Codex-only.
    """
    # Agent runtime: "codex" (native Codex app-server), "claude_code" or "opencode". CLI: --agent-backend.
    backend: str = option('codex', choices=('codex', 'claude_code', 'opencode'))
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    opencode: OpenCodeConfig = field(default_factory=OpenCodeConfig)
    # Evidence helper client (visual review, self-briefing); see EvidenceConfig.
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.backend == 'codex' and self.evidence.client == 'native' and self.evidence.upstream is not None:
            # CodexEvidenceAgent always uses the actor's provider; an evidence upstream would be silently ignored.
            raise ConfigError('evidence_upstream_requires_relay_client: set agent.evidence.client="relay" or remove '
                              'agent.evidence.upstream (the native Codex evidence agent uses the actor provider)')

    def to_dict(self):
        # The default evidence section is omitted so agent snapshots of earlier checkouts stay identical.
        value = super().to_dict()
        if self.evidence == EvidenceConfig():
            value.pop('evidence', None)
        return value


def _section_type(annotation):
    """The Section dataclass of a field annotated ``S`` or ``S | None``; None for any other annotation."""
    if is_dataclass(annotation):
        return annotation
    if get_origin(annotation) in (Union, types.UnionType):
        options = [item for item in get_args(annotation) if item is not type(None)]
        if len(options) == 1 and is_dataclass(options[0]):
            return options[0]
    return None


def _construct(cls, value, prefix='config', base=None):
    if not isinstance(value, dict):
        raise ConfigError(f'config_object_required:{prefix}')
    known = {item.name: item for item in fields(cls)}
    unknown = set(value) - known.keys()
    if unknown:
        raise ConfigError(f'unknown_config_keys:{prefix}: {", ".join(sorted(unknown))}')
    hints = get_type_hints(cls)
    values = {}
    for key, item in known.items():
        annotation = hints[key]
        if key not in value:
            continue
        part = value[key]
        section = _section_type(annotation)
        if section is not None and (part is not None or is_dataclass(annotation)):
            # An optional section (``S | None``) stays None when null; a required one must be an object.
            part = _construct(section, part, prefix + '.' + key, base)
        elif get_origin(annotation) is tuple and isinstance(part, list):
            part = tuple(part)
        if item.metadata.get('path') and isinstance(part, str) and part.strip():
            path = Path(part).expanduser()
            part = str(((base or Path.cwd()) / path).resolve())
        values[key] = part
    return cls(**values)


def _merge(base, overrides):
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


_EXTERNAL_PERCEPTION_SWITCHES = (
    'vggt_height', 'object_table', 'scene3d', 'object_table_observe', 'scene3d_observe', 'held_perception'
)
EXTERNAL_PERCEPTION_FIELDS = (
    'external_perception', 'vggt_height', 'vggt_height_mode', 'vggt_endpoint',
    'vggt_support_z_m', 'vggt_timeout_s', 'object_table', 'object_table_max_objects',
    'object_table_max_height_m', 'object_table_head_exclude_px', 'scene3d',
    'scene3d_voxel_m', 'scene3d_z_above_m', 'scene3d_z_below_m',
    'object_table_observe', 'scene3d_observe', 'obs3d_timeout_s',
    'obs3d_max_objects', 'obs3d_action_overlay', 'pose_skill', 'held_perception',
)


def _handoff_compat(value):
    """An explicit legacy bool or master external_perception switch keeps its strategy across config layers."""
    features = value.get('features')
    if isinstance(features, dict):
        updated = dict(features)
        if type(updated.get('airborne_handoff')) is bool and 'handoff_preference' not in updated:
            updated['handoff_preference'] = 'airborne' if updated['airborne_handoff'] else 'surface'
        if type(updated.get('external_perception')) is bool:
            master = updated['external_perception']
            for flag in _EXTERNAL_PERCEPTION_SWITCHES:
                updated.setdefault(flag, master)
            if not master:
                updated.setdefault('pose_skill', False)
            if 'vggt_height_mode' not in updated:
                updated['vggt_height_mode'] = 'optional'
        if updated != features:
            value = {**value, 'features': updated}
    return value


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ConfigError(f'cannot_read_config:{path}: {error}') from error
    if not isinstance(value, dict):
        raise ConfigError(f'config_object_required:{path}')
    return value


def _legacy_environment(environ):
    result = {}
    aliases = {
        'CODEX_HOME': ('model', 'codex_home'), 'AURIGA_PRICE_CARD': ('model', 'price_card'),
        'AURIGA_ROBODOJO_ROOT': ('backend', 'root'), 'AURIGA_ROBODOJO_PYTHON': ('backend', 'python'),
        'AURIGA_PI05_PYTHON': ('policy', 'python'), 'AURIGA_PI05_CHECKPOINT': ('policy', 'checkpoint'),
        'AURIGA_PI05_SERVER_URL': ('policy', 'server_url'),
        'AURIGA_GRASP_MEMORY_DIR': ('features', 'grasp_memory_dir'),
    }
    # Keep the old standalone image file as a compatibility input. An explicit
    # unified images section has higher priority. Packaged installs need no file.
    image_path = Path(environ.get('AURIGA_IMAGE_CONFIG', Path(__file__).resolve().parents[2] / 'configs/images.json')).expanduser()
    if environ.get('AURIGA_IMAGE_SETTINGS'):
        try:
            result['images'] = json.loads(environ['AURIGA_IMAGE_SETTINGS'])
        except ValueError as error:
            raise ConfigError('invalid_AURIGA_IMAGE_SETTINGS') from error
    elif 'AURIGA_IMAGE_CONFIG' in environ or image_path.exists():
        result['images'] = _read_json(image_path)
    for variable, (section, key) in aliases.items():
        if variable in environ:
            result.setdefault(section, {})[key] = environ[variable]
    return result


@dataclass(frozen=True)
class Config(Section):
    version: int = option(1, choices=(1,))
    run: RunConfig = field(default_factory=RunConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    images: ImagesConfig = field(default_factory=ImagesConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def to_dict(self):
        # Unused agent sections are omitted so Codex-only manifests and snapshots stay byte-identical.
        value = super().to_dict()
        if self.agent == AgentConfig():
            value.pop('agent', None)
        elif self.agent.evidence == EvidenceConfig():
            value['agent'].pop('evidence', None)
        return value

    def __post_init__(self):
        super().__post_init__()
        if not self.features.airborne_handoff and self.features.handoff_preference == 'airborne':
            raise ConfigError('airborne_handoff_preference_requires_airborne_handoff_enabled')
        if self.features.motion_execution_mode == 'feedback':
            if self.features.default_motion_speed_scale != 1:
                raise ConfigError('feedback_motion_execution_requires_default_speed_1')
            if self.features.motion_review:
                raise ConfigError('feedback_motion_execution_requires_motion_review_disabled')
            if self.features.tracking_guard:
                raise ConfigError('feedback_motion_execution_requires_tracking_guard_disabled')
        from .observation_policy import allows_depth
        if not allows_depth(self.features.observation_profile) and any((self.features.depth_skills,
                self.features.depth_auto_feedback, self.features.depth_event_payload)):
            raise ConfigError('depth_features_require_rgbd_observation_profile')
        if self.features.depth_event_payload and not self.features.depth_auto_feedback:
            raise ConfigError('depth_event_payload_requires_depth_auto_feedback')
        if 0 < self.images.trim_max_inline_images < 24:
            raise ConfigError('trim_max_inline_images_requires_0_or_at_least_24')
        if self.features.observation_profile == 'rgb_proprio_v1' and self.features.motion_review:
            object.__setattr__(self, 'features', replace(self.features, motion_review=False))

    @classmethod
    def from_dict(cls, value, *, base=None):
        """Read an archived/full or partial document without consulting environment."""
        # Resolve defaults too, so all exported deployment paths are absolute.
        if not isinstance(value, dict):
            raise ConfigError('config_object_required')
        # The default agent section is omitted from to_dict(); merge it back so partial agent documents inherit
        # backend-specific defaults (e.g. the Anthropic upstream of claude_code).
        defaults = {**cls().to_dict(), 'agent': AgentConfig().to_dict()}
        return _construct(cls, _merge(defaults, _handoff_compat(value)), base=Path(base) if base else None)

    @classmethod
    def load(cls, path=None, *, environ=None, overrides=None):
        environ = dict(os.environ if environ is None else environ)
        value = _merge(cls().to_dict(), _legacy_environment(environ))
        # Resolve legacy/default paths from the caller's cwd before overlaying a
        # file whose relative paths are interpreted next to that file.
        value = cls.from_dict(value).to_dict()
        selected = path or environ.get('AURIGA_CONFIG')
        default = Path(__file__).resolve().parents[2] / 'configs/auriga.json'
        if not default.exists():
            default = Path(__file__).resolve().with_name('default_config.json')
        sources = [default] if default.exists() else []
        if LOCAL_CONFIG_PATH.exists():
            sources.append(LOCAL_CONFIG_PATH)
        if selected:
            sources.append(Path(selected))
        for source in sources:
            source = source.expanduser().resolve()
            document = _read_json(source)
            # Validate unknown keys and types even when later CLI overrides exist.
            # Only normalize fields present in the document; absent defaults must
            # not erase lower-priority legacy settings.
            def normalize(data, schema):
                if not isinstance(data, dict):
                    raise ConfigError('config_object_required')
                declared = {item.name: item for item in fields(schema)}
                hints = get_type_hints(schema)
                out = {}
                for key, part in data.items():
                    if key not in declared:
                        raise ConfigError(f'unknown_config_keys:{schema.__name__}.{key}')
                    item = declared[key]
                    section = _section_type(hints[key])
                    if section is not None and (part is not None or is_dataclass(hints[key])):
                        part = normalize(part, section)
                    elif item.metadata.get('path') and isinstance(part, str):
                        part = str((source.parent / Path(part).expanduser()).resolve()) if part.strip() else part
                    out[key] = part
                return out
            normalized = _handoff_compat(normalize(document, cls))
            cls.from_dict(normalized)  # Validate the file before CLI overrides.
            value = _merge(value, normalized)
        if overrides:
            value = _merge(value, _handoff_compat(overrides))
        return cls.from_dict(value)

    def with_overrides(self, overrides):
        return type(self).from_dict(_merge(self.to_dict(), _handoff_compat(overrides)))

    def validate_launch(self):
        run = self.run
        if run.benchmark and self.features.motion_execution_mode != 'resampled':
            raise ConfigError('benchmark_requires_resampled_unified_action; feedback_is_diagnostic_only')
        if not run.replay and "task_plan" in self.features.disabled_tools:
            raise ConfigError("required_task_plan_tool_cannot_be_disabled")
        if run.resume and run.replay:
            raise ConfigError('--resume and --replay are mutually exclusive')
        if (run.resume or run.replay) and not run.run_dir:
            raise ConfigError('--resume/--replay requires --run-dir')
        if run.benchmark and (run.max_actions is not None or run.wall_budget_s is not None):
            raise ConfigError('--benchmark uses official native step and model-call limits; remove --max-actions and --wall-budget-s')
        if run.benchmark and (run.resume or run.replay or run.prompt):
            raise ConfigError('--benchmark requires a fresh run and uses the native instruction, not --prompt')
        # The default hook is dormant outside the benchmark continuation loop.
        # Explicit initial task-completion instructions still require benchmark.
        if self.features.completion_guidance in {'initial', 'both'} and not run.benchmark and not (run.resume or run.replay):
            raise ConfigError('--completion-guidance requires --benchmark')
        if run.task_notes is not None or run.task_notes_source is not None:
            if not run.benchmark:
                raise ConfigError('--task-notes requires --benchmark')
            from ..cortex.context.task_notes import resolve_task_notes
            resolve_task_notes(False, run.task_notes, run.task_notes_source)
        return self


# Maps existing CLI/case names to their canonical location. This is shared by
# launchers rather than duplicating defaults or constructing ad-hoc namespaces.
RUN_FIELDS = frozenset(item.name for item in fields(RunConfig))
FEATURE_FIELDS = frozenset(item.name for item in fields(FeaturesConfig))
CLI_FIELDS = {**{name: ('run', name) for name in RUN_FIELDS},
              **{name: ('features', name) for name in FEATURE_FIELDS},
              'image_feedback_policy': ('images', 'feedback_policy'),
              'model': ('model', 'name'), 'agent_backend': ('agent', 'backend'), 'effort': ('model', 'effort'),
              'codex_home': ('model', 'codex_home'), 'price_card': ('model', 'price_card')}


def cli_overrides(values):
    result = {}
    for name, value in values.items():
        if name in CLI_FIELDS:
            section, key = CLI_FIELDS[name]
            result.setdefault(section, {})[key] = value
    return result


def namespace_values(config):
    return {name: getattr(getattr(config, section), key) for name, (section, key) in CLI_FIELDS.items()}


MOTION_THRESHOLD_FIELDS = (
    'tight_position_tolerance_m',
    'default_position_tolerance_m',
    'soft_position_tolerance_m',
    'relaxed_position_tolerance_m',
    'compliant_position_tolerance_m',
    'coarse_position_tolerance_m',
    'max_position_tolerance_m',
    'tight_orientation_tolerance_rad',
    'orientation_tolerance_rad',
    'relaxed_orientation_tolerance_rad',
    'coarse_orientation_tolerance_rad',
    'max_orientation_tolerance_rad',
    'contact_divergence_threshold_m',
    'stall_guard_strict_divergence_m',
    'stall_guard_contact_stop_divergence_m',
    'stall_guard_compliant_divergence_m',
    'stall_guard_min_divergence_m',
    'stall_guard_max_divergence_m',
    'stall_guard_step_velocity_threshold_m',
    'stall_guard_consecutive_steps',
    'max_gripper_aperture_m',
    'grip_empty_threshold_mm',
    'grip_thin_threshold_mm',
    'grip_holding_threshold_mm',
    'grip_solid_threshold_mm',
    'grip_non_empty_min_aperture_m',
    'grip_thin_min_aperture_m',
    'grip_thin_max_aperture_m',
    'grip_rigid_min_aperture_m',
    'grip_rigid_max_aperture_m',
    'grip_wide_min_aperture_m',
    'grip_wide_max_aperture_m',
    'max_min_grip_m',
    'closed_jaw_goal_max',
    'checkpoint_empty_jaw_measured_max',
    'joint_travel_standard_max_rad',
    'joint_travel_standard_backtracking_rad',
    'joint_travel_extended_max_rad',
    'joint_travel_extended_backtracking_rad',
    'segment_start_joint_drift_rad',
    'resume_max_joint_drift_rad',
    'resume_max_ee_drift_m',
    'resume_max_orientation_drift_rad',
    'resume_max_jaw_drift_normalized',
    'batch_plan_cache_size',
    'hotspot_xy_radius_m',
    'hotspot_z_radius_m',
    'ledger_max_recent_outcomes',
    'ledger_max_hotspots',
    'target_exemption_radius_m',
    'clearance_height_m',
)

_DEFAULT_FEATURES = FeaturesConfig()
_DEFAULT_MOTION_THRESHOLDS = {name: getattr(_DEFAULT_FEATURES, name) for name in MOTION_THRESHOLD_FIELDS}


def motion_thresholds(source=None):
    """提取统一配置中的运动、接触、夹爪、行程与恢复经验阈值，支持 Config/FeaturesConfig/Harness/Adapter 或字典输入。"""
    result = dict(_DEFAULT_MOTION_THRESHOLDS)
    if source is None:
        return result
    if isinstance(source, dict):
        for name in MOTION_THRESHOLD_FIELDS:
            if name in source and source[name] is not None:
                result[name] = source[name]
        return result
    explicit_dict = getattr(source, 'motion_thresholds', None)
    if isinstance(explicit_dict, dict):
        for name in MOTION_THRESHOLD_FIELDS:
            if name in explicit_dict and explicit_dict[name] is not None:
                result[name] = explicit_dict[name]
    cfg = getattr(source, 'config', source)
    feat = getattr(cfg, 'features', cfg)
    for target in (feat, source):
        if target is None or isinstance(target, dict):
            continue
        for name in MOTION_THRESHOLD_FIELDS:
            val = getattr(target, name, None)
            if val is not None and type(val) in (int, float):
                result[name] = val
    return result
