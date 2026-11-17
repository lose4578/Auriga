# Unified Configuration and Feature Flags

`auriga.core.config.Config` is the unified configuration entry point for new runs. The default configuration file in the repository is `configs/auriga.json`, and a complete annotated example is available in [`configs/auriga.example.json`](../../configs/auriga.example.json). In practice, you only need to specify the fields you want to override; all omitted fields inherit defaults from the `Config` class. Because JSON does not support comments, this document serves as the reference for every configuration field, dependency, and legacy environment variable mapping.

When building a wheel, `scripts/build_package.py` copies the repository default file into the package as `auriga/default_config.json`. Installed wheels fall back to this packaged copy when running outside a repository checkout, so new experiments share identical defaults whether run from source or from a wheel. Explicit `--config` / `AURIGA_CONFIG` overrides still take precedence, and `Config.from_dict()` preserves historical defaults when loading older archives.

## Usage and Override Precedence

All project commands below run inside the Miniforge `Auriga` environment; for environment creation and verifying the resolved Python interpreter and Codex SDK/CLI versions, see the [Running Guide](running.md#environment-setup).

```sh
# Run with the repository default configuration
mamba run -n Auriga python -m auriga.apps.gateway --run-dir runs/my-run --gpu 4

# Use a custom experiment configuration; CLI flags override settings for this run only
mamba run -n Auriga python -m auriga.apps.gateway --config configs/my-experiment.json \
  --run-dir runs/my-run --gpu 4 --no-enable-wait

# Validate and print the full effective configuration without starting models, GPUs, or services
mamba run -n Auriga python -m auriga.apps.gateway --config configs/my-experiment.json --print-config

# Point all unified-config entry points to a specific configuration file
export AURIGA_CONFIG=/absolute/path/auriga.json
```

Configuration layers merge in the following order:
1. `Config` class defaults
2. Compatible legacy environment variables
3. Repository default file `configs/auriga.json` (or the packaged fallback in installed wheels)
4. Local repository override file `configs/auriga.local.json` (if present)
5. Explicit JSON configuration file (`--config` or `AURIGA_CONFIG`)
6. Explicit CLI flags

Nested objects merge field-by-field, so an experiment configuration that only overrides `features` still inherits the default model and local Codex home, while an explicit configuration specifying legacy model settings can still override the newer defaults. `--config` takes precedence over `AURIGA_CONFIG`; when both are omitted, the merged public and local defaults are used. Every configuration file is strictly validated on load, and relative paths inside a file resolve relative to that file's directory. Unpassed CLI flags never overwrite file values. Unknown field names, invalid types, non-finite numbers, and unsupported enum values are rejected before any service, model, or robot worker starts.

Relative paths in JSON files resolve relative to the directory containing that file, whereas CLI and legacy environment variable paths resolve relative to the current working directory (`~` is expanded in both cases). Saved effective snapshots store resolved absolute paths. The minimal valid JSON file is `{"version": 1}`, as the `Config` class supplies all defaults. Because `configs/auriga.example.json` uses `~` paths, review its `backend` and `policy` sections when copying it to another machine.

Array fields replace the previous list as a whole—for example, passing `--tag trial` replaces the tags defined in the config file, and `--clear-tags` empties the list. `null` in JSON is only valid for optional fields that explicitly permit `null` in the schema; never use `null` to disable a boolean flag (use `false` instead).

For example, the file below keeps the default calibrated observation profile, disables `robot_wait` and $\pi_{0.5}$, and enables tracking protection and compact actor payloads:

```json
{
  "version": 1,
  "features": {
    "enable_wait": false,
    "pi05": false,
    "tracking_guard": true,
    "actor_payload_profile": "compact_v1"
  },
  "images": {
    "reuse_exact_images": false
  }
}
```

New runs enable `robot_wait` by default. Disabling it only removes the model's ability to call `robot_wait`; it does not change whether observations advance physical simulation time (`robot_observe`, model thinking time, and wall-clock sleeps never advance the paused simulation).

Python callers share the same immutable `Config` object so individual modules never re-read environment variables or diverge in defaults:

```python
from auriga.core.config import Config

config = Config.load("configs/my-experiment.json").validate_launch()
assert config.features.enable_wait is False

# Returns a new Config instance without mutating the original; an explicit False is never treated as "omitted".
candidate = config.with_overrides({"features": {"enable_wait": True}})
snapshot = candidate.to_dict()

# Loading from an archived dictionary does not read current environment variables or default files.
restored = Config.from_dict(snapshot)
```

`Workbench`, `Harness`, `RobotBackend`, `CodexSession`, and the Experiment Library all accept `config=`, while preserving existing explicit constructor keyword arguments for backward compatibility. Loading a configuration and running `validate_launch()` do not start any external services.

## Run and Model Settings

| Field | Default / Values | Purpose and Constraints |
| --- | --- | --- |
| `run.task` | `stack_bowls` | Native task name. |
| `run.seed` / `run.layout_id` | `0` / `0` | Non-negative integers specifying initial environment conditions. |
| `run.gpu` | `4` | GPU index for the robot simulator and local policy worker; the evaluation queue assigns the actual GPU per case. |
| `run.run_dir` | `null` | Explicit directory for a new run; when `null`, the Experiment Library allocates one automatically. |
| `run.experiment_root` | `~/exp/auriga` | Root archive directory used when allocating run directories automatically. |
| `run.category` / `run.title` / `run.tag` | `main` / `null` / `[]` | Archive category, human-readable title, and list of tags. |
| `run.port` | `0` | Loopback-only HTTP port; `0` selects an ephemeral free port automatically. |
| `run.prompt` | `null` | Custom prompt for development runs; when `null`, uses the default interface check prompt (official benchmarks always use the native public task instruction). |
| `run.benchmark` | `false` | Enables the official benchmark lifecycle: native task instruction, same-thread turn continuation, independent native evaluation, and final cleanup. |
| `run.exit_after_turn` | `false` | Exits after the first turn completes; benchmark runs ignore this and continue until protocol termination conditions are met. |
| `run.resume` / `run.replay` | `false` / `false` | Mutually exclusive flags requiring an existing `run_dir`; official benchmarks only allow fresh runs. |
| `run.max_actions` | `null` | Upper bound on high-level actions for development checks; rejected in official `--benchmark` runs. |
| `run.wall_budget_s` | `null` | Whole-episode wall-clock budget in seconds for development runs; rejected in official `--benchmark` runs. |
| `run.task_notes` / `run.task_notes_source` | `null` / `null` | Custom public benchmark notes and provenance appended after default operational experience; must be provided together as non-empty strings (custom body up to 8,000 characters). |
| `model.name` / `model.effort` | `gpt-6.1-sol` / `high` (repo default for new runs) | Primary actor model identifier and reasoning effort tier, configured independently. |
| `model.codex_home` | `~/.codex` (repo default for new runs) | Host directory from which local Codex provider routing and credentials are read; host extensions are not inherited, sessions run in a per-run private directory, and OS isolation is governed by `features.sandbox_enabled`. |
| `model.price_card` | `null` | Optional path to an explicit pricing card JSON file; cost estimation never alters raw token evidence. |
| `model.native_memories` | `true` (repo default for new runs) | Controls reading and automatic consolidation of private native Codex memories without modifying host configuration or Auriga's `operation_memory`. |
| `model.transport_recovery.max_rounds` | `10` | Maximum cooldown-and-resend rounds (1 HTTP attempt per round) after the 12 automatic retries for explicitly retryable errors are exhausted or after an upstream `transport_incomplete` disconnect; `0` disables recovery, maximum `20`. |
| `model.transport_recovery.initial_delay_s` / `backoff_multiplier` / `max_delay_s` | `60` / `2` / `600` | Cooldown schedule across rounds: `60`, `120`, `240`, `480`, `600`... seconds. |
| `model.transport_recovery.max_episode_delay_s` | `3600` | Maximum cumulative transport-recovery cooldown seconds across an entire episode (single proxy lifecycle); once exceeded, the run terminates normally. |
| `model.transport_recovery.transport_incomplete` | `true` | Whether to also recover from upstream transport disconnects that lack a terminal response (e.g., `IncompleteRead`); `false` only recovers explicitly retryable provider errors. |

Configuration files never store API tokens or duplicate provider secrets. Provider endpoints, authentication methods, and upstream URLs remain managed by the native Codex configuration and credential files inside `model.codex_home`, and provider API keys in environment variables continue to be resolved as usual.

On a local workstation, you can use a Git-ignored `configs/auriga.local.json` to point to a dedicated provider home—for example, `{"version": 1, "model": {"codex_home": "~/.config/auriga/providers/my-provider"}}`. Keep only the directory path in `configs/auriga.local.json`, and place the provider's `config.toml` and key-bearing `auth.json` inside that external directory with directory permissions `0700` and file permissions `0600` (including `configs/auriga.local.json`). Never place raw API keys inside unified JSON configuration files, because resolved configurations are written into manifests, experiment archives, and web reports. Using a dedicated Codex home prevents provider changes from affecting interactive Codex CLI sessions or already running queues, and `configs/auriga.local.json` is never bundled into wheels. An explicit `--config` still overrides `configs/auriga.local.json`, and neither `Config.from_dict()` nor already frozen queues read the local override file, ensuring historical snapshots are never reinterpreted when local defaults change.

Both `configs/auriga.json` and `configs/auriga.example.json` explicitly select `gpt-6.1-sol`, `high`, and `~/.codex` for new runs, while local defaults or explicit configs can point to another isolated home. At startup, Auriga reads the active provider from `config.toml` in the selected home and reads credentials from `auth.json` or the provider's declared authentication settings, keeping real credentials strictly inside the host relay. The actor's private per-run home receives placeholder credentials without copying real API keys, `.credentials.json`, or host extensions from the source directory, and never writes back to the source directory. Never commit secrets to tracked config files, manifests, CLI arguments, or logs, and never check a source Codex home into Git. `model.price_card` defaults to `null` so pricing rules from another provider or model are not applied inadvertently.

The primary actor's model name and reasoning effort are governed strictly by the unified configuration and are not overridden by eponymous preferences inside the source Codex home. `Harness` pins `web_search = "disabled"`, `service_tier = "default"`, and the `Responses` wire protocol; the native provider identifier `auriga_model_proxy` points to the local accounting proxy, which forwards requests to the external upstream defined by the selected local Codex provider. When enabled, the optional auxiliary visual review model reuses the provider, proxy, and authentication prepared for the primary session, running `gpt-6.1-sol` / `high` in its own isolated home and thread without re-reading the source Codex home.

Default model selection for new runs lives solely in the repository configuration file; `Config.from_dict()` does not consult `configs/auriga.json`, so older archives lacking explicit model fields continue to resolve to their historical `gpt-6-astra` / `~/.codex` defaults. Evaluation queues with a frozen `auriga_config` continue using their frozen snapshot. When launching a new experiment from an older manifest template, pass `--config configs/auriga.json` explicitly or regenerate the manifest. Note that a queue freezes the `codex_home` path rather than inlining secret files, so each case reads the provider files at that path when it launches; if you switch providers for future runs, create a new provider home directory rather than mutating a directory referenced by an active queue. The legacy `~/.config/auriga/experiments.json` file supports only `codex_home` and `price_card` paths; omit `price_card` entirely (rather than setting it to JSON `null`) when no price card is used.

`manifest.auriga_config` stores the resolved effective configuration snapshot; even if a historical snapshot has only a subset of fields, missing fields are interpreted using historical defaults. Supply partial configuration overrides for new runs via `--config`, which loads the current defaults first and freezes the complete resolved snapshot.

## Code Execution Sandbox

`features.sandbox_enabled` is a global boolean toggle defaulting to `false` in both the `Config` class and `configs/auriga.json`; set it explicitly to `true` to enable OS-level isolation. It simultaneously controls the Linux bubblewrap namespace around the primary agent session of every backend (Codex with its native terminal, subagents, shell commands and memory consolidation; Claude Code and OpenCode with their native tools and subagents), the bubblewrap namespace around the auxiliary Codex evidence agent, and Landlock/seccomp isolation for `program_run`. Its purpose for benchmarks is that the model cannot read the RoboDojo / simulator source (reward hacking); the published Harness source stays readable at `/harness`. RoboDojo comparisons across backends therefore run with the sandbox on for every arm; see [Agent Backends](agent-backends.md#sandbox-features-sandbox_enabled-all-backends) for the per-backend mounts. When set to `false`, Auriga neither probes nor loads these isolation mechanisms, so `bwrap`, `libseccomp`, unprivileged user namespaces, and Landlock are not required on the host (standard runtime dependencies still apply).

For example, save the following as `configs/sandbox.json` to enable the sandbox for a new run:

```json
{
  "version": 1,
  "features": {
    "sandbox_enabled": true
  }
}
```

```sh
mamba run -n Auriga python -m auriga.apps.gateway --config configs/sandbox.json --print-config
mamba run -n Auriga python -m auriga.apps.gateway --config configs/sandbox.json \
  --run-dir runs/sandbox-trial --gpu 4
```

When disabled (`false`, the default), each run still uses an isolated per-run `HOME` / `CODEX_HOME`, the local model relay, and placeholder credentials without inheriting host MCP servers, plugins, or skills, and `program_run` retains its FD broker, tool permission checks, action review, budgets, and resource limits (`RLIMIT`). However, executed code runs with the launching user's host filesystem and network privileges—private directories and placeholder tokens do not form an OS security boundary, and formal observation restrictions are enforced solely at the application layer. Consequently, unsandboxed runs do not provide OS-level proof that host files, credentials, evaluators, or observer artifacts were inaccessible; always record whether a run executed with or without the sandbox.

When the sandbox is disabled, the read-only copy of public Harness source code is staged at `<run>/private/actor-source` instead of being bind-mounted at `/harness`. The `sandbox_enabled` setting is persisted in the effective configuration snapshot; editing a config file on disk never hot-switches a running experiment or a frozen queue. Resuming a session across different sandbox modes is rejected (start a new run instead), and legacy full-access sessions lacking isolation markers cannot be migrated in place.

The isolation, mount, and controlled network rules below apply when `features.sandbox_enabled=true` is explicitly enabled. During `CodexSession._prepare()`, `Harness` launches the entire native Codex app-server inside a Linux bubblewrap sandbox. Isolated mount, PID, IPC, and network namespaces cover the primary thread, subagents, shell tool, and memory consolidation, running under the current user's UID/GID with all Linux capabilities cleared. `bwrap`, `sudo`, and `setpriv` are discovered from the current `PATH` and platform default `PATH`. Auriga prefers unprivileged user namespaces; only when the host kernel disables unprivileged user namespaces does it fall back to `sudo -n` to create the namespace and immediately drop privileges via `setpriv`. If sandbox initialization fails, startup aborts rather than falling back to unsandboxed host execution. The launch gate (`factory.backend_requirements`) accepts the sandbox for `codex`, `claude_code` and `opencode` alike; a host without bubblewrap is refused when the session starts, before the native agent process exists (`codex_sandbox_bubblewrap_not_found_on_path_or_os_defpath` for Codex, `agent_sandbox_bubblewrap_not_found_on_path_or_os_defpath` for Claude Code and OpenCode). On a host where the gateway user has supplementary groups, the unprivileged path cannot drop them inside the user namespace, so the rootful `sudo -n` fallback is used there as well.

The run directory, source snapshot path, and native Codex binary are resolved from the run configuration, the installed `auriga` package, and the pinned `openai-codex-cli-bin` package respectively, without hardcoding usernames, checkout paths, or Conda installation directories. IPC sockets use `XDG_RUNTIME_DIR` (or Python's platform temporary directory when unset); the runtime directory must be owned by the current user with `0700` permissions and have a short enough path to fit Unix domain socket limits. System `python3` is discovered from the platform default `PATH`, its standard library path is queried from that interpreter, and required ELF shared libraries are resolved via `ldd`—Auriga never bind-mounts all of `/usr`, the Conda environment, or host `site-packages` / `dist-packages`. Paths such as `/harness`, `/tmp`, and `/usr/bin` refer to mount points inside the sandbox rather than host machine configuration paths.

System utilities are exposed inside the sandbox according to the explicit allowlist in [`system_runtime.py`](../../auriga/cortex/session/system_runtime.py), covering standard shell utilities, file operations, text processing, archiving, and local Git, while `rg` is discovered from the active environment. Optional utilities that are not installed on the host are simply omitted. Inside the sandbox, Python can install scientific computing packages through the private PyPI proxy, whereas other host binaries, compiler toolchains, and system package managers in `/usr` are not exposed.

Inside the outer bubblewrap container, Codex runs with `approval_policy="never"` and `sandbox_mode="danger-full-access"` so native tools can operate freely **within the outer OS sandbox**. OS isolation never relies on the model voluntarily following prompt instructions. The generated native configuration sets:

```toml
web_search = "disabled"
approval_policy = "never"
sandbox_mode = "danger-full-access"
model_verbosity = "high"
service_tier = "default"

[features]
shell_tool = true
unified_exec = true
memories = true
multi_agent = true
multi_agent_v2 = true
hooks = true
goals = true
browser_use = false
browser_use_external = false
computer_use = false
apps = false
plugins = false
artifact = false
image_generation = false
network_proxy = false
code_mode_host = false
fast_mode = false
```

Inside the sandbox, `HOME` and `CODEX_HOME` point to the run's actor workspace and private session directory, system runtime dependencies are mounted read-only, and `/tmp` is private to the sandbox. `/harness` exposes a read-only snapshot of the public Harness, `ToolSpec`, action, and program interfaces as defined by the allowlist in `auriga/cortex/session/actor_source.py`. It excludes RoboDojo, backend task implementations, evaluators, reference policies, `.git` metadata, historical patches, experiment databases, the host home directory, and sibling repositories. `/harness` is provided for inspecting interface contracts; the full checkout is never mounted to satisfy arbitrary imports.

The isolated network namespace has no direct access to the host network or the public Internet. Traffic is forwarded exclusively over per-run Unix domain sockets to the local accounting model relay, the read-only account relay, and the controlled Python package proxy. The package proxy permits HTTPS requests strictly to `pypi.org` and `files.pythonhosted.org` so `python3 -m pip install <package>` installs into the actor's private workspace and cache; arbitrary websites, host loopback services, SSH, Git remotes, and alternate package mirrors are unreachable. Web search, browser tools, and external app/MCP integrations are disabled, and the model relay rejects server-side web search or remote tool invocations.

Host provider routing and credentials are read by `Harness` and remain inside the host relay; the sandboxed Codex process receives only the local relay URL and placeholder credentials. Supported host authentication methods include API keys or ChatGPT access tokens in `auth.json`, provider `env_key` variables, and bearer tokens, with custom auth headers restricted to `Authorization` and `ChatGPT-Account-Id`. Real `auth.json` files, `.credentials.json`, MCP servers, plugins, skills, and host conversation history/memories are never copied into the run, and the source Codex home (or CC Switch configuration) is never modified. Note that ChatGPT access tokens are read once when the host gateway starts—the sandbox cannot refresh host login sessions if a token expires mid-run.

Dynamic robot tools continue to use the direct tool catalog with `code_mode=false` and `code_mode_host=false` so image outputs are delivered as structured image items in `Responses`. The model name and reasoning effort come from the unified configuration, while native background requests (such as memory consolidation or thread titling) are accounted separately under their actual role and model. `model.native_memories=true` enables per-run native file memories (set to `false` to disable), which remain completely separate from Auriga's `operation_memory` and `grasp-memory`.

The standalone auxiliary evidence agent runs in its own separate bubblewrap namespace with an empty workspace and dedicated IPC socket—mounting neither the primary actor's workspace, `/harness`, nor the PyPI proxy—and receives only the selected evidence payload with all tools disabled. `program_run` retains its dedicated FD broker and Landlock/seccomp filter, restricting Python imports to the standard library and approved scientific computing packages while disabling host `.pth` files and unapproved Conda/`site-packages` directories. All physical robot actions from programs still pass through the single `Harness.execute` admission point.

## Feature Flags

| Field | Default (New Runs) | Controlled Behavior |
| --- | --- | --- |
| `features.sandbox_enabled` | `false` | Global toggle for the bubblewrap namespace of the primary agent (every backend), the auxiliary Codex evidence agent, and `program_run` Landlock/seccomp isolation (enabled only when explicitly set to `true`). When `false`, code runs with the launching user's host file/network permissions while retaining application-level tool validation, action review, and budgets; see [Code Execution Sandbox](#code-execution-sandbox). |
| `features.observation_profile` | `rgb_proprio_calibrated_v1` | Permits RGB images, proprioceptive state, exact camera/robot calibration, and the public robot model. Also supports legacy `rgb_proprio_v1`, while explicit `rgbd_proprio_calibrated_v1` is required to enable native depth capture and numeric depth tools. |
| `features.motion_review` | `true` | Enables advisory motion preview, cached plan execution, and optional grasp planning/memory tools; numerical risk diagnostics do not block execution. Automatically disabled under the uncalibrated legacy observation profile. |
| `features.default_task_notes` | `true` | Injects versioned operational experience notes into new benchmark runs; custom `run.task_notes` can still be used when disabled, and non-benchmark smoke runs do not inject default notes. |
| `features.vision_compare` | `true` | Enables RGB feature comparison (`vision_compare`) and RGB skills that depend on it; disabling it does not affect explicitly enabled RGB-D skills. |
| `features.robotic_view` | `false` | [Wrist View Solver](robotic-view.md): computes `link6` poses from a caller-specified camera viewpoint; requires calibrated observations and returns preview parameters without moving the robot. |
| `features.stereo_guide` | `false` | [Stereo Epipolar Guide](stereo-guide.md): renders the epipolar line of a source pixel onto another synchronized camera view; requires calibrated zero-distortion pinhole cameras and does not perform automatic feature matching or triangulation. |
| `features.depth_auto_feedback` | `false` | Automatic pre/post-action numeric depth verification for registered targets; requires an RGB-D profile and adds no extra captures or physical actions. |
| `features.depth_skills` | `false` | Bounded grasp-and-lift and place-release-retract depth skills (requires RGB-D profile). Because the `skill_run` execution tool was removed in the 2026-10 refactoring, this flag currently only affects the `enabled` field in the `depth_query` `skill_help` catalog. |
| `features.depth_event_payload` | `false` | Allows routine intermediate actions with reliable depth measurements to deliver numeric summaries in place of inline images while still archiving full images; requires both an RGB-D profile and `depth_auto_feedback`. |
| `features.enable_wait` | `true` | Exposes `robot_wait` to explicitly advance simulation by 1–50 native steps and return temporal image frames. |
| `features.pi05` | `false` | Exposes the $\pi_{0.5}$ short-horizon policy prefix tool matched to the ARX X5. |
| `features.visual_review_model` | `null` | Disables auxiliary visual/operation verification by default; set explicitly to `gpt-6.1-sol` to enable `vision_review`, which shares the episode's model request budget. |
| `features.tracking_guard` | `false` | Halts end-effector motion when sustained joint tracking error exceeds thresholds, returning partial execution evidence. |
| `features.joint_replan` | `true` | Searches for closer valid $2\pi$-equivalent joint targets and substitutes the trajectory only when verified and shorter in native steps. |
| `features.motion_execution_mode` | `resampled` | Official actor and benchmark runs use `resampled` unified actions; `feedback` is restricted to trusted legacy fixed-input diagnostics, requires `motion_review=false` and `tracking_guard=false`, and is rejected prior to benchmark launch. |
| `features.default_motion_speed_scale` | `1.0` | Default arm motion speed multiplier (`0.25`–`2.0`) when a stage specifies neither a speed preset nor an explicit scale; explicit stage settings override this default, gripper speed defaults independently to `1.0`, and `feedback` mode permits only `1.0`. |
| `features.airborne_handoff` | `true` | Permits airborne bimanual handoff templates; `--no-airborne-handoff` restores the `surface_only` hard constraint requiring an intermediate support surface. Simultaneous dual-arm motion, independent progression, and arm queues remain available in both modes. |
| `features.handoff_preference` | `surface` | `--handoff-preference surface/airborne` selects the handoff guidance preference when airborne handoffs are permitted. The default combination (`airborne_handoff=true` + `handoff_preference="surface"`) resolves to `prefer_surface`: the model chooses autonomously, with physical support surfaces recommended rather than forced. `true + "airborne"` resolves to `prefer_airborne`. When legacy boolean configs omit `handoff_preference`, `true` restores legacy `prefer_airborne` and `false` restores `surface_only`; historical archives likewise preserve their recorded policy. Setting `airborne_handoff=false` with `handoff_preference="airborne"` is rejected. |
| `features.actor_payload_profile` | `lean_v1` | Strips routine storage metadata and queryable long references without rounding numerical measurements; also supports `full_v1` and `compact_v1` (program interfaces and raw archive evidence always retain full data). |
| `features.completion_guidance` | `hook` | Benchmark runs default to state-aware home-pose checking on turn completion; `initial` injects the reminder only in the initial prompt, `both` enables both, and `none` disables home-pose checking while retaining generic continuation. |
| `features.grasp_memory_dir` | `~/.local/share/auriga/grasp-memory` | Cross-run grasp memory directory, consulted only when grasp memory features are enabled. |
| `features.disabled_tools` | `[]` | Disables specific known tool names individually (taking precedence over feature-group enables). `robot_capabilities` cannot be disabled in benchmark runs because the gateway queries it through the registry at startup; disabling `operation_memory` also omits its section from the startup instructions. |
| `features.lean_projection` | `false` | Upstream request history projection ([`auriga/core/presentation/lean_projection.py`](../../auriga/core/presentation/lean_projection.py), disabled by default; rewrites only outbound upstream request bodies without mutating the Codex thread history): replaces inline images in non-anchor tool outputs of closed historical blocks with text placeholders, and replaces verbose text outputs of 6 high-frequency tools (`robot_action`, `program_run`, `robot_observe`, `robot_wait`, `geometry_query`, `history_view_observation`) with deterministic summaries; when enabled, observation payloads also stop repeating static fields delivered in earlier turns. **Potential performance trade-off**: compressing historical feedback and images discards pixel details, full trajectory samples, and long `stdout` logs from non-anchor steps in closed blocks (unless the model explicitly calls `history_view_observation` to re-read them), which may affect long-horizon spatial reasoning or multi-step failure recovery; keep it `false` by default. |
| `features.lean_projection_k` | `1` | Number of most recent tool steps in the tail block that retain full inline images (`1`–`8`, active only when `lean_projection_block=0`). |
| `features.lean_projection_block` | `8` | Block-aligned mode (default strategy when `lean_projection` is enabled): omits the separate tail window; tool steps in the currently open block and the first step (anchor) of each closed block are sent verbatim, while non-anchor steps in closed blocks are replaced with summaries/placeholders. Because rewrites occur only once every `B` steps, request `n+1` within a block is a strict prefix extension of request `n`, and on block boundaries the anchor request remains a valid prefix so only the newly closed block needs to be re-prefilled. In block mode, the decision-boundary reminder (`_task_decision_context`) is placed in fixed slots rather than appended at the very end of every request (one slot immediately before the task message in the initial request, plus one slot per block created on the first request in that block that has a saved plan, positioned right before the latest model turn / after the previous tool output). Each slot retains the snapshot from when it was created, closed-block slots remain in place, and the "no plan yet" reminder appears only in the initial slot (see the `lean_projection.py` docstring). Setting `0` selects sliding tail-window mode (note that on tested `Responses`-compatible relays, upstream KV caching does not reuse prefixes once a tail window shifts, so cache hits stop at the last request before the shift). |
| `features.lean_request_dump` | `0` | Diagnostic setting: writes the first `N` outbound upstream request bodies (with inline image bytes replaced by digests) to `private/request-dump/`; `0` disables dumping. |
| `features.external_perception` | `true` | Master toggle for external vision models (VGGT + SAM). Both the `Config` class and `configs/auriga.json` enable `vggt_height` (`vggt_height_mode="optional"`), `object_table`, `scene3d`, `object_table_observe`, `scene3d_observe`, and `held_perception` by default against `features.vggt_endpoint` (default `http://127.0.0.1:8331`). If the service is unreachable or a request fails, the corresponding fields are marked `unknown` / `unavailable` with a diagnostic reason while observations and action receipts are delivered normally without aborting the episode. Setting `external_perception=false` while leaving all sub-flags at their default `true` disables all external perception calls; setting all sub-flags to `false` while `external_perception=true` enables all of them; any mixed combination honors each individual sub-flag. CLI flags follow `--[no-]external-perception` and matching per-sub-flag options; see `docs/vggt_height.md`, `docs/object_table.md`, `docs/scene3d.md`, and `docs/obs3d.md`. |
| `features.held_perception` | `true` | In-hand object and robot-mesh mask perception (enabled by default with `external_perception`, requiring `object_table` or `object_table_observe` and a service supporting `object-table-v5-hp1`, with automatic fallback to `object-table-v5`): replaces the 6 cm link capsule approximation in the object table with a forward-kinematics robot mesh mask rendered from measured joint angles (dilated by 3 px). Any arm whose gripper is closed around an object receives a dedicated held-object row (`quality: "held_fk_depth"`, `support: "held_<arm>"`) whose dimensions and principal axis are carried rigidly with `link6` from the pre-grasp object table row and centered at the FK grasp center (translating slender objects along their principal axis by the carry offset, subject to a 45-degree axis-conflict gate). When no pre-grasp row exists, dimensions are estimated from VGGT points inside the SAM mask rescaled by the FK grasp depth. Held rows bypass airborne-to-tabletop re-projection, `value.held[id]` reports the estimation method, uncertainty, and source row, and one sentence is appended to developer instructions; when disabled, the tool catalog, instructions, and object table remain unchanged (see `docs/held_perception.md`). |
| `features.record_video` | `true` | Automatically starts native three-camera video recording. |
| `features.export_video` | `true` | Exports recorded videos during run finalization (skipped automatically when `record_video=false`). |
| `features.terminal` | `true` | Starts the native read-only Codex terminal; disable for headless runs without a TUI. |

`feedback` motion execution mode is reserved exclusively for trusted Python diagnostics such as `scripts/probes/feedback_motion_probe.py --mode feedback`; it does not support the public `robot_action` tool and cannot be used for unified dual-arm manipulation by the actor or programs. Both `Config.validate_launch()` and the evaluation queue reject `feedback` mode before launching models or workers for a benchmark. Non-benchmark diagnostic workbenches may pass `--motion-execution-mode feedback --no-motion-review --no-tracking-guard`, though this does not expose public arm action tools; historical archives recorded in `feedback` mode remain readable and replayable. In `feedback` mode, the worker plans once per waypoint, selects the closest $2\pi$-equivalent joint target within limits (verified for full-link URDF equivalence and target pose accuracy), and repeatedly issues commands through the native `take_action` entry point until position error is within the requested tolerance, orientation error is at most `0.1 rad`, and all active-arm joint velocities are at most `0.1 rad/s` for two consecutive native steps. If neither best position/orientation error nor joint target error improves sufficiently over 15 native steps, execution stops with `feedback_stalled` regardless of instantaneous velocity (which also bounds cases where position is reached but velocity fails to settle); each action remains bounded by 10 seconds of simulated time (default 250 native steps), the action wall-clock deadline, cancellation, and native terminal conditions without altering the episode's official step limit.

`feedback` mode preserves sequential user waypoints and `after_motion` gripper dwell times while explicitly rejecting cached trajectory previews and `during_motion` fractional gripper timing. Because closed-loop feedback tracks the planned endpoint rather than following the initial open-loop trajectory verbatim, collision checks performed during initial planning do not guarantee a collision-free execution path. `reference_resampled_steps` provides only a diagnostic estimate based on initial trajectory row counts, whereas actual step accounting always uses `executed_native_steps` from the receipt. Step accounting for `robot_wait` and joint policy prefixes is unchanged.

In controlled comparisons on RoboDojo bottle manipulation tasks, `feedback` mode reduced step counts on short-horizon motions but did not improve full-task success rates; the comparison surfaced a stall-detection gap under high instantaneous velocity and long-windup joint targets, both of which have since been fixed. `feedback` mode is retained solely for trusted fixed-input diagnostics, while official unified actions default to and require `resampled`.

Whether a tool is enabled in configuration and whether the model chooses to call it are separate questions. `features.disabled_tools` filters the model's tool catalog, direct actor calls, and `program_run` broker calls simultaneously. Physical actions initiated from inside a program pass through the exact same `robot_action` execution boundary and approval workflow and cannot bypass them via legacy action names. Disabling `program_run` or individual tools is never a substitute for validating native observation and execution protocols.

Depth sensing is disabled by default across native capture, model tools, programs, and history queries. For the three explicit RGB-D feature flags, numeric interfaces, and skill parameters, see [RGB-D Numeric Tools](rgbd-numeric-tools.md). The public `depth_query` tool is exposed only under the `rgbd_proprio_calibrated_v1` profile (legacy `depth_move` was removed in the 2026-10 refactoring), and adding `depth_query` to `disabled_tools` also disables the depth alias on `geometry_query`. Combining any depth feature flag with a non-RGB-D observation profile raises a configuration validation error at startup.

### State-Aware Completion Guidance (`HomePoseCheck`)

`completion_guidance="hook"` or `"both"` uses the [`HomePoseCheck`](../../auriga/runtime/actions/home_check.py) component. At the start of each episode—after native reset and scene settling, and before the first actor action—`HomePoseCheck` records the measured dual-arm `ee_pose` (`link6` position and `quaternion_xyzw` orientation in the `env_origin` frame) from the first public observation. This initial reference is preserved in observation history and the home reference record without being overwritten by subsequent observations, and is re-recorded only when an explicit environment reset or session resume starts a new episode. New benchmark runs default to `hook`, which neither injects an upfront home-pose reminder into the initial prompt nor sends unconditional home-pose warnings on every turn; non-benchmark development runs do not trigger benchmark turn continuation. Resumed and replayed runs honor their archived setting (defaulting to `none` for legacy archives lacking the field).

When the primary actor emits `turn.completed`, the native episode has not ended, and standard scheduling guards permit continuation, `HomePoseCheck` compares the latest complete observation against the recorded initial pose using RoboDojo's default `all_robot_back_to_origin` geometry rule: the absolute difference along every Cartesian position axis for both arms must be **at most 0.15 m**, and the shortest quaternion rotation angle must be **at most 20 deg** (inclusive at the boundary). Quaternions `q` and `-q` represent the same orientation, and full-turn joint windup and gripper aperture are ignored. While this matches the terminal home condition of arrangement tasks, it does not represent the complete success criteria of every task; verify task-specific homing conventions before relying on it for other task families.

| Check Outcome | Continuation Prompt Sent |
| --- | --- |
| `away`: valid dual-arm measurements exist and at least one arm exceeds tolerance | Explicitly reminds the model that the arms have not returned to the initial pose, provides the history reference to the initial observation, and notes that other task conditions may also remain unmet. |
| `home`: both arms are within tolerance | Sends the standard continuation prompt ("The task is not finished yet; please continue."). |
| `unknown`: missing initial/current measurements, stale observation, mismatched identity/frame, or invalid data | Sends the same standard continuation prompt without diagnosing missing evidence as an unhomed robot. |

`HomePoseCheck` reads only permitted robot-state artifacts—it never issues actions, automatically resets the arms, or queries the private evaluator or object ground truth. Admitting any physical action invalidates the current observation evidence; intermediate progress snapshots cannot substitute for a terminal observation, so only a complete post-action terminal capture or a subsequent `robot_observe` restores a valid check state. While the simulation is paused, the same complete observation can be checked repeatedly, and read-only `history_*` queries do not alter the active observation evidence. Native terminal states, cumulative model request budgets, timeouts, unconfirmed actions, and active program guards always take precedence over the continuation hook.

Each episode logs a `completion_guidance.home_reference` event recording the initial reference pose, while the manifest and task-prompt configuration record the policy version, tolerances, initial reference, and the three prompt templates. Every continuation attempt logs a `benchmark.actor_turn_continuation` event containing the check outcome, per-arm errors, initial and current observation references, `continuation_kind`, and the exact prompt selected. Note that this event records the initiation of a continuation turn; verify subsequent session messages to confirm upstream delivery. Under state-aware `hook` mode, the static `continuation_prompt` field is `null`—inspect `continuation_prompts` and the per-turn events instead.

For example, launching a benchmark with the default state-aware `hook` requires no extra flags:

```bash
mamba run -n Auriga python -m auriga.apps.gateway --benchmark --task arrange_largest_number_random \
  --run-dir runs/arrangement-home-check
```

Pass `--completion-guidance none` to disable home-pose checking, or `--completion-guidance both` to also include the initial prompt reminder. Queue manifest cases inherit the unified configuration (`hook` by default) when the field is omitted, and can override it per case with `"completion_guidance": "none"` or `"both"`.

All existing CLI flags remain supported alongside their negated counterparts, including `--no-default-task-notes`, `--no-vision-compare`, `--no-enable-wait`, `--no-pi05`, `--no-tracking-guard`, `--no-joint-replan`, `--no-motion-review`, `--no-terminal`, `--no-record-video`, and `--no-export-video`. `--no-visual-review-model` disables an auxiliary review model enabled in a config file. Passing `--disabled-tools robot_wait vision_review` replaces the disabled tool list, while passing `--disabled-tools` with no arguments clears the list (it never force-enables tools that are unavailable due to feature flags or observation profiles).

## Current Actor Tools and Availability Controls

The active tool catalog is generated from public `ToolSpec` declarations in `ToolRegistry` and filtered by the resolved configuration, observation profile, and backend capabilities (`auriga.runtime.harness.TOOLS` is retained only as a legacy compatibility export and does not represent the live catalog). The table below lists the 20 built-in tool declarations plus the 2 optional view tools; optional tools are omitted from the catalog when disabled. Any tool can be disabled via `features.disabled_tools`, and configuration flags never bypass role checks, observation permissions, or physical action admission.

| Tool | Baseline Availability / Prerequisites | Purpose |
| --- | --- | --- |
| `task_plan` | Enabled by default | Submit an initial task plan before the first physical action, or revise and read the active plan. |
| `robot_observe` | Enabled by default | Capture current permitted camera images and synchronized measured robot state. |
| `robot_capabilities` | Enabled by default | Query available capabilities, units, coordinate frames, TCP definitions, and budgets. |
| `robot_information` | Enabled by default | Read startup documentation, native URDF, fixed physical specifications, and permitted calibration. |
| `robot_contact_overlay` | Calibrated observation profile | Render robot reference geometry overlaid onto permitted RGB images. |
| `robot_action` | Calibrated observation profile, `resampled` mode, backend supporting unified actions | Preview multi-stage motion batches, execute cached plans, and inspect trajectories across single or dual arms with observation checkpoints. |
| `robot_action_status` | Enabled by default | Query the execution status of a previously submitted action without re-executing it. |
| `robot_wait` | `features.enable_wait=true` | Hold dual-arm joint and gripper targets while explicitly advancing simulation steps. |
| `robot_policy_step` | `features.pi05=true` and matching policy service running | Execute a bounded $\pi_{0.5}$ joint-space policy prefix for a local sub-goal. |
| `history_list_actions` | Enabled by default | Paginate through immutable historical action requests and execution receipts. |
| `history_view_observation` | Enabled by default | Inspect historical camera images and robot states with optional camera selection, ROI cropping, and zoom. |
| `history_read_trace` | Enabled by default | Read permitted planned, commanded, and measured joint/pose trajectories. |
| `history_annotate_action` | Enabled by default | Append actor diagnostic notes to a historical action without mutating measurements or state. |
| `geometry_query` | Calibrated observation profile (depth queries also require an RGB-D profile) | Compute pixel rays, 3D-to-2D projections, and permitted numerical geometry queries. |
| `robotic_view` | `features.robotic_view=true`, calibrated profile, and wrist cameras with published fixed mount chains | Read-only solver converting a desired camera viewpoint into the corresponding `link6` pose. |
| `stereo_guide` | `features.stereo_guide=true` and two zero-distortion pinhole cameras in the same formal observation | Render a source pixel and its corresponding epipolar line on a target camera view without automatic stereo matching. |
| `depth_query` | Explicit RGB-D observation profile | Query numeric depth measurements and candidate geometric hypotheses without executing physical actions. |
| `program_run` | Controlled program runner service | Execute bounded Python programs whose physical actions require primary-model approval by default. |
| `program_status` | Controlled program runner service | Read program execution status, historical source code, and action outcomes. |
| `vision_compare` | `features.vision_compare=true` | Compute pixel-level feature motion between two permitted RGB observations. |
| `vision_review` | `features.visual_review_model` and auxiliary review service enabled | Optional auxiliary visual or operation evidence review that preserves uncertainty and grants no action authority. |
| `operation_memory` | Enabled by default | Query or append evidence-attributed operational lessons scoped to the current task or shared across tasks. |

`robot_action` previews a motion batch and executes the exact cached plan by default; `features.motion_review` governs legacy single-arm adapters and does not disable the unified batch interface on its own. Legacy action names, queues, templates, skills, `scene_*`, `operation_verify`, and `depth_move` were removed in the 2026-10 refactoring and do not appear in actor or program catalogs. Legacy Python adapters and historical feature flags continue to be interpreted under their compatibility contracts; see [Unified Actions and Program Approval](unified-actions.md) and [Tool Plugins](tool-plugins.md).

Synchronized gripper actuation within a `robot_action` `kind="move"` stage is specified per request via `gripper_goal_normalized` and an optional `gripper_motion={"start_fraction": 0.2, "end_fraction": 0.8}` window satisfying `0 <= start_fraction < end_fraction <= 1`. When `gripper_motion` is omitted, the gripper command is issued after the arm motion reaches its stage target; standalone gripper actuation uses `kind="gripper"` with `goal_normalized`. These are per-stage request parameters rather than global configuration flags. Execution follows the cached trajectory, and command completion alone never proves physical contact, stable grasp, or task success.

### Optional View Tools

`robotic_view` and `stereo_guide` are independent boolean flags that both default to `false`. Either tool can be enabled on its own:

```json
{
  "version": 1,
  "features": {
    "robotic_view": true,
    "stereo_guide": false
  }
}
```

Gateway CLI flags `--robotic-view` / `--no-robotic-view` and `--stereo-guide` / `--no-stereo-guide` override the configuration file. Individual cases in a queue manifest can explicitly override `"robotic_view": true` or `"stereo_guide": false` (inheriting the unified configuration when omitted; neither field accepts strings or `null`). Session resumption and replay honor the archived flags, treating missing fields in older archives as `false`.

When disabled, the corresponding plugin is not registered and neither the tool nor its documentation is delivered to the actor or programs. Even when enabled via feature flags, passing `--disabled-tools robotic_view stereo_guide` can still suppress them by name. Tool documentation is delivered exclusively through each tool's `ToolSpec` without modifying global task prompts or Task Notes, and neither tool moves the robot, captures new observations, or infers task success. For parameter schemas and usage examples, see [Wrist View Solver](robotic-view.md) and [Stereo Epipolar Guide](stereo-guide.md).

## Agent Backend (`agent`)

The optional `agent` section selects the agent runtime that drives the actor. `agent.backend` is `codex` (default), `claude_code` or `opencode`; `--agent-backend` overrides it for one run, and every other `agent.*` field is config-file only. While `agent` equals its default it is omitted from exported snapshots, `effective_config.json` and manifests, so Codex runs stay byte-identical and old archives read as `codex`. `model.name`, `model.effort`, `model.price_card` and `model.transport_recovery` apply to every backend; `model.codex_home` and `model.native_memories` are Codex-only.

| Field | Default | Meaning |
|---|---|---|
| `agent.backend` | `codex` | Agent runtime. |
| `agent.claude_code.cli_path` | `null` | Absolute Claude Code CLI path; `null` = the CLI bundled with `claude-agent-sdk==0.2.180` (optional extra `auriga-harness[claude]`). |
| `agent.claude_code.upstream` | Anthropic | Upstream model endpoint (see below); defaults `anthropic_messages`, `https://api.anthropic.com`, `ANTHROPIC_API_KEY`, `x-api-key`, `anthropic`. |
| `agent.claude_code.context_delivery` / `agent.opencode.context_delivery` | `full` | `full` = complete task context each turn (sets `Harness.context_epoch_volatile`, so no image or static metadata delivered earlier is assumed present); `delta` = only what changed since the last delivery. |
| `agent.claude_code.subagents` / `agent.opencode.subagents` | `true` / `true` | Allow the native subagent tool. |
| `agent.claude_code.auto_compact_window_tokens` | `null` | Token window (100000..1000000) at which native auto-compaction triggers; `null` = CLI default. |
| `agent.opencode.executable` | `null` | Absolute path of the platform binary of the pinned `opencode-ai` release. |
| `agent.opencode.upstream` | unset | Upstream model endpoint; must supply `wire_api`, `base_url`, `api_key_env` (or `source: "codex_home"`). |
| `agent.opencode.context_limit_tokens` / `output_limit_tokens` | `200000` / `32000` | Model context window and per-call generation limit, tokens. |
| `agent.evidence.client` | `native` | Client of the evidence helpers (`features.visual_review_model`, `features.self_briefing`). `native` = the backend's own helper where it has one (Codex: `CodexEvidenceAgent`, a fresh ephemeral app-server thread); backends without one use the relay client. `relay` = `RelayEvidenceClient` on every backend. |
| `agent.evidence.upstream` | `null` | Upstream (same fields as `upstream` below) for `features.visual_review_model` requests, bound on the run's relay as `"evidence"`; `null` = the actor's upstream. Relay client only; `wire_api` must be `openai_responses` or `anthropic_messages` (`source: "codex_home"` implies `openai_responses`). |

### Evidence helpers (`agent.evidence`)

`RelayEvidenceClient` ([`auriga/cortex/agents/evidence.py`](../../auriga/cortex/agents/evidence.py)) answers one evidence packet with one fresh, streamed, tool-free request through the run's own model relay: no agent process, home directory, history or memory. It takes the same keywords and `review_evidence(...)` call as `CodexEvidenceAgent` and raises the same error strings, so `VisualReviewer` and self-briefing run unchanged and keep validating the answer schema themselves. Because the request goes through the relay, it uses the shared 180-call budget (charged only for a delivered usable answer), attribution (`source` is `sol_visual_review` or `self_briefing`, the thread is `evidence-<request_id>`), retries, the transport-recovery gate, the blocked-transport gate and the egress scrub. Self-briefing always asks the actor's model on the actor's upstream. Visual review asks `features.visual_review_model` on the `"evidence"` binding when `agent.evidence.upstream` is set, else on the actor's upstream; a non-Codex backend whose actor wire is not `openai_responses` refuses to launch with `visual_review_model` unless `agent.evidence.upstream` is set (`evidence_upstream_required_for_visual_review_model`). Before the first benchmark that enables an evidence feature on a new (upstream, model) pair, the user must authorise one live probe that the provider accepts the structured-output request; until then a rejection is an ordinary recorded failure (visual review raises to the tool, self-briefing is skipped), never a silent fallback.

Fairness rule: the default keeps Codex on `CodexEvidenceAgent`, so existing Codex experiments stay comparable with their history. A harness comparison whose arms enable `self_briefing` or `visual_review_model` must set `agent.evidence.client="relay"` on its Codex arm. The effective client is recorded as manifest `evidence_client` (only written when an evidence feature is on) and `agent_backend.evidence_client`; the queue writes a `comparison_evidence_path_mismatch` warning to `state.json` (`warnings`) and `events.jsonl` (`queue_warning`) when one manifest's evidence-enabled cases use different clients. Switching Codex from native to relay is its own single-variable change: the relay request carries no Codex harness context and names its format `evidence_answer` (Codex sends `codex_output_schema`, `text.verbosity` and `reasoning.context`). The default `agent.evidence` section is omitted from snapshots, and a frozen checkout without `EvidenceConfig` rejects a non-default one with `unified_config_incompatible:agent.evidence`.

`upstream` fields: `source` (`explicit`, `codex_home`, or `claude_settings` for Claude Code: endpoint, key and forwarded provider variables from the official settings file named by `settings`, see [Agent Backends](agent-backends.md#configuration-per-backend)), `wire_api` (`anthropic_messages`, `openai_responses`, `openai_chat`), `base_url` (http(s) URL without credentials, query or fragment), `api_key_env` (the NAME of the environment variable holding the key; the value is never stored in the config, manifests or queue records), `auth_scheme` (`bearer` or `x-api-key`) and `provider` (a label for price cards and resume identity). In the queue, a case may set `agent_backend`, `model` and `effort`; a frozen checkout whose config schema has no `AgentConfig` rejects `agent_backend` cases with `unified_config_incompatible:agent` rather than running them as Codex. Experiment-library group conditions and report conditions expose `agent_backend` (`codex` for archives that predate the section). Mixed-backend comparisons (Codex, Claude Code and OpenCode cases in one manifest, e.g. one `experiment_condition` per backend) record the backend as an explicit condition at every level; the additional fields are listed in the table below.

`claude_code` launch gate and runtime (`cortex/agents/factory.py`, `cortex/agents/claude_code/session.py`): besides the shared refusals (`features.lean_projection`; `features.visual_review_model` without a usable evidence upstream; `model.effort` must be `low`, `medium`, `high`, `xhigh` or `max`; `features.sandbox_enabled=true` is accepted, see [Code Execution Sandbox](#code-execution-sandbox)), the launch is refused with `claude_code_upstream_requires_anthropic_messages` unless `agent.claude_code.upstream` is an explicit `anthropic_messages` endpoint with `base_url` and `api_key_env`, and with `agent_backend_unavailable:claude_code: <reason>; install auriga-harness[claude]` unless `claude-agent-sdk==0.2.180` is installed and the CLI (bundled, or `cli_path`) reports `2.1.321` (one `--version` run in an empty temporary home). At start the gateway reads the key from the named environment variable (`claude_code_upstream_key_missing` when unset); the CLI itself only sees a placeholder key and the run-local relay and MCP endpoints. A resumed run is gated against its archived configuration (not the invocation's) and must match the recorded `runtime/agent` identity (backend, session id, isolation, tool digest, model, provider, counting, wire, SDK and CLI versions) and still hold the CLI session history under `private/claude-config`; a resume after an unclean exit invalidates the model context epoch and schedules the decision reminder. `apps/evaluation/checkpoint.py` and `turn_probe.py` refuse non-Codex archives (`checkpoint_requires_codex_archive:<backend>`); a replay viewer of a Claude Code archive needs no runtime. The session API can apply model changes before the next prompt (the CLI first sends one charged `max_tokens: 1` validation request through the relay; a refused change keeps the previous model); effort changes apply from the next session start. A native turn the CLI starts by itself while the Auriga turn is over (a background shell or monitor finishing, a scheduled wake-up) is adopted as an Auriga turn (`agent.turn_started {origin: "native_wakeup"}`, `capabilities.native_wakeup`); its tool calls carry that turn id, a `run_turn` during it joins it, and the gateway `prompt` command waits for it to end like any running turn. Registry tool calls outside a running turn are refused, and with `agent.claude_code.subagents` enabled the session always uses `context_delivery=full` (recorded as `agent_backend.context_delivery_policy`).

`opencode` launch gate and runtime (`cortex/agents/factory.py`, `cortex/agents/opencode/session.py`): besides the shared refusals, the launch is refused with `opencode_upstream_wire_unsupported:openai_chat` (the relay serves OpenCode only on `anthropic_messages` and `openai_responses`), `opencode_upstream_incomplete` unless an explicit upstream names `wire_api`, `base_url` and `api_key_env` (or uses `source: "codex_home"`, which implies `openai_responses` and reuses the Codex credential source of `model.codex_home`), and `agent_backend_unavailable:opencode: <reason>` unless `agent.opencode.executable` is an absolute executable whose `--version` is `1.19.5`. At start the session also requires `GET /global/health` to report `1.19.5`, the `GET /doc` route list to equal the checked-in `opencode_routes.json`, and `rg` on the host, and records the `build` agent's effective native tool ids read back from the server (`agent_backend.native_tools.enabled`); `opencode serve` runs on a free loopback port with a per-run password, its own private XDG directories and read-only config directories. A turn is admitted when `prompt_async` answers 204 (a 4xx restores the previous turn; a 5xx, timeout or dropped connection is unknown admission and disconnects, never retried) and ends from `GET /session/status` plus `GET /session/{id}/message` snapshots, never from `/event` alone (the stream has no replay; every gap is reconciled, and a watchdog reconciles every 10 s while a turn runs); a session that stays idle without a terminal answer fails the turn with `opencode_turn_incomplete`. Operator steers join at the next step boundary. An interrupt (also with no host turn while the main session is busy, and after an unknown admission while the server lives) first blocks the relay for the main session and every known subagent session (cancels in-flight calls and denies new ones until the next turn: after an abort during a running `task` the main agent still makes one more model call, which must not be forwarded or charged), then aborts the main session and every subagent session of the turn. `model.effort` is the reasoning effort on both wires: the generated relay model declares one `variant` per tier and every prompt and steer selects `model.effort` (OpenCode passes the variant on to subagents and to compaction). `anthropic_messages` accepts `low`, `medium`, `high`, `xhigh`, `max` (request body `output_config.effort` plus `thinking: {type: adaptive, display: summarized}`); `openai_responses` accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh` (`reasoning: {effort, summary: auto}` plus `include: [reasoning.encrypted_content]`). Another value refuses the launch with `agent_effort_unsupported:opencode:<effort>`, and a server that reports other variants for `relay/<model>` than the generated ones refuses the start (`opencode_variants_mismatch`). The mapping is the identity (no per-model table): a tier the model rejects fails at the provider. `set_effort` applies from the next prompt. Provenance: `agent_backend.native_effort = {variant, applied, channel: prompt_variant, provider_options, declared_variants, wire_field}` and `output_limit_tokens` (OpenCode sends it as `max_tokens` / `max_output_tokens`; Claude Code sends up to 128000 on its own, a recorded harness difference). The model is fixed for the run (`set_model` to another model is refused with `set_model_unsupported_by_backend:opencode`). With `agent.opencode.subagents=true` (the default), `context_delivery` is forced to `full` (design R4.3) and child sessions are followed: their transcript rows carry the child session id and `parent_item_id` = the `task` tool part, and are never final-answer text. Resume requires the recorded identity (backend, session id, isolation, tool digest, model, provider, counting, wire, CLI version) and the session in the run's OpenCode database; an unclean previous exit invalidates the model context epoch. A disconnect also blocks the relay for those sessions, a steer after an interrupt is refused (`turn_interrupt_requested`), and an interrupt that lands before OpenCode started the loop is confirmed by a second idle (a loop that starts later is aborted again). Close aborts whenever the server lives (a busy main session makes the close unclean), stops the server's process tree including every detached bash-tool group seen while it ran (re-snapshotted every 2 s), and verifies that none survives; after a server crash the teardown is reported unconfirmed.

#### Mixed-backend experiment fields (report and library)

Per-run fields from `python -m auriga.apps.evaluation.report` (`conditions` and the run row) and per-group fields from the experiment library. A missing record is `null` (unknown), never a default guess, except where the old-archive default is stated.

| Field | Where | Meaning and unit | Source / old-archive default |
|---|---|---|---|
| `conditions.agent_backend` | report, library | Backend kind: `codex`, `claude_code`, `opencode`. | Manifest `agent_backend.kind`; library also `auriga_config.agent.backend` and the per-case override, and the runs' recorded kind wins when it differs (warning `recorded_agent_backend_differs_from_configured`; disagreeing attempts give `null`, warning `recorded_agent_backend_differs_between_attempts`). Archives without the block: `codex`. |
| `conditions.agent_cli` | report | CLI version string only (no path, no hash). | `agent_backend.cli.version`; Codex falls back to manifest `codex_cli`. |
| `conditions.evidence_client` | report, library | Effective client of the evidence helpers: `codex_agent` or `relay` (see the fairness rule above). Part of the comparison condition: arms with an evidence feature enabled must share it. | `agent_backend.evidence_client`; library derives it from the final backend and `agent.evidence.client`. Pre-backend Codex archives: `codex_agent`; other backends without a record: `null`. |
| `conditions.native_effort` | report (object), library (canonical JSON string) | The effort the backend actually applied (`{"effort": ...}` Claude Code, `{"variant": ...}` OpenCode, `{"model_reasoning_effort": ...}` Codex), next to the shared `model.effort` knob. | `agent_backend.native_effort`; `null` when not recorded. |
| `conditions.counting_basis` | report | Comparison basis of the model-call counting rule. Codex records `delivered_usable_actor_helper_compact_responses_v2`, the other backends `delivered_usable_generation_completions_v1`; `COUNTING_EQUIVALENCE` (`cortex/session/usage.py`) maps both to `all_delivered_usable_generation_completions`, so they compare as one basis. An undeclared string is its own basis. | `agent_backend.counting`, else `model_requests.counting`. |
| `counting` | report row | The raw recorded counting string (shown beside the basis). | `summary.model_requests.counting`. |
| `served_models` | `summary.json`, report row | `{model name reported by the provider: delivered completions}` (unitless counts); a response without a model name counts under `unknown`. Shown next to the configured `conditions.model`; no equality is claimed (provider aliasing). | Written by the Codex session and the MCP backends into `runtime/model_requests.served_models`; report falls back to it, and for old Codex archives derives it from `model.*` events. |
| `model_usage_relay` | `summary.json`, report row | Token counters (`input_tokens`, `cached_input_tokens`, `cache_write_input_tokens`, `output_tokens`, `reasoning_output_tokens`, `total_tokens`; tokens) summed over every delivered completion seen by the relay, plus `calls` and `unknown_calls`. A total is `null` as soon as one call lacks that counter. The same measurement for every backend; Codex uses the `account_responses_usage` field mapping. | `runtime/model_usage_relay`. Codex runs resumed from before this record keep their earlier calls as unknown. |
| `model_usage_relay_source` | report row | `recorded` or `derived_from_model_events` (old Codex archives, same field mapping), `null` when none. | |
| `usage_methods` | report row | Per counter `{native, relay, delta}` where `native` is the backend's own cumulative usage (`token_usage`) and `delta = relay - native` (tokens, `null` when either is unknown). Two measurement methods side by side; never merged. | |
| `agent_turn_status` | `summary.json` | Terminal status of the last agent turn for every backend; `codex_turn_status` keeps the same value for compatibility. | |

For Claude Code and OpenCode, `token_accounting.source` is `relay.<wire>.usage.sum` and the report keeps that label; price cards without cache-write pricing give partial estimates there (`cache_write_pricing_unknown`).

## Machine Paths and Image Pipeline

| Configuration Field | Legacy Env Var / Default | Description |
| --- | --- | --- |
| `backend.root` | `AURIGA_ROBODOJO_ROOT` / `~/code/robodojo` | Standalone RoboDojo checkout directory. |
| `backend.environment` | `robodojo` | Explicit environment constructor identifier; unknown environments abort startup. |
| `backend.robot_model_id` | `arx-x5` | Supported robot binding for the selected environment (RoboDojo currently accepts only `arx-x5`). |
| `backend.python` | `AURIGA_ROBODOJO_PYTHON` / `~/miniforge3/envs/RoboDojo/bin/python` | Python interpreter for the Isaac Sim robot worker. |
| `policy.python` | `AURIGA_PI05_PYTHON` / `~/.local/share/auriga-openpi-arx/bin/python` | Dedicated Python interpreter for the OpenPI policy worker. |
| `policy.checkpoint` | `AURIGA_PI05_CHECKPOINT` | Official $\pi_{0.5}$ checkpoint weights matched to the ARX X5 (full default path in `configs/auriga.example.json`). |
| `policy.server_url` | `AURIGA_PI05_SERVER_URL` / `null` | Optional URL of a pre-launched policy inference server to reuse. |
| `features.grasp_memory_dir` | `AURIGA_GRASP_MEMORY_DIR` | Directory for cross-run grasp memory storage. |
| `model.codex_home` | `CODEX_HOME` | Source directory for native Codex provider routing and credentials. |
| `model.price_card` | `AURIGA_PRICE_CARD` | Optional pricing card JSON path for token cost estimation. |

| `images` Field | Default | Constraints and Behavior |
| --- | --- | --- |
| `transport_format` | `webp` | `webp` or `jpeg`; automatically falls back to `jpeg` if WebP encoding is unavailable. |
| `webp_quality` | `85` | Integer in `1`–`100`. |
| `webp_method` | `4` | Integer compression effort in `0`–`6`. |
| `jpeg_quality` | `95` | Model transport JPEG quality, integer in `1`–`100`. |
| `storage_jpeg_quality` | `95` | Archived RGB artifact JPEG quality, integer in `1`–`100`. |
| `feedback_policy` | `all` | `all` returns all camera views; `adaptive` deduplicates routine action and template feedback views that are unchanged relative to the last image delivered to the model, while preserving all views on the initial observation, after context compaction/session switches, on explicit `robot_observe` calls, on failures, and on critical handoff checks (overridable via `--image-feedback-policy`). |
| `reuse_exact_images` | `true` | Reuses byte-identical images within a single outbound model request without merging distinct `detail` levels or altering compaction requests. |
| `trim_max_inline_images` | `60` | When `trim_oversized_requests` is enabled, outbound requests whose inline image count exceeds this threshold are trimmed using the same "keep head and tail, replace middle" window down to roughly $2/3$ of the cap (40 images by default), staying below typical 64-image provider limits. Set `0` to disable count-based trimming; non-zero values must be at least `24`. |
| `trim_oversized_requests` | `true` | When an outbound `/responses` request body exceeds 4.5 MB, `ModelRequestProxy` retains the initial and most recent images while replacing middle-turn images with text placeholders carrying their `observation_id` (triggering at most once per native compaction cycle), and trims further before retrying if the upstream returns `image_compression_failed`. Setting `false` forwards the full Codex history verbatim without retrying `image_compression_failed`. Operates independently of `features.lean_projection`; see [Image Pipeline](image-pipeline.md). |

Legacy `configs/images.json`, `AURIGA_IMAGE_CONFIG`, and `AURIGA_IMAGE_SETTINGS` remain supported as compatibility inputs, with the `images` section of the unified configuration taking precedence. Camera resolution, camera ordering, simulation sampling rates, and action timing are fixed protocol properties rather than image compression settings, so pixel coordinate contracts remain invariant.

## Evaluation Queue, Experiment Library, and Standalone Workflows

`scripts/benchmark_queue.py --config ...` uses the same unified configuration file for both queue scheduling and per-case gateway runs. The queue manifest still defines the list of cases to run, and explicit per-case fields in the manifest can override specific experimental conditions while omitted flags inherit from the unified configuration. At startup, the queue writes `effective_config.json` for auditing and OOM retries so subsequent edits to configuration files on disk never alter an active queue's conditions. Similarly, `scripts/benchmark_group.py --config ...` embeds the resolved configuration directly into the generated queue manifest.

Each manifest case may include an optional non-empty string `experiment_condition` to label its experimental group; this field is preserved in the queue manifest and `state.json` strictly for experiment library grouping and is never passed to gateway CLI flags, runtime configs, or model context.

During preflight, the queue statically parses `auriga/core/config.py` in each target frozen runtime checkout without executing target code. Audited backward-compatible differences—where an older frozen checkout predates newly added configuration fields (`features.sandbox_enabled`, `robotic_view`, `stereo_guide`, `lean_compact`, motion threshold fields, external perception fields, and the `analysis` section)—are handled automatically: when those fields equal their current class defaults (such as `sandbox_enabled=false`), the per-case launch snapshot omits the unknown keys and records the projection in `manifest.config_compatibility` inside `config.json`. If any new field is set to a non-default value or an unaudited schema difference is detected, the queue rejects the manifest before GPU admission. The queue manifest retains the complete unified configuration snapshot, and OOM recovery validates the exact same compatibility projection.

| Configuration Field | Default | Purpose |
| --- | --- | --- |
| `queue.max_parallel` | `60` | Maximum concurrent cases for new queues (an explicit `null` also resolves to `60`, while an explicit ceiling in the queue manifest takes precedence). |
| `queue.memory_per_case_mib` | `11264` | Per-case GPU memory reservation (`11 GiB`). |
| `queue.gpu_headroom_mib` | `0` | Additional unallocated GPU memory headroom required per device. |
| `queue.wait_timeout_s` | `1800` | Timeout in seconds when waiting for eligible GPU resources. |
| `queue.oom_retry.max_retries` | `2` | Automatic fresh-episode retries after a confirmed native GPU OOM (`0` disables retries). |
| `queue.oom_retry.initial_delay_s` | `60` | Cooldown delay in seconds before the first OOM retry. |
| `queue.oom_retry.backoff_multiplier` | `2` | Exponential backoff multiplier between subsequent OOM retries. |
| `queue.oom_retry.max_delay_s` | `300` | Maximum cooldown delay in seconds between OOM retries. |
| `queue.wake_enabled` | `true` | Enables or disables the queue completion notification hook. |
| `queue.wake_thread` / `wake_codex` / `wake_remote` | `null` / `codex` / `null` | Target session UUID, Codex executable path, and app-server socket endpoint for completion notifications; when `wake_thread` is `null`, falls back to `~/.config/auriga/completion-thread` if present, and `wake_remote` defaults to the user's shared control socket (see [Experiment Completion Hook](experiment-completion-hook.md)). |
| `library.root` | `~/exp/auriga` | Root directory scanned by the persistent Experiment Library service. |
| `library.port` | `8780` | Loopback HTTP port for the Experiment Library service. |
| `library.scan_interval` | `1` | Background directory scan interval in seconds. |
| `library.public_origin` | `null` | Exact HTTPS origin permitted when deployed behind an authenticating reverse proxy. |
| `library.task_source_root` | `null` | Optional directory path from which task scoring source code is read for review panels. |
| `scene.root` | `runs/local_scene` | Data directory for the standalone scene service (the embedded gateway service always uses `<run>/twin-service`). |
| `scene.port` | `8770` | Loopback HTTP port for the standalone scene service. |
| `scene.max_workers` | `2` | Maximum concurrent bounded CPU worker jobs in the scene service. |
| `scene.max_queue` | `4` | Maximum queued jobs in the scene service. |
| `scene.max_blob_store_bytes` | `536870912` | Total blob storage budget for the scene service (default `512 MiB`). |

The legacy experiment settings file pointed to by `AURIGA_EXPERIMENT_CONFIG` and existing queue manifest / case fields remain supported. New queues require every `case.cwd` to point to a checkout that supports unified configuration; frozen historical checkouts lacking `--config` fail preflight with `unified_config_requires_supported_gateway` before GPU admission or output directory creation. OOM recovery for legacy queues reuses their recorded launch commands without requiring checkout upgrades. Official benchmark budgets cannot be overridden by the queue, and OOM retries trigger only when native OOM evidence and process cleanup are both confirmed—writing each retry to a fresh directory while preserving all failed attempts.

You can dynamically update the concurrency ceiling of a running queue:

```bash
mamba run -n Auriga python scripts/benchmark_queue.py --output /path/to/running-queue --set-max-parallel 15
```

This command atomically updates `control.json` (`{"max_parallel": 15}`) inside the target queue directory without reading current configuration files or launching tasks; it accepts only positive integers and rejects terminated queues or legacy queues that predate `control.json`. The scheduler reads `control.json` before every poll cycle and before every case admission: raising `max_parallel` triggers an immediate admission check on the next loop iteration without waiting for the 90-second GPU sampling interval, whereas lowering `max_parallel` never kills active cases and simply pauses backfilling until active runs drop below the new ceiling. GPU memory reservations and shared lease rules still apply. Invalid or unreadable `control.json` files are ignored while retaining the previous limit and logging `queue_control_invalid` once per distinct consecutive error.

The CLI prints `requested` once `control.json` is written; confirm actual application via `state.json.scheduling.max_parallel` and the `max_parallel_changed` event in `events.jsonl` (which records the old value, new value, and active run count at the time of change). The initial `manifest.json`, `config.json`, and `effective_config.json` remain immutable startup snapshots while runtime scheduling adjustments are tracked in `state.json` and `events.jsonl`. Queue processes started from older code require restarting under the current entry point to support live `control.json` updates.

`scripts/motion_review_eval.py prepare --config ...` inherits the model, reasoning effort, provider home, machine paths, and image settings from the unified configuration, freezing the resolved configuration into the paired evaluation protocol and generated queue manifests; summary aggregation verifies the saved configuration rather than reading current defaults, and previously generated Astra protocols continue to use their recorded model and effort.

In `scripts/reference_episode.py`, both the direct API client and the robot backend share a single resolved `Config` instance; `--config`, `--model`, `--effort`, and `--codex-home` can override settings explicitly, and `--print-config` prints the effective configuration without reading credentials or launching the model or robot. New reference runs default to `~/.codex` with `gpt-6.1-sol` / `high`, sending the model name, `reasoning.effort`, and `service_tier="default"` in each request. Because this standalone client calls the upstream API directly without the Codex app-server, it reads credentials strictly from the selected provider's declared `env_key`, its bearer token configuration, or the API key in the selected home's `auth.json`—it does not support interactive ChatGPT web login tokens and never falls back to `OPENAI_API_KEY` in the environment unless `env_key` explicitly declares it. Missing providers, endpoints, or credentials fail fast before any model request is sent.

The standalone operational and diagnostic commands below are one-shot workflows controlled by their own CLI flags or HTTP payloads rather than global boolean feature flags:

| Workflow | Entry Point and Controls |
| --- | --- |
| Standalone scene client replay | `scene_client` flags (`URL`, `delay`, `duplicate`, `reverse`) are replay test parameters; the scene service itself supports the unified `scene` config section plus `--config` and `--print-config`. |
| Fixed API reference policy | `scripts/reference_episode.py` accepts `--config`, model override flags, `--max-calls` (`1`–`170`), `--action-timeout-s` (`0.1`–`300`), and task/seed/GPU flags under its standalone reference protocol. |
| Native policy diagnostics | `native_policy_validate.py`, `pi05_atomic_handoff.py`, and related probes accept experiment-specific inputs such as `prefix`, `prompt`, `noise seed`, `hold-limit`, `episode`, and `ee-smoke`. |
| Motion review preflight (removed in 2026-10) | `motion_review_preflight.py --output ... --wait-timeout-s 600` used a diagnostic wait timeout independent of the 1,800-second queue default. |
| Geometry validation | `geometry_validate.py` flags (`synthetic`, `raw-recording`, `ROI`, `heldout`) select the dataset and evaluation slice for a single validation run. |
| Observation image compaction | `compact_observation_images.py --apply` explicitly removes verified raw images from a sealed archive and never runs implicitly during episodes. |
| Persistent library service installation | `install_library_service.py --install` installs and enables the user `systemd` unit (printing the unit definition to `stdout` when `--install` is omitted). |
| Observer endpoints | `/v1/observer/record`, native evaluation, hidden scene snapshots, and video export are invoked explicitly by the harness/operator and are never registered as actor tools. |
| Archive import and reporting | `auriga.apps.library.experiments` and `benchmark_report.py` accept one-off input/output paths and filter flags. |

## Resumption, Replay, and Reproducibility

Every run archive saves the complete resolved configuration alongside legacy manifest fields so older analysis scripts continue to work. The saved snapshot captures the full effective runtime conditions rather than just the partial overrides supplied by the user.

When resuming or replaying an existing run, experimental semantics are restored strictly from the archive rather than reading current default files on disk (while local service parameters such as the HTTP listening port can still be changed). Legacy manifests lacking `enable_wait` keep it `false`, those lacking `vision_compare` keep it `true`, those lacking `actor_payload_profile` default to `full_v1`, and those lacking `default_task_notes` keep it `false`. Unified configuration snapshots from before `default_task_notes` or `vision_compare` were introduced likewise restore their respective historical defaults. Task Notes always reuse the exact archived body, source, and version without regenerating or re-appending current default notes. Legacy image settings lacking `reuse_exact_images` keep it `false`, and those lacking `trim_max_inline_images` restore `0` (byte-size trimming only). If an archived model request ledger has no `transport_recovery` record, automatic transport recovery remains disabled on resume; otherwise, it resumes under the recorded policy. Remember that `--resume` creates a fresh simulator episode/generation rather than restoring mid-episode physics state, while preserving cumulative thread budgets and token accounting.

Consequently, changing a default setting in `configs/auriga.json` affects only newly launched runs. To run a controlled paired ablation, create two fresh run directories, save their respective resolved configurations, and verify the actual tool catalog and observation profile in each `manifest.json`—never assume historical runs automatically migrated to newer defaults.

## Fixed Runtime Protocol and Boundaries

The constants and capability boundaries below are intentionally fixed in code rather than exposed as arbitrary JSON configuration toggles, ensuring that editing a config file cannot inadvertently alter benchmark rules or violate the actor information boundary:

| Scope | Fixed Protocol Rule |
| --- | --- |
| Official model budget | `180` delivered valid actor / helper / compaction responses per episode; no hard per-request wall-clock timeout; explicitly retryable upstream API errors are automatically retried with backoff by `Harness` up to `12` times (with SDK-internal retries set to `0`), followed by bounded cooldown resends governed by `model.transport_recovery` (see [Model-Led Operation](model-led-operation.md)); failed or incomplete attempts are accounted separately. |
| Official run supervision | `900`-second environment initialization deadline; during an active run, stops only after `1,800` consecutive seconds with zero model, program, robot, or simulator activity; no whole-episode wall-clock cap on official benchmarks. |
| Decision boundaries | No artificial cap on read-only tool queries between physical actions; completing a model turn does not terminate the native episode. |
| Native actions | Per-task native step limits, task success criteria, and control frequencies are governed by RoboDojo, with a single physical action writer and idempotent receipts. |
| Unified action schema | `robot_action` accepts `1`–`32` stages per batch (up to `16` stages per arm); dual arms execute along a shared timeline with intra-arm ordering, cross-arm dependencies, and `observe` checkpoints enforced by the batch contract. `robot_wait` and policy prefixes execute at most `50` native steps per call, and action timeouts are bounded to `0.1`–`300` seconds. |
| Legacy atomic actions | Legacy single-arm `EEAction` requests support at most `8` sequential waypoints for trusted compatibility callers only and do not restrict `robot_action` batches. |
| Motion review contact diagnostics | Final continuous approach window is within `60 mm` / `20 deg` of the stage goal; placement contact allows up to `3 mm` geometric penetration tolerance (`2 mm` default), reporting numerical risk without blocking execution while preserving raw overlap facts. |
| Tracking guard | Triggers when joint tracking error exceeds `0.5 rad` for `5` consecutive native steps, halting without automatic trajectory modification. |
| Joint replanning | Attempted only when at least `15` seconds of action timeout remain, substituting a candidate trajectory only if it is fully validated and shorter in native steps. |
| Program resource limits and budgets (enforced in both sandbox modes) | `1 GiB` address space, `8 MiB` file size limit, `128 KiB` source code limit; per-program active wall-clock budget defaults to `180` seconds (maximum `300` seconds), excluding wait time for auxiliary model calls and primary-model action approval; no cap on tool call counts; single-threaded numerical libraries and a `5`-second safety margin for action admission. |
| Standalone scene service limits | Maximum `32 MiB` per blob; per-job queue and wall-clock budgets default to `30` seconds each (validated per request), while worker count, queue depth, and total blob storage are configured under `scene`. |
| Native Codex capabilities | Native dynamic tools return text and images directly, with optional local tool search loading registered tools on demand; shell, file operations, and native memories are enabled while web search and external extensions are disabled. OS namespaces and `program_run` Landlock/seccomp isolation are governed by `features.sandbox_enabled`, and native thread admission restricts robot tool calls strictly to the primary actor thread. |
| Autonomous execution | Startup instructions require the primary model to decide plans, parameters, and program action approvals autonomously without asking a human operator; native user-input requests immediately receive empty answers, command/file approvals on the active session automatically return `acceptForSession`, and requested permissions are granted at session scope without human intervention. Harness observation projection and action admission remain enforced at all times, and Landlock is active when `features.sandbox_enabled=true`. |
| Observation permissions | Native depth is forbidden by default and permitted only under the explicit RGB-D profile; hidden object poses, contact ground truth, and private evaluator state are forbidden across all profiles. |
| Video timing | Three-camera recording samples at `5 Hz` in simulation time plus observation boundary frames, preserving the exact non-uniform frame mapping rather than recording at the full native physics rate. |
| Stopping and scoring | Unconfirmed stops, `unknown` action outcomes, and infrastructure failures remain unscored under the scoring policy and can never be configured to fabricate task success or hide failures. |

Direct tool mode is pinned via a per-run private `model_catalog_json` that preserves the native model catalog from Codex SDK/CLI `0.160.0` while setting `tool_mode` to `"direct"` (because native model metadata overrides feature flags, disabling `code_mode` alone is insufficient). Model names, reasoning effort tiers, context windows, and other capabilities remain unchanged; for catalog provenance, checksums, and upgrade procedures, see [`third_party/codex/README.md`](../../third_party/codex/README.md).

If future research requires experimentally varying any of these fixed protocol constants, introduce an explicit versioned configuration field with validation, manifest logging, and regression tests, and clearly separate official benchmark runs from exploratory development runs.

For `--benchmark` and `--exit-after-turn` runs (interactive development workbenches do not auto-exit on idle), runtime supervision measures inactivity from the most recent genuine activity: initiating a model request, receiving real upstream transport bytes, emitting code execution events/output, or advancing robot/simulation execution all reset the 30-minute (`1,800`-second) inactivity timer. Merely having a pending request, synthetic keepalives, runtime status heartbeats, or live child processes does not reset the timer. As long as any subsystem makes genuine progress, the episode is never terminated for total elapsed wall-clock time; if all subsystems remain silent for 1,800 seconds, active requests are cancelled and the run enters the stop and resource-cleanup sequence, recording `runtime_inactivity_timeout` as an unscored infrastructure interruption. Explicitly retryable API errors are retried automatically by `Harness`; once immediate retries are exhausted or an upstream transport disconnect occurs—and provided no physical action is in flight or unconfirmed and no program is active—the proxy cools down and resends the identical request according to `model.transport_recovery`, terminating only on non-retryable errors, rejected recovery gates, or exhausted recovery budgets. See [Model-Led Operation](model-led-operation.md) for details.

Reaching the 900-second environment startup deadline closes action admission and initiates resource cleanup, though it does not guarantee that the entire gateway process will exit within 900 or 905 seconds. [`RuntimeSupervision`](../../auriga/runtime/supervision.py) retains ownership of the startup task while [`Workbench`](../../auriga/apps/gateway/workbench.py) polls published robot worker, SDK bridge, app-server, and terminal subprocess handles every 50 ms (including processes spawned after the stop request was issued), sending `SIGTERM` first and escalating to `SIGKILL` after 5 seconds while recording signal and exit evidence.

In [`RobotProcess`](../../auriga/robot/backends/robodojo/robot_process.py), a single request deadline covers acquiring the write lock, writing to the pipe, sending cancellation, and awaiting the worker response; partial writes or missing receipts remain `unknown` without automatic action replay. After a pipe failure, planned robot close waits total at most 27 seconds, whereas a standard close takes at most 237 seconds across the worst-case branch (covering the close RPC, process exit, and reader thread join). These bounds exclude OS scheduling delays or uninterruptible kernel I/O; Python or native threads that cannot be terminated safely remain owned by their respective objects rather than being falsely logged as released. Enforcing a hard whole-process exit deadline requires an external process supervisor. For CPU subprocess regression tests, see [`test_robot_process_deadline.py`](../../tests/test_robot_process_deadline.py) and [`test_runtime_shutdown.py`](../../tests/test_runtime_shutdown.py).

`features.arm_queue` and `features.motion_cost_preview` are enabled by default and can be overridden via `--[no-]arm-queue` and `--[no-]motion-cost-preview`. Both require `resampled` execution mode, and `motion_cost_preview` additionally requires a calibrated observation profile. Disabling `arm_queue` also rejects raw `ArmQueueAction` calls, and disabling `motion_cost_preview` also rejects corresponding program calls. Queue manifests may specify `eligible_gpu_indices` as a list of unique non-negative GPU indices; when omitted, all visible GPUs remain eligible for automatic admission, and when set, excluded GPUs are never selected while eligible GPUs still obey shared GPU leases and memory checks.

## Bounded Continuation, Execution Checks, and Manipulation Evidence

Four independent boolean flags under `features` default to `true` for new runs, controlled via CLI flags `--[no-]bounded-continuation`, `--[no-]motion-safety-guard`, `--[no-]trajectory-preview`, and `--[no-]manipulation-protocol` (and matching snake_case fields on queue manifest cases). When resuming or replaying legacy manifests that lack these fields, they default to `false` to preserve historical behavior:

- `bounded_continuation`: The first completed `final` answer establishes a baseline; thereafter, if no physical action progress occurs, the episode terminates after `3` consecutive `final` answers with no new tool evidence or after at most `6` recovery `final` turns in total (native episode termination always takes precedence). New read-only tool exploration grants only bounded recovery turns, and the private evaluator score is never consulted to make this decision.
- `motion_safety_guard`: Inspects compiled joint trajectories prior to execution for abnormal joint detours/windup and remaining native step budget; if rejected, returns diagnostic evidence without altering joint commands or auto-replanning while preserving already executed step counts.
- `trajectory_preview`: Attaches candidate joint configurations, public forward-kinematics poses, and synchronized dual-arm model clearance distances to motion cost previews, and enables `robot_action(mode="inspect")` to query trajectory states at specific native steps or project them onto current RGB images. Uses strictly the public robot model, measured joint feedback, public calibration, and candidate commands—never task object ground truth (see [Motion Safety and Trajectory Evidence](motion-safety-and-trajectory-evidence.md)).
- `manipulation_protocol`: Aligns Task Notes, `task_plan`, and bounded templates around explicit contact locations, placement geometry, and observation evidence. Unknown geometry may be recorded as `unknown`, and completed gripper releases are labeled `released_unverified` until verified by observation. Bimanual handoff permissions and preferences remain governed by `airborne_handoff` and `handoff_preference`, with full dual-arm parallel execution preserved.
