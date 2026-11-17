# Documentation Index

If you are new to Auriga, start with the main [README](../README.md) ([English](../README.md) / [简体中文](../README.zh-CN.md)) and the [Auriga Technical Report (PDF)](Auriga-Technical-Report.pdf).

The tables below organize all documentation by topic. Development rules and architectural boundaries are governed by [AGENTS.md](../AGENTS.md), while the authoritative runtime tool contracts live in [`auriga/runtime/tools/definitions.py`](../auriga/runtime/tools/definitions.py) and [`auriga/runtime/tools/builtins.py`](../auriga/runtime/tools/builtins.py).

---

## 🚀 Getting Started & Operations

| Document | What It Covers |
| :--- | :--- |
| [Running Guide](development/running.md) | Preparing the environment, starting interactive or benchmark runs, pausing/stopping, resuming sessions, replaying archives, and scheduling batch evaluations. |
| [RoboDojo Environment Setup](development/robodojo-setup.md) | Pinned RoboDojo / IsaacLab commits, required compatibility patches, and how Auriga locates the isolated RoboDojo interpreter. |
| [Configuration Guide](development/configuration.md) | Configuring models, machine paths, the OS sandbox, observation conditions, tool availability, video recording, the agent backend and its [evidence helper client](development/configuration.md#evidence-helpers-agentevidence). |
| [Experiment Library & Web Service](development/experiment-library.md) | Archiving, searching, live monitoring, side-by-side comparison, replay, and the optional `experiment-groups.json` registry format. |
| [HD Offline Replay](development/hd-replay.md) | Re-simulating archived trajectories in RoboDojo and exporting high-definition videos without calling the model. |
| [Experiment Completion Hook](development/experiment-completion-hook.md) | How the evaluation queue notifies callers when a batch finishes. |
| [Benchmark OOM Retries](development/benchmark-oom-retries.md) | When and how the queue schedules a fresh episode retry after a native GPU out-of-memory failure. |
| [Token & Cost Accounting](development/token-accounting.md) | How token usage is metered and converted into cost estimates via pricing cards. |
| [Web Workbench & Validation](../web/README.md) | Building the frontend bundle and validating the terminal, phase timeline, and video replay in headless browsers. |

---

## 🛠 Architecture & Contributor Guides

| Document | Purpose |
| :--- | :--- |
| [AGENTS.md](../AGENTS.md) | Essential reading for developers and coding agents: architectural boundaries, first principles, experiment discipline, and delivery checks. |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Environment setup, CPU/web/package checks, and pull request guidelines. |
| [Implementation Architecture](development/architecture.md) | Six-layer package structure, call paths, state ownership, and module dependency rules. |
| [Agent Backends](development/agent-backends.md) | The swappable agent runtime (Codex / Claude Code / OpenCode): `AgentSession` contract, MCP tool transport, relay wire adapters, native capability policy per backend, comparison protocol for harness experiments, and known limitations; [prompt parity](development/reports/agent-backend-prompt-parity.md) and [spike findings](development/reports/agent-backend-spikes.md). |
| [Claude Code backport validation](development/reports/auriga-claude-backport.md) | Source commits, Auriga compatibility scope, CPU/Web/package results and environment limitations. |
| [Extension Guide](development/extensions.md) | Choosing the right extension layer: operation memory, task notes, plugins, tools, or robot backends. |
| [Tool Contracts & Plugins](development/tool-plugins.md) | Declaring schemas, roles/effects, narrow `ToolContext` access, typed receipts, and dual actor/program regressions. |
| [Tool Output Contract](development/tool-output-contract.md) & [Cost Summary & Input Diagnostics](development/tool-input-output.md) | Structured tool responses, on-demand `help` requests, preview cost summaries, and actionable validation errors. |
| [Robot & Environment Integration](development/robot-environment-integration.md) | Onboarding a new manipulator or simulator/real-world environment and validating supported combinations. |
| [Robot Backend Boundaries](development/robot-backend-boundaries.md) | Separating pure validation, trajectory compilation, IPC transport, and native worker execution. |
| [Validation & Release Guide](development/validation.md) | Selecting test suites by change scope, host-capability gating of sandbox tests, distinguishing software checks from native task evidence, and cutting releases. |
| [Changelog](../CHANGELOG.md) | Version-by-version release notes and compatibility history. |

---

## 🤖 Runtime Interfaces & Manipulation Behavior

| Document | Topic |
| :--- | :--- |
| [Unified Actions & Program Review](development/unified-actions.md) | Task plans, multi-stage `robot_action` batches, inter-arm dependencies, program action review, and operation memory. |
| [Program & Task Result Semantics](development/program-result-semantics.md) | Distinguishing program exit status, physical motion completion, and native evaluator task scores. |
| [Robot Information Interface](development/robot-information.md) | Robot kinematic structure, URDF, native meshes, camera calibration, and physical specifications. |
| [Actor Information Boundary](development/actor-information-boundary.md) | What data the main model, auxiliary models, and helper programs are permitted to read—and how provenance metadata is stripped. |
| [Image Pipeline & Storage](development/image-pipeline.md) | Model-bound images, archive images, compression, deduplication, and video encoding. |
| [Adaptive Camera View Selection](development/adaptive-view-selection.md) | How post-action feedback selects relevant camera views. |
| [Dual-Arm Synchronized Motion](development/dual-arm-motion.md) & [In-Motion Gripper Commands](development/synchronized-gripper.md) | Coordinated dual-arm trajectories, handovers, and synchronized gripper actuation during motion. |
| [Per-Action Motion Speed](development/motion-speed.md) | Semantics, bounds, and defaults of stage speed scaling factors. |
| [Motion Planning Failure Feedback](development/motion-failure-feedback.md) | Planning diagnostics, candidate trajectory feedback, and partial execution receipts. |
| [Joint Travel Admission & Trajectory Evidence](development/motion-safety-and-trajectory-evidence.md) | Principal-branch joint constraints, safety admission checks, and executed trajectory evidence. |
| [Motion Cost & Contact Overlay](development/motion-cost-and-contact-overlay.md) | Preview cost summaries and `robot_contact_overlay` visual projections. |
| [Model-Led Operation](development/model-led-operation.md) | Division of responsibility between model transport, risk reference summaries, and optional visual verification. |
| [Turn Progress Supervision](development/actor-progress.md) | Bounded continuation policy when an actor ends a benchmark turn prematurely. |
| [Wrist Camera Viewpoint Solver](development/robotic-view.md) & [Stereo Epipolar Guide](development/stereo-guide.md) | Two optional geometric tools (disabled by default). |
| [RGB-D Numeric Tools](development/rgbd-numeric-tools.md) | Depth queries, limitations, and configuration under explicit RGB-D observation conditions. |
| [$\pi_{0.5}$ Inference Service](development/pi05-inference-service.md) | Interface and execution boundaries for the optional local policy bridge (disabled by default). |
| [Official Public Task Rules](development/public-task-rules.md) | Verbatim snapshot, prompt injection, and backward compatibility for the 42 official RoboDojo task specifications. |
| [Default Task Notes](development/task-notes.md) | General manipulation notes injected at startup, versioning, and feature toggles. |

---

## 👁 3D Perception & Context Notes

| Document | Topic |
| :--- | :--- |
| [obs3d](obs3d.md) | 3D scene perception attached to default observations. |
| [object_table](object_table.md), [scene3d](scene3d.md), [vggt_height](vggt_height.md) | Structured 3D object table, point-cloud handles (`scene3d_helpers`), and support-plane height estimation via the [VGGT Service](../services/vggt_height/README.md). |
| [held_perception](held_perception.md) | In-hand object state and geometry perception. |
| [pose_skill](pose_skill.md) | Pose targets and grasp geometry references. |
| [lean_context](lean_context.md) | Prompt-cache-preserving history projection (disabled by default). |
| [scene_truth](scene_truth.md) | Observer-only ground-truth recording for offline geometric auditing (never exposed to the model). |

---

## 📐 Historical & Foundational Design Specifications

The documents in `docs/design/` preserve the foundational architecture specifications. Where early design drafts mention standalone motion tools, templates, skills, or `scene_*` tool names, the current implementation has converged onto the unified `robot_action` and program review interface described above.

| Document | Scope |
| :--- | :--- |
| [Embodied Harness Design](design/embodied-harness-design.md) | Core philosophy, division of labor between the model and executor, observation boundaries, programs, and long-term roadmap. |
| [Shared Interface & Event Protocol](design/shared-protocol.md) | Run and action identities, `ObservationBundle`, `EEAction`, event streams, snapshots, and media references. |
| [Web Workbench & Codex Integration](design/web-workbench-and-codex.md) | Coupling native Codex sessions, PTY terminals, multi-camera robot views, and phase-synchronized history inspection. |
| [Local Scene & Short-Horizon Rollout](design/local-scene-and-rollout.md) | Constructing queryable 3D scenes and snapshots from permitted observations (full physical rollout worker remains future work). |

> [!NOTE]
> The live Cortex tool catalogue and input validation are assembled from [`auriga/runtime/tools/builtins.py`](../auriga/runtime/tools/builtins.py) and [`auriga/runtime/tools/definitions.py`](../auriga/runtime/tools/definitions.py). Availability at runtime depends on configuration, observation conditions, backend capabilities, and external service readiness. Models can pass `request={"kind": "help"}` to supported query tools for on-demand field documentation.
