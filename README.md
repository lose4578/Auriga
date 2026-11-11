<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/auriga-lockup-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/auriga-lockup-light.svg" />
  <img src="docs/assets/auriga-lockup-light.svg" alt="Auriga" width="320" />
</picture>

<p><b>An Agent Harness for Embodied Manipulation</b></p>

[![Project Page](https://img.shields.io/badge/Project%20Page-auriga.stelledge.com-0A66C2.svg)](https://auriga.stelledge.com/)
[![Demo](https://img.shields.io/badge/Demo-Videos%20%26%20Rollouts-008080.svg)](https://auriga.stelledge.com/demos.html)
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B.svg)](https://arxiv.org/abs/2611.01112)
[![Papers with Code: RoboDojo](https://paperswithcode.co/api/v1/papers/2611.01112/leaderboard-badge.svg?eval=47675&live=1)](https://paperswithcode.co/api/v1/papers/2611.01112/leaderboard-badge-link?eval=47675)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](README.md) · [简体中文](README.zh-CN.md) · [Tech Report (PDF)](docs/Auriga-Technical-Report.pdf) · [Demo](https://auriga.stelledge.com/demos.html) · [Documentation](docs/README.md) · [Contributor Guide](AGENTS.md) · [Contributing](CONTRIBUTING.md) · [Citation](#-citation)

</div>

---

**Auriga** is an agent harness for embodied robot manipulation. A model runs in a persistent Codex session, looks at camera images and robot state, plans, calls tools, writes and reviews small helper programs, and moves a robot—while the harness owns the execution boundary, the run state, and the evidence record. The default environment is the [RoboDojo](https://robodojo-benchmark.com/) simulation benchmark with a dual-arm **ARX X5**.

Instead of hardcoding task-specific scripts or handing control to an open-loop policy, Auriga keeps **the model in the lead** and treats tools as scene-neutral instruments:
- **Deliberate before moving**: The model localizes objects in 3D via calibrated multi-view geometry and previews multi-stage dual-arm trajectories while the world is paused.
- **Closed-loop execution**: Every physical action goes through a single gateway (`Harness.execute`) that checks stage tolerances and grip guards in real time, pausing immediately on stalls or slips and returning fresh visual + state receipts so the model can recover in place.

<p align="center">
  <img src="docs/assets/auriga-teaser.png" alt="Auriga Episode Overview" width="100%" />
</p>

---

## Architecture

<p align="center">
  <img src="docs/assets/auriga-overview.png" alt="Auriga Architecture and Episode Loop" width="100%" />
</p>

A run is an **observe → decide → act → feedback** loop:
1. Before its first physical move, the model submits a short task plan (`task_plan`).
2. `robot_action(action="preview")` plans a batch of stages (single or dual arm, gripper moves, inter-arm dependencies, and optional checkpoints) without advancing the simulator.
3. `robot_action(action="execute", plan_id=...)` runs the cached plan through [`Harness.execute`](auriga/runtime/harness.py) and monitors arrival tolerances and grip guards.
4. In RoboDojo, simulated physics is paused while the model thinks and only advances through actions or explicit `robot_wait` calls. Task scores come strictly from the native evaluator.

---

## Benchmark Results

Mean Score (0–100) and Success Rate (SR) on the 42-task [RoboDojo](https://robodojo-benchmark.com/) benchmark against representative baselines:

| Method | Type | Average (Score / SR) | Generalization | Precision | Long-Horizon | Memory | Open |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| WALL-OSS-1.0-lite | VLA | 42.6 / 35.0% | 46.7 | 38.2 | 49.9 | 57.4 | 20.8 |
| GO-3 Nano Preview | Agent + WAM | 61.9 / 56.4% | 36.5 | **78.7** | **68.2** | 64.5 | 61.7 |
| RoboProbe *(Ref. Harness)* | Agent | 29.0 / 22.5% | 33.4 | 12.7 | 21.5 | 43.0 | 34.4 |
| HexaShell-Preview | Agent | 53.2 / 46.9% | 39.2 | 41.3 | 42.9 | 84.5 | 58.0 |
| PyRUA-Lean | Agent | 55.3 / 50.4% | 48.3 | 44.8 | 54.3 | 68.4 | 60.6 |
| **Auriga (Ours)** | **Agent** | **64.0 / 57.7%** | **52.9** | 46.3 | 62.1 | **93.4** | **65.4** |

---

## Tools

Ten core tools are visible upfront; the rest load on demand through native Codex tool search. Each tool declares its input/output schema, physical units (`m`, `rad`, `deg`, `s`), defaults, and bounds in [`auriga/runtime/tools/definitions.py`](auriga/runtime/tools/definitions.py).

| Tool | Purpose |
| :--- | :--- |
| `task_plan` | Submit and revise the task plan and attention list; re-injected before each decision. |
| `robot_observe` | Current camera images plus joint / end-effector / gripper state; optional zoomed ROI crops. |
| `robot_information`, `robot_capabilities` | Robot structure, URDF, camera calibration, native geometry, and runtime capabilities. |
| `robot_action` | Preview, execute, and inspect single- or dual-arm stage batches. |
| `robot_wait` | Advance simulation to watch motion or let objects settle. |
| `robot_action_status` | Query the original request and execution receipt of an action. |
| `program_run`, `program_status` | Run closed-loop helper or numeric programs; review the physical actions they request. |
| `geometry_query`, `robot_contact_overlay` | Calibrated 3D geometry (stereo triangulation, pixel-to-plane) and gripper contact overlays. |
| `vision_compare` | Measure local visual changes between two observations. |
| `history_*` | Inspect past actions, trajectories, and observations; attach annotations. |
| `operation_memory` | Query or record operating experience within the episode and across tasks. |
| *Optional*: `robotic_view`, `stereo_guide`, `vision_review`, `depth_query`, `robot_policy_step` | Wrist-camera viewpoint solver, epipolar guide, auxiliary visual review, RGB-D queries (explicit depth condition only), and local $\pi_{0.5}$ policy step. |

See [Unified Actions & Program Review](docs/development/unified-actions.md) and [Tool Contracts & Plugins](docs/development/tool-plugins.md).

---

## Requirements & Installation

### Prerequisites

- **OS & GPU**: Linux `x86_64` and an NVIDIA GPU capable of running Isaac Sim 5.1.
- **Conda / Mamba**: [Miniforge](https://github.com/conda-forge/miniforge) (`mamba`). The project uses a single conda environment named `Auriga` (Python 3.12, see [`environment.yml`](environment.yml)); do not create a `.venv`.
- **Node.js**: Node.js 22 and `npm` to build the Web UI.
- **RoboDojo**: RoboDojo with Isaac Sim, IsaacLab, and task assets, pinned and patched as described in [RoboDojo Environment Setup](docs/development/robodojo-setup.md) (runs in its own interpreter).
- **Codex Home**: `~/.codex` (by default) with credentials for a Responses-compatible model provider. The default model is `gpt-6.1-sol` with reasoning effort `high`; the Codex SDK/CLI version is pinned in `requirements.lock.txt`.
- **Optional VGGT Service**: The [VGGT perception service](services/vggt_height/README.md) (`http://127.0.0.1:8331`) provides 3D object tables, point clouds, and height estimates. If the service is not running, those attachments are marked unavailable and the run continues; set `"external_perception": false` to disable it.

### Install

```bash
git clone https://github.com/lose4578/Auriga.git
cd Auriga

# 1. Create the Auriga conda environment
mamba env create -f environment.yml

# 2. Build the Web Workbench & Experiment Library UI
npm --prefix web ci && npm --prefix web run build

# 3. Run CPU checks (no GPU, simulator, or model needed)
mamba run -n Auriga python -m unittest discover -s tests
```

---

## Configuration

Settings merge in order: [`configs/auriga.json`](configs/auriga.json) → `configs/auriga.local.json` (git-ignored) → `--config <file>` → CLI flags.

Put machine-specific paths (RoboDojo interpreter/checkout, Codex home, model) in `configs/auriga.local.json`:

```json
{
  "version": 1,
  "model": {
    "name": "gpt-6.1-sol",
    "reasoning_effort": "high",
    "codex_home": "~/.codex"
  },
  "features": {
    "sandbox_enabled": false,
    "external_perception": true
  },
  "paths": {
    "runs_dir": "runs",
    "archive_root": "~/exp/auriga",
    "robodojo_python": "/path/to/miniforge3/envs/robodojo/bin/python",
    "robodojo_root": "/path/to/RoboDojo"
  }
}
```

Inspect the effective merged configuration without starting any service:

```bash
mamba run -n Auriga python -m auriga.apps.gateway --print-config
```

See the [Configuration Guide](docs/development/configuration.md) and the [annotated example](configs/auriga.example.json) for all fields.

---

## Quick Start

### 1. Run a Single Episode

Pick a free GPU and a new run directory:

```bash
mamba run -n Auriga python -u -m auriga.apps.gateway \
  --benchmark --task general_pickup --seed 0 --layout-id 0 \
  --gpu 0 --port 8767 --run-dir runs/my-first-task
```

Open **<http://127.0.0.1:8767>** to watch the cameras, native Codex terminal, action previews, and run record.

### 2. Replay, Browse & Render

```bash
# Read-only replay of an archived run directory (no GPU, no model)
mamba run -n Auriga python -m auriga.apps.gateway --replay --run-dir runs/my-first-task --port 8767

# Experiment library: browse runs, comparisons, and replays at http://127.0.0.1:8780
mamba run -n Auriga python -m auriga.apps.library

# Re-simulate an archived trajectory and render HD videos (needs simulator, never calls the model)
mamba run -n Auriga python -m auriga.apps.replay.hd_replay --source-run runs/my-first-task --output runs/my-first-task-hd
```

### 3. Batch Evaluation

Batch evaluation uses a manifest and the queue (`python -m auriga.apps.evaluation.queue`). Fixed benchmark suites in [`benchmarks/`](benchmarks) generate manifests:
- [`gpt-as-policy-10`](benchmarks/gpt-as-policy-10/README.md): 10 tasks × 5 scenes (50 cases).
- [`robodojo-42-15`](benchmarks/robodojo-42-15/README.md): All 42 official tasks × 15 scenes (630 cases).

See the [Running Guide](docs/development/running.md) for session resume, interactive control, batch scheduling, reports, and audits.

---

## Security

Generated code runs as your OS user by default (`features.sandbox_enabled=false`): native Codex commands, sub-agents, and `program_run` have your user's file and network access, and only the harness's application-level boundaries apply (observation filtering, action admission, program review).

On Linux hosts with `bubblewrap`, Landlock ABI ≥ 4, and `libseccomp`, you can enable OS isolation with `{"version": 1, "features": {"sandbox_enabled": true}}`; startup fails explicitly instead of downgrading when isolation is unavailable. See [Code Execution Sandbox](docs/development/configuration.md#code-execution-sandbox). Keep credentials in `~/.codex`, never in the repository.

---

## Repository Layout

```text
auriga/        Python package, six layers; module imports only point downward
  core/          contracts, configuration, information boundary, storage, model-facing presentation
  robot/         robot model and calibration, motion contracts and planners, RoboDojo backend
  perception/    image measurement, overlays, optional external perception, RGB-D, scene store
  cortex/        Codex session and sandbox, actor startup context, turn supervision
  runtime/       Harness, robot_action orchestration, tool registry and plugins, programs, policy bridge
  apps/          gateway/workbench, experiment library, evaluation queue, HD replay, reference baseline
web/           workbench and experiment-library frontend (built into the package)
configs/       default configuration, full example, image settings, presets, pricing cards
benchmarks/    fixed benchmark groups (cases and partitions)
scripts/       packaging, benchmark-group generator, compatibility entry points, probes
services/      optional VGGT 3D perception service
contracts/     protocol fixtures used by tests
tests/         CPU contract and regression tests
third_party/   pinned third-party material with licenses (Codex model catalog, RoboProbe)
docs/          development guides, design documents, and perception notes
```

**Code tour**: [`auriga/apps/gateway/cli.py`](auriga/apps/gateway/cli.py) builds a [`Workbench`](auriga/apps/gateway/workbench.py), which starts the Codex session ([`auriga/cortex/session/codex.py`](auriga/cortex/session/codex.py)) and the [`Harness`](auriga/runtime/harness.py). Tools are declared in [`auriga/runtime/tools/definitions.py`](auriga/runtime/tools/definitions.py) and assembled in [`auriga/runtime/tools/builtins.py`](auriga/runtime/tools/builtins.py); `robot_action` lives in [`auriga/runtime/actions/unified_actions.py`](auriga/runtime/actions/unified_actions.py). Physical execution goes from [`Harness.execute`](auriga/runtime/harness.py) over [`auriga/robot/backends/robodojo/robot_process.py`](auriga/robot/backends/robodojo/robot_process.py) to [`auriga/robot/backends/robodojo/robot_worker.py`](auriga/robot/backends/robodojo/robot_worker.py) inside the RoboDojo interpreter.

---

## Where to Start

| You want to… | Read |
| :--- | :--- |
| Run tasks and benchmarks | [Running Guide](docs/development/running.md), [Configuration](docs/development/configuration.md), [RoboDojo Setup](docs/development/robodojo-setup.md) |
| Understand how a call flows | [Implementation Architecture](docs/development/architecture.md), [Shared Protocol](docs/design/shared-protocol.md) |
| Add or change a tool | [Tool Contracts & Plugins](docs/development/tool-plugins.md), [Extension Guide](docs/development/extensions.md) |
| Add a robot or environment | [Robot & Environment Integration](docs/development/robot-environment-integration.md), [Backend Boundaries](docs/development/robot-backend-boundaries.md) |
| Know what the model may see | [Actor Information Boundary](docs/development/actor-information-boundary.md) |
| Work on 3D perception | [obs3d](docs/obs3d.md), [object_table](docs/object_table.md), [scene3d](docs/scene3d.md), [vggt_height](docs/vggt_height.md) |
| Work on the Web UI | [web/README.md](web/README.md) |
| Browse and replay experiments | [Experiment Library](docs/development/experiment-library.md), [HD Replay](docs/development/hd-replay.md) |
| Contribute | [AGENTS.md](AGENTS.md), [Validation Guide](docs/development/validation.md), [CONTRIBUTING.md](CONTRIBUTING.md) |

Full index: [docs/README.md](docs/README.md).

---

## License & Third-Party Material

Auriga is released under the [MIT License](LICENSE). [`third_party/`](third_party) keeps upstream licenses and provenance for vendored material (Codex model catalog and RoboProbe reference code, both Apache-2.0). RoboDojo, Isaac Sim, IsaacLab, cuRobo, and model weights are not distributed with this repository.

---

## 🌟 Citation

If you use Auriga in your research or engineering work, please cite the repository:

```bibtex
@software{auriga2026code,
  title        = {Auriga: An Agent Harness for Embodied Manipulation},
  author       = {{StellarEdge AI Technical Team}},
  year         = {2026},
  url          = {https://github.com/lose4578/Auriga},
  version      = {0.2.0},
  license      = {MIT}
}
```
