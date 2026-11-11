<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/auriga-lockup-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/auriga-lockup-light.svg" />
  <img src="docs/assets/auriga-lockup-light.svg" alt="Auriga" width="320" />
</picture>

<p><b>面向具身操作的智能体执行框架（Agent Harness for Embodied Manipulation）</b></p>

[![Project Page](https://img.shields.io/badge/Project%20Page-auriga.stelledge.com-0A66C2.svg)](https://auriga.stelledge.com/)
[![Demo](https://img.shields.io/badge/Demo-Videos%20%26%20Rollouts-008080.svg)](https://auriga.stelledge.com/demos.html)
[![Tech Report](https://img.shields.io/badge/Tech%20Report-PDF-B31B1B.svg)](docs/Auriga-Technical-Report.pdf)
[![Papers with Code: RoboDojo](https://paperswithcode.co/api/v1/papers/2611.01112/leaderboard-badge.svg?eval=47675&live=1)](https://paperswithcode.co/api/v1/papers/2611.01112/leaderboard-badge-link?eval=47675)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](environment.yml)
[![Node.js 22](https://img.shields.io/badge/Node.js-22-339933.svg)](web/package.json)

[English](README.md) · [简体中文](README.zh-CN.md) · [技术报告 (PDF)](docs/Auriga-Technical-Report.pdf) · [演示视频 (Demo)](https://auriga.stelledge.com/demos.html) · [文档索引](docs/README.md) · [开发约定](AGENTS.md) · [参与贡献](CONTRIBUTING.md) · [引用](#-引用)

</div>

---

**Auriga** 是面向具身机器人操作的智能体执行框架（agent harness）。模型在持续的 Codex 会话中查看相机图像和机器人状态、制定计划、调用工具、编写并审批小程序、控制机器人；Harness 负责执行边界、运行状态和证据记录。当前默认运行环境是 [RoboDojo](https://robodojo-benchmark.com/) 仿真基准与双臂 **ARX X5**。

不同于为特定任务编写固定脚本或直接交给开环策略，Auriga 的核心思路是**由模型主导决策，由 Harness 提供与具体任务无关的闭环感知与执行边界**：
- **先推演再执行**：模型通过标定多视角几何与三维点云定位目标，在仿真暂停状态下预览双臂多阶段候选轨迹，确认无误后再真正驱动机械臂。
- **单入口闭环与原位恢复**：所有物理动作统一经过 `Harness.execute`，执行中实时校验阶段到位容差与夹爪门禁；一旦发生卡阻或抓空滑脱即刻暂停，并返回最新画面与实测残差回执，让模型在当前状态下直接修正恢复。

<p align="center">
  <img src="docs/assets/auriga-teaser.png" alt="Auriga 单局闭环操作概览" width="100%" />
</p>

---

## 架构与运行流程

<p align="center">
  <img src="docs/assets/auriga-overview.png" alt="Auriga 系统架构与闭环流程" width="100%" />
</p>

一次运行围绕 **观察 → 判断 → 行动 → 反馈** 循环展开：
1. 首次物理动作前，模型提交简短的任务计划（`task_plan`）。
2. `robot_action(action="preview")` 先整批预览各阶段（单/双臂运动、夹爪动作、依赖关系、可选观察检查点），不推进仿真世界。
3. `robot_action(action="execute", plan_id=...)` 按 `plan_id` 经 [`Harness.execute`](auriga/runtime/harness.py) 执行同一缓存计划，并实时监控到位容差与夹爪门禁。
4. RoboDojo 中模型思考期间物理世界保持暂停，仅通过动作或显式 `robot_wait` 推进时间。任务分数来自原生 evaluator，不采信模型自报。

---

## 基准结果

在 [RoboDojo](https://robodojo-benchmark.com/) 42 个任务上的五个能力维度平均得分（Score, 0–100）与总成功率（SR）对比：

| 方法 | 类型 | 平均 (Score / SR) | Generalization | Precision | Long-Horizon | Memory | Open |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| WALL-OSS-1.0-lite | VLA | 42.6 / 35.0% | 46.7 | 38.2 | 49.9 | 57.4 | 20.8 |
| GO-3 Nano Preview | Agent + WAM | 61.9 / 56.4% | 36.5 | **78.7** | **68.2** | 64.5 | 61.7 |
| RoboProbe *(参考基线)* | Agent | 29.0 / 22.5% | 33.4 | 12.7 | 21.5 | 43.0 | 34.4 |
| HexaShell-Preview | Agent | 53.2 / 46.9% | 39.2 | 41.3 | 42.9 | 84.5 | 58.0 |
| PyRUA-Lean | Agent | 55.3 / 50.4% | 48.3 | 44.8 | 54.3 | 68.4 | 60.6 |
| **Auriga (Ours)** | **Agent** | **64.0 / 57.7%** | **52.9** | 46.3 | 62.1 | **93.4** | **65.4** |

---

## 工具列表

初始会话中直接暴露 10 个核心工具，其余工具通过 Codex 原生工具搜索按需加载。每个工具在 [`auriga/runtime/tools/definitions.py`](auriga/runtime/tools/definitions.py) 中声明输入/输出 schema、物理单位（`m`、`rad`、`deg`、`s`）、默认值和范围。

| 工具 | 作用 |
| :--- | :--- |
| `task_plan` | 提交和修订任务计划与关注清单；每次决策前重新注入。 |
| `robot_observe` | 当前图像及关节 / 末端 / 夹爪状态；支持按需局部放大裁剪。 |
| `robot_information`, `robot_capabilities` | 机器人结构、URDF、相机标定、原生几何与运行时能力。 |
| `robot_action` | 单/双臂阶段批次的预览、执行与证据检查。 |
| `robot_wait` | 推进仿真，观察运动或等待物体落稳。 |
| `robot_action_status` | 查询动作的原始请求与执行回执。 |
| `program_run`, `program_status` | 运行闭环反馈或数值程序，审批其请求的物理动作。 |
| `geometry_query`, `robot_contact_overlay` | 标定三维几何（双目三角化、像素—平面交点）与夹爪接触参考叠加图。 |
| `vision_compare` | 量化两次观察之间的局部图像变化。 |
| `history_*` | 回看动作、轨迹和观察，并添加注释。 |
| `operation_memory` | 查询或记录当前任务与跨任务的操作经验。 |
| *可选*：`robotic_view`, `stereo_guide`, `vision_review`, `depth_query`, `robot_policy_step` | 腕部相机视点解算、双目极线引导、辅助视觉复核、RGB-D 数值查询（仅限显式深度条件）、$\pi_{0.5}$ 局部策略。 |

详见 [Unified Actions](docs/development/unified-actions.md) 和 [Tool Plugins](docs/development/tool-plugins.md)。

---

## 环境要求与安装

### 运行要求

- **系统与 GPU**：Linux `x86_64`，能运行 Isaac Sim 5.1 的 NVIDIA GPU。
- **Conda / Mamba**：[Miniforge](https://github.com/conda-forge/miniforge)（`mamba`）。项目统一使用名为 `Auriga` 的 conda 环境（Python 3.12，见 [`environment.yml`](environment.yml)），不要创建 `.venv`。
- **Node.js**：Node.js 22 和 `npm`（用于构建前端网页）。
- **RoboDojo**：按 [RoboDojo 环境配置](docs/development/robodojo-setup.md) 固定版本并打好补丁的 RoboDojo、Isaac Sim、IsaacLab 与任务资产（使用其独立解释器）。
- **Codex 配置**：默认 `~/.codex`，配置支持 Responses 协议的模型供应商凭据。默认模型为 `gpt-6.1-sol`（推理档位 `high`），SDK/CLI 版本固定在 `requirements.lock.txt`。
- **可选 VGGT 感知服务**：[VGGT 感知服务](services/vggt_height/README.md)（默认访问 `http://127.0.0.1:8331`）提供三维对象表、点云与高度估计。服务未启动时相应附加结果标为不可用，运行继续；无需使用时可设 `"external_perception": false`。

### 安装步骤

```bash
git clone https://github.com/lose4578/Auriga.git
cd Auriga

# 1. 创建 Auriga conda 环境
mamba env create -f environment.yml

# 2. 构建 Web 工作台与实验库前端
npm --prefix web ci && npm --prefix web run build

# 3. 运行 CPU 检查（无需 GPU、仿真器或模型）
mamba run -n Auriga python -m unittest discover -s tests
```

---

## 配置说明

配置按以下顺序合并覆盖：[`configs/auriga.json`](configs/auriga.json) → `configs/auriga.local.json`（Git 忽略）→ `--config <file>` → 命令行参数。

将本机相关路径（RoboDojo 解释器与目录、Codex home、模型）写入 `configs/auriga.local.json`：

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

只打印合并后的有效配置（不启动任何服务）：

```bash
mamba run -n Auriga python -m auriga.apps.gateway --print-config
```

全部可配字段见 [Configuration Guide](docs/development/configuration.md) 和 [完整示例](configs/auriga.example.json)。

---

## 快速开始

### 1. 运行单局任务

选择空闲 GPU 和新的运行目录：

```bash
mamba run -n Auriga python -u -m auriga.apps.gateway \
  --benchmark --task general_pickup --seed 0 --layout-id 0 \
  --gpu 0 --port 8767 --run-dir runs/my-first-task
```

打开 **<http://127.0.0.1:8767>** 查看相机画面、原生 Codex 终端、动作预览与运行记录。

### 2. 回放、检索与高清导出

```bash
# 只读回放运行目录（无需 GPU 和模型）
mamba run -n Auriga python -m auriga.apps.gateway --replay --run-dir runs/my-first-task --port 8767

# 实验库：在 http://127.0.0.1:8780 检索运行、对比方案与回放录像
mamba run -n Auriga python -m auriga.apps.library

# 按归档动作重新仿真并导出高清视频（需要仿真器，不调用模型）
mamba run -n Auriga python -m auriga.apps.replay.hd_replay --source-run runs/my-first-task --output runs/my-first-task-hd
```

### 3. 批量评测

批量评测使用 manifest 和队列（`python -m auriga.apps.evaluation.queue`）；[`benchmarks/`](benchmarks) 中的固定任务组可生成 manifest：
- [`gpt-as-policy-10`](benchmarks/gpt-as-policy-10/README.md)：10 个任务 × 5 个场景（50 个 case）。
- [`robodojo-42-15`](benchmarks/robodojo-42-15/README.md)：官方 42 个任务 × 15 个场景（630 个 case）。

会话恢复、交互控制、批量运行、报告与审计见 [Running Guide](docs/development/running.md)。

---

## 安全边界

生成的代码默认以启动用户的身份运行（`features.sandbox_enabled=false`）：原生 Codex 命令、子代理和 `program_run` 可访问当前用户有权访问的文件和网络，由 Harness 的应用层边界负责隔离（观察过滤、动作准入、程序审核）。

在具备 `bubblewrap`、Landlock ABI ≥ 4 和 `libseccomp` 的 Linux 主机上，可通过 `{"version": 1, "features": {"sandbox_enabled": true}}` 开启 OS 级沙箱；隔离条件不满足时启动直接失败，不会静默降级。详见 [Code Execution Sandbox](docs/development/configuration.md#code-execution-sandbox)。请将凭据保存在 `~/.codex`，不要提交到仓库。

---

## 目录结构

```text
auriga/        Python 包，六层结构；模块级 import 只指向下层
  core/          协议契约、配置、信息边界、存储、模型侧展示
  robot/         机器人模型与标定、动作契约与运动规划算法、RoboDojo 后端
  perception/    图像测量、叠加图、可选外部感知、RGB-D、场景存储
  cortex/        Codex 会话与沙箱、actor 启动上下文、回合监督
  runtime/       Harness、robot_action 编排、工具注册与插件、程序、策略桥接
  apps/          网关/工作台、实验库、评测队列、高清重放、参考基线
web/           工作台与实验库前端（构建后打入包内）
configs/       默认配置、完整示例、图像设置、预设与价格卡
benchmarks/    固定 benchmark 任务组（case 与分组）
scripts/       打包、任务组生成器、兼容入口与开发探针
services/      可选 VGGT 三维感知服务
contracts/     测试使用的协议夹具
tests/         CPU 契约与回归测试
third_party/   带许可证的第三方固定资料（Codex 模型目录、RoboProbe）
docs/          英文开发指南、设计文档与感知说明
```

**代码导读**：[`auriga/apps/gateway/cli.py`](auriga/apps/gateway/cli.py) 构建 [`Workbench`](auriga/apps/gateway/workbench.py)，启动 Codex 会话（[`auriga/cortex/session/codex.py`](auriga/cortex/session/codex.py)）和 [`Harness`](auriga/runtime/harness.py)。工具在 [`auriga/runtime/tools/definitions.py`](auriga/runtime/tools/definitions.py) 声明、由 [`auriga/runtime/tools/builtins.py`](auriga/runtime/tools/builtins.py) 装配；`robot_action` 位于 [`auriga/runtime/actions/unified_actions.py`](auriga/runtime/actions/unified_actions.py)。物理执行从 [`Harness.execute`](auriga/runtime/harness.py) 经 [`auriga/robot/backends/robodojo/robot_process.py`](auriga/robot/backends/robodojo/robot_process.py) 发往在 RoboDojo 解释器中运行的 [`auriga/robot/backends/robodojo/robot_worker.py`](auriga/robot/backends/robodojo/robot_worker.py)。

---

## 文档入口

| 你想要…… | 阅读 |
| :--- | :--- |
| 运行任务和 benchmark | [Running Guide](docs/development/running.md)、[Configuration](docs/development/configuration.md)、[RoboDojo Setup](docs/development/robodojo-setup.md) |
| 理解一次调用怎样完成 | [Implementation Architecture](docs/development/architecture.md)、[Shared Protocol](docs/design/shared-protocol.md) |
| 新增或修改工具 | [Tool Contracts & Plugins](docs/development/tool-plugins.md)、[Extension Guide](docs/development/extensions.md) |
| 接入新机械臂或环境 | [Robot & Environment Integration](docs/development/robot-environment-integration.md)、[Backend Boundaries](docs/development/robot-backend-boundaries.md) |
| 了解模型能看到什么 | [Actor Information Boundary](docs/development/actor-information-boundary.md) |
| 了解三维感知 | [obs3d](docs/obs3d.md)、[object_table](docs/object_table.md)、[scene3d](docs/scene3d.md)、[vggt_height](docs/vggt_height.md) |
| 开发网页 | [web/README.md](web/README.md) |
| 检索和回放实验 | [Experiment Library](docs/development/experiment-library.md)、[HD Replay](docs/development/hd-replay.md) |
| 参与贡献 | [AGENTS.md](AGENTS.md)、[Validation Guide](docs/development/validation.md)、[CONTRIBUTING.md](CONTRIBUTING.md) |

完整索引见 [docs/README.md](docs/README.md)。

---

## 许可证与第三方资料

Auriga 以 [MIT 许可证](LICENSE) 发布。[`third_party/`](third_party) 保留随仓库分发的第三方资料的上游许可证和来源（Codex 模型目录与 RoboProbe 参考代码，均为 Apache-2.0）。RoboDojo、Isaac Sim、IsaacLab、cuRobo 和模型权重不随本仓库分发。

---

## 🌟 引用

如果本项目对你的工作有帮助，欢迎引用：

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
