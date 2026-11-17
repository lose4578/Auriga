"""切片节点离线感知对比与 Codex 原生 thread/fork 单步反事实探针（Turn Probe）。

支持两种零仿真 GPU 的极速验证模式：
1. `probe_perception_slice` / `probe_perception_batch`：
   在归档切片节点上用当前工作区代码重算 3D 感知（`object_table` / `stereo_matches` / `obs3d`），
   直接量化对比新旧感知输出（物体召回、语义标签保留、双目极线残差过滤、桌面重锚定与夹持态保持）。
2. `probe_forked_turn`：
   在归档切片节点处替换最新感知输出，调用 Codex 原生 `client.thread_fork` 分叉出带完整历史上下文的新线程，
   让 GPT 模型继续推理直至发出下一步 `robot_action`，直接对比新旧代码下模型规划的 3D 抓取/放置轨迹差异。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace

from ...cortex.context.actor_instructions import instructions as actor_instructions
from ...core.presentation.actor_output import ActorOutput
from ...core.presentation.actor_presentation import ActorPresentation
from .checkpoint import (
    ForkedCodexSession,
    _hydrate_runtime_perception_state,
    prepare_checkpoint_slice,
    recompute_tool_output,
    require_codex_archive,
)
from ...core.config import Config
from ...core.contracts import canonical
from ...cortex.context.actor_instructions import CALIBRATED_OBSERVATION_INSTRUCTIONS, LEGACY_OBSERVATION_INSTRUCTIONS
from ...robot.model.geometry import GeometryTools
from ...core.observation_policy import allows_depth, has_calibration
from ...perception.depth.depth_runtime import INSTRUCTIONS as DEPTH_INSTRUCTIONS
from ...perception.external.obs3d import developer_instructions as obs3d_instructions
from ...perception.scene.store import SceneStore
from ...core.storage.store import Store, open_read_store
from ...cortex.context.task_notes import handoff_policy
from ...cortex.context.task_planning import TaskPlanning
from ...core.tool_output import StructuredText, ToolResponse
from ...runtime.tools.builtins import create_registry


def _extract_old_table_and_stereo(recomputed: dict, source_run: Path, call_id: str) -> tuple[list, list]:
    """从归档工具结果中提取旧版 rows 与 stereo_matches。"""
    if recomputed["kind"] == "object_table":
        old_val = (recomputed.get("old_data") or {}).get("value") or {}
        return list(old_val.get("rows") or []), list(old_val.get("stereo_matches") or [])
    old_rows = []
    with open_read_store(source_run) as store:
        for rec in store.records("tool_result"):
            if (rec.get("correlation") or {}).get("tool_call_id") == call_id and rec.get("raw_ref"):
                raw = json.loads(store.resolve_artifact(rec["raw_ref"], ("observer", "actor"))[1])
                for item in (raw.get("wire_response") or {}).get("contentItems", []):
                    if item.get("type") == "inputText" and '"object_table"' in str(item.get("text", "")):
                        ot = (json.loads(item["text"]) or {}).get("object_table") or {}
                        if isinstance(ot, dict):
                            old_rows = list(ot.get("rows") or [])
                break
    return old_rows, []


def probe_perception_slice(
    source_run: str | Path,
    call_id: str,
    *,
    config_overrides: dict | None = None,
) -> dict:
    """对单个切片节点执行新旧感知输出对比。"""
    source_run = Path(source_run).resolve()
    t0 = time.monotonic()
    recomputed = recompute_tool_output(source_run, call_id, config_overrides=config_overrides)
    elapsed_s = round(time.monotonic() - t0, 3)

    old_rows, old_stereo = _extract_old_table_and_stereo(recomputed, source_run, call_id)
    if recomputed["kind"] == "object_table":
        new_val = (recomputed.get("new_data") or {}).get("value") or {}
        new_rows = list(new_val.get("rows") or [])
        new_stereo = list(new_val.get("stereo_matches") or [])
        new_hints = new_val.get("grasp_hints") or {}
        new_delta = None
        new_warn = None
        new_ledger = None
    else:
        nd = recomputed.get("new_data") or {}
        new_ot = nd.get("object_table") or {}
        new_rows = list(new_ot.get("rows") or []) if isinstance(new_ot, dict) else []
        new_stereo = []
        new_hints = new_ot.get("grasp_hints") or {} if isinstance(new_ot, dict) else {}
        new_delta = nd.get("object_state_delta")
        new_warn = nd.get("repeat_failure_warning")
        new_ledger = nd.get("episode_ledger")

    def _labelled_count(rows):
        return sum(1 for r in rows if len(r) >= 11 and r[10] is not None)

    def _floating_unknown_count(rows):
        return sum(1 for r in rows if len(r) >= 10 and r[8] == "unknown" and "floating" in str(r[9]))

    old_max_skew = max((float(m.get("skew_gap_m") or 0.0) for m in old_stereo), default=0.0)
    new_max_skew = max((float(m.get("skew_gap_m") or 0.0) for m in new_stereo), default=0.0)
    old_clipped_stereo = sum(1 for m in old_stereo if m.get("clipped_at_border"))

    return {
        "case": source_run.name,
        "call_id": call_id,
        "tool": recomputed["tool"],
        "kind": recomputed["kind"],
        "elapsed_s": elapsed_s,
        "active_labels": recomputed.get("active_labels"),
        "old_row_count": len(old_rows),
        "new_row_count": len(new_rows),
        "old_labelled_rows": _labelled_count(old_rows),
        "new_labelled_rows": _labelled_count(new_rows),
        "old_floating_unknown": _floating_unknown_count(old_rows),
        "new_floating_unknown": _floating_unknown_count(new_rows),
        "old_stereo_count": len(old_stereo),
        "new_stereo_count": len(new_stereo),
        "old_clipped_stereo_count": old_clipped_stereo,
        "old_max_skew_gap_m": round(old_max_skew, 4),
        "new_max_skew_gap_m": round(new_max_skew, 4),
        "old_rows": old_rows,
        "new_rows": new_rows,
        "old_stereo_matches": old_stereo,
        "new_stereo_matches": new_stereo,
        "new_grasp_hints": new_hints,
        "object_state_delta": new_delta,
        "repeat_failure_warning": new_warn,
        "episode_ledger": new_ledger,
    }


def probe_perception_batch(slices: list[tuple[str | Path, str] | dict], *, max_workers: int = 4) -> list[dict]:
    """并行对多个 (source_run, call_id) 或字典规范的切片执行感知重算对比。"""
    pairs = [
        (s["source_run"], s["call_id"]) if isinstance(s, dict) else (s[0], s[1])
        for s in slices
    ]
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [pool.submit(probe_perception_slice, run_dir, cid) for run_dir, cid in pairs]
        return [f.result() for f in futures]


def _extract_archived_next_calls(source_run: Path, after_call_id: str, max_calls: int = 6) -> list[dict]:
    """提取原归档运行在切点之后紧接着发出的工具调用序列（用于与分叉线程对比）。"""
    with open_read_store(source_run) as store:
        calls = store.records("tool_call")
        started = False
        out = []
        for rec in calls:
            p = rec.get("params") or {}
            cid = p.get("callId")
            if not started:
                if cid == after_call_id:
                    started = True
                continue
            out.append({
                "call_id": cid,
                "tool": p.get("tool"),
                "arguments": p.get("arguments"),
            })
            if p.get("tool") == "robot_action":
                req = (p.get("arguments") or {}).get("request") or {}
                if req.get("mode") in {"execute", "direct", None}:
                    break
            if len(out) >= max_calls:
                break
        return out


def _build_developer_instructions(slice_dir: Path, store: Store, cfg: Config, pres: ActorPresentation, out: ActorOutput) -> str:
    """构建与真实 Workbench 启动一致的开发者指令（含 robot_information、capabilities 与 3D 感知说明）。"""
    obs_profile = cfg.features.observation_profile
    observation_contract = (
        DEPTH_INSTRUCTIONS if allows_depth(obs_profile)
        else CALIBRATED_OBSERVATION_INSTRUCTIONS if has_calibration(obs_profile)
        else LEGACY_OBSERVATION_INSTRUCTIONS
    )
    handoff = handoff_policy(cfg.features.airborne_handoff, handoff_preference=cfg.features.handoff_preference)
    instructions = actor_instructions(
        observation_contract,
        handoff["text"],
        source_root=None if cfg.features.sandbox_enabled else slice_dir / "private" / "actor-source",
        operation_memory="operation_memory" not in cfg.features.disabled_tools,
    )
    instructions += obs3d_instructions(cfg.features)
    if getattr(cfg.features, "pose_skill", False):
        from ...perception.external.pose_skill import INSTRUCTIONS as pose_skill_instructions
        instructions += pose_skill_instructions
    if getattr(cfg.features, "held_perception", False):
        from ...perception.external.held_perception import INSTRUCTION as held_perception_instruction
        instructions += "\n" + held_perception_instruction
    robot_info = store.read_state("runtime", "robot_information")
    if robot_info:
        from ...robot.model.robot_information import robot_information_text
        presented_info = out.present(robot_info)
        instructions = robot_information_text(presented_info) + "\n\nHarness instructions\n" + instructions
    return instructions


class ProbeRobotAdapter:
    """零仿真探针专用的轻量机器人适配器：支持真实 unified_actions._preview（含 audit_batch_grasp_3d 与 geometry_review）及 program_run 内部动作预检。"""

    def __init__(self, image_settings: dict, *, simulate_risks: list[str] | None = None):
        self.image_settings = image_settings
        self.simulate_risks = list(simulate_risks or [])
        self._plan_seq = 0
        self._batch_plans = {}
        self.previewed_actions = []
        self.executed_actions = []

    def preview_batch(self, action: dict) -> dict:
        self._plan_seq += 1
        plan_id = f"probe-native-plan-{self._plan_seq}"
        stages = action.get("stages") or []
        stage_costs = []
        risks = list(self.simulate_risks)
        for idx, st in enumerate(stages):
            kind = st.get("kind", "move")
            stage_costs.append({
                "stage_id": st.get("id", f"s{idx}"),
                "kind": kind,
                "arm": st.get("arm"),
                "status": "planned",
                "nominal_native_steps": 25 if kind == "move" else 10,
                "nominal_sim_time_s": 1.0 if kind == "move" else 0.4,
            })
            # 轻量双臂目标间距检测：若同阶段双臂目标间距 < 0.08m，生成 inter_arm_proxy_intersection 风险以验证降级链路
            if kind == "move" and st.get("arm") == "both":
                arms_cfg = st.get("arms") or {}
                l_pos = (arms_cfg.get("left") or {}).get("grasp_position_m") or (arms_cfg.get("left") or {}).get("position_m")
                r_pos = (arms_cfg.get("right") or {}).get("grasp_position_m") or (arms_cfg.get("right") or {}).get("position_m")
                if isinstance(l_pos, (list, tuple)) and isinstance(r_pos, (list, tuple)) and len(l_pos) == 3 and len(r_pos) == 3:
                    dist = sum((float(l_pos[k]) - float(r_pos[k])) ** 2 for k in range(3)) ** 0.5
                    if dist < 0.08 and "inter_arm_proxy_intersection" not in risks:
                        risks.append("inter_arm_proxy_intersection")
        total_steps = sum(c["nominal_native_steps"] for c in stage_costs)
        raw = {
            "status": "planned",
            "plan_id": plan_id,
            "planned_native_steps": total_steps,
            "nominal_sim_time_s": round(total_steps * 0.04, 2),
            "stage_costs": {c["stage_id"]: c for c in stage_costs},
            "motion_safety_checks": [
                {"stage_id": c["stage_id"], "enabled": True, "status": "pass", "start_state_source": "measured"}
                for c in stage_costs if c["kind"] == "move"
            ],
            "geometry_review": {
                "status": "risk" if risks else "clear",
                "risks": risks,
            },
            "trajectory_data": {
                "plan_id": plan_id,
                "planned_native_steps": total_steps,
                "arms": {},
            },
        }
        self._batch_plans[plan_id] = deepcopy(raw)
        self.previewed_actions.append(deepcopy(action))
        return raw


def _extract_tool_error(res: dict | None) -> str | None:
    """从工具返回包中提取结构化错误原因（当 success=False 时）。"""
    if not isinstance(res, dict) or res.get("success", True) is not False:
        return None
    for item in res.get("contentItems") or []:
        if isinstance(item, dict) and item.get("type") == "inputText":
            txt = str(item.get("text") or "").strip()
            if txt:
                return txt[:240]
    return "tool_call_failed"


def probe_forked_turn(
    source_run: str | Path,
    until_call_id: str,
    *,
    work_dir: str | Path | None = None,
    max_probe_calls: int = 4,
    turn_timeout_s: float = 180.0,
    stop_on_preview: bool = False,
    recompute_cutpoint: bool = True,
    continuation_prompt: str = "Continue the task from the latest tool result above.",
    config_overrides: dict | None = None,
) -> dict:
    """在切点处利用 Codex 原生 `thread/fork` 启动反事实探针，完整支持 preview -> execute 与 program_run 内部动作拦截。"""
    source_run = Path(source_run).resolve()
    require_codex_archive(source_run)
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix=f"slice-probe-{source_run.name[:24]}-"))
    else:
        work_dir = Path(work_dir).resolve()

    t0 = time.monotonic()
    prepared = prepare_checkpoint_slice(
        source_run,
        work_dir,
        until_call_id=until_call_id,
        recompute_cutpoint=recompute_cutpoint,
        config_overrides=config_overrides,
    )
    spec = prepared["spec"]
    archived_next = _extract_archived_next_calls(source_run, until_call_id)

    cfg_dict = json.loads((work_dir / "effective_config.json").read_text())
    cfg = Config.from_dict(cfg_dict)
    if config_overrides:
        cfg = cfg.with_overrides(config_overrides)

    with open_read_store(work_dir) as ro_store:
        identity = ro_store.read_state("runtime", "identity") or {}
    store = Store(work_dir, identity)
    observations = store.records("observation")
    latest_obs = observations[-1] if observations else None
    capabilities = store.read_state("runtime", "capabilities") or {}

    from ...runtime.harness import Harness
    from ...runtime.programs.runner import ProgramRunner

    probe_robot = ProbeRobotAdapter(image_settings=cfg.images.to_dict())
    harness = Harness(
        store,
        probe_robot,
        cfg.run.max_actions,
        actor_payload_profile=cfg.features.actor_payload_profile,
        enable_wait=cfg.features.enable_wait,
        observation_profile=cfg.features.observation_profile,
        motion_review=False,
        vision_compare=cfg.features.vision_compare,
        config=cfg,
        task_plan_required=True,
    )
    harness.latest = latest_obs
    harness.observe = lambda correlation=None: {**harness.latest, "action_budget": harness.action_budget()}
    harness.capabilities = capabilities
    limit = (capabilities.get("action_budget") or {}).get("native_step_limit")
    if isinstance(limit, int) and limit > 0:
        harness.native_budget_limit = limit
        harness.native_budget_known = True
        harness.native_action_count = sum(
            int(r.get("executed_native_steps") or 0) for r in store.records("action_receipt")
        )
    pres = harness.actor_presentation
    out = harness.actor_output

    recorded_calls = []
    stop_triggered = threading.Event()
    session_ref = {}

    def _trigger_stop():
        stop_triggered.set()
        sess = session_ref.get("session")
        if sess and sess.turn_id:
            threading.Thread(target=lambda: sess.interrupt(), daemon=True).start()

    def _probe_execute(action, correlation=None):
        """在 Harness.execute 出口拦截物理动作（同时覆盖直接调用与 program_run 脚本内调用）。"""
        probe_robot.executed_actions.append({"action": deepcopy(action), "correlation": deepcopy(correlation)})
        obs_id = (harness.latest or {}).get("observation_id")
        act_id = action.get("action_id", "act-probe-1")
        receipt = {
            "action_id": act_id,
            "status": "completed",
            "batch_status": "completed",
            "result_observation_id": obs_id,
            "executed_native_steps": 0,
            "stage_results": {
                s.get("id", f"s{i}"): {"status": "completed"}
                for i, s in enumerate(action.get("stages") or [])
                if isinstance(s, dict)
            },
        }
        harness.store.save("action_request", act_id, action)
        harness.store.save("action_receipt", act_id, receipt)
        return {
            "request": action,
            "receipt": receipt,
            "current_observation_id": obs_id,
            "result_observation_is_current": True,
        }

    harness.execute = _probe_execute
    harness.programs = ProgramRunner(harness.program_services())

    _hydrate_runtime_perception_state(harness, store, out, None)
    # 缓存切点处新算出的 object_table，供下一步 robot_action 的 audit_batch_grasp_3d 直接消费
    if prepared.get("recomputed") and prepared["recomputed"].get("kind") == "object_table":
        val = (prepared["recomputed"].get("new_data") or {}).get("value")
        if isinstance(val, dict) and latest_obs:
            from ...perception.external.obs3d import _remember
            shown = dict(columns=val["columns"], rows=val["rows"], table_z=val.get("support_z_m"))
            if val.get("grasp_hints"):
                shown["grasp_hints"] = val["grasp_hints"]
            _remember(harness, latest_obs["observation_id"], "object_table", (shown, None, prepared["recomputed"]["new_data"]))

    tools = harness.tool_registry.catalog(harness, role="actor")
    instructions = _build_developer_instructions(work_dir, store, cfg, pres, out)
    tool_errors = []

    def probe_tool_handler(params):
        name = params.get("tool")
        raw_args = deepcopy(params.get("arguments") or {})
        call_entry = {
            "call_id": params.get("callId"),
            "tool": name,
            "arguments": raw_args,
        }
        recorded_calls.append(call_entry)
        try:
            res = harness.handle_tool(params)
            err = _extract_tool_error(res)
            if err:
                call_entry["error"] = err
                tool_errors.append({"call_id": params.get("callId"), "tool": name, "error": err})
        finally:
            if (
                probe_robot.executed_actions
                or (stop_on_preview and probe_robot.previewed_actions)
                or len(recorded_calls) >= max_probe_calls
            ):
                _trigger_stop()
        return res

    initial_plan_ids = set(getattr(harness, "unified_action_plans", {}).keys())
    session = ForkedCodexSession(work_dir, store, probe_tool_handler, config=cfg, task_planning=harness.task_planning)
    session_ref["session"] = session
    try:
        forked_tid = session.start_forked(tools, instructions, spec["source_thread_id"])
        harness.thread_id = forked_tid
        session.run_turn(continuation_prompt)
        session.turn_done.wait(timeout=turn_timeout_s)
    finally:
        if harness.programs:
            harness.programs.close()
        session.close()
        previewed_plans = [
            {
                "plan_id": rec.get("summary", {}).get("plan_id"),
                "source_stages": rec.get("source_stages"),
                "grasp_3d_review": rec.get("summary", {}).get("grasp_3d_review"),
                "planned_native_steps": rec.get("summary", {}).get("planned_native_steps"),
            }
            for pid, rec in getattr(harness, "unified_action_plans", {}).items()
            if pid not in initial_plan_ids
        ]
        store.close()

    return {
        "case": source_run.name,
        "until_call_id": until_call_id,
        "work_dir": str(work_dir),
        "source_thread_id": spec["source_thread_id"],
        "forked_thread_id": forked_tid,
        "closed_open_turn": spec.get("rollout", {}).get("closed_open_turn", False),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "archived_next_calls": archived_next,
        "forked_next_calls": recorded_calls,
        "previewed_plans": previewed_plans,
        "executed_actions_count": len(probe_robot.executed_actions),
        "tool_error_count": len(tool_errors),
        "tool_errors": tool_errors,
    }


def _resolve_cutpoint_ids(
    source_run: Path,
    until_call_id: str | None = None,
    until_action_id: str | None = None,
    action_index: int | None = None,
    cold_start: bool = False,
) -> tuple[str | None, str | None]:
    """支持按 call_id、action_id、动作序号 action_index（0=第1步后、-1=末步后）或 cold_start（首帧观察后、零历史工具惯性）解析切点。"""
    source_run = Path(source_run).resolve()
    if cold_start and not until_call_id and not until_action_id:
        with open_read_store(source_run) as store:
            store.identity = store.read_state("runtime", "identity") or {}
            for rec in store.records("tool_call"):
                p = rec.get("params") or {}
                if p.get("tool") == "robot_observe" and p.get("callId"):
                    return str(p["callId"]), None
        raise ValueError(f"源运行未找到初始 robot_observe 调用: {source_run}")
    if not until_call_id and not until_action_id and action_index is None:
        raise ValueError("必须指定 until_call_id、until_action_id、action_index 或 cold_start 之一")
    if action_index is not None and not until_action_id and not until_call_id:
        with open_read_store(source_run) as store:
            store.identity = store.read_state("runtime", "identity") or {}
            reqs = store.records("action_request")
            if not reqs:
                raise ValueError(f"源运行无动作记录: {source_run}")
            until_action_id = reqs[int(action_index)]["action_id"]
    if until_action_id and not until_call_id:
        with open_read_store(source_run) as store:
            store.identity = store.read_state("runtime", "identity") or {}
            last_actor_call_id = None
            for ev in store.events(limit=100000):
                corr = ev.get("correlation") or {}
                payload = ev.get("payload") or {}
                tcid = corr.get("tool_call_id") or payload.get("callId")
                if isinstance(tcid, str) and tcid.startswith("call_"):
                    last_actor_call_id = tcid
                elif ev.get("type") == "codex.raw" and payload.get("direction") == "received":
                    msg = payload.get("message") or {}
                    if msg.get("method") == "item/tool/call":
                        cid = (msg.get("params") or {}).get("callId")
                        if isinstance(cid, str) and cid.startswith("call_"):
                            last_actor_call_id = cid
                if corr.get("action_id") == until_action_id:
                    raw_cid = corr.get("tool_call_id")
                    until_call_id = last_actor_call_id if (not raw_cid or not str(raw_cid).startswith("call_")) and last_actor_call_id else raw_cid
    return until_call_id, until_action_id


def run_live_slice(
    source_run: str | Path,
    *,
    until_call_id: str | None = None,
    until_action_id: str | None = None,
    action_index: int | None = None,
    cold_start: bool = False,
    work_dir: str | Path | None = None,
    gpu_id: int = 7,
    max_slice_actions: int = 2,
    max_turns: int = 3,
    turn_timeout_s: float = 240.0,
    recompute_cutpoint: bool = True,
    continuation_prompt: str = "Continue the task from the latest tool result above.",
    config_overrides: dict | None = None,
) -> dict:
    """从任意切片节点（短程/阶段/半程/长程/冷启动）启动带真实 RoboDojo 物理快进与 Codex thread/fork 的闭环切片实验。"""
    from ...robot.backends.factory import create_backend
    from ...runtime.harness import Harness
    from ...runtime.programs.runner import ProgramRunner

    source_run = Path(source_run).resolve()
    require_codex_archive(source_run)
    explicit_call_id = until_call_id
    until_call_id, until_action_id = _resolve_cutpoint_ids(
        source_run,
        until_call_id=until_call_id,
        until_action_id=until_action_id,
        action_index=action_index,
        cold_start=cold_start,
    )
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix=f"live-slice-{source_run.name[:24]}-"))
    else:
        work_dir = Path(work_dir).resolve()

    t0 = time.monotonic()
    prepared = prepare_checkpoint_slice(
        source_run,
        work_dir,
        until_call_id=explicit_call_id if until_action_id else until_call_id,
        until_action_id=until_action_id,
        recompute_cutpoint=recompute_cutpoint,
        config_overrides=config_overrides,
    )
    spec = prepared["spec"]
    effective_call_id = spec.get("until_call_id")
    archived_next = _extract_archived_next_calls(source_run, effective_call_id) if effective_call_id else []

    cfg_dict = json.loads((work_dir / "effective_config.json").read_text())
    cfg = Config.from_dict(cfg_dict)
    if config_overrides:
        cfg = cfg.with_overrides(config_overrides)

    with open_read_store(work_dir) as ro_store:
        identity = ro_store.read_state("runtime", "identity") or {}
    store = Store(work_dir, identity)
    observations = store.records("observation")
    latest_obs = observations[-1] if observations else None

    robot = create_backend(
        task=cfg.run.task,
        gpu_id=int(gpu_id),
        seed=cfg.run.seed,
        layout_id=cfg.run.layout_id,
        run_dir=work_dir / "private" / "robot",
        observation_profile=cfg.features.observation_profile,
        tracking_guard=cfg.features.tracking_guard,
        joint_replan=cfg.features.joint_replan,
        config=cfg,
    )
    harness = Harness(
        store,
        robot,
        cfg.run.max_actions,
        actor_payload_profile=cfg.features.actor_payload_profile,
        enable_wait=cfg.features.enable_wait,
        observation_profile=cfg.features.observation_profile,
        motion_review=cfg.features.motion_review,
        vision_compare=cfg.features.vision_compare,
        config=cfg,
        task_plan_required=True,
    )
    pres = harness.actor_presentation
    out = harness.actor_output

    recorded_calls = []
    executed_slice_actions = []
    stop_triggered = threading.Event()
    session_ref = {}

    def _trigger_stop():
        stop_triggered.set()
        sess = session_ref.get("session")
        if sess and sess.turn_id:
            threading.Thread(target=lambda: sess.interrupt(), daemon=True).start()

    orig_harness_execute = harness.execute

    def _live_slice_execute(action, correlation=None):
        res = orig_harness_execute(action, correlation=correlation)
        rcpt = res.get("receipt") or {}
        executed_slice_actions.append({
            "action_id": action.get("action_id"),
            "status": rcpt.get("status"),
            "batch_status": rcpt.get("batch_status"),
            "reason": rcpt.get("reason"),
            "executed_native_steps": rcpt.get("executed_native_steps"),
            "stage_results": rcpt.get("stage_results"),
            "result_observation_id": rcpt.get("result_observation_id"),
        })
        if harness.episode_done.is_set() or (max_slice_actions > 0 and len(executed_slice_actions) >= max_slice_actions):
            _trigger_stop()
        return res

    harness.execute = _live_slice_execute
    harness.programs = ProgramRunner(harness.program_services())

    ff_info = {}
    eval_after = {}
    forked_tid = None
    session = None
    try:
        t_boot0 = time.monotonic()
        harness.capabilities = deepcopy(robot.start())
        limit = (harness.capabilities.get("action_budget") or {}).get("native_step_limit")
        if isinstance(limit, int) and limit > 0:
            harness.native_budget_limit, harness.native_budget_known = limit, True
        if observations:
            harness.capture_home_reference(observations[0])
            harness.latest = latest_obs
        t_ff0 = time.monotonic()
        ff_info = robot.fast_forward_slice(spec)
        ff_info["fast_forward_wall_s"] = round(time.monotonic() - t_ff0, 3)
        ff_info["worker_boot_wall_s"] = round(t_ff0 - t_boot0, 3)
        harness.native_action_count = int(ff_info.get("native_actions_dispatched") or 0)

        _hydrate_runtime_perception_state(harness, store, out, None)
        if prepared.get("recomputed") and prepared["recomputed"].get("kind") == "object_table":
            val = (prepared["recomputed"].get("new_data") or {}).get("value")
            if isinstance(val, dict) and latest_obs:
                from ...perception.external.obs3d import _remember
                shown = dict(columns=val["columns"], rows=val["rows"], table_z=val.get("support_z_m"))
                if val.get("grasp_hints"):
                    shown["grasp_hints"] = val["grasp_hints"]
                _remember(harness, latest_obs["observation_id"], "object_table", (shown, None, prepared["recomputed"]["new_data"]))

        tools = harness.tool_registry.catalog(harness, role="actor")
        instructions = _build_developer_instructions(work_dir, store, cfg, pres, out)
        tool_errors = []

        def live_tool_handler(params):
            entry = {
                "call_id": params.get("callId"),
                "tool": params.get("tool"),
                "arguments": deepcopy(params.get("arguments") or {}),
            }
            recorded_calls.append(entry)
            res = harness.handle_tool(params)
            err = _extract_tool_error(res)
            if err:
                entry["error"] = err
                tool_errors.append({"call_id": params.get("callId"), "tool": params.get("tool"), "error": err})
            if isinstance(res, dict):
                for item in res.get("contentItems") or []:
                    if isinstance(item, dict) and item.get("type") == "inputText":
                        txt = str(item.get("text") or "")
                        if txt.startswith("{"):
                            try:
                                parsed = json.loads(txt)
                                entry["result_summary"] = {
                                    k: parsed.get(k)
                                    for k in ("status", "reason", "failed_stage_id", "failed_arm", "plan_id", "executed_native_steps", "auto_executed_from_clean_preview", "downgraded_from_execute_if_clean")
                                    if k in parsed
                                }
                            except Exception:
                                pass
                        break
            return res

        session = ForkedCodexSession(work_dir, store, live_tool_handler, config=cfg, task_planning=harness.task_planning)
        session_ref["session"] = session
        forked_tid = session.start_forked(tools, instructions, spec["source_thread_id"])
        harness.thread_id = forked_tid

        for turn_idx in range(max(1, int(max_turns))):
            prompt = continuation_prompt if turn_idx == 0 else "Continue the task from the latest feedback."
            session.run_turn(prompt)
            finished_turn = session.turn_done.wait(timeout=turn_timeout_s)
            if not finished_turn:
                _trigger_stop()
                session.turn_done.wait(timeout=15.0)
                break
            if stop_triggered.is_set() or harness.episode_done.is_set():
                break
            if max_slice_actions > 0 and len(executed_slice_actions) >= max_slice_actions:
                break

        eval_after = robot.evaluate(final_check=True)
    finally:
        if harness.programs:
            harness.programs.close()
        if session is not None:
            session.close()
        try:
            robot.close()
        except Exception:
            pass
        store.close()

    score_before = ff_info.get("score_before")
    score_after = eval_after.get("normalized_score", eval_after.get("score"))
    delta_score = (
        round(float(score_after) - float(score_before), 4)
        if score_before is not None and score_after is not None
        else None
    )
    return {
        "case": source_run.name,
        "until_call_id": effective_call_id,
        "until_action_id": spec.get("until_action_id"),
        "target_sim_step": spec.get("target_sim_step"),
        "work_dir": str(work_dir),
        "source_thread_id": spec["source_thread_id"],
        "forked_thread_id": forked_tid,
        "fast_forward": ff_info,
        "score_before": score_before,
        "score_after": score_after,
        "delta_score": delta_score,
        "task_success": eval_after.get("task_success", eval_after.get("success")),
        "executed_slice_actions": executed_slice_actions,
        "forked_next_calls": recorded_calls,
        "archived_next_calls": archived_next,
        "tool_error_count": len(tool_errors),
        "tool_errors": tool_errors,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }


def _check_slice_assertions(result: dict, assertions: dict | None) -> list[dict]:
    """校验单个切片重算结果是否满足黄金切片集声明的断言指标。"""
    if not assertions:
        return []
    new_s = result.get("new_summary") or result
    raw_rows = new_s.get("rows") if "rows" in new_s else result.get("new_rows") or []
    rows_by_id = {}
    for r in raw_rows:
        if isinstance(r, dict):
            rows_by_id[r.get("id")] = r
        elif isinstance(r, (list, tuple)) and len(r) >= 10:
            rows_by_id[r[0]] = {
                "id": r[0],
                "centre_m": r[1],
                "length_m": float(r[4]) if r[4] is not None else 0.0,
                "width_m": float(r[5]) if r[5] is not None else 0.0,
                "support": r[8],
                "quality": r[9],
                "flags": str(r[9]).split(","),
                "label": r[10] if len(r) >= 11 else None,
            }
    checks = []
    if "max_tool_errors" in assertions or result.get("mode") in {"turn_fork", "live_slice"}:
        thr = int(assertions.get("max_tool_errors", 0))
        actual = int(result.get("tool_error_count") or 0)
        if "max_tool_errors" in assertions or actual > thr:
            checks.append({
                "name": f"tool_errors <= {thr}",
                "passed": actual <= thr,
                "actual": actual,
            })
    if "max_skew_below" in assertions:
        thr = float(assertions["max_skew_below"])
        actual = float(new_s.get("max_stereo_skew_m", result.get("new_max_skew_gap_m", 0.0)) or 0.0)
        checks.append({
            "name": f"max_skew <= {thr}",
            "passed": actual <= thr,
            "actual": actual,
        })
    if "min_labelled_count" in assertions:
        thr = int(assertions["min_labelled_count"])
        actual = int(new_s.get("labelled_count", result.get("new_labelled_rows", 0)) or 0)
        checks.append({
            "name": f"labelled_count >= {thr}",
            "passed": actual >= thr,
            "actual": actual,
        })
    if "max_floating_unknown" in assertions:
        thr = int(assertions["max_floating_unknown"])
        actual = int(new_s.get("floating_unknown_count", result.get("new_floating_unknown", 0)) or 0)
        checks.append({
            "name": f"floating_unknown <= {thr}",
            "passed": actual <= thr,
            "actual": actual,
        })
    for oid, exp_sup in (assertions.get("expect_support") or {}).items():
        actual = (rows_by_id.get(oid) or {}).get("support")
        checks.append({
            "name": f"{oid}.support == {exp_sup}",
            "passed": actual == exp_sup,
            "actual": actual,
        })
    for oid, exp_floating in (assertions.get("expect_floating") or {}).items():
        flags = (rows_by_id.get(oid) or {}).get("flags") or []
        actual = any("floating" in str(f) for f in flags)
        checks.append({
            "name": f"{oid}.floating == {exp_floating}",
            "passed": actual == bool(exp_floating),
            "actual": actual,
        })
    for oid, min_len in (assertions.get("min_length_m") or {}).items():
        actual = float((rows_by_id.get(oid) or {}).get("length_m") or 0.0)
        checks.append({
            "name": f"{oid}.length_m >= {min_len}",
            "passed": actual >= float(min_len),
            "actual": actual,
        })
    for oid, hint_key in (assertions.get("expect_hint_field") or {}).items():
        hints = result.get("new_grasp_hints") or {}
        actual = (hints.get(oid) or {}).get(hint_key)
        checks.append({
            "name": f"{oid}.hint.{hint_key} != None",
            "passed": actual is not None,
            "actual": actual,
        })
    if "fork_planned_min" in assertions:
        thr = int(assertions["fork_planned_min"])
        plans = result.get("previewed_plans") or []
        actual = max(
            sum(1 for p in plans if p.get("plan_id") or (p.get("plan") or {}).get("status") == "planned"),
            int(result.get("executed_actions_count") or 0),
        )
        checks.append({
            "name": f"fork_planned >= {thr}",
            "passed": actual >= thr,
            "actual": actual,
        })
    if "live_completed_min" in assertions:
        thr = int(assertions["live_completed_min"])
        acts = result.get("executed_slice_actions") or []
        actual = sum(
            1
            for a in acts
            if a.get("status") == "completed"
            or (a.get("status") == "stopped" and a.get("reason") == "observation_checkpoint")
        )
        checks.append({
            "name": f"live_completed >= {thr}",
            "passed": actual >= thr,
            "actual": actual,
        })
    if "live_actions_min" in assertions:
        thr = int(assertions["live_actions_min"])
        acts = result.get("executed_slice_actions") or []
        actual = sum(1 for a in acts if a.get("status") in {"completed", "stopped"})
        checks.append({
            "name": f"live_actions >= {thr}",
            "passed": actual >= thr,
            "actual": actual,
        })
    if "score_after_min" in assertions:
        thr = float(assertions["score_after_min"])
        actual = float(result.get("score_after") or 0.0)
        checks.append({
            "name": f"score_after >= {thr}",
            "passed": actual >= thr,
            "actual": actual,
        })
    if "delta_score_min" in assertions:
        thr = float(assertions["delta_score_min"])
        actual = float(result.get("delta_score") or 0.0)
        checks.append({
            "name": f"delta_score >= {thr}",
            "passed": actual >= thr,
            "actual": actual,
        })
    if "escalation_level_min" in assertions:
        thr = int(assertions["escalation_level_min"])
        warns = result.get("repeat_failure_warning") or []
        actual = max((int(w.get("escalation_level") or 0) for w in warns if isinstance(w, dict)), default=0)
        checks.append({
            "name": f"escalation_level >= {thr}",
            "passed": actual >= thr,
            "actual": actual,
        })
    if "consecutive_no_motion_min" in assertions:
        thr = int(assertions["consecutive_no_motion_min"])
        ledger = result.get("episode_ledger") or {}
        delta = result.get("object_state_delta") or {}
        actual = max(
            int(ledger.get("consecutive_no_object_motion") or 0),
            int(delta.get("consecutive_no_object_motion") or 0),
        )
        checks.append({
            "name": f"consecutive_no_motion >= {thr}",
            "passed": actual >= thr,
            "actual": actual,
        })
    if "expect_unmoved_after_unocclusion" in assertions:
        exp = bool(assertions["expect_unmoved_after_unocclusion"])
        delta = result.get("object_state_delta") or {}
        actual = bool(delta.get("unmoved_after_unocclusion"))
        checks.append({
            "name": f"unmoved_after_unocclusion == {exp}",
            "passed": actual == exp,
            "actual": actual,
        })
    if "expect_upward_stop_dz" in assertions:
        exp = bool(assertions["expect_upward_stop_dz"])
        warns = result.get("repeat_failure_warning") or []
        actual = any(w.get("upward_stop_dz_m") is not None for w in warns if isinstance(w, dict))
        checks.append({
            "name": f"upward_stop_dz_present == {exp}",
            "passed": actual == exp,
            "actual": actual,
        })
    if "expect_pre_slip_aperture" in assertions:
        exp = bool(assertions["expect_pre_slip_aperture"])
        warns = result.get("repeat_failure_warning") or []
        actual = any(w.get("pre_slip_aperture_m") is not None for w in warns if isinstance(w, dict))
        checks.append({
            "name": f"pre_slip_aperture_present == {exp}",
            "passed": actual == exp,
            "actual": actual,
        })
    return checks


def _run_single_suite_spec(spec: dict, idx: int, gpus: list[int]) -> dict:
    mode = spec.get("mode", "perception")
    src = Path(spec["source_run"]).resolve()
    cid, aid = _resolve_cutpoint_ids(
        src,
        until_call_id=spec.get("call_id") or spec.get("until_call_id"),
        until_action_id=spec.get("action_id") or spec.get("until_action_id"),
        action_index=spec.get("action_index"),
        cold_start=bool(spec.get("cold_start", False)),
    )
    if mode == "perception":
        res = probe_perception_slice(src, cid, config_overrides=spec.get("config_overrides"))
        res["mode"] = "perception"
        return res
    if mode == "turn_fork":
        res = probe_forked_turn(
            src,
            cid,
            max_probe_calls=int(spec.get("max_probe_calls", 4)),
            recompute_cutpoint=bool(spec.get("recompute_cutpoint", True)),
            config_overrides=spec.get("config_overrides"),
        )
        res["mode"] = "turn_fork"
        res["kind"] = "turn_fork"
        return res
    if mode == "live_slice":
        gpu_id = int(spec["gpu"]) if "gpu" in spec else gpus[idx % len(gpus)]
        res = run_live_slice(
            src,
            until_call_id=spec.get("call_id") or spec.get("until_call_id") or (cid if not aid else None),
            until_action_id=aid,
            cold_start=bool(spec.get("cold_start", False)),
            gpu_id=gpu_id,
            max_slice_actions=int(spec.get("max_slice_actions", 1)),
            max_turns=int(spec.get("max_turns", 2)),
            turn_timeout_s=float(spec.get("turn_timeout_s", 360.0)),
            recompute_cutpoint=bool(spec.get("recompute_cutpoint", True)),
            config_overrides=spec.get("config_overrides"),
        )
        res["mode"] = "live_slice"
        res["kind"] = "live_slice"
        res["gpu_id"] = gpu_id
        return res
    raise ValueError(f"未知切片模式: {mode}")


def run_slice_suite(suite_path: str | Path, *, max_workers: int = 4, gpus: list[int] | None = None) -> dict:
    """并发运行黄金切片回归集（支持 perception / turn_fork / live_slice 混合大批量并行），汇总对比与断言通过率。"""
    suite_path = Path(suite_path).resolve()
    suite = json.loads(suite_path.read_text())
    slices = suite.get("slices") or []
    gpu_list = gpus or [4, 5, 6, 7]
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [pool.submit(_run_single_suite_spec, spec, idx, gpu_list) for idx, spec in enumerate(slices)]
        results = [f.result() for f in futures]

    total_checks = 0
    passed_checks = 0
    for item, spec in zip(results, slices):
        cid_s = str(item.get("call_id") or item.get("until_call_id") or item.get("until_action_id") or "")[:12]
        item["slice_id"] = spec.get("slice_id") or f"{item['case']}@{cid_s}"
        item["description"] = spec.get("description", "")
        checks = _check_slice_assertions(item, spec.get("assertions"))
        item["checks"] = checks
        item["all_passed"] = all(c["passed"] for c in checks)
        total_checks += len(checks)
        passed_checks += sum(1 for c in checks if c["passed"])

    report = {
        "suite_name": suite.get("name", suite_path.stem),
        "suite_path": str(suite_path),
        "slice_count": len(results),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "all_passed": passed_checks == total_checks,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "results": results,
    }
    report["markdown"] = format_suite_markdown(report)
    return report


def format_suite_markdown(report: dict) -> str:
    """将黄金切片集回归报告格式化为 Markdown Diff 表。"""
    lines = [
        f"### 黄金切片回归报告：{report['suite_name']}",
        f"- **切片数**：`{report['slice_count']}` | **断言通过**：`{report['passed_checks']}/{report['total_checks']}` | **总耗时**：`{report['elapsed_s']}s`",
        "",
    ]
    perc_rows = [r for r in report["results"] if r.get("mode", "perception") == "perception"]
    if perc_rows:
        lines.extend([
            "| 切片 ID | 切点类型 | 物体数 (旧→新) | 标签数 (旧→新) | 双目命中 (旧→新) | MaxSkew (旧→新) | 悬浮未知 (旧→新) | 断言状态 |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for r in perc_rows:
            checks = r.get("checks") or []
            status = "PASS" if r.get("all_passed", True) else "FAIL: " + ", ".join(c["name"] for c in checks if not c["passed"])
            lines.append(
                f"| `{r['slice_id']}` | `{r['kind']}` "
                f"| `{r.get('old_row_count')} -> {r.get('new_row_count')}` "
                f"| `{r.get('old_labelled_rows')} -> {r.get('new_labelled_rows')}` "
                f"| `{r.get('old_stereo_count')} -> {r.get('new_stereo_count')}` "
                f"| `{r.get('old_max_skew_gap_m')} -> {r.get('new_max_skew_gap_m')}` "
                f"| `{r.get('old_floating_unknown')} -> {r.get('new_floating_unknown')}` "
                f"| {status} |"
            )
        lines.append("")
    exec_rows = [r for r in report["results"] if r.get("mode") in {"turn_fork", "live_slice"}]
    if exec_rows:
        lines.extend([
            "| 切片 ID | 模式 | 切点动作 | 耗时 | 规划/执行结果 | 工具报错数 | 分数变化 (前→后) | 断言状态 |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for r in exec_rows:
            checks = r.get("checks") or []
            status = "PASS" if r.get("all_passed", True) else "FAIL: " + ", ".join(c["name"] for c in checks if not c["passed"])
            err_cnt = int(r.get("tool_error_count") or 0)
            if r.get("mode") == "turn_fork":
                plans = r.get("previewed_plans") or []
                p_str = f"plans={len(plans)}, exec={r.get('executed_actions_count', 0)}"
                s_str = "N/A"
            else:
                acts = r.get("executed_slice_actions") or []
                a_statuses = [f"{a.get('action_id')}:{a.get('status')}" for a in acts]
                p_str = ", ".join(a_statuses) if a_statuses else "no_action"
                s_str = f"{r.get('score_before')} -> {r.get('score_after')} (Δ={r.get('delta_score')})"
            lines.append(
                f"| `{r['slice_id']}` | `{r.get('mode')}` | `{r.get('until_action_id') or r.get('call_id')}` "
                f"| `{r.get('elapsed_s')}s` | `{p_str}` | `{err_cnt}` | `{s_str}` | {status} |"
            )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auriga 中间态切片感知重算、黄金切片集回归、thread/fork 与真实物理闭环切片工具")
    parser.add_argument("--mode", choices=("perception", "turn_fork", "live_slice", "suite"), default="perception")
    parser.add_argument("--source-run", default=None, help="源归档实验 case 目录")
    parser.add_argument("--call-id", default=None, help="切点 tool_call ID")
    parser.add_argument("--action-id", default=None, help="切点 action_id（用于动作节点切片）")
    parser.add_argument("--action-index", type=int, default=None, help="切点动作序号（0=第1步动作后，-1=末步动作后）")
    parser.add_argument("--cold-start", action="store_true", help="在首帧 robot_observe 后切入，测试零历史惯性的冷启动推理")
    parser.add_argument("--suite", default=None, help="suite 模式下的黄金切片集 JSON 路径")
    parser.add_argument("--work-dir", default=None, help="turn_fork / live_slice 模式下的切片工作目录")
    parser.add_argument("--gpu", type=int, default=7, help="live_slice 模式使用的 GPU 编号")
    parser.add_argument("--gpus", default="4,5,6,7", help="suite 模式下并行分配 live_slice 的 GPU 列表，逗号分隔")
    parser.add_argument("--max-slice-actions", type=int, default=2, help="live_slice 模式最多执行的物理动作数（0=直到回合结束）")
    parser.add_argument("--max-turns", type=int, default=3, help="live_slice 模式最多进行的推理轮次数")
    parser.add_argument("--recompute-cutpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-probe-calls", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)

    if args.mode == "suite" or args.suite:
        if not args.suite:
            raise ValueError("suite 模式必须指定 --suite 路径")
        gpu_list = [int(x.strip()) for x in str(args.gpus).split(",") if x.strip()]
        res = run_slice_suite(args.suite, max_workers=args.max_workers, gpus=gpu_list)
        print(res["markdown"])
        return
    if not args.source_run:
        raise ValueError("必须指定 --source-run")
    if args.mode == "live_slice":
        res = run_live_slice(
            args.source_run,
            until_call_id=args.call_id,
            until_action_id=args.action_id,
            action_index=args.action_index,
            cold_start=args.cold_start,
            work_dir=args.work_dir,
            gpu_id=args.gpu,
            max_slice_actions=args.max_slice_actions,
            max_turns=args.max_turns,
            recompute_cutpoint=args.recompute_cutpoint,
        )
    elif args.mode == "perception":
        cid, _ = _resolve_cutpoint_ids(
            Path(args.source_run), until_call_id=args.call_id, until_action_id=args.action_id, action_index=args.action_index, cold_start=args.cold_start
        )
        res = probe_perception_slice(args.source_run, cid)
    else:
        cid, _ = _resolve_cutpoint_ids(
            Path(args.source_run), until_call_id=args.call_id, until_action_id=args.action_id, action_index=args.action_index, cold_start=args.cold_start
        )
        res = probe_forked_turn(
            args.source_run,
            cid,
            work_dir=args.work_dir,
            max_probe_calls=args.max_probe_calls,
            recompute_cutpoint=args.recompute_cutpoint,
        )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
