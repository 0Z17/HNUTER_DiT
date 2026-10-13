"""Reproducible headless MPPI flight and actuator-motion measurements.

Use a saved interactive plan rather than sampling again. ``--module-source``
can point to a previous copy of mujoco_execution.py for a before/after run.
Both variants execute the complete MuJoCo model and the same plan/MPPI seed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


TILT_JOINT_NAMES = ("rj2", "lj2", "rj1", "lj1")
TILT_ACTUATOR_NAMES = ("tilt_rj2", "tilt_lj2", "tilt_rj1", "tilt_lj1")
ROTOR_ACTUATOR_NAMES = ("motor_xy1", "motor_xy2", "motor_xy3", "motor_xy4")


def quaternion_roll_pitch_degrees(quaternions: Any) -> np.ndarray:
    """XYZ Euler roll/pitch from unit quaternions in wxyz convention."""
    q = np.asarray(quaternions, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.degrees(np.stack((
        np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
        np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)),
    ), axis=-1))


def series_motion_metrics(times: Any, values: Any) -> dict[str, Any]:
    """Motion/command variation, normalized by observed simulation duration.

    Total variation per second is the mean per channel; it separates a shorter
    flight from an actual reduction in servo activity. RMS finite-difference
    rates are weighted by sample intervals to handle nonuniform sampling.
    """
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or v.ndim != 2 or len(t) != len(v) or len(t) < 2:
        raise ValueError("需要至少两个时间点及对应的二维测量数组")
    dt = np.diff(t)
    if not np.isfinite(t).all() or not np.isfinite(v).all() or np.any(dt <= 0):
        raise ValueError("测量值必须有限，时间必须严格递增")
    delta = np.diff(v, axis=0)
    variation = np.abs(delta).sum(axis=0)
    duration = float(t[-1] - t[0])
    return {
        "total_variation_per_channel": variation.tolist(),
        "mean_total_variation_per_second": float(variation.mean() / duration),
        "rate_rms": float(np.sqrt(np.sum(delta**2 / dt[:, None]) / (duration * v.shape[1]))),
        "rate_p95": float(np.percentile(np.abs(delta / dt[:, None]), 95)),
        "maximum_step": float(np.abs(delta).max()),
    }


def summarize_flight(records: list[dict[str, Any]], final: dict[str, Any]) -> dict[str, Any]:
    if len(records) < 2:
        raise RuntimeError("仿真没有产生足够的动态记录: " + str(final.get("message")))
    times = np.asarray([row["time_s"] for row in records])
    angular = np.asarray([row["angular_velocity"] for row in records])
    reference_angular = np.asarray([row.get("reference_angular_velocity", [0, 0, 0]) for row in records])
    joint_velocity = np.asarray([row["tilt_joint_velocity"] for row in records])
    rotor = np.asarray([row["rotor_commands"] for row in records])
    attitude = quaternion_roll_pitch_degrees([row["pose"][3:] for row in records])
    desired_attitude = quaternion_roll_pitch_degrees([row["reference_pose"][3:] for row in records])
    attitude_error = (attitude - desired_attitude + 180) % 360 - 180
    rotor_imbalance = np.abs(rotor[:, :2].sum(axis=1) - rotor[:, 2:].sum(axis=1)) / np.maximum(rotor.sum(axis=1), 1e-9)
    weights = np.r_[np.diff(times), np.diff(times)[-1]]
    terminal = times >= times[-1] - 1.0
    return {
        **{key: final.get(key) for key in (
            "state", "success", "message", "contact_count", "goal_error_m",
            "attitude_error_deg", "position_rmse_m", "maximum_tracking_error_m",
            "retiming", "reference_audit", "options", "seed",
        )},
        "simulation_duration_s": float(times[-1]),
        "wall_runtime_s": float(final.get("elapsed_s", 0)),
        "sample_count": len(records),
        "roll_pitch_tracking_rmse_deg": np.sqrt(np.average(attitude_error**2, axis=0, weights=weights)).tolist(),
        "body_angular_velocity_rms_rps": float(np.sqrt(np.average(np.mean(angular**2, axis=1), weights=weights))),
        "body_angular_rate_tracking_rmse_rps": float(np.sqrt(np.average(np.mean((angular-reference_angular)**2, axis=1), weights=weights))) if all("reference_angular_velocity" in row for row in records) else None,
        "tilt_joint_velocity_rms_rps": float(np.sqrt(np.average(np.mean(joint_velocity**2, axis=1), weights=weights))),
        "tilt_joint_velocity_p95_rps": float(np.percentile(np.abs(joint_velocity), 95)),
        "terminal_1s_tilt_joint_velocity_rms_rps": float(np.sqrt(np.average(np.mean(joint_velocity[terminal]**2, axis=1), weights=weights[terminal]))),
        "terminal_1s_body_angular_velocity_rms_rps": float(np.sqrt(np.average(np.mean(angular[terminal]**2, axis=1), weights=weights[terminal]))),
        "tilt_joint_position": series_motion_metrics(times, [row["tilt_joint_position"] for row in records]),
        "tilt_command": series_motion_metrics(times, [row["tilt_commands"] for row in records]),
        "rotor_command": series_motion_metrics(times, rotor),
        "left_right_rotor_imbalance_mean_fraction": float(np.average(rotor_imbalance, weights=weights)),
        "left_right_rotor_imbalance_p95_fraction": float(np.percentile(rotor_imbalance, 95)),
        "measurement_note": "50 ms control-boundary samples; tilt joint rates come from full MuJoCo qvel, not command differences. Absolute bank/pitch is expected to follow the planned pose; attitude RMSE measures tracking.",
    }


def compare_flights(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare matched flights; never attribute a changed scene to the fix."""
    for key in ("scene_geometry_hash", "candidate_id", "seed"):
        if before.get(key) != after.get(key):
            raise ValueError(f"before/after 的 {key} 不一致")
    if before.get("plan_sha256") and after.get("plan_sha256"):
        if before["plan_sha256"] != after["plan_sha256"]:
            raise ValueError("before/after 的完整规划不一致")
    metrics = {
        "simulation_duration_s": (before["simulation_duration_s"], after["simulation_duration_s"]),
        "tilt_joint_velocity_rms_rps": (before["tilt_joint_velocity_rms_rps"], after["tilt_joint_velocity_rms_rps"]),
        "tilt_command_variation_per_second": (before["tilt_command"]["mean_total_variation_per_second"], after["tilt_command"]["mean_total_variation_per_second"]),
        "terminal_1s_tilt_joint_velocity_rms_rps": (before["terminal_1s_tilt_joint_velocity_rms_rps"], after["terminal_1s_tilt_joint_velocity_rms_rps"]),
        "goal_error_m": (before["goal_error_m"], after["goal_error_m"]),
        "position_rmse_m": (before["position_rmse_m"], after["position_rmse_m"]),
    }
    return {
        "same_scene_candidate_and_mppi_seed": True,
        "before_success": before["success"], "after_success": after["success"],
        "simulation_speedup_factor": before["simulation_duration_s"] / after["simulation_duration_s"],
        "metrics": {key: {
            "before": previous, "after": current,
            "change_percent": 100 * (current / previous - 1) if previous > 0 else None,
        } for key, (previous, current) in metrics.items()},
        "note": "Duration uses simulation time. Tilt command variation is normalized per second, and joint velocity is physical MuJoCo qvel. Wall runtime depends on host load.",
    }


def _load_execution(source: Path | None) -> Any:
    if source is None:
        from . import mujoco_execution
        return mujoco_execution
    name = "hnuter_dit._benchmark_execution_snapshot"
    specification = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def run_benchmark(plan: dict[str, Any], candidate_id: int, *, source: Path | None = None,
                  seed: int = 1, options: dict[str, Any] | None = None,
                  timeout_s: float = 600, asset_root: Path | None = None,
                  collector_root: Path | None = None) -> dict[str, Any]:
    source_path = source or Path(__file__).with_name("mujoco_execution.py")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    execution = _load_execution(source)
    dependency_hashes = {}
    for symbol in ("TimedPoseReference", "StabilizedFlightControl"):
        value = getattr(execution, symbol, None)
        module = sys.modules.get(value.__module__) if value is not None else None
        if module is not None and getattr(module, "__file__", None):
            dependency_hashes[symbol] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    trajectory = next(row for row in plan["trajectories"] if row["index"] == candidate_id)
    records: list[dict[str, Any]] = []
    controllers = []
    original_runtime = execution._external_runtime

    def recording_runtime(root):
        mujoco, controller_type, module = original_runtime(root)

        class RecordingController(controller_type):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                controllers.append((mujoco, self))

        return mujoco, RecordingController, module

    execution._external_runtime = recording_runtime

    def record(status):
        if "actuator_controls" not in status or not controllers:
            return
        now = float(status["simulation_time_s"])
        if records and now <= records[-1]["time_s"]:
            return
        mujoco, controller = controllers[-1]
        model, data = controller.model, controller.data
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in TILT_JOINT_NAMES]
        tilts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in TILT_ACTUATOR_NAMES]
        rotors = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ROTOR_ACTUATOR_NAMES]
        if min(*joints, *tilts, *rotors) < 0:
            raise RuntimeError("HNUTER 模型缺少 benchmark 要求的倾转关节或旋翼")
        records.append({
            "time_s": now, "pose": status["current_pose"],
            "reference_pose": status["reference_pose"],
            "angular_velocity": controller.get_mppi_pose_state()[10:13].tolist(),
            "reference_angular_velocity": controller.target_attitude_rate.tolist(),
            "reference_velocity": controller.target_velocity.tolist(),
            "mppi_commanded_acceleration": controller.target_acceleration.tolist(),
            "mppi_commanded_angular_acceleration": controller.target_angular_acceleration.tolist(),
            "tilt_joint_position": data.qpos[model.jnt_qposadr[joints]].tolist(),
            "tilt_joint_velocity": data.qvel[model.jnt_dofadr[joints]].tolist(),
            "tilt_commands": data.ctrl[tilts].tolist(),
            "rotor_commands": data.ctrl[rotors].tolist(),
            "tracking_error_m": status["tracking_error_m"],
        })

    manager = execution.MujocoExecutionManager(
        asset_root or execution.DEFAULT_MUJOCO_ROOT,
        viewer=False, realtime=False, on_update=record,
    )
    if hasattr(manager, "set_reference_validator"):
        # Match the interactive executor's smoothed-curve audit without
        # loading the diffusion checkpoint or regenerating the saved plan.
        from .interactive import InteractiveDiTService
        from .service import DEFAULT_COLLECTOR, ReferenceCollector
        collector = ReferenceCollector(collector_root or DEFAULT_COLLECTOR)
        base = SimpleNamespace(collector=collector, _checker=collector.collision_checker_type)
        InteractiveDiTService(base, manager)
    started = time.monotonic()
    try:
        manager.start(plan["scene"], trajectory["poses_wxyz"], seed=seed, options=options)
        while True:
            status = manager.status()
            if status["state"] in {"completed", "failed", "stopped"}:
                break
            if time.monotonic() - started > timeout_s:
                manager.stop()
                raise TimeoutError("MuJoCo benchmark 超过超时时间")
            time.sleep(.10)
        result = summarize_flight(records, status)
        result.update(candidate_id=candidate_id, scene_geometry_hash=execution.scene_geometry_hash(plan["scene"]),
                      source=str(source) if source else "current", source_sha256=source_hash,
                      dependency_source_sha256=dependency_hashes,
                      plan_sha256=plan_hash, joint_names=TILT_JOINT_NAMES, records=records)
        return result
    finally:
        manager.close()
        execution._external_runtime = original_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("artifacts/interactive_demo_smoke/plan.json"))
    parser.add_argument("--candidate", type=int)
    parser.add_argument("--module-source", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--options", default="{}", help="ExecutionConfig options as JSON")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--mujoco-root", type=Path)
    parser.add_argument("--collector-root", type=Path,
                        help="提供场景几何与 URDF 验碰的采集仓库路径")
    parser.add_argument("--baseline", type=Path, help="Matched before-run JSON for a normalized comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    candidate = args.candidate if args.candidate is not None else plan["final_candidate_id"]
    result = run_benchmark(plan, candidate, source=args.module_source, seed=args.seed,
                           options=json.loads(args.options), timeout_s=args.timeout_s,
                           asset_root=args.mujoco_root, collector_root=args.collector_root)
    if args.baseline:
        result["comparison"] = compare_flights(json.loads(args.baseline.read_text()), result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, ensure_ascii=False))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
