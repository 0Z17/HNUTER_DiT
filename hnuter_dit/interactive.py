"""Interactive Base DiT best-of-eight planning and MuJoCo MPPI execution."""

from __future__ import annotations

import argparse
import copy
from http.server import ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlsplit
import uuid
import webbrowser

import numpy as np

from .service import DEFAULT_COLLECTOR, DEFAULT_MODEL, DiTTrajectoryService
from .web import DiTWebHandler, WEB_ROOT


def _require_measured_clearance(checker: Any) -> None:
    """The collector's OBB fallback returns placeholders, not metric distances."""
    if (getattr(checker, "supports_clearance_gradient", None) is False
            or checker.backend_name == "urdf_primitive_obb_conservative"):
        raise RuntimeError(
            "交互动力学执行需要 COAL 真实净空后端；当前 OBB 后备后端的 0.10 m "
            "是占位值，不能用于执行复检。请安装 simulation 依赖并重新启动。"
        )


def densify_poses(
    poses: Any, max_translation_step_m: float = 0.02,
    max_rotation_step_deg: float = 3.0,
) -> np.ndarray:
    """Audit the same straight segments/geodesic rotations used by execution.

    This is a sampled geometric audit, not a continuous collision certificate.
    Quaternion interpolation follows the shortest arc in wxyz convention.
    """
    path = np.asarray(poses, dtype=np.float64).copy()
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2:
        raise ValueError("轨迹必须是至少两个 waypoint 的 [N, 7] pose7 数组")
    if not np.isfinite(path).all():
        raise ValueError("轨迹包含非有限值")
    if not (math.isfinite(max_translation_step_m) and max_translation_step_m > 0
            and math.isfinite(max_rotation_step_deg) and max_rotation_step_deg > 0):
        raise ValueError("复检的平移和旋转间距必须为有限正数")
    norms = np.linalg.norm(path[:, 3:], axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        raise ValueError("轨迹四元数必须归一化")
    path[:, 3:] /= norms[:, None]
    dense = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        qa, qb = start[3:].copy(), end[3:].copy()
        dot = float(qa @ qb)
        if dot < 0:
            qb = -qb
            dot = -dot
        theta = math.acos(float(np.clip(dot, -1, 1)))
        steps = max(
            1, math.ceil(float(np.linalg.norm(end[:3] - start[:3])) / max_translation_step_m),
            math.ceil(math.degrees(2 * theta) / max_rotation_step_deg),
        )
        if steps > 10000 or len(dense) + steps > 100000:
            raise ValueError("轨迹复检采样过大")
        for fraction in np.arange(1, steps + 1) / steps:
            position = start[:3] + fraction * (end[:3] - start[:3])
            if theta < 1e-6:
                quaternion = (1 - fraction) * qa + fraction * qb
            else:
                quaternion = (
                    math.sin((1 - fraction) * theta) * qa
                    + math.sin(fraction * theta) * qb
                ) / math.sin(theta)
            quaternion /= np.linalg.norm(quaternion)
            dense.append(np.concatenate((position, quaternion)))
    return np.asarray(dense)


def rank_candidates(
    result: dict[str, Any], scene: dict[str, Any], base_service: Any,
    minimum_clearance_m: float = 0.02,
) -> dict[str, Any]:
    """Reject invalid/interpolated collisions, rank remaining paths deterministically."""
    checker = base_service._checker(scene)
    _require_measured_clearance(checker)
    bounds = scene.get("sampling_space", {}).get("position_bounds", {
        "min": [-3.3, -3.3, 0.65], "max": [3.3, 3.3, 3.5],
    })
    lower, upper = np.asarray(bounds["min"]), np.asarray(bounds["max"])
    records = []
    for trajectory in result["trajectories"]:
        candidate_id = int(trajectory["index"])
        trajectory["id"] = candidate_id
        metrics = trajectory["metrics"]
        record = {
            "id": candidate_id, "eligible": False, "score": None,
            "reason": "原始 waypoint 未通过验碰、边界或姿态检查",
            "dense_sample_count": 0, "minimum_dense_clearance_m": None,
        }
        if metrics["valid"]:
            dense = densify_poses(trajectory["poses_wxyz"])
            endpoints_match = True
            for pose, label in ((dense[0], "start_pose"), (dense[-1], "goal_pose")):
                expected = np.asarray(result[label], dtype=np.float64)
                endpoints_match &= bool(
                    np.allclose(pose[:3], expected[:3], atol=1e-5, rtol=0)
                    and abs(float(pose[3:] @ expected[3:])) > 1 - 1e-5
                )
            clearance = np.asarray(checker.clearance(dense[:, :3], dense[:, 3:]))
            if clearance.shape != (len(dense),) or not np.isfinite(clearance).all():
                raise RuntimeError("碰撞复检没有返回有限净空值")
            attitudes = np.asarray([
                base_service.collector.quaternion_roll_pitch_degrees(q) for q in dense[:, 3:]
            ])
            in_bounds = bool(np.all((dense[:, :3] >= lower) & (dense[:, :3] <= upper)))
            attitude_valid = bool(np.all(
                (np.abs(attitudes[:, 0]) <= 40) & (np.abs(attitudes[:, 1]) <= 70)
            ))
            minimum = float(clearance.min())
            eligible = endpoints_match and in_bounds and attitude_valid and minimum > minimum_clearance_m
            record.update(
                eligible=eligible, dense_sample_count=len(dense),
                minimum_dense_clearance_m=minimum,
            )
            if eligible:
                # All terms are independent of candidate order; original index
                # breaks ties. Clearance is capped to avoid rewarding long detours.
                score = (
                    float(metrics["path_length_m"])
                    + 0.015 * float(metrics.get("p95_turn_angle_deg", 0))
                    + 0.005 * float(metrics.get("maximum_turn_angle_deg", 0))
                    + 2.0 * float(metrics.get("backtracking_fraction", 0))
                    - 0.3 * min(minimum, 0.3)
                )
                record.update(score=score, reason="通过 waypoint 与加密 SE(3) 复检")
            elif not endpoints_match:
                record["reason"] = "轨迹起终点与规划请求不一致"
            elif not in_bounds:
                record["reason"] = "插值轨迹越界"
            elif not attitude_valid:
                record["reason"] = "插值姿态超限"
            else:
                record["reason"] = f"插值最小净空 {minimum:.4f} m 未超过 {minimum_clearance_m:.3f} m"
        trajectory["execution_audit"] = copy.deepcopy(record)
        records.append(record)
    eligible_records = [item for item in records if item["eligible"]]
    final = min(eligible_records, key=lambda item: (item["score"], item["id"])) if eligible_records else None
    result["final_candidate_id"] = final["id"] if final else None
    result["selection"] = {
        "message": f"最终轨迹 #{final['id'] + 1}：通过加密复检后综合长度、转角、回退与净空选择" if final else "本批次无可执行候选，请更换推理 seed 或调整场景/端点后重新生成",
        "candidates": records,
        "minimum_clearance_m": minimum_clearance_m,
        "translation_step_m": 0.02, "rotation_step_deg": 3.0,
        "audit_type": "sampled SE(3), not continuous certification",
        "score_formula": "length_m + 0.015*p95_turn_deg + 0.005*max_turn_deg + 2*backtracking - 0.3*min(clearance_m,0.3)",
    }
    return result


class InteractiveDiTService:
    """Keep an immutable server-side plan paired with its exact world geometry."""

    def __init__(
        self, base_service: Any, execution: Any, *, scene_seed: int = 42,
        execution_seed: int = 1, execution_options: Any = None,
        minimum_clearance_m: float = 0.02,
    ) -> None:
        if not math.isfinite(minimum_clearance_m) or minimum_clearance_m < 0:
            raise ValueError("minimum clearance 必须为有限非负数")
        self.base = base_service
        self.execution = execution
        self.scene_seed = scene_seed
        self.execution_seed = execution_seed
        self.execution_options = execution_options
        self.minimum_clearance_m = minimum_clearance_m
        self._lock = threading.RLock()
        self._plan: dict[str, Any] | None = None
        self._execution_plan_id: str | None = None
        if hasattr(execution, "set_reference_validator"):
            execution.set_reference_validator(self.validate_execution_reference)

    def validate_execution_reference(
        self, scene: dict[str, Any], poses: np.ndarray,
    ) -> dict[str, Any]:
        """Recheck the C2 curve that TOPP-RA/MPPI actually executes."""
        with self._lock:
            dense = densify_poses(poses, 0.01, 1.0)
            checker = self.base._checker(scene)
            _require_measured_clearance(checker)
            clearance = np.asarray(checker.clearance(dense[:, :3], dense[:, 3:]))
            bounds = scene.get("sampling_space", {}).get("position_bounds", {
                "min": [-3.3, -3.3, 0.65], "max": [3.3, 3.3, 3.5],
            })
            attitudes = np.asarray([
                self.base.collector.quaternion_roll_pitch_degrees(q) for q in dense[:, 3:]
            ])
            if not np.all((dense[:, :3] >= bounds["min"]) & (dense[:, :3] <= bounds["max"])):
                raise ValueError("TOPP-RA 平滑参考超出规划边界，请选择其他候选")
            if np.any(np.abs(attitudes[:, 0]) > 40) or np.any(np.abs(attitudes[:, 1]) > 70):
                raise ValueError("TOPP-RA 平滑参考姿态超限，请选择其他候选")
            if not np.isfinite(clearance).all() or float(clearance.min()) <= self.minimum_clearance_m:
                raise ValueError("TOPP-RA 平滑参考未通过净空复检，请选择其他候选或重新规划")
            return {
                "checked": True, "collision_backend": checker.backend_name,
                "sample_count": len(dense), "minimum_clearance_m": float(clearance.min()),
                "translation_step_m": 0.01, "rotation_step_deg": 1.0,
            }

    def catalog(self) -> dict[str, Any]:
        result = self.base.catalog()
        result["execution"] = {
            "enabled": True, "batch_size": 8, "guidance_mode": "smooth",
            "guidance_steps": 8, "initial_scene_seed": self.scene_seed,
            "controller": "MPPI + HNUTER actuator allocation",
            "retiming": "TOPP-RA continuous SE(3)",
            "minimum_clearance_m": self.minimum_clearance_m,
            "viewer": bool(getattr(self.execution, "viewer_enabled", True)),
        }
        result["initial_scene_seed"] = self.scene_seed
        return result

    def _editable(self) -> None:
        if self.execution.status()["state"] in {"starting", "running", "stopping"}:
            raise RuntimeError("MPPI 正在执行，请先停止再修改场景或重新规划")

    @staticmethod
    def _condition_key(request: dict[str, Any]) -> str:
        return json.dumps({key: request.get(key) for key in ("scene", "start_pose", "goal_pose")}, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def generate_scene_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._editable()
            result = self.base.generate_scene_payload(request)
            self._plan = None
            return result

    def randomize_endpoints(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._editable()
            result = self.base.randomize_endpoints(request)
            self._plan = None
            return result

    def validate(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._editable()
            if (self._plan
                    and request.get("invalidate_plan_id") == self._plan["result"]["plan_id"]
                    and self._condition_key(request) != self._plan["condition_key"]):
                self._plan = None
            return self.base.validate(request)

    def sample_trajectories(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._editable()
            self._plan = None
            request = copy.deepcopy(request)
            request.update(count=8, guidance_mode="smooth", guidance_steps=8)
            result = self.base.sample_trajectories(request)
            if len(result["trajectories"]) != 8:
                raise RuntimeError("Base DiT 未返回完整的一批 8 条候选")
            for label in ("start_pose", "goal_pose"):
                actual = np.asarray(result[label], dtype=np.float64)
                expected = np.asarray(request[label], dtype=np.float64)
                if (actual.shape != (7,) or expected.shape != (7,)
                        or not np.isfinite(actual).all() or not np.isfinite(expected).all()
                        or not np.allclose(actual[:3], expected[:3], atol=1e-5, rtol=0)
                        or abs(float(actual[3:] @ expected[3:])) < 1 - 1e-5):
                    raise RuntimeError("Base DiT 返回的起终点与当前请求不一致")
            rank_candidates(result, request["scene"], self.base, self.minimum_clearance_m)
            result["plan_id"] = uuid.uuid4().hex
            result["scene"] = copy.deepcopy(request["scene"])
            self._plan = {
                "condition_key": self._condition_key(request),
                "scene": copy.deepcopy(request["scene"]),
                "result": copy.deepcopy(result),
            }
            return result

    def _require_plan(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._plan is None or request.get("plan_id") != self._plan["result"]["plan_id"]:
            raise ValueError("规划已失效或 plan_id 不匹配，请先生成当前场景的 8 条候选")
        return self._plan

    @staticmethod
    def _candidate(plan: dict[str, Any], candidate_id: Any) -> dict[str, Any]:
        result = plan["result"]
        candidate = next((item for item in result["trajectories"] if item["index"] == candidate_id), None)
        if candidate is None or not candidate["execution_audit"]["eligible"]:
            raise ValueError("该候选未通过执行复检，无法作为最终轨迹")
        return candidate

    def select_plan(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._editable()
            plan = self._require_plan(request)
            candidate = self._candidate(plan, request.get("candidate_id"))
            plan["result"]["final_candidate_id"] = candidate["index"]
            plan["result"]["selection"]["message"] = f"手工选择最终轨迹 #{candidate['index'] + 1}，已通过加密复检"
            return copy.deepcopy(plan["result"])

    def preview_execution(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._editable()
            validation = self.validate({
                **request,
                "invalidate_plan_id": self._plan["result"]["plan_id"] if self._plan else None,
            })
            if not validation["valid"]:
                raise ValueError("场景或端点无效: " + "; ".join(validation["issues"]))
            self.execution.preview(copy.deepcopy(request["scene"]), request["start_pose"], request["goal_pose"])
            self._execution_plan_id = None
            return self.execution_status()

    def start_execution(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._editable()
            plan = self._require_plan(request)
            candidate = self._candidate(plan, request.get("candidate_id", plan["result"]["final_candidate_id"]))
            self.execution.start(
                copy.deepcopy(plan["scene"]), copy.deepcopy(candidate["poses_wxyz"]),
                candidates=[copy.deepcopy(item["poses_wxyz"]) for item in plan["result"]["trajectories"]],
                seed=self.execution_seed, options=self.execution_options,
            )
            plan["result"]["final_candidate_id"] = candidate["index"]
            self._execution_plan_id = plan["result"]["plan_id"]
            return self.execution_status()

    def stop_execution(self, request: dict[str, Any]) -> dict[str, Any]:
        del request
        self.execution.stop()
        return self.execution_status()

    def execution_status(self) -> dict[str, Any]:
        result = self.execution.status()
        result["plan_id"] = self._execution_plan_id
        result.setdefault("elapsed_s", result.get("simulation_time_s", 0))
        return result

    def close(self) -> None:
        self.execution.close()


class InteractiveWebHandler(DiTWebHandler):
    service: InteractiveDiTService

    def log_message(self, format: str, *args: Any) -> None:
        if urlsplit(self.path).path != "/api/execution/status":
            super().log_message(format, *args)

    def do_GET(self) -> None:  # noqa: N802
        if urlsplit(self.path).path == "/api/execution/status":
            self._json(200, self.service.execution_status())
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        routes = {
            "/api/plan/select": self.service.select_plan,
            "/api/execution/preview": self.service.preview_execution,
            "/api/execution/start": self.service.start_execution,
            "/api/execution/stop": self.service.stop_execution,
        }
        operation = routes.get(urlsplit(self.path).path)
        if operation is None:
            super().do_POST()
            return
        try:
            self._json(200, operation(self._payload()))
        except (ValueError, KeyError, TypeError, RuntimeError, ImportError, OSError) as error:
            self._json(400, {"error": str(error)})


def handler_type(service: InteractiveDiTService) -> type[InteractiveWebHandler]:
    class BoundHandler(InteractiveWebHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)
    BoundHandler.service = service
    return BoundHandler


def main() -> None:
    from .mujoco_execution import DEFAULT_MUJOCO_ROOT, ExecutionConfig, MujocoExecutionManager

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--collector-root", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument("--mujoco-root", type=Path, default=DEFAULT_MUJOCO_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4, help="CPU 推理线程数，避免小批次过多线程开销")
    parser.add_argument("--scene-seed", type=int, default=42)
    parser.add_argument("--execution-seed", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1.0, help="TOPP-RA 最大参考速度 m/s")
    parser.add_argument("--max-accel", type=float, default=1.5, help="TOPP-RA 最大参考加速度 m/s²")
    parser.add_argument("--angular-speed", type=float, default=0.7, help="最大参考角速度 rad/s")
    parser.add_argument("--angular-accel", type=float, default=1.5, help="最大参考角加速度 rad/s²")
    parser.add_argument("--mppi-samples", type=int, default=192)
    parser.add_argument("--mppi-horizon", type=int, default=32)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--minimum-clearance", type=float, default=0.02)
    parser.add_argument("--headless", action="store_true", help="真实动力学仿真，但不打开 MuJoCo 窗口")
    parser.add_argument("--no-realtime", action="store_true", help="不按墙钟时间限速")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads 必须为正整数")
    import torch
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        torch.set_num_threads(args.cpu_threads)
    options = ExecutionConfig(
        speed_mps=args.speed, num_samples=args.mppi_samples,
        horizon=args.mppi_horizon, control_dt=args.control_dt,
        max_acceleration_mps2=args.max_accel, max_angular_speed_rps=args.angular_speed,
        max_angular_acceleration_rps2=args.angular_accel,
    )
    options.validate()
    execution = MujocoExecutionManager(
        asset_root=args.mujoco_root, viewer=not args.headless,
        realtime=not args.no_realtime,
    )
    service = InteractiveDiTService(
        DiTTrajectoryService(args.checkpoint, args.collector_root, args.device),
        execution, scene_seed=args.scene_seed, execution_seed=args.execution_seed,
        execution_options=options, minimum_clearance_m=args.minimum_clearance,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_type(service))
    print(f"Base DiT + MuJoCo MPPI: http://{args.host}:{args.port}\n"
          f"batch=8 guidance=smooth/8 device={service.base.runtime.device} "
          f"scene_seed={args.scene_seed} execution_seed={args.execution_seed}", flush=True)
    if args.open_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
