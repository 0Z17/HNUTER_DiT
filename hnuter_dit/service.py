"""Scene generation, DiT sampling, and exact trajectory audit service."""

from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any

import numpy as np
import torch

from .model import (
    DiffusionSchedule,
    build_model,
    ddim_sample,
    pose7_to_pose9,
    pose9_to_pose7_numpy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT / "artifacts" / "base_free_flight_159103traj_dit_v001"
    / "models" / "dit_cross" / "best.pt"
)
DEFAULT_COLLECTOR = Path("/home/z017/research/HNUTER_trajectory_collector")
EXPECTED_PARAMETER_COUNT = 11_964_169
WARNING_FRAGMENTS = (
    "路线", "证书", "route", "condition", "引导", "保护通道",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} 必须是有限的 {size} 维向量")
    return result


class ReferenceCollector:
    """Load the collector's authoritative free-flight domain functions."""

    def __init__(self, root: Path = DEFAULT_COLLECTOR) -> None:
        self.root = root.resolve()
        if not (self.root / "obstacle_scene_builder.py").is_file():
            raise FileNotFoundError(f"找不到轨迹采集仓库: {self.root}")
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from obstacle_scene_builder import (  # noqa: PLC0415
            FAMILY_BUILDERS,
            FAMILY_MINIMUMS,
            obstacle_tokens,
            quaternion_roll_pitch_degrees,
            validate_scene,
        )
        from expert_trajectory_collector.campaign.free_flight import (  # noqa: PLC0415
            generate_scene,
            sample_condition,
        )
        from obstacle_scene_experts import (  # noqa: PLC0415
            ConservativeURDFCollisionChecker,
        )

        self.families = tuple(sorted(FAMILY_BUILDERS))
        self.family_minimums = dict(FAMILY_MINIMUMS)
        self.generate_scene = generate_scene
        self.sample_condition = sample_condition
        self.obstacle_tokens = obstacle_tokens
        self.validate_scene = validate_scene
        self.quaternion_roll_pitch_degrees = quaternion_roll_pitch_degrees
        self.collision_checker_type = ConservativeURDFCollisionChecker


class DiTModelRuntime:
    def __init__(self, checkpoint_path: Path = DEFAULT_MODEL, device: str = "auto") -> None:
        self.checkpoint_path = checkpoint_path.resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到 DiT checkpoint: {self.checkpoint_path}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法访问 GPU")
        self.device = torch.device(
            "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        if checkpoint.get("model_type") != "dit_cross":
            raise RuntimeError("checkpoint 不是 environment-conditioned dit_cross")
        if checkpoint.get("prediction_type") != "velocity":
            raise RuntimeError("checkpoint prediction_type 不是 velocity")
        self.model = build_model(
            checkpoint["model_type"], **checkpoint["architecture"]
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.configure_environment_normalization(
            checkpoint["environment_feature_mean"],
            checkpoint["environment_feature_std"],
        )
        self.model.eval()
        self.parameter_count = sum(value.numel() for value in self.model.parameters())
        if self.parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                f"模型参数量 {self.parameter_count} 与预期 {EXPECTED_PARAMETER_COUNT} 不一致"
            )
        self.path_mean = np.asarray(checkpoint["path_mean"], dtype=np.float32)
        self.path_std = np.asarray(checkpoint["path_std"], dtype=np.float32)
        self.sequence_length = int(checkpoint["architecture"]["sequence_length"])
        self.diffusion_steps = int(checkpoint["diffusion_steps"])
        self.schedule = DiffusionSchedule.cosine(self.diffusion_steps, self.device)
        self.training_step = int(checkpoint["training_step"])
        self.validation_loss = float(checkpoint["validation_loss"])
        self.source_dataset = str(checkpoint["source_dataset"])
        self.checkpoint_sha256 = _sha256_file(self.checkpoint_path)
        status_path = self.checkpoint_path.parents[2] / "run_status.json"
        if status_path.is_file():
            expected = json.loads(status_path.read_text(encoding="utf-8")).get(
                "selected_checkpoint_sha256"
            )
            if expected and expected != self.checkpoint_sha256:
                raise RuntimeError("best.pt SHA-256 与 run_status.json 不一致")
        self.lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": "dit_cross",
            "preset": "small_environment",
            "parameter_count": self.parameter_count,
            "training_step": self.training_step,
            "validation_loss": self.validation_loss,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_dataset": self.source_dataset,
            "device": str(self.device),
            "cuda_device": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda" else None
            ),
            "trajectory_shape": [self.sequence_length, 9],
            "environment_tokens": [32, 10],
            "prediction_type": "velocity",
        }

    def sample(
        self, tokens: np.ndarray, mask: np.ndarray,
        start_pose: np.ndarray, goal_pose: np.ndarray,
        *, count: int, sampling_steps: int, seed: int,
    ) -> tuple[np.ndarray, float]:
        raw_condition = np.concatenate((
            pose7_to_pose9(start_pose), pose7_to_pose9(goal_pose)
        )).astype(np.float32)
        condition_mean = np.tile(self.path_mean, 2)
        condition_std = np.tile(self.path_std, 2)
        normalized = (raw_condition - condition_mean) / condition_std
        conditions = torch.from_numpy(
            np.repeat(normalized[None], count, axis=0)
        ).float().to(self.device)
        obstacle_tensor = torch.from_numpy(tokens).float().to(self.device)
        mask_tensor = torch.from_numpy(mask).bool().to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with self.lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            normalized_paths = ddim_sample(
                self.model, conditions, self.schedule, self.sequence_length,
                sampling_steps, obstacle_tensor, mask_tensor, generator,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
        paths9 = normalized_paths.cpu().numpy() * self.path_std + self.path_mean
        paths7 = pose9_to_pose7_numpy(paths9).astype(np.float32)
        paths7[:, 0] = start_pose
        paths7[:, -1] = goal_pose
        return paths7, elapsed


class DiTTrajectoryService:
    def __init__(
        self, checkpoint_path: Path = DEFAULT_MODEL,
        collector_root: Path = DEFAULT_COLLECTOR, device: str = "auto",
    ) -> None:
        self.collector = ReferenceCollector(collector_root)
        self.runtime = DiTModelRuntime(checkpoint_path, device)
        self._checker_cache: OrderedDict[str, Any] = OrderedDict()
        self._checker_lock = threading.Lock()

    def catalog(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "HNUTER DiT Trajectory Lab",
            "families": list(self.collector.families),
            "family_minimums": self.collector.family_minimums,
            "model": self.runtime.describe(),
            "workspace_bounds": {"min": [-4, -4, 0], "max": [4, 4, 4]},
            "sampling_bounds": {"min": [-3.3, -3.3, 0.65], "max": [3.3, 3.3, 3.5]},
        }

    @staticmethod
    def _scene_hash(scene: dict[str, Any]) -> str:
        geometry = [{
            "id": item.get("id"),
            "collision": item.get("collision", False),
            "role": item.get("role"),
            "type": item.get("type"),
            "pose": item.get("pose"),
            "size_xyz": item.get("size_xyz"),
        } for item in scene.get("obstacles", [])]
        return hashlib.sha256(json.dumps(
            geometry, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()

    def _checker(self, scene: dict[str, Any]) -> Any:
        key = self._scene_hash(scene)
        with self._checker_lock:
            if key in self._checker_cache:
                checker = self._checker_cache.pop(key)
                self._checker_cache[key] = checker
                return checker
            checker = self.collector.collision_checker_type(scene)
            self._checker_cache[key] = checker
            while len(self._checker_cache) > 8:
                self._checker_cache.popitem(last=False)
            return checker

    def generate_scene_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        seed = int(request.get("seed", random.SystemRandom().randrange(1, 2**31)))
        family = str(request.get("family", "random"))
        if family == "random":
            family = random.Random(f"family/{seed}").choice(self.collector.families)
        if family not in self.collector.families:
            raise ValueError(f"未知场景族: {family}")
        minimum = self.collector.family_minimums[family]
        count_min = max(minimum, int(request.get("obstacle_count_min", 8)))
        count_max = max(count_min, min(32, int(request.get("obstacle_count_max", 18))))
        last_error: Exception | None = None
        for attempt in range(12):
            scene_seed = seed + attempt
            try:
                scene = self.collector.generate_scene({
                    "family": family,
                    "seed": scene_seed,
                    "sample_ranges": True,
                    "obstacle_count_min": count_min,
                    "obstacle_count_max": count_max,
                    "size_min": float(request.get("size_min", 0.35)),
                    "size_max": float(request.get("size_max", 1.0)),
                    "gap_width_min": float(request.get("gap_width_min", 1.50)),
                    "gap_width_max": float(request.get("gap_width_max", 1.95)),
                    "global_yaw_min": float(request.get("global_yaw_min", -35)),
                    "global_yaw_max": float(request.get("global_yaw_max", 35)),
                    "translation_max": float(request.get("translation_max", 0.30)),
                })
                scene = self.collector.sample_condition(
                    scene, 0, seed + 7919, maximum_attempts=64, timeout_s=4.0
                )
                return self._present_scene(scene)
            except ValueError as error:
                last_error = error
        raise ValueError(f"连续 12 个 seed 未生成有效场景: {last_error}")

    def randomize_endpoints(self, request: dict[str, Any]) -> dict[str, Any]:
        scene = copy.deepcopy(self._required_scene(request))
        seed = int(request.get("seed", random.SystemRandom().randrange(1, 2**31)))
        condition = self.collector.sample_condition(
            scene, int(request.get("condition_index", 0)), seed,
            maximum_attempts=96, timeout_s=6.0,
        )
        return self._present_scene(condition)

    @staticmethod
    def _required_scene(request: dict[str, Any]) -> dict[str, Any]:
        scene = request.get("scene")
        if not isinstance(scene, dict):
            raise ValueError("请求缺少 scene JSON 对象")
        return scene

    def _present_scene(self, scene: dict[str, Any]) -> dict[str, Any]:
        pair = (scene.get("precheck_pairs") or [{}])[0]
        return {
            "scene": scene,
            "start_pose": pair.get("start_pose"),
            "goal_pose": pair.get("goal_pose"),
            "validation": self.validate({
                "scene": scene,
                "start_pose": pair.get("start_pose"),
                "goal_pose": pair.get("goal_pose"),
            }),
        }

    def _validate_pose(
        self, pose: Any, label: str, scene: dict[str, Any], checker: Any,
    ) -> tuple[np.ndarray | None, list[str]]:
        issues: list[str] = []
        try:
            value = _finite_vector(pose, 7, label)
        except ValueError as error:
            return None, [str(error)]
        quaternion_norm = float(np.linalg.norm(value[3:7]))
        if abs(quaternion_norm - 1) > 1e-3:
            issues.append(f"{label} 四元数未归一化")
        else:
            value[3:7] /= quaternion_norm
        bounds = scene.get("sampling_space", {}).get("position_bounds", {
            "min": [-3.3, -3.3, 0.65], "max": [3.3, 3.3, 3.5]
        })
        lower = np.asarray(bounds["min"], dtype=np.float64)
        upper = np.asarray(bounds["max"], dtype=np.float64)
        if np.any(value[:3] < lower) or np.any(value[:3] > upper):
            issues.append(f"{label} 位置超出采样边界")
        if not issues and not checker.is_collision_free(value[:3], value[3:7]):
            issues.append(f"{label} 与障碍物碰撞或姿态超限")
        return value, issues

    def validate(self, request: dict[str, Any]) -> dict[str, Any]:
        scene = self._required_scene(request)
        scene_issues = list(self.collector.validate_scene(scene))
        warnings = [
            item for item in scene_issues
            if any(fragment in item for fragment in WARNING_FRAGMENTS)
        ]
        issues = [item for item in scene_issues if item not in warnings]
        checker = self._checker(scene)
        start, start_issues = self._validate_pose(
            request.get("start_pose"), "起点", scene, checker
        )
        goal, goal_issues = self._validate_pose(
            request.get("goal_pose"), "终点", scene, checker
        )
        issues.extend(start_issues)
        issues.extend(goal_issues)
        if start is not None and goal is not None:
            if np.linalg.norm(start[:3] - goal[:3]) < 0.5:
                issues.append("起点和终点距离必须至少为 0.5 m")
        tokens, mask = self.collector.obstacle_tokens(scene)
        return {
            "valid": not issues,
            "issues": issues,
            "warnings": warnings,
            "active_tokens": int(sum(mask)),
            "collision_backend": checker.backend_name,
        }

    def sample_trajectories(self, request: dict[str, Any]) -> dict[str, Any]:
        scene = self._required_scene(request)
        count = int(request.get("count", 4))
        sampling_steps = int(request.get("sampling_steps", 25))
        seed = int(request.get("seed", 1))
        if not 1 <= count <= 16:
            raise ValueError("轨迹数量必须在 [1, 16]")
        if not 2 <= sampling_steps <= self.runtime.diffusion_steps:
            raise ValueError(
                f"DDIM steps 必须在 [2, {self.runtime.diffusion_steps}]"
            )
        validation = self.validate(request)
        if not validation["valid"]:
            raise ValueError("场景或端点无效: " + "; ".join(validation["issues"]))
        start = _finite_vector(request["start_pose"], 7, "起点").astype(np.float32)
        goal = _finite_vector(request["goal_pose"], 7, "终点").astype(np.float32)
        start[3:7] /= np.linalg.norm(start[3:7])
        goal[3:7] /= np.linalg.norm(goal[3:7])
        token_values, mask_values = self.collector.obstacle_tokens(scene)
        tokens = np.asarray(token_values, dtype=np.float32)
        mask = np.asarray(mask_values, dtype=bool)
        paths, elapsed = self.runtime.sample(
            tokens, mask, start, goal, count=count,
            sampling_steps=sampling_steps, seed=seed,
        )
        checker = self._checker(scene)
        bounds = scene.get("sampling_space", {}).get("position_bounds", {
            "min": [-3.3, -3.3, 0.65], "max": [3.3, 3.3, 3.5]
        })
        lower = np.asarray(bounds["min"], dtype=np.float64)
        upper = np.asarray(bounds["max"], dtype=np.float64)
        trajectories: list[dict[str, Any]] = []
        for index, path in enumerate(paths):
            clearance = checker.clearance(path[:, :3], path[:, 3:7])
            attitudes = np.asarray([
                self.collector.quaternion_roll_pitch_degrees(quaternion)
                for quaternion in path[:, 3:7]
            ])
            in_bounds = np.all((path[:, :3] >= lower) & (path[:, :3] <= upper), axis=1)
            attitude_valid = (
                (np.abs(attitudes[:, 0]) <= 40.0)
                & (np.abs(attitudes[:, 1]) <= 70.0)
            )
            collision_free = clearance > 0
            valid = bool(np.all(collision_free & in_bounds & attitude_valid))
            length = float(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1).sum())
            trajectories.append({
                "index": index,
                "poses_wxyz": path.tolist(),
                "clearance_m": clearance.tolist(),
                "metrics": {
                    "valid": valid,
                    "minimum_clearance_m": float(np.min(clearance)),
                    "path_length_m": length,
                    "collision_sample_count": int(np.count_nonzero(~collision_free)),
                    "out_of_bounds_sample_count": int(np.count_nonzero(~in_bounds)),
                    "attitude_violation_sample_count": int(np.count_nonzero(~attitude_valid)),
                    "maximum_abs_roll_deg": float(np.max(np.abs(attitudes[:, 0]))),
                    "maximum_abs_pitch_deg": float(np.max(np.abs(attitudes[:, 1]))),
                },
            })
        valid_count = sum(item["metrics"]["valid"] for item in trajectories)
        return {
            "schema_version": "hnuter_dit_web_predictions_v001",
            "scene_id": scene.get("environment_id", "edited_scene"),
            "start_pose": start.tolist(),
            "goal_pose": goal.tolist(),
            "trajectories": trajectories,
            "summary": {
                "requested_count": count,
                "valid_count": valid_count,
                "valid_rate": valid_count / count,
                "sampling_steps": sampling_steps,
                "seed": seed,
                "neural_sampling_time_s": elapsed,
                "milliseconds_per_path": 1000 * elapsed / count,
                "collision_backend": checker.backend_name,
                "model_device": str(self.runtime.device),
            },
            "validation": validation,
        }
