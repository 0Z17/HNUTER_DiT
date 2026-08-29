"""Controlled v001-v002 evaluation on the frozen test split and 90° rotations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .service import DiTModelRuntime, ReferenceCollector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/home/z017/research/HNUTER_datasets/free_flight_159103traj_dit_v001")
DEFAULT_BASELINE = ROOT / "artifacts/base_free_flight_159103traj_dit_v001/models/dit_cross/best.pt"
DEFAULT_CANDIDATE = ROOT / "artifacts/base_free_flight_159103traj_dit_relgeom_v002/models/dit_rel_cross/best.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/base_free_flight_159103traj_dit_relgeom_v002/comparison_v001_vs_v002.json"


def quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(first), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(second), -1, 0)
    return np.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), axis=-1)


def rotate_pose7(value: np.ndarray, angle: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = value.copy()
    cosine, sine = math.cos(angle), math.sin(angle)
    result[..., 0] = cosine * value[..., 0] - sine * value[..., 1]
    result[..., 1] = sine * value[..., 0] + cosine * value[..., 1]
    qz = np.asarray([math.cos(0.5 * angle), 0, 0, math.sin(0.5 * angle)])
    result[..., 3:7] = quaternion_multiply(qz, value[..., 3:7])
    return result.astype(np.float32)


def rotate_scene(scene: dict[str, Any], angle: float) -> dict[str, Any]:
    result = copy.deepcopy(scene)
    cosine, sine = math.cos(angle), math.sin(angle)
    qz = np.asarray([math.cos(0.5 * angle), 0, 0, math.sin(0.5 * angle)])
    for obstacle in result.get("obstacles", []):
        pose = obstacle.get("pose", {})
        if "position" not in pose or "quaternion_wxyz" not in pose:
            continue
        position = np.asarray(pose["position"], dtype=np.float64)
        pose["position"] = [
            float(cosine * position[0] - sine * position[1]),
            float(sine * position[0] + cosine * position[1]),
            float(position[2]),
        ]
        pose["quaternion_wxyz"] = quaternion_multiply(
            qz, np.asarray(pose["quaternion_wxyz"])
        ).tolist()
    result["environment_id"] = str(result.get("environment_id", "scene")) + "_yaw90"
    return result


def select_conditions(dataset: Path, count: int, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (
        dataset / "conditions/index.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            groups[str(row["family"])].append(row)
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    families = sorted(groups)
    quotient, remainder = divmod(count, len(families))
    for index, family in enumerate(families):
        take = quotient + (index < remainder)
        candidates = groups[family]
        choice = rng.choice(len(candidates), size=min(take, len(candidates)), replace=False)
        selected.extend(candidates[int(item)] for item in choice)
    return sorted(selected, key=lambda row: row["condition_id"])


def trajectory_metrics(path: np.ndarray, scene: dict[str, Any], checker: Any,
                       collector: ReferenceCollector) -> dict[str, Any]:
    clearance = checker.clearance(path[:, :3], path[:, 3:7])
    attitudes = np.asarray([
        collector.quaternion_roll_pitch_degrees(value) for value in path[:, 3:7]
    ])
    bounds = scene["sampling_space"]["position_bounds"]
    lower, upper = np.asarray(bounds["min"]), np.asarray(bounds["max"])
    in_bounds = np.all((path[:, :3] >= lower) & (path[:, :3] <= upper), axis=1)
    attitude_valid = (np.abs(attitudes[:, 0]) <= 40) & (np.abs(attitudes[:, 1]) <= 70)
    collision_free = clearance > 0
    return {
        "physical_collision_free": bool(np.all(collision_free)),
        "valid": bool(np.all(collision_free & in_bounds & attitude_valid)),
        "minimum_clearance_m": float(clearance.min()),
        "path_length_m": float(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1).sum()),
    }


def summarize(records: list[dict[str, Any]], model_name: str, scenario: str) -> dict[str, Any]:
    subset = [row for row in records if row["model"] == model_name and row["scenario"] == scenario]
    conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        conditions[row["condition_id"]].append(row)
    return {
        "candidate_count": len(subset),
        "condition_count": len(conditions),
        "physical_collision_free_rate": float(np.mean([row["physical_collision_free"] for row in subset])),
        "valid_rate": float(np.mean([row["valid"] for row in subset])),
        "best_of_n_task_solved_rate": float(np.mean([
            any(item["valid"] for item in group) for group in conditions.values()
        ])),
        "mean_minimum_clearance_m": float(np.mean([row["minimum_clearance_m"] for row in subset])),
        "mean_path_length_m": float(np.mean([row["path_length_m"] for row in subset])),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    collector = ReferenceCollector()
    runtimes = {
        "v001_absolute_cross_attention": DiTModelRuntime(args.baseline, args.device),
        "v002_canonical_relative_geometry": DiTModelRuntime(args.candidate, args.device),
    }
    environments = {
        row["environment_index"]: row
        for row in (
            json.loads(line) for line in (
                args.dataset / "environments/index.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        )
    }
    conditions = select_conditions(args.dataset, args.condition_count, args.seed)
    records: list[dict[str, Any]] = []
    equivariance: dict[str, list[tuple[float, float]]] = defaultdict(list)
    angle = math.pi / 2
    for condition_index, condition in enumerate(conditions):
        environment_row = environments[int(condition["environment_index"])]
        scene = json.loads(
            (args.dataset / environment_row["path"]).read_text(encoding="utf-8")
        )
        rotated_scene = rotate_scene(scene, angle)
        start = np.asarray(condition["start_pose7_wxyz"], dtype=np.float32)
        goal = np.asarray(condition["goal_pose7_wxyz"], dtype=np.float32)
        rotated_start, rotated_goal = rotate_pose7(start, angle), rotate_pose7(goal, angle)
        scene_tokens, scene_mask = collector.obstacle_tokens(scene)
        rotated_tokens, rotated_mask = collector.obstacle_tokens(rotated_scene)
        checker = collector.collision_checker_type(scene)
        rotated_checker = collector.collision_checker_type(rotated_scene)
        sample_seed = args.seed + 1009 * condition_index
        for model_name, runtime in runtimes.items():
            original, _ = runtime.sample(
                np.asarray(scene_tokens, dtype=np.float32), np.asarray(scene_mask, dtype=bool),
                start, goal, count=args.samples_per_condition,
                sampling_steps=args.ddim_steps, seed=sample_seed,
            )
            rotated, _ = runtime.sample(
                np.asarray(rotated_tokens, dtype=np.float32), np.asarray(rotated_mask, dtype=bool),
                rotated_start, rotated_goal, count=args.samples_per_condition,
                sampling_steps=args.ddim_steps, seed=sample_seed,
            )
            restored = rotate_pose7(rotated, -angle)
            for sample_index in range(args.samples_per_condition):
                base = {
                    "model": model_name, "family": condition["family"],
                    "condition_id": condition["condition_id"], "sample_index": sample_index,
                }
                records.append(base | {"scenario": "original"} | trajectory_metrics(
                    original[sample_index], scene, checker, collector
                ))
                records.append(base | {"scenario": "yaw_rotated_90deg"} | trajectory_metrics(
                    rotated[sample_index], rotated_scene, rotated_checker, collector
                ))
                position_rms = float(np.sqrt(np.mean(np.square(
                    original[sample_index, :, :3] - restored[sample_index, :, :3]
                ))))
                dots = np.clip(np.abs(np.sum(
                    original[sample_index, :, 3:7] * restored[sample_index, :, 3:7], axis=1
                )), 0, 1)
                rotation_rms = float(np.sqrt(np.mean(np.square(
                    np.degrees(2 * np.arccos(dots))
                ))))
                equivariance[model_name].append((position_rms, rotation_rms))
        print(f"evaluated {condition_index + 1}/{len(conditions)} conditions", flush=True)
    summaries = {
        model: {
            scenario: summarize(records, model, scenario)
            for scenario in ("original", "yaw_rotated_90deg")
        } | {
            "yaw_equivariance_position_rms_m": float(np.mean([v[0] for v in equivariance[model]])),
            "yaw_equivariance_rotation_rms_deg": float(np.mean([v[1] for v in equivariance[model]])),
        }
        for model in runtimes
    }
    result = {
        "schema_version": "hnuter_dit_v001_v002_comparison_v001",
        "dataset": str(args.dataset), "split": "test",
        "condition_selection": "family_stratified_fixed_seed",
        "condition_count": len(conditions),
        "samples_per_condition": args.samples_per_condition,
        "ddim_steps": args.ddim_steps, "seed": args.seed,
        "collision_backend": "full HNUTER URDF + COAL at 128 waypoints",
        "models": {name: runtime.describe() for name, runtime in runtimes.items()},
        "summaries": summaries, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--condition-count", type=int, default=100)
    parser.add_argument("--samples-per-condition", type=int, default=8)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    args.dataset, args.baseline, args.candidate, args.output = (
        args.dataset.resolve(), args.baseline.resolve(),
        args.candidate.resolve(), args.output.resolve(),
    )
    return args


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps(result["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
