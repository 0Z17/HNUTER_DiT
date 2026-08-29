"""Obstacle-perturbation stress test on frozen v001 test environments."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_models import (
    DEFAULT_BASELINE, DEFAULT_CANDIDATE, DEFAULT_DATASET,
    select_conditions, trajectory_metrics,
)
from .service import DiTModelRuntime, ReferenceCollector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/base_free_flight_159103traj_dit_relgeom_v002/obstacle_shift_comparison.json"


def point_segment_projection(point: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    delta = goal - start
    fraction = float(np.dot(point - start, delta) / max(np.dot(delta, delta), 1e-9))
    return start + np.clip(fraction, 0.15, 0.85) * delta


def shift_relevant_obstacle(
    scene: dict[str, Any], start: np.ndarray, goal: np.ndarray, distance: float,
) -> tuple[dict[str, Any], str, list[float]]:
    result = copy.deepcopy(scene)
    candidates = [
        item for item in result.get("obstacles", [])
        if item.get("collision", False) and item.get("type") == "box"
        and item.get("role") != "floor"
    ]
    if not candidates:
        raise ValueError("scene contains no movable collision box")
    def score(item: dict[str, Any]) -> float:
        center = np.asarray(item["pose"]["position"], dtype=np.float64)
        projection = point_segment_projection(center, start[:3], goal[:3])
        return float(np.linalg.norm(center - projection))
    obstacle = min(candidates, key=score)
    center = np.asarray(obstacle["pose"]["position"], dtype=np.float64)
    projection = point_segment_projection(center, start[:3], goal[:3])
    direction = projection - center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        segment = goal[:3] - start[:3]
        direction = np.asarray([-segment[1], segment[0], 0.0])
        norm = float(np.linalg.norm(direction))
    displacement = distance * direction / max(norm, 1e-9)
    obstacle["pose"]["position"] = (center + displacement).tolist()
    result["environment_id"] = str(result.get("environment_id", "scene")) + "_shifted"
    return result, str(obstacle.get("id", "unknown")), displacement.tolist()


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
    metrics: dict[str, list[dict[str, Any]]] = {name: [] for name in runtimes}
    for condition_index, condition in enumerate(conditions):
        environment_row = environments[int(condition["environment_index"])]
        scene = json.loads((args.dataset / environment_row["path"]).read_text(encoding="utf-8"))
        start = np.asarray(condition["start_pose7_wxyz"], dtype=np.float32)
        goal = np.asarray(condition["goal_pose7_wxyz"], dtype=np.float32)
        shifted, obstacle_id, displacement = shift_relevant_obstacle(
            scene, start, goal, args.shift_distance
        )
        original_tokens, original_mask = collector.obstacle_tokens(scene)
        shifted_tokens, shifted_mask = collector.obstacle_tokens(shifted)
        shifted_checker = collector.collision_checker_type(shifted)
        sample_seed = args.seed + 1009 * condition_index
        for model_name, runtime in runtimes.items():
            original, _ = runtime.sample(
                np.asarray(original_tokens, dtype=np.float32), np.asarray(original_mask, dtype=bool),
                start, goal, count=args.samples_per_condition,
                sampling_steps=args.ddim_steps, seed=sample_seed,
            )
            changed, _ = runtime.sample(
                np.asarray(shifted_tokens, dtype=np.float32), np.asarray(shifted_mask, dtype=bool),
                start, goal, count=args.samples_per_condition,
                sampling_steps=args.ddim_steps, seed=sample_seed,
            )
            for sample_index in range(args.samples_per_condition):
                row = trajectory_metrics(
                    changed[sample_index], shifted, shifted_checker, collector
                )
                row.update({
                    "condition_id": condition["condition_id"],
                    "family": condition["family"], "sample_index": sample_index,
                    "moved_obstacle_id": obstacle_id,
                    "displacement_xyz": displacement,
                    "trajectory_response_rms_m": float(np.sqrt(np.mean(np.square(
                        changed[sample_index, :, :3] - original[sample_index, :, :3]
                    )))),
                    "trajectory_response_max_m": float(np.linalg.norm(
                        changed[sample_index, :, :3] - original[sample_index, :, :3], axis=1
                    ).max()),
                })
                metrics[model_name].append(row)
        print(f"evaluated obstacle shift {condition_index + 1}/{len(conditions)}", flush=True)
    summaries = {}
    for model_name, rows in metrics.items():
        condition_groups = {
            condition_id: [row for row in rows if row["condition_id"] == condition_id]
            for condition_id in sorted({row["condition_id"] for row in rows})
        }
        summaries[model_name] = {
            "candidate_count": len(rows),
            "physical_collision_free_rate": float(np.mean([
                row["physical_collision_free"] for row in rows
            ])),
            "best_of_n_task_solved_rate": float(np.mean([
                any(row["valid"] for row in group) for group in condition_groups.values()
            ])),
            "mean_minimum_clearance_m": float(np.mean([
                row["minimum_clearance_m"] for row in rows
            ])),
            "mean_trajectory_response_rms_m": float(np.mean([
                row["trajectory_response_rms_m"] for row in rows
            ])),
            "mean_trajectory_response_max_m": float(np.mean([
                row["trajectory_response_max_m"] for row in rows
            ])),
        }
    result = {
        "schema_version": "hnuter_dit_obstacle_shift_comparison_v001",
        "dataset": str(args.dataset), "split": "test",
        "condition_count": len(conditions),
        "samples_per_condition": args.samples_per_condition,
        "shift_distance_m": args.shift_distance,
        "selection": "collision box nearest the start-goal segment, moved toward the segment",
        "ddim_steps": args.ddim_steps, "seed": args.seed,
        "collision_backend": "full HNUTER URDF + COAL at 128 waypoints",
        "summaries": summaries, "records": metrics,
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
    parser.add_argument("--shift-distance", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    for name in ("dataset", "baseline", "candidate", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps(result["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
