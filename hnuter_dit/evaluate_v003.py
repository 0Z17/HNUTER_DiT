"""Controlled v001/v002/v003 and late-guidance evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_models import (
    DEFAULT_BASELINE, DEFAULT_DATASET, rotate_pose7, rotate_scene,
    select_conditions, trajectory_metrics,
)
from .evaluate_obstacle_shift import shift_relevant_obstacle
from .guidance import CollisionGuidanceConfig
from .service import DiTModelRuntime, ReferenceCollector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V002 = (
    ROOT / "artifacts/base_free_flight_159103traj_dit_relgeom_v002"
    / "models/dit_rel_cross/best.pt"
)
DEFAULT_V003 = (
    ROOT / "artifacts/base_free_flight_159103traj_dit_canonical_cross_v003"
    / "models/dit_cross/best.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/base_free_flight_159103traj_dit_canonical_cross_v003"
    / "comparison_v003_smooth_guidance.json"
)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition_id"]].append(row)
    return {
        "candidate_count": len(rows),
        "condition_count": len(groups),
        "physical_collision_free_rate": float(np.mean([
            row["physical_collision_free"] for row in rows
        ])),
        "valid_rate": float(np.mean([row["valid"] for row in rows])),
        "best_of_n_task_solved_rate": float(np.mean([
            any(item["valid"] for item in group) for group in groups.values()
        ])),
        "mean_minimum_clearance_m": float(np.mean([
            row["minimum_clearance_m"] for row in rows
        ])),
        "mean_path_length_m": float(np.mean([
            row["path_length_m"] for row in rows
        ])),
        "mean_path_efficiency": float(np.mean([
            row["path_efficiency"] for row in rows
        ])),
        "mean_p95_turn_angle_deg": float(np.mean([
            row["p95_turn_angle_deg"] for row in rows
        ])),
        "mean_maximum_turn_angle_deg": float(np.mean([
            row["maximum_turn_angle_deg"] for row in rows
        ])),
        "mean_acceleration_rms_m_per_index2": float(np.mean([
            row["acceleration_rms_m_per_index2"] for row in rows
        ])),
        "mean_jerk_rms_m_per_index3": float(np.mean([
            row["jerk_rms_m_per_index3"] for row in rows
        ])),
        "mean_backtracking_fraction": float(np.mean([
            row["backtracking_fraction"] for row in rows
        ])),
    }


def _paired_comparison(
    candidate: list[dict[str, Any]], reference: list[dict[str, Any]], seed: int,
) -> dict[str, Any]:
    candidate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reference_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate:
        candidate_groups[row["condition_id"]].append(row)
    for row in reference:
        reference_groups[row["condition_id"]].append(row)
    condition_ids = sorted(candidate_groups)
    if condition_ids != sorted(reference_groups):
        raise RuntimeError("paired comparison condition IDs do not match")
    values: dict[str, list[float]] = defaultdict(list)
    for condition_id in condition_ids:
        left = sorted(candidate_groups[condition_id], key=lambda row: row["sample_index"])
        right = sorted(reference_groups[condition_id], key=lambda row: row["sample_index"])
        if len(left) != len(right):
            raise RuntimeError("paired comparison sample counts do not match")
        values["physical_collision_free_rate_delta"].append(float(np.mean([
            float(a["physical_collision_free"]) - float(b["physical_collision_free"])
            for a, b in zip(left, right)
        ])))
        values["valid_rate_delta"].append(float(np.mean([
            float(a["valid"]) - float(b["valid"])
            for a, b in zip(left, right)
        ])))
        values["best_of_n_task_solved_rate_delta"].append(
            float(any(row["valid"] for row in left))
            - float(any(row["valid"] for row in right))
        )
        values["mean_minimum_clearance_m_delta"].append(float(np.mean([
            a["minimum_clearance_m"] - b["minimum_clearance_m"]
            for a, b in zip(left, right)
        ])))
        values["mean_path_length_m_delta"].append(float(np.mean([
            a["path_length_m"] - b["path_length_m"]
            for a, b in zip(left, right)
        ])))
        for metric in (
            "p95_turn_angle_deg", "maximum_turn_angle_deg",
            "acceleration_rms_m_per_index2", "jerk_rms_m_per_index3",
            "backtracking_fraction",
        ):
            values[f"mean_{metric}_delta"].append(float(np.mean([
                a[metric] - b[metric] for a, b in zip(left, right)
            ])))
    matrix = {name: np.asarray(items, dtype=np.float64) for name, items in values.items()}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(condition_ids), size=(10_000, len(condition_ids)))
    return {
        name: {
            "estimate": float(data.mean()),
            "bootstrap_95_ci": [
                float(np.percentile(data[indices].mean(axis=1), 2.5)),
                float(np.percentile(data[indices].mean(axis=1), 97.5)),
            ],
        }
        for name, data in matrix.items()
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    collector = ReferenceCollector()
    v001 = DiTModelRuntime(args.v001, args.device)
    v002 = DiTModelRuntime(args.v002, args.device)
    v003 = DiTModelRuntime(args.v003, args.device)
    smooth_guidance = CollisionGuidanceConfig(
        mode="smooth",
        steps=args.guidance_steps,
        step_size_m=args.guidance_step_size,
        safe_distance_m=args.guidance_safe_distance,
        sigma_m=args.guidance_sigma,
        smoothing_passes=args.guidance_smoothing_passes,
    )
    legacy_guidance = CollisionGuidanceConfig(
        mode="pointwise", steps=8, step_size_m=0.20,
        safe_distance_m=0.04, sigma_m=0.05, smoothing_passes=2,
    )
    variants = {
        "v001_absolute_standard_cross": (v001, None),
        "v002_canonical_relative_cross": (v002, None),
        "v003_canonical_standard_cross": (v003, None),
        "v003_legacy_pointwise_guidance": (v003, legacy_guidance),
        "v003_smooth_multiobjective_guidance": (v003, smooth_guidance),
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
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {scenario: [] for scenario in ("original", "yaw_rotated_90deg", "obstacle_shift")}
        for name in variants
    }
    timings: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in variants
    }
    equivariance: dict[str, list[tuple[float, float]]] = defaultdict(list)
    angle = math.pi / 2
    for condition_index, condition in enumerate(conditions):
        environment_row = environments[int(condition["environment_index"])]
        scene = json.loads((args.dataset / environment_row["path"]).read_text(encoding="utf-8"))
        start = np.asarray(condition["start_pose7_wxyz"], dtype=np.float32)
        goal = np.asarray(condition["goal_pose7_wxyz"], dtype=np.float32)
        rotated_scene = rotate_scene(scene, angle)
        rotated_start = rotate_pose7(start, angle)
        rotated_goal = rotate_pose7(goal, angle)
        shifted_scene, obstacle_id, displacement = shift_relevant_obstacle(
            scene, start, goal, args.shift_distance
        )
        scene_tokens, scene_mask = collector.obstacle_tokens(scene)
        rotated_tokens, rotated_mask = collector.obstacle_tokens(rotated_scene)
        shifted_tokens, shifted_mask = collector.obstacle_tokens(shifted_scene)
        arrays = {
            "original": (
                np.asarray(scene_tokens, dtype=np.float32),
                np.asarray(scene_mask, dtype=bool), start, goal,
                scene, collector.collision_checker_type(scene),
            ),
            "yaw_rotated_90deg": (
                np.asarray(rotated_tokens, dtype=np.float32),
                np.asarray(rotated_mask, dtype=bool), rotated_start, rotated_goal,
                rotated_scene, collector.collision_checker_type(rotated_scene),
            ),
            "obstacle_shift": (
                np.asarray(shifted_tokens, dtype=np.float32),
                np.asarray(shifted_mask, dtype=bool), start, goal,
                shifted_scene, collector.collision_checker_type(shifted_scene),
            ),
        }
        sample_seed = args.seed + 1009 * condition_index
        for model_name, (runtime, guide) in variants.items():
            sampled: dict[str, np.ndarray] = {}
            for scenario, (tokens, mask, scenario_start, scenario_goal, _, _) in arrays.items():
                paths, elapsed = runtime.sample(
                    tokens, mask, scenario_start, scenario_goal,
                    count=args.samples_per_condition,
                    sampling_steps=args.ddim_steps, seed=sample_seed,
                    guidance=guide,
                )
                sampled[scenario] = paths
                timings[model_name][scenario].append(elapsed)
            restored = rotate_pose7(sampled["yaw_rotated_90deg"], -angle)
            for sample_index in range(args.samples_per_condition):
                for scenario in arrays:
                    _, _, _, _, scenario_scene, checker = arrays[scenario]
                    row = trajectory_metrics(
                        sampled[scenario][sample_index], scenario_scene,
                        checker, collector,
                    )
                    row.update({
                        "condition_id": condition["condition_id"],
                        "family": condition["family"],
                        "sample_index": sample_index,
                    })
                    if scenario == "obstacle_shift":
                        response = (
                            sampled[scenario][sample_index, :, :3]
                            - sampled["original"][sample_index, :, :3]
                        )
                        row.update({
                            "moved_obstacle_id": obstacle_id,
                            "displacement_xyz": displacement,
                            "trajectory_response_rms_m": float(np.sqrt(np.mean(np.square(response)))),
                            "trajectory_response_max_m": float(
                                np.linalg.norm(response, axis=1).max()
                            ),
                        })
                    records[model_name][scenario].append(row)
                original = sampled["original"][sample_index]
                position_rms = float(np.sqrt(np.mean(np.square(
                    original[:, :3] - restored[sample_index, :, :3]
                ))))
                dots = np.clip(np.abs(np.sum(
                    original[:, 3:7] * restored[sample_index, :, 3:7], axis=1
                )), 0, 1)
                rotation_rms = float(np.sqrt(np.mean(np.square(
                    np.degrees(2 * np.arccos(dots))
                ))))
                equivariance[model_name].append((position_rms, rotation_rms))
        print(f"evaluated {condition_index + 1}/{len(conditions)} conditions", flush=True)

    summaries: dict[str, Any] = {}
    for model_name in variants:
        summaries[model_name] = {}
        for scenario in records[model_name]:
            summary = _summarize(records[model_name][scenario])
            summary["mean_neural_sampling_time_s_per_batch"] = float(np.mean(
                timings[model_name][scenario]
            ))
            summary["milliseconds_per_path"] = (
                1000 * summary["mean_neural_sampling_time_s_per_batch"]
                / args.samples_per_condition
            )
            if scenario == "obstacle_shift":
                summary["mean_trajectory_response_rms_m"] = float(np.mean([
                    row["trajectory_response_rms_m"]
                    for row in records[model_name][scenario]
                ]))
                summary["mean_trajectory_response_max_m"] = float(np.mean([
                    row["trajectory_response_max_m"]
                    for row in records[model_name][scenario]
                ]))
            summaries[model_name][scenario] = summary
        summaries[model_name]["yaw_equivariance_position_rms_m"] = float(np.mean([
            value[0] for value in equivariance[model_name]
        ]))
        summaries[model_name]["yaw_equivariance_rotation_rms_deg"] = float(np.mean([
            value[1] for value in equivariance[model_name]
        ]))
    result = {
        "schema_version": "hnuter_dit_v003_smooth_guidance_comparison_v002",
        "dataset": str(args.dataset), "split": "test",
        "condition_selection": "family_stratified_fixed_seed",
        "condition_count": len(conditions),
        "samples_per_condition": args.samples_per_condition,
        "ddim_steps": args.ddim_steps, "seed": args.seed,
        "shift_distance_m": args.shift_distance,
        "collision_backend": "full HNUTER URDF + COAL at 128 waypoints",
        "guidance": {
            "legacy_pointwise": legacy_guidance.__dict__,
            "smooth_multiobjective": smooth_guidance.__dict__,
        },
        "models": {
            name: runtime.describe() | {
                "guidance_enabled": guide is not None and guide.steps > 0
            }
            for name, (runtime, guide) in variants.items()
        },
        "summaries": summaries,
        "paired_comparisons": {
            comparison: {
                scenario: _paired_comparison(
                    records[candidate][scenario], records[reference][scenario],
                    args.seed + 7919 * comparison_index + 101 * scenario_index,
                )
                for scenario_index, scenario in enumerate(
                    ("original", "yaw_rotated_90deg", "obstacle_shift")
                )
            }
            for comparison_index, (comparison, candidate, reference) in enumerate((
                (
                    "v003_minus_v002",
                    "v003_canonical_standard_cross",
                    "v002_canonical_relative_cross",
                ),
                (
                    "v003_smooth_minus_unguided",
                    "v003_smooth_multiobjective_guidance",
                    "v003_canonical_standard_cross",
                ),
                (
                    "v003_smooth_minus_legacy_pointwise",
                    "v003_smooth_multiobjective_guidance",
                    "v003_legacy_pointwise_guidance",
                ),
                (
                    "v003_legacy_pointwise_minus_unguided",
                    "v003_legacy_pointwise_guidance",
                    "v003_canonical_standard_cross",
                ),
            ))
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--v001", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--v002", type=Path, default=DEFAULT_V002)
    parser.add_argument("--v003", type=Path, default=DEFAULT_V003)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--condition-count", type=int, default=100)
    parser.add_argument("--samples-per-condition", type=int, default=8)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--shift-distance", type=float, default=0.30)
    parser.add_argument("--guidance-steps", type=int, default=8)
    parser.add_argument("--guidance-step-size", type=float, default=0.12)
    parser.add_argument("--guidance-safe-distance", type=float, default=0.04)
    parser.add_argument("--guidance-sigma", type=float, default=0.05)
    parser.add_argument("--guidance-smoothing-passes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    for name in ("dataset", "v001", "v002", "v003", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps(result["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
