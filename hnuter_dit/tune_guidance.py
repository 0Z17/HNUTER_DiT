"""Tune late collision-guidance settings on the frozen validation split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_models import DEFAULT_DATASET, select_conditions, trajectory_metrics
from .evaluate_obstacle_shift import shift_relevant_obstacle
from .evaluate_v003 import DEFAULT_V003
from .guidance import CollisionGuidanceConfig
from .service import DiTModelRuntime, ReferenceCollector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "artifacts/base_free_flight_159103traj_dit_canonical_cross_v003"
    / "guidance_tuning_smooth_validation.json"
)


def _variants() -> dict[str, CollisionGuidanceConfig | None]:
    return {
        "unguided": None,
        "legacy_pointwise_g8_s0.20": CollisionGuidanceConfig(
            mode="pointwise", steps=8, step_size_m=0.20,
            smoothing_passes=2,
        ),
        "smooth_g5_s0.05": CollisionGuidanceConfig(steps=5, step_size_m=0.05),
        "smooth_g8_s0.04": CollisionGuidanceConfig(steps=8, step_size_m=0.04),
        "smooth_g8_s0.06": CollisionGuidanceConfig(steps=8, step_size_m=0.06),
        "smooth_g8_s0.08": CollisionGuidanceConfig(steps=8, step_size_m=0.08),
        "smooth_g8_s0.10": CollisionGuidanceConfig(steps=8, step_size_m=0.10),
        "smooth_g8_s0.12": CollisionGuidanceConfig(steps=8, step_size_m=0.12),
        "smooth_g8_s0.15": CollisionGuidanceConfig(steps=8, step_size_m=0.15),
    }


def tune(args: argparse.Namespace) -> dict[str, Any]:
    collector = ReferenceCollector()
    runtime = DiTModelRuntime(args.checkpoint, args.device)
    variants = _variants()
    environments = {
        row["environment_index"]: row
        for row in (
            json.loads(line) for line in (
                args.dataset / "environments/index.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        )
    }
    conditions = select_conditions(
        args.dataset, args.condition_count, args.seed, split="validation"
    )
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    timings: dict[str, list[float]] = defaultdict(list)
    for condition_index, condition in enumerate(conditions):
        environment_row = environments[int(condition["environment_index"])]
        scene = json.loads((args.dataset / environment_row["path"]).read_text(encoding="utf-8"))
        start = np.asarray(condition["start_pose7_wxyz"], dtype=np.float32)
        goal = np.asarray(condition["goal_pose7_wxyz"], dtype=np.float32)
        shifted, _, _ = shift_relevant_obstacle(
            scene, start, goal, args.shift_distance
        )
        token_values, mask_values = collector.obstacle_tokens(shifted)
        tokens = np.asarray(token_values, dtype=np.float32)
        mask = np.asarray(mask_values, dtype=bool)
        checker = collector.collision_checker_type(shifted)
        sample_seed = args.seed + 1009 * condition_index
        for name, guidance in variants.items():
            paths, elapsed = runtime.sample(
                tokens, mask, start, goal,
                count=args.samples_per_condition,
                sampling_steps=args.ddim_steps, seed=sample_seed,
                guidance=guidance,
            )
            timings[name].append(elapsed)
            for sample_index, path in enumerate(paths):
                records[name].append(
                    trajectory_metrics(path, shifted, checker, collector) | {
                        "condition_id": condition["condition_id"],
                        "sample_index": sample_index,
                    }
                )
        print(f"tuned {condition_index + 1}/{len(conditions)} conditions", flush=True)
    summaries: dict[str, dict[str, float]] = {}
    for name, rows in records.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row["condition_id"]].append(row)
        summaries[name] = {
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
            "mean_neural_sampling_time_s_per_batch": float(np.mean(timings[name])),
        }
    baseline = summaries["unguided"]
    smooth_names = [name for name in summaries if name.startswith("smooth_")]
    limits = {
        "mean_path_length_m": baseline["mean_path_length_m"] + 0.25,
        "mean_p95_turn_angle_deg": max(
            1.25 * baseline["mean_p95_turn_angle_deg"],
            baseline["mean_p95_turn_angle_deg"] + 1.0,
        ),
        "mean_acceleration_rms_m_per_index2": (
            1.25 * baseline["mean_acceleration_rms_m_per_index2"]
        ),
        "mean_jerk_rms_m_per_index3": (
            1.25 * baseline["mean_jerk_rms_m_per_index3"]
        ),
        "mean_backtracking_fraction": baseline["mean_backtracking_fraction"] + 0.01,
    }
    feasible = [
        name for name in smooth_names
        if all(summaries[name][metric] <= limit for metric, limit in limits.items())
    ]
    violations = {
        name: float(sum(
            max(summaries[name][metric] / max(limit, 1e-12) - 1.0, 0.0)
            for metric, limit in limits.items()
        ))
        for name in smooth_names
    }
    selection_pool = feasible
    used_shape_fallback = False
    if not selection_pool:
        used_shape_fallback = True
        minimum_violation = min(violations.values())
        selection_pool = [
            name for name, value in violations.items()
            if np.isclose(value, minimum_violation)
        ]
    selected = max(selection_pool, key=lambda name: (
        summaries[name]["best_of_n_task_solved_rate"],
        summaries[name]["physical_collision_free_rate"],
        summaries[name]["mean_minimum_clearance_m"],
        -summaries[name]["mean_path_length_m"],
    ))
    result = {
        "schema_version": "hnuter_dit_smooth_guidance_tuning_validation_v002",
        "dataset": str(args.dataset), "split": "validation",
        "scenario": "closest obstacle shifted toward start-goal segment",
        "condition_count": len(conditions),
        "samples_per_condition": args.samples_per_condition,
        "ddim_steps": args.ddim_steps,
        "shift_distance_m": args.shift_distance,
        "selection_rule": (
            "shape constraints first; then max best-of-n, collision-free rate, "
            "clearance, negative length"
        ),
        "shape_constraint_limits": limits,
        "shape_constraint_violations": violations,
        "feasible_variants": feasible,
        "selection_used_shape_fallback": used_shape_fallback,
        "selected_variant": selected,
        "selected_config": variants[selected].__dict__,
        "summaries": summaries,
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
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_V003)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--condition-count", type=int, default=30)
    parser.add_argument("--samples-per-condition", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--shift-distance", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    for name in ("dataset", "checkpoint", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> None:
    result = tune(parse_args())
    print(json.dumps({
        "selected_variant": result["selected_variant"],
        "selected_config": result["selected_config"],
        "summaries": result["summaries"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
