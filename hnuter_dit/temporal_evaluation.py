"""Held-out, time-preserving evaluation for the temporal DiT pilot.

Future obstacle truth belongs here, never in a model request or candidate
selector. This module scores one preselected random draw per decision window.
The sampled 42-sphere metric is a geometric proxy; native/Isaac contact results
are reported separately. A guard-induced stop does not count as model success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .collision_loss import ROBOT_SPHERE_CENTERS, ROBOT_SPHERE_RADII


# These gates are declared before sampling the held-out pilot. Passing the
# offline gates alone is insufficient to authorize a larger collection run.
PILOT_GATES = {
    "minimum_test_groups": 16,
    "minimum_feasible_safe_progress_rate": .80,
    "minimum_gain_over_latest_only": .10,
    "minimum_gain_over_frozen_history": .05,
    "minimum_gain_over_untrained": .10,
    "maximum_native_collision_rate": 0.,
    "minimum_native_episodes": 8,
    "maximum_native_tracking_error_m": .12,
    "maximum_inference_p95_seconds": .5,
    "industrial_contact_free_goal_required": True,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_report(output, report):
    path = Path(output) / "report.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rotation6_matrix(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("rotation6 must be finite with shape (6,)")
    x, y = value[:3], value[3:]
    if np.linalg.norm(x) < 1e-8:
        raise ValueError("rotation6 first axis is degenerate")
    x = x / np.linalg.norm(x)
    y = y - x * (x @ y)
    if np.linalg.norm(y) < 1e-8:
        raise ValueError("rotation6 second axis is degenerate")
    y = y / np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


def timed_proxy_clearance(positions, times, boxes, mask, rotation, *, spacing_s=.02, control_points=None):
    """Sample robot spheres and moving OBBs at the SAME physical times.

    ``boxes[T,M,10]`` are ground truth in the current canonical frame. Their
    centers/extents interpolate linearly and orientations use shortest SLERP.
    Mask changes are rejected: teleporting an obstacle is not an interpolation.
    Empty truth is invalid for this bounded collection, not certified free.
    """
    positions, times = np.asarray(positions, float), np.asarray(times, float)
    boxes, mask = np.asarray(boxes, float), np.asarray(mask, bool)
    rotation = np.asarray(rotation, float)
    if (times.ndim != 1 or len(times) < 3 or times[0] != 0 or
            np.any(np.diff(times) <= 0) or positions.shape != (len(times), 3) or
            boxes.ndim != 3 or boxes.shape[0] != len(times) or boxes.shape[2] != 10 or
            mask.shape != boxes.shape[:2] or rotation.shape != (3, 3) or
            not all(np.isfinite(a).all() for a in (positions, times, boxes, rotation)) or
            not np.isfinite(spacing_s) or spacing_s <= 0):
        raise ValueError("invalid fixed-time proxy input")
    if not np.array_equal(mask, np.broadcast_to(mask[:1], mask.shape)):
        raise ValueError("changing ground-truth mask cannot be interpolated")
    active = np.flatnonzero(mask[0])
    if not len(active):
        raise ValueError("missing evaluation collision truth")
    sample_times = np.unique(np.r_[times, np.arange(0, times[-1], spacing_s), times[-1]])
    if control_points is None:
        path = CubicSpline(times, positions)(sample_times)
    else:
        from scipy.interpolate import BSpline
        from .temporal_trajectory import KNOTS
        spline = BSpline(KNOTS, control_points, 3)
        if not np.allclose(spline(times), positions, atol=2e-5, rtol=2e-5):
            raise ValueError("positions disagree with the evaluated B-spline")
        path = spline(sample_times)
    spheres = path[:, None, :] + np.asarray(ROBOT_SPHERE_CENTERS) @ rotation.T
    radii = np.asarray(ROBOT_SPHERE_RADII)
    minimum = np.inf
    for index in active:
        box = boxes[:, index]
        if (box[:, 3:6] <= 0).any():
            raise ValueError("ground-truth full extents must be positive")
        centers = np.stack([np.interp(sample_times, times, box[:, j]) for j in range(3)], -1)
        extents = np.stack([np.interp(sample_times, times, box[:, j]) for j in range(3, 6)], -1)
        quats = box[:, [7, 8, 9, 6]]
        matrices = Slerp(times, Rotation.from_quat(quats))(sample_times).as_matrix()
        local = np.einsum("tsi,tij->tsj", spheres - centers[:, None], matrices)
        delta = np.abs(local) - extents[:, None] / 2
        signed = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(-1), 0) - radii
        minimum = min(minimum, float(signed.min()))
    return minimum


def score_trajectory(positions, times, condition, teacher_positions, gt_boxes, gt_mask,
                     *, max_speed=.5, max_acceleration=.6, clearance_margin=.04, control_points=None):
    """Fixed-time feasibility, progress and timed geometric collision metrics."""
    positions, times = np.asarray(positions, float), np.asarray(times, float)
    condition, teacher = np.asarray(condition, float), np.asarray(teacher_positions, float)
    if positions.shape != (len(times), 3) or teacher.shape != positions.shape or condition.shape != (12,):
        raise ValueError("trajectory scoring input shape mismatch")
    if not np.isfinite(positions).all():
        return {"finite": False, "feasible_safe_progress": False,
                "safe": False, "feasible": False, "adequate_progress": False}
    if control_points is None:
        spline = CubicSpline(times, positions)
        dense_times = np.unique(np.r_[times, np.linspace(times[0], times[-1], 1601)])
        speed = float(np.linalg.norm(spline(dense_times, 1), axis=1).max())
        acceleration = float(np.linalg.norm(spline(dense_times, 2), axis=1).max())
    else:
        from .temporal_trajectory import FixedTimeTrajectory
        exact = FixedTimeTrajectory(control_points, quaternion=[1., 0., 0., 0.],
                                    max_speed=1e6, max_acceleration=1e6)
        spline = exact.spline
        if not np.allclose(spline(times), positions, atol=2e-5, rtol=2e-5):
            raise ValueError("predicted positions disagree with spline controls")
        speed, acceleration = exact.observed_maxima["speed"], exact.observed_maxima["acceleration"]
    initial_speed_error = float(np.linalg.norm(spline(0, 1) - condition[3:6]))
    goal = condition[:3]
    initial_distance = float(np.linalg.norm(goal))
    final_distance = float(np.linalg.norm(goal - positions[-1]))
    teacher_final_distance = float(np.linalg.norm(goal - teacher[-1]))
    progress = initial_distance - final_distance
    teacher_progress = initial_distance - teacher_final_distance
    # A teacher may legitimately wait throughout a short horizon. Report that
    # class separately; do not turn "stayed put" into universal task success.
    adequate = progress >= max(0., .8 * teacher_progress) - .05
    if teacher_progress > .20:
        adequate = adequate and progress > .15
    minimum = timed_proxy_clearance(positions, times, gt_boxes, gt_mask, rotation6_matrix(condition[6:]),
                                    control_points=control_points)
    feasible = (speed <= max_speed * (1 + 1e-5) and acceleration <= max_acceleration * (1 + 1e-5) and
                np.linalg.norm(positions[0]) <= .005 and initial_speed_error <= .05)
    safe = minimum >= clearance_margin
    return {"finite": True, "feasible_safe_progress": bool(feasible and safe and adequate),
            "safe": bool(safe), "feasible": bool(feasible), "adequate_progress": bool(adequate),
            "timed_proxy_minimum_clearance_m": minimum,
            "proxy_collision": bool(minimum <= 0),
            "maximum_speed_mps": speed, "maximum_acceleration_mps2": acceleration,
            "initial_velocity_error_mps": initial_speed_error,
            "goal_progress_m": progress, "teacher_progress_m": teacher_progress,
            "final_goal_distance_m": final_distance, "teacher_final_goal_distance_m": teacher_final_distance,
            "teacher_low_progress_class": bool(teacher_progress <= .20),
            "goal_within_035m": bool(final_distance <= .35)}


def summarize_rows(rows):
    if not rows:
        raise ValueError("no evaluated rows")
    keys = ("feasible_safe_progress", "safe", "feasible", "adequate_progress", "proxy_collision",
            "goal_within_035m", "teacher_low_progress_class")
    result = {key + "_rate": float(np.mean([r.get(key, False) for r in rows])) for key in keys}
    result["samples"] = len(rows)
    result["groups"] = len({r["group_id"] for r in rows})
    for key in ("goal_progress_m", "timed_proxy_minimum_clearance_m", "maximum_speed_mps", "maximum_acceleration_mps2"):
        values = [r[key] for r in rows if key in r]
        result[key + "_mean"] = float(np.mean(values)) if values else None
    latency = [r["sampling_seconds"] for r in rows if "sampling_seconds" in r]
    if latency:
        result["sampling_seconds_mean"] = float(np.mean(latency))
        result["sampling_seconds_p95"] = float(np.percentile(latency, 95))
    # Each group is the split unit; balance paired episodes at the group level.
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(float(row["feasible_safe_progress"]))
    result["group_balanced_feasible_safe_progress_rate"] = float(np.mean([np.mean(v) for v in grouped.values()]))
    result["by_teacher_progress"] = {}
    for label, keep in (("low_progress_at_most_020m", lambda r: r.get("teacher_progress_m", 0) <= .20),
                        ("moving_more_than_020m", lambda r: r.get("teacher_progress_m", 0) > .20)):
        subset = [row for row in rows if keep(row)]
        result["by_teacher_progress"][label] = {
            "samples": len(subset),
            "feasible_safe_progress_rate": float(np.mean([r["feasible_safe_progress"] for r in subset])) if subset else None,
            "proxy_collision_rate": float(np.mean([r.get("proxy_collision", False) for r in subset])) if subset else None,
        }
    return result


def paired_group_difference(temporal_rows, comparison_rows, *, seed=4103):
    """Bootstrap whole paired groups, never correlated windows independently."""
    def rates(rows):
        groups = {}
        for row in rows:
            groups.setdefault(row["group_id"], []).append(float(row["feasible_safe_progress"]))
        return {key: np.mean(values) for key, values in groups.items()}
    first, second = rates(temporal_rows), rates(comparison_rows)
    if set(first) != set(second):
        raise ValueError("paired comparison requires identical held-out groups")
    deltas = np.asarray([first[key] - second[key] for key in sorted(first)])
    if not len(deltas):
        raise ValueError("empty paired comparison")
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(5000, len(deltas)), replace=True).mean(axis=1)
    return {"groups": len(deltas), "difference": float(deltas.mean()),
            "bootstrap_95_percent_interval": np.percentile(samples, [2.5, 97.5]).tolist(),
            "bootstrap_unit": "entire paired scene group", "bootstrap_seed": seed}


def validate_runtime_role(description, name, dataset_sha256, temporal_description=None):
    mode, state = ("latest", "trained") if name == "latest_only" else ("temporal", "untrained_initial" if name == "untrained" else "trained")
    if (description.get("history_mode") != mode or description.get("training_state") != state or
            description.get("source_dataset", {}).get("dataset_sha256") != dataset_sha256):
        raise ValueError(f"{name} checkpoint role or frozen dataset identity mismatch")
    step = description.get("training_step", -1)
    if (state == "trained" and step < 1) or (state == "untrained_initial" and step != 0):
        raise ValueError(f"{name} checkpoint has an incompatible training step")
    if temporal_description is not None:
        if description.get("architecture") != temporal_description.get("architecture"):
            raise ValueError("comparison checkpoint architectures differ")
        if name == "frozen_history" and description.get("checkpoint_sha256") != temporal_description.get("checkpoint_sha256"):
            raise ValueError("frozen-history ablation must use exactly the temporal weights")


def industrial_evidence_valid(industrial, checkpoint_sha256):
    if not industrial or not checkpoint_sha256:
        return False
    runs = industrial.get("runs", [])
    profiles = {run.get("dynamic_profile") for run in runs}
    if not {"constant_incoming", "constant_outgoing"} <= profiles:
        return False
    for run in runs:
        planner, collision = run.get("planner", {}), run.get("collision", {})
        statuses = run.get("plan_status_counts")
        if statuses is None:
            statuses = {}
            for plan in planner.get("plans", []):
                key = plan.get("status", "unknown")
                statuses[key] = statuses.get(key, 0) + 1
        no_rejected_plan = not any("rejected" in status and count for status, count in statuses.items())
        evidence = (run.get("passed") is True and run.get("completed") is True and
                    run.get("mode") == "temporal" and run.get("goal_reached") is True and
                    run.get("dynamic_obstacle") is True and run.get("steps", 0) > 0 and
                    collision.get("samples") == run.get("steps") and
                    collision.get("complete") is True and collision.get("collision_detected") is False and
                    planner.get("failed") is False and planner.get("temporal_checkpoint_used") is True and
                    planner.get("checkpoint", {}).get("checkpoint_sha256") == checkpoint_sha256 and
                    planner.get("accepted_plans", 0) > 0 and
                    planner.get("time_in_state_seconds", {}).get("model_reference", 0) > 0 and
                    planner.get("guard_interventions", 1) == 0 and planner.get("guard_rejections", 1) == 0 and
                    no_rejected_plan)
        if not evidence:
            return False
    return True


def expansion_decision(summaries, *, native=None, industrial=None, checkpoint_sha256=None):
    """A missing required measurement fails closed; no automatic scaling."""
    temporal = summaries.get("temporal", {})
    rate = temporal.get("group_balanced_feasible_safe_progress_rate", 0.)
    checks = {
        "enough_independent_test_groups": temporal.get("groups", 0) >= PILOT_GATES["minimum_test_groups"],
        "adequate_absolute_performance": rate >= PILOT_GATES["minimum_feasible_safe_progress_rate"],
        "latency_within_budget": temporal.get("sampling_seconds_p95", float("inf")) <= PILOT_GATES["maximum_inference_p95_seconds"],
    }
    for name, minimum in (("latest_only", .10), ("frozen_history", .05), ("untrained", .10)):
        other = summaries.get(name)
        checks["improvement_over_" + name] = bool(other is not None and rate - other.get(
            "group_balanced_feasible_safe_progress_rate", 1.) >= minimum - 1e-12)
    checks["native_contact_validation"] = bool(native and native.get("episodes", 0) >= 8 and
                                               native.get("collision_rate", 1.) == 0 and native.get("complete") and
                                               native.get("maximum_tracking_error_m", float("inf")) <= .12)
    checks["industrial_goal_without_contact"] = industrial_evidence_valid(industrial, checkpoint_sha256)
    return {"expand": bool(all(checks.values())), "checks": checks, "predeclared_thresholds": PILOT_GATES,
            "scope": "bounded synthetic dynamic pilot; offline proxy is not a physical flight certificate"}


def evaluate_dataset(args):
    from .temporal_data import load_dataset, sample_from_dataset, native_rollout
    from .temporal_inference import TemporalRuntime

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = load_dataset(args.dataset, verify=True)
    indices = np.flatnonzero(dataset["split"] == 2)
    if not len(indices):
        raise ValueError("frozen dataset has no held-out test split")
    report = {
        "schema": "temporal_dit_pilot_evaluation_v1", "dataset_freeze": dataset["_freeze"],
        "completed": False, "stage": "test_sampling",
        "selection": "one fixed-seed candidate per window; no future-truth candidate selection or guard",
        "sampling_seed": args.seed, "ddim_steps": args.steps,
        "collision_metric": "42-sphere/time-indexed OBB proxy at 20ms sampling, not physical contacts",
        "limits": {"speed_mps": .5, "acceleration_mps2": .6, "clearance_margin_m": .04},
        "variants": {}, "gate": expansion_decision({}),
    }
    # Write the thresholds before inference so they cannot accidentally become
    # an undocumented after-the-fact definition of pilot success.
    (output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    predicted, predicted_controls = {}, {}
    for name, checkpoint, ablate in (
        ("temporal", args.temporal, False), ("latest_only", args.latest_only, False),
        ("frozen_history", args.temporal, True), ("untrained", args.untrained, False),
    ):
        runtime = TemporalRuntime(checkpoint, device=args.device)
        description = runtime.describe()
        validate_runtime_role(description, name, dataset['_freeze']['dataset_sha256'],
                              report['variants'].get('temporal', {}).get('runtime'))
        rows, positions, control_points = [], [], []
        for index in indices:
            sample = sample_from_dataset(dataset, int(index))
            tokens, masks = sample["history_tokens"].copy(), sample["history_mask"].copy()
            if ablate:
                tokens[...] = tokens[-1]
                masks[...] = masks[-1]
            seed = args.seed + int(sample["group_id"]) * 1000
            before = time.perf_counter()
            result = runtime.sample(tokens, masks, sample["history_times"], sample["condition"],
                                    seed=seed, steps=args.steps)
            elapsed = time.perf_counter() - before
            pred = np.asarray(result["positions"], dtype=float)
            times = np.asarray(result["future_times"], dtype=float)
            if times.shape != sample["future_times"].shape or not np.allclose(times, sample["future_times"], atol=1e-8):
                raise ValueError("runtime changed the physical prediction clock")
            row = score_trajectory(pred, times, sample["condition"], sample["positions"],
                                   sample["gt_boxes"], sample["gt_mask"], control_points=result["control_points"])
            row.update(index=int(index), group_id=int(sample["group_id"]),
                       episode_id=int(sample["episode_id"]), pair_variant=int(sample["pair_variant"]),
                       teacher_wait_seconds=float(sample["teacher_wait_s"]),
                       sampling_seed=seed, sampling_seconds=elapsed)
            rows.append(row)
            positions.append(pred)
            control_points.append(result["control_points"])
        predicted[name] = np.stack(positions)
        predicted_controls[name] = np.stack(control_points)
        report["variants"][name] = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                                     "runtime": description, "summary": summarize_rows(rows), "rows": rows}
        save_report(output, report)
        print(json.dumps({"variant": name, **report["variants"][name]["summary"]}), flush=True)
        del runtime
    np.savez_compressed(output / "predictions.npz", indices=indices, **predicted,
                        **{name + "_control_points": value for name, value in predicted_controls.items()})
    report["paired_comparisons"] = {name: paired_group_difference(report["variants"]["temporal"]["rows"],
                                                               report["variants"][name]["rows"])
                                     for name in ("latest_only", "frozen_history", "untrained")}
    # Teacher scores establish that the labels pass the same geometric and
    # dynamic criteria, independently of the collection's physical acceptance.
    teacher_rows = []
    for index in indices:
        sample = sample_from_dataset(dataset, int(index))
        row = score_trajectory(sample["positions"], sample["future_times"], sample["condition"],
                               sample["positions"], sample["gt_boxes"], sample["gt_mask"])
        row.update(group_id=int(sample["group_id"]), index=int(index))
        teacher_rows.append(row)
    report["teacher"] = {"summary": summarize_rows(teacher_rows), "rows": teacher_rows}
    report["stage"] = "native_rollout"
    save_report(output, report)
    # Do not choose the successful offline paths. The fixed first N held-out
    # samples include both members of each paired group, including failures.
    native_reports = []
    for local_index, index in enumerate(indices[:args.native_count]):
        sample = sample_from_dataset(dataset, int(index))
        native = native_rollout(sample, predicted["temporal"][local_index], sample["future_times"],
                                control_points=predicted_controls["temporal"][local_index],
                                output_dir=output / "native" / f"sample_{index:04d}")
        native_reports.append({"index": int(index), "group_id": int(sample["group_id"]), **native})
        report["native_progress"] = native_reports
        save_report(output, report)
        print(json.dumps({"native_sample": int(index), "report": native}), flush=True)
    # Strictly require explicit completion/contact fields from the collector's
    # physical evaluator. Missing evidence cannot be interpreted as no contact.
    native_summary = {
        "episodes": len(native_reports),
        "complete": bool(native_reports and all(r.get("completed") is True and r.get("steps") == 8000 and
                          r.get("simulation_seconds") == 8. and r.get("no_retiming") is True for r in native_reports)),
        "collision_rate": float(np.mean([r.get("collision_detected", True) for r in native_reports]))
                          if native_reports else None,
        "maximum_tracking_error_m": max((r.get("position_tracking_max_error_m", float("inf")) for r in native_reports), default=None),
        "rows": native_reports,
    }
    report["native"] = native_summary
    industrial = json.loads(Path(args.industrial_report).read_text()) if args.industrial_report else None
    report["industrial"] = industrial
    report["gate"] = expansion_decision({k: v["summary"] for k, v in report["variants"].items()},
                                         native=native_summary, industrial=industrial,
                                         checkpoint_sha256=report['variants']['temporal']['checkpoint_sha256'])
    report.update(completed=True, stage="complete")
    save_report(output, report)
    print(json.dumps({"report": str(output / "report.json"), "gate": report["gate"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--temporal", required=True)
    parser.add_argument("--latest-only", required=True)
    parser.add_argument("--untrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=10403)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--native-count", type=int, default=8)
    parser.add_argument("--industrial-report")
    args = parser.parse_args()
    if args.native_count < 0 or args.native_count % 2:
        parser.error("native-count must be nonnegative and even to retain paired groups")
    evaluate_dataset(args)


if __name__ == "__main__":
    main()
