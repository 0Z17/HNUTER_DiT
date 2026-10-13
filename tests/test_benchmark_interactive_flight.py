"""Check the normalization behind the before/after physical-flight report."""

import json
import sys

import numpy as np
import pytest

from hnuter_dit.benchmark_interactive_flight import (
    compare_flights, quaternion_roll_pitch_degrees, series_motion_metrics, summarize_flight,
)


def test_variation_normalized_by_duration_and_nonuniform_time():
    # A 2 rad/s command has the same measured activity regardless of flight
    # duration or uneven observation intervals.
    for times in ([0, .25, 1, 2], [0, .5, 4]):
        values = np.asarray(times)[:, None] * [[2, -2]]
        metrics = series_motion_metrics(times, values)
        assert metrics["mean_total_variation_per_second"] == pytest.approx(2)
        assert metrics["rate_rms"] == pytest.approx(2)
    with pytest.raises(ValueError, match="严格递增"):
        series_motion_metrics([0, 1, 1], [[0], [1], [1]])


def test_attitude_errors_track_planned_bank_and_ignore_quaternion_sign():
    half_roll = np.radians(25) / 2
    banked = [np.cos(half_roll), np.sin(half_roll), 0, 0]
    np.testing.assert_allclose(quaternion_roll_pitch_degrees([banked, -np.asarray(banked)]),
                               [[25, 0], [25, 0]], atol=1e-12)
    records = [{
        "time_s": float(index), "pose": [0, 0, 1, *banked],
        "reference_pose": [0, 0, 1, *banked],
        "angular_velocity": [0, 0, 0],
        "tilt_joint_position": [0, 0, 0, 0],
        "tilt_joint_velocity": [1, -1, 1, -1],
        "tilt_commands": [0, 0, 0, 0],
        "rotor_commands": [10, 10, 5, 5],
    } for index in range(3)]
    result = summarize_flight(records, {"success": True})
    assert result["tilt_joint_velocity_rms_rps"] == pytest.approx(1)
    assert result["left_right_rotor_imbalance_mean_fraction"] == pytest.approx(1 / 3)
    np.testing.assert_allclose(result["roll_pitch_tracking_rmse_deg"], 0, atol=1e-12)


def test_comparison_requires_same_scene_candidate_and_seed():
    before = {"scene_geometry_hash": "scene_a", "candidate_id": 3, "seed": 1}
    after = {**before, "scene_geometry_hash": "scene_b"}
    with pytest.raises(ValueError, match="scene_geometry_hash"):
        compare_flights(before, after)


def test_cli_forwards_custom_collector_path_for_saved_plan(tmp_path, monkeypatch, capsys):
    from hnuter_dit import benchmark_interactive_flight as benchmark

    plan = {"final_candidate_id": 4, "scene": {"environment_id": "saved_scene"}}
    plan_path = tmp_path / "exported_plan.json"
    plan_path.write_text(json.dumps(plan))
    output_path = tmp_path / "results" / "flight.json"
    collector_path = tmp_path / "portable_collector"
    mujoco_path = tmp_path / "portable_mujoco"
    calls = []

    def execute_saved_plan(received_plan, candidate_id, **options):
        calls.append((received_plan, candidate_id, options))
        return {"success": True, "records": [], "scene_id": "saved_scene"}

    monkeypatch.setattr(benchmark, "run_benchmark", execute_saved_plan)
    monkeypatch.setattr(sys, "argv", [
        "hnuter_dit.benchmark_interactive_flight", "--plan", str(plan_path),
        "--collector-root", str(collector_path), "--mujoco-root", str(mujoco_path),
        "--seed", "7", "--options", '{"speed_mps": 0.8}',
        "--output", str(output_path),
    ])
    benchmark.main()

    received_plan, candidate_id, options = calls[0]
    assert received_plan == plan
    assert candidate_id == 4
    assert options["collector_root"] == collector_path
    assert options["asset_root"] == mujoco_path
    assert options["seed"] == 7
    assert options["options"] == {"speed_mps": 0.8}
    assert json.loads(output_path.read_text())["success"] is True
    assert json.loads(capsys.readouterr().out)["scene_id"] == "saved_scene"
