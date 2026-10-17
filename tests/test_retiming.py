"""Continuous TOPP-RA references, physical angular derivatives, and audits."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("toppra")
from scipy.spatial.transform import Rotation

from hnuter_dit.retiming import RetimingConfig, TimedPoseReference


def _poses(position, angles=None):
    position = np.asarray(position, dtype=float)
    if angles is None:
        quaternion = np.tile([1., 0, 0, 0], (len(position), 1))
    else:
        quaternion = Rotation.from_euler("xyz", angles).as_quat()[:, [3, 0, 1, 2]]
    return np.column_stack((position, quaternion))


def test_dense_waypoints_do_not_force_a_stop_at_each_point():
    source = _poses(np.column_stack((np.linspace(0, 5, 128), np.zeros(128), np.ones(128))))
    reference = TimedPoseReference(source)
    assert reference.motion_duration < 8
    arrivals, _ = reference.sample(reference.knot_times + reference.config.initial_hold_s)
    np.testing.assert_allclose(arrivals[:, :3], source[:, :3], atol=1e-9)
    assert np.min(np.linalg.norm(arrivals[12:-12, 3:6], axis=1)) > .5
    assert reference.describe()["method"] == "toppra_continuous_se3_c2"
    assert reference.describe()["stops_at_waypoints"] is False


def test_noncommuting_rotations_have_true_body_angular_rate_and_acceleration():
    source = _poses(
        [[0, 0, 1], [.4, .1, 1.1], [.8, -.1, 1.2], [1.2, 0, 1]],
        [[0, 0, 0], [.2, -.3, .3], [-.1, .2, .5], [.3, .1, .7]],
    )
    config = RetimingConfig(speed_mps=.8, max_acceleration_mps2=1.2,
                            max_angular_speed_rps=.6, max_angular_acceleration_rps2=1.1)
    reference = TimedPoseReference(source, config)
    times = np.linspace(config.initial_hold_s + .1, reference.duration - .1, 1001)
    dt = 1e-5
    state, acceleration = reference.sample(times)
    plus, _ = reference.sample(times + dt)
    minus, _ = reference.sample(times - dt)
    rotation = Rotation.from_quat(state[:, [7, 8, 9, 6]])
    plus_rotation = Rotation.from_quat(plus[:, [7, 8, 9, 6]])
    minus_rotation = Rotation.from_quat(minus[:, [7, 8, 9, 6]])
    numerical_omega = ((rotation.inv() * plus_rotation).as_rotvec()
                       - (rotation.inv() * minus_rotation).as_rotvec()) / (2 * dt)
    np.testing.assert_allclose(state[:, 10:], numerical_omega, atol=2e-6)
    np.testing.assert_allclose(acceleration, (plus[:, [3, 4, 5, 10, 11, 12]]
                                             - minus[:, [3, 4, 5, 10, 11, 12]]) / (2 * dt), atol=2e-4)

    dense, feedforward = reference.sample(np.linspace(0, reference.duration + .1, 24001))
    assert np.linalg.norm(dense[:, 3:6], axis=1).max() <= config.speed_mps
    assert np.linalg.norm(dense[:, 10:], axis=1).max() <= config.max_angular_speed_rps
    assert np.linalg.norm(feedforward[:, :3], axis=1).max() <= config.max_acceleration_mps2
    assert np.linalg.norm(feedforward[:, 3:], axis=1).max() <= config.max_angular_acceleration_rps2
    np.testing.assert_allclose(np.linalg.norm(dense[:, 6:10], axis=1), 1, atol=1e-12)


def test_smoothed_geometry_stays_in_tube_and_has_independent_dense_audit():
    source = _poses([[0, 0, 1], [1, 0, 1], [1, 1, 1], [2, 1, 1]])
    reference = TimedPoseReference(source)
    curve = reference.curve
    parameters = np.linspace(0, 1, 40001)
    poses = curve.geometry(parameters)
    error = np.linalg.norm(poses[:, :3] - curve._original_positions(parameters), axis=1)
    assert error.max() <= curve.position_tolerance_m
    assert curve.position_deviation_m <= curve.position_tolerance_m
    audited = reference.audit_poses(max_spacing_m=.006, max_rotation_rad=.006)
    np.testing.assert_allclose(audited[[0, -1]], source[[0, -1]], atol=1e-10)
    assert np.linalg.norm(np.diff(audited[:, :3], axis=0), axis=1).max() <= .006
    # Continuity is checked at geometric knots, including the rounded corners.
    for knot in curve.parameters[1:-1]:
        before = curve.derivatives([knot - 1e-8])
        after = curve.derivatives([knot + 1e-8])
        np.testing.assert_allclose(before[0], after[0], atol=1e-4)
        np.testing.assert_allclose(before[1], after[1], atol=.005)


def test_small_learned_waypoint_noise_is_smoothed_and_jerk_is_bounded():
    phase = np.linspace(0, 1, 128)
    random = np.random.default_rng(42)
    noise = random.normal(0, .001, (128, 2))
    noise[[0, -1]] = 0
    source = _poses(np.column_stack((5 * phase, .2 * np.sin(2 * np.pi * phase) + noise[:, 0],
                                    1 + .1 * np.sin(np.pi * phase) + noise[:, 1])))
    reference = TimedPoseReference(source)
    assert len(reference.curve.parameters) < 64
    assert reference.duration < 14
    times = np.linspace(reference.config.initial_hold_s, reference.duration, 30001)
    _, acceleration = reference.sample(times)
    jerk = np.diff(acceleration, axis=0) / np.diff(times)[:, None]
    assert np.linalg.norm(jerk[:, :3], axis=1).max() <= reference.max_linear_jerk_mps3 * 1.02
    assert reference.validation["estimated_max_linear_jerk_mps3"] <= reference.max_linear_jerk_mps3
    assert reference.describe()["kinematic_validation"]["valid"] is True


def test_start_and_end_holds_have_zero_kinematics_and_acceleration_is_eased():
    reference = TimedPoseReference(_poses([[0, 0, 1], [2, 0, 1]]))
    times = [-1, 0, reference.config.initial_hold_s, reference.duration, reference.duration + 1]
    state, acceleration = reference.sample(times)
    np.testing.assert_array_equal(state[:, 3:6], 0)
    np.testing.assert_array_equal(state[:, 10:], 0)
    np.testing.assert_array_equal(acceleration, 0)
    _, near = reference.sample([reference.config.initial_hold_s + 1e-6, reference.duration - 1e-6])
    np.testing.assert_allclose(near, 0, atol=1e-6)


def test_invalid_inputs_and_failed_solver_are_never_silently_replaced(monkeypatch):
    import toppra.algorithm

    with pytest.raises(ValueError, match="没有位移"):
        TimedPoseReference(_poses([[0, 0, 1], [0, 0, 1]]))
    with pytest.raises(ValueError, match="speed_mps"):
        TimedPoseReference(_poses([[0, 0, 1], [1, 0, 1]]), replace(RetimingConfig(), speed_mps=0))
    monkeypatch.setattr(toppra.algorithm.TOPPRA, "compute_trajectory", lambda *args: None)
    with pytest.raises(RuntimeError, match="TOPP-RA"):
        TimedPoseReference(_poses([[0, 0, 1], [1, 0, 1]]))


def test_antipodal_quaternions_and_duplicate_poses_do_not_create_angular_spikes():
    source = _poses([[0, 0, 1], [0, 0, 1], [1, 0, 1], [2, 0, 1]])
    source[1::2, 3:] *= -1
    reference = TimedPoseReference(source)
    state, acceleration = reference.sample(np.linspace(0, reference.duration, 1001))
    np.testing.assert_allclose(state[:, 6:10], np.tile([1, 0, 0, 0], (1001, 1)), atol=1e-12)
    np.testing.assert_array_equal(state[:, 10:], 0)
    np.testing.assert_array_equal(acceleration[:, 3:], 0)


def test_orientation_only_motion_is_retimed_with_angular_limits():
    source = _poses([[0, 0, 1], [0, 0, 1]], [[0, 0, 0], [.8, -.3, .4]])
    reference = TimedPoseReference(source)
    state, acceleration = reference.sample(np.linspace(0, reference.duration, 20001))
    np.testing.assert_allclose(state[:, :3], np.tile(source[0, :3], (20001, 1)), atol=1e-12)
    assert np.linalg.norm(state[:, 10:], axis=1).max() > .1
    assert np.linalg.norm(state[:, 10:], axis=1).max() <= reference.config.max_angular_speed_rps
    assert np.linalg.norm(acceleration[:, 3:], axis=1).max() <= reference.config.max_angular_acceleration_rps2
