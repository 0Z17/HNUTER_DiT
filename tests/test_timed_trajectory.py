import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hnuter_dit.timed_trajectory import TimedTrajectory


def path():
    # Uneven spacing, repeated poses, sign-equivalent quaternions, and coupled
    # noncommuting rotations exercise more than a straight-line special case.
    phase = np.array([0., .04, .04, .2, .6, .85, 1.])
    xyz = np.c_[2 * phase, .3 * np.sin(np.pi * phase), 2 + .1 * phase]
    q = Rotation.from_euler('xyz', np.c_[.3 * np.sin(phase * np.pi),
                                       .2 * phase, .7 * phase]).as_quat()[:, [3, 0, 1, 2]]
    q[2] *= -1
    return np.c_[xyz, q]


def test_derivatives_match_independent_position_and_rotation_differences():
    trajectory = TimedTrajectory(path())
    t = np.linspace(.01, .99, 31) * trajectory.duration
    h = 1e-4
    before, center, after = [trajectory.sample(t + offset) for offset in (-h, 0, h)]
    np.testing.assert_allclose((after['pos'] - before['pos']) / (2*h), center['vel'], atol=2e-7)
    np.testing.assert_allclose((after['vel'] - before['vel']) / (2*h), center['acc'], atol=2e-7)
    qbefore, qcenter, qafter = [Rotation.from_quat(row['quaternion'][:, [1, 2, 3, 0]])
                              for row in (before, center, after)]
    # R(t)^T R(t+h) is a body-frame increment. A world-frame derivative would
    # fail here because rotations around x/y/z are deliberately combined.
    body_rate = ((qcenter.inv() * qafter).as_rotvec()
                 - (qcenter.inv() * qbefore).as_rotvec()) / (2*h)
    np.testing.assert_allclose(body_rate, center['angular_velocity'], atol=2e-7)
    np.testing.assert_allclose((after['angular_velocity'] - before['angular_velocity']) / (2*h),
                               center['angular_acceleration'], atol=2e-7)


def test_endpoints_holds_and_dense_limits():
    poses = path()
    trajectory = TimedTrajectory(poses, max_speed=.35, max_acceleration=.25,
                                 max_angular_speed=.2, max_angular_acceleration=.3)
    endpoints = trajectory.sample([0., trajectory.duration])
    np.testing.assert_allclose(endpoints['pos'], poses[[0, -1], :3], atol=1e-12)
    for key in ('vel', 'acc', 'angular_velocity', 'angular_acceleration'):
        np.testing.assert_allclose(endpoints[key], 0, atol=1e-12)
    sample = trajectory.sample(np.linspace(0, trajectory.duration, 30001))
    for key, limit in [('vel', .35), ('acc', .25), ('angular_velocity', .2), ('angular_acceleration', .3)]:
        assert np.max(np.linalg.norm(sample[key], axis=1)) <= limit * (1+1e-6)
    np.testing.assert_allclose(np.linalg.norm(sample['quaternion'], axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(trajectory.sample(-1)['pos'], poses[0, :3], atol=1e-12)
    np.testing.assert_allclose(trajectory.sample(trajectory.duration+1)['pos'], poses[-1, :3], atol=1e-12)


def test_initial_motion_transition_is_c2_and_never_clamps_positions():
    v0, a0 = np.array([.15, -.03, .02]), np.array([.02, 0., 0.])
    trajectory = TimedTrajectory(path(), start_velocity=v0, start_acceleration=a0)
    start = trajectory.sample(0.)
    np.testing.assert_allclose(start['vel'], v0)
    np.testing.assert_allclose(start['acc'], a0)
    h = 1e-5
    np.testing.assert_allclose((trajectory.sample(h)['pos'] - start['pos']) / h, v0, atol=1e-6)
    end = trajectory._bridge_duration
    before, after = trajectory.sample(end-h), trajectory.sample(end+h)
    np.testing.assert_allclose(before['pos'], after['pos'], atol=1e-5)
    np.testing.assert_allclose(before['vel'], after['vel'], atol=1e-5)
    np.testing.assert_allclose(before['acc'], after['acc'], atol=1e-5)
    assert np.max(np.linalg.norm(trajectory.sample(np.linspace(0, trajectory.duration, 30001))['vel'], axis=1)) <= .5


def test_pure_attitude_stationary_and_invalid_inputs():
    values = np.tile([0., 0., 2., 1., 0., 0., 0.], (4, 1))
    static = TimedTrajectory(values)
    np.testing.assert_allclose(static.sample(.25)['vel'], 0.)
    values[-1, 3:] = [np.cos(.5), 0., 0., np.sin(.5)]
    turning = TimedTrajectory(values)
    assert turning.duration > static.duration
    np.testing.assert_allclose(turning.sample(turning.duration/2)['pos'], values[0, :3])
    with pytest.raises(ValueError, match='start_velocity'):
        TimedTrajectory(values, start_velocity=[1., 0., 0.])
    values[0, 3:] = 0.
    with pytest.raises(ValueError, match='quaternion'):
        TimedTrajectory(values)
