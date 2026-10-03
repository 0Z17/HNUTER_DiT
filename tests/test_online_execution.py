"""Execution behavior without loading a checkpoint or creating a GPU worker."""
from concurrent.futures import Future
from types import SimpleNamespace
import time

import numpy as np
import pytest

from hnuter_dit.observation_guard import ObservationGuard
from hnuter_dit.online import BrakeReference, OnlinePlanner
from hnuter_dit.timed_trajectory import TimedTrajectory


def state(position=(0, 0, 0), velocity=(0, 0, 0)):
    return dict(position=np.asarray(position, dtype=float),
                velocity=np.asarray(velocity, dtype=float),
                quaternion=np.array([1., 0., 0., 0.]))


def scan(points=((3, 0, 0),), timestamp=1., position=(0, 0, 0)):
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(position) + [0, 0, .13]
    return SimpleNamespace(points=np.asarray(points, dtype=float).reshape(-1, 3),
                           world_T_sensor=transform, sim_time=timestamp)


def trajectory(goal=(2, 0, 0)):
    poses = np.array([[0, 0, 0, 1, 0, 0, 0], [*goal, 1, 0, 0, 0]], dtype=float)
    return TimedTrajectory(poses), poses


def planner(frame=None):
    """Skip only external worker construction; exercise real planner methods."""
    instance = OnlinePlanner.__new__(OnlinePlanner)
    instance.mode, instance.goal, instance.seed = 'online', np.array([2., 0., 0.]), 7
    instance.future = instance.path = instance.brake = None
    instance.path_start = instance.brake_start = 0.
    instance.hold = None
    instance.observation = frame
    instance.guard = ObservationGuard(frame) if frame is not None else None
    instance.last_request = -np.inf
    instance.events, instance.guard_events, instance.planning_seconds = [], [], []
    instance.closed, instance.raw_paths, instance.last_state = False, [], None
    return instance


def finished_plan(reference, poses, scan_time, planning_seconds=.1):
    future = Future()
    future.set_result((dict(scan_time=scan_time, planning_seconds=planning_seconds,
                            status='candidate'), reference, poses))
    return future


def test_brake_reference_derivatives_match_motion_and_limits():
    measured = state((-.5, 2, 1), (.2, -.3, .1))
    brake = BrakeReference(measured)
    first, last = brake.sample(0), brake.sample(brake.duration)
    np.testing.assert_array_equal(first['pos'], measured['position'])
    np.testing.assert_array_equal(first['vel'], measured['velocity'])
    np.testing.assert_allclose(last['vel'], 0, atol=1e-14)
    np.testing.assert_allclose(last['acc'], 0, atol=1e-14)
    # Independent finite differences test the derivatives consumed by SE(3).
    h = brake.duration * 1e-5
    for fraction in (.05, .2, .5, .8, .95):
        t = fraction * brake.duration
        before, sample, after = brake.sample(t-h), brake.sample(t), brake.sample(t+h)
        np.testing.assert_allclose((after['pos']-before['pos'])/(2*h), sample['vel'], atol=2e-9)
        np.testing.assert_allclose((after['vel']-before['vel'])/(2*h), sample['acc'], atol=2e-9)
    samples = [brake.sample(t) for t in np.linspace(0, brake.duration, 301)]
    assert max(np.linalg.norm(value['acc']) for value in samples) <= .6 + 1e-12
    assert max(np.linalg.norm(value['vel']) for value in samples) <= np.linalg.norm(measured['velocity']) + 1e-12
    held = brake.sample(brake.duration + 2)
    np.testing.assert_array_equal(held['pos'], last['pos'])
    np.testing.assert_array_equal(held['vel'], last['vel'])


def test_stale_scan_brakes_from_measured_motion_then_holds():
    instance = planner(scan(timestamp=0.))
    instance.path, _ = trajectory()
    measured = state((.2, 0, 0), (.4, 0, 0))
    command = instance.command(measured, .5)
    assert instance.path is None and instance.brake is not None
    assert instance.guard_events[-1]['reason'] == 'stale_scan'
    np.testing.assert_array_equal(command['pos'], measured['position'])
    np.testing.assert_array_equal(command['vel'], measured['velocity'])
    duration = instance.brake.duration
    stopped = instance.command(measured, .5+duration)
    assert instance.brake is None
    np.testing.assert_allclose(stopped['vel'], 0, atol=1e-14)
    np.testing.assert_allclose(instance.command(measured, 3+duration)['pos'], stopped['pos'])
    assert len(instance.guard_events) == 1  # Do not restart a deceleration every step.


@pytest.mark.parametrize('points', [[], [(3, 0, 0)]])
def test_latest_unknown_scan_stops_active_unobserved_motion(points):
    instance = planner()
    instance.path, _ = trajectory((0, 2, 0))
    instance.path_start = 1.
    instance.future = Future()  # An outstanding inference must not bypass sensing.
    measured = state(velocity=(0, .3, 0))
    instance.observe(scan(points, timestamp=1.), measured, 1.)
    assert instance.path is None and instance.brake is not None
    assert not instance.guard_events[-1]['allowed']
    desired = instance.command(measured, 1.)
    np.testing.assert_array_equal(desired['pos'], measured['position'])
    np.testing.assert_array_equal(desired['vel'], measured['velocity'])


@pytest.mark.parametrize('scan_time,planning_seconds', [(0., .1), (.9, 2.1)])
def test_old_or_slow_inference_cannot_activate_a_candidate(scan_time, planning_seconds):
    instance = planner(scan(timestamp=1.))
    candidate, poses = trajectory()
    instance.future = finished_plan(candidate, poses, scan_time, planning_seconds)
    desired = instance.command(state(), 1.)
    assert instance.path is None
    assert instance.events[-1]['status'] == 'stale_plan_rejected'
    np.testing.assert_array_equal(desired['pos'], [0, 0, 0])
    assert not instance.raw_paths


def test_fresh_plan_does_not_activate_into_unknown_direction():
    instance = planner(scan(timestamp=1.))
    candidate, poses = trajectory((0, 2, 0))
    instance.future = finished_plan(candidate, poses, .9)
    desired = instance.command(state(), 1.)
    assert instance.path is None
    assert instance.events[-1]['status'] == 'prefix_rejected'
    assert instance.events[-1]['execution_guard']['reason'] == 'unknown_centerline'
    np.testing.assert_array_equal(desired['pos'], [0, 0, 0])


def test_fresh_observed_candidate_activates_without_position_jump():
    instance = planner(scan(timestamp=1.))
    candidate, poses = trajectory()
    instance.future = finished_plan(candidate, poses, .9)
    desired = instance.command(state(), 1.)
    assert instance.path is candidate
    assert instance.events[-1]['status'] == 'accepted'
    np.testing.assert_array_equal(desired['pos'], [0, 0, 0])
    np.testing.assert_allclose(desired['vel'], 0, atol=1e-14)


def test_failed_worker_request_becomes_braking_not_control_loop_exception():
    instance = planner(scan(timestamp=1.))
    instance.path, _ = trajectory()
    instance.path_start = 1.
    instance.future = Future()
    instance.future.set_exception(RuntimeError('DiT request timed out'))
    measured = state(velocity=(.3, 0, 0))
    desired = instance.command(measured, 1.)
    assert instance.path is None and instance.brake is not None
    np.testing.assert_array_equal(desired['pos'], measured['position'])
    np.testing.assert_array_equal(desired['vel'], measured['velocity'])
    assert instance.events or instance.guard_events


def test_pending_deadline_brakes_and_late_result_cannot_reactivate():
    instance = planner(scan(timestamp=1.))
    instance.path, poses = trajectory()
    candidate = instance.path
    instance.path_start = 1.
    pending = Future()
    instance.future = pending
    instance.request_wall_time = time.perf_counter() - 2.1
    measured = state(velocity=(.3, 0, 0))
    desired = instance.command(measured, 1.)
    assert instance.path is None and instance.brake is not None
    np.testing.assert_array_equal(desired['pos'], measured['position'])
    np.testing.assert_array_equal(desired['vel'], measured['velocity'])
    pending.set_result((dict(scan_time=1., planning_seconds=.1, status='candidate'), candidate, poses))
    # Scan age, speed and start drift all pass now: only the expired request
    # identity must prevent this otherwise-valid delayed reply being used.
    instance.command(state(), 1.1)
    assert instance.path is None
    assert not instance.raw_paths


def test_old_plan_start_does_not_jump_reference_back_twenty_centimeters():
    measured = state((.15, 0, 0))
    frame = scan(((3, 0, 0), (-3, 0, 0)), timestamp=1., position=measured['position'])
    instance = planner(frame)
    candidate, poses = trajectory()
    instance.future = finished_plan(candidate, poses, .9)
    desired = instance.command(measured, 1.)
    assert np.linalg.norm(desired['pos'] - measured['position']) <= .05


def test_old_plan_attitude_does_not_step_back_to_previous_orientation():
    measured = state()
    yaw = .25
    measured['quaternion'] = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
    frame = scan(timestamp=1.)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    frame.world_T_sensor[:3, :3] = rotation
    frame.points = np.array([[3., 0, 0]]) @ rotation
    instance = planner(frame)
    candidate, poses = trajectory()
    instance.future = finished_plan(candidate, poses, .9)
    desired = instance.command(measured, 1.)
    dot = abs(desired['quaternion'] @ measured['quaternion'])
    assert 2 * np.arccos(np.clip(dot, 0, 1)) <= .1


def test_candidate_selection_skips_shorter_unknown_prefix_for_observed_route():
    # A measured plane ahead supplies hit rays at and above the sensor horizon.
    # A downward route is geometrically clear but its departure is unobserved.
    y, z = np.meshgrid(np.linspace(-1., 1., 21), np.linspace(0., 3., 101))
    frame = scan(np.c_[np.full(y.size, 3.), y.ravel(), z.ravel()], timestamp=1.)
    phase = np.linspace(0., 1., 41)
    paths = np.zeros((2, len(phase), 7))
    paths[:, :, 0] = 2 * phase
    paths[:, :, 3] = 1.
    paths[0, :, 2] = -.1 * np.sin(np.pi * phase)
    paths[1, :, 2] = .3 * np.sin(np.pi * phase)
    paths[:, [0, -1], 2] = 0.
    lengths = np.linalg.norm(np.diff(paths[:, :, :3], axis=1), axis=2).sum(axis=1)
    assert lengths[0] < lengths[1]
    np.testing.assert_array_equal(paths[0, [0, -1]], paths[1, [0, -1]])

    class FixedCandidates:
        def sample(self, mode, tokens, mask, start, goal, **options):
            return paths.copy(), {'algorithm': 'test_candidates_no_model', 'inference_seconds': 0.}

    instance = planner(frame)
    instance.runtime = FixedCandidates()
    # Exercise the real condition encoding, spline, clearance, guard and search.
    result, reference, selected = instance._plan(frame, state(), 1.)
    assert result['observed_collision_free'] == 2
    assert result['selected'] == 1
    assert reference is not None
    np.testing.assert_array_equal(selected, paths[1])
    assert result['candidate_execution_guards']['0']['reason'] == 'unknown_centerline'
    assert not result['candidate_execution_guards']['0']['allowed']
    assert result['candidate_execution_guards']['1']['allowed']
