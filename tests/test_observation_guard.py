from types import SimpleNamespace
import numpy as np
import pytest
from hnuter_dit.observation_guard import ObservationGuard


def frame(points,stamp=1.):
    transform=np.eye(4);transform[2,3]=.13
    return SimpleNamespace(points=np.asarray(points,dtype=float).reshape(-1,3),world_T_sensor=transform,sim_time=stamp)


def poses(xs):
    return np.array([[x,0,0,1,0,0,0] for x in xs],dtype=float)


def test_visible_prefix_and_stale_scan():
    guard=ObservationGuard(frame([[3,0,0],[3,.1,0],[3,-.1,0]]))
    assert guard.check(poses([0,.2,.8]),sim_time=1.1)['allowed']
    assert guard.check(poses([0,.2,.8]),sim_time=1.4)['reason']=='stale_scan'


def test_unobserved_direction_and_empty_frame_do_not_become_free():
    guard=ObservationGuard(frame([[3,0,0]]))
    path=poses([0,.2,.8]);path[1:,1]=path[1:,0];path[:,0]=0
    result=guard.check(path,sim_time=1.)
    assert not result['allowed'] and result['reason']=='unknown_centerline'
    assert not ObservationGuard(frame([])).check(poses([0,.8]),sim_time=1.)['allowed']
    assert not ObservationGuard(frame([[.05,0,0]])).check(poses([0,.8]),sim_time=1.)['allowed']


def test_near_blind_zone_alone_is_not_evidence_for_advancing():
    result=ObservationGuard(frame([[3,0,0]])).check(poses([0,.05]),sim_time=1.)
    assert not result['allowed']
    assert result['unknown_fraction']==1.


def test_observed_obstacle_checks_robot_extent_not_only_sensor():
    guard=ObservationGuard(frame([[1.,.4,-.13],[3,0,0]]))
    result=guard.check(poses([0,.2,.8]),sim_time=1.)
    assert not result['allowed'] and result['reason']=='observed_obstacle'


def test_invalid_pose_timestamp_and_quaternion_rejected():
    bad=frame([[3,0,0]]);bad.world_T_sensor[0,0]=2
    with pytest.raises(ValueError):ObservationGuard(bad)
    with pytest.raises(ValueError):ObservationGuard(frame([[3,0,0]],float('nan')))
    guard=ObservationGuard(frame([[3,0,0]]));bad_poses=poses([0,.8]);bad_poses[1,3:]=0
    with pytest.raises(ValueError):guard.check(bad_poses,sim_time=1.)


def ray_frame(angles_degrees, ranges):
    angles = np.deg2rad(np.asarray(angles_degrees, dtype=float))
    directions = np.c_[np.cos(angles), np.sin(angles), np.zeros(len(angles))]
    ranges = np.asarray(ranges, dtype=float)
    hits = np.isfinite(ranges) & (ranges > .1)
    value = frame(directions[hits] * ranges[hits, None])
    value.ray_directions = directions
    value.ranges = ranges
    return value


@pytest.mark.parametrize('missing_range', [np.nan, np.inf])
def test_nearest_emitted_miss_is_not_filled_by_neighboring_hit(missing_range):
    value = ray_frame([0., 1.], [missing_range, 5.])
    result = ObservationGuard(value).check(poses([0, .2, .8]), sim_time=1.)
    assert not result['allowed']
    assert result['reason'] == 'unknown_centerline'
    assert result['unknown_fraction'] == 1.
    assert result['ray_evidence'] == 'nearest_emitted_ray_including_misses'
    # This hit alone would have passed the old return-neighbor heuristic.
    assert ObservationGuard(frame(value.points)).check(poses([0, .2, .8]), sim_time=1.)['allowed']


def test_two_degree_emitted_hit_is_accepted_only_inside_transverse_bound():
    guard = ObservationGuard(ray_frame([2.], [5.]))
    near = guard.check(poses([0, .2, 1.]), sim_time=1.)
    assert near['allowed']  # 1 m * sin(2 deg) = 3.49 cm < 5 cm.
    assert near['angular_tolerance_degrees'] == 3.
    assert near['max_transverse_beam_distance_m'] == .05
    far = guard.check(poses([0, .2, 2.]), sim_time=1.)
    assert not far['allowed']  # 2 m * sin(2 deg) = 6.98 cm > 5 cm.
    assert far['reason'] == 'unknown_centerline'


def test_over_three_degrees_is_unknown_even_with_small_transverse_distance():
    guard = ObservationGuard(ray_frame([3.1], [5.]))
    result = guard.check(poses([0, .2]), sim_time=1.)
    assert .2 * np.sin(np.deg2rad(3.1)) < .05
    assert not result['allowed'] and result['reason'] == 'unknown_centerline'


def test_points_only_frame_retains_one_point_five_degree_default():
    value = ray_frame([2.], [5.])
    result = ObservationGuard(frame(value.points)).check(poses([0, .2, .8]), sim_time=1.)
    assert not result['allowed']
    assert result['angular_tolerance_degrees'] == 1.5
    assert result['ray_evidence'] == 'returns_only'


@pytest.mark.parametrize('directions,ranges', [
    ([[1., 0.]], [5.]),
    ([[1., 0., 0.]], [5., 6.]),
    ([[0., 0., 0.]], [5.]),
    ([[2., 0., 0.]], [5.]),
    ([[np.nan, 0., 0.]], [5.]),
])
def test_malformed_full_ray_geometry_is_rejected(directions, ranges):
    value = frame([[5, 0, 0]])
    value.ray_directions, value.ranges = np.asarray(directions), np.asarray(ranges)
    with pytest.raises(ValueError, match='ray frame'):
        ObservationGuard(value)


@pytest.mark.parametrize('missing_attribute', ['ray_directions', 'ranges'])
def test_incomplete_full_ray_frame_cannot_fall_back_to_hit_only_evidence(missing_attribute):
    value = ray_frame([0., 1.], [np.nan, 5.])
    delattr(value, missing_attribute)
    with pytest.raises(ValueError, match='ray frame'):
        ObservationGuard(value)
