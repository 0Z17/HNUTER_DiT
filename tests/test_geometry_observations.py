from types import SimpleNamespace
import numpy as np
import pytest

from hnuter_dit.geometry_observations import (encode_geometry_history, reencode_dataset,
    input_spec, preprocessing_configuration, GRID_SIZE)
from hnuter_dit.temporal_observations import HISTORY_TIMES


def frames(points=None):
    if points is None:
        points = np.array([[1., 0, 0], [2, .2, 0], [0, 0, -1.]])
    return [SimpleNamespace(points=points.copy(), world_T_sensor=np.eye(4), sim_time=t+.6) for t in HISTORY_TIMES]


def state():
    return dict(position=np.zeros(3), velocity=np.zeros(3), quaternion=[1.,0,0,0])


@pytest.mark.parametrize('representation', ['point','voxel'])
def test_causal_geometry_pose_permutation_and_empty(representation):
    original = frames()
    encoded = encode_geometry_history(original, state(), [2,0,0], representation)
    spec = input_spec(representation)
    assert encoded['history_tokens'].shape == tuple(spec['tokens_shape'])
    assert encoded['history_mask'].shape == tuple(spec['mask_shape'])
    shuffled = frames(original[0].points[::-1])
    again = encode_geometry_history(shuffled, state(), [2,0,0], representation)
    np.testing.assert_array_equal(encoded['history_tokens'], again['history_tokens'])
    # Translating world origins, current state and goal together changes nothing.
    shift = np.array([12., -8., 4.])
    shifted = frames()
    for scan in shifted:
        scan.world_T_sensor[:3,3] = shift
    moved_state = state(); moved_state['position'] = shift
    moved = encode_geometry_history(shifted, moved_state, shift+[2,0,0], representation)
    np.testing.assert_array_equal(encoded['history_tokens'], moved['history_tokens'])
    np.testing.assert_array_equal(encoded['condition'], moved['condition'])
    empty = encode_geometry_history(frames(np.empty((0,3))), state(), [2,0,0], representation)
    assert not empty['history_mask'].any()
    if representation == 'voxel':
        assert empty['history_tokens'][:,2].all()
        assert not empty['history_tokens'][:,:2].any()


def test_free_evidence_is_only_before_a_measured_endpoint_and_unknown_stays_unknown():
    encoded = encode_geometry_history(frames(np.array([[2.,0,0]])), state(), [2,0,0], 'voxel')
    grid = encoded['history_tokens'][0]
    # .25m cells: measured x=2 lies in x40; x=.5 is traversed; x=3 is unseen.
    assert grid[0,40,32,32] == 1 and grid[1,40,32,32] == 0
    assert grid[1,34,32,32] == 1
    assert grid[2,44,32,32] == 1 and grid[2,32,40,32] == 1
    assert np.all(grid.sum(axis=0) == 1)
    # Even returns outside the ROI can supply causal free-ray evidence within it.
    farther = encode_geometry_history(frames(np.array([[10.,0,0]])), state(), [2,0,0], 'voxel')
    assert not farther['history_tokens'][:,0].any()
    assert farther['history_tokens'][0,1,40,32,32]
    assert farther['history_tokens'][0,2,63,63,63]


@pytest.mark.parametrize('representation', ['point','voxel'])
def test_offline_reencoding_matches_live_and_ignores_future_fields(representation):
    scans = frames(np.array([[1.,0,0],[0,2.,0],[0,0,-1.]]))
    data = dict(history_ray_ranges=np.tile([1.,2.,1.,np.nan], (1,4,1)),
        ray_directions=np.tile([[1.,0,0],[0,1.,0],[0,0,-1.],[0,-1.,0]],(4,1,1)),
        history_sensor_poses=np.tile(np.eye(4),(1,4,1,1)),
        origin_world=np.zeros((1,3)), world_to_canonical=np.eye(3)[None],
        gt_boxes=np.full((1,65,1,10),99999.), teacher_wait_s=np.array([99999.]))
    offline = reencode_dataset(data, representation)
    live = encode_geometry_history(scans, state(), [2,0,0], representation)
    np.testing.assert_array_equal(offline['history_tokens'][0], live['history_tokens'])
    np.testing.assert_array_equal(offline['history_mask'][0], live['history_mask'])
    data['gt_boxes'] *= -1; data['teacher_wait_s'] *= -1
    np.testing.assert_array_equal(offline['history_tokens'], reencode_dataset(data, representation)['history_tokens'])
    assert offline['_preprocessing'] == preprocessing_configuration(representation)


def test_history_time_rigid_pose_and_ranges_validation():
    wrong = frames(); wrong[0].sim_time += .01
    with pytest.raises(ValueError, match='timestamps'):
        encode_geometry_history(wrong, state(), [2,0,0], 'point')
    wrong = frames(); wrong[0].world_T_sensor[0,0] = 2
    with pytest.raises(ValueError, match='rigid'):
        encode_geometry_history(wrong, state(), [2,0,0], 'voxel')
