"""Geometry causal IPC dispatch; no Isaac or GPU dependency."""
from types import SimpleNamespace
import numpy as np
import pytest

from hnuter_dit.geometry_observations import input_spec
from hnuter_dit.planner_process import PlannerProcess
from hnuter_dit.temporal_online import TemporalOnlinePlanner


@pytest.mark.parametrize('representation', ['point', 'voxel'])
def test_geometry_ipc_uses_checkpoint_shape_and_keeps_only_causal_fields(representation):
    client = PlannerProcess.__new__(PlannerProcess)
    client.runtime_kind = 'temporal'
    client._description = {'input_representation': representation}
    spec = input_spec(representation)
    tokens = np.zeros(spec['tokens_shape'], dtype=spec['tokens_dtype'])
    encoding = dict(history_tokens=tokens, history_mask=np.zeros(spec['mask_shape'], bool),
        history_times=np.array([-.6,-.4,-.2,0.]), condition=np.array([1,0,0,0,0,0,1,0,0,0,1,0.]),
        future_obstacles='never transmitted', gt_boxes=np.ones((65, 4, 10)))
    requests = []
    def request(payload, fields):
        requests.append(payload)
        return dict(positions=np.zeros((65,3)), velocities=np.zeros((65,3)), accelerations=np.zeros((65,3)),
                    future_times=np.arange(65)*.125, control_points=np.zeros((16,3))), {}
    client._request = request
    assert client.sample_temporal(encoding, 7)['positions'].shape == (65,3)
    assert requests[0]['input_representation'] == representation
    assert requests[0]['history_tokens'].dtype == np.dtype(spec['tokens_dtype'])
    assert 'future_obstacles' not in requests[0] and 'gt_boxes' not in requests[0]
    with pytest.raises(ValueError, match='conditioning'):
        client.sample_temporal(dict(encoding, history_tokens=np.zeros((4,32,10))), 7)
    with pytest.raises(ValueError, match='conditioning'):
        client.sample_temporal(dict(encoding, history_mask=np.zeros(spec['mask_shape'], np.uint8)), 7)


@pytest.mark.parametrize('representation', ['point', 'voxel'])
def test_online_dispatches_measured_history_using_checkpoint_representation(monkeypatch, representation):
    import hnuter_dit.geometry_observations as module
    planner = TemporalOnlinePlanner.__new__(TemporalOnlinePlanner)
    planner.input_representation = representation
    planner.goal = np.array([2.,0.,2.])
    planner.seed = 7
    planner.max_speed, planner.max_acceleration = .5, .6
    calls = []
    def encode(frames, state, goal, *, representation, query_time):
        calls.append((frames, state, representation, query_time))
        return dict(metadata={'causal': True}, condition=np.zeros(12), origin_world=np.array([0.,0.,2.]),
                    world_to_canonical=np.eye(3))
    monkeypatch.setattr(module, 'encode_geometry_history', encode)
    prediction = dict(metadata={}, future_times=np.arange(65)*.125, control_points=np.zeros((16,3)),
        positions=np.zeros((65,3)), velocities=np.zeros((65,3)), accelerations=np.zeros((65,3)))
    planner.runtime = SimpleNamespace(sample_temporal=lambda encoding, seed: prediction)
    state = dict(quaternion=np.array([1.,0.,0.,0.]))
    result, path = planner._plan(['measured history'], state, .6)
    assert calls[0] == (['measured history'], state, representation, .6)
    assert result['status'] == 'candidate' and path.duration == 8.
