"""Temporal IPC and execution semantics; no Isaac or GPU initialization."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import BSpline

from hnuter_dit.planner_process import PlannerProcess
from hnuter_dit.temporal_online import HistoryBuffer, TemporalOnlinePlanner
from hnuter_dit.temporal_trajectory import FixedTimeTrajectory, KNOTS, FUTURE_TIMES


def linear_controls(velocity):
    greville = np.array([KNOTS[i+1:i+4].mean() for i in range(16)])
    return greville[:, None] * np.asarray(velocity)[None]


def state(position=(0., 0., 2.), velocity=(0., 0., 0.)):
    return dict(position=np.asarray(position), velocity=np.asarray(velocity),
                quaternion=np.array([1., 0., 0., 0.]), angular_velocity=np.zeros(3), rotation_matrix=np.eye(3))


def frame(t):
    transform = np.eye(4)
    transform[:3, 3] = [0., 0., 2.13]
    return SimpleNamespace(points=np.array([[3., 0., 0.], [3., 1., 0.]]),
                           world_T_sensor=transform, sim_time=t, frame_index=round(t*10),
                           point_timestamps=np.full(2, t))


def prediction(controls):
    spline = BSpline(KNOTS, controls, 3)
    return dict(control_points=controls, future_times=FUTURE_TIMES.copy(),
                positions=spline(FUTURE_TIMES), velocities=spline(FUTURE_TIMES, 1),
                accelerations=spline(FUTURE_TIMES, 2), inference_seconds=0., metadata={'fixture': True})


def test_fixed_time_preserves_wait_and_does_not_erase_duplicate_controls():
    controls = np.zeros((16, 3))
    controls[8:, 0] = np.linspace(.01, .1, 8)
    path = FixedTimeTrajectory(controls, quaternion=[1, 0, 0, 0])
    sample = path.sample(np.linspace(0, 2, 101))
    assert np.max(np.abs(sample['pos'])) == 0
    assert np.max(np.abs(sample['vel'])) == 0
    assert np.max(np.abs(sample['acc'])) == 0
    assert path.sample(7.)['pos'][0] > 0
    assert path.duration == 8.
    assert path.control_points.shape == (16, 3)
    with pytest.raises(ValueError, match='fixed'):
        path.sample(8.1)


def test_fixed_time_derivatives_and_limits_are_physical_without_retiming():
    velocity = np.array([.15, -.1, .03])
    controls = linear_controls(velocity)
    path = FixedTimeTrajectory(controls, quaternion=[2, 0, 0, 0])
    times = np.linspace(0, 8, 2001)
    samples = path.sample(times)
    np.testing.assert_allclose(samples['pos'], times[:, None]*velocity, atol=1e-12)
    np.testing.assert_allclose(samples['vel'], np.tile(velocity, (len(times), 1)), atol=1e-12)
    assert np.abs(samples['acc']).max() < 1e-12
    assert path.observed_maxima['speed'] == pytest.approx(np.linalg.norm(velocity))
    with pytest.raises(ValueError, match='speed limit exceeded'):
        FixedTimeTrajectory(controls, quaternion=[1, 0, 0, 0], max_speed=.1)


def test_cubic_velocity_extrema_include_between_knot_peaks():
    rng = np.random.default_rng(52)
    controls = np.cumsum(rng.normal(0, .003, size=(16, 3)), axis=0)
    path = FixedTimeTrajectory(controls, quaternion=[1, 0, 0, 0], max_speed=1., max_acceleration=1.)
    sampled = path.sample(np.linspace(0, 8, 30001))
    for field, limit in (('vel', 'speed'), ('acc', 'acceleration')):
        maximum = np.linalg.norm(sampled[field], axis=1).max()
        assert path.observed_maxima[limit] >= maximum-1e-10
        assert path.observed_maxima[limit] == pytest.approx(maximum, abs=1e-7)


def test_prediction_world_rotation_and_spline_evidence_are_checked():
    controls = linear_controls([.1, 0, 0])
    encoded = dict(origin_world=np.array([-25., -35., 2.2]),
                   world_to_canonical=np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]))
    result = prediction(controls)
    path = FixedTimeTrajectory.from_prediction(result, encoded, [1, 0, 0, 0])
    np.testing.assert_allclose(path.sample(2)['pos'], [-25., -34.8, 2.2])
    np.testing.assert_allclose(path.sample(2)['vel'], [0., .1, 0.], atol=1e-12)
    result['positions'][10, 0] += .01
    with pytest.raises(ValueError, match='disagree'):
        FixedTimeTrajectory.from_prediction(result, encoded, [1, 0, 0, 0])


def test_history_uses_four_real_causal_scans_and_owns_array_copies():
    history = HistoryBuffer()
    first = frame(0.)
    history.append(first, state(), 0.)
    first.points[:] = 123.
    for index in range(1, 7):
        history.append(frame(index*.1), state(), index*.1)
    selection = history.select(.6)
    np.testing.assert_allclose([item[0].sim_time for item in selection], [0., .2, .4, .6])
    assert selection[0][0].points[0, 0] == 3.
    assert not selection[0][0].points.flags.writeable
    with pytest.raises(ValueError, match='not available'):
        history.select(.65)
    with pytest.raises(ValueError, match='increasing'):
        history.append(frame(.6), state(), .6)
    invalid = frame(.7)
    invalid.sim_time = float('nan')
    with pytest.raises(ValueError, match='timestamp'):
        history.append(invalid, state(), .7)


class FakeRuntime:
    def __init__(self, **kwargs):
        self.calls = 0
        self.closed = False
    def sample_temporal(self, encoding, seed):
        self.calls += 1
        return prediction(np.zeros((16, 3)))
    def describe(self):
        return {'checkpoint': 'test_double', 'fixture': True}
    def close(self):
        self.closed = True


@pytest.fixture
def planner(monkeypatch):
    import hnuter_dit.temporal_online as module
    monkeypatch.setattr(module, 'PlannerProcess', FakeRuntime)
    result = TemporalOnlinePlanner(goal_position=[0, 3, 2], checkpoint='test_double', device='cpu')
    result.guard = SimpleNamespace(check=lambda poses, **kwargs: dict(
        allowed=True, reason='test_clear', minimum_clearance_m=1.))
    result.observation = frame(10.)
    yield result
    result.close()


def plan_result(time=10.):
    return dict(scan_time=time, planning_seconds=.1, status='candidate')


def test_activation_consumes_latency_on_original_time_axis(planner):
    path = FixedTimeTrajectory(linear_controls([.05, 0, 0])+[0, 0, 2], quaternion=[1, 0, 0, 0])
    measured = state((.0125, 0., 2.), (.01, 0., 0.))
    planner._activate(plan_result(), path, measured, 10.25)
    assert planner.events[-1]['status'] == 'accepted'
    assert planner.path_start == 10.
    planner.observation = frame(10.3)
    assert planner.command(measured, 10.3)['pos'][0] == pytest.approx(.015)
    assert planner.events[-1]['consumed_prediction_seconds'] == .25


def test_activation_rejects_stale_and_velocity_discontinuous_predictions(planner):
    stationary = FixedTimeTrajectory(np.zeros((16, 3))+[0, 0, 2], quaternion=[1, 0, 0, 0])
    planner._activate(plan_result(), stationary, state(), 10.501)
    assert planner.events[-1]['status'] == 'stale_plan_rejected'
    fast = FixedTimeTrajectory(linear_controls([.2, 0, 0])+[0, 0, 2], quaternion=[1, 0, 0, 0])
    planner._activate(plan_result(), fast, state(), 10.)
    assert planner.events[-1]['status'] == 'state_join_rejected'
    assert planner.path is None


def test_model_wait_and_guard_brake_are_accounted_separately(planner):
    planner.guard = SimpleNamespace(check=lambda poses, **kwargs: dict(
        allowed=False, reason='unknown_centerline', minimum_clearance_m=1.))
    path = FixedTimeTrajectory(np.zeros((16, 3))+[0, 0, 2], quaternion=[1, 0, 0, 0])
    planner._activate(plan_result(), path, state(), 10.)
    assert planner.events[-1]['execution_guard']['reason'] == 'model_stationary_prediction'
    planner.command(state(), 10.)
    planner.command(state(), 10.1)
    planner._brake(state(), 10.2, dict(allowed=False, reason='observed_obstacle'))
    planner.command(state(), 10.2)
    report = planner.close()
    assert report['time_in_state_seconds']['model_wait'] == pytest.approx(.2)
    assert report['guard_interventions'] == 1
    assert report['accepted_plans'] == 1


def test_active_reference_is_not_restarted_by_new_observations(planner, monkeypatch):
    import hnuter_dit.temporal_online as module
    monkeypatch.setattr(module, 'ObservationGuard', lambda frame: SimpleNamespace(check=lambda *a, **k: dict(
        allowed=True, reason='test_clear', minimum_clearance_m=1.)))
    planner.path = FixedTimeTrajectory(linear_controls([.05, 0, 0])+[0, 0, 2], quaternion=[1, 0, 0, 0])
    planner.path_start = 0.
    for i in range(10):
        planner.observe(frame(i*.1), state(), i*.1)
    assert planner.future is None and planner.runtime.calls == 0
    assert planner.path_start == 0.


def test_moving_history_is_outside_pilot_support(planner, monkeypatch):
    import hnuter_dit.temporal_online as module
    monkeypatch.setattr(module, 'ObservationGuard', lambda frame: None)
    for i in range(7):
        planner.observe(frame(i*.1), state(velocity=(.05, 0., 0.)), i*.1)
    assert planner.future is None
    assert planner.waiting_reason == 'moving_start_unsupported'


def test_temporal_atomic_ipc_transmits_no_oracle_fields(tmp_path, monkeypatch):
    script = tmp_path/'worker.py'
    script.write_text('''import json,sys,time
from pathlib import Path
import numpy as np
d=Path(sys.argv[1]); sequence=0
def status(v):
 p=d/'status.tmp';p.write_text(json.dumps(v));p.replace(d/'status.json')
status(dict(state='ready',model=dict(fixture=True)))
while True:
 if (d/'request.npz').exists():
  with np.load(d/'request.npz') as r:
   new=int(r['sequence']); keys=list(r.files)
  if new>sequence:
   sequence=new
   with (d/'response.tmp').open('wb') as f:
    np.savez(f,sequence=sequence,positions=np.zeros((65,3)),velocities=np.zeros((65,3)),accelerations=np.zeros((65,3)),control_points=np.zeros((16,3)),future_times=np.arange(65)*.125,metadata=json.dumps(dict(input_keys=keys,inference_seconds=0.)))
   (d/'response.tmp').replace(d/'response.npz')
 if (d/'stop.json').exists():
  status(dict(state='stopped',sequence=sequence));break
 time.sleep(.005)
''')
    checkpoint = tmp_path/'fixture.pt'
    checkpoint.write_bytes(b'protocol fixture, not model weights')
    monkeypatch.setattr(PlannerProcess, '_worker_script', script)
    encoded = dict(history_tokens=np.zeros((4,32,10)), history_mask=np.zeros((4,32), bool),
                   history_times=np.array([-.6,-.4,-.2,0.]), condition=np.zeros(12),
                   gt_boxes=np.ones((65,5,10)), obstacle_schedule='must never cross IPC')
    with PlannerProcess(tmp_path, python=sys.executable, device='cpu', temporal_checkpoint=checkpoint) as client:
        result = client.sample_temporal(encoded, 7)
        assert result['positions'].shape == (65,3)
        assert 'gt_boxes' not in result['metadata']['input_keys']
        assert 'obstacle_schedule' not in result['metadata']['input_keys']
        config = json.loads((client.output_dir/'config.json').read_text())
        assert config['runtime_kind'] == 'temporal' and len(config['expected_checkpoint_sha256']) == 64
        with pytest.raises(ValueError, match='relative timestamps'):
            client.sample_temporal(dict(encoded, history_times=np.zeros(4)), 7)
        with pytest.raises(ValueError, match='conditioning'):
            client.sample_temporal(dict(encoded, history_mask=np.ones((4,32), dtype=int)), 7)
        with pytest.raises(ValueError, match=r'\[1,100\]'):
            client.sample_temporal(encoded, 7, steps=101)


def test_runner_requires_explicit_temporal_checkpoint(tmp_path):
    from hnuter_dit.industrial_runner import arguments
    with pytest.raises(SystemExit):
        arguments(['--mode', 'temporal'])
    with pytest.raises(SystemExit):
        arguments(['--mode', 'online', '--temporal-checkpoint', str(tmp_path/'missing.pt')])
    with pytest.raises(SystemExit):
        arguments(['--dynamic-profile', 'constant_incoming'])


def test_real_temporal_worker_loads_explicit_cpu_checkpoint_and_decodes_same_spline(tmp_path):
    # A tiny untrained fixture exercises the actual loader/worker, independently
    # of the protocol double. It is never reported as trained model evidence.
    import torch
    from hnuter_dit.temporal_model import (TemporalDiT, CHECKPOINT_SCHEMA, CONDITION_SCHEMA,
        HISTORY_SCHEMA, TRAJECTORY_SCHEMA, HISTORY_TIMES)
    model = TemporalDiT(dimension=16, depth=1, heads=2, history_depth=1)
    normalization = dict(control_mean=[0.]*3, control_std=[1.]*3,
                         environment_mean=[0.]*6, environment_std=[1.]*6,
                         condition_mean=[0.]*12, condition_std=[1.]*12, fit_split='train')
    model.set_normalization(normalization)
    checkpoint = tmp_path/'untrained_protocol_fixture.pt'
    torch.save(dict(schema=CHECKPOINT_SCHEMA, model_type='temporal_dit_bspline',
        condition_schema=CONDITION_SCHEMA, history_schema=HISTORY_SCHEMA,
        trajectory_schema=TRAJECTORY_SCHEMA, prediction_type='velocity', diffusion_steps=100,
        future_times=FUTURE_TIMES.tolist(), history_times=HISTORY_TIMES.tolist(),
        normalization=normalization, architecture=model.architecture, history_mode='temporal',
        model_state=model.state_dict(), training_state='untrained_initial', training_step=0,
        source_dataset={'fixture': True}), checkpoint)
    encoded = dict(history_tokens=np.zeros((4,32,10), np.float32), history_mask=np.zeros((4,32), bool),
                   history_times=HISTORY_TIMES, condition=np.array([1,0,0,0,0,0,1,0,0,0,1,0], np.float32),
                   origin_world=np.zeros(3), world_to_canonical=np.eye(3))
    with PlannerProcess(tmp_path, python=sys.executable, device='cpu', temporal_checkpoint=checkpoint) as client:
        result = client.sample_temporal(encoded, 9, steps=2)
        assert client.describe()['runtime_kind'] == 'temporal'
        assert result['metadata']['checkpoint_sha256'] == client.describe()['checkpoint_sha256']
        assert result['metadata']['training_state'] == 'untrained_initial'
        decoded = FixedTimeTrajectory.from_prediction(result, encoded, [1,0,0,0], max_speed=100., max_acceleration=100.)
        np.testing.assert_allclose(decoded.sample(FUTURE_TIMES)['pos'], result['positions'], atol=2e-5)
    assert client._process.poll() == 0
