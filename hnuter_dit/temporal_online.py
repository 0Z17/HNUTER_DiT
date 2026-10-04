"""Causal temporal-model references through an isolated inference worker.

The pilot was trained on stationary starts. It finishes an active reference or
brakes to a measured hold before requesting another one. Scene geometry and
future obstacle schedules are never available to this planner.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time

import numpy as np

from .observation_guard import ObservationGuard
from .online import BrakeReference
from .planner_process import PlannerProcess
from .temporal_observations import HISTORY_TIMES, encode_history
from .temporal_trajectory import FixedTimeTrajectory


class HistoryBuffer:
    """Own immutable copies of 10 Hz measurements, selecting exact 0.2 s gaps."""

    def __init__(self):
        self.samples = deque(maxlen=8)

    def append(self, frame, state, sim_time):
        if (not np.isfinite(sim_time) or not np.isfinite(float(frame.sim_time))
                or abs(float(frame.sim_time)-sim_time) > 1e-6):
            raise ValueError("temporal observation timestamp differs from physics time")
        if self.samples and sim_time <= self.samples[-1][0].sim_time:
            raise ValueError("temporal observations must have increasing simulation times")
        fields = dict(sim_time=float(sim_time))
        for name in ('points', 'world_T_sensor', 'point_timestamps', 'ray_directions', 'ranges'):
            if hasattr(frame, name):
                fields[name] = np.array(getattr(frame, name), copy=True)
                fields[name].flags.writeable = False
        fields['frame_index'] = getattr(frame, 'frame_index', None)
        snapshot = {key: np.array(value, copy=True) for key, value in state.items()}
        self.samples.append((SimpleNamespace(**fields), snapshot))

    def select(self, query_time):
        selected = []
        for age in HISTORY_TIMES:
            candidates = [item for item in self.samples if abs(item[0].sim_time-query_time-age) < 1e-5]
            if len(candidates) != 1:
                raise ValueError("four causal scans at [-.6,-.4,-.2,0] are not available")
            selected.append(candidates[0])
        return selected


class TemporalOnlinePlanner:
    def __init__(self, *, goal_position, checkpoint, device='cuda', seed=7, output_dir=None,
                 max_speed=.3, max_acceleration=.4):
        self.goal = np.asarray(goal_position, dtype=float)
        if self.goal.shape != (3,) or not np.isfinite(self.goal).all():
            raise ValueError("temporal goal must be finite xyz")
        if not np.isfinite([max_speed, max_acceleration]).all() or min(max_speed, max_acceleration) <= 0:
            raise ValueError("temporal motion limits must be finite and positive")
        self.mode, self.seed = 'temporal', int(seed)
        self.max_speed, self.max_acceleration = float(max_speed), float(max_acceleration)
        self.runtime = PlannerProcess(output_dir=output_dir, device=device, temporal_checkpoint=checkpoint)
        self.input_representation = self.runtime.describe().get('input_representation', 'box')
        if self.input_representation not in ('box', 'point', 'voxel'):
            self.runtime.close()
            raise ValueError('unsupported temporal checkpoint input representation')
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='temporal-inference')
        self.history = HistoryBuffer()
        self.future = self.path = self.brake = self.hold = self.guard = None
        self.observation = None
        self.path_start = self.brake_start = 0.
        self.last_request = -np.inf
        self.request_wall_time = None
        self.request_expired = False
        self.closed = self.failed = False
        self.failure_reason = None
        self.events, self.guard_events, self.raw_paths = [], [], []
        self.planning_seconds = []
        self.waiting_reason = 'history_warmup'
        self.prediction_wait = False
        self.time_in_state = {}
        self.last_command_time = None
        self.last_command_category = 'history_warmup'
        self.start_speed_limit = .03

    def observe(self, frame, state, sim_time):
        self.history.append(frame, state, sim_time)
        self.observation = self.history.samples[-1][0]
        self.guard = ObservationGuard(self.observation)
        if self.hold is None:
            self.hold = dict(pos=state['position'].copy(), quaternion=state['quaternion'].copy())
        if self.path is not None:
            verdict = self._guard_path(self.path, self.path_start, sim_time)
            if not verdict['allowed']:
                self._brake(state, sim_time, verdict)
        # Do not restart an ongoing prediction every scan: the learned time
        # axis includes waiting, and repeated restarts would erase its progress.
        if self.path is not None or self.brake is not None or self.future is not None or self.failed:
            return
        if sim_time-self.last_request < 1.:
            return
        try:
            history = self.history.select(sim_time)
        except ValueError:
            self.waiting_reason = 'history_warmup'
            return
        speeds = [np.linalg.norm(item[1]['velocity']) for item in history]
        displacements = [np.linalg.norm(item[1]['position']-state['position']) for item in history]
        if max(speeds) > self.start_speed_limit or max(displacements) > .03:
            self.waiting_reason = 'moving_start_unsupported'
            return
        if np.linalg.norm(state['position']-self.goal) < .20:
            self.waiting_reason = 'goal_neighborhood_hold'
            return
        snapshot = {key: np.array(value, copy=True) for key, value in state.items()}
        self.request_wall_time = time.perf_counter()
        self.request_expired = False
        self.last_request = float(sim_time)
        self.future = self.pool.submit(self._plan, [item[0] for item in history], snapshot, float(sim_time))
        self.waiting_reason = 'inference_pending'

    def _plan(self, frames, state, sim_time):
        began = time.perf_counter()
        if self.input_representation == 'box':
            encoding = encode_history(frames, state, self.goal, query_time=sim_time)
        else:
            from .geometry_observations import encode_geometry_history
            encoding = encode_geometry_history(frames, state, self.goal,
                representation=self.input_representation, query_time=sim_time)
        prediction = self.runtime.sample_temporal(encoding, self.seed)
        result = dict(scan_time=sim_time, model=prediction['metadata'], history=encoding['metadata'],
                      condition=encoding['condition'].tolist(), planning_seconds=time.perf_counter()-began,
                      status='candidate', future_times=np.asarray(prediction['future_times']).tolist(),
                      canonical_control_points=np.asarray(prediction['control_points']).tolist())
        try:
            path = FixedTimeTrajectory.from_prediction(prediction, encoding, state['quaternion'],
                       max_speed=self.max_speed, max_acceleration=self.max_acceleration)
        except ValueError as exc:
            result.update(status='trajectory_rejected', rejection_reason=str(exc))
            result['planning_seconds'] = time.perf_counter()-began
            return result, None
        result['derivative_maxima'] = path.observed_maxima
        result['world_control_points'] = path.control_points.tolist()
        result['planning_seconds'] = time.perf_counter()-began
        return result, path

    @staticmethod
    def _is_wait(path, relative_time):
        times = np.linspace(relative_time, min(path.duration, relative_time+.5), 9)
        sample = path.sample(times)
        return bool(np.linalg.norm(sample['pos']-sample['pos'][0], axis=1).max() <= .002
                    and np.linalg.norm(sample['vel'], axis=1).max() <= .01)

    def _guard_path(self, path, reference_time, sim_time):
        if self.guard is None:
            return dict(allowed=False, reason='missing_scan')
        relative = sim_time-reference_time
        if relative < 0 or relative >= path.duration:
            return dict(allowed=False, reason='prediction_horizon_ended', kind='horizon')
        self.prediction_wait = self._is_wait(path, relative)
        if self.prediction_wait:
            sample = path.sample(np.linspace(relative, min(relative+.5, path.duration), 9))
            poses = np.c_[sample['pos'], sample['quaternion']]
        else:
            samples = path.sample(np.linspace(relative, path.duration, 161))
            distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(samples['pos'], axis=0), axis=1))]
            stop = min(len(distance), int(np.searchsorted(distance, .8))+1)
            poses = np.c_[samples['pos'][:stop], samples['quaternion'][:stop]]
            from .path_validation import dense_poses
            poses = dense_poses(poses, spacing=.04)
        verdict = self.guard.check(poses, sim_time=sim_time)
        if self.prediction_wait and verdict['reason'] == 'unknown_centerline' and verdict.get('minimum_clearance_m', 0) >= .20:
            # A model-predicted stationary hold needs no claim that an unseen
            # route is free. This exception never authorizes moving references.
            verdict = dict(verdict, allowed=True, reason='model_stationary_prediction',
                           certification='stationary model prediction only; no free-space claim')
        return verdict

    def _brake(self, state, sim_time, reason):
        self.path = None
        self.prediction_wait = False
        self.brake = BrakeReference(state)
        self.brake_start = float(sim_time)
        self.guard_events.append(dict(time=float(sim_time), kind=reason.get('kind', 'guard'), **{k:v for k,v in reason.items() if k != 'kind'}))
        self.waiting_reason = reason['reason']

    def _activate(self, result, candidate, state, sim_time):
        reference_time = float(result['scan_time'])
        age = float(sim_time-reference_time)
        result.update(activation_time=float(sim_time), observation_age_sim_seconds=age)
        self.planning_seconds.append(result['planning_seconds'])
        if candidate is not None:
            if self.request_expired or age < 0 or age > .5 or result['planning_seconds'] > 2.:
                result['status'] = 'stale_plan_rejected'
            elif np.linalg.norm(state['velocity']) > self.start_speed_limit:
                result['status'] = 'moving_start_rejected'
            else:
                predicted = candidate.sample(age)
                position_gap = float(np.linalg.norm(predicted['pos']-state['position']))
                velocity_gap = float(np.linalg.norm(predicted['vel']-state['velocity']))
                q = np.asarray(state['quaternion']) / np.linalg.norm(state['quaternion'])
                attitude_gap = float(2*np.arccos(np.clip(abs(q@predicted['quaternion']), 0, 1)))
                result.update(activation_position_gap_m=position_gap, activation_velocity_gap_mps=velocity_gap,
                              activation_attitude_gap_rad=attitude_gap)
                if position_gap > .05 or velocity_gap > .08 or attitude_gap > .1:
                    result['status'] = 'state_join_rejected'
                else:
                    verdict = self._guard_path(candidate, reference_time, sim_time)
                    result['execution_guard'] = verdict
                    if verdict['allowed']:
                        self.path, self.path_start, self.brake = candidate, reference_time, None
                        self.raw_paths.append(candidate.sample(np.arange(65)*.125)['pos'].tolist())
                        result.update(status='accepted', trajectory_duration=candidate.duration,
                                      consumed_prediction_seconds=age, reference_sim_time=reference_time)
                    else:
                        result['status'] = 'guard_rejected'
        self.events.append(result)
        if result['status'] != 'accepted':
            self.waiting_reason = result['status']

    def command(self, state, sim_time):
        if self.hold is None:
            self.hold = dict(pos=state['position'].copy(), quaternion=state['quaternion'].copy())
        if self.path is not None:
            if self.observation is None or sim_time-self.observation.sim_time > .3:
                self._brake(state, sim_time, dict(allowed=False, reason='stale_scan'))
            elif sim_time-self.path_start >= self.path.duration:
                self._brake(state, sim_time, dict(allowed=False, reason='prediction_horizon_ended', kind='horizon'))
        if (self.future is not None and not self.future.done() and not self.request_expired
                and time.perf_counter()-self.request_wall_time > 2.):
            self.request_expired = True
            self._brake(state, sim_time, dict(allowed=False, reason='planning_deadline_exceeded'))
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            try:
                result, candidate = future.result()
                self._activate(result, candidate, state, sim_time)
            except Exception as exc:
                self.failed = True
                self.failure_reason = f'{type(exc).__name__}: {exc}'
                self.events.append(dict(status='planner_failed', activation_time=float(sim_time), error=self.failure_reason))
                self._brake(state, sim_time, dict(allowed=False, reason='planner_failed', error=self.failure_reason))
        if self.path is not None:
            desired = self.path.sample(sim_time-self.path_start)
            category = 'model_wait' if self.prediction_wait else 'model_reference'
        elif self.brake is not None:
            desired = self.brake.sample(sim_time-self.brake_start)
            category = 'braking'
            if sim_time-self.brake_start >= self.brake.duration:
                self.hold = dict(pos=desired['pos'].copy(), quaternion=desired['quaternion'].copy())
                self.brake = None
        else:
            desired = dict(self.hold, vel=np.zeros(3), acc=np.zeros(3))
            category = self.waiting_reason
        if self.last_command_time is not None:
            delta = sim_time-self.last_command_time
            if delta < 0:
                raise ValueError('temporal command time moved backwards')
            self.time_in_state[self.last_command_category] = self.time_in_state.get(self.last_command_category, 0.)+delta
        self.last_command_time, self.last_command_category = float(sim_time), category
        return desired

    def close(self):
        if not self.closed:
            try:
                self.pool.shutdown(wait=True, cancel_futures=True)
                if self.future is not None and not self.future.cancelled():
                    try:
                        result, _ = self.future.result()
                        result['status'] = 'not_executed_after_run'
                        self.events.append(result)
                        self.planning_seconds.append(result['planning_seconds'])
                    except Exception as exc:
                        self.failed, self.failure_reason = True, f'{type(exc).__name__}: {exc}'
                        self.events.append(dict(status='planner_failed', error=self.failure_reason))
                    self.future = None
            finally:
                try:
                    self.runtime.close()
                finally:
                    self.closed = True
        values = np.asarray(self.planning_seconds)
        return dict(mode='temporal', input_representation=self.input_representation,
                    failed=self.failed, failure_reason=self.failure_reason,
                    checkpoint=self.runtime.describe(), plans=self.events, guard_events=self.guard_events,
                    accepted_plans=sum(item['status']=='accepted' for item in self.events),
                    guard_rejections=sum(item['status']=='guard_rejected' for item in self.events),
                    guard_interventions=sum(item['kind']=='guard' for item in self.guard_events),
                    horizon_completions=sum(item['kind']=='horizon' for item in self.guard_events),
                    time_in_state_seconds=self.time_in_state,
                    planning_seconds_mean=float(values.mean()) if len(values) else None,
                    planning_seconds_max=float(values.max()) if len(values) else None,
                    geometric_paths=self.raw_paths, temporal_checkpoint_loaded=True,
                    temporal_checkpoint_used=any('model' in item for item in self.events),
                    model_inference_calls=sum('model' in item for item in self.events),
                    executed_model_reference=bool(self.raw_paths),
                    prediction_contract='position-only cubic B-spline; fixed 8 s at scan time; measured attitude held',
                    start_velocity_support=dict(training='stationary anchors only', maximum_runtime_start_speed_mps=self.start_speed_limit),
                    motion_limits=dict(max_speed_mps=self.max_speed, max_acceleration_mps2=self.max_acceleration),
                    input_scope='Four past measured LiDAR frames and ego state only; no future obstacle truth',
                    safety_scope='Current observed-surface/centerline guard; stationary predictions do not certify unobserved free space')
