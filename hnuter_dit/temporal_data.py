"""Small, causal MuJoCo ray-history/physically executed temporal teacher pilot.

This is a stationary-start, constant-velocity crossing, wait/go curriculum.
The teacher may inspect future geometry; student conditioning never does.
Only episodes passing native 1 kHz actuator-driven flight are accepted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from .temporal_observations import (FUTURE_TIMES, HISTORY_TIMES, TEMPORAL_SCHEMA,
                                    encode_history, quaternion_matrix)

DEFAULT_SIM_ROOT = Path('/home/z017/research/isaacsim-industrial-reactor')
DT = .001
ANCHOR_TIME = .6
STUDENT_KEYS = ('history_tokens', 'history_mask', 'history_times', 'condition')
TRAINING_KEYS = STUDENT_KEYS + ('positions', 'future_times', 'episode_id', 'group_id', 'split', 'sample_id')
_MODULES = {}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _snapshot_sources(output, sources):
    """Preserve the exact text used for collection before any worker starts."""
    directory = Path(output)/'source_snapshot'
    directory.mkdir()
    hashes, snapshots = {}, {}
    for source in sources:
        source = Path(source).resolve()
        destination = directory/source.name
        if destination.exists():
            raise ValueError('source snapshot basenames must be unique')
        content = source.read_bytes()
        destination.write_bytes(content)
        hashes[str(source)] = hashlib.sha256(content).hexdigest()
        snapshots[str(source)] = str(destination.relative_to(output))
    return hashes, snapshots


def _module(path, name):
    path = Path(path).resolve()
    if str(path) not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        _MODULES[str(path)] = value
    return _MODULES[str(path)]


def scenario_for_group(group_id, seed=73):
    """Draw one group before its approaching/receding velocity is selected."""
    random = np.random.default_rng(np.random.SeedSequence([seed, group_id]))
    axis = group_id % 2
    origin = np.array([0., 0., random.uniform(2.0, 2.4)])
    goal = origin.copy()
    goal[axis] += random.uniform(1.8, 2.1)
    box = origin.copy()
    # The broad propeller span along body Y needs a farther crossing plane;
    # otherwise the actor can collide with the stationary start during a wait.
    box[axis] += random.uniform(.9, 1.05) if axis == 0 else random.uniform(1.25, 1.4)
    box[1-axis] -= random.uniform(.95, 1.1)
    return dict(group_id=int(group_id), origin=origin.tolist(), goal=goal.tolist(),
                route_axis=axis, motion_axis=1-axis, obstacle_at_anchor=box.tolist(),
                speed=float(random.uniform(.22, .28)), full_extents=[.7, .7, 2.5],
                source='native_mujoco_constant_velocity_prismatic',
                teacher_family='direct_quintic_with_wait_only',
                static_boxes=[[5.5, 0., 2.5, .2, 24., 5.],
                              [0., 5.5, 2.5, 24., .2, 5.],
                              [0., 0., -.05, 24., 24., .1]])


def _xml(scenario, sim_root):
    source = Path(sim_root)/'robots/hnuter/source/hnuter.xml'
    root = ET.parse(source).getroot()
    root.find('compiler').set('meshdir', str(source.parent/'meshes'))
    world = root.find('worldbody')
    for index, box in enumerate(scenario['static_boxes']):
        ET.SubElement(world, 'geom', name=f'temporal_static_{index}', type='box',
                      pos=' '.join(map(str, box[:3])), size=' '.join(str(x/2) for x in box[3:]),
                      contype='1', conaffinity='1', group='0')
    velocity = scenario['velocity']
    initial = np.array(scenario['obstacle_at_anchor'], dtype=float)
    initial[scenario['motion_axis']] -= velocity*ANCHOR_TIME
    body = ET.SubElement(world, 'body', name='temporal_obstacle', pos=' '.join(map(str, initial)))
    axis = np.eye(3)[scenario['motion_axis']]
    ET.SubElement(body, 'joint', name='temporal_slide', type='slide', axis=' '.join(map(str, axis)),
                  limited='false', damping='0', armature='0')
    sx, sy, sz = scenario['full_extents']
    inertia = 10*np.array([sy*sy+sz*sz, sx*sx+sz*sz, sx*sx+sy*sy])/12
    ET.SubElement(body, 'inertial', pos='0 0 0', mass='10', diaginertia=' '.join(map(str, inertia)))
    ET.SubElement(body, 'geom', name='temporal_obstacle_geom', type='box',
                  size=' '.join(str(x/2) for x in scenario['full_extents']),
                  contype='1', conaffinity='1', group='0')
    return ET.tostring(root, encoding='unicode')


class NativeWorld:
    """One original HNUTER + evaluator obstacle; only actuators advance motion."""
    def __init__(self, scenario, sim_root=DEFAULT_SIM_ROOT):
        import mujoco
        self.mj, self.scenario, self.sim_root = mujoco, scenario, Path(sim_root)
        source = self.sim_root/'tests/reference/hnuter_control.py'
        controller = _module(source, '_temporal_native_hnuter').HnuterController
        self._directory = tempfile.TemporaryDirectory(prefix='hnuter-temporal-model-')
        path = Path(self._directory.name)/'world.xml'
        path.write_text(_xml(scenario, self.sim_root))
        self.controller = controller(path)
        self.model, self.data = self.controller.model, self.controller.data
        if not np.isclose(self.model.opt.timestep, DT):
            raise ValueError('native model must integrate at 1 ms')
        self.controller.set_freejoint_pose(scenario['origin'])
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'temporal_slide')
        self.obstacle_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'temporal_obstacle')
        self.robot_geoms = self.model.body_rootid[self.model.geom_bodyid] == self.controller._body_id
        self.model.geom_group[self.robot_geoms] = 5  # Sensor filter only; contacts unchanged.
        self.data.qvel[self.model.jnt_dofadr[joint]] = scenario['velocity']
        self.data.qpos[self.model.jnt_qposadr[joint]] = 0.
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        self._directory.cleanup()

    def state(self):
        return self.controller.get_state()

    def snapshot(self):
        return dict(teacher_native_qpos=self.data.qpos.copy(), teacher_native_qvel=self.data.qvel.copy(),
                    teacher_native_act=self.data.act.copy(), teacher_native_time=np.float64(self.data.time))

    def restore(self, sample):
        self.data.qpos[:] = sample['teacher_native_qpos']
        self.data.qvel[:] = sample['teacher_native_qvel']
        self.data.act[:] = sample['teacher_native_act']
        self.data.time = float(sample['teacher_native_time'])
        self.data.qacc_warmstart[:] = 0
        self.mj.mj_forward(self.model, self.data)

    def step(self, desired):
        self.controller.set_desired_state(desired)
        self.controller.update_control()
        # The horizontal frictionless slider carries its initialized velocity.
        # Gravity has no component along this axis; contacts can change actual
        # velocity. No position write, prescribed future pose, or servo clock.
        self.mj.mj_step(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        collision = False
        for item in self.data.contact[:self.data.ncon]:
            a, b = int(item.geom1), int(item.geom2)
            if a >= 0 and b >= 0 and self.robot_geoms[a] != self.robot_geoms[b] and item.dist <= 0:
                collision = True
                break
        return collision

    def capture(self, index, rays):
        module = _module(self.sim_root/'perception/mid360.py', '_temporal_mid360')
        directions = module.scan_directions(index, module.Mid360Config(rays_per_frame=rays))
        pose = module.sensor_pose(self.state())
        world = np.ascontiguousarray(directions.astype(float) @ pose[:3, :3].T).ravel()
        ranges, geomids = np.empty(rays, dtype=np.float64), np.empty(rays, dtype=np.int32)
        self.mj.mj_multiRay(self.model, self.data, pose[:3, 3].copy(), world,
                            np.array([1, 1, 1, 1, 1, 0], dtype=np.uint8), True, -1,
                            geomids, ranges, None, rays, 40.)
        valid = np.isfinite(ranges) & (ranges >= .1) & (ranges <= 40.)
        ranges[~valid] = np.nan
        points = (directions[valid]*ranges[valid, None]).astype(np.float32)
        return module.LidarFrame(points, pose, float(self.data.time), rays, ranges.astype(np.float32),
                                 directions, np.full(len(points), self.data.time), index)

    def boxes_world(self):
        boxes = [[*self.data.xpos[self.obstacle_body], *self.scenario['full_extents'], 1., 0., 0., 0.]]
        boxes += [[*item, 1., 0., 0., 0.] for item in self.scenario['static_boxes']]
        return np.asarray(boxes, dtype=float)


class WaitGoTeacher:
    def __init__(self, origin, goal, wait):
        self.origin, self.delta, self.wait = np.asarray(origin), np.asarray(goal)-origin, float(wait)
        distance = np.linalg.norm(self.delta)
        self.travel = max(1., 1.875*distance/.45, math.sqrt(5.773502692*distance/.5))
        self.duration = self.wait+self.travel

    def sample(self, t):
        u = float(np.clip((t-self.wait)/self.travel, 0, 1))
        s, ds, dds = u**3*(10-15*u+6*u*u), 30*u*u*(1-u)**2, 60*u*(2*u*u-3*u+1)
        return dict(pos=self.origin+self.delta*s, vel=self.delta*ds/self.travel,
                    acc=self.delta*dds/self.travel**2, quaternion=np.array([1., 0., 0., 0.]))


def _teacher_proxy_clearance(teacher, scenario):
    from .collision_loss import ROBOT_SPHERE_CENTERS, ROBOT_SPHERE_RADII
    times = np.arange(0, teacher.duration+.025, .025)
    positions = np.array([teacher.sample(t)['pos'] for t in times])
    centers = positions[:, None, :] + np.asarray(ROBOT_SPHERE_CENTERS)
    moving = np.tile(scenario['obstacle_at_anchor'], (len(times), 1)).astype(float)
    moving[:, scenario['motion_axis']] += scenario['velocity']*times
    delta = np.abs(centers-moving[:, None, :])-np.asarray(scenario['full_extents'])/2
    distance = np.linalg.norm(np.maximum(delta, 0), axis=-1)+np.minimum(delta.max(axis=-1), 0)-np.asarray(ROBOT_SPHERE_RADII)
    return float(distance.min())


def _run_reference(world, reference, duration, collect=False):
    start = float(world.data.time)
    end_step = int(round(duration/DT))
    collision, first_contact = False, None
    errors, velocities, actual_positions, obstacle_boxes = [], [], [], []
    sampled = []
    for step in range(end_step+1):
        t = step*DT
        state = world.state()
        desired = reference.sample(t)
        errors.append(float(np.linalg.norm(state['position']-desired['pos'])))
        velocities.append(float(np.linalg.norm(state['velocity'])))
        if collect and step % 125 == 0 and t <= FUTURE_TIMES[-1]+1e-8:
            actual_positions.append(state['position'].copy())
            obstacle_boxes.append(world.boxes_world())
            sampled.append(t)
        if not all(np.isfinite(state[k]).all() for k in ('position', 'quaternion', 'velocity')):
            raise ValueError('non-finite native state')
        if step < end_step and world.step(desired):
            collision = True
            first_contact = t+DT if first_contact is None else first_contact
    final = world.state()
    goal_distance = float(np.linalg.norm(final['position']-np.asarray(world.scenario['goal'])))
    report = dict(completed=True, steps=end_step, simulation_seconds=end_step*DT,
                  collision_detected=collision, first_contact_time=first_contact,
                  goal_reached=bool(goal_distance <= .2 and np.linalg.norm(final['velocity']) <= .10),
                  final_goal_distance_m=goal_distance, final_position=final['position'].tolist(),
                  final_speed_mps=float(np.linalg.norm(final['velocity'])),
                  position_tracking_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                  position_tracking_max_error_m=max(errors), max_speed_mps=max(velocities),
                  reference_time_origin=start, physical_backend=f'MuJoCo {world.mj.__version__}',
                  collision_truth='original HNUTER collision geoms vs environment contacts at every 1 ms; sensor group filter does not change contacts')
    return report, np.asarray(actual_positions), np.asarray(obstacle_boxes)


def _canonical_boxes(boxes, origin, rotation):
    result = boxes.copy()
    result[..., :3] = (result[..., :3]-origin) @ rotation.T
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    result[..., 6:] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    return result.astype(np.float32)


def collect_episode(scenario, sim_root=DEFAULT_SIM_ROOT, rays=20_000):
    world = NativeWorld(scenario, sim_root)
    try:
        frames = []
        hover = dict(pos=np.asarray(scenario['origin']), vel=np.zeros(3), acc=np.zeros(3),
                     quaternion=np.array([1., 0., 0., 0.]))
        for step in range(601):
            if step % 200 == 0:
                frames.append(world.capture(step//100, rays))
            if step < 600 and world.step(hover):
                return None, dict(accepted=False, reason='contact_during_history')
        current = world.state()
        if np.linalg.norm(current['velocity']) > .03:
            return None, dict(accepted=False, reason='not_stationary_start')
        encoding = encode_history(frames, current, scenario['goal'])
        snapshot = world.snapshot()
        attempts = []
        for wait in (0., 2., 4., 6., 8.):
            teacher = WaitGoTeacher(current['position'], np.asarray(scenario['goal']), wait)
            proxy = _teacher_proxy_clearance(teacher, scenario)
            if proxy < .18:
                attempts.append(dict(wait_s=wait, proxy_clearance_m=proxy, reason='proxy_rejected'))
                continue
            world.restore(snapshot)
            duration = max(8., teacher.duration+.75)
            report, actual_positions, boxes = _run_reference(world, teacher, duration, collect=True)
            accepted = (not report['collision_detected'] and report['goal_reached']
                        and report['position_tracking_max_error_m'] < .12 and report['max_speed_mps'] <= .5)
            attempts.append(dict(wait_s=wait, proxy_clearance_m=proxy, accepted=bool(accepted), **report))
            if not accepted:
                continue
            origin, rotation = encoding['origin_world'], encoding['world_to_canonical']
            # Learn the command that passed physical validation, not the
            # controller's resulting tracking error (which would be applied a
            # second time at deployment). Keep the full actual trajectory as
            # evaluator evidence, outside the model/training allowlists.
            reference_positions = np.stack([teacher.sample(t)['pos'] for t in FUTURE_TIMES])
            record = {key: encoding[key] for key in ('history_tokens', 'history_mask', 'condition', 'origin_world', 'world_to_canonical')}
            record.update(positions=((reference_positions-origin) @ rotation.T).astype(np.float32),
                          teacher_actual_positions=((actual_positions-origin) @ rotation.T).astype(np.float32),
                          gt_boxes=_canonical_boxes(boxes, origin, rotation),
                          gt_mask=np.ones(boxes.shape[:2], dtype=bool),
                          history_ray_ranges=np.stack([f.ranges for f in frames]),
                          history_sensor_poses=np.stack([f.world_T_sensor for f in frames]),
                          teacher_wait_s=np.float32(wait), teacher_duration_s=np.float32(duration),
                          teacher_scenario_json=json.dumps(scenario, sort_keys=True),
                          teacher_report_json=json.dumps(report, sort_keys=True),
                          teacher_latest_token_hash=_array_hash(encoding['history_tokens'][-1]),
                          teacher_latest_ray_hash=_array_hash(frames[-1].ranges),
                          **snapshot)
            return record, dict(accepted=True, wait_s=wait, attempts=attempts,
                                latest_token_hash=record['teacher_latest_token_hash'],
                                latest_ray_hash=record['teacher_latest_ray_hash'])
        return None, dict(accepted=False, reason='no_valid_physical_teacher', attempts=attempts)
    finally:
        world.close()


def sample_from_dataset(dataset, index):
    count = len(dataset['positions'])
    if not 0 <= index < count:
        raise IndexError(index)
    constants = {'history_times', 'future_times', 'ray_directions', '_freeze'}
    sample = {key: (value if key in constants else value[index]) for key, value in dataset.items()}
    sample['teacher_scenario'] = json.loads(str(sample['teacher_scenario_json']))
    return sample


def native_rollout(sample, positions, times=FUTURE_TIMES, *, control_points=None,
                   sim_root=DEFAULT_SIM_ROOT, output_dir=None):
    """Execute one supplied timed candidate; no oracle selection or retiming.

    The candidate is canonical xyz at the dataset's anchor. The robot and
    obstacle restart from that actual native snapshot, not the earlier scan
    clock. This evaluator alone consumes teacher scenario/native state fields.
    """
    from scipy.interpolate import BSpline, CubicSpline
    positions, times = np.asarray(positions, dtype=float), np.asarray(times, dtype=float)
    if positions.shape != (65, 3) or times.shape != (65,) or not np.isfinite(positions).all() or not np.allclose(times, FUTURE_TIMES):
        raise ValueError('expected finite positions[65,3] on fixed .125 s grid')
    scenario = sample.get('teacher_scenario') or json.loads(str(sample['teacher_scenario_json']))
    world = NativeWorld(scenario, sim_root)
    try:
        world.restore(sample)
        anchor_quaternion = world.state()['quaternion'].copy()
        rotation, origin = np.asarray(sample['world_to_canonical']), np.asarray(sample['origin_world'])
        points = positions @ rotation + origin
        if np.linalg.norm(points[0]-world.state()['position']) > .02:
            raise ValueError('candidate must start at current measured position')
        if control_points is not None:
            controls = np.asarray(control_points, dtype=float)
            if controls.shape != (16, 3) or not np.isfinite(controls).all():
                raise ValueError('control_points must be finite [16,3] canonical coefficients')
            knots = np.r_[np.zeros(4), np.arange(1, 13)*8/13, np.full(4, 8.)]
            canonical_spline = BSpline(knots, controls, 3)
            if not np.allclose(canonical_spline(times), positions, atol=2e-5, rtol=0):
                raise ValueError('supplied positions disagree with temporal B-spline coefficients')
            spline = BSpline(knots, controls @ rotation + origin, 3)
            representation = 'original_cubic_BSpline_16_controls_analytic_derivatives'
        else:
            spline = CubicSpline(times, points, bc_type=((1, world.state()['velocity']), (2, np.zeros(3))))
            representation = 'CubicSpline_interpolated_positions_no_retiming'
        class Reference:
            def sample(self, t):
                return dict(pos=spline(t), vel=spline(t, 1), acc=spline(t, 2), quaternion=anchor_quaternion)
        report, actual, boxes = _run_reference(world, Reference(), float(times[-1]), collect=True)
        report.update(sample_id=str(sample.get('sample_id', 'unknown')), no_retiming=True,
                      oracle_selection=False, reference_representation=representation)
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            np.savez_compressed(output_dir/'native_trace.npz', times=times, positions_world=actual,
                                obstacle_boxes_world=boxes, requested_positions_canonical=positions)
        return report
    finally:
        world.close()


def _split_groups(groups, seed):
    if groups < 4:
        raise ValueError('at least four groups required for disjoint train/validation/test')
    holdout = max(1, groups//6)
    result = np.zeros(groups, dtype=np.int8)
    order = np.random.default_rng(seed+919).permutation(groups)
    result[order[-2*holdout:-holdout]], result[order[-holdout:]] = 1, 2
    return result


def _collect_group(task):
    group, split, seed, rays, sim_root = task
    base = scenario_for_group(group, seed)
    pair, audit = [], []
    for variant in (0, 1):
        scenario = dict(base, velocity=base['speed']*(1 if variant == 0 else -1), pair_variant=variant)
        began = time.perf_counter()
        record, validation = collect_episode(scenario, sim_root, rays)
        validation.update(group_id=group, pair_variant=variant, wall_seconds=time.perf_counter()-began)
        audit.append(validation)
        if record is not None:
            record.update(group_id=np.int32(group), episode_id=np.int32(2*group+variant),
                          pair_variant=np.int8(variant), split=np.int8(split),
                          sample_id=f'g{group:04d}_v{variant}_t0600')
        pair.append(record)
        print(json.dumps(dict(event='episode', group=group, variant=variant,
                              accepted=record is not None, wait_s=validation.get('wait_s'),
                              wall_seconds=validation['wall_seconds'])), flush=True)
    if not all(item is not None for item in pair):
        return [], audit, dict(group_id=group, quarantined='incomplete_physical_pair')
    latest_equal = np.array_equal(pair[0]['history_tokens'][-1], pair[1]['history_tokens'][-1])
    condition_close = np.allclose(pair[0]['condition'], pair[1]['condition'], atol=1e-6, rtol=0)
    row = dict(group_id=group, latest_tokens_equal=bool(latest_equal),
               latest_token_hashes=[str(item['teacher_latest_token_hash']) for item in pair],
               conditions_equal_within_1e_6=bool(condition_close),
               teacher_wait_s=[float(item['teacher_wait_s']) for item in pair],
               dynamic_decision_differs=abs(float(pair[0]['teacher_wait_s']-pair[1]['teacher_wait_s'])) >= 1.)
    if not latest_equal or not condition_close:
        row['quarantined'] = 'counterfactual_latest_or_condition_mismatch'
        pair = []
    return pair, audit, row


def collect_dataset(output, groups=96, seed=73, rays=20_000, sim_root=DEFAULT_SIM_ROOT, workers=1):
    output, sim_root = Path(output), Path(sim_root)
    if (output/'dataset.npz').exists() or (output/'freeze.json').exists():
        raise FileExistsError('refusing to replace an existing temporal dataset')
    output.mkdir(parents=True, exist_ok=True)
    if not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError('workers must be in [1,4]')
    split = _split_groups(groups, seed)
    records, audit, pair_audit = [], [], []
    started = time.perf_counter()
    source_hashes, source_snapshots = _snapshot_sources(output, [
        Path(__file__), Path(__file__).with_name('temporal_observations.py'),
        sim_root/'tests/reference/hnuter_control.py', sim_root/'robots/hnuter/source/hnuter.xml'])
    tasks = [(group, int(split[group]), seed, rays, str(sim_root)) for group in range(groups)]
    context = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) if workers > 1 else nullcontext()
    with context as pool:
        outcomes = pool.map(_collect_group, tasks) if pool is not None else map(_collect_group, tasks)
        # map consumes in group order, independent of worker completion order.
        for pair, episode_audit, row in outcomes:
            records.extend(pair)
            audit.extend(episode_audit)
            pair_audit.append(row)
            (output/'collection_audit.json').write_text(json.dumps(dict(episodes=audit, pairs=pair_audit), indent=2)+'\n')
    if not records:
        raise RuntimeError('No paired physically valid episodes; inspect collection_audit.json')
    if any(_hash(path) != digest for path, digest in source_hashes.items()):
        raise RuntimeError('collection source changed during execution; audit retained but dataset not frozen')
    arrays = {key: np.asarray([record[key] for record in records]) for key in records[0]}
    module = _module(sim_root/'perception/mid360.py', '_temporal_mid360')
    arrays.update(history_times=HISTORY_TIMES, future_times=FUTURE_TIMES,
                  ray_directions=np.stack([module.scan_directions(index, module.Mid360Config(rays_per_frame=rays)) for index in (0, 2, 4, 6)]))
    np.savez_compressed(output/'dataset.npz', **arrays)
    accepted_groups = np.unique(arrays['group_id'])
    freeze = dict(schema=TEMPORAL_SCHEMA, status='frozen', dataset_file='dataset.npz',
                  dataset_sha256=_hash(output/'dataset.npz'), samples=len(records),
                  independent_groups=len(accepted_groups), episodes=len(np.unique(arrays['episode_id'])),
                  requested_groups=groups, seed=seed, windows_per_episode=1, workers=workers,
                  split_codes={'train': 0, 'validation': 1, 'test': 2},
                  split_samples={name: int((arrays['split'] == code).sum()) for name, code in [('train', 0), ('validation', 1), ('test', 2)]},
                  student_allowlist=list(STUDENT_KEYS), training_allowlist=list(TRAINING_KEYS),
                  teacher_future_fields='gt_* and teacher_*: evaluator/label audit only, forbidden model inputs',
                  target_semantics='physically_validated_teacher_reference',
                  data_time_semantics='fixed physical future times from latest history anchor; positions are the reference commands of the accepted native teacher rollout; teacher_actual_positions stores executed physical positions separately; no arc-length resampling, no terminal goal clamp',
                  stationary_start_only=True, maximum_start_speed_mps=.03,
                  limits={'reference_speed_mps': .5, 'reference_acceleration_mps2': .6},
                  sensor=dict(rays_per_frame=rays, history_rate_hz=5., observation_backend='MuJoCo mj_multiRay against actual geoms',
                              direction_pattern='MID360 envelope approximation, not Livox timing or optics', no_return='NaN/unknown'),
                  teacher_family='constant_velocity_crossing_wait_go',
                  quality_gates=dict(no_native_contact=True, native_dt_s=DT, final_goal_m=.2,
                                     final_speed_mps=.10, tracking_max_m=.12, actual_max_speed_mps=.5),
                  source_sha256=source_hashes, source_snapshots=source_snapshots,
                  wall_seconds=time.perf_counter()-started,
                  accepted_pairs_with_different_wait=sum(row.get('dynamic_decision_differs', False) and not row.get('quarantined') for row in pair_audit),
                  failures_preserved_in='collection_audit.json')
    (output/'freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
    load_dataset(output)
    return freeze


def load_dataset(root, verify=True):
    root = Path(root)
    freeze = json.loads((root/'freeze.json').read_text())
    if freeze.get('schema') != TEMPORAL_SCHEMA or freeze.get('status') != 'frozen':
        raise ValueError('unsupported or unfrozen temporal dataset')
    path = root/freeze['dataset_file']
    if verify and _hash(path) != freeze['dataset_sha256']:
        raise ValueError('temporal dataset hash mismatch')
    if verify and 'source_snapshots' in freeze:
        if set(freeze['source_snapshots']) != set(freeze['source_sha256']):
            raise ValueError('source snapshot provenance keys disagree')
        for original, snapshot in freeze['source_snapshots'].items():
            if _hash(root/snapshot) != freeze['source_sha256'][original]:
                raise ValueError('source snapshot hash mismatch: '+snapshot)
    with np.load(path, allow_pickle=False) as value:
        data = {key: value[key].copy() for key in value.files}
    n = len(data['positions'])
    expected = {'history_tokens': (n, 4, 32, 10), 'history_mask': (n, 4, 32),
                'condition': (n, 12), 'positions': (n, 65, 3), 'split': (n,),
                'episode_id': (n,), 'group_id': (n,), 'pair_variant': (n,)}
    for key, shape in expected.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError('invalid temporal array: '+key)
    if freeze.get('target_semantics') == 'physically_validated_teacher_reference':
        actual = data.get('teacher_actual_positions')
        if actual is None or actual.shape != (n, 65, 3) or not np.isfinite(actual).all():
            raise ValueError('reference targets require actual physical trajectory evidence')
    if n != freeze['samples'] or not np.allclose(data['history_times'], HISTORY_TIMES) or not np.allclose(data['future_times'], FUTURE_TIMES):
        raise ValueError('temporal counts/time grids disagree')
    if not np.allclose(data['positions'][:, 0], 0, atol=1e-5) or not set(data['split'].tolist()) <= {0, 1, 2}:
        raise ValueError('invalid initial position or split')
    for group in np.unique(data['group_id']):
        members = data['group_id'] == group
        if len(np.unique(data['split'][members])) != 1 or set(data['pair_variant'][members]) != {0, 1}:
            raise ValueError('group split leakage or incomplete counterfactual pair')
    if data['gt_boxes'].shape != (n, 65, 4, 10) or data['gt_mask'].shape != (n, 65, 4):
        raise ValueError('invalid evaluator future geometry')
    for key in ('split', 'group_id', 'episode_id', 'pair_variant'):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError('integer identity/split array required: '+key)
    for tokens, mask in ((data['history_tokens'], data['history_mask']), (data['gt_boxes'], data['gt_mask'])):
        if mask.dtype != np.bool_ or not np.isfinite(tokens).all():
            raise ValueError('boolean masks and finite box tokens required')
        active = tokens[mask]
        if (active[:, 3:6] <= 0).any() or not np.allclose(np.linalg.norm(active[:, 6:], axis=1), 1., atol=1e-5):
            raise ValueError('active box extents/quaternions invalid')
    if data['origin_world'].shape != (n, 3) or data['world_to_canonical'].shape != (n, 3, 3):
        raise ValueError('invalid canonical transform shapes')
    matrices = data['world_to_canonical']
    if (not np.isfinite(data['origin_world']).all() or not np.isfinite(matrices).all()
            or not np.allclose(matrices.transpose(0, 2, 1) @ matrices, np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(matrices), 1, atol=1e-6)):
        raise ValueError('invalid canonical rigid rotations')
    if freeze['independent_groups'] != len(np.unique(data['group_id'])) or freeze['episodes'] != len(np.unique(data['episode_id'])):
        raise ValueError('freeze group/episode counts disagree')
    if len(np.unique(data['sample_id'])) != n:
        raise ValueError('duplicate sample IDs')
    for name, code in [('train', 0), ('validation', 1), ('test', 2)]:
        if freeze['split_samples'][name] != int((data['split'] == code).sum()):
            raise ValueError('freeze split counts disagree')
    for episode in np.unique(data['episode_id']):
        members = data['episode_id'] == episode
        if len(np.unique(data['group_id'][members])) != 1 or len(np.unique(data['split'][members])) != 1:
            raise ValueError('episode leaks across group/split')
    for group in np.unique(data['group_id']):
        members = np.flatnonzero(data['group_id'] == group)
        if len(members) != 2 or not np.array_equal(data['history_tokens'][members[0], -1], data['history_tokens'][members[1], -1]):
            raise ValueError('counterfactual latest token mismatch')
        if not np.array_equal(data['history_mask'][members[0], -1], data['history_mask'][members[1], -1]):
            raise ValueError('counterfactual latest mask mismatch')
        if not np.allclose(data['condition'][members[0]], data['condition'][members[1]], atol=1e-6, rtol=0):
            raise ValueError('counterfactual current condition mismatch')
    ranges = data['history_ray_ranges']
    rays = freeze['sensor']['rays_per_frame']
    if (ranges.shape != (n, 4, rays) or data['ray_directions'].shape != (4, rays, 3)
            or data['history_sensor_poses'].shape != (n, 4, 4, 4)):
        raise ValueError('invalid raw history ray evidence shape')
    if np.isinf(ranges).any() or (ranges[np.isfinite(ranges)] < .1).any() or (ranges[np.isfinite(ranges)] > 40).any():
        raise ValueError('invalid ray ranges; misses must be NaN')
    for group in np.unique(data['group_id']):
        members = np.flatnonzero(data['group_id'] == group)
        if not np.array_equal(ranges[members[0], -1], ranges[members[1], -1], equal_nan=True):
            raise ValueError('counterfactual latest raw ray mismatch')
    for name, values in [('teacher_latest_token_hash', data['history_tokens'][:, -1]),
                         ('teacher_latest_ray_hash', ranges[:, -1])]:
        if name in data and any(str(digest) != _array_hash(value) for digest, value in zip(data[name], values)):
            raise ValueError('stored latest evidence hash mismatch: '+name)
    if not np.isfinite(data['ray_directions']).all() or not np.allclose(np.linalg.norm(data['ray_directions'], axis=-1), 1., atol=1e-5):
        raise ValueError('invalid raw ray directions')
    poses = data['history_sensor_poses']
    if (not np.isfinite(poses).all() or not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(poses[..., :3, :3].swapaxes(-1, -2) @ poses[..., :3, :3], np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(poses[..., :3, :3]), 1., atol=1e-6)):
        raise ValueError('invalid raw history sensor rigid poses')
    data['_freeze'] = freeze
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=int, default=4)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--rays', type=int, default=20_000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--sim-root', type=Path, default=DEFAULT_SIM_ROOT)
    args = parser.parse_args(argv)
    if args.groups < 4 or args.rays < 100:
        parser.error('groups >= 4 and rays >= 100 required')
    print(json.dumps(collect_dataset(args.output, args.groups, args.seed, args.rays, args.sim_root, args.workers), indent=2))


if __name__ == '__main__':
    main()
