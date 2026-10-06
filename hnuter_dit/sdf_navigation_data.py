"""Continuous 3-D experts and causal moving-anchor nvblox observation windows.

Teacher geometry is privileged. Student inputs never contain boxes or a route
class. This schema is deliberately incompatible with stationary-start v1.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial import Polynomial as P
from scipy.spatial.transform import Rotation
from .sdf_data import StaticWorld
from .temporal_data import DEFAULT_SIM_ROOT
from .sdf_pose import angular_derivatives

SCHEMA = 'nvblox_continuous_navigation_v1'
KINDS = ('open', 'left', 'right', 'over', 'under', 'height_change')
DURATION = 30.
DT = .001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenario(index, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    kind = KINDS[index % len(KINDS)]
    origin = np.array([0., 0., 3.])
    target = np.array([rng.uniform(5.8, 6.2), 0., rng.uniform(-.2, .2), 0., 0., rng.uniform(-.12, .12)])
    if kind == 'height_change':
        target[2] = float(rng.choice([-1, 1]) * rng.uniform(.8, 1.2))
    boxes = [[3., 0., -.1, 18., 18., .2], [9., 0., 3., .2, 14., 6.]]
    if kind in ('left', 'right'):
        # A tall, asymmetric block makes the shorter horizontal passage switch.
        side = 1 if kind == 'right' else -1
        boxes.append([3., side*.65, 3., .65, 2.1, 8.])
    if kind == 'over':
        boxes.append([3., 0., 1.5, .65, 14., 3.])
    if kind == 'under':
        boxes.append([3., 0., 5., .65, 14., 4.])
    if len(boxes) > 2:
        boxes[-1][0] += float(rng.uniform(-.2, .2))
        boxes[-1][3] = float(rng.uniform(.5, .8))
    # Whole scenes, including all overlapping windows, share one split.
    bucket = int.from_bytes(hashlib.sha256(f'{seed}:{index}'.encode()).digest()[:4], 'little') % 10
    return dict(index=index, seed=seed, kind=kind, origin=origin.tolist(), target=target.tolist(),
                boxes=boxes, split=0 if bucket < 7 else (1 if bucket < 9 else 2))


class ContinuousReference:
    def __init__(self, origin, target, lateral=0., vertical=0., duration=DURATION):
        self.origin = np.asarray(origin)
        self.target = np.asarray(target)
        self.duration = duration
        progress = P([0, 0, 0, 10, -15, 6])
        bump = P([0, 0, 0, 64, -192, 192, -64])
        self.polys = [progress*x for x in self.target]
        self.polys[1] += lateral*bump
        self.polys[2] += vertical*bump

    def sample(self, stamp):
        scalar = np.ndim(stamp) == 0
        t = np.atleast_1d(stamp).astype(float)
        if not np.isfinite(t).all() or np.any(t < -1e-8) or np.any(t > self.duration+1e-8):
            raise ValueError('invalid trajectory time')
        u = np.clip(t/self.duration, 0, 1)
        x, v, a = [np.stack([p.deriv(k)(u)/self.duration**k for p in self.polys], -1) for k in range(3)]
        omega, alpha = angular_derivatives(x[:, 3:], v[:, 3:], a[:, 3:])
        result = dict(pos=x[:, :3]+self.origin, vel=v[:, :3], acc=a[:, :3],
                      quaternion=Rotation.from_euler('xyz', x[:, 3:]).as_quat()[:, [3, 0, 1, 2]],
                      angular_velocity=omega, angular_acceleration=alpha)
        return {k: value[0] if scalar else value for k, value in result.items()}

    def metrics(self, boxes):
        from .collision_loss import ROBOT_SPHERE_CENTERS, ROBOT_SPHERE_RADII
        t = np.linspace(0, self.duration, 1501)
        samples = self.sample(t)
        r = Rotation.from_quat(samples['quaternion'][:, [1, 2, 3, 0]]).as_matrix()
        centers = samples['pos'][:, None] + np.einsum('tij,sj->tsi', r, ROBOT_SPHERE_CENTERS)
        boxes = np.asarray(boxes)
        delta = abs(centers[:, :, None]-boxes[None, None, :, :3])-boxes[None, None, :, 3:]/2
        distance = np.linalg.norm(np.maximum(delta, 0), axis=-1)+np.minimum(delta.max(-1), 0)-np.asarray(ROBOT_SPHERE_RADII)[None, :, None]
        jerk = np.stack([p.deriv(3)(t/self.duration)/self.duration**3 for p in self.polys[:3]], -1)
        length = float(np.linalg.norm(np.diff(samples['pos'], axis=0), axis=1).sum())
        return dict(clearance_m=float(distance.min()), length_m=length,
                    speed_max_mps=float(np.linalg.norm(samples['vel'], axis=1).max()),
                    acceleration_max_mps2=float(np.linalg.norm(samples['acc'], axis=1).max()),
                    jerk_mean_square=float(np.mean(np.sum(jerk**2, axis=1))),
                    duration_s=self.duration)


def expert(config, origin):
    candidates = [(0., 0.)]
    for amplitude in (.8, 1.2, 1.6, 2., 2.4):
        candidates += [(amplitude, 0.), (-amplitude, 0.), (0., amplitude), (0., -amplitude)]
    best = None
    for lateral, vertical in candidates:
        path = ContinuousReference(origin, config['target'], lateral, vertical)
        metrics = path.metrics(config['boxes'])
        if metrics['clearance_m'] < .22 or metrics['speed_max_mps'] > .7 or metrics['acceleration_max_mps2'] > .9:
            continue
        score = metrics['length_m'] + 10*metrics['jerk_mean_square']
        if best is None or score < best[0]:
            best = (score, path, dict(metrics, lateral_m=lateral, vertical_m=vertical))
    if best is None:
        raise RuntimeError('no feasible expert in finite 3-D candidate family')
    return best[1:]


def collect_episode(index, seed, output, sim_root, rays):
    config = scenario(index, seed)
    world = StaticWorld(config, sim_root)
    report = dict(index=index, config=config, accepted=False)
    began = time.perf_counter()
    frames, windows, trace = [], [], []
    try:
        hold = dict(pos=np.asarray(config['origin']), quaternion=np.array([1., 0., 0., 0.]), vel=np.zeros(3), acc=np.zeros(3))
        for _ in range(600):
            if world.step(hold):
                raise RuntimeError('warmup contact')
        path, metrics = expert(config, world.state()['position'])
        report['expert'] = metrics
        errors, contacts = [], 0
        for step in range(round(DURATION/DT)+1):
            t = step*DT
            state = world.state()
            if not all(np.isfinite(state[k]).all() for k in ('position', 'quaternion', 'velocity')):
                raise RuntimeError('nonfinite physics state')
            desired = path.sample(t)
            errors.append(float(np.linalg.norm(state['position']-desired['pos'])))
            if step % 200 == 0:
                frames.append(world.capture(len(frames), rays))
            if step % 100 == 0:
                trace.append([float(world.data.time), *state['position'], *state['velocity']])
            # No stop or reinitialization at these moving decision boundaries.
            if step % 2000 == 0 and 2 <= t <= DURATION-4:
                future = path.sample(t+np.linspace(0, 4, 81))
                windows.append(dict(stamp=float(world.data.time), scan_ids=list(range(len(frames)-4, len(frames))),
                    position=state['position'].copy(), quaternion=state['quaternion'].copy(),
                    velocity=state['velocity'].copy(), angular_velocity=state['angular_velocity'].copy(), reference_acceleration=desired['acc'],
                    future=np.concatenate([future[k] for k in ('pos','quaternion','vel','acc','angular_velocity','angular_acceleration')], axis=1)))
            if step < round(DURATION/DT) and world.step(desired):
                contacts += 1
                break
        final_error = float(np.linalg.norm(world.state()['position']-path.sample(DURATION)['pos']))
        report.update(contact_steps=contacts, completed=step==round(DURATION/DT), tracking_max_m=max(errors),
                      final_error_m=final_error, windows=len(windows), scans=len(frames))
        report['accepted'] = bool(report['completed'] and contacts==0 and max(errors)<.15 and final_error<.12)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        world.close()
    if frames and windows:
        name = f'episode_{index:05d}.npz'
        np.savez_compressed(output/name,
            scan_ranges=np.stack([f.ranges for f in frames]), scan_directions=np.stack([f.ray_directions for f in frames]),
            scan_poses=np.stack([f.world_T_sensor for f in frames]), scan_times=np.array([f.sim_time for f in frames]),
            window_times=np.array([w['stamp'] for w in windows]), history_indices=np.array([w['scan_ids'] for w in windows]),
            positions=np.stack([w['position'] for w in windows]), quaternions=np.stack([w['quaternion'] for w in windows]),
            velocities=np.stack([w['velocity'] for w in windows]), angular_velocities=np.stack([w['angular_velocity'] for w in windows]), reference_accelerations=np.stack([w['reference_acceleration'] for w in windows]),
            future=np.stack([w['future'] for w in windows]), future_times=np.linspace(0,4,81),
            global_goal=path.sample(DURATION)['pos'], split=np.int64(config['split']), accepted=np.bool_(report['accepted']), trace=np.array(trace))
        report.update(file=name, sha256=sha(output/name))
    report['wall_seconds'] = time.perf_counter()-began
    return report


def collect(args):
    if args.count < 1 or args.rays < 1:
        raise ValueError('positive count and rays required')
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    sources = {name: sha(Path(__file__).with_name(name)) for name in ('sdf_navigation_data.py','sdf_data.py','temporal_data.py','sdf_mapping.py','sdf_pose.py','collision_loss.py')}
    sources['controller'] = sha(args.sim_root/'tests/reference/hnuter_control.py')
    sources['mjcf'] = sha(args.sim_root/'robots/hnuter/source/hnuter.xml')
    for index in range(args.count):
        reports.append(collect_episode(index, args.seed, args.output, args.sim_root, args.rays))
        manifest = dict(schema=SCHEMA, requested=args.count, seed=args.seed, rays=args.rays, reports=reports,
                        accepted=sum(r['accepted'] for r in reports), kinds=dict(Counter(r['config']['kind'] for r in reports)),
                        source_hashes=sources, physics='original actuator-driven MuJoCo, dt=0.001; no flight pose writes',
                        teacher='privileged box geometry; sampled sphere clearance plus every-step physical contact checks')
        (args.output/'collection.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps({k:v for k,v in reports[-1].items() if k!='config'}), flush=True)


def validate_history(times, stamp):
    times = np.asarray(times)
    if times.shape != (4,) or not np.isfinite(times).all() or not np.isfinite(stamp) or np.any(np.diff(times) <= 0) or times[-1] > stamp+1e-8 or stamp-times[-1] > .001 or stamp-times[0] > .601:
        raise ValueError('noncausal or stale history')


def prepare(args):
    from .sdf_mapping import LocalSDF
    collection = json.loads((args.source/'collection.json').read_text())
    if collection['schema'] != SCHEMA or len(collection['reports']) != collection['requested']:
        raise ValueError('incomplete or incompatible collection')
    if len({r['index'] for r in collection['reports']}) != collection['requested']:
        raise ValueError('duplicate scene IDs')
    args.output.mkdir(parents=True, exist_ok=False)
    mapper = LocalSDF(args.sim_root)
    records = []
    for row in collection['reports']:
        if not row['accepted']:
            continue  # Failures remain in manifest, never resampled or counted as successes.
        path = args.source/row['file']
        if sha(path) != row['sha256']:
            raise ValueError('source hash mismatch')
        with np.load(path, allow_pickle=False) as d:
            for j, stamp in enumerate(d['window_times']):
                ids = d['history_indices'][j]
                times = d['scan_times'][ids]
                validate_history(times, stamp)
                # All four moving observations are queried in the CURRENT common frame.
                # Clear map per window; no later scans or episode-end map can leak in.
                mapper.reset(d['positions'][j])
                for idx in ids:
                    ranges=d['scan_ranges'][idx]; valid=np.isfinite(ranges)
                    mapper.integrate(d['scan_directions'][idx,valid]*ranges[valid,None], d['scan_poses'][idx], d['scan_times'][idx])
                fields, ages = mapper.inputs()
                future = d['future'][j].copy()
                future[:,:3] -= d['positions'][j]
                name=f"window_{row['index']:05d}_{j:03d}.npz"
                np.savez_compressed(args.output/name, fields=fields.half().cpu().numpy(), ages=ages.cpu().numpy(),
                    current_quaternion=d['quaternions'][j], current_velocity=d['velocities'][j], current_angular_velocity=d['angular_velocities'][j],
                    current_reference_acceleration=d['reference_accelerations'][j],
                    goal=d['global_goal']-d['positions'][j], future=future, future_times=d['future_times'],
                    origin=d['positions'][j], stamp=stamp, scene_id=np.int64(row['index']), split=d['split'])
                records.append(dict(file=name, sha256=sha(args.output/name), source=row['file'], source_sha256=row['sha256'],
                    scene_id=row['index'], split=int(d['split']), stamp=float(stamp), scan_times=times.tolist(),
                    speed_mps=float(np.linalg.norm(d['velocities'][j])), known_fraction=float(fields[-1,3].float().mean().item())))
        print(json.dumps(dict(mapped_scene=row['index'], windows=len(records))), flush=True)
    result=dict(schema=SCHEMA, collection_sha256=sha(args.source/'collection.json'), mapper=mapper.describe(), windows=records,
                rejected_episodes=[r['index'] for r in collection['reports'] if not r['accepted']],
                future_columns='relative_xyz, quaternion_wxyz, world_velocity, world_acceleration, body_omega, body_alpha',
                frame='world-aligned, origin at current measured body position; quaternion stays world wxyz',
                scope='causal four-scan local maps, not persistent SLAM; reference acceleration is prior command, not measured acceleration')
    (args.output/'mapping.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('collect','prepare'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--sim-root',type=Path,default=DEFAULT_SIM_ROOT)
    p.add_argument('--count',type=int,default=6)
    p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--rays',type=int,default=20000)
    args=p.parse_args()
    if args.mode=='prepare' and args.source is None:p.error('prepare requires --source')
    (collect if args.mode=='collect' else prepare)(args)


if __name__=='__main__':
    main()
