"""Factory mesh + external OMPL/COAL teacher + causal moving nvblox windows.

External collector is read-only. No claim of real-time inference or Isaac mesh
contact dynamics: native actuator dynamics and external full-URDF mesh checks
are separate audited stages. No geometric truth is a student input.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from .sdf_navigation_data import sha
from .sdf_data import StaticWorld
from .sdf_pose import angular_derivatives
from .temporal_data import _module

SCHEMA='factory_causal_navigation_v1'


def atomic_json(path, value):
    path=Path(path); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def import_collector(root):
    root=Path(root).resolve()
    sys.path.insert(0,str(root))
    import coal_collision as cc
    import ompl_se3_planner as om
    if Path(cc.__file__).resolve().parent != root or Path(om.__file__).resolve().parent != root:
        raise RuntimeError('collector module path mismatch')
    return cc,om


class TimedPath:
    """One C2 spatial spline with a global quintic clock, no waypoint stops."""
    def __init__(self, states, duration):
        states=np.asarray(states,float)
        if states.ndim!=2 or states.shape[1]!=7 or len(states)<2 or not np.isfinite(states).all():
            raise ValueError('finite pose7 path required')
        d=np.linalg.norm(np.diff(states[:,:3],axis=0),axis=1)
        keep=np.r_[True,d>1e-6];states=states[keep]
        if len(states)<2:raise ValueError('translation path required')
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(states[:,:3],axis=0),axis=1))]
        angles=np.unwrap(Rotation.from_quat(states[:,[4,5,6,3]]).as_euler('xyz'),axis=0)
        # Dense OMPL samples otherwise force an almost sharp corner. Fit a
        # coarser spatial spline, then independently recheck the entire curve.
        knots=np.linspace(0,arc[-1],max(3,int(np.ceil(arc[-1]/.8))+1))
        values=np.c_[states[:,:3],angles]
        controls=np.stack([np.interp(knots,arc,values[:,k]) for k in range(6)],axis=1)
        self.spline=CubicSpline(knots/arc[-1],controls,bc_type='natural')
        self.duration=float(duration)
        if not np.isfinite(self.duration) or self.duration<=0:raise ValueError('positive duration required')

    def sample(self,t):
        scalar=np.ndim(t)==0;t=np.atleast_1d(t).astype(float)
        if not np.isfinite(t).all() or np.any(t<0) or np.any(t>self.duration+1e-7):raise ValueError('invalid reference time')
        u=np.clip(t/self.duration,0,1);s=10*u**3-15*u**4+6*u**5
        sd=30*u**2*(1-u)**2/self.duration;sdd=(60*u-180*u**2+120*u**3)/self.duration**2
        x=self.spline(s);v=self.spline(s,1)*sd[:,None]
        a=self.spline(s,2)*sd[:,None]**2+self.spline(s,1)*sdd[:,None]
        w,alpha=angular_derivatives(x[:,3:],v[:,3:],a[:,3:])
        result=dict(pos=x[:,:3],quaternion=Rotation.from_euler('xyz',x[:,3:]).as_quat()[:,[3,0,1,2]],
                    vel=v[:,:3],acc=a[:,:3],angular_velocity=w,angular_acceleration=alpha)
        return {k:v[0] if scalar else v for k,v in result.items()}


def geometry_and_plan(args,output):
    cc,om=import_collector(args.collector_root)
    import trimesh
    mesh_path=args.sim_root/'_build/collision.npz'
    with np.load(mesh_path) as data:vertices=data['vertices'].copy();faces=data['faces'].copy()
    # Export the exact triangles: no hull, OBB reduction or decimation.
    trimesh.Trimesh(vertices,faces,process=False).export(output/'factory.ply')
    urdf=args.collector_root/'expert_trajectory_collector/assets/HDJQR-0102-0055.SLDASM.urdf'
    env=cc.StaticCollisionObject('factory',cc.CollisionGeometry.mesh(output/'factory.ply'),cc.CollisionPose.identity())
    checker=cc.CoalCollisionChecker.from_urdf(urdf,[env],safety_margin=.15)
    mission=json.loads(args.mission.read_text());points=np.array([mission['start_position'],*mission['waypoints']],float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or not np.isfinite(points).all():raise ValueError('invalid mission')
    lower=points.min(0)-[3,3,2];upper=points.max(0)+[3,3,2]
    states=[];segments=[];began=time.perf_counter()
    planner=om.OMPLSE3Planner(lower,upper,collision_checker=checker,seed=args.seed,validity_resolution=.002,planner_range=.6)
    for i,(start,goal) in enumerate(zip(points[:-1],points[1:])):
        result=planner.plan(om.SE3Pose(start,[1,0,0,0]),om.SE3Pose(goal,[1,0,0,0]),solve_time=args.solve_time,
                            interpolation_resolution=.15,minimum_waypoints=8)
        states.extend(result.states if i==0 else result.states[1:])
        segments.append(dict(planner=result.planner_name,seconds=result.planning_time_s,raw_states=result.raw_state_count))
    states=np.array(states);np.savez_compressed(output/'geometry.npz',states=states,mission_points=points)
    length=np.linalg.norm(np.diff(states[:,:3],axis=0),axis=1).sum()
    reference=TimedPath(states,max(20.,float(length)*4.))
    dense=reference.sample(np.linspace(0,reference.duration,2001))
    # Uniform retiming, not moving control points or per-waypoint braking.
    scale=max(1.,np.linalg.norm(dense['vel'],axis=1).max()/.6,
              np.sqrt(np.linalg.norm(dense['acc'],axis=1).max()/.6),
              np.linalg.norm(dense['angular_velocity'],axis=1).max()/.3)
    reference.duration=float(np.ceil(reference.duration*scale*10)/10)
    if reference.duration>args.max_duration:raise RuntimeError('retimed path exceeds explicit duration budget')
    dense=reference.sample(np.linspace(0,reference.duration,2001))
    clearance=checker.clearance(dense['pos'],dense['quaternion'])
    if np.min(clearance)<.15:raise RuntimeError('smoothed path failed independent mesh clearance gate')
    angles=Rotation.from_quat(dense['quaternion'][:,[1,2,3,0]]).as_euler('xyz')
    if np.abs(angles[:,:2]).max()>.2:raise RuntimeError('expert outside validated small roll/pitch curriculum')
    audit=dict(schema=SCHEMA,mesh_sha256=sha(mesh_path),urdf_sha256=sha(urdf),mission_sha256=sha(args.mission),
        collector_sources={n:sha(args.collector_root/n) for n in ('coal_collision.py','ompl_se3_planner.py')},
        geometry_sha256=sha(output/'geometry.npz'),source_sha256=sha(__file__),triangles=len(faces),vertices=len(vertices),
        segments=segments,geometry_wall_seconds=time.perf_counter()-began,reference_duration=reference.duration,
        reference_min_clearance_m=float(np.min(clearance)),trajectory_length_m=float(length),
        speed_max_mps=float(np.linalg.norm(dense['vel'],axis=1).max()),
        acceleration_max_mps2=float(np.linalg.norm(dense['acc'],axis=1).max()),
        stage='full mesh OMPL/COAL geometry and sampled retiming audit; not rollout',mission=mission)
    atomic_json(output/'geometry.json',audit)
    return checker,reference,vertices,faces,points,audit


def collect(args):
    args.output.mkdir(parents=True,exist_ok=False)
    output=args.output;began=time.perf_counter();report=dict(schema=SCHEMA,accepted=False,stage='initialization')
    world=None
    try:
        checker,ref,vertices,faces,points,geometry=geometry_and_plan(args,output)
        from .factory_mesh_sensor import MeshSensor
        sensor=MeshSensor(vertices,faces,args.sim_root,args.rays)
        # Factory contact responses are NOT inserted as a convex MuJoCo mesh.
        # The reference robot dynamics are preserved; COAL checks actual poses.
        world=StaticWorld(dict(origin=points[0],boxes=[]),args.sim_root)
        for _ in range(600):
            state=world.state()
            if not checker.is_collision_free(state['position'],state['quaternion']):raise RuntimeError('warmup mesh collision')
            if world.step(dict(pos=points[0],quaternion=np.array([1,0,0,0]),vel=np.zeros(3),acc=np.zeros(3))):raise RuntimeError('native warmup contact')
        frames=[];windows=[];traces=[];errors=[];angles=[];scan_seconds=[];mesh_checks=0;waypoint_index=1
        count=round(ref.duration/.001)
        for step in range(count+1):
            t=step*.001;state=world.state();desired=ref.sample(t)
            if not all(np.isfinite(state[k]).all() for k in ('position','velocity','quaternion','angular_velocity')):raise RuntimeError('nonfinite state')
            if not checker.is_collision_free(state['position'],state['quaternion']):raise RuntimeError('actual full URDF mesh collision/margin violation')
            mesh_checks+=1
            errors.append(float(np.linalg.norm(state['position']-desired['pos'])))
            angles.append(float(2*np.arccos(np.clip(abs(state['quaternion']@desired['quaternion']),0,1))))
            if step%100==0:
                start=time.perf_counter();frames.append(sensor.capture(state,float(world.data.time),len(frames)));scan_seconds.append(time.perf_counter()-start)
                traces.append([float(world.data.time),*state['position'],*state['quaternion'],*state['velocity']])
            if waypoint_index < len(points)-1 and np.linalg.norm(points[waypoint_index]-state['position']) < .5:
                waypoint_index += 1
            if step%1000==0 and 1<=t<=ref.duration-4.1:
                ids=np.arange(len(frames)-7,len(frames),2)
                future=ref.sample(t+.1+np.linspace(0,4,81))
                # A public mission waypoint is conditioning; never the teacher path.
                goal=points[waypoint_index]
                windows.append(dict(time=float(world.data.time),ids=ids,position=state['position'].copy(),
                    quaternion=state['quaternion'].copy(),velocity=state['velocity'].copy(),omega=state['angular_velocity'].copy(),
                    acc=desired['acc'],goal=goal,snapshot=world.snapshot(),
                    bridge=np.concatenate([ref.sample(t+np.linspace(0,.1,101))[k] for k in ('pos','quaternion','vel','acc','angular_velocity','angular_acceleration')],axis=1),
                    future=np.concatenate([future[k] for k in ('pos','quaternion','vel','acc','angular_velocity','angular_acceleration')],axis=1)))
            if step<count and world.step(desired):raise RuntimeError('native robot contact')
        report.update(stage='rollout_complete',geometry=geometry,physics_steps=count,actual_mesh_checks=mesh_checks,
            tracking_max_m=max(errors),attitude_tracking_max_rad=max(angles),windows=len(windows),scans=len(frames),
            final_error_m=errors[-1],scan_wall_p50_s=float(np.median(scan_seconds)),scan_wall_p95_s=float(np.percentile(scan_seconds,95)),
            source_sha256=sha(__file__),sensor_source_sha256=sha(Path(__file__).with_name('factory_mesh_sensor.py')),
            physics='native unchanged actuator dynamics; external full-URDF COAL mesh query every 1 ms; no factory contact response',
            seed=args.seed,split=args.split,split_group=geometry['mesh_sha256'])
        report['accepted']=bool(max(errors)<.15 and max(angles)<.08 and errors[-1]<.12 and len(windows)>0)
        np.savez_compressed(output/'episode.npz',scan_ranges=np.stack([f.ranges for f in frames]),
            scan_directions=np.stack([f.ray_directions for f in frames]),scan_poses=np.stack([f.world_T_sensor for f in frames]),
            scan_times=np.array([f.sim_time for f in frames]),history_indices=np.stack([w['ids'] for w in windows]),
            window_times=np.array([w['time'] for w in windows]),positions=np.stack([w['position'] for w in windows]),
            quaternions=np.stack([w['quaternion'] for w in windows]),velocities=np.stack([w['velocity'] for w in windows]),
            angular_velocities=np.stack([w['omega'] for w in windows]),reference_accelerations=np.stack([w['acc'] for w in windows]),
            goals=np.stack([w['goal'] for w in windows]),future=np.stack([w['future'] for w in windows]),future_times=.1+np.linspace(0,4,81),
            teacher_native_qpos=np.stack([w['snapshot']['teacher_native_qpos'] for w in windows]),
            teacher_native_qvel=np.stack([w['snapshot']['teacher_native_qvel'] for w in windows]),
            teacher_native_act=np.stack([w['snapshot']['teacher_native_act'] for w in windows]),
            teacher_native_time=np.array([w['snapshot']['teacher_native_time'] for w in windows]),
            previous_reference_bridge=np.stack([w['bridge'] for w in windows]),
            trace=np.array(traces),split=np.int64(args.split),accepted=np.bool_(report['accepted']))
        report['episode_sha256']=sha(output/'episode.npz')
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',accepted=False)
        raise
    finally:
        if world is not None:world.close()
        report['wall_seconds']=time.perf_counter()-began
        atomic_json(output/'rollout.json',report)


def prepare(args):
    from .sdf_mapping import LocalSDF
    from .sdf_navigation_data import validate_history
    source=args.source;report=json.loads((source/'rollout.json').read_text())
    if not report['accepted'] or report['schema']!=SCHEMA or sha(source/'episode.npz')!=report['episode_sha256']:
        raise ValueError('accepted compatible rollout with verified hash required')
    args.output.mkdir(parents=True,exist_ok=False);mapper=LocalSDF(args.sim_root);mapper.torch.set_num_threads(2);records=[]
    with np.load(source/'episode.npz',allow_pickle=False) as archive:
        # NPZ indexing decompresses the entire member: materialize each once,
        # rather than re-decoding the full flight for every individual scan.
        d={key:archive[key] for key in archive.files}
        for j,stamp in enumerate(d['window_times']):
            ids=d['history_indices'][j];validate_history(d['scan_times'][ids],stamp)
            # Heading-aligned measured body frame (yaw only), common across history.
            yaw=Rotation.from_quat(d['quaternions'][j][[1,2,3,0]]).as_euler('xyz')[2]
            frame=Rotation.from_euler('z',yaw).as_matrix();origin=d['positions'][j]
            mapper.reset(origin,frame)
            started=time.perf_counter()
            prefix=np.flatnonzero((d['scan_times'] >= stamp-args.map_history_seconds-1e-8)&(d['scan_times'] <= stamp+1e-8))
            for idx in prefix:
                ranges=d['scan_ranges'][idx];keep=np.isfinite(ranges)
                mapper.integrate(d['scan_directions'][idx,keep]*ranges[keep,None],d['scan_poses'][idx],d['scan_times'][idx])
            fields,ages=mapper.inputs()
            future=d['future'][j].copy();future[:,:3]=(future[:,:3]-origin)@frame
            rotations=Rotation.from_quat(future[:,[4,5,6,3]]).as_matrix()
            future[:,3:7]=Rotation.from_matrix(frame.T[None]@rotations).as_quat()[:,[3,0,1,2]]
            future[:,7:10]=future[:,7:10]@frame;future[:,10:13]=future[:,10:13]@frame
            current_r=frame.T@Rotation.from_quat(d['quaternions'][j][[1,2,3,0]]).as_matrix()
            name=f'window_{j:04d}.npz'
            np.savez_compressed(args.output/name,fields=fields.half().cpu().numpy(),ages=ages.cpu().numpy(),
                current_velocity=d['velocities'][j]@frame,current_angular_velocity=d['angular_velocities'][j],
                current_quaternion=Rotation.from_matrix(current_r).as_quat()[[3,0,1,2]],
                current_reference_acceleration=d['reference_accelerations'][j]@frame,goal=(d['goals'][j]-origin)@frame,
                future=future,future_times=d['future_times'],execution_delay_s=np.float32(.1),origin=origin,
                world_from_local=frame,stamp=stamp,split=d['split'],split_group=np.array(report['split_group']))
            records.append(dict(file=name,sha256=sha(args.output/name),stamp=float(stamp),scan_times=(stamp+ages.cpu().numpy().astype(float)).tolist(), integrated_scan_times=d['scan_times'][prefix].tolist(),
                map_prepare_wall_seconds=time.perf_counter()-started,known_fraction=float(fields[-1,3].float().mean().item())))
            print(json.dumps(records[-1]),flush=True)
    atomic_json(args.output/'mapping.json',dict(schema=SCHEMA,records=records,mapper=mapper.describe(),
        source_sha256=report['episode_sha256'],source_rollout_sha256=sha(source/'rollout.json'),
        split_group=report['split_group'],split=report['split'],execution_delay_s=.1,map_history_seconds=args.map_history_seconds,
        contract='yaw-aligned local positions/linear derivatives, local wxyz; body angular derivatives; 100 ms future label offset is a training design, not measured model latency',
        scope='four causal 5 Hz snapshots of bounded past-only 10 Hz integrated map; reconstructed per window; no future geometry input'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('collect','prepare'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--sim-root',type=Path,default=Path('/home/z017/research/isaacsim-industrial-reactor'))
    p.add_argument('--collector-root',type=Path,default=Path('/home/z017/research/HNUTER_trajectory_collector'))
    p.add_argument('--mission',type=Path);p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--split',type=int,choices=(0,1,2),default=0);p.add_argument('--rays',type=int,default=20000)
    p.add_argument('--map-history-seconds',type=float,default=5.)
    p.add_argument('--solve-time',type=float,default=2.);p.add_argument('--max-duration',type=float,default=90.)
    args=p.parse_args()
    if not .6 <= args.map_history_seconds <= 20.:p.error('map history must be 0.6 to 20 s')
    if args.mode=='collect' and args.mission is None:p.error('collect requires mission')
    if args.mode=='prepare' and args.source is None:p.error('prepare requires source')
    (collect if args.mode=='collect' else prepare)(args)

if __name__=='__main__':main()
