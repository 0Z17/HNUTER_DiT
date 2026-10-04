"""Collect measured MID-360 histories and actuator-validated static pose experts."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET
import numpy as np
from .temporal_data import NativeWorld, _module, DEFAULT_SIM_ROOT
from .sdf_pose import HORIZON, PROGRESS, PoseSpline, full_controls, proxy_clearance, SCHEMA


def scenario(index,seed=811,background_randomization=False):
    rng=np.random.default_rng(np.random.SeedSequence([seed,index]))
    side=1 if index%2 else -1
    origin=np.array([0.,0.,2.3])
    target=np.r_[rng.uniform(4.2,4.6),0.,rng.uniform(-.10,.15),
                 rng.uniform(-.12,.12),rng.uniform(-.12,.12),rng.uniform(-.35,.35)]
    obstacle=[target[0]*rng.uniform(.47,.53),side*rng.uniform(.30,.50),2.3,
              rng.uniform(.55,.8),rng.uniform(.75,1.0),3.2]
    # A second static barrier blocks the side already occupied by the main box.
    # Only ray returns, never side/geometry, are presented to the learned policy.
    barrier=[obstacle[0],side*1.8,2.3,1.0,.65,3.2]
    boxes=[obstacle,barrier,[6.,0.,2.5,.2,14.,5.],
           [0.,6.,2.5,16.,.2,5.],[0.,-6.,2.5,16.,.2,5.],
           [0.,0.,-.05,16.,16.,.1]]
    if background_randomization:
        # Sensor-domain diversity: close/open backgrounds are independent of
        # route side. All returns still come from actual collision geometry.
        boxes[2][0]=float(rng.choice([6.,9.,16.]))
        boxes[3][1]=float(rng.choice([4.5,7.,16.]))
        boxes[4][1]=-float(rng.choice([4.5,7.,16.]))
        for _ in range(4):
            boxes.append([float(rng.uniform(-2,7)),float(rng.choice([-1,1])*rng.uniform(3.5,6.)),
                float(rng.uniform(1.5,3.)),float(rng.uniform(.4,1.1)),float(rng.uniform(.4,.8)),float(rng.uniform(2.,5.))])
    return dict(index=int(index),seed=seed,origin=origin.tolist(),target=target.tolist(),
                boxes=boxes,side=side,background_randomization=background_randomization,
                split=0 if index<64 else (1 if index<80 else 2))


class StaticWorld(NativeWorld):
    def __init__(self, config, sim_root=DEFAULT_SIM_ROOT):
        import mujoco
        self.mj,self.scenario,self.sim_root=mujoco,config,Path(sim_root)
        source=self.sim_root/'robots/hnuter/source/hnuter.xml'
        xml=ET.parse(source).getroot(); xml.find('compiler').set('meshdir',str(source.parent/'meshes'))
        for index,box in enumerate(config['boxes']):
            ET.SubElement(xml.find('worldbody'),'geom',name=f'sdf_static_{index}',type='box',
                pos=' '.join(map(str,box[:3])),size=' '.join(str(x/2) for x in box[3:]),
                contype='1',conaffinity='1',group='0')
        self._directory=tempfile.TemporaryDirectory(prefix='hnuter-static-sdf-')
        path=Path(self._directory.name)/'world.xml'; path.write_text(ET.tostring(xml,encoding='unicode'))
        cls=_module(self.sim_root/'tests/reference/hnuter_control.py','_static_sdf_native').HnuterController
        self.controller=cls(path); self.model,self.data=self.controller.model,self.controller.data
        assert np.isclose(self.model.opt.timestep,.001)
        self.controller.set_freejoint_pose(config['origin'])
        self.robot_geoms=self.model.body_rootid[self.model.geom_bodyid]==self.controller._body_id
        self.model.geom_group[self.robot_geoms]=5
        mujoco.mj_forward(self.model,self.data)


def expert(config,origin,start_rotation):
    target=np.array(config['target']); best=None
    # Geometry-aware expert searches smooth detours; no route is supplied at inference.
    for direction in (-1,1):
        for amplitude in (1.35,1.55,1.75,1.95,2.15):
            free=PROGRESS[:,None]*target
            free[:,1]+=direction*amplitude*np.array([.45,.90,1.,1.,.90,.45])
            path=PoseSpline(full_controls(free,target),origin=origin,start_rotation=start_rotation)
            clearance=proxy_clearance(path,config['boxes'])
            audit=path.audit()
            if clearance<.22 or audit['speed_bound_mps']>.65 or audit['acceleration_bound_mps2']>.9:
                continue
            points=path.sample(np.linspace(0,HORIZON,201))['pos']
            length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
            if best is None or length<best[0]:best=(length,path,clearance)
    if best is None:raise RuntimeError('expert search failed')
    return best[1],best[2]


def run_path(world,path,record=False,stop_on_contact=False):
    times=np.arange(round(HORIZON/.001)+1)*.001
    desired=path.sample(times)
    contacts=0; max_position=0.; max_attitude=0.; trace=[]
    for step,t in enumerate(times):
        state=world.state()
        if not all(np.isfinite(state[key]).all() for key in ('position','quaternion','velocity')):
            raise RuntimeError('non-finite physical state during pose evaluation')
        position_error=np.linalg.norm(state['position']-desired['pos'][step])
        q=state['quaternion']; qt=desired['quaternion'][step]
        angle=2*np.arccos(np.clip(abs(q@qt),0,1))
        max_position=max(max_position,float(position_error)); max_attitude=max(max_attitude,float(angle))
        if record and step%100==0:
            trace.append(dict(t=float(t),position=state['position'].tolist(),quaternion=q.tolist()))
        if step<len(times)-1:
            contacts+=world.step({key:value[step] for key,value in desired.items()})
            if contacts and stop_on_contact:break
    final=world.state(); goal=path.sample(HORIZON)
    distance=float(np.linalg.norm(final['position']-goal['pos']))
    angle=float(2*np.arccos(np.clip(abs(final['quaternion']@goal['quaternion']),0,1)))
    return dict(contact_steps=contacts,goal_distance_m=distance,goal_angle_rad=angle,
        final_speed_mps=float(np.linalg.norm(final['velocity'])),tracking_max_m=max_position,
        attitude_tracking_max_rad=max_attitude,steps=min(step+1,len(times)-1),completed=bool(step==len(times)-1),
        passed=bool(step==len(times)-1 and contacts==0 and distance<.12 and angle<.08 and max_position<.15),trace=trace)


def collect_one(arguments):
    index,seed,sim_root,directory,background_randomization=arguments
    config=scenario(index,seed,background_randomization); world=StaticWorld(config,sim_root)
    began=time.perf_counter()
    try:
        hold=dict(pos=np.array(config['origin']),quaternion=np.array([1.,0,0,0]),vel=np.zeros(3),acc=np.zeros(3))
        frames=[]
        for step in range(601):
            if step%200==0:frames.append(world.capture(step//200,20000))
            if step<600 and world.step(hold):raise RuntimeError('warmup contact')
        state=world.state(); snapshot=world.snapshot()
        path,clearance=expert(config,state['position'],state['rotation_matrix'])
        report=run_path(world,path)
        if not report['passed']:raise RuntimeError('expert physical validation failed: '+str(report))
        np.savez_compressed(Path(directory)/f'episode_{index:03}.npz',
            history_ray_ranges=np.stack([f.ranges for f in frames]),
            ray_directions=np.stack([f.ray_directions for f in frames]),
            history_sensor_poses=np.stack([f.world_T_sensor for f in frames]),
            history_times=np.array([f.sim_time for f in frames]),
            origin=state['position'],start_rotation=state['rotation_matrix'],
            target=np.array(config['target'],np.float32),free_controls=path.controls[3:9].astype(np.float32),
            split=np.int64(config['split']),**snapshot)
        report.update(index=index,config=config,proxy_clearance_m=clearance,
            trajectory_audit=path.audit(),elapsed_seconds=time.perf_counter()-began)
        return report
    finally:world.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=96)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--seed',type=int,default=811)
    parser.add_argument('--background-randomization',action='store_true')
    parser.add_argument('--sim-root',type=Path,default=DEFAULT_SIM_ROOT)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    reports=[]; start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[(i,args.seed,args.sim_root,args.output,args.background_randomization) for i in range(args.count)]
        for report in pool.map(collect_one,jobs):
            reports.append(report)
            print(json.dumps({k:report[k] for k in ('index','passed','contact_steps','elapsed_seconds')}),flush=True)
            (args.output/'collection.json').write_text(json.dumps(dict(schema=SCHEMA,
                reports=reports,count=len(reports),requested_count=args.count,elapsed_seconds=time.perf_counter()-start),indent=2)+'\n')


if __name__=='__main__':main()
