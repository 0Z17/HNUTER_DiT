"""Collect a bounded 48-episode task pilot with real actuator/sensor execution."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from .task_spec import scenario, expert, descriptor, target_condition, reference_metrics, surface_geometry, OPERATE, SCHEMA
from .task_physics import TaskWorld, execute_reference
from .task_inspection import inspection_measurement, summarize_inspection
from .temporal_data import DEFAULT_SIM_ROOT, DT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def warmup(world, rays=20000):
    hold = dict(pos=np.array(world.scenario['origin']),quaternion=np.array([1.,0,0,0]),vel=np.zeros(3),acc=np.zeros(3))
    frames=[]
    for step in range(601):
        if step%200==0:
            frames.append(world.capture(step//200,rays))
        if step<600 and world.step(hold):
            raise RuntimeError('unexpected contact during stationary sensing')
    return frames


def execute(world, reference, config):
    """All methods use identical physics/controller/force feedback and gates."""
    started=float(world.data.time)
    if config['task']=='surface':
        report, trace = execute_reference(world,reference)
        for row in trace:
            row['t']=row['time']-started
        selected=[row for row in trace if OPERATE[0]<=row['t']<=OPERATE[1]]
        geometry=surface_geometry(np.array([x['position'] for x in selected]),
                                  np.array([x['quaternion'] for x in selected]),config) if selected else {'passed':False}
        report['actual_task_geometry']=geometry
        progress=float(selected[-1]['tool_pole'][0]-selected[0]['tool_pole'][0]) if selected else 0.
        minimum_progress=.4*(config['goal'][0]-config['origin'][0])
        report['operate_x_progress_m']=progress
        report['minimum_operate_x_progress_m']=minimum_progress
        # Contact penetration and force are assessed by physical metrics. The
        # stricter nominal-plane diagnostic is reported separately, not used to
        # silently relabel a compliant physical contact as geometrically exact.
        task_valid=bool(selected and geometry['footprint_fraction']==1. and
                        geometry['forbidden_patch_fraction']==0. and geometry['normal_error_max_deg']<=3. and
                        progress>=minimum_progress)
        report['physical_execution_passed']=report['passed']
        report['passed']=bool(report['passed'] and task_valid)
    else:
        trace=[];views=[];max_position=0.;max_angle=0.
        times=np.arange(20000)*DT
        desired=reference.sample(times)
        for step,t in enumerate(times):
            command={key:value[step] for key,value in desired.items()}
            row=world.step_task(command,phase='operate' if OPERATE[0]<=t<=OPERATE[1] else 'free')
            max_position=max(max_position,float(np.linalg.norm(np.array(row['position'])-command['pos'])))
            angle=2*np.arccos(np.clip(abs(np.array(row['quaternion'])@command['quaternion']),0,1))
            max_angle=max(max_angle,float(angle))
            if step%10==0 or step==len(times)-1:
                row['t']=float(t+DT)
                row['measurement']=inspection_measurement(world,config)
                views.append(dict(t=row['t'],measurement=row['measurement']))
                trace.append(row)
            if row['forbidden_contact']:
                break
        final=world.state();goal=reference.sample(20.)
        distance=float(np.linalg.norm(final['position']-goal['pos']))
        actual=summarize_inspection(views,config,operate_interval=OPERATE)
        physical=bool(step+1==20000 and world.forbidden_steps==0 and distance<.12 and max_position<.15 and world.saturated_steps/world.steps<=.01)
        report=dict(schema=SCHEMA,completed=step+1==20000,steps=step+1,simulation_seconds=(step+1)*DT,
            metadata=world.metadata(),scenario=config,forbidden_contact_steps=world.forbidden_steps,
            first_forbidden=world.first_forbidden,saturated_steps=world.saturated_steps,
            saturation_fraction=world.saturated_steps/world.steps,
            maximum_continuous_saturation_s=world.maximum_saturation_streak*DT,
            max_position_tracking_error_m=max_position,max_attitude_tracking_error_rad=max_angle,
            final_goal_error_m=distance,final_speed_mps=float(np.linalg.norm(final['velocity'])),
            inspection=actual,physical_execution_passed=physical,passed=bool(physical and actual['passed']))
    report['reference_geometry']=reference_metrics(reference,config)
    report['trajectory_audit']=reference.audit()
    return report,trace


def collect_one(arguments):
    index,seed,sim_root,directory=arguments
    config=scenario(index,seed);world=TaskWorld(config,sim_root);began=time.perf_counter()
    output=Path(directory);trace=[]
    try:
        frames=warmup(world)
        state=world.state();snapshot=world.snapshot()
        path=expert(config,state['position'],state['rotation_matrix'])
        report,trace=execute(world,path,config)
        # Rejected episodes and reasons remain recorded. Only accepted ones can
        # enter a prepared training file; no repeated draw until success.
        np.savez_compressed(output/f'episode_{index:03}.npz',
            history_ray_ranges=np.stack([f.ranges for f in frames]),
            ray_directions=np.stack([f.ray_directions for f in frames]),
            history_sensor_poses=np.stack([f.world_T_sensor for f in frames]),
            history_times=np.array([f.sim_time for f in frames]),
            origin=state['position'],start_rotation=state['rotation_matrix'],
            target=target_condition(config,state['position'],state['rotation_matrix']).astype(np.float32),
            task_descriptor=descriptor(config,state['position'],state['rotation_matrix']),
            free_controls=path.controls[3:9].astype(np.float32),
            split=np.int64(config['split']),task=np.array(config['task']),layout=np.int64(config['layout']),
            accepted=np.bool_(report['passed']),**snapshot)
        (output/f'trace_{index:03}.json').write_text(json.dumps(trace,separators=(',',':'))+'\n')
    except Exception as exc:
        report=dict(passed=False,error=f'{type(exc).__name__}: {exc}',completed=False)
    finally:
        world.close()
    report.update(index=index,config=config,elapsed_seconds=time.perf_counter()-began)
    (output/f'report_{index:03}.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def collect(output,indices,workers=4,seed=20261004,sim_root=DEFAULT_SIM_ROOT):
    if len(indices)>48 or len(set(indices))!=len(indices):
        raise ValueError('pilot is capped at 48 unique attempts')
    output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('task_spec.py'),Path(__file__).with_name('task_physics.py'),
             Path(__file__).with_name('task_inspection.py'),Path(sim_root)/'tests/reference/hnuter_control.py',
             Path(sim_root)/'robots/hnuter/source/hnuter.xml']
    snap=output/'source_snapshot';snap.mkdir()
    for file in sources:(snap/file.name).write_bytes(file.read_bytes())
    provenance={str(file):sha(file) for file in sources}
    reports=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for report in pool.map(collect_one,[(i,seed,sim_root,output) for i in indices]):
            reports.append(report)
            print(json.dumps({key:report.get(key) for key in ('index','passed','error','elapsed_seconds')}),flush=True)
            summary=dict(schema=SCHEMA,seed=seed,requested_indices=indices,count=len(reports),
                accepted_count=sum(bool(row['passed']) for row in reports),reports=reports,
                source_sha256=provenance,elapsed_seconds=time.perf_counter()-start,
                split_unit='layout; paired task conditions and tasks never cross splits',
                scope='bounded native MuJoCo pilot; not Isaac industrial execution, SLAM or real-world contact')
            (output/'collection.json').write_text(json.dumps(summary,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--indices',type=int,nargs='+',default=list(range(48)))
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20261004)
    p.add_argument('--sim-root',type=Path,default=DEFAULT_SIM_ROOT)
    args=p.parse_args();collect(args.output,args.indices,args.workers,args.seed,args.sim_root)


if __name__=='__main__':main()
