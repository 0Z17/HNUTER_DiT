"""Paired 4.1-second actuator rollouts from saved expert native snapshots."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .factory_expert_data import import_collector,atomic_json
from .sdf_data import StaticWorld
from .sdf_pose import PoseSpline,full_controls
from .moving_pose import MovingReference
from .sdf_navigation_data import sha

def unpack(row):
    return dict(pos=row[:3],quaternion=row[3:7],vel=row[7:10],acc=row[10:13],angular_velocity=row[13:16],angular_acceleration=row[16:19])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sim-root',type=Path,default=Path('/home/z017/research/isaacsim-industrial-reactor'));p.add_argument('--collector-root',type=Path,default=Path('/home/z017/research/HNUTER_trajectory_collector'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.predictions/'predictions.json').read_text());protocol=json.loads((a.root/'protocol.json').read_text());gate=protocol['short_horizon_pass_gate']
    cc,_=import_collector(a.collector_root)
    mesh=a.root/'route_08/factory.ply';urdf=a.collector_root/'expert_trajectory_collector/assets/HDJQR-0102-0055.SLDASM.urdf'
    checker=cc.CoalCollisionChecker.from_urdf(urdf,[cc.StaticCollisionObject('factory',cc.CollisionGeometry.mesh(mesh),cc.CollisionPose.identity())],safety_margin=gate['mesh_margin_m'])
    results=[]
    for route in (8,9,10,11):
        with np.load(a.root/f'route_{route:02}/episode.npz') as archive:raw={k:archive[k] for k in archive.files}
        rep=json.loads((a.root/f'route_{route:02}/rollout.json').read_text());flight=raw['window_times']-raw['scan_times'][0]
        selected=sorted(set(int(np.argmin(abs(flight-t))) for t in (4.,rep['geometry']['reference_duration']/2,rep['geometry']['reference_duration']-6)))
        rows=[r for r in manifest['results'] if r['route']==route and r['window'] in selected]
        world=StaticWorld(dict(origin=raw['positions'][0],boxes=[]),a.sim_root)
        try:
            for row in rows:
                j=row['window'];snapshot={k:raw[k][j] for k in ('teacher_native_qpos','teacher_native_qvel','teacher_native_act','teacher_native_time')}
                world.restore(snapshot);state=world.state();np.testing.assert_allclose(state['position'],raw['positions'][j],atol=1e-10,rtol=0)
                np.testing.assert_allclose(state['velocity'],raw['velocities'][j],atol=1e-10,rtol=0)
                with np.load(a.predictions/row['file']) as archive:d={k:archive[k] for k in archive.files}
                c=d['condition'];method=row['method'];frame=d['world_from_local'];origin=d['origin']
                if method.startswith('moving'):ref=MovingReference(d['controls'],c)
                elif method=='legacy_frozen':
                    start=Rotation.from_quat(c[[10,11,12,9]]).as_matrix();target=np.r_[c[:3],Rotation.from_matrix(start.T).as_euler('xyz')]
                    ref=PoseSpline(full_controls(d['controls'],target),start_rotation=start)
                maximum_tracking=0.;maximum_speed=0.;maximum_acc=0.;violation=False;contact=False;trace=[];previous=None;jump=0.;began=time.perf_counter()
                for step in range(4100):
                    if step<100:desired=unpack(raw['previous_reference_bridge'][j,step])
                    else:
                        t=step*.001
                        if method=='constant_velocity':
                            desired=dict(pos=c[3:6]*t,vel=c[3:6],acc=np.zeros(3),quaternion=c[9:13],angular_velocity=np.zeros(3),angular_acceleration=np.zeros(3))
                        else:desired=ref.sample(t)
                        desired=dict(desired)
                        for k in ('pos','vel','acc'):desired[k]=desired[k]@frame.T+(origin if k=='pos' else 0)
                        desired['quaternion']=Rotation.from_matrix(frame@Rotation.from_quat(desired['quaternion'][[1,2,3,0]]).as_matrix()).as_quat()[[3,0,1,2]]
                    state=world.state()
                    if not all(np.isfinite(state[k]).all() for k in ('position','velocity','quaternion')):violation=True;break
                    if not checker.is_collision_free(state['position'],state['quaternion']):violation=True;break
                    maximum_tracking=max(maximum_tracking,float(np.linalg.norm(state['position']-desired['pos'])))
                    maximum_speed=max(maximum_speed,float(np.linalg.norm(state['velocity'])))
                    maximum_acc=max(maximum_acc,float(np.linalg.norm(desired['acc'])))
                    if step==100:jump=float(np.linalg.norm(desired['pos']-previous))
                    previous=desired['pos'].copy()
                    if step%50==0:trace.append([step*.001,*state['position'],*state['velocity']])
                    contact=bool(world.step(desired))
                    if contact:break
                final=world.state()
                if not checker.is_collision_free(final['position'],final['quaternion']):violation=True
                truth=raw['future'][j,-1,:3];error=float(np.linalg.norm(final['position']-truth))
                passed=bool(step==4099 and not violation and not contact and error<gate['teacher_final_error_m'] and maximum_tracking<gate['tracking_model_reference_max_m'] and maximum_speed<gate['actual_speed_max_mps'] and maximum_acc<gate['reference_acceleration_max_mps2'])
                result=dict(route=route,window=j,method=method,passed=passed,steps=step+1,mesh_violation=violation,native_contact=contact,
                    teacher_final_error_m=error,tracking_reference_max_m=maximum_tracking,speed_max_mps=maximum_speed,
                    reference_acceleration_max_mps2=maximum_acc,reference_switch_jump_m=jump,wall_seconds=time.perf_counter()-began)
                results.append(result);np.savez_compressed(a.output/f"trace_{route}_{j}_{method}.npz",trace=np.array(trace))
                print(json.dumps(result),flush=True)
                atomic_json(a.output/'rollouts.json',dict(protocol_sha256=sha(a.root/'protocol.json'),predictions_sha256=sha(a.predictions/'predictions.json'),gate=gate,results=results,
                    scope='Paired short horizon replans from expert states; 100 ms previous reference bridge; not autonomous full-mission deployment; external full-URDF mesh checks plus native actuator dynamics'))
        finally:world.close()
if __name__=='__main__':main()
