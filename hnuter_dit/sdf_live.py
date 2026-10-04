"""Wall-clock paced native-physics static SDF pose flights with 10 Hz sensing."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .sdf_data import StaticWorld,scenario
from .sdf_online import SDFOnlinePlanner


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int,default=96)
    p.add_argument('--sim-root',default='/sim');p.add_argument('--duration',type=float,default=24.)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    config=scenario(args.index);world=StaticWorld(config,args.sim_root)
    goal=np.array(config['origin'])+np.array(config['target'])[:3]
    planner=SDFOnlinePlanner(goal_position=goal,goal_rpy=config['target'][3:],checkpoint=args.checkpoint,output_dir=args.output/'worker')
    contacts=0;trace=[];captures=[];times=[];start=time.perf_counter();report={}
    try:
        for step in range(round(args.duration/.001)):
            state=world.state();stamp=step*.001
            if step%100==0:
                began=time.perf_counter();frame=world.capture(step//100,20000)
                capture_seconds=time.perf_counter()-began;captures.append(capture_seconds)
                planner.observe(frame,state,stamp,capture_wall_time=began)
            command=planner.command(state,stamp)
            if step%20==0:trace.append(dict(t=stamp,pos=state['position'].tolist(),quaternion=state['quaternion'].tolist(),desired=command['pos'].tolist(),status=planner.last_command_category))
            contacts+=world.step(command)
            remaining=start+(step+1)*.001-time.perf_counter()
            if remaining>0:time.sleep(remaining)
        final=world.state();r=Rotation.from_quat(final['quaternion'][[1,2,3,0]])
        angle=float((Rotation.from_euler('xyz',config['target'][3:]).inv()*r).magnitude())
        report=dict(index=args.index,config=config,steps=step+1,simulation_seconds=(step+1)*.001,
            loop_wall_seconds=time.perf_counter()-start,contact_steps=contacts,
            final_goal_distance_m=float(np.linalg.norm(final['position']-goal)),final_goal_angle_rad=angle,
            final_speed_mps=float(np.linalg.norm(final['velocity'])),
            raycast_seconds=dict(median=float(np.median(captures)),p95=float(np.percentile(captures,95)),maximum=float(max(captures))))
    finally:
        report['planner']=planner.close();world.close()
        report['passed']=bool(report.get('contact_steps')==0 and report.get('final_goal_distance_m',99)<.12 and
            report.get('final_goal_angle_rad',99)<.08 and not report['planner']['failed'] and report['planner']['accepted_plans']>0)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.output/'trajectory.json').write_text(json.dumps(trace)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='planner'}),flush=True)
    raise SystemExit(0 if report['passed'] else 2)


if __name__=='__main__':main()
