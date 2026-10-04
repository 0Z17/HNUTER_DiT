#!/usr/bin/env python3
"""Archive compact static-SDF evidence and plot actual industrial flight traces."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
    root=args.root;raw=root/'results/static_sdf';artifact=root/'artifacts/static_sdf_pose_pilot_v1'
    artifact.mkdir(parents=True,exist_ok=True);(artifact/'models/sdf').mkdir(parents=True,exist_ok=True)
    checkpoint=raw/'train2000_diverse_seed31/best.pt';dest=artifact/'models/sdf/best.pt'
    if dest.exists() and sha(dest)!=sha(checkpoint):raise RuntimeError('refusing to replace a different frozen checkpoint')
    shutil.copyfile(checkpoint,dest)
    initial=artifact/'models/initial_sdf/best.pt';initial.parent.mkdir(parents=True,exist_ok=True)
    if initial.exists() and sha(initial)!=sha(raw/'train2000_seed29/best.pt'):raise RuntimeError('initial checkpoint differs')
    shutil.copyfile(raw/'train2000_seed29/best.pt',initial)
    collection=read(raw/'diverse96_seed919/collection.json')
    native=read(raw/'test_diverse_native/native.json');offline=read(raw/'test_diverse_frozen/offline.json')
    files={'training':raw/'train2000_diverse_seed31/training.json','mapping':raw/'mapped_diverse96/mapping.json',
        'initial_training':raw/'train2000_seed29/training.json','initial_mapping':raw/'mapped96/mapping.json',
        'initial_offline':raw/'test_frozen/offline.json','initial_native':raw/'test_native/native.json',
        'offline':raw/'test_diverse_frozen/offline.json','native':raw/'test_diverse_native/native.json',
        'industrial96_initial_failure':raw/'industrial96/report.json',
        'industrial96_wallclock':raw/'industrial96_wallclock/report.json',
        'industrial97_warmup_failure':raw/'industrial97_warmed/report.json',
        'industrial97_initial_single':raw/'industrial97_final/report.json',
        'industrial97_initial_candidates':raw/'industrial97_candidates/report.json',
        'industrial100_final':raw/'industrial100_final/report.json',
        'industrial101_final':raw/'industrial101_final/report.json',
        'live96_initial':raw/'live96/report.json'}
    for name in ('live96_final','live97_final'):
        path=raw/name/'report.json'
        if path.exists():files[name]=path
    entries={}
    for name,path in files.items():
        doc=read(path)
        # Detailed per-scan metrics remain in local raw output; counts/timing
        # summaries, plan decisions and original report hashes are retained.
        if 'planner' in doc:
            planner=doc['planner'];scans=planner.pop('scans',[])
            planner['recorded_scan_responses']=len(scans)
            planner['planning_responses']=[r for r in scans if r.get('plan_requested')]
            steady=[r['sensor_to_response_wall_seconds'] for r in scans if r['scan_time']>=.6]
            planner['steady_response_seconds']=dict(count=len(steady),median=float(np.median(steady)),p95=float(np.percentile(steady,95)),maximum=float(max(steady))) if steady else None
            planner['steady_deadline_misses_100ms']=sum(x>.1 for x in steady)
            lidar_path=path.with_name('lidar_frames.json')
            if lidar_path.exists():
                captures={round(r['t'],8):r.get('capture_wall_seconds') for r in read(lidar_path)}
                completion=[captures.get(round(r['scan_time'],8)) for r in scans if r['scan_time']>=.6]
                if completion and all(x is not None for x in completion):
                    completion=np.asarray(completion)+np.asarray(steady)
                    intervals=np.diff(completion)
                    planner['steady_processed_update_intervals_seconds']=dict(median=float(np.median(intervals)),p95=float(np.percentile(intervals,95)),maximum=float(max(intervals)))
        (artifact/f'{name}.json').write_text(json.dumps(doc,indent=2)+'\n')
        entries[name]=dict(path=str(path.relative_to(root)),sha256=sha(path))
    report=dict(schema='static_sdf_pose_validation_v1',date='2026-10-04',checkpoint_sha256=sha(checkpoint),
        selected_step=offline['selected_step'],collection=dict(episodes=len(collection['reports']),
        physical_successes=sum(r['passed'] for r in collection['reports']),contacts=sum(r['contact_steps'] for r in collection['reports']),
        max_position_tracking_m=max(r['tracking_max_m'] for r in collection['reports']),
        collection_seconds=collection['elapsed_seconds'],split_counts=dict(train=64,validation=16,test=16),seed=919,background_randomization=True),
        training=dict(steps=2000,seed=31,batch_size=16,learning_rate=.0003,ema=.99,from_scratch=True),
        held_out_native=native['summary'],offline_ablation={k:dict(passed=v['passed'],count=v['count']) for k,v in offline['results'].items()},
        inference_seconds=offline['inference_seconds'],evidence=entries,
        sources={str(path.relative_to(root)):sha(path) for path in sorted((root/'hnuter_dit').glob('sdf_*.py'))},
        limitations=['single small curriculum and training seed','stationary start, up to three observation-driven attempts then fixed 20s pose plan with continued sensing',
            'history has no demonstrated benefit over latest observation in this static curriculum',
            'unknown space is not certified free','industrial physics loop is slower than wall-clock real time',
            'modest orientation changes, no learned contact or narrow-gap attitude maneuvers'])
    report['flights']={}
    for name in files:
        if 'industrial' not in name and 'live' not in name:continue
        doc=read(files[name]);planner=doc.get('planner',{})
        report['flights'][name]={key:doc.get(key) for key in ('passed','simulation_seconds','loop_wall_seconds','final_goal_distance_m','final_goal_angle_rad','contact_steps','raycast_seconds','sensor_startup_warmup_seconds','sensing_wall_intervals_after_history_warmup_seconds')}
        report['flights'][name].update({key:planner.get(key) for key in ('accepted_plans','guard_interventions','dropped_scans','deadline_misses_100ms','response_seconds','map_seconds')})
        archived=read(artifact/f'{name}.json')['planner']
        for key in ('steady_response_seconds','steady_deadline_misses_100ms','steady_processed_update_intervals_seconds','plan_attempts'):
            report['flights'][name][key]=archived.get(key)
        if 'collision' in doc:report['flights'][name]['collision']=doc['collision']
    (root/'docs/static_sdf_pose_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.spatial.transform import Rotation
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,name in enumerate(('industrial100_final','industrial101_final')):
        doc=read(files[name]);trace=read(files[name].with_name('trajectory.json'));origin=np.array(doc['initial_position'])
        xyz=np.array([r['position'] for r in trace])-origin
        ax=axes[0,col]
        for box in doc['static_sdf_fixture']['boxes'][:2]:
            ax.add_patch(Rectangle((box[0]-box[3]/2,box[1]-box[4]/2),box[3],box[4],color='#d97936',alpha=.7))
        ax.plot(xyz[:,0],xyz[:,1],color='#147d92',lw=2,label='Actual actuator-driven flight')
        ax.scatter([0,doc['goal_position'][0]-origin[0]],[0,0],c=['#444','#349445'],s=45)
        ax.set(xlabel='Local X (m)',ylabel='Local Y (m)',title=f'Static scene {100+col}: actual flight',xlim=(-.4,5),ylim=(-2.4,2.4),aspect='equal')
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
        q=np.array([r['quaternion'] for r in trace]);rpy=Rotation.from_quat(q[:,[1,2,3,0]]).as_euler('xyz')*180/np.pi
        ax=axes[1,col]
        for i,label in enumerate(('Roll','Pitch','Yaw')):ax.plot([r['t'] for r in trace],rpy[:,i],label=label)
        ax.set(xlabel='Simulation time (s)',ylabel='Measured attitude (deg)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Frozen SDF-conditioned DiT — actual industrial physics and measured attitude')
    fig.savefig(root/'docs/static_sdf_pose_flights.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(checkpoint_sha256=report['checkpoint_sha256'],physical=native['summary'],flights={k:v['passed'] for k,v in report['flights'].items()}),indent=2))


if __name__=='__main__':main()
