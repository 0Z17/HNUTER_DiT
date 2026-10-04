#!/usr/bin/env python3
"""Bind factory demo frames and measured perception to the physical run."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from export_flight_video import digest,validate_manifest,validate_flight_identity,locate_ffmpeg


def validate(run):
    run=Path(run)
    frames=validate_manifest(run/'video/manifest.json')
    report=validate_flight_identity(run/'video/manifest.json',frames,run/'report.json')
    side=json.loads((run/'perception_video/manifest.json').read_text())
    second=validate_manifest(run/f"perception_video/{side['camera']['camera']}/manifest.json")
    rows=json.loads((run/'trajectory.json').read_text())
    by_step={round(r['t']/report['physics_dt']):r for r in rows}
    by_step[report['steps']]=dict(position=report['final_position'],quaternion=report['final_quaternion'])
    if not len(frames['frames'])==len(second['frames'])==len(side['frames']):raise ValueError('capture count mismatch')
    for a,b,c in zip(frames['frames'],second['frames'],side['frames']):
        step=a['physics_step'];stamp=a['simulation_time'];row=by_step[step]
        if not step==b['physics_step']==c['physics_step']:raise ValueError('capture step mismatch')
        if not np.array_equal(a['position'],row['position']) or not np.array_equal(a['quaternion_wxyz'],row['quaternion']):
            raise ValueError('camera telemetry differs from physical log')
        if a['position']!=b['position'] or a['quaternion_wxyz']!=b['quaternion_wxyz']:raise ValueError('cameras at different states')
        if c['scan'] is not None and c['scan']['time']>stamp+1e-8:raise ValueError('future LiDAR')
        if c['map_scan_time'] is not None and c['map_scan_time']>stamp+1e-8:raise ValueError('future map')
    positions=np.array([r['position'] for r in rows]+[report['final_position']])
    planner=report['planner'];accepted=[e for e in planner['plans'] if e['status']=='accepted']
    if not report['passed'] or not planner['mission_completed'] or len(accepted)!=len(report['factory_mission']['waypoints']):
        raise ValueError('factory mission did not complete successfully')
    if report['collision']['collision_detected'] or not report['collision']['complete'] or planner['guard_interventions']:
        raise ValueError('collision or intervention in completed mission')
    return dict(schema='hnuter_factory_video_qa_v1',passed=True,report_sha256=digest(run/'report.json'),
        manifest_sha256=digest(run/'video/manifest.json'),checked_frames=len(frames['frames']),
        steps=report['steps'],simulation_seconds=report['simulation_seconds'],
        measured_distance_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
        accepted_legs=len(accepted),rejected_attempts=sum(e['status']=='rejected' for e in planner['plans']),
        segment_activation_times=[e['activation_time'] for e in accepted],
        final_goal_distance_m=report['final_goal_distance_m'],final_goal_angle_rad=report['final_goal_angle_rad'],
        recording_perception_wait_count=len(report.get('recording_perception_barrier',{}).get('waits',[])),
        recording_perception_wait_seconds=sum(w['wall_seconds'] for w in report.get('recording_perception_barrier',{}).get('waits',[])),
        checkpoint_sha256=planner['checkpoint']['checkpoint_sha256'],
        checks='Every camera state equals the physical log exactly; both cameras synchronized; perception timestamps causal; complete physical contact checks; all mission legs accepted without execution guard intervention.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mp4',type=Path);parser.add_argument('--ffmpeg')
    args=parser.parse_args();result=validate(args.run)
    if args.mp4:
        decoded=subprocess.run([locate_ffmpeg(args.ffmpeg),'-v','error','-nostats','-progress','pipe:1',
            '-i',str(args.mp4),'-map','0:v:0','-f','null','-'],capture_output=True,text=True,check=True)
        counts=[int(line.split('=',1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
        if not counts or counts[-1]!=result['checked_frames']:raise ValueError('decoded video frame count mismatch')
        result['mp4']=dict(sha256=digest(args.mp4),decoded_frames=counts[-1],decode_exit_code=decoded.returncode)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
