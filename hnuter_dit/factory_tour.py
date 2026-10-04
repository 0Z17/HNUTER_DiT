"""Stationary-start local DiT plans connected by surveyed mission goals.

Only the active next goal enters the local policy. Each leg uses a rigid yaw
coordinate change and fresh history queried from a bounded world GPU map. Robot state is
never written. There is no geometric path repair or fallback flight reference.
"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .sdf_online import SDFOnlinePlanner,SDFProcess


def load_mission(path):
    mission=json.loads(Path(path).read_text())
    points=np.asarray([mission['start_position'],*mission['waypoints']],dtype=float)
    yaws=np.asarray(mission['goal_yaws'],dtype=float)
    if mission.get('schema')!='hnuter_factory_mission_v1' or points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or yaws.shape!=(len(points)-1,):
        raise ValueError('Invalid factory mission')
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    if not np.isfinite(points).all() or not np.isfinite(yaws).all() or np.any((lengths<4.1)|(lengths>4.7)):
        raise ValueError('Factory legs must stay near the learned 4.2–4.6 m curriculum')
    if np.max(np.abs(np.diff(np.r_[0,yaws])))>.35+1e-6:raise ValueError('Excessive mission yaw change')
    return mission


def local_state(state,origin,frame):
    r=frame.T@state['rotation_matrix']
    return dict(state,position=(state['position']-origin)@frame,velocity=state['velocity']@frame,
        rotation_matrix=r,quaternion=Rotation.from_matrix(r).as_quat()[[3,0,1,2]],
        euler=Rotation.from_matrix(r).as_euler('xyz'))


def world_command(command,origin,frame):
    result=dict(command)
    for key in ('pos','vel','acc'):
        result[key]=np.asarray(command[key])@frame.T+(origin if key=='pos' else 0)
    r=Rotation.from_quat(np.asarray(command['quaternion'])[[1,2,3,0]]).as_matrix()
    result['quaternion']=Rotation.from_matrix(frame@r).as_quat()[[3,0,1,2]]
    # Angular velocity/acceleration remain expressed in the same physical body.
    return result


class FactorySDFPlanner:
    def __init__(self,*,mission,checkpoint,goal_position=None,goal_rpy=None,seed=23,output_dir=None,
                 device='cuda',max_speed=.65,max_acceleration=.9):
        self.mission=mission;self.waypoints=np.asarray(mission['waypoints']);self.yaws=mission['goal_yaws']
        self.runtime=SDFProcess(output_dir=output_dir,temporal_checkpoint=checkpoint,device=device,request_timeout=10.)
        self.options=dict(checkpoint=checkpoint,seed=seed,device=device,max_speed=max_speed,max_acceleration=max_acceleration,
            injected_runtime=self.runtime)
        self.active=None;self.segment=-1;self.record_visualization=False
        self.events=[];self.raw_paths=[];self.scans=[];self.legs=[]
        self.failed=False;self.failure_reason=None;self.completed_time=None;self.closed=False
        self.last_command_category='mission_initializing';self.settle_start=None;self.recorded_scans=self.recorded_events=0
        self.guard_interventions=0

    def _start(self,state,stamp):
        self.segment+=1;self.segment_start=float(stamp);self.world_origin=state['position'].copy()
        direction=self.waypoints[self.segment]-self.world_origin
        self.world_frame=Rotation.from_euler('z',np.arctan2(direction[1],direction[0])).as_matrix()
        target=direction@self.world_frame
        r=self.world_frame.T@Rotation.from_euler('z',self.yaws[self.segment]).as_matrix()
        context=dict(map_world_origin=self.world_origin,map_world_frame=self.world_frame,map_global_start=np.float64(stamp))
        self.active=SDFOnlinePlanner(goal_position=target,goal_rpy=Rotation.from_matrix(r).as_euler('xyz'),map_context=context,**self.options)
        self.active.record_visualization=self.record_visualization
        self.recorded_scans=self.recorded_events=0;self.settle_start=None
        self.events.append(dict(status='segment_started',segment=self.segment,sim_time=stamp,
            measured_origin=self.world_origin.tolist(),goal=self.waypoints[self.segment].tolist(),world_frame=self.world_frame.tolist()))

    def _collect(self):
        child=self.active
        context=dict(segment=self.segment,world_origin=self.world_origin.tolist(),world_frame=self.world_frame.tolist(),segment_start=self.segment_start)
        for scan in child.scans[self.recorded_scans:]:
            self.scans.append(dict(scan,**context,map_local_scan_time=scan['scan_time'],
                scan_time=scan['scan_time']+self.segment_start,response_time=scan['response_time']+self.segment_start))
        self.recorded_scans=len(child.scans)
        for event in child.events[self.recorded_events:]:
            entry=dict(event,**context)
            if 'activation_time' in entry:entry['activation_time']+=self.segment_start
            if 'scan_time' in entry:entry['scan_time']+=self.segment_start
            if 'sim_time' in entry:entry['sim_time']+=self.segment_start
            if child.origin is not None:entry['local_origin']=child.origin.tolist();entry['start_rotation']=child.rotation.tolist()
            self.events.append(entry)
            if event['status']=='accepted':self.raw_paths.append(entry)
        self.recorded_events=len(child.events)
        self.guard_interventions=sum(x['guard_interventions'] for x in self.legs)+child.guard_interventions
        if child.failed or child.guard_interventions or (child.planned and not child.raw_paths):
            self.failed=True;self.failure_reason=child.failure_reason or 'local plan rejected or stopped; no fallback'

    def observe(self,frame,state,sim_time,capture_wall_time=None):
        if self.active is None:self._start(state,sim_time)
        if self.failed:return
        if self.completed_time is None and self.active.last_command_category=='pose_hold' and self.active.raw_paths:
            distance=float(np.linalg.norm(state['position']-self.waypoints[self.segment]))
            angle=float((Rotation.from_euler('z',self.yaws[self.segment]).inv()*Rotation.from_matrix(state['rotation_matrix'])).magnitude())
            if distance<.12 and angle<.08 and np.linalg.norm(state['velocity'])<.03:
                if self.settle_start is None:self.settle_start=sim_time
                if sim_time-self.settle_start>=.5-1e-8:
                    if self.segment+1==len(self.waypoints):self.completed_time=sim_time
                    else:
                        self.legs.append(self.active.close());self._start(state,sim_time)
            else:self.settle_start=None
        pose=np.eye(4);pose[:3,:3]=self.world_frame.T@frame.world_T_sensor[:3,:3]
        pose[:3,3]=(frame.world_T_sensor[:3,3]-self.world_origin)@self.world_frame
        local=local_state(state,self.world_origin,self.world_frame)
        transformed=replace(frame,world_T_sensor=pose,sim_time=sim_time-self.segment_start)
        self.active.observe(transformed,local,sim_time-self.segment_start,capture_wall_time=capture_wall_time)
        self._collect()

    def command(self,state,sim_time):
        if self.active is None:raise RuntimeError('Mission requires a measured initial scan')
        local=local_state(state,self.world_origin,self.world_frame)
        command=self.active.command(local,sim_time-self.segment_start)
        self._collect()
        self.last_command_category=f'leg {self.segment+1}/{len(self.waypoints)}: {self.active.last_command_category}'
        return world_command(command,self.world_origin,self.world_frame)

    def recording_refresh_required(self):
        return bool(self.active is not None and self.active.recording_refresh_required())

    def wait_for_recorded_observation(self,frame,state,sim_time,capture_wall_time):
        pose=np.eye(4);pose[:3,:3]=self.world_frame.T@frame.world_T_sensor[:3,:3]
        pose[:3,3]=(frame.world_T_sensor[:3,3]-self.world_origin)@self.world_frame
        local=local_state(state,self.world_origin,self.world_frame)
        transformed=replace(frame,world_T_sensor=pose,sim_time=sim_time-self.segment_start)
        self.active.wait_for_recorded_observation(transformed,local,sim_time-self.segment_start,capture_wall_time)
        self._collect()

    def close(self):
        if not self.closed:
            if self.active is not None:self.legs.append(self.active.close())
            self.runtime.close();self.closed=True
            for leg in self.legs:
                if leg['failed']:
                    self.failed=True;self.failure_reason=leg['failure_reason'] or 'local planner failed during shutdown'
        return dict(mode='factory_sdf',failed=self.failed,failure_reason=self.failure_reason,checkpoint=self.runtime.describe(),
            mission=self.mission,mission_completed=self.completed_time is not None,completed_time=self.completed_time,
            legs=self.legs,plans=self.events,accepted_plans=len(self.raw_paths),guard_interventions=self.guard_interventions,
            scans=self.scans,position_only=False,executed_model_reference=bool(self.raw_paths),
            contract='Surveyed mission goals; bounded measured world nvblox map, fresh local history and frozen DiT per 20-second stationary-start leg. Rigid coordinate changes only; no local path repair or expert fallback.')
