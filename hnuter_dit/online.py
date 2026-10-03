"""Asynchronous current-LiDAR DiT planning and bounded physical references.

Baseline freezes the first planning scan. Online mode uses measured scans only;
world collision geometry is reserved for the separate physical evaluator.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from .planner_process import PlannerProcess
from .lidar_condition import frame_to_tokens
from .timed_trajectory import TimedTrajectory
from .observation_guard import ObservationGuard


class BrakeReference:
    """C2 velocity decay from the measured state, without teleportation."""
    def __init__(self, state):
        self.position=state['position'].copy()
        self.velocity=state['velocity'].copy()
        self.quaternion=state['quaternion'].copy()
        self.duration=max(.4, 1.875*np.linalg.norm(self.velocity)/.6)

    def sample(self,t):
        s=float(np.clip(t/self.duration,0,1))
        p=self.position+self.velocity*self.duration*(s-2.5*s**4+3*s**5-s**6)
        v=self.velocity*(1-10*s**3+15*s**4-6*s**5)
        a=self.velocity/self.duration*(-30*s**2+60*s**3-30*s**4)
        return dict(pos=p,vel=v,acc=a,quaternion=self.quaternion.copy(),
                    angular_velocity=np.zeros(3),angular_acceleration=np.zeros(3))


class OnlinePlanner:
    def __init__(self, *, mode, goal_position, device='cuda', seed=7, output_dir=None, max_speed=.3, max_acceleration=.4):
        if mode not in ('baseline','online'):
            raise ValueError('mode must be baseline or online')
        self.mode,self.goal,self.seed=mode,np.asarray(goal_position,dtype=float),seed
        if self.goal.shape!=(3,) or not np.isfinite(self.goal).all():
            raise ValueError('goal must be finite xyz')
        if not np.isfinite([max_speed,max_acceleration]).all() or min(max_speed,max_acceleration)<=0:
            raise ValueError('motion limits must be finite and positive')
        self.max_speed=float(max_speed)
        self.max_acceleration=float(max_acceleration)
        self.runtime=PlannerProcess(output_dir=output_dir,device=device)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='dit-planning')
        self.future=None
        self.path=self.brake=None
        self.path_start=self.brake_start=0.
        self.hold=None
        self.observation=self.guard=None
        self.last_request=-np.inf
        self.events=[]
        self.guard_events=[]
        self.planning_seconds=[]
        self.closed=False
        self.raw_paths=[]
        self.last_state=None
        self.failed=False
        self.failure_reason=None
        self.request_wall_time=None
        self.request_expired=False

    def observe(self,frame,state,sim_time):
        self.observation=frame
        self.last_state={k:np.array(v,copy=True) for k,v in state.items()}
        if self.hold is None:
            self.hold={'pos':state['position'].copy(),'quaternion':state['quaternion'].copy()}
        if self.mode=='online':
            self.guard=ObservationGuard(frame)
            if self.path is not None:
                verdict=self._guard_path(self.path,self.path_start,sim_time)
                if not verdict['allowed']:
                    self._brake(state,sim_time,verdict)
        if sim_time<.5 or self.future is not None or getattr(self,'failed',False):
            return
        if self.mode=='baseline' and (self.path is not None or self.events):
            return
        if self.mode=='online' and (sim_time-self.last_request<1. or np.linalg.norm(state['position']-self.goal)<.25):
            return
        self.last_request=float(sim_time)
        snapshot={k:np.array(v,copy=True) for k,v in state.items()}
        self.request_wall_time=time.perf_counter()
        self.request_expired=False
        self.future=self.pool.submit(self._plan,frame,snapshot,float(sim_time))

    def _plan(self,frame,state,sim_time):
        from .path_validation import observed_clearance
        started=time.perf_counter()
        tokens,mask,metadata=frame_to_tokens(frame,state['position'],self.goal)
        start=np.r_[state['position'],state['quaternion']]
        goal=np.r_[self.goal,[1.,0.,0.,0.]]
        paths,model_info=self.runtime.sample(self.mode,tokens,mask,start,goal,seed=self.seed,count=8)
        distances=observed_clearance(paths,tokens[mask])
        lengths=np.linalg.norm(np.diff(paths[:,:,:3],axis=1),axis=2).sum(axis=1)
        eligible=np.isfinite(paths).all(axis=(1,2)) & (distances>.05)
        if self.mode=='online':
            direct=float(np.linalg.norm(goal[:3]-start[:3]))
            eligible &= lengths<=max(2.5*direct,direct+3.)
        selected=None
        reference=None
        reference_clearance=None
        candidate_guards={}
        captured_guard=ObservationGuard(frame) if self.mode=='online' else None
        for index in np.argsort(np.where(eligible,lengths,np.inf)):
            if not eligible[index]:
                break
            try:
                candidate=TimedTrajectory(paths[index],max_speed=getattr(self,'max_speed',.3),max_acceleration=getattr(self,'max_acceleration',.4))
                # Inspect the time-parameterized geometry as well as raw knots.
                count=min(4096,max(32,int(np.ceil(lengths[index]/.04))))
                interpolated=np.array([np.r_[s['pos'],s['quaternion']] for s in
                    [candidate.sample(t) for t in np.linspace(0,candidate.duration,count)]])
                minimum=float(observed_clearance(interpolated[None],tokens[mask])[0])
                if minimum>.05:
                    if captured_guard is not None:
                        verdict=self._guard_path(candidate,sim_time,sim_time,guard=captured_guard)
                        candidate_guards[str(int(index))]=verdict
                        if not verdict['allowed']:
                            continue
                    selected,reference,reference_clearance=int(index),candidate,minimum
                    break
            except ValueError:
                continue
        result=dict(scan_time=sim_time,tokens=metadata,model=model_info,
            planning_seconds=time.perf_counter()-started,candidates=len(paths),
            observed_collision_free=int(eligible.sum()),candidate_clearance_m=distances.tolist(),
            candidate_lengths_m=lengths.tolist(),selected=selected,
            interpolated_observed_clearance_m=reference_clearance,
            candidate_execution_guards=candidate_guards,
            status='candidate' if selected is not None else 'no_executable_candidate')
        return result,reference,None if selected is None else paths[selected]

    def _guard_path(self,path,start,sim_time,guard=None):
        guard=self.guard if guard is None else guard
        if guard is None:
            return dict(allowed=False,reason='missing_scan')
        t0=float(np.clip(sim_time-start,0,path.duration))
        # Inspect a spatial execution horizon, not a near-zero-speed time slice.
        times=np.linspace(t0,path.duration,100)
        poses=[]
        distance=0.
        previous=None
        for t in times:
            state=path.sample(t)
            pose=np.r_[state['pos'],state['quaternion']]
            if previous is not None:
                distance+=np.linalg.norm(pose[:3]-previous[:3])
            poses.append(pose)
            previous=pose
            if distance>=.8:
                break
        from .path_validation import dense_poses
        poses=dense_poses(np.asarray(poses),spacing=.04)
        return guard.check(poses,sim_time=sim_time)

    def _brake(self,state,sim_time,reason):
        self.path=None
        self.brake=BrakeReference(state)
        self.brake_start=float(sim_time)
        self.guard_events.append(dict(time=float(sim_time),**reason))

    def command(self,state,sim_time):
        if self.hold is None:
            self.hold={'pos':state['position'].copy(),'quaternion':state['quaternion'].copy()}
        if self.mode=='online' and self.path is not None and (
                self.observation is None or sim_time-self.observation.sim_time>.3):
            self._brake(state,sim_time,dict(allowed=False,reason='stale_scan'))
        if (self.mode=='online' and self.future is not None and not self.future.done()
                and getattr(self,'request_wall_time',None) is not None
                and time.perf_counter()-self.request_wall_time>2.
                and not getattr(self,'request_expired',False)):
            self.request_expired=True
            self._brake(state,sim_time,dict(allowed=False,reason='planning_deadline_exceeded'))
        if self.future is not None and self.future.done():
            future,self.future=self.future,None
            try:
                result,candidate,poses=future.result()
            except Exception as exc:
                self.failed=True
                self.failure_reason=f'{type(exc).__name__}: {exc}'
                self.events.append(dict(status='planner_failed',activation_time=float(sim_time),
                                        error=self.failure_reason))
                self._brake(state,sim_time,dict(allowed=False,reason='planner_failed',error=self.failure_reason))
                result=None
            if result is not None:
                self._activate_result(result,candidate,poses,state,sim_time)
        if self.path is not None:
            return self.path.sample(sim_time-self.path_start)
        if self.brake is not None:
            desired=self.brake.sample(sim_time-self.brake_start)
            if sim_time-self.brake_start>=self.brake.duration:
                self.hold={'pos':desired['pos'].copy(),'quaternion':desired['quaternion'].copy()}
                self.brake=None
            return desired
        return dict(self.hold,vel=np.zeros(3),acc=np.zeros(3))

    def _activate_result(self,result,candidate,poses,state,sim_time):
        result['activation_time']=float(sim_time)
        result['observation_age_sim_seconds']=float(sim_time-result['scan_time'])
        self.planning_seconds.append(result['planning_seconds'])
        if candidate is not None:
            drift=float(np.linalg.norm(state['position']-poses[0,:3]))
            result['start_drift_m']=drift
            current_q=np.asarray(state['quaternion'])
            old_q=np.asarray(poses[0,3:])
            angle=2.*np.arccos(np.clip(abs(np.dot(current_q,old_q))/(np.linalg.norm(current_q)*np.linalg.norm(old_q)),0.,1.))
            result['start_attitude_drift_rad']=float(angle)
            if self.mode=='online' and self.path is not None:
                result['status']='kept_valid_trajectory'
            elif drift>.05 or angle>.1 or (self.mode=='online' and (getattr(self,'request_expired',False) or sim_time-result['scan_time']>.5 or result['planning_seconds']>2.)):
                result['status']='stale_plan_rejected'
            elif self.mode=='online' and np.linalg.norm(state['velocity'])>.08:
                result['status']='wait_for_braking'
            else:
                verdict=(self._guard_path(candidate,float(sim_time),sim_time) if self.mode=='online'
                         else dict(allowed=True,reason='static_baseline_observed_geometry_only'))
                result['execution_guard']=verdict
                if verdict['allowed']:
                    self.path=candidate
                    self.brake=None
                    self.path_start=float(sim_time)
                    self.raw_paths.append(poses.tolist())
                    result['status']='accepted'
                    result['trajectory_duration']=self.path.duration
                else:
                    result['status']='prefix_rejected'
        self.events.append(result)

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True,cancel_futures=True)
            if self.future is not None and not self.future.cancelled():
                try:
                    result,_,_=self.future.result()
                    result['status']='not_executed_after_run'
                    self.events.append(result)
                    self.planning_seconds.append(result['planning_seconds'])
                except Exception as exc:
                    self.failed=True
                    self.failure_reason=f'{type(exc).__name__}: {exc}'
                    self.events.append(dict(status='planner_failed',error=self.failure_reason))
                self.future=None
            self.runtime.close()
            self.closed=True
        values=np.asarray(self.planning_seconds)
        return dict(mode=self.mode,motion_limits=dict(max_speed_mps=self.max_speed,max_acceleration_mps2=self.max_acceleration,braking_acceleration_mps2=.6),failed=self.failed,failure_reason=self.failure_reason,checkpoint=self.runtime.describe(),plans=self.events,guard_events=self.guard_events,
                    accepted_plans=sum(e['status']=='accepted' for e in self.events),
                    planning_seconds_mean=float(values.mean()) if len(values) else None,
                    planning_seconds_max=float(values.max()) if len(values) else None,
                    geometric_paths=self.raw_paths,online_weights_retrained=False,
                    unknown_space_policy='Online: valid-return sensor-centerline evidence required outside current 0.1m neighborhood; no-return directions remain unknown. Observed 42-sphere clearance is a limited guard, not full-volume safety certification. Baseline: frozen observed geometry only.')
