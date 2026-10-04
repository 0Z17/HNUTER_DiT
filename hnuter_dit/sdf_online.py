"""Asynchronous measured-scan GPU mapping and pose trajectory execution.

This static pilot requests one full-horizon plan from a measured stationary
anchor. It keeps sensing and checking observed clearance during execution.
It does not claim moving-start replanning or unknown-space certification.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .planner_process import PlannerProcess,_worker_environment
from .sdf_pose import PoseSpline,HORIZON
from .online import BrakeReference


class SDFProcess(PlannerProcess):
    _worker_script=Path(__file__).with_name('sdf_worker.py')

    @staticmethod
    def worker_environment():
        env=_worker_environment()
        env['LD_LIBRARY_PATH']=os.environ.get('NVBLOX_LIBRARY_PATH','/opt/nvblox-venv/lib/python3.12/site-packages/nvidia/npp/lib')
        return env

    def scan(self,payload):return self._request(payload,('controls',))


class SDFOnlinePlanner:
    def __init__(self,*,goal_position,checkpoint,goal_rpy=(.08,-.06,.25),seed=23,output_dir=None,
                 device='cuda',max_speed=.65,max_acceleration=.9,injected_runtime=None,map_context=None):
        self.goal=np.asarray(goal_position);self.goal_rpy=np.asarray(goal_rpy)
        if self.goal.shape!=(3,) or self.goal_rpy.shape!=(3,) or not np.isfinite(np.r_[self.goal,self.goal_rpy]).all():raise ValueError('invalid pose goal')
        self.owns_runtime=injected_runtime is None
        self.map_context={} if map_context is None else dict(map_context)
        self.runtime=injected_runtime if injected_runtime is not None else SDFProcess(output_dir=output_dir,temporal_checkpoint=checkpoint,device=device,request_timeout=10.)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='sdf-mapping')
        self.seed=seed;self.max_speed=max_speed;self.max_acceleration=max_acceleration
        self.future=self.path=self.brake=self.hold=None
        self.origin=self.rotation=self.target=None
        self.submitted=0;self.dropped=0;self.events=[];self.raw_paths=[];self.scans=[]
        self.failed=False;self.failure_reason=None;self.closed=False;self.planned=False;self.path_start=0.
        self.last_command_category='history_warmup';self.pending=None;self.guard_interventions=0
        self.last_map_time=None;self.last_map_wall=None;self.deadline_misses=0
        self.plan_attempts=0;self.last_plan_request=-np.inf
        self.record_visualization=False

    def _stop(self,state,sim_time,reason):
        if self.brake is None:
            self.brake=BrakeReference(state);self.brake_start=sim_time
        self.path=None;self.planned=True
        self.events.append(dict(status='execution_stopped',reason=reason,sim_time=sim_time))

    def observe(self,frame,state,sim_time,capture_wall_time=None):
        if not np.isfinite(sim_time) or abs(float(frame.sim_time)-sim_time)>1e-6:
            raise ValueError('scan and physics timestamps must match')
        if not all(np.isfinite(state[key]).all() for key in ('position','velocity','rotation_matrix','quaternion')):
            raise ValueError('nonfinite measured state')
        if self.hold is None:
            self.hold=dict(pos=state['position'].copy(),quaternion=state['quaternion'].copy(),vel=np.zeros(3),acc=np.zeros(3))
            self.origin=state['position'].copy();self.rotation=state['rotation_matrix'].copy()
            angles=Rotation.from_matrix(self.rotation.T@Rotation.from_euler('xyz',self.goal_rpy).as_matrix()).as_euler('xyz')
            self.target=np.r_[self.goal-self.origin,angles]
        self._poll(state,sim_time)
        if self.failed:return
        if self.future is not None:self.dropped+=1;return
        plan=bool(not self.planned and self.plan_attempts<3 and sim_time>=.6-1e-6 and sim_time-self.last_plan_request>=1.-1e-6
                  and self.scans and self.scans[-1].get('history_ready',False))
        if plan and (np.linalg.norm(state['velocity'])>.03 or np.linalg.norm(state['position']-self.origin)>.03):
            self.failed=True;self.failure_reason='stationary start contract exceeded';self._stop(state,sim_time,self.failure_reason);return
        if plan:
            self.plan_attempts+=1;self.last_plan_request=sim_time
        payload=dict(points=np.array(frame.points),pose=np.array(frame.world_T_sensor),stamp=np.float64(sim_time),
            reset=np.bool_(self.submitted==0),origin=self.origin,start_rotation=self.rotation,target=self.target,
            plan=np.bool_(plan),seed=np.int64(self.seed),elapsed=np.float64(max(0,sim_time-self.path_start) if self.path else 0))
        if self.record_visualization:payload["record_visualization"]=np.bool_(True)
        payload.update(self.map_context)
        began=time.perf_counter() if capture_wall_time is None else float(capture_wall_time)
        if not np.isfinite(began) or began>time.perf_counter():raise ValueError('invalid wall-clock scan timestamp')
        self.pending=dict(stamp=sim_time,plan=plan,started=began)
        self.future=self.pool.submit(self.runtime.scan,payload);self.submitted+=1

    def _poll(self,state,sim_time):
        if self.future is None or not self.future.done():return
        pending=self.pending;future=self.future;self.future=self.pending=None
        try:
            arrays,metadata=future.result();elapsed=time.perf_counter()-pending['started']
            self.deadline_misses+=int(elapsed>.1)
            self.last_map_time=pending['stamp'];self.last_map_wall=pending['started']
            metadata.update(scan_time=pending['stamp'],response_time=sim_time,observation_age_sim_seconds=sim_time-pending['stamp'],
                            sensor_to_response_wall_seconds=elapsed,plan_requested=pending['plan'])
            self.scans.append(metadata)
            observed=metadata.get('observed_clearance',{})
            if pending['plan']:
                self.planned=self.plan_attempts>=3
                path=PoseSpline(arrays['controls'],self.origin,start_rotation=self.rotation)
                audit=path.audit()
                fresh=sim_time-pending['stamp']<=.3 and np.linalg.norm(state['position']-self.origin)<=.03 and elapsed<=.3
                feasible=metadata.get('feasible') and audit['speed_bound_mps']<=self.max_speed and audit['acceleration_bound_mps2']<=self.max_acceleration
                accepted=bool(fresh and feasible and observed.get('minimum_observed_support',False) and not observed.get('observed_violation',False))
                event=dict(status='accepted' if accepted else 'rejected',scan_time=pending['stamp'],activation_time=sim_time,
                    fresh=bool(fresh),feasible=bool(feasible),observed_clearance=observed,controls=arrays['controls'].tolist())
                self.events.append(event)
                if accepted:
                    self.planned=True;self.path=path;self.path_start=sim_time;self.raw_paths.append(event)
            elif self.path is not None and observed.get('observed_violation',False):
                self.guard_interventions+=1;self._stop(state,sim_time,'observed_sdf_clearance')
        except Exception as exc:
            self.failed=True;self.failure_reason=f'{type(exc).__name__}: {exc}'
            self._stop(state,sim_time,self.failure_reason)

    def command(self,state,sim_time):
        self._poll(state,sim_time)
        if self.path is not None and (self.last_map_time is None or sim_time-self.last_map_time>.4 or time.perf_counter()-self.last_map_wall>.4):
            self.guard_interventions+=1;self._stop(state,sim_time,'stale_perception')
        if self.path is not None:
            if sim_time-self.path_start<=HORIZON:
                self.last_command_category='sdf_pose_reference';return self.path.sample(sim_time-self.path_start)
            self.hold=self.path.sample(HORIZON);self.path=None
        if self.brake is not None:
            self.last_command_category='braking';desired=self.brake.sample(sim_time-self.brake_start)
            if sim_time-self.brake_start>=self.brake.duration:self.hold=desired;self.brake=None
            return desired
        self.last_command_category='pose_hold' if self.planned else 'history_warmup'
        return self.hold

    def recording_refresh_required(self):
        return bool(self.path is not None and
            (self.last_map_wall is None or time.perf_counter()-self.last_map_wall>.25))

    def wait_for_recorded_observation(self,frame,state,sim_time,capture_wall_time):
        """Offline recording barrier: caller holds physics fixed throughout.

        Accept only a newly captured current-state map. This never relabels a
        previous scan or weakens the normal command-time freshness guard.
        """
        deadline=time.perf_counter()+.3
        while time.perf_counter()<deadline:
            self._poll(state,sim_time)
            if self.failed:raise RuntimeError(self.failure_reason)
            if self.last_map_time is not None and abs(self.last_map_time-sim_time)<1e-8:
                if time.perf_counter()-self.last_map_wall<=.3:return
                raise RuntimeError('current-state observation already exceeded recording freshness deadline')
            if self.future is None:self.observe(frame,state,sim_time,capture_wall_time=capture_wall_time)
            time.sleep(.001)
        raise RuntimeError('recording barrier did not obtain a fresh current-state observation within 300 ms')

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True,cancel_futures=True)
            if self.future is not None and not self.future.cancelled():
                try:self.future.result()
                except Exception as exc:self.failed=True;self.failure_reason=str(exc)
            if self.owns_runtime:self.runtime.close()
            self.closed=True
        durations=np.array([r['sensor_to_response_wall_seconds'] for r in self.scans])
        mapping=np.array([r['map_seconds'] for r in self.scans])
        summary=lambda x:dict(median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(x.max())) if len(x) else None
        return dict(mode='sdf_pose',failed=self.failed,failure_reason=self.failure_reason,checkpoint=self.runtime.describe(),
            plans=self.events,accepted_plans=len(self.raw_paths),guard_interventions=self.guard_interventions,
            scans=self.scans,submitted_scans=self.submitted,dropped_scans=self.dropped,deadline_misses_100ms=self.deadline_misses,
            plan_attempts=self.plan_attempts,max_stationary_plan_attempts=3,
            response_seconds=summary(durations),map_seconds=summary(mapping),goal_rpy=self.goal_rpy.tolist(),
            executed_model_reference=bool(self.raw_paths),position_only=False,
            contract='static environment; stationary-start full 20s position and attitude spline; continued measured-map monitoring; no geometric path repair',
            safety_scope='Observed ESDF only, unknown space is not certified free; actual contacts evaluated independently')
