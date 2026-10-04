"""Colocated nvblox mapping and learned pose inference; maps never cross IPC."""
from __future__ import annotations
import hashlib
import time
from pathlib import Path
import numpy as np
from .sdf_mapping import LocalSDF


class SDFRuntime:
    def __init__(self,checkpoint,sim_root='/sim'):
        self.map=LocalSDF(sim_root)
        from .sdf_model import load_model,sample
        from .sdf_pose import PoseSpline,full_controls
        self.model,self.payload=load_model(checkpoint); self.sampler=sample
        self.PoseSpline,self.full_controls=PoseSpline,full_controls
        self.torch=self.map.torch;self.torch.set_num_threads(4)
        self.checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.path=None
        # Warm network kernels without touching real map history.
        fields=self.torch.zeros((1,4,6,32,32,32),device='cuda')
        ages=self.torch.tensor([[-.6,-.4,-.2,0]],device='cuda')
        target=self.torch.tensor([[4.4,0,0,.1,-.1,.2]],device='cuda')
        sample(self.model,fields,ages,target);self.torch.cuda.synchronize()

    def reset(self,origin,start_rotation,target):
        self.origin=np.array(origin);self.rotation=np.array(start_rotation);self.target=np.array(target)
        if self.target.shape!=(6,) or not np.isfinite(self.target).all():raise ValueError('invalid pose target')
        if np.any(np.abs(self.target[3:])>.7):raise ValueError('pose target exceeds pilot angular support')
        self.map.reset(origin);self.path=None

    def scan(self,points,pose,stamp,plan=False,seed=23,elapsed=0.):
        start=time.perf_counter();metadata=self.map.integrate(points,pose,stamp)
        controls=np.zeros((0,6))
        if plan:
            fields,ages=self.map.inputs();self.torch.cuda.synchronize();begin=time.perf_counter()
            target=self.torch.as_tensor(self.target,dtype=self.torch.float32,device='cuda')[None]
            free=self.sampler(self.model,fields[None],ages[None],target,seed=seed).cpu().numpy()[0]
            self.torch.cuda.synchronize();metadata['inference_seconds']=time.perf_counter()-begin
            controls=self.full_controls(free,self.target)
            self.path=self.PoseSpline(controls,self.origin,start_rotation=self.rotation)
            audit=self.path.audit();metadata['trajectory_audit']=audit
            metadata['feasible']=bool(audit['speed_bound_mps']<=.65 and audit['acceleration_bound_mps2']<=.9 and audit['max_local_angle_rad']<=.7)
            first_clearance=self.clearance(self.path,0.)
            candidates=[dict(index=0,feasible=metadata['feasible'],observed_clearance=first_clearance,controls=controls.tolist())]
            selected=0
            if not metadata['feasible'] or first_clearance['observed_violation'] or not first_clearance['minimum_observed_support']:
                # Additional candidates are learned samples, not geometric
                # edits or expert routes. Only measured map queries select.
                begin=time.perf_counter()
                alternatives=self.sampler(self.model,fields[None].expand(4,-1,-1,-1,-1,-1),
                    ages[None].expand(4,-1),target.expand(4,-1),seed=seed+1).cpu().numpy()
                self.torch.cuda.synchronize();metadata['resample_inference_seconds']=time.perf_counter()-begin
                best=None
                for i,alternative in enumerate(alternatives,1):
                    candidate_controls=self.full_controls(alternative,self.target)
                    candidate=self.PoseSpline(candidate_controls,self.origin,start_rotation=self.rotation)
                    candidate_audit=candidate.audit();clearance=self.clearance(candidate,0.)
                    feasible=bool(candidate_audit['speed_bound_mps']<=.65 and candidate_audit['acceleration_bound_mps2']<=.9 and candidate_audit['max_local_angle_rad']<=.7)
                    candidates.append(dict(index=i,feasible=feasible,observed_clearance=clearance,controls=candidate_controls.tolist()))
                    minimum=clearance['minimum_known_m']
                    if feasible and minimum is not None and minimum>=.05 and clearance['known_fraction']>=.05 and (best is None or minimum>best[0]):
                        best=(minimum,i,candidate,candidate_controls,candidate_audit)
                if best is not None:
                    _,selected,self.path,controls,audit=best
                    metadata['trajectory_audit']=audit;metadata['feasible']=True
            metadata['candidate_selection']=dict(selected=selected,candidates=candidates,
                method='one DiT sample; if infeasible or observed-collision, sample four more; select by observed ESDF clearance, no path repair')
        if self.path is not None:
            metadata['observed_clearance']=self.clearance(self.path,float(elapsed))
        self.torch.cuda.synchronize();metadata['worker_wall_seconds']=time.perf_counter()-start
        return controls,metadata

    def clearance(self,path,elapsed):
        from scipy.spatial.transform import Rotation
        from .collision_loss import ROBOT_SPHERE_CENTERS,ROBOT_SPHERE_RADII
        t=self.torch
        values=path.sample(np.linspace(np.clip(elapsed,0,path.duration),path.duration,81))
        rotations=Rotation.from_quat(values['quaternion'][:,[1,2,3,0]]).as_matrix()
        centers=values['pos'][:,None]+np.einsum('tij,sj->tsi',rotations,np.array(ROBOT_SPHERE_CENTERS))
        queries=t.as_tensor(centers.reshape(-1,3),device='cuda',dtype=t.float32)
        spheres=t.cat((queries,t.zeros((len(queries),1),device='cuda')),dim=-1)
        out=t.full((len(queries),1),self.map.native.unknown,device='cuda')
        distances=self.map.native.mapper.query_esdf(out,spheres,0)[:,0]
        known=t.isfinite(distances)&((distances-self.map.native.unknown).abs()>1e-4)
        radii=t.tensor(ROBOT_SPHERE_RADII,device='cuda').repeat(len(values['pos']))
        clearance=distances-radii
        minimum=float(clearance[known].min().item()) if bool(known.any()) else None
        return dict(minimum_known_m=minimum,known_fraction=float(known.float().mean().item()),
                    observed_violation=bool(minimum is not None and minimum<.05),
                    minimum_observed_support=bool(minimum is not None and float(known.float().mean().item())>=.05),
                    collision_certified=False,unknown_policy='unknown is not free; partial observed guard only')

    def describe(self):
        return dict(schema=self.payload['schema'],checkpoint_sha256=self.checkpoint_sha256,
            selected_step=self.payload['step'],mapper=self.map.describe(),pose_output=True,
            input_scope='measured historical scans, measured start pose and requested goal; no obstacle truth')
