"""Actual pinned nvblox CUDA local distance fields; no synthetic box rasterizer."""
from __future__ import annotations
from collections import deque
from pathlib import Path
import sys
import time
import numpy as np

GRID=32
CELL=.25
CHANNELS=('tsdf_clipped','weight_log','esdf_clipped','esdf_known','tsdf_known','latest_hits')
CONFIG=dict(voxel_size=.15,max_range=20.,esdf_max_distance=5.,width=720,height=128,vertical_fov_degrees=104.)


class LocalSDF:
    """One bounded planning-anchor map with four measured snapshots on GPU.

Static pilot flights stay inside a fixed 8m local window for <=30s. Explicitly
reset before a new episode; no permanent global SLAM or moving-object model.
"""
    def __init__(self,sim_root='/sim'):
        sys.path.insert(0,str(Path(sim_root)/'perception/nvblox'))
        from worker import NativeMapper
        from projection import spherical_range_image
        self.native=NativeMapper(CONFIG); self.project=spherical_range_image
        self.torch=self.native.torch; self.history=deque(maxlen=4)
        self.origin=None; self.frame=None; self.last_time=None; self.first_time=None
        self.integration_times=[]
        t=self.torch
        axis=(t.arange(GRID,device='cuda')+.5)*CELL-4
        self.local=t.stack(t.meshgrid(axis+2,axis,axis,indexing='ij'),dim=-1).reshape(-1,3)

    def reset(self,origin,frame=None):
        self.origin=np.asarray(origin,dtype=np.float64)
        self.frame=np.eye(3) if frame is None else np.asarray(frame,dtype=np.float64)
        if self.origin.shape!=(3,) or not np.isfinite(self.origin).all() or self.frame.shape!=(3,3) or not np.allclose(self.frame.T@self.frame,np.eye(3),atol=1e-6):
            raise ValueError('invalid map anchor')
        t=self.torch; self.native.clear();self.history.clear();self.last_time=self.first_time=None
        self.queries=self.local@t.as_tensor(self.frame.T,device='cuda',dtype=t.float32)+t.as_tensor(self.origin,device='cuda',dtype=t.float32)
        self.spheres=t.cat((self.queries,t.zeros((len(self.queries),1),device='cuda')),dim=-1)
        self.tsdf=t.zeros((len(self.queries),2),device='cuda')
        self.esdf=t.full((len(self.queries),1),self.native.unknown,device='cuda')

    def integrate(self,points,pose,stamp):
        if self.origin is None:raise ValueError('map needs explicit planning anchor')
        points=np.asarray(points);pose=np.asarray(pose);stamp=float(stamp)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.isfinite(stamp):
            raise ValueError('invalid measured scan')
        if not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-5):
            raise ValueError('invalid sensor rotation')
        if self.last_time is not None and stamp<=self.last_time:raise ValueError('scan time must increase')
        if self.first_time is not None and stamp-self.first_time>30:raise ValueError('static pilot map lifetime exceeded')
        t=self.torch;t.cuda.synchronize();began=time.perf_counter()
        depth=self.project(points,width=720,height=128,vertical_fov_degrees=104.,max_range=20.)
        self.native.mapper.integrate_depth(t.as_tensor(depth,device='cuda'),t.as_tensor(np.ascontiguousarray(pose),dtype=t.float32),self.native.sensor,None,0)
        self.native.mapper.update_esdf(0)
        self.tsdf=self.native.mapper.query_tsdf(self.tsdf,self.queries,0)
        self.esdf=self.native.mapper.query_esdf(self.esdf,self.spheres,0)
        known=t.isfinite(self.esdf[:,0])&((self.esdf[:,0]-self.native.unknown).abs()>1e-4)
        observed=self.tsdf[:,1]>0
        # Same real measured returns provide a sparse surface channel.
        world=t.as_tensor(points,dtype=t.float32,device='cuda')@t.as_tensor(pose[:3,:3].T,dtype=t.float32,device='cuda')+t.as_tensor(pose[:3,3],dtype=t.float32,device='cuda')
        local=(world-t.as_tensor(self.origin,dtype=t.float32,device='cuda'))@t.as_tensor(self.frame,dtype=t.float32,device='cuda')
        ijk=((local-t.tensor([-2.,-4.,-4.],device='cuda'))/CELL).floor().long()
        valid=((ijk>=0)&(ijk<GRID)).all(dim=-1);ijk=ijk[valid]
        hits=t.zeros(GRID**3,device='cuda'); hits[ijk[:,0]*GRID**2+ijk[:,1]*GRID+ijk[:,2]]=1
        features=t.stack((t.where(observed,self.tsdf[:,0].clamp(-.75,.75)/.75,0),
            t.log1p(self.tsdf[:,1].clamp(0,10))/np.log(11),
            t.where(known,self.esdf[:,0].clamp(-3,3)/3,0),known.float(),observed.float(),hits))
        features=features.reshape(len(CHANNELS),GRID,GRID,GRID)
        if not bool(t.isfinite(features).all()):raise RuntimeError('nonfinite native SDF')
        self.latest_features=features
        if not self.history or stamp-self.history[-1][0]>=.2-1e-5:
            self.history.append((stamp,features))
        self.last_time=stamp
        if self.first_time is None:self.first_time=stamp
        t.cuda.synchronize();seconds=time.perf_counter()-began;self.integration_times.append(seconds)
        return dict(map_seconds=seconds,known_fraction=float(known.float().mean().item()),scan_time=stamp)

    def inputs(self):
        if len(self.history)!=4:raise ValueError('four causal SDF snapshots required')
        # Actual asynchronous timestamps are retained; never relabel an old
        # measurement as current. Include the newest integrated observation.
        snapshots=list(self.history)
        snapshots[-1]=(self.last_time,self.latest_features)
        times=np.array([x[0] for x in snapshots]);ages=times-self.last_time
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0) or np.any(np.diff(times)>.4+1e-6) or -ages[0]>1.:
            raise ValueError('causal history is too sparse or unordered')
        return self.torch.stack([x[1] for x in snapshots]), self.torch.tensor(ages,device='cuda',dtype=self.torch.float32)

    def describe(self):
        return dict(native=self.native.metadata,config=CONFIG,grid=GRID,cell=CELL,
            local_bounds=[[-2,6],[-4,4],[-4,4]],channels=CHANNELS,
            unknown_policy='zero-valued distance with separate explicit known flags; never certified free',
            history='four cumulative causal snapshots, nominal 0.2 s apart; actual ages retained under bounded asynchronous jitter, common planning-anchor coordinates')
