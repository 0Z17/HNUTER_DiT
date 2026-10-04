#!/usr/bin/env python3
"""Pinned nvblox interface probe, not a policy or representation-quality comparison."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0, '/sim')
sys.path.insert(0, '/sim/perception/nvblox')
from perception.nvblox.worker import NativeMapper
from perception.nvblox.projection import spherical_range_image


def summarize(values):
    a = np.asarray(values)
    return dict(count=len(a), median_ms=float(np.median(a)), p95_ms=float(np.percentile(a,95)),
                mean_ms=float(a.mean()), max_ms=float(a.max()))


def main():
    dataset=Path('/data/results/geometry_pilot/collection96_seed107/dataset.npz')
    with np.load(dataset, allow_pickle=False) as z:
        names=('history_ray_ranges','ray_directions','history_sensor_poses','origin_world','world_to_canonical','split')
        data={name:z[name] for name in names}
    indices=np.flatnonzero(data['split']==2)[:8]
    config=dict(voxel_size=.25, max_range=20., esdf_max_distance=5., width=720, height=128, vertical_fov_degrees=104.)
    native=NativeMapper(config)
    torch=native.torch
    torch.set_num_threads(4)
    axis=(np.arange(64)+.5)*.25-8.
    local=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    sphere=torch.from_numpy(np.linalg.norm(local,axis=-1)<=8).cuda()
    rows=[]
    for repeat in range(-1,3):
        selected=indices[:2] if repeat<0 else indices
        for index in selected:
            native.clear();torch.cuda.synchronize()
            world=local@data['world_to_canonical'][index]+data['origin_world'][index]
            queries=torch.as_tensor(world,dtype=torch.float32,device='cuda')
            query_spheres=torch.cat((queries,torch.zeros((len(queries),1),device='cuda')),dim=-1)
            tsdf=torch.zeros((len(queries),2),device='cuda')
            esdf=torch.full((len(queries),1),native.unknown,device='cuda')
            for frame in range(4):
                ranges=data['history_ray_ranges'][index,frame];keep=np.isfinite(ranges)
                points=data['ray_directions'][frame,keep]*ranges[keep,None]
                pose=data['history_sensor_poses'][index,frame]
                torch.cuda.synchronize();start=time.perf_counter()
                depth=spherical_range_image(points,width=720,height=128,vertical_fov_degrees=104.,max_range=20.)
                projected=time.perf_counter()
                depth_gpu=torch.as_tensor(depth,device='cuda',dtype=torch.float32)
                pose_cpu=torch.as_tensor(np.ascontiguousarray(pose),dtype=torch.float32)
                native.mapper.integrate_depth(depth_gpu,pose_cpu,native.sensor,None,0)
                torch.cuda.synchronize();integrated=time.perf_counter()
                native.mapper.update_esdf(0)
                torch.cuda.synchronize();updated=time.perf_counter()
                tsdf=native.mapper.query_tsdf(tsdf,queries,0)
                esdf=native.mapper.query_esdf(esdf,query_spheres,0)
                torch.cuda.synchronize();queried=time.perf_counter()
                # Example scalar distance + weight + known channels stay on GPU.
                known=(torch.isfinite(esdf[:,0]) & ((esdf[:,0]-native.unknown).abs()>1e-4) & sphere)
                features=torch.stack((tsdf[:,0].clamp(-1.,1.),tsdf[:,1],torch.where(known,esdf[:,0],0.),known.float()),dim=-1)
                torch.cuda.synchronize();finished=time.perf_counter()
                if repeat>=0:
                    rows.append(dict(repeat=repeat,index=int(index),frame=frame,
                        projection_ms=1000*(projected-start), transfer_tsdf_integration_ms=1000*(integrated-projected),
                        esdf_update_ms=1000*(updated-integrated), dense_query_ms=1000*(queried-updated),
                        pack_gpu_features_ms=1000*(finished-queried), total_ms=1000*(finished-start),
                        known_query_count=int(known.sum().item()), query_count=len(queries),
                        output_finite=bool(torch.isfinite(features).all().item())))
        print(json.dumps(dict(repeat=repeat,measured_frames=len(rows))),flush=True)
    assert all(r['output_finite'] for r in rows)
    report=dict(schema='nvblox_dense_roi_latency_probe_v1',date='2026-10-04',config=config,
        environment=native.metadata, dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        simulation_worker_sha256=hashlib.sha256(Path('/sim/perception/nvblox/worker.py').read_bytes()).hexdigest(),
        protocol=dict(windows=8,frames_per_window=4,repeats=3,warmup_frames=8,
            grid_shape=[64,64,64],query_count=64**3,output_device='cuda',
            output_semantics=['clipped_tsdf_distance','tsdf_weight','esdf_distance_unknown_zero','known_mask'],
            scope='CPU spherical projection + transfer + incremental TSDF + ESDF + full ROI TSDF/ESDF queries + GPU feature packing',
            exclusions=['ray acquisition','grid creation','map reset','CPU export','IPC','neural encoding','diffusion','physics','rendering'],
            limitation='Four incrementally fused snapshots, not four independent occupancy scans. Static TSDF only: no dynamic separation, field accuracy or learned-policy behavior validated. Do not rank this as an equivalent fourth encoder.'),
        summary={k:summarize([r[k] for r in rows]) for k in rows[0] if k.endswith('_ms')},rows=rows)
    Path('/out/report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':main()
