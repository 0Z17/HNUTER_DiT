"""Frozen held-out pose evaluation; raw model quality is separate from guards."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np


def infer(dataset,checkpoint,output):
    sys.path.extend(str(p) for p in sorted(Path('/isaac-sim/extscache').glob('omni.kit.pip_archive-*/pip_prebundle')))
    import torch
    from .sdf_model import load_model,sample
    from .sdf_pose import full_controls,PoseSpline,proxy_clearance,PROGRESS
    torch.set_num_threads(4);model,payload=load_model(checkpoint)
    with np.load(dataset,allow_pickle=False) as z:
        indices=np.flatnonzero(z['split']==2)
        fields=torch.as_tensor(z['fields'][indices],device='cuda');ages=torch.as_tensor(z['ages'][indices],device='cuda')
        targets=torch.as_tensor(z['target'][indices],device='cuda')
        truth=z['free'][indices];boxes=z['evaluator_boxes'][indices];origins=z['evaluator_origins'][indices];rotations=z['evaluator_rotations'][indices]
    output.mkdir(parents=True,exist_ok=False)
    results={};predictions={};timings=[]
    for mode in ('model','latest','empty','straight'):
        rows=[];frees=[]
        # Batch-one, deterministic identical seed across interventions.
        for row,index in enumerate(indices):
            if mode=='straight':free=PROGRESS[:,None]*targets[row].cpu().numpy()
            else:
                torch.cuda.synchronize();start=time.perf_counter()
                free=sample(model,fields[row:row+1],ages[row:row+1],targets[row:row+1],seed=23,
                            ablation=None if mode=='model' else mode).cpu().numpy()[0]
                torch.cuda.synchronize()
                if mode=='model':timings.append(time.perf_counter()-start)
            target=targets[row].cpu().numpy();path=PoseSpline(full_controls(free,target),origins[row],start_rotation=rotations[row])
            audit=path.audit();clearance=proxy_clearance(path,boxes[row])
            feasible=audit['speed_bound_mps']<=.65 and audit['acceleration_bound_mps2']<=.9 and audit['max_local_angle_rad']<=.7
            rows.append(dict(index=int(index),proxy_clearance_m=clearance,feasible=bool(feasible),
                safe=bool(clearance>.1),passed=bool(feasible and clearance>.1),control_rmse=float(np.sqrt(np.mean((free-truth[row])**2))),audit=audit))
            frees.append(free)
        results[mode]=dict(count=len(rows),passed=sum(r['passed'] for r in rows),rows=rows)
        predictions[mode]=np.stack(frees)
    np.savez_compressed(output/'predictions.npz',indices=indices,**predictions)
    report=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),selected_step=payload['step'],
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),seed=23,split='test',results=results,
        inference_seconds=dict(median=float(np.median(timings)),p95=float(np.percentile(timings,95))),
        caveat='Sampled sphere proxy is not native collision truth; native evaluation recorded separately. First call includes cold-start effects.')
    (output/'offline.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['passed'] for k,v in results.items()}),flush=True)


def native_one(job):
    collection,predictions,mode,row,sim_root=job
    from .sdf_data import StaticWorld,run_path
    from .sdf_pose import PoseSpline,full_controls
    with np.load(predictions,allow_pickle=False) as z:index=int(z['indices'][row]);free=z[mode][row]
    configs=json.loads((Path(collection)/'collection.json').read_text())['reports'];cfg=next(r['config'] for r in configs if r['index']==index)
    with np.load(Path(collection)/f'episode_{index:03}.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    world=StaticWorld(cfg,sim_root)
    try:
        world.restore(data);path=PoseSpline(full_controls(free,data['target']),data['origin'],start_rotation=data['start_rotation'])
        report=run_path(world,path,record=True,stop_on_contact=True);report.update(index=index,mode=mode)
        return report
    finally:world.close()


def native(collection,predictions,output,sim_root,workers):
    with np.load(predictions) as z:count=len(z['indices'])
    jobs=[(collection,predictions,mode,row,sim_root) for mode in ('model','empty','straight') for row in range(count)]
    reports=[];start=time.perf_counter();output.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(native_one,jobs):
            trace=row.pop('trace'); (output/f"{row['mode']}_{row['index']}_trace.json").write_text(json.dumps(trace)+'\n')
            reports.append(row);print(json.dumps(row),flush=True)
            summary={mode:dict(count=sum(r['mode']==mode for r in reports),passed=sum(r['mode']==mode and r['passed'] for r in reports),
                collision_episodes=sum(r['mode']==mode and r['contact_steps']>0 for r in reports)) for mode in ('model','empty','straight')}
            (output/'native.json').write_text(json.dumps(dict(reports=reports,summary=summary,elapsed_seconds=time.perf_counter()-start,
                safety_guard=False,geometry_repair=False,truth='real original HNUTER collision geometry contacts checked at every 1 ms physics step'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('infer','native'))
    p.add_argument('--dataset',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--collection',type=Path)
    p.add_argument('--predictions',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--sim-root',default='/home/z017/research/isaacsim-industrial-reactor')
    args=p.parse_args()
    if args.mode=='infer':infer(args.dataset,args.checkpoint,args.output)
    else:native(args.collection,args.predictions,args.output,args.sim_root,args.workers)
