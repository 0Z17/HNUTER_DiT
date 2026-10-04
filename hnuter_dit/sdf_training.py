"""Prepare causal GPU map data and train a fresh small static pose diffusion pilot."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source,output,sim_root):
    from .sdf_mapping import LocalSDF
    mapper=LocalSDF(sim_root); fields=[]; targets=[]; frees=[]; splits=[]; ages=[]; source_hashes={}
    collection=json.loads((source/'collection.json').read_text())
    configs={row['index']:row['config'] for row in collection['reports']}
    boxes=[];origins=[];rotations=[]
    for path in sorted(source.glob('episode_*.npz')):
        with np.load(path,allow_pickle=False) as data:
            mapper.reset(data['origin'])
            for i in range(4):
                ranges=data['history_ray_ranges'][i];keep=np.isfinite(ranges)
                points=data['ray_directions'][i,keep]*ranges[keep,None]
                mapper.integrate(points,data['history_sensor_poses'][i],data['history_times'][i])
            value,age=mapper.inputs();fields.append(value.half().cpu().numpy());ages.append(age.cpu().numpy())
            targets.append(data['target']);frees.append(data['free_controls']);splits.append(data['split'])
            source_hashes[path.name]=digest(path)
            index=int(path.stem.split('_')[-1]);boxes.append(configs[index]['boxes'])
            origins.append(data['origin']);rotations.append(data['start_rotation'])
        print(json.dumps(dict(prepared=len(fields),episode=path.name)),flush=True)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'dataset.npz',fields=np.stack(fields),ages=np.stack(ages),target=np.stack(targets),free=np.stack(frees),split=np.array(splits),
        evaluator_boxes=np.array(boxes),evaluator_origins=np.array(origins),evaluator_rotations=np.array(rotations))
    report=dict(schema='static_sdf_causal_data_v1',mapper=mapper.describe(),source=str(source),source_hashes=source_hashes,
        episodes=len(fields),dataset_sha256=digest(output/'dataset.npz'),causal=True,
        input_scope='measured LiDAR return positions, simultaneous sensor poses, current pose and requested goal only',
        map_seconds=dict(median=float(np.median(mapper.integration_times)),p95=float(np.percentile(mapper.integration_times,95))))
    (output/'mapping.json').write_text(json.dumps(report,indent=2)+'\n')


def train(dataset,output,steps,seed):
    import sys
    sys.path.extend(str(p) for p in sorted(Path('/isaac-sim/extscache').glob('omni.kit.pip_archive-*/pip_prebundle')))
    import torch
    from .sdf_model import SDFPoseDiT,sample
    from .sdf_mapping import CHANNELS
    from .sdf_pose import SCHEMA,PoseSpline,full_controls,proxy_clearance
    from .model import DiffusionSchedule
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed)
    with np.load(dataset,allow_pickle=False) as data:
        split=data['split'].copy();train_ids=np.flatnonzero(split==0);val_ids=np.flatnonzero(split==1)
        # Held-out test tensors are never transferred or used during training.
        selected=np.r_[train_ids,val_ids]
        arrays={key:torch.as_tensor(data[key][selected],device='cuda') for key in ('fields','ages','target','free')}
        evaluator={key:data[key][val_ids].copy() for key in ('evaluator_boxes','evaluator_origins','evaluator_rotations')}
    if not len(train_ids) or not len(val_ids):raise ValueError('train and validation splits required')
    model=SDFPoseDiT().cuda();residual=arrays['free'][:len(train_ids)]-model.baseline(arrays['target'][:len(train_ids)])
    model.mean.copy_(residual.mean(0));model.std.copy_(residual.std(0).clamp_min(.04))
    ema=copy.deepcopy(model).eval();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    schedule=DiffusionSchedule.cosine(100,torch.device('cuda'));generator=torch.Generator(device='cuda').manual_seed(seed+1)
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();best=None;logs=[]
    for step in range(1,steps+1):
        idx=torch.randint(len(train_ids),(16,),device='cuda',generator=generator)
        target=arrays['target'][idx].float();clean=model.normalized(arrays['free'][idx].float(),target)
        t=torch.randint(100,(len(idx),),device='cuda',generator=generator);alpha=schedule.alpha_bars[t,None,None]
        noise=torch.randn(clean.shape,device='cuda',generator=generator)
        noisy=alpha.sqrt()*clean+(1-alpha).sqrt()*noise;truth=alpha.sqrt()*noise-(1-alpha).sqrt()*clean
        memory=model.encode(arrays['fields'][idx],arrays['ages'][idx])
        predicted=model.forward_cached(noisy,t,target,memory)
        loss=(predicted-truth).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        with torch.no_grad():
            for p,e in zip(model.parameters(),ema.parameters()):e.lerp_(p,.01)
        if step%250==0 or step==steps:
            s=slice(len(train_ids),len(selected))
            with torch.inference_mode():prediction=sample(ema,arrays['fields'][s],arrays['ages'][s],arrays['target'][s],seed=23).cpu().numpy()
            errors=[];safe=[];valid=[]
            for j,index in enumerate(val_ids):
                target=arrays['target'][len(train_ids)+j].cpu().numpy()
                path=PoseSpline(full_controls(prediction[j],target),origin=evaluator['evaluator_origins'][j],start_rotation=evaluator['evaluator_rotations'][j])
                audit=path.audit();safe.append(proxy_clearance(path,evaluator['evaluator_boxes'][j])>.1)
                valid.append(audit['speed_bound_mps']<=.65 and audit['acceleration_bound_mps2']<=.9 and audit['max_local_angle_rad']<=.7)
                errors.append(float(np.mean((prediction[j]-arrays['free'][len(train_ids)+j].cpu().numpy())**2)))
            success=int(np.sum(np.array(safe)&np.array(valid)));error=float(np.mean(errors));score=(success,-error)
            row=dict(step=step,loss=float(loss.item()),validation_safe=sum(safe),validation_feasible=sum(valid),
                     validation_success=success,validation_count=len(val_ids),validation_mse=error,elapsed_seconds=time.perf_counter()-start)
            logs.append(row);print(json.dumps(row),flush=True)
            if best is None or score>best:
                best=score
                torch.save(dict(schema=SCHEMA,channels=list(CHANNELS),architecture=ema.architecture,
                    model={k:v.cpu() for k,v in ema.state_dict().items()},step=step,seed=seed,source_dataset_sha256=digest(dataset),
                    validation=row,initialization='random, all parameters trained from scratch',training_count=len(train_ids),validation_count=len(val_ids)),output/'best.pt')
            (output/'training.json').write_text(json.dumps(dict(seed=seed,steps=steps,log=logs,train_ids=train_ids.tolist(),validation_ids=val_ids.tolist(),test_excluded=True),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','train'))
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sim-root',default='/sim');parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--seed',type=int,default=29)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.source,args.output,args.sim_root)
    else:train(args.source,args.output,args.steps,args.seed)


if __name__=='__main__':main()
