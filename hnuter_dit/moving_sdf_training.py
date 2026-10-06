"""Explicit within-factory route holdout pilot (not unseen-scene validation)."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .sdf_navigation_data import sha

def bootstrap():
    import sys
    sys.path.extend(str(p) for p in sorted(Path('/isaac-sim/extscache').glob('omni.kit.pip_archive-*/pip_prebundle')))
    import torch
    torch.set_num_threads(2)
    return torch

def condition(d):
    return np.concatenate([d['goal'],d['current_velocity'],d['current_angular_velocity'],d['current_quaternion'],
                            d['current_reference_acceleration'],np.atleast_1d(d['execution_delay_s'])]).astype(np.float32)

def records(root,roles):
    protocol=json.loads((root/'protocol.json').read_text())
    if protocol['schema']!='within_factory_route_holdout_v1':raise ValueError('explicit same-factory protocol required')
    rows=[]
    for key,role in protocol['roles'].items():
        if role not in roles:continue
        folder=root/f'mapped_{int(key):02}'
        m=json.loads((folder/'mapping.json').read_text())
        for j,r in enumerate(m['records']):
            rows.append(dict(route=int(key),window=j,role=role,path=str(folder/r['file']),sha256=r['sha256']))
    return rows

def train(args):
    torch=bootstrap()
    from .sdf_model import load_model
    from .moving_sdf_model import MovingSDFDiT,moving_sample
    from .model import DiffusionSchedule
    torch.manual_seed(77);np.random.seed(77)
    base,_=load_model(args.base);model=MovingSDFDiT().cuda();model.initialize_from(base)
    rows=records(args.root,{'train','validation'});conditions=[];labels=[];memory=[]
    for row in rows:
        if sha(row['path'])!=row['sha256']:raise ValueError('dataset hash mismatch')
        with np.load(row['path']) as d:
            conditions.append(condition(d));pose=d['future'][np.arange(10,81,10),:7]
            labels.append(np.c_[pose[:,:3],Rotation.from_quat(pose[:,[4,5,6,3]]).as_euler('xyz')])
            with torch.inference_mode():
                memory.append(model.encode(torch.as_tensor(d['fields'][None],device='cuda'),torch.as_tensor(d['ages'][None],device='cuda'))[0])
    c=torch.as_tensor(np.stack(conditions),device='cuda');y=torch.as_tensor(np.stack(labels),device='cuda',dtype=torch.float32)
    mem=torch.stack(memory).detach();train_ids=torch.tensor([i for i,r in enumerate(rows) if r['role']=='train'],device='cuda')
    val_ids=torch.tensor([i for i,r in enumerate(rows) if r['role']=='validation'],device='cuda')
    residual=y[train_ids]-model.baseline(c[train_ids]);model.mean.copy_(residual.mean(0));model.std.copy_(residual.std(0).clamp_min(.02))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.01)
    ema=copy.deepcopy(model).eval();schedule=DiffusionSchedule.cosine(100,torch.device('cuda'))
    rng=torch.Generator(device='cuda').manual_seed(78);logs=[];best=float('inf');began=time.perf_counter()
    args.output.mkdir(exist_ok=False,parents=True)
    for step in range(1,args.steps+1):
        ids=train_ids[torch.randint(len(train_ids),(16,),device='cuda',generator=rng)]
        target=y[ids];clean=model.normalized(target,c[ids]);t=torch.randint(100,(16,),device='cuda',generator=rng)
        a=schedule.alpha_bars[t,None,None];noise=torch.randn(clean.shape,device='cuda',generator=rng)
        x=a.sqrt()*clean+(1-a).sqrt()*noise;truth=a.sqrt()*noise-(1-a).sqrt()*clean
        v=model.forward_cached(x,t,c[ids],mem[ids]);estimate=model.denormalized(a.sqrt()*x-(1-a).sqrt()*v,c[ids])
        err=estimate-target;curv=err[:,2:]-2*err[:,1:-1]+err[:,:-2]
        loss=(v-truth).square().mean()+10*err.square().mean()+2*curv.square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        with torch.no_grad():
            for p,e in zip(model.parameters(),ema.parameters()):e.lerp_(p,.01)
        if step%250==0 or step==args.steps:
            pred=moving_sample(ema,c[val_ids],mem[val_ids]);mse=float((pred-y[val_ids]).square().mean())
            row=dict(step=step,loss=float(loss),validation_control_mse=mse,elapsed_seconds=time.perf_counter()-began);logs.append(row);print(json.dumps(row),flush=True)
            if mse<best:
                best=mse;torch.save(dict(schema='moving_sdf_dit_v1',model={k:v.cpu() for k,v in ema.state_dict().items()},
                    step=step,base_sha256=sha(args.base),protocol_sha256=sha(args.root/'protocol.json'),seed=77,validation=row),args.output/'best.pt')
            (args.output/'training.json').write_text(json.dumps(dict(log=logs,train_count=len(train_ids),validation_count=len(val_ids),records=rows,
                parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                initialization='transfer frozen SDF encoder and compatible DiT weights; new motion condition/order; train-only normalization; test excluded'),indent=2)+'\n')


def infer(args):
    torch=bootstrap()
    from .sdf_model import load_model,sample
    from .moving_sdf_model import MovingSDFDiT,moving_sample
    from .moving_pose import MovingReference
    from .sdf_pose import PoseSpline,full_controls
    base,_=load_model(args.base);payload=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    if payload['base_sha256']!=sha(args.base) or payload['protocol_sha256']!=sha(args.root/'protocol.json'):raise ValueError('checkpoint provenance mismatch')
    model=MovingSDFDiT().cuda();model.load_state_dict(payload['model']);model.eval();rows=records(args.root,{'test'})
    args.output.mkdir(exist_ok=False,parents=True);results=[];times=np.arange(81)*.05+.1
    # Warm-up only synthetic zero tensors, never fit held-out data.
    with torch.inference_mode():
        warm_fields=torch.zeros((1,4,6,32,32,32),device='cuda');warm_ages=torch.tensor([[-.6,-.4,-.2,0]],device='cuda');warm_c=torch.zeros((1,17),device='cuda')
        for _ in range(2):moving_sample(model,warm_c,model.encode(warm_fields,warm_ages))
        for index,row in enumerate(rows):
            if sha(row['path'])!=row['sha256']:raise ValueError('test hash mismatch')
            with np.load(row['path']) as archive:d={k:archive[k] for k in archive.files}
            c=condition(d);ct=torch.as_tensor(c[None],device='cuda');f=torch.as_tensor(d['fields'][None],device='cuda');ages=torch.as_tensor(d['ages'][None],device='cuda')
            for method in ('legacy_frozen','constant_velocity','moving_finetuned','moving_finetuned_empty_map'):
                torch.cuda.synchronize();began=time.perf_counter();controls=np.empty((0,6))
                if method.startswith('moving'):
                    memory=model.encode(torch.zeros_like(f) if method.endswith('empty_map') else f,ages)
                    controls=moving_sample(model,ct,memory,seed=23)[0].cpu().numpy();reference=MovingReference(controls,c);p=reference.sample(times)
                elif method=='legacy_frozen':
                    start=Rotation.from_quat(c[[10,11,12,9]]).as_matrix();target=np.r_[c[:3],Rotation.from_matrix(start.T).as_euler('xyz')]
                    controls=sample(base,f,ages,torch.as_tensor(target[None],device='cuda',dtype=torch.float32),seed=23)[0].cpu().numpy()
                    reference=PoseSpline(full_controls(controls,target),start_rotation=start);p=reference.sample(times)
                else:
                    p=dict(pos=times[:,None]*c[None,3:6],quaternion=np.tile(c[9:13],(81,1)),vel=np.tile(c[3:6],(81,1)),acc=np.zeros((81,3)),angular_velocity=np.zeros((81,3)),angular_acceleration=np.zeros((81,3)))
                torch.cuda.synchronize();elapsed=time.perf_counter()-began
                error=np.linalg.norm(p['pos']-d['future'][:,:3],axis=1)
                name=f'case_{index:04d}_{method}.npz'
                np.savez_compressed(args.output/name,controls=controls,condition=c,pose=np.c_[p['pos'],p['quaternion']],times=times,
                    origin=d['origin'],world_from_local=d['world_from_local'],truth=d['future'],**{k:p[k] for k in ('vel','acc','angular_velocity','angular_acceleration')})
                results.append(dict(**row,method=method,file=name,ade_m=float(error.mean()),fde_m=float(error[-1]),
                    compute_seconds=elapsed,sampled_speed_max=float(np.linalg.norm(p['vel'],axis=1).max()),sampled_acc_max=float(np.linalg.norm(p['acc'],axis=1).max())))
            print(json.dumps(dict(inferred=index+1,total=len(rows))),flush=True)
            (args.output/'predictions.json').write_text(json.dumps(dict(checkpoint_sha256=sha(args.checkpoint),base_sha256=sha(args.base),results=results),indent=2)+'\n')


def prepare(args):
    import gc
    from types import SimpleNamespace
    from .factory_expert_data import prepare as map_episode
    rows=json.loads((args.root/'collection_status.json').read_text())
    if len(rows)!=12 or any(not r['accepted'] for r in rows):
        raise ValueError('pilot collection incomplete or has failures; report them, do not silently drop/redraw')
    for row in sorted(rows,key=lambda r:r['route']):
        i=row['route']
        map_episode(SimpleNamespace(source=args.root/f'route_{i:02}',output=args.root/f'mapped_{i:02}',sim_root=args.sim_root,map_history_seconds=5.))
        gc.collect()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','train','infer'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path);p.add_argument('--output',type=Path);p.add_argument('--sim-root',type=Path,default=Path('/sim'));p.add_argument('--checkpoint',type=Path);p.add_argument('--steps',type=int,default=2000)
    a=p.parse_args()
    if a.mode!='prepare' and (a.base is None or a.output is None):p.error('train/infer require base and output')
    if a.mode=='infer' and a.checkpoint is None:p.error('infer requires checkpoint')
    {'prepare':prepare,'train':train,'infer':infer}[a.mode](a)
if __name__=='__main__':main()
