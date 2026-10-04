"""Measured nvblox preparation and frozen-prior task-residual pilot training."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source,output,sim_root):
    from .sdf_mapping import LocalSDF
    mapper=LocalSDF(sim_root);rows=[];values=[];ages=[];hashes={}
    collection=json.loads((source/'collection.json').read_text())
    reports={row['index']:row for row in collection['reports']}
    recorded={int(path.stem.split('_')[-1]) for path in source.glob('episode_*.npz')}
    requested=set(collection['requested_indices'])
    if recorded!=requested or set(reports)!=requested:
        raise ValueError('incomplete collection; missing attempts must not silently disappear from evaluation')
    entries=[]
    for path in sorted(source.glob('episode_*.npz')):
        index=int(path.stem.split('_')[-1]);report=reports[index]
        with np.load(path,allow_pickle=False) as data:
            if bool(data['accepted']) != bool(report['passed']):raise ValueError('accepted array/report mismatch')
            mapper.reset(data['origin'])
            for i in range(4):
                ranges=data['history_ray_ranges'][i];keep=np.isfinite(ranges)
                mapper.integrate(data['ray_directions'][i,keep]*ranges[keep,None],
                                 data['history_sensor_poses'][i],data['history_times'][i])
            fields,age=mapper.inputs();values.append(fields.half().cpu().numpy());ages.append(age.cpu().numpy())
            rows.append({key:data[key].copy() for key in ('target','task_descriptor','free_controls','split','task','layout','origin','start_rotation','accepted')})
        entries.append(dict(index=index,config=report['config']));hashes[path.name]=digest(path)
        print(json.dumps(dict(prepared=len(rows),index=index)),flush=True)
    if not rows:raise ValueError('no recorded episodes')
    output.mkdir(parents=True,exist_ok=False)
    arrays={key:np.stack([r[key] for r in rows]) for key in rows[0]}
    np.savez_compressed(output/'dataset.npz',fields=np.stack(values),ages=np.stack(ages),
                        episode_index=np.array([e['index'] for e in entries]),**arrays)
    report=dict(schema='task_causal_sdf_data_v1',mapper=mapper.describe(),episodes=len(rows),
        source=str(source),source_hashes=hashes,entries=entries,dataset_sha256=digest(output/'dataset.npz'),
        collection_source_sha256=collection['source_sha256'],
        input_scope='four real past MID360/nvblox histories, current pose, goal, external geometric task command',
        excluded_inputs='expert direction/amplitude, future state, force labels, validity labels, scene boxes/layout ID',
        training_filter='accepted experts only; all test attempts retained for unbiased task evaluation',
        accepted_experts=sum(bool(row['accepted']) for row in rows),
        causal=True,map_seconds=dict(median=float(np.median(mapper.integration_times)),p95=float(np.percentile(mapper.integration_times,95))))
    (output/'mapping.json').write_text(json.dumps(report,indent=2)+'\n')


def rotation_torch(angles,torch):
    r,p,y=angles.unbind(-1);cr,sr=r.cos(),r.sin();cp,sp=p.cos(),p.sin();cy,sy=y.cos(),y.sin()
    return torch.stack((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,
        sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,-sp,cp*sr,cp*cr),-1).reshape(*angles.shape[:-1],3,3)


def raw_task_loss(free,target,task,rotation,basis,kind,torch):
    """Differentiable geometry on the raw clean estimate; decoder is absent."""
    controls=torch.cat((torch.zeros_like(free[:,:3]),free,target[:,None].expand(-1,3,-1)),1)
    pose=torch.einsum('kc,bcd->bkd',basis,controls)
    rot=rotation[:,None]@rotation_torch(pose[:,:,3:],torch)
    if kind=='surface':
        tip=pose[:,:,:3]+torch.einsum('bkij,bj->bki',rot,task[:,21:24])
        normal=torch.einsum('bij,bj->bi',rotation,task[:,3:6])
        plane=((tip-task[:,None,:3])*normal[:,None]).sum(-1).square().mean()
        alignment=(rot[:,:,:,2]-normal[:,None]).square().sum(-1).mean()
        return 5.*plane+.5*alignment
    camera=pose[:,:,:3]+torch.einsum('bkij,bj->bki',rot,task[:,18:21])
    delta=task[:,None,:3]-camera;distance=delta.norm(dim=-1).clamp_min(1e-4)
    cosine=(rot[:,:,:,0]*delta).sum(-1)/distance
    aim=(1.-cosine).mean()
    lo,hi=task[:,9,None],task[:,10,None]
    range_error=(torch.relu(lo-distance).square()+torch.relu(distance-hi).square()).mean()
    return .5*aim+range_error


def train(source,base_path,output,kind,steps=1500,seed=41,output_residual=True,physical_loss_weight=.5):
    sys.path.extend(str(p) for p in sorted(Path('/isaac-sim/extscache').glob('omni.kit.pip_archive-*/pip_prebundle')))
    import torch
    from scipy.interpolate import BSpline
    from .sdf_model import load_model
    from .task_adapter import FrozenSDFTaskAdapter,sample_task
    from .model import DiffusionSchedule
    from .sdf_pose import KNOTS
    from .task_spec import OPERATE
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed)
    with np.load(source/'dataset.npz',allow_pickle=False) as data:
        for layout in np.unique(data['layout']):
            if len(np.unique(data['split'][data['layout']==layout]))!=1:
                raise ValueError('a layout crosses dataset splits')
        select=(data['task']==kind)&data['accepted']
        train_ids=np.flatnonzero(select&(data['split']==0));val_ids=np.flatnonzero(select&(data['split']==1))
        if not len(train_ids) or not len(val_ids):raise ValueError(f'no accepted training/validation experts for {kind}')
        ids=np.r_[train_ids,val_ids]
        arrays={key:torch.as_tensor(data[key][ids],device='cuda').float() for key in
                ('fields','ages','target','task_descriptor','free_controls','start_rotation')}
        episode_ids=data['episode_index'][ids].tolist()
        layout_ids=data['layout'][ids].tolist()
    base,payload=load_model(base_path,'cuda');base_before={key:value.detach().cpu().clone() for key,value in base.state_dict().items()}
    model=FrozenSDFTaskAdapter(base,output_residual=output_residual).cuda()
    # Only frozen environment memory is cached, including no held-out tensors.
    memory=torch.cat([model.encode(arrays['fields'][i:i+4],arrays['ages'][i:i+4]) for i in range(0,len(ids),4)])
    del arrays['fields'],arrays['ages']
    optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=8e-4,weight_decay=.01)
    ema=copy.deepcopy(model).eval()
    schedule=DiffusionSchedule.cosine(100,torch.device('cuda'))
    generator=torch.Generator(device='cuda').manual_seed(seed+1)
    basis=torch.as_tensor(BSpline(KNOTS,np.eye(12),3)(np.linspace(*OPERATE,33)),device='cuda',dtype=torch.float32)
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();logs=[];best=float('inf')
    for step in range(1,steps+1):
        idx=torch.randint(len(train_ids),(16,),device='cuda',generator=generator)
        target=arrays['target'][idx];task=arrays['task_descriptor'][idx]
        clean=model.normalized(arrays['free_controls'][idx],target)
        t=torch.randint(100,(len(idx),),device='cuda',generator=generator);alpha=schedule.alpha_bars[t,None,None]
        noise=torch.randn(clean.shape,device='cuda',generator=generator)
        noisy=alpha.sqrt()*clean+(1-alpha).sqrt()*noise;truth=alpha.sqrt()*noise-(1-alpha).sqrt()*clean
        predicted=model.forward_cached(noisy,t,target,memory[idx],task)
        raw=model.denormalized(alpha.sqrt()*noisy-(1-alpha).sqrt()*predicted,target)
        velocity=(predicted-truth).square().mean()
        geometry=raw_task_loss(raw,target,task,arrays['start_rotation'][idx],basis,kind,torch)
        imitation=(raw-arrays['free_controls'][idx]).square().mean()
        loss=velocity+geometry+physical_loss_weight*imitation
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.adapter_parameters()),5.,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():
            for p,e in zip(model.adapter_parameters(),ema.adapter_parameters()):e.lerp_(p,.02)
        if step%250==0 or step==steps:
            val=slice(len(train_ids),len(ids))
            with torch.inference_mode():
                prediction=sample_task(ema,None,None,arrays['target'][val],arrays['task_descriptor'][val],
                    seed=23,memory=memory[val])
                error=float((prediction-arrays['free_controls'][val]).square().mean().item())
            row=dict(step=step,loss=float(loss.item()),velocity_loss=float(velocity.item()),
                raw_geometry_loss=float(geometry.item()),raw_imitation_loss=float(imitation.item()),
                validation_raw_control_mse=error,elapsed_seconds=time.perf_counter()-started)
            logs.append(row);print(json.dumps(row),flush=True)
            if error<best:
                best=error
                torch.save(dict(schema='task_causal_sdf_adapter_v1',task=kind,configuration=ema.adapter_configuration(),
                    adapter={k:v.cpu() for k,v in ema.adapter_state_dict().items()},base_sha256=digest(base_path),
                    dataset_sha256=digest(source/'dataset.npz'),step=step,seed=seed,validation=row),output/'best.pt')
            unchanged=all(torch.equal(before,model.base.state_dict()[key].cpu()) for key,before in base_before.items())
            no_grad=all(p.grad is None and not p.requires_grad for p in model.base.parameters())
            if not unchanged or not no_grad:raise RuntimeError('frozen prior contract violated')
            report=dict(task=kind,seed=seed,steps=steps,log=logs,train_dataset_ids=train_ids.tolist(),
                validation_dataset_ids=val_ids.tolist(),used_episode_ids=episode_ids,used_layout_ids=layout_ids,
                test_excluded=True,base_sha256=digest(base_path),frozen_base_bitwise_unchanged=unchanged,
                frozen_normalization_bitwise_unchanged=True,base_parameter_gradients_absent=no_grad,
                configuration=model.adapter_configuration(),checkpoint_selection='lowest validation raw-control MSE',
                losses='velocity MSE + raw preprojection task geometry + weighted raw physical-control imitation MSE',
                physical_loss_weight=physical_loss_weight,optimizer=dict(name='AdamW',lr=8e-4,weight_decay=.01,batch_size=16),
                decoder_used_during_training=False,source_dataset_sha256=digest(source/'dataset.npz'))
            (output/'training.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','train'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sim-root',default='/sim');p.add_argument('--base',type=Path)
    p.add_argument('--task',choices=['surface','inspection']);p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--seed',type=int,default=41);p.add_argument('--block-only',action='store_true')
    p.add_argument('--physical-loss-weight',type=float,default=.5)
    args=p.parse_args()
    if args.mode=='prepare':prepare(args.source,args.output,args.sim_root)
    else:train(args.source,args.base,args.output,args.task,args.steps,args.seed,not args.block_only,args.physical_loss_weight)


if __name__=='__main__':main()
