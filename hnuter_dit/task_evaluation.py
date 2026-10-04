"""Matched-seed raw/decoded/shuffled baselines with actual task execution."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
from .task_training import digest


def predict(source,base_path,adapters,output,seed=23,block_only_adapter=None):
    sys.path.extend(str(p) for p in sorted(Path('/isaac-sim/extscache').glob('omni.kit.pip_archive-*/pip_prebundle')))
    import torch
    from .sdf_model import load_model
    from .task_adapter import FrozenSDFTaskAdapter,sample_task
    from .task_spec import descriptor,decode_geometry
    mapping=json.loads((source/'mapping.json').read_text());entries=mapping['entries']
    with np.load(source/'dataset.npz',allow_pickle=False) as data:
        ids=np.flatnonzero(data['split']==2)
        arrays={key:data[key][ids].copy() for key in data.files}
    output.mkdir(parents=True,exist_ok=False);rows=[];base_hash=digest(base_path)
    adapter_hashes={};unavailable=[];dataset_hash=digest(source/'dataset.npz')
    # No task or checkpoint selection here: one fixed checkpoint and candidate
    # seed are evaluated on every registered held-out attempt, including expert
    # failures. Failed expert test cases are not filtered out.
    for kind in ('surface','inspection'):
        base,_=load_model(base_path,'cuda')
        path=adapters/kind/'best.pt'
        payload=torch.load(path,map_location='cuda',weights_only=False) if path.is_file() else None
        if payload is not None and payload['base_sha256']!=base_hash:raise ValueError('adapter/base hash mismatch')
        if payload is not None:
            if (payload.get('schema')!='task_causal_sdf_adapter_v1' or payload.get('task')!=kind or
                    payload.get('dataset_sha256')!=dataset_hash):
                raise ValueError('adapter task/schema/dataset provenance mismatch')
            adapter_hashes[kind]=digest(path)
        else:
            unavailable.append(dict(task=kind,reason='no accepted-data adapter checkpoint',
                missing_methods=['adapter_raw','adapter_decoded','shuffled_raw','shuffled_decoded']))
        model=FrozenSDFTaskAdapter(base,output_residual=True if payload is None else payload['configuration']['output_residual']).cuda().eval()
        if payload is not None:model.load_adapter_state_dict(payload['adapter'])
        block_model=None
        if kind=='inspection' and block_only_adapter is not None:
            block_payload=torch.load(block_only_adapter,map_location='cuda',weights_only=False)
            if (block_payload.get('schema')!='task_causal_sdf_adapter_v1' or block_payload.get('task')!=kind or
                block_payload.get('base_sha256')!=base_hash or block_payload.get('dataset_sha256')!=dataset_hash or
                block_payload['configuration']['output_residual']):
                raise ValueError('block-only ablation provenance/configuration mismatch')
            block_model=FrozenSDFTaskAdapter(base,output_residual=False).cuda().eval()
            block_model.load_adapter_state_dict(block_payload['adapter'])
            adapter_hashes['inspection_block_only']=digest(block_only_adapter)
        for local,index in enumerate(ids):
            if str(arrays['task'][local])!=kind:continue
            entry=entries[int(index)];config=entry['config'];episode=entry['index']
            fields=torch.tensor(arrays['fields'][local:local+1],device='cuda')
            ages=torch.tensor(arrays['ages'][local:local+1],device='cuda').float()
            target=torch.tensor(arrays['target'][local:local+1],device='cuda').float()
            task=torch.tensor(arrays['task_descriptor'][local:local+1],device='cuda').float()
            origin=arrays['origin'][local];rotation=arrays['start_rotation'][local]
            swapped=dict(config)
            # Counterfactual geometry replaces only task input; the world,
            # endpoints, map memory and diffusion noise are held constant.
            if kind=='surface':
                patch=list(config['keepout_in_pad_xy']);patch[2:]=[-patch[3],-patch[2]];swapped['keepout_in_pad_xy']=patch
            else:
                point=list(config['inspection_target']);point[1]*=-1;swapped['inspection_target']=point
            shuffled=torch.tensor(descriptor(swapped,origin,rotation)[None],device='cuda')
            with torch.inference_mode():
                memory=model.encode(fields,ages)
                torch.cuda.synchronize();began=time.perf_counter()
                raw_base=sample_task(model,None,None,target,task,seed=seed,enabled=False,memory=memory)[0].cpu().numpy()
                torch.cuda.synchronize();base_seconds=time.perf_counter()-began
                if payload is not None:
                    began=time.perf_counter()
                    raw_adapter=sample_task(model,None,None,target,task,seed=seed,memory=memory)[0].cpu().numpy()
                    torch.cuda.synchronize();adapter_seconds=time.perf_counter()-began
                    raw_shuffled=sample_task(model,None,None,target,shuffled,seed=seed,memory=memory)[0].cpu().numpy()
                if block_model is not None:
                    raw_block=sample_task(block_model,None,None,target,task,seed=seed,memory=memory)[0].cpu().numpy()
            linear=model.base.progress.cpu().numpy()[:,None]*arrays['target'][local]
            methods={'linear_decoded':decode_geometry(linear,config,origin,rotation),
                'base_raw':raw_base,'base_decoded':decode_geometry(raw_base,config,origin,rotation)}
            if payload is not None:
                methods.update(adapter_raw=raw_adapter,adapter_decoded=decode_geometry(raw_adapter,config,origin,rotation),
                    shuffled_raw=raw_shuffled,shuffled_decoded=decode_geometry(raw_shuffled,config,origin,rotation))
            if block_model is not None:
                methods.update(block_only_raw=raw_block,block_only_decoded=decode_geometry(raw_block,config,origin,rotation))
            for method,free in methods.items():
                filename=f'{episode:03}_{method}.npz'
                np.savez_compressed(output/filename,free=free,target=arrays['target'][local],origin=origin,rotation=rotation)
                rows.append(dict(index=episode,config=config,method=method,prediction=filename,
                    seed=seed,base_denoising_seconds=base_seconds,
                    adapter_denoising_seconds=adapter_seconds if payload is not None else None,
                    source_expert_accepted=bool(arrays['accepted'][local])))
            print(json.dumps(dict(predicted=episode,task=kind,adapter_available=payload is not None)),flush=True)
    (output/'predictions.json').write_text(json.dumps(dict(schema='task_pilot_predictions_v1',rows=rows,
        base_sha256=base_hash,adapter_sha256=adapter_hashes,unavailable=unavailable,
        source_dataset_sha256=dataset_hash,seed=seed,
        candidates_per_method_per_task=1,test_filter='all recorded test attempts, irrespective of expert success',
        decoder_scope='surface plane/normal or inspection control-point look-at only; no tangential/path search'),indent=2)+'\n')


def run_one(arguments):
    row,predictions,collection,output,sim_root=arguments
    from .task_data import warmup,execute
    from .task_physics import TaskWorld
    from .sdf_pose import PoseSpline,full_controls
    config=row['config'];world=None;began=time.perf_counter()
    try:
        world=TaskWorld(config,sim_root)
        warmup(world,rays=32)
        with np.load(Path(collection)/f"episode_{row['index']:03}.npz") as original:
            snapshot=world.snapshot()
            if any(not np.allclose(original[key],snapshot[key],atol=1e-10,rtol=0) for key in snapshot):
                raise RuntimeError('initial physical qpos/qvel/act/time snapshot mismatch')
        with np.load(Path(predictions)/row['prediction']) as data:
            reference=PoseSpline(full_controls(data['free'],data['target']),origin=data['origin'],start_rotation=data['rotation'])
        report,trace=execute(world,reference,config)
        (Path(output)/f"{row['index']:03}_{row['method']}_trace.json").write_text(json.dumps(trace,separators=(',',':'))+'\n')
    except Exception as exc:
        report=dict(passed=False,completed=False,error=f'{type(exc).__name__}: {exc}')
    finally:
        if world is not None:world.close()
    report.update(index=row['index'],method=row['method'],task=config['task'],elapsed_seconds=time.perf_counter()-began)
    (Path(output)/f"{row['index']:03}_{row['method']}.json").write_text(json.dumps(report,indent=2)+'\n')
    return report


def execute_predictions(predictions,collection,output,sim_root,workers=6):
    spec=json.loads((predictions/'predictions.json').read_text());output.mkdir(parents=True,exist_ok=False)
    reports=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for report in pool.map(run_one,[(row,predictions,collection,output,sim_root) for row in spec['rows']]):
            reports.append(report)
            print(json.dumps({key:report.get(key) for key in ('index','task','method','passed','error')}),flush=True)
            aggregate={}
            for row in reports:
                key=row['task']+'/'+row['method'];group=aggregate.setdefault(key,dict(count=0,passed=0,physical_passed=0,forbidden_contacts=0,contact_unknown=0))
                group['count']+=1;group['passed']+=bool(row['passed'])
                group['physical_passed']+=bool(row.get('physical_execution_passed',False))
                group['forbidden_contacts']+=int(row.get('forbidden_contact_steps',0))
                group['contact_unknown']+=int('forbidden_contact_steps' not in row)
            (output/'evaluation.json').write_text(json.dumps(dict(schema='task_pilot_physical_evaluation_v1',
                predictions_sha256=digest(predictions/'predictions.json'),reports=reports,aggregate=aggregate,
                elapsed_seconds=time.perf_counter()-start,completed=len(reports)==len(spec['rows']),
                candidates_per_case=1,scope='single-seed small native MuJoCo pilot; no selection on test results'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['predict','execute'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path);p.add_argument('--adapters',type=Path);p.add_argument('--collection',type=Path)
    p.add_argument('--block-only-adapter',type=Path)
    p.add_argument('--sim-root',type=Path,default=Path('/home/z017/research/isaacsim-industrial-reactor'))
    p.add_argument('--workers',type=int,default=6);p.add_argument('--seed',type=int,default=23)
    args=p.parse_args()
    if args.mode=='predict':predict(args.source,args.base,args.adapters,args.output,args.seed,args.block_only_adapter)
    else:execute_predictions(args.source,args.collection,args.output,args.sim_root,args.workers)


if __name__=='__main__':main()
