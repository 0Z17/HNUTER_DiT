#!/usr/bin/env python3
"""Compact evidence from preserved task pilot reports; does not rerun or filter tests."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report(run, output):
    collection=json.loads((run/'collection/collection.json').read_text())
    evaluation=json.loads((run/'evaluation/evaluation.json').read_text())
    predictions=json.loads((run/'predictions/predictions.json').read_text())
    selection=json.loads((run/'selection.json').read_text())
    if collection['count']!=48 or not evaluation['completed']:
        raise ValueError('report requires all collection attempts and planned evaluations')
    groups={}
    for task in ('surface','inspection'):
        rows=[r for r in collection['reports'] if r['config']['task']==task]
        splits={str(s):dict(attempted=sum(r['config']['split']==s for r in rows),
            accepted=sum(r['config']['split']==s and r['passed'] for r in rows)) for s in range(3)}
        groups[task]=dict(attempted=len(rows),accepted=sum(r['passed'] for r in rows),splits=splits,
            forbidden_contact_steps=sum(r['forbidden_contact_steps'] for r in rows),
            strict_actual_geometry_passed=sum(r.get('actual_task_geometry',{}).get('passed',False) for r in rows) if task=='surface' else None,
            whole_operation_force_passed=sum(r.get('whole_operate_force_interval_fraction',0)>=.9 for r in rows) if task=='surface' else None)
        if task=='surface':
            for key in ('operating_contact_fraction','operating_force_interval_fraction','whole_operate_contact_fraction',
                        'whole_operate_force_interval_fraction','maximum_actual_penetration_m','max_position_tracking_error_m'):
                values=[r[key] for r in rows if key in r]
                if values:groups[task][key]=dict(min=min(values),median=float(np.median(values)),max=max(values))
    grouped=defaultdict(list)
    for row in evaluation['reports']:grouped[row['task']+'/'+row['method']].append(row)
    compact={}
    for key,rows in grouped.items():
        value=dict(attempted=len(rows),passed=sum(r['passed'] for r in rows),
            physical_passed=sum(r.get('physical_execution_passed',False) for r in rows),
            raw_reference_geometry_passed=sum(r.get('reference_geometry',{}).get('passed',False) for r in rows),
            forbidden_contact_steps=sum(r.get('forbidden_contact_steps',0) for r in rows),
            contact_unknown=sum('forbidden_contact_steps' not in r for r in rows),
            failures=[dict(index=r['index'],error=r.get('error'),first_forbidden=r.get('first_forbidden')) for r in rows if not r['passed']])
        if key.startswith('surface/'):
            value['strict_actual_geometry_passed']=sum(r.get('actual_task_geometry',{}).get('passed',False) for r in rows)
            for field in ('plane_error_mean_m','plane_error_max_m','normal_error_max_deg'):
                numbers=[r['reference_geometry'][field] for r in rows if field in r.get('reference_geometry',{})]
                if numbers:value['reference_'+field]=dict(median=float(np.median(numbers)),max=max(numbers))
            value['whole_operation_force_passed']=sum(r.get('whole_operate_force_interval_fraction',0)>=.9 for r in rows)
        else:
            for field in ('effective_fraction','longest_continuous_effective_s'):
                numbers=[r['inspection'][field] for r in rows if 'inspection' in r]
                if numbers:value[field]=dict(min=min(numbers),median=float(np.median(numbers)),max=max(numbers))
        compact[key]=value
    timings={}
    unique={r['index']:r for r in predictions['rows']}
    for field in ('base_denoising_seconds','adapter_denoising_seconds'):
        values=[r[field] for r in unique.values() if r[field] is not None]
        timings[field]=dict(median=float(np.median(values)),p95=float(np.percentile(values,95)))
    training={}
    for file in sorted((run/'models').glob('*/training.json')):
        tr=json.loads(file.read_text())
        training[file.parent.name]=dict(best_validation_raw_control_mse=min(r['validation_raw_control_mse'] for r in tr['log']),
            steps=tr['steps'],seconds=tr['log'][-1]['elapsed_seconds'],training_count=len(tr['train_dataset_ids']),
            validation_count=len(tr['validation_dataset_ids']),seed=tr['seed'],
            frozen_base_bitwise_unchanged=tr['frozen_base_bitwise_unchanged'],
            frozen_normalization_bitwise_unchanged=tr['frozen_normalization_bitwise_unchanged'],
            base_parameter_gradients_absent=tr['base_parameter_gradients_absent'],test_excluded=tr['test_excluded'],
            output_residual=tr['configuration']['output_residual'],
            physical_loss_weight=tr.get('physical_loss_weight',.5),checkpoint_sha256=sha(file.parent/'best.pt'))
    summary=dict(schema='task_adaptive_pilot_evidence_v1',date='2026-10-04',
        purpose='New frozen-SDF task extension aligned with manuscript principles, not canonical v003 replication',
        collection=groups,evaluation=compact,training=training,selection=selection,
        source_sha256={str(p.relative_to(run)):sha(p) for p in [run/'collection/collection.json',run/'prepared/dataset.npz',run/'prepared/mapping.json',
            run/'evaluation/evaluation.json',run/'predictions/predictions.json',run/'collection_task_audit.json']},
        base_sha256=predictions['base_sha256'],adapter_sha256=predictions['adapter_sha256'],
        timing=dict(**timings,scope='single-case 20-step cached-memory GPU denoising only; excludes map, encoder, decoder, physics',
                    collection_wall_seconds=collection['elapsed_seconds'],evaluation_wall_seconds=evaluation['elapsed_seconds']),
        limitations=['Single seed; four held-out cases per task; narrow workbench distribution.',
          'Soft numerical contact is uncalibrated; post-settle physical gates differ from full-window force and strict actual geometry.',
          '48 trials contain 6 failed contact experts; failures excluded only from positive training, not hidden in reports.',
          'Single-target ray visibility is not RGB recognition, target area coverage or sensing-plane estimation.',
          'Task geometry is an external command; MID360 has a near-downward blind region.',
          'Four stationary-start measured histories, no in-flight replanning or industrial contact deployment.',
          'Each task has an independent adapter; no simultaneous task blending/conflict handling demonstrated.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['evaluation'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();report(args.run,args.output)
