#!/usr/bin/env python3
"""Produce a compact provenance report and fixed-sample comparison figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnuter_dit.temporal_data import load_dataset
from hnuter_dit.temporal_evaluation import expansion_decision, file_sha256, industrial_evidence_valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--temporal-training', type=Path, required=True)
    parser.add_argument('--latest-training', type=Path, required=True)
    parser.add_argument('--industrial', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation.get('completed') is not True:
        raise ValueError('cannot publish an incomplete held-out evaluation')
    data = load_dataset(args.dataset)
    dataset_sha = data['_freeze']['dataset_sha256']
    if dataset_sha != evaluation.get('dataset_freeze', {}).get('dataset_sha256'):
        raise ValueError('report dataset differs from the evaluated frozen dataset')
    histories, training = {}, {}
    for name, directory in (('temporal', args.temporal_training), ('latest_only', args.latest_training)):
        training[name] = {
            'summary': json.loads((directory/'summary.json').read_text()),
            'representation_audit': json.loads((directory/'representation_audit.json').read_text()),
            'configuration': json.loads((directory/'configuration.json').read_text()),
        }
        histories[name] = json.loads((directory/'training_history.json').read_text())
        observed = file_sha256(directory/'best.pt')
        if (observed != evaluation['variants'][name]['checkpoint_sha256'] or
                observed != training[name]['summary']['checkpoint_files']['best.pt'] or
                training[name]['configuration']['source_dataset']['dataset_sha256'] != dataset_sha):
            raise ValueError(name + ' training directory does not match the evaluated weights/dataset')
    if (file_sha256(args.temporal_training/'initial.pt') != evaluation['variants']['untrained']['checkpoint_sha256'] or
            evaluation['variants']['frozen_history']['checkpoint_sha256'] != evaluation['variants']['temporal']['checkpoint_sha256']):
        raise ValueError('ablation or initial checkpoint provenance mismatch')
    for key in ('steps', 'batch_size', 'learning_rate', 'seed', 'validation_interval', 'sampling_steps', 'ema_decay'):
        if training['temporal']['summary']['configuration'][key] != training['latest_only']['summary']['configuration'][key]:
            raise ValueError('the paired training budgets/settings differ: ' + key)
    industrial_runs = []
    for path in args.industrial:
        value = json.loads(path.read_text())
        planner = value.get('planner', {})
        industrial_runs.append({
            'report_path': str(path), 'report_sha256': file_sha256(path),
            **{key: value.get(key) for key in ('passed', 'completed', 'mode', 'dynamic_profile', 'goal_reached',
                    'final_goal_distance_m', 'final_speed_mps', 'first_goal_time', 'steps', 'loop_wall_seconds',
                    'wall_seconds_including_cleanup', 'simulation_seconds', 'dynamic_obstacle')},
            'collision': value.get('collision'),
            'planner': {key: planner.get(key) for key in ('failed', 'failure_reason', 'accepted_plans',
                'guard_rejections', 'guard_interventions', 'horizon_completions', 'time_in_state_seconds',
                'planning_seconds_mean', 'planning_seconds_max', 'start_velocity_support', 'motion_limits',
                'temporal_checkpoint_used', 'checkpoint')},
            'plan_status_counts': {status: sum(plan.get('status') == status for plan in planner.get('plans', []))
                                  for status in sorted({plan.get('status', 'unknown') for plan in planner.get('plans', [])})},
        })
    profiles = {run['dynamic_profile'] for run in industrial_runs}
    industrial = {'passed': bool({'constant_incoming', 'constant_outgoing'} <= profiles and all(r['passed'] is True for r in industrial_runs)),
                  'required': 'both incoming and outgoing constant-speed industrial transfer trials', 'runs': industrial_runs}
    industrial['system_passed'] = industrial['passed']
    industrial['unassisted_model_passed'] = industrial_evidence_valid(industrial, evaluation['variants']['temporal']['checkpoint_sha256'])
    summaries = {key: value['summary'] for key, value in evaluation['variants'].items()}
    compact = {
        'schema': 'temporal_mid360_pilot_validation_v1', 'date': '2026-10-04',
        'dataset': data['_freeze'], 'training': training,
        'evaluation_report': {'path': str(args.evaluation), 'sha256': file_sha256(args.evaluation)},
        'held_out': summaries, 'paired_comparisons': evaluation['paired_comparisons'],
        'teacher': evaluation['teacher']['summary'], 'native_execution': evaluation['native'],
        'industrial': industrial,
        'gate': expansion_decision(summaries, native=evaluation['native'], industrial=industrial,
                                  checkpoint_sha256=evaluation['variants']['temporal']['checkpoint_sha256']),
        'scope': 'Newly trained position-only model; stationary starts and constant-speed crossing curriculum.',
        'limitations': [
            'Known constant-speed futures are teacher/evaluation truth, never student inputs.',
            'Native candidate rollouts last the fixed 8-second horizon; intentional waits need not reach the goal.',
            'No guard or ground-truth candidate selection in the held-out model comparison.',
            'Industrial current-scan guards can reject a learned reference; intervention does not prove model success.',
            'No general dynamic-environment, moving-start, full-attitude or wall-clock real-time claim.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact, indent=2, allow_nan=False)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(args.evaluation.parent/'predictions.npz', allow_pickle=False) as result:
        indices = result['indices']
        predictions = {name: result[name] for name in summaries}
    if len(indices) < 2 or data['group_id'][indices[0]] != data['group_id'][indices[1]]:
        raise ValueError('first two held-out samples must be the first complete pair')
    colors = {'temporal': '#2166AC', 'latest_only': '#D98216', 'frozen_history': '#777777', 'untrained': '#C54E5A'}
    labels = {'temporal': 'Trained history', 'latest_only': 'Trained latest only', 'frozen_history': 'Frozen-history ablation', 'untrained': 'Untrained'}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for local, ax in enumerate(axes[0]):
        index = indices[local]
        ax.plot(data['future_times'], data['positions'][index, :, 0], color='black', linewidth=2., label='Physical-validated teacher')
        for name, values in predictions.items():
            ax.plot(data['future_times'], values[local, :, 0], color=colors[name],
                    linewidth=1.8 if name == 'temporal' else 1.1, alpha=.95,
                    linestyle='-' if name in ('temporal', 'latest_only') else '--', label=labels[name])
        direction = 'Approaching' if int(data['pair_variant'][index]) == 0 else 'Receding'
        ax.set(title=f'First held-out pair: {direction} obstacle (group {data["group_id"][index]})',
               xlabel='Seconds after observation', ylabel='Goal-direction position (m)')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    names = list(summaries)
    axes[1, 0].bar([labels[name].replace(' ', '\n') for name in names],
        [100*summaries[name]['group_balanced_feasible_safe_progress_rate'] for name in names],
        color=[colors[name] for name in names])
    axes[1, 0].axhline(80, color='black', linestyle=':', label='Predeclared 80% gate')
    axes[1, 0].set(ylim=(0, 105), ylabel='Feasible + safe + sufficient progress (%)', title='Independent paired-group test')
    axes[1, 0].legend(fontsize=8)
    for name, history in histories.items():
        axes[1, 1].plot([entry['step'] for entry in history],
                       [entry['validation']['sampled_position_rmse_m'] for entry in history],
                       color=colors[name], label=labels[name])
    axes[1, 1].set(xlabel='Training steps', ylabel='Sampled trajectory RMSE (m)', title='Validation only: checkpoint selection')
    axes[1, 1].grid(alpha=.2)
    axes[1, 1].legend(fontsize=8)
    figure.suptitle('Temporal DiT pilot: fixed physical clock, one candidate per condition', fontsize=14)
    figure.savefig(args.output.with_suffix('.png'), dpi=160)
    plt.close(figure)
    print(json.dumps({'report': str(args.output), 'expand': compact['gate']['expand'], 'checks': compact['gate']['checks']}))


if __name__ == '__main__':
    main()
