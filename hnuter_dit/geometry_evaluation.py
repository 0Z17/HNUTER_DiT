"""Controlled geometry-encoder comparison with validation-only model selection.

The encoder is the intervention. Data, temporal decoder, training budget and
sampling seeds are held fixed. Future geometry is used only by the evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .temporal_data import load_dataset, native_rollout, sample_from_dataset
from .temporal_evaluation import (file_sha256, paired_group_difference,
                                  score_trajectory, summarize_rows)

REPRESENTATIONS = ('box', 'point', 'voxel')
NEW_REPRESENTATIONS = ('point', 'voxel')
SCHEMA = 'geometry_encoder_comparison_v1'
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1]/'docs/geometry_encoder_protocol.json'
PROTOCOL = dict(
    training_steps=2000, batch_size=32, training_seed=7,
    sampling_seed=20404, ddim_steps=20, candidate_count=1,
    maximum_speed_mps=.5, maximum_acceleration_mps2=.6,
    proxy_clearance_margin_m=.04, trajectory_controls=16,
    horizon_seconds=8., native_episodes_per_representation=8,
    selection='validation only: highest group-balanced feasible-safe-progress rate; then lowest position RMSE; then point before voxel',
    uncertainty='paired whole-group bootstrap; 5000 resamples; no window-level independence assumption',
    future_truth='evaluator only; never candidate selection or student condition',
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_protocol_dataset(freeze, protocol):
    if (freeze.get('seed') != protocol['dataset']['collection_seed']
            or freeze.get('requested_groups') != protocol['dataset']['groups_requested']):
        raise ValueError('dataset collection seed/size differs from the fresh preregistered pilot')
    if (freeze.get('samples') != 192 or freeze.get('independent_groups') != 96
            or freeze.get('split_samples') != {'train':128, 'validation':32, 'test':32}):
        raise ValueError('controlled comparison requires all 96 accepted pairs and 128/32/32 split samples')


def split_data(data, indices):
    """Slice before preprocessing; validation does not inspect test observations."""
    constants = {'history_times', 'future_times', 'ray_directions', '_freeze'}
    count = len(data['positions'])
    result = {}
    for key, value in data.items():
        if key in constants:
            result[key] = value
        elif isinstance(value, np.ndarray) and value.ndim and len(value) == count:
            result[key] = value[indices].copy()
        else:
            raise ValueError('unrecognized dataset array partition: '+key)
    return result


def validate_comparison(descriptions, dataset_sha256):
    """Reject weights from different data, budgets or shared trajectory heads."""
    if set(descriptions) != set(REPRESENTATIONS):
        raise ValueError('box, point and voxel descriptions are required')
    first = descriptions['box']
    required_budget = {'steps': 2000, 'batch_size': 32, 'seed': 7, 'learning_rate': .0002,
                       'ema_decay': .99, 'validation_interval': 100, 'sampling_steps': 20}
    required_architecture = dict(dimension=128, depth=4, heads=4, history_depth=2)
    for name in REPRESENTATIONS:
        desc = descriptions[name]
        if (desc.get('input_representation') != name or desc.get('history_mode') != 'temporal'
                or desc.get('training_state') != 'trained' or desc.get('training_step', 0) <= 0
                or desc.get('source_dataset', {}).get('dataset_sha256') != dataset_sha256):
            raise ValueError(name+' checkpoint representation/training/dataset mismatch')
        budget = desc.get('training_budget', {})
        if any(budget.get(k) != v for k, v in required_budget.items()):
            raise ValueError(name+' checkpoint does not have the declared training budget')
        # Representation names are the intended intervention, not a budget.
        comparable = {k: v for k, v in budget.items() if k != 'input_representation'}
        baseline_budget = {k: v for k, v in first.get('training_budget', {}).items() if k != 'input_representation'}
        if comparable != baseline_budget:
            raise ValueError('comparison training budgets differ')
        if desc.get('architecture') != first.get('architecture'):
            raise ValueError('shared DiT/head architectures differ')
        if desc.get('architecture') != required_architecture:
            raise ValueError('shared architecture differs from the preregistered model')
        if desc.get('control_points') != 16 or desc.get('future_times') != (np.arange(65)*.125).tolist():
            raise ValueError('comparison must retain the 16-control, fixed 8-second head')
        if not isinstance(desc.get('geometry_config'), dict) or not desc['geometry_config']:
            raise ValueError('missing geometry preprocessing configuration')
        if desc.get('encoder_parameter_count', 0) <= 0 or desc.get('parameter_count', 0) < desc['encoder_parameter_count']:
            raise ValueError('missing/invalid encoder parameter counts')
    return True


def select_new_encoder(variants):
    """Only call on validation rows; no test statistic is accepted here."""
    for name in REPRESENTATIONS:
        if variants[name].get('split') != 'validation':
            raise ValueError('encoder selection requires validation-only results')
    def key(name):
        summary = variants[name]['summary']
        rate = summary['group_balanced_feasible_safe_progress_rate']
        error = summary['position_rmse_m']
        if not np.isfinite(rate) or not np.isfinite(error):
            raise ValueError('nonfinite validation selection metric')
        return (-rate, error, NEW_REPRESENTATIONS.index(name))
    chosen = min(NEW_REPRESENTATIONS, key=key)
    selected_rate = variants[chosen]['summary']['group_balanced_feasible_safe_progress_rate']
    box_rate = variants['box']['summary']['group_balanced_feasible_safe_progress_rate']
    return dict(representation=chosen, rule=PROTOCOL['selection'], selected_before_test=True,
                validation_rate=selected_rate, box_validation_rate=box_rate,
                validation_gain_over_box=selected_rate-box_rate,
                improvement_over_box_observed=bool(selected_rate > box_rate),
                selection_is_not_success_certificate=True)


def _prepare(data, representation):
    from .geometry_observations import reencode_dataset
    return reencode_dataset(data, representation)


def _runtime(checkpoint, device):
    from .temporal_inference import TemporalRuntime
    return TemporalRuntime(checkpoint, device=device)


def evaluate_variant(runtime, encoded, indices, *, seed=20404, steps=20, frozen_history=False):
    sample_data = {key: value for key, value in encoded.items() if not key.startswith('_') or key == '_freeze'}
    rows, predicted, controls = [], [], []
    for local, index in enumerate(indices):
        sample = sample_from_dataset(sample_data, local)
        tokens, masks = sample['history_tokens'].copy(), sample['history_mask'].copy()
        if frozen_history:
            tokens[...] = tokens[-1]
            masks[...] = masks[-1]
        sample_seed = seed + int(sample['group_id'])*1000
        began = time.perf_counter()
        result = runtime.sample(tokens, masks, sample['history_times'], sample['condition'],
                                seed=sample_seed, steps=steps)
        elapsed = time.perf_counter()-began
        positions, times = np.asarray(result['positions']), np.asarray(result['future_times'])
        cp = np.asarray(result['control_points'])
        if times.shape != (65,) or not np.allclose(times, sample['future_times'], atol=1e-8, rtol=0):
            raise ValueError('runtime changed the physical prediction clock')
        if positions.shape != (65, 3) or cp.shape != (16, 3) or not np.isfinite(cp).all():
            raise ValueError('runtime output shape/control validity mismatch')
        row = score_trajectory(positions, times, sample['condition'], sample['positions'],
                               sample['gt_boxes'], sample['gt_mask'], control_points=cp)
        row.update(index=int(index), group_id=int(sample['group_id']), episode_id=int(sample['episode_id']),
                   pair_variant=int(sample['pair_variant']), sampling_seed=sample_seed,
                   sampling_seconds=elapsed, model_reported_seconds=float(result['inference_seconds']),
                   position_rmse_m=float(np.sqrt(np.mean(np.sum((positions-sample['positions'])**2, axis=-1)))))
        rows.append(row)
        predicted.append(positions)
        controls.append(cp)
    summary = summarize_rows(rows)
    summary['position_rmse_m'] = float(np.sqrt(np.mean([r['position_rmse_m']**2 for r in rows])))
    return dict(rows=rows, summary=summary), np.stack(predicted), np.stack(controls)


def _native_summary(rows):
    completed = [r.get('completed') is True and r.get('steps') == 8000 and
                 r.get('simulation_seconds') == 8. and r.get('no_retiming') is True for r in rows]
    known_contacts = [r['collision_detected'] for r in rows if isinstance(r.get('collision_detected'), (bool, np.bool_))]
    tracking = [float(r['position_tracking_max_error_m']) for r in rows
                if r.get('position_tracking_max_error_m') is not None and np.isfinite(r['position_tracking_max_error_m'])]
    complete = len(rows) == 8 and all(completed)
    all_evidence = len(known_contacts) == 8 and len(tracking) == 8
    return dict(episodes=len(rows), expected_episodes=8, complete=complete,
        failed_or_incomplete_episodes=sum(not value for value in completed),
        contact_known_episodes=len(known_contacts), contact_unknown_episodes=len(rows)-len(known_contacts),
        collision_rate=float(np.mean(known_contacts)) if known_contacts else None,
        collision_rate_denominator='episodes with explicit actual-contact evidence only; failures remain unknown',
        maximum_tracking_error_m=max(tracking) if tracking else None,
        physical_validation_passed=bool(complete and all_evidence and not any(known_contacts) and max(tracking) <= .12),
        goal_reached_rate=float(np.mean([r.get('goal_reached') is True for r in rows])) if rows else None,
        note='Fixed 8-second physical prefix; a valid learned wait need not reach the goal within this horizon.', rows=rows)


def native_results(data, indices, positions, controls, output):
    """Attempt every fixed sample; an exception is failed evidence, not contact."""
    for local, index in enumerate(indices[:8]):
        sample = sample_from_dataset(data, int(index))
        directory = Path(output)/f'sample_{index:04d}'
        try:
            result = native_rollout(sample, positions[local], sample['future_times'],
                                    control_points=controls[local], output_dir=directory)
        except Exception as exc:
            result = dict(completed=False, collision_detected=None, goal_reached=None,
                          error=f'{type(exc).__name__}: {exc}', physical_validation_passed=False,
                          contact_status='unknown: rollout did not return complete contact evidence')
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory/'evaluation_failure.json', result)
        yield dict(index=int(index), group_id=int(sample['group_id']), **result)


def run_evaluation(args):
    if args.command == 'test' and args.native_count != 8:
        raise ValueError('formal geometry comparison requires eight fixed native episodes per representation')
    data = load_dataset(args.dataset, verify=True)
    protocol_path = Path(args.protocol).resolve()
    preregistered = json.loads(protocol_path.read_text())
    if preregistered.get('schema') != 'geometry_encoder_pilot_protocol_v1':
        raise ValueError('unrecognized preregistered geometry protocol')
    validate_protocol_dataset(data['_freeze'], preregistered)
    protocol_digest = file_sha256(protocol_path)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selection_path = Path(args.selection).resolve() if args.command == 'test' else None
    if selection_path:
        previous = json.loads(selection_path.read_text())
        if (previous.get('schema') != SCHEMA or previous.get('phase') != 'validation'
                or previous.get('completed') is not True or not previous.get('selected_new_encoder', {}).get('selected_before_test')):
            raise ValueError('test requires a completed validation selection report')
        if previous['dataset_sha256'] != data['_freeze']['dataset_sha256']:
            raise ValueError('selection dataset SHA differs from test dataset')
        if previous['preregistered_protocol']['sha256'] != protocol_digest:
            raise ValueError('preregistered protocol changed after validation selection')
        checkpoints = {name: previous['variants'][name]['checkpoint'] for name in REPRESENTATIONS}
        for name, path in checkpoints.items():
            if file_sha256(path) != previous['variants'][name]['checkpoint_sha256']:
                raise ValueError('selected checkpoint changed: '+name)
        seed, steps = previous['sampling_seed'], previous['ddim_steps']
        phase, split = 'test', 2
    else:
        checkpoints = {name: str(Path(getattr(args, name)).resolve()) for name in REPRESENTATIONS}
        seed, steps, phase, split = args.seed, args.steps, 'validation', 1
    if seed != preregistered['evaluation']['sampling_seed'] or steps != preregistered['shared_training']['sampling_steps']:
        raise ValueError('sampling settings differ from the preregistered protocol')
    indices = np.flatnonzero(data['split'] == split)
    if not len(indices):
        raise ValueError('requested split has no samples')
    report = dict(schema=SCHEMA, phase=phase, completed=False, protocol=PROTOCOL,
                  dataset_path=str(Path(args.dataset).resolve()), dataset_sha256=data['_freeze']['dataset_sha256'],
                  dataset_freeze=data['_freeze'], sampling_seed=seed, ddim_steps=steps,
                  preregistered_protocol=dict(path=str(protocol_path), sha256=protocol_digest, content=preregistered),
                  variants={}, native={}, timing='sampling includes input tensor transfer and model output transfer; raw-ray preprocessing separately measured as a split batch; excludes IPC, simulation, rendering',
                  collision_metric='20 ms timed 42-sphere/OBB proxy; native contacts reported separately',
                  test_used_for_model_selection=False, automatic_expansion_authorized=False)
    if selection_path:
        report.update(selection_report=str(selection_path), selection_sha256=file_sha256(selection_path),
                      selected_new_encoder=previous['selected_new_encoder'])
    write_json(output/'protocol.json', report)
    predictions, control_points, descriptions = {}, {}, {}
    subset = split_data(data, indices)
    for name in REPRESENTATIONS:
        before = time.perf_counter()
        encoded = _prepare(subset, name)
        preprocessing_seconds = time.perf_counter()-before
        runtime = _runtime(checkpoints[name], args.device)
        desc = runtime.describe()
        descriptions[name] = desc
        digest = file_sha256(checkpoints[name])
        if desc.get('checkpoint_sha256') != digest:
            raise ValueError('runtime checkpoint identity mismatch')
        variant, predictions[name], control_points[name] = evaluate_variant(runtime, encoded, indices, seed=seed, steps=steps)
        variant.update(split=phase, checkpoint=checkpoints[name], checkpoint_sha256=digest,
                       runtime=desc, geometry_config_sha256=config_hash(desc['geometry_config']),
                       preprocessing_split_batch_seconds=preprocessing_seconds,
                       preprocessing_timing_scope=('loads already saved box tokens; excludes their collection-time construction'
                           if name == 'box' else 'reconstructs split raw rays and encodes history on CPU as one batch'))
        report['variants'][name] = variant
        if phase == 'test' and name in NEW_REPRESENTATIONS:
            ablation, predictions[name+'_frozen_history'], control_points[name+'_frozen_history'] = evaluate_variant(
                runtime, encoded, indices, seed=seed, steps=steps, frozen_history=True)
            ablation.update(split=phase, checkpoint=checkpoints[name], checkpoint_sha256=digest,
                            ablation='all four geometry frames/masks replaced with latest; physical history ages unchanged')
            report['variants'][name+'_frozen_history'] = ablation
        write_json(output/'report.json', report)
        print(json.dumps(dict(phase=phase, variant=name, **variant['summary'])), flush=True)
        del runtime, encoded
    validate_comparison(descriptions, report['dataset_sha256'])
    # Preserve all candidates before physical execution, even if a particular
    # native rollout fails or the evaluator is later interrupted.
    np.savez_compressed(output/'predictions.npz', indices=indices, **predictions,
                        **{name+'_control_points': value for name, value in control_points.items()})
    if phase == 'validation':
        report['selected_new_encoder'] = select_new_encoder(report['variants'])
    else:
        for name in REPRESENTATIONS:
            old = previous['variants'][name]
            if old['geometry_config_sha256'] != report['variants'][name]['geometry_config_sha256']:
                raise ValueError('preprocessing changed after validation selection')
        report['paired_comparisons'] = {name+'_minus_box': paired_group_difference(
            report['variants'][name]['rows'], report['variants']['box']['rows']) for name in NEW_REPRESENTATIONS}
        report['history_ablations'] = {name: paired_group_difference(
            report['variants'][name]['rows'], report['variants'][name+'_frozen_history']['rows']) for name in NEW_REPRESENTATIONS}
        # Fixed first N test samples, including both members of each group and
        # failed offline predictions. No score-based subsampling.
        for name in REPRESENTATIONS:
            rows = []
            for row in native_results(data, indices, predictions[name], control_points[name], output/'native'/name):
                rows.append(row)
                report['native'][name] = _native_summary(rows)
                write_json(output/'report.json', report)
    report.update(completed=True)
    write_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('select', 'test'):
        command = sub.add_parser(name)
        command.add_argument('--dataset', required=True)
        command.add_argument('--output', required=True)
        command.add_argument('--device', default='cuda')
        command.add_argument('--protocol', default=str(DEFAULT_PROTOCOL))
        if name == 'select':
            for rep in REPRESENTATIONS:
                command.add_argument('--'+rep, required=True)
            command.add_argument('--seed', type=int, default=20404)
            command.add_argument('--steps', type=int, default=20)
        else:
            command.add_argument('--selection', required=True, help='Completed validation report.json')
            command.add_argument('--native-count', type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == 'test' and args.native_count != 8:
        parser.error('native-count must be 8 under the preregistered comparison protocol')
    run_evaluation(args)


if __name__ == '__main__':
    main()
