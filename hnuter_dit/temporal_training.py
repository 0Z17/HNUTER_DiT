"""Train a new causal temporal DiT; old static v003 weights are untouched.

Only training rows fit normalization or gradients. Validation selects best.pt;
test rows are excluded even from the representation audit and validation logs.
No future obstacle state, teacher schedule, episode clock, or object identity
is a model input. Targets retain their physical time, including waits.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .model import DiffusionSchedule
from .geometry_observations import reencode_dataset, preprocessing_configuration, SCHEMAS
from .temporal_inference import DIFFUSION_STEPS, file_sha256, sample_model, validate_inputs
from .temporal_model import (TemporalDiT, CHECKPOINT_SCHEMA, CONDITION_SCHEMA,
    HISTORY_SCHEMA, TRAJECTORY_SCHEMA, FUTURE_TIMES, HISTORY_TIMES,
    fit_control_points, decode_control_points, normalization_from_training)


def _atomic_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _atomic_checkpoint(path, payload):
    temporary = Path(str(path) + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _state_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _quantiles(value):
    return dict(zip(('min', 'p50', 'p90', 'p95', 'max'),
                    np.quantile(value, [0, .5, .9, .95, 1]).tolist()))


def audit_representation(positions, controls, splits):
    """Training/validation representation audit, never uses held-out test rows."""
    report = {}
    for label, number in (('train', 0), ('validation', 1)):
        indices = np.flatnonzero(splits == number)
        fitted = decode_control_points(controls[indices])['positions']
        error = np.linalg.norm(fitted-positions[indices], axis=-1)
        dense = decode_control_points(controls[indices], np.linspace(0, 8, 1025))
        speed = np.linalg.norm(dense['velocities'], axis=-1).max(axis=-1)
        acceleration = np.linalg.norm(dense['accelerations'], axis=-1).max(axis=-1)
        report[label] = dict(rows=len(indices), position_rmse_m=float(np.sqrt(np.mean(error**2))),
            position_max_error_m=float(error.max()), position_row_rmse_m=_quantiles(np.sqrt((error**2).mean(axis=-1))),
            max_speed_m_s=_quantiles(speed), max_acceleration_m_s2=_quantiles(acceleration),
            fraction_over_speed_0_5=float(np.mean(speed > .5 + 1e-6)),
            fraction_over_acceleration_0_6=float(np.mean(acceleration > .6 + 1e-6)))
    return dict(splits=report, dense_samples=1025, endpoint_velocity_exact=True,
        final_goal_constrained=False, retimed=False, test_rows_used=False,
        derivative_bounds='Dense audit; execution/evaluation separately use analytic extrema.')


def _validate_data(data, input_representation="box"):
    required = ('history_tokens', 'history_mask', 'history_times', 'condition', 'positions', 'future_times', 'split', 'group_id')
    if any(key not in data for key in required):
        raise ValueError('dataset is missing required temporal arrays')
    positions = np.asarray(data['positions'], dtype=np.float32)
    split = np.asarray(data['split'])
    if positions.shape != (len(split), 65, 3) or not np.isfinite(positions).all() or np.any(~np.isin(split, [0, 1, 2])):
        raise ValueError('invalid positions or split labels')
    if not np.array_equal(np.asarray(data['future_times']), FUTURE_TIMES):
        raise ValueError('future_times must be the fixed 65-point 8-second grid')
    validate_inputs(*(data[key] for key in ('history_tokens', 'history_mask', 'history_times', 'condition')), input_representation=input_representation)
    for group in np.unique(data['group_id']):
        if len(np.unique(split[np.asarray(data['group_id']) == group])) != 1:
            raise ValueError('a group is shared between data splits')
    if not np.any(split == 0) or not np.any(split == 1):
        raise ValueError('training and validation must both be nonempty')
    return split, positions


def diffusion_loss(model, schedule, batch, generator, position_weight=.2,
                   velocity_weight=.02, acceleration_weight=.002):
    """Supervised v prediction plus decoded position/derivative reconstruction."""
    clean = (batch['controls'][:, 2:] - model.control_mean) / model.control_std
    t = torch.randint(DIFFUSION_STEPS, (len(clean),), device=clean.device, generator=generator)
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    alpha = schedule.alpha_bars[t, None, None]
    noisy = alpha.sqrt() * clean + (1-alpha).sqrt() * noise
    target = alpha.sqrt() * noise - (1-alpha).sqrt() * clean
    prediction = model(noisy, t, batch['condition'], batch['history_tokens'], batch['history_mask'], batch['history_times'])
    denoising = (prediction-target).square().mean()
    predicted_clean = alpha.sqrt() * noisy - (1-alpha).sqrt() * prediction
    decoded = model.decode(model.full_controls(predicted_clean, batch['condition']))
    teacher = model.decode(batch['controls'])
    # Physically validated teacher references supervise position; fitted derivatives
    # supervise shape without inventing a new temporal resampling of the label.
    position = (decoded['positions']-batch['positions']).square().mean()
    velocity = (decoded['velocities']-teacher['velocities']).square().mean()
    acceleration = (decoded['accelerations']-teacher['accelerations']).square().mean()
    total = denoising + position_weight*position + velocity_weight*velocity + acceleration_weight*acceleration
    return total, dict(total=total, diffusion_v_mse=denoising, position_mse=position,
                       velocity_mse=velocity, acceleration_mse=acceleration)


@torch.no_grad()
def _validation(model, schedule, tensors, indices, batch_size, seed, steps):
    model.eval()
    sums = dict(diffusion_v_mse=0., sampled_position_squared_error=0.)
    for offset in range(0, len(indices), batch_size):
        chosen = indices[offset:offset+batch_size]
        batch = {key: value[chosen].to(next(model.parameters()).device) for key, value in tensors.items()}
        generator = torch.Generator(device=batch['condition'].device).manual_seed(seed+offset)
        _, loss = diffusion_loss(model, schedule, batch, generator)
        sampled = sample_model(model, schedule, batch['history_tokens'], batch['history_mask'],
            batch['history_times'], batch['condition'], seed=seed+offset, steps=steps)
        sums['diffusion_v_mse'] += float(loss['diffusion_v_mse']) * len(chosen)
        sums['sampled_position_squared_error'] += float((sampled['positions']-batch['positions']).square().sum(-1).mean()) * len(chosen)
    return dict(diffusion_v_mse=sums['diffusion_v_mse']/len(indices),
        sampled_position_rmse_m=float(np.sqrt(sums['sampled_position_squared_error']/len(indices))),
        rows=len(indices), sampling_steps=steps, fixed_seed=seed, split='validation')


def train_arrays(data, output_dir, *, source_dataset=None, history_mode='temporal',
                 steps=2000, batch_size=32, learning_rate=2e-4, seed=7, device='cuda',
                 dimension=128, depth=4, heads=4, history_depth=2,
                 validation_interval=100, sampling_steps=20, ema_decay=.99,
                 position_weight=.2, velocity_weight=.02, acceleration_weight=.002, input_representation='box'):
    """Train a fresh model. No resume/overwrite: each output is a separate run."""
    if steps < 1 or batch_size < 1 or validation_interval < 1 or not 0 <= ema_decay < 1:
        raise ValueError('invalid training budget or EMA setting')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('initial.pt', 'best.pt', 'last.pt')):
        raise FileExistsError('output already has checkpoints; choose a fresh run directory')
    preprocessing_started = time.perf_counter()
    if input_representation != 'box':
        if data.get('_input_representation') == input_representation:
            if data.get('_preprocessing') != preprocessing_configuration(input_representation):
                raise ValueError('cached geometry preprocessing configuration does not match selected representation')
        else:
            data = reencode_dataset(data, input_representation)
    preprocessing_seconds = time.perf_counter()-preprocessing_started
    split, positions = _validate_data(data, input_representation)
    geometry_config = preprocessing_configuration(input_representation)
    train_indices, val_indices = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    # Only train/validation targets are ever fit or loaded into model tensors.
    usable = np.flatnonzero(split != 2)
    condition = np.asarray(data['condition'], dtype=np.float32)
    controls = np.zeros((len(split), 16, 3), np.float32)
    controls[usable] = fit_control_points(positions[usable], condition[usable, 3:6])
    audit = audit_representation(positions, controls, split)
    _atomic_json(output/'representation_audit.json', audit)
    normalization_mask = np.asarray(data['history_mask']).copy()
    if history_mode == 'latest':
        normalization_mask[:, :-1] = False
    normalization = normalization_from_training(controls, condition, data['history_tokens'],
                                                  normalization_mask, train_indices, input_representation)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device)
    model = TemporalDiT(dimension, depth, heads, history_depth, history_mode, input_representation).to(device)
    model.set_normalization(normalization)
    schedule = DiffusionSchedule.cosine(DIFFUSION_STEPS, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    averaged = copy.deepcopy(model).eval()
    # Do not tensorize gt_* / teacher_* fields, indices, IDs or absolute time.
    age = np.asarray(data['history_times'], dtype=np.float32)
    if age.shape == (4,):
        age = np.broadcast_to(age, (len(split), 4)).copy()
    arrays = dict(history_tokens=data['history_tokens'], history_mask=data['history_mask'],
                  history_times=age, condition=condition, positions=positions, controls=controls)
    storage_device = torch.device('cpu') if input_representation == 'voxel' else device
    tensors = {key: torch.as_tensor(np.asarray(value)[usable].copy(), device=storage_device) for key, value in arrays.items()}
    remap = {old: new for new, old in enumerate(usable.tolist())}
    train_rows = np.array([remap[index] for index in train_indices])
    val_rows = np.array([remap[index] for index in val_indices])
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    configuration = dict(history_mode=history_mode, input_representation=input_representation,
        optimizer='AdamW', weight_decay=.01, gradient_clip_norm=1., steps=steps, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed, device=str(device), validation_interval=validation_interval,
        sampling_steps=sampling_steps, ema_decay=ema_decay, position_weight=position_weight,
        velocity_weight=velocity_weight, acceleration_weight=acceleration_weight)
    source_dataset = dict(source_dataset or data.get('_freeze', {}))
    source_dataset['max_training_start_speed_mps'] = float(np.linalg.norm(condition[train_indices, 3:6], axis=-1).max())
    shared = dict(schema=CHECKPOINT_SCHEMA, model_type='temporal_dit_bspline',
        condition_schema=CONDITION_SCHEMA, history_schema=SCHEMAS[input_representation], trajectory_schema=TRAJECTORY_SCHEMA,
        input_representation=input_representation, geometry_config=geometry_config,
        architecture=model.architecture, history_mode=history_mode, prediction_type='velocity',
        diffusion_steps=DIFFUSION_STEPS, future_times=FUTURE_TIMES.tolist(), history_times=HISTORY_TIMES.tolist(),
        normalization=normalization, source_dataset=source_dataset, configuration=configuration,
        initialization='Random weights with training-only normalization; normalization is an initialization prior.',
        input_allowlist=['history_tokens', 'history_mask', 'history_times', 'condition'],
        label_allowlist=['positions'], checkpoint_selection='validation sampled position RMSE; no test rows',
        torch_version=torch.__version__)
    initial_state = _state_cpu(model)
    initial_hash = _state_sha256(initial_state)
    _atomic_checkpoint(output/'initial.pt', dict(shared, model_state=initial_state,
        training_step=0, training_state='untrained_initial', weights_sha256=initial_hash))
    _atomic_json(output/'configuration.json', dict(shared, train_rows=len(train_rows), validation_rows=len(val_rows),
        test_rows_excluded=int(np.sum(split == 2)), parameter_count=sum(p.numel() for p in model.parameters()),
        encoder_parameter_count=model.encoder_parameter_count(), preprocessing_seconds=preprocessing_seconds,
        tensor_storage_device=str(storage_device)))
    print(json.dumps(dict(event='ready', output=str(output), representation_audit=audit,
                         parameter_count=sum(p.numel() for p in model.parameters()))), flush=True)
    started = time.perf_counter()
    best = float('inf')
    history = []
    accumulated = {}
    interval_start = time.perf_counter()
    for step in range(1, steps+1):
        model.train()
        chosen = rng.choice(train_rows, min(batch_size, len(train_rows)), replace=False)
        batch = {key: value[chosen].to(next(model.parameters()).device) for key, value in tensors.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = diffusion_loss(model, schedule, batch, generator, position_weight, velocity_weight, acceleration_weight)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for average, current in zip(averaged.parameters(), model.parameters()):
                average.lerp_(current, 1-ema_decay)
        for key, value in metrics.items():
            accumulated[key] = accumulated.get(key, 0.) + float(value.detach())
        if step % validation_interval == 0 or step == steps:
            validation = _validation(averaged, schedule, tensors, val_rows, batch_size, seed+100000, sampling_steps)
            interval_steps = step - (history[-1]['step'] if history else 0)
            record = dict(step=step, elapsed_seconds=time.perf_counter()-started,
                interval_seconds=time.perf_counter()-interval_start, gradient_norm=float(norm),
                train={key: value/interval_steps for key, value in accumulated.items()}, validation=validation)
            history.append(record)
            state = _state_cpu(averaged)
            checkpoint = dict(shared, model_state=state, training_step=step, training_state='trained',
                weights_sha256=_state_sha256(state), validation=validation, elapsed_seconds=record['elapsed_seconds'])
            if validation['sampled_position_rmse_m'] < best:
                best = validation['sampled_position_rmse_m']
                _atomic_checkpoint(output/'best.pt', checkpoint)
            _atomic_checkpoint(output/'last.pt', checkpoint)
            _atomic_json(output/'training_history.json', history)
            print(json.dumps(record), flush=True)
            accumulated = {}
            interval_start = time.perf_counter()
    summary = dict(configuration=configuration, best_validation_position_rmse_m=best,
        steps_completed=steps, elapsed_seconds=time.perf_counter()-started, preprocessing_seconds=preprocessing_seconds,
        initial_weights_sha256=initial_hash, last_weights_sha256=checkpoint['weights_sha256'],
        weights_changed=initial_hash != checkpoint['weights_sha256'], test_rows_used=False,
        checkpoint_files={name: file_sha256(output/name) for name in ('initial.pt', 'best.pt', 'last.pt')})
    _atomic_json(output/'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', '--output-dir', dest='output', required=True)
    parser.add_argument('--history-mode', choices=('temporal', 'latest'), default='temporal')
    parser.add_argument('--input-representation', choices=('box', 'point', 'voxel'), default='box')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dimension', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--history-depth', type=int, default=2)
    parser.add_argument('--validation-interval', type=int, default=100)
    parser.add_argument('--sampling-steps', type=int, default=20)
    parser.add_argument('--ema-decay', type=float, default=.99)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    from .temporal_data import load_dataset
    data = load_dataset(args.dataset, verify=True)
    source = dict(data['_freeze'])
    source['root'] = str(Path(args.dataset).resolve())
    source['dataset_sha256'] = file_sha256(Path(args.dataset)/source['dataset_file'])
    fields = vars(args).copy()
    fields.pop('dataset'); fields.pop('cpu_threads')
    output = fields.pop('output')
    train_arrays(data, output, source_dataset=source, **fields)


if __name__ == '__main__':
    main()
