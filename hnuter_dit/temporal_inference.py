"""Strict inference for newly trained fixed-time temporal DiT checkpoints.

Sampling runs the learned model; there is no geometric prior, collision
guidance, endpoint clamp, or retiming. A trajectory is not certified safe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .model import DiffusionSchedule
from .geometry_observations import input_spec, preprocessing_configuration, SCHEMAS, ROI_RADIUS
from .temporal_model import (TemporalDiT, CHECKPOINT_SCHEMA, CONDITION_SCHEMA,
    HISTORY_SCHEMA, TRAJECTORY_SCHEMA, FUTURE_TIMES, HISTORY_TIMES, FREE_CONTROLS)

DIFFUSION_STEPS = 100


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(history_tokens, history_mask, history_times, condition, input_representation="box"):
    spec = input_spec(input_representation)
    tokens = np.asarray(history_tokens, dtype=None if input_representation == "voxel" else np.float32)
    mask = np.asarray(history_mask)
    ages = np.asarray(history_times, dtype=np.float32)
    current = np.asarray(condition, dtype=np.float32)
    single = tokens.ndim == len(spec["tokens_shape"])
    if single:
        tokens, mask, current = tokens[None], mask[None], current[None]
    batch = len(tokens)
    if (batch == 0 or tokens.shape != (batch, *spec['tokens_shape'])
            or mask.shape != (batch, *spec['mask_shape']) or current.shape != (batch, 12)):
        raise ValueError('history tokens/mask shapes do not match ' + input_representation)
    if mask.dtype != np.bool_:
        raise ValueError('history_mask must be boolean; missing returns must stay unknown')
    if ages.shape == (4,):
        ages = np.broadcast_to(ages, (batch, 4)).copy()
    if ages.shape != (batch, 4) or not np.allclose(ages, HISTORY_TIMES, atol=1e-5, rtol=0):
        raise ValueError('history_times must be [-.6,-.4,-.2,0] relative to current state')
    if not all(np.isfinite(value).all() for value in (tokens, current, ages)):
        raise ValueError('temporal observations and condition must be finite')
    if input_representation == 'box':
        active = tokens[mask]
        if len(active) and (np.any(active[:, 3:6] <= 0) or np.any(np.linalg.norm(active[:, 6:], axis=-1) < 1e-6)):
            raise ValueError('observed boxes require positive full lengths and nonzero wxyz rotation')
        if len(active) and not np.allclose(np.linalg.norm(active[:, 6:], axis=-1), 1., atol=1e-3, rtol=0):
            raise ValueError('observed box quaternions must be unit wxyz rotations')
    elif input_representation == 'point':
        if np.any(np.linalg.norm(tokens[mask], axis=-1) > ROI_RADIUS+1e-5):
            raise ValueError('measured points must lie inside the canonical 8m ROI')
    else:
        # Validate per row to keep temporary masks bounded even for a dataset.
        for grid, known in zip(tokens, mask):
            if (np.any((grid != 0)&(grid != 1)) or np.any((grid[:,0] != 0)&(grid[:,1] != 0))
                    or not np.array_equal(known, (grid[:,0]+grid[:,1]).astype(bool))
                    or not np.array_equal(grid[:,2], (~known).astype(grid.dtype))):
                raise ValueError('voxel channels must be disjoint occupied/free/unknown and mask must mark known cells')
        tokens = tokens.astype(np.uint8, copy=False)
    first, second = current[:, 6:9], current[:, 9:12]
    if not (np.allclose(np.linalg.norm(first, axis=-1), 1., atol=1e-3, rtol=0)
            and np.allclose(np.linalg.norm(second, axis=-1), 1., atol=1e-3, rtol=0)
            and np.allclose((first*second).sum(axis=-1), 0., atol=1e-3, rtol=0)):
        raise ValueError('condition rotation6D must contain two orthonormal rotation columns')
    return tokens, mask, ages, current, single


@torch.no_grad()
def sample_model(model, schedule, history_tokens, history_mask, history_times,
                 condition, seed, steps=20):
    """DDIM eta=0 in normalized control space, retaining the physical clock."""
    if not isinstance(steps, int) or not 1 <= steps <= DIFFUSION_STEPS:
        raise ValueError('steps must be an integer between 1 and 100')
    device = condition.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    value = torch.randn((len(condition), FREE_CONTROLS, 3), generator=generator, device=device)
    observations, valid = model.encode_history(history_tokens, history_mask, history_times)
    context = model.condition_context(condition)
    indices = torch.linspace(DIFFUSION_STEPS - 1, 0, steps, device=device).round().long()
    for index, timestep in enumerate(indices):
        t = timestep.expand(len(condition))
        prediction = model.forward_with_context(value, t, context, observations, valid)
        alpha = schedule.alpha_bars[timestep]
        clean = alpha.sqrt() * value - (1-alpha).sqrt() * prediction
        noise = (1-alpha).sqrt() * value + alpha.sqrt() * prediction
        if index + 1 < len(indices):
            previous = schedule.alpha_bars[indices[index+1]]
            value = previous.sqrt() * clean + (1-previous).sqrt() * noise
        else:
            value = clean
    controls = model.full_controls(value, condition)
    return dict(control_points=controls, **model.decode(controls))


class TemporalRuntime:
    def __init__(self, checkpoint, device='cuda'):
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f'temporal checkpoint not found: {self.checkpoint_path}')
        self.checkpoint_sha256 = file_sha256(self.checkpoint_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        payload = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        self.input_representation = payload.get('input_representation', 'box')
        self.input_spec = input_spec(self.input_representation)
        geometry_config = preprocessing_configuration(self.input_representation)
        if self.input_representation != 'box' and payload.get('geometry_config') != geometry_config:
            raise ValueError('checkpoint geometry preprocessing configuration does not match this runtime')
        expected = dict(schema=CHECKPOINT_SCHEMA, model_type='temporal_dit_bspline',
            condition_schema=CONDITION_SCHEMA, history_schema=SCHEMAS[self.input_representation],
            trajectory_schema=TRAJECTORY_SCHEMA, prediction_type='velocity',
            diffusion_steps=DIFFUSION_STEPS)
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f'incompatible temporal checkpoint {key}: expected {value!r}')
        if payload.get('future_times') != FUTURE_TIMES.tolist() or payload.get('history_times') != HISTORY_TIMES.tolist():
            raise ValueError('checkpoint physical time grids do not match this runtime')
        if payload.get('normalization', {}).get('fit_split') != 'train':
            raise ValueError('checkpoint must contain training-only normalization provenance')
        if payload.get('training_state') not in ('untrained_initial', 'trained') or payload.get('training_step', -1) < 0:
            raise ValueError('checkpoint training state is invalid')
        self.model = TemporalDiT(**payload['architecture'], history_mode=payload['history_mode'], input_representation=self.input_representation)
        self.model.set_normalization(payload['normalization'])
        expected_norm = {name: value.clone() for name, value in self.model.named_buffers() if name.endswith(('_mean', '_std'))}
        self.model.load_state_dict(payload['model_state'], strict=True)
        for name, value in expected_norm.items():
            if not torch.equal(value, getattr(self.model, name)):
                raise ValueError('checkpoint normalization metadata differs from model buffers')
        if not all(bool(torch.isfinite(value).all()) for value in self.model.state_dict().values()):
            raise ValueError('checkpoint model has nonfinite weights')
        self.model.to(self.device).eval()
        self.schedule = DiffusionSchedule.cosine(DIFFUSION_STEPS, self.device)
        self.payload = {key: payload[key] for key in ('schema', 'architecture', 'history_mode',
            'training_step', 'training_state', 'source_dataset', 'normalization')}
        self.payload.update(input_representation=self.input_representation, geometry_config=geometry_config,
            preprocessing_sha256=geometry_config['configuration_sha256'], input_spec=self.input_spec,
            training_budget={key: value for key, value in payload.get('configuration', {}).items() if key != 'device'})
        self.startup_warmup_seconds = 0.
        if self.device.type == 'cuda':
            # Complete CUDA initialization before the IPC worker reports ready.
            dummy_tokens = np.zeros(self.input_spec['tokens_shape'], dtype=self.input_spec['tokens_dtype'])
            if self.input_representation == 'voxel':
                dummy_tokens[:,2] = 1
            dummy_condition = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], np.float32)
            start = time.perf_counter()
            self.sample(dummy_tokens, np.zeros(self.input_spec['mask_shape'], bool), HISTORY_TIMES, dummy_condition, seed=0)
            self.startup_warmup_seconds = time.perf_counter() - start

    def describe(self):
        return dict(self.payload, checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_path=str(self.checkpoint_path), future_times=FUTURE_TIMES.tolist(),
            history_times=HISTORY_TIMES.tolist(), start_velocity_support=dict(
                representation='Exact initial derivative; this alone is not evidence of moving-start training.',
                training='stationary anchors only' if self.payload['source_dataset'].get('stationary_start_only') else 'see source_dataset',
                max_training_start_speed_mps=self.payload['source_dataset'].get('max_training_start_speed_mps')),
            position_only=True, no_retiming=True, control_points=16, spline_degree=3,
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            encoder_parameter_count=self.model.encoder_parameter_count(),
            torch_version=torch.__version__, device=str(self.device),
            startup_warmup_seconds=self.startup_warmup_seconds)

    def sample(self, history_tokens, history_mask, history_times, condition, seed, steps=20):
        tokens, mask, ages, current, single = validate_inputs(history_tokens, history_mask, history_times, condition, self.input_representation)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        tensors = [torch.as_tensor(value, device=self.device) for value in (tokens, mask, ages, current)]
        with torch.inference_mode():
            result = sample_model(self.model, self.schedule, *tensors, seed=seed, steps=steps)
        arrays = {key: value.detach().cpu().numpy() for key, value in result.items()}
        elapsed = time.perf_counter() - start
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError('temporal model produced nonfinite output')
        if single:
            arrays = {key: value[0] for key, value in arrays.items()}
        return dict(arrays, future_times=FUTURE_TIMES.copy(), inference_seconds=elapsed,
            metadata=dict(checkpoint_sha256=self.checkpoint_sha256, schema=CHECKPOINT_SCHEMA,
                input_representation=self.input_representation, preprocessing_sha256=self.payload['preprocessing_sha256'],
                training_step=self.payload['training_step'], training_state=self.payload['training_state'],
                source_dataset=self.payload['source_dataset'], history_mode=self.model.history_mode,
                seed=int(seed), diffusion_steps=DIFFUSION_STEPS, model_evaluations=steps,
                batch_size=len(tokens), timing_includes_transfer=True,
                position_only=True, retimed=False, collision_certified=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='NPZ with only causal history_* and condition used')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args(argv)
    runtime = TemporalRuntime(args.checkpoint, args.device)
    with np.load(args.input, allow_pickle=False) as data:
        result = runtime.sample(*(data[name] for name in ('history_tokens', 'history_mask', 'history_times', 'condition')),
                                seed=args.seed, steps=args.steps)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{key: value for key, value in result.items() if isinstance(value, np.ndarray)})
    output.with_suffix('.json').write_text(json.dumps(dict(runtime.describe(),
        sample=result['metadata'], inference_seconds=result['inference_seconds']), indent=2) + '\n')


if __name__ == '__main__':
    main()
