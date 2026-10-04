#!/usr/bin/env python3
"""Latency-only replay of the frozen geometry pilot; no training or control changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)), max_ms=float(a.max()))


def reconstruct(data, index):
    frames = []
    for f in range(4):
        ranges = data['history_ray_ranges'][index, f]
        keep = np.isfinite(ranges)
        frames.append(SimpleNamespace(points=data['ray_directions'][f, keep]*ranges[keep, None],
            world_T_sensor=data['history_sensor_poses'][index, f], sim_time=float([0., .2, .4, .6][f])))
    origin, rotation = data['origin_world'][index], data['world_to_canonical'][index]
    condition = data['condition'][index]
    body = rotation.T @ np.column_stack((condition[6:9], condition[9:12],
                                         np.cross(condition[6:9], condition[9:12])))
    xyzw = Rotation.from_matrix(body).as_quat()
    state = dict(position=origin, velocity=rotation.T@condition[3:6],
                 quaternion=xyzw[[3, 0, 1, 2]])
    goal = origin + rotation.T@condition[:3]
    return frames, state, goal


def denoise_cached(runtime, observations, valid, context, condition, seed=20404):
    """Same 20 DDIM updates as sample_model; checked against that function below."""
    model, schedule = runtime.model, runtime.schedule
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    value = torch.randn((len(condition), 14, 3), generator=generator, device=condition.device)
    indices = torch.linspace(99, 0, 20, device=condition.device).round().long()
    for index, timestep in enumerate(indices):
        prediction = model.forward_with_context(value, timestep.expand(len(condition)), context, observations, valid)
        alpha = schedule.alpha_bars[timestep]
        clean = alpha.sqrt()*value - (1-alpha).sqrt()*prediction
        noise = (1-alpha).sqrt()*value + alpha.sqrt()*prediction
        if index+1 < len(indices):
            previous = schedule.alpha_bars[indices[index+1]]
            value = previous.sqrt()*clean + (1-previous).sqrt()*noise
        else:
            value = clean
    controls = model.full_controls(value, condition)
    return dict(control_points=controls, **model.decode(controls))


def timed_gpu(function):
    torch.cuda.synchronize()
    before = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, 1000*(time.perf_counter()-before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        raise ValueError('positive repeats and a new output directory are required')
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    from hnuter_dit.geometry_observations import encode_geometry_history
    from hnuter_dit.temporal_inference import TemporalRuntime, validate_inputs, sample_model
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark protocol')
    args.output.mkdir(parents=True)
    dataset = root/'results/geometry_pilot/collection96_seed107/dataset.npz'
    allowed = ('history_ray_ranges', 'history_sensor_poses', 'ray_directions', 'origin_world',
               'world_to_canonical', 'condition', 'split', 'group_id')
    with np.load(dataset, allow_pickle=False) as raw:
        data = {k: raw[k] for k in allowed}
    indices = np.flatnonzero(data['split'] == 2)
    assert len(indices) == 32
    windows = {int(i): reconstruct(data, int(i)) for i in indices}
    names = ['box', 'point', 'voxel']
    runtimes = {name: TemporalRuntime(root/f'artifacts/geometry_encoder_pilot_v1/models/{name}/best.pt') for name in names}
    records, stage_records, checks, payload_bytes = [], [], {}, {}
    def encoding(name, index):
        frames, state, goal = windows[int(index)]
        return encode_geometry_history(frames, state, goal, name, query_time=.6)
    def sample(name, encoded):
        return runtimes[name].sample(*(encoded[k] for k in ('history_tokens', 'history_mask', 'history_times', 'condition')),
                                     seed=20404, steps=20)
    for name in names:
        for index in indices[:5]:
            sample(name, encoding(name, index))
    rng = np.random.default_rng(4104)
    # Measured actual raw-return -> CPU encoding -> public runtime -> CPU trajectory,
    # alternating representations per window to reduce order/thermal effects.
    for repeat in range(args.repeats):
        for index in rng.permutation(indices):
            for name in rng.permutation(names):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                encoded = encoding(name, index)
                ready = time.perf_counter()
                prediction = sample(name, encoded)
                torch.cuda.synchronize()
                end = time.perf_counter()
                assert np.isfinite(prediction['positions']).all()
                records.append(dict(representation=str(name), index=int(index), repeat=repeat,
                    preprocess_ms=1000*(ready-begin), runtime_ms=1000*(end-ready), total_ms=1000*(end-begin)))
        print(json.dumps(dict(event='end_to_end_repeat_complete', repeat=repeat, measurements=len(records))), flush=True)
    # Separately synchronized stage timings are diagnostic; their percentiles
    # must not be added to manufacture an end-to-end percentile.
    with torch.inference_mode():
        for repeat in range(args.repeats):
            for index in rng.permutation(indices):
                for name in rng.permutation(names):
                    encoded = encoding(name, index)
                    fields = [encoded[k] for k in ('history_tokens', 'history_mask', 'history_times', 'condition')]
                    payload_bytes[name] = sum(x.nbytes for x in fields)
                    begin = time.perf_counter()
                    arrays = validate_inputs(*fields, name)[:4]
                    validation_ms = 1000*(time.perf_counter()-begin)
                    tensors, transfer_ms = timed_gpu(lambda: [torch.as_tensor(v, device='cuda') for v in arrays])
                    tokens, mask, ages, condition = tensors
                    runtime = runtimes[name]
                    observations, history_ms = timed_gpu(lambda: runtime.model.encode_history(tokens, mask, ages))
                    context, context_ms = timed_gpu(lambda: runtime.model.condition_context(condition))
                    output, ddim_ms = timed_gpu(lambda: denoise_cached(runtime, *observations, context, condition))
                    cpu, output_ms = timed_gpu(lambda: {k: v.cpu().numpy() for k, v in output.items()})
                    if name not in checks:
                        reference = sample_model(runtime.model, runtime.schedule, *tensors, seed=20404, steps=20)
                        error = max(float(np.max(np.abs(cpu[k]-reference[k].cpu().numpy()))) for k in cpu)
                        assert error <= 1e-6, (name, error)
                        checks[name] = dict(cached_denoising_matches_production_max_abs_error=error)
                    stage_records.append(dict(representation=str(name), index=int(index), repeat=repeat,
                        cpu_validation_ms=validation_ms, host_to_device_ms=transfer_ms,
                        history_encoder_ms=history_ms, condition_mlp_ms=context_ms,
                        cached_ddim_and_spline_ms=ddim_ms, device_to_host_ms=output_ms))
            print(json.dumps(dict(event='stage_repeat_complete', repeat=repeat)), flush=True)
    report = dict(schema='geometry_latency_benchmark_v1', date='2026-10-04',
        protocol=dict(samples=32, groups=16, repeats=args.repeats, warmup_windows_per_representation=5,
            batch_size=1, ddim_steps=20, seed=20404, interleave_seed=4104,
            scope='resident measured-return windows -> rebuilt CPU representation -> CUDA runtime -> CPU trajectory',
            exclusions=['sensor raycasting/acquisition', 'file loading', 'ray-range reconstruction', 'IPC',
                        'nvblox mapping', 'collision guard', 'controller', 'physics', 'rendering'],
            stages='separately synchronized wall-clock durations; not additive percentiles',
            box_recomputed_from_raw_returns=True, no_training_or_model_selection=True),
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), torch_threads=torch.get_num_threads(),
            openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'), omp_num_threads=os.environ.get('OMP_NUM_THREADS')),
        dataset_sha256=digest(dataset), script_sha256=digest(__file__),
        checkpoints={name:runtimes[name].checkpoint_sha256 for name in names},
        source_sha256={f:digest(root/'hnuter_dit'/f) for f in ['geometry_observations.py','lidar_condition.py',
            'temporal_observations.py','temporal_inference.py','temporal_model.py']},
        valid_returns_per_frame=summarize([len(f.points) for frames, _, _ in windows.values() for f in frames]),
        checks=checks, input_payload_bytes=payload_bytes, summaries={}, stage_summaries={})
    # Counts have no time units.
    report['valid_returns_per_frame']={k.replace('_ms',''):v for k,v in report['valid_returns_per_frame'].items()}
    for name in names:
        rows = [r for r in records if r['representation'] == name]
        stages = [r for r in stage_records if r['representation'] == name]
        report['summaries'][name]={k:summarize([r[k] for r in rows]) for k in ['preprocess_ms','runtime_ms','total_ms']}
        report['stage_summaries'][name]={k:summarize([r[k] for r in stages]) for k in stages[0] if k.endswith('_ms')}
    for filename, value in [('report.json', report), ('measurements.json', records), ('stages.json', stage_records)]:
        (args.output/filename).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['summaries'], indent=2), flush=True)


if __name__ == '__main__':
    main()
