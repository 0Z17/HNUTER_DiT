"""Standalone worker; do not import this module into the Isaac/Kit process."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import traceback


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, allow_nan=False) + '\n')
    temporary.replace(path)


def run(directory):
    # The isolated venv's torch/numpy must win. This narrowly appends small Kit
    # prerequisites; deprecated ml_archive is deliberately never appended.
    for archive in sorted(Path('/isaac-sim/extscache').glob('omni.kit.pip_archive-*/pip_prebundle')):
        sys.path.append(str(archive))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from hnuter_dit.service import DiTModelRuntime
    from hnuter_dit.guidance import CollisionGuidanceConfig
    from hnuter_dit.online_diffusion import sample_online

    config = json.loads((directory/'config.json').read_text())
    runtime = DiTModelRuntime(device=config['device'])
    description = dict(runtime.describe(), torch_version=torch.__version__,
                       torch_module_path=torch.__file__, python_executable=sys.executable)
    atomic_json(directory/'status.json', dict(state='ready', model=description))
    sequence = 0
    while True:
        request_path = directory/'request.npz'
        if request_path.exists():
            with np.load(request_path, allow_pickle=False) as request:
                requested_sequence = int(request['sequence'])
                if requested_sequence > sequence:
                    mode = str(request['mode'])
                    tokens, mask = request['tokens'].copy(), request['mask'].copy()
                    start, goal = request['start'].copy(), request['goal'].copy()
                    seed, count = int(request['seed']), int(request['count'])
                else:
                    mode = None
            if mode is not None:
                sequence = requested_sequence
                began = time.perf_counter()
                try:
                    if mode == 'baseline':
                        paths, elapsed = runtime.sample(tokens, mask, start, goal, count=count,
                            sampling_steps=25, seed=seed,
                            guidance=CollisionGuidanceConfig(mode='smooth', steps=8, step_size_m=.12))
                        metadata = dict(algorithm='unchanged_v003_smooth_guidance_baseline',
                            inference_seconds=elapsed, model_evaluations=25,
                            weights_retrained=False, checkpoint_sha256=runtime.checkpoint_sha256,
                            seed=seed, candidate_count=count, collision_certified=False)
                    elif mode == 'online':
                        paths, metadata = sample_online(runtime, tokens, mask, start, goal, seed, count)
                    else:
                        raise ValueError('unsupported planner mode: ' + mode)
                    metadata['worker_wall_seconds'] = time.perf_counter()-began
                    metadata['sequence'] = sequence
                    metadata['process_isolated'] = True
                except Exception as exc:
                    traceback.print_exc()
                    paths = np.empty((0, 128, 7), dtype=np.float32)
                    metadata = dict(error=f'{type(exc).__name__}: {exc}', sequence=sequence)
                temporary = directory/'response.tmp'
                with temporary.open('wb') as stream:
                    np.savez(stream, sequence=np.int64(sequence), paths=paths,
                             metadata=json.dumps(metadata, allow_nan=False))
                temporary.replace(directory/'response.npz')
        stop_path = directory/'stop.json'
        if stop_path.exists() and sequence >= int(json.loads(stop_path.read_text()).get('sequence', 0)):
            atomic_json(directory/'status.json', dict(state='stopped', sequence=sequence, model=description))
            return
        time.sleep(.01)


def main():
    directory = Path(sys.argv[1]).resolve()
    try:
        run(directory)
    except BaseException as exc:
        traceback.print_exc()
        atomic_json(directory/'status.json', dict(state='error', error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
