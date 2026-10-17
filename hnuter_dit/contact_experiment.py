"""The approved five-budget/three-seed Contact research training matrix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .contact_adapter import sha256_file
from .train_contact_adapter import DEFAULT_BASE, DEFAULT_DATASET, json_hash, source_hashes, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'artifacts/contact_adapter_multifamily_v001'
BUDGETS = (500, 1000, 2000, 4000, 8000)
SEEDS = (17, 29, 43)


def training_matrix() -> list[dict]:
    rows = [dict(id=f'geometry_n{size:04d}_seed{seed}', variant='geometry', train_size=size,
                 seed=seed, extra_args=[]) for size in BUDGETS for seed in SEEDS]
    for seed in SEEDS:
        rows.append(dict(id=f'no_contact_loss_n8000_seed{seed}', variant='no_contact_loss', train_size=8000,
                         seed=seed, extra_args=['--contact-weight', '0', '--normal-weight', '0', '--smooth-weight', '0']))
        rows.append(dict(id=f'no_geometry_tokens_n8000_seed{seed}', variant='no_geometry_tokens', train_size=8000,
                         seed=seed, extra_args=['--task-conditioning', 'none']))
    return rows


def protocol(base: Path, dataset: Path) -> dict:
    return dict(schema_version='contact_research_protocol_v001',
        base_checkpoint=str(base.resolve()), base_sha256=sha256_file(base),
        training_dataset=str(dataset.resolve()), dataset_freeze_sha256=sha256_file(dataset/'dataset_freeze.json'),
        budgets=list(BUDGETS), model_seeds=list(SEEDS), data_subset_seed=20261011,
        optimizer_updates=8000, batch_size=64, learning_rate=1e-4,
        subset_policy='nested whole source maps; paired expert trajectories stay together',
        baseline='frozen v003 with identical obstacles/endpoints/DDIM/latents, projection off/on',
        variants=training_matrix(),
        checkpoint_selection='minimum common fixed-noise loss on all 1000 validation experts; never benchmark-tuned',
        new_test_conditions=10000, new_source_maps=1000, scene_families=10,
        primary_candidates_per_condition=1,
        multimodal_subset_conditions=1000, multimodal_candidates_per_condition=8,
        sampling_steps=25, diffusion_steps=100, clip_x0=4., latent_seed=20261011,
        scenarios=['original', 'obstacle_shift_0.30m'], projection=['off', 'on'],
        feasibility_selection='all raw stress cases and a separately frozen witness-certified feasible subset',
        inference_inputs='only requested geometry, endpoints, physical obstacles; oracle and route metadata excluded',
        uncertainty='source-map cluster bootstrap; model seeds reported separately and mean/std across 3 seeds',
        projection_interpolation='off: world XYZ+SO3 SLERP; on: unwrapped intrinsic linear lift',
        thermal=dict(cpu_pause_c=90, cpu_resume_c=80, gpu_pause_c=83, gpu_resume_c=75, stable_cool_seconds=10),
        limitations=['geometric continuous-contact paths; no approach/retreat, forces or dynamics',
                    'new-map IID within the same ten scene families, not semantic OOD',
                    'bounded feasibility solver failure is unknown, not a proof of infeasibility'])


def ensure_protocol(output: Path, base: Path, dataset: Path) -> dict:
    value = protocol(base, dataset)
    digest = json_hash(value)
    path = output/'protocol.json'
    if path.exists():
        prior = json.loads(path.read_text())
        if prior['protocol_sha256'] != digest:
            raise RuntimeError('immutable experiment protocol differs; choose a fresh version')
        return prior
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, value | dict(protocol_sha256=digest, created_utc=datetime.now(timezone.utc).isoformat()))
    return value | dict(protocol_sha256=digest)


def run_training(output: Path, base: Path, dataset: Path) -> None:
    definition = ensure_protocol(output, base, dataset)
    status_path = output/'experiment_status.json'
    began = time.monotonic()
    completed = []
    for row in definition['variants']:
        directory = output/'models'/row['id']
        state_path = directory/'run_status.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['status'] == 'complete':
                if state['step'] != 8000 or sha256_file(directory/'best.pt') != state['selected_checkpoint_sha256']:
                    raise RuntimeError('completed model is not the planned frozen checkpoint')
                saved = json.loads((directory/'resolved_training_config.json').read_text())
                expected_weights = dict(contact=0. if row['variant']=='no_contact_loss' else 5.,
                    normal=0. if row['variant']=='no_contact_loss' else .5,
                    smooth=0. if row['variant']=='no_contact_loss' else .1, collision=.002)
                expected = dict(base_sha256=definition['base_sha256'], dataset_freeze_sha256=definition['dataset_freeze_sha256'],
                    train_size=row['train_size'], seed=row['seed'], training_steps=8000, batch_size=64,
                    data_subset_seed=20261011, source_hashes=source_hashes(), loss_weights=expected_weights)
                if any(saved.get(key)!=value for key,value in expected.items()) or saved['adapter_config']['task_conditioning'] != ('none' if row['variant']=='no_geometry_tokens' else 'geometry'):
                    raise RuntimeError('completed model configuration differs from its approved matrix row')
                completed.append(row['id'])
                continue
        command = [sys.executable, '-m', 'hnuter_dit.train_contact_adapter', '--base', str(base),
                   '--dataset', str(dataset), '--output', str(directory), '--train-size', str(row['train_size']),
                   '--seed', str(row['seed']), '--steps', '8000', '--batch-size', '64',
                   '--data-subset-seed', '20261011', *row['extra_args']]
        if (directory/'last.pt').is_file():
            command.append('--resume')
        directory.mkdir(parents=True, exist_ok=True)
        write_json(status_path, dict(status='training', current_model=row['id'], complete_models=completed,
                                    total_models=len(definition['variants']), protocol_sha256=definition['protocol_sha256']))
        print(f"START {row['id']} ({len(completed)+1}/{len(definition['variants'])})", flush=True)
        with (directory/'console.log').open('a', buffering=1) as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(status_path, dict(status='training_failed', current_model=row['id'], complete_models=completed,
                                        exit_code=result.returncode, log=str(directory/'console.log')))
            raise RuntimeError(f"training failed: {row['id']}; inspect {directory/'console.log'}")
        state = json.loads(state_path.read_text())
        if state['status'] != 'complete':
            raise RuntimeError('trainer ended without the full planned budget')
        completed.append(row['id'])
        print(f"COMPLETE {row['id']} best_step={state['best_step']} elapsed_s={state['elapsed_s']:.1f}", flush=True)
    write_json(status_path, dict(status='training_complete_evaluation_pending', complete_models=completed,
                                total_models=len(definition['variants']), elapsed_s=time.monotonic()-began,
                                protocol_sha256=definition['protocol_sha256']))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--write-protocol', action='store_true')
    args = parser.parse_args()
    # The verified environment has many ROS site plugins and CPU BLAS threads.
    # Bound the research child processes without changing global machine state.
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    if args.write_protocol:
        definition = ensure_protocol(args.output, args.base, args.dataset)
        print(json.dumps({'protocol_sha256':definition['protocol_sha256'], 'models':len(definition['variants'])}))
    else:
        run_training(args.output, args.base, args.dataset)


if __name__ == '__main__':
    main()
