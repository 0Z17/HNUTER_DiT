"""Frozen Contact benchmark: identical latents, paired decoders, full URDF audit.

Generation and scoring are separate resumable stages. No oracle is ever read by
this module. Feasibility booleans were fixed before generation and only annotate
scores, never filter an inference request or select a trajectory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .contact_adapter import sha256_file
from .contact_benchmark import _read_gz_jsonl, _parse_input_record, parse_input_condition
from .contact_experiment import DEFAULT_OUTPUT, training_matrix
from .contact_inference import ContactInferenceRuntime, INFERENCE_VERSION
from .contact_metrics import METRICS_VERSION, evaluate_contact_path
from .contact_projection import PROJECTION_VERSION, ContactProjectionSpec, project_contact_pose9
from .contact_metrics import _backend
from .train_contact_adapter import DEFAULT_BASE, json_hash, write_json

DEFAULT_BENCHMARK = Path('/home/z017/research/HNUTER_datasets/surface_contact_adapter_benchmark_10000cond_v001')


def load_benchmark(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'frozen':
        raise RuntimeError('freeze all original/shifted feasibility before model inference')
    for name in ('runtime_inputs.npz', 'index.jsonl'):
        if sha256_file(root/name) != manifest['input_files_sha256'][name]:
            raise RuntimeError(f'frozen benchmark checksum mismatch: {name}')
    with np.load(root/'runtime_inputs.npz', allow_pickle=False) as archive:
        inputs = {name:archive[name].copy() for name in archive.files}
    if len(np.unique(inputs['condition_id'])) != manifest['condition_count']:
        raise RuntimeError('runtime cache repeats or drops distinct conditions')
    if len(inputs['conditions']) != 2*manifest['condition_count']:
        raise RuntimeError('runtime scenario cache has a shortfall')
    return manifest, inputs


def multimodal_maps(inputs: dict[str, np.ndarray], maps_per_family: int = 10) -> set[str]:
    families: dict[str, set[str]] = defaultdict(set)
    for family, map_id in zip(inputs['family'], inputs['map_id']):
        families[str(family)].add(str(map_id))
    selected = set()
    import hashlib
    for family, maps in sorted(families.items()):
        ranked = sorted(maps, key=lambda key:hashlib.sha256(f'20261011/multimodal/{key}'.encode()).hexdigest())
        selected.update(ranked[:maps_per_family])
    return selected


def requests_for_map(inputs: dict[str, np.ndarray], map_id: str, multi: set[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero(inputs['map_id'] == map_id)
    count = 8 if map_id in multi else 1
    return np.repeat(rows, count), np.tile(np.arange(count), len(rows))


def model_contract(args: argparse.Namespace, model_id: str, manifest: dict) -> tuple[Path, dict]:
    adapter = None if model_id == 'base' else args.experiment/'models'/model_id/'best.pt'
    if adapter is not None:
        state = json.loads((adapter.parent/'run_status.json').read_text())
        if state['status'] != 'complete' or state['step'] != 8000 or sha256_file(adapter) != state['selected_checkpoint_sha256']:
            raise RuntimeError('evaluation requires a completed, validation-selected model')
    code = Path(__file__).parent
    backend = _backend()
    import importlib.metadata
    collector = backend[3].parents[2]
    definition = dict(model_id=model_id, base_sha256=sha256_file(args.base),
        adapter_sha256=None if adapter is None else sha256_file(adapter),
        benchmark_manifest_sha256=sha256_file(args.benchmark/'manifest.json'),
        benchmark_conditions=manifest['condition_count'],
        primary_candidates=1, multimodal_maps_per_family=10, multimodal_candidates=8,
        ddim_steps=25, latent_seed=20261011, inference_version=INFERENCE_VERSION,
        projection_version=PROJECTION_VERSION, metrics_version=METRICS_VERSION,
        reference_backend=dict(urdf_sha256=backend[4], coal_source_sha256=sha256_file(collector/'coal_collision.py'),
                               coal_version=importlib.metadata.version('coal'), numpy_version=np.__version__, torch_version=torch.__version__),
        inference_sources={name:sha256_file(code/name) for name in ('contact_inference.py','contact_adapter.py','model.py','geometry.py')},
        evaluation_sources={name:sha256_file(code/name) for name in ('evaluate_contact_experiment.py','contact_metrics.py','contact_projection.py','contact_benchmark.py')})
    directory = args.experiment/'evaluation'/model_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'contract.json'
    digest = json_hash(definition)
    if path.exists():
        if json.loads(path.read_text())['contract_sha256'] != digest:
            raise RuntimeError('evaluation resume changed its model, inputs or code')
    else:
        write_json(path, definition | dict(contract_sha256=digest, created_utc=datetime.now(timezone.utc).isoformat()))
    return directory, definition | dict(contract_sha256=digest)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def generate(args: argparse.Namespace, model_id: str, manifest: dict, inputs: dict) -> None:
    directory, contract = model_contract(args, model_id, manifest)
    multi = multimodal_maps(inputs)
    maps = sorted(set(map(str, inputs['map_id'])))
    if args.limit_maps:
        maps = maps[:args.limit_maps]
    missing = []
    completed = 0
    for map_id in maps:
        stem = map_id.rsplit('/', 1)[-1]
        path = directory/'raw'/f'{stem}.npz'
        meta = path.with_suffix('.json')
        if path.exists() and meta.exists():
            saved = json.loads(meta.read_text())
            if saved['contract_sha256'] != contract['contract_sha256'] or sha256_file(path) != saved['raw_sha256']:
                raise RuntimeError('generated shard checksum/contract mismatch')
            completed += 1
        else:
            missing.append(map_id)
    if not missing:
        write_json(directory/'generation_status.json', dict(status='pilot_complete' if args.limit_maps else 'complete',
            complete_maps=completed, total_maps=len(maps), contract_sha256=contract['contract_sha256']))
        return
    runtime = ContactInferenceRuntime(args.base, None if model_id == 'base' else args.experiment/'models'/model_id/'best.pt')
    began = time.monotonic()
    for start in range(0, len(missing), 16):
        block = missing[start:start+16]
        requests = [requests_for_map(inputs, map_id, multi) for map_id in block]
        rows, candidates = np.concatenate([r[0] for r in requests]), np.concatenate([r[1] for r in requests])
        paths = []
        for offset in range(0, len(rows), args.batch_size):
            paths.append(runtime.sample(inputs, rows[offset:offset+args.batch_size], candidates[offset:offset+args.batch_size]))
        paths = np.concatenate(paths)
        offset = 0
        for map_id, (map_rows, map_candidates) in zip(block, requests):
            stem = map_id.rsplit('/', 1)[-1]
            path = directory/'raw'/f'{stem}.npz'
            count = len(map_rows)
            atomic_npz(path, world_pose9=paths[offset:offset+count], input_rows=map_rows, candidate_indices=map_candidates)
            write_json(path.with_suffix('.json'), dict(contract_sha256=contract['contract_sha256'], raw_sha256=sha256_file(path),
                                                      map_id=map_id, candidates=count, inference_dtype='float32', batch_size=args.batch_size))
            offset += count
        completed += len(block)
        write_json(directory/'generation_status.json', dict(status='generating', complete_maps=completed, total_maps=len(maps),
                                                           elapsed_s=time.monotonic()-began, contract_sha256=contract['contract_sha256']))
        print(f'GENERATE {model_id} {completed}/{len(maps)} maps elapsed={time.monotonic()-began:.1f}s', flush=True)
    write_json(directory/'generation_status.json', dict(status='pilot_complete' if args.limit_maps else 'complete',
        complete_maps=completed, total_maps=len(maps), contract_sha256=contract['contract_sha256']))


def _scalar_dict(values: dict) -> dict:
    return {key: (float(value) if value.is_floating_point() else bool(value) if value.dtype == torch.bool else int(value))
            for key,value in values.items()}


def score_map(args: argparse.Namespace, directory: Path, contract: dict, inputs: dict,
              map_id: str, multi: set[str], manifest: dict) -> int:
    stem = map_id.rsplit('/', 1)[-1]
    raw_path = directory/'raw'/f'{stem}.npz'
    raw_meta = json.loads(raw_path.with_suffix('.json').read_text())
    if raw_meta['contract_sha256'] != contract['contract_sha256'] or sha256_file(raw_path) != raw_meta['raw_sha256']:
        raise RuntimeError('score input raw SHA mismatch')
    score_path = directory/'scores'/f'{stem}.jsonl.gz'
    meta_path = score_path.with_name(f'{stem}.json')
    if score_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['contract_sha256'] != contract['contract_sha256'] or meta['raw_sha256'] != raw_meta['raw_sha256'] or sha256_file(score_path) != meta['scores_sha256']:
            raise RuntimeError('scored shard checksum/contract mismatch')
        return meta['score_rows']
    with np.load(raw_path, allow_pickle=False) as archive:
        paths = archive['world_pose9']; rows = archive['input_rows']; candidates = archive['candidate_indices']
    if paths.shape != (len(rows),128,9) or rows.ndim != 1 or candidates.shape != rows.shape:
        raise RuntimeError('raw shard has a path/request shortfall or malformed shape')
    expected_rows, expected_candidates = requests_for_map(inputs, map_id, multi)
    if not np.array_equal(rows, expected_rows) or not np.array_equal(candidates, expected_candidates):
        raise RuntimeError('raw request list differs from frozen sampling protocol')
    condition_shard = args.benchmark/str(inputs['input_shard'][rows[0]])
    expected_input_hash = manifest['input_files_sha256'][str(inputs['input_shard'][rows[0]])]
    if sha256_file(condition_shard) != expected_input_hash:
        raise RuntimeError('whitelist condition shard changed')
    records = [_parse_input_record(record) for record in _read_gz_jsonl(condition_shard)]
    output_rows = []
    prepared = {}
    for path, row, candidate in zip(paths, rows, candidates):
        scenario = str(inputs['scenario'][row])
        if int(row) not in prepared:
            condition = parse_input_condition(records[int(inputs['input_row'][row])]['conditions'][scenario])
            if json_hash(condition) != str(inputs['input_sha256'][row]):
                raise RuntimeError('runtime input differs from whitelist condition')
            prepared[int(row)] = condition, ContactProjectionSpec.from_condition(condition)
        condition, spec = prepared[int(row)]
        common = dict(model_id=contract['model_id'], case_id=str(inputs['case_id'][row]),
            condition_id=str(inputs['condition_id'][row]), map_id=map_id, family=str(inputs['family'][row]),
            kind=str(inputs['kind'][row]), difficulty=str(inputs['difficulty'][row]), scenario=scenario,
            candidate_index=int(candidate), multimodal_subset=map_id in multi,
            feasible=bool(inputs['feasible_mask'][row]), feasibility_status=str(inputs['feasibility_status'][row]),
            input_sha256=str(inputs['input_sha256'][row]))
        off = evaluate_contact_path(path, condition)
        output_rows.append(common | dict(projection='off', metrics=off))
        try:
            projected = project_contact_pose9(path, spec)
            on = evaluate_contact_path(projected.world_pose9, condition, projected_intrinsic=projected.intrinsic_states)
            on.update(projection_before=_scalar_dict(projected.before), projection_after=_scalar_dict(projected.after),
                      projection_geometry_valid=bool(projected.geometry_valid))
        except ValueError as error:
            on = dict(candidate_success=False, numeric_candidate_success=False, invalid=True,
                      invalid_reason=str(error), projection_failed=True)
        output_rows.append(common | dict(projection='on', metrics=on))
    score_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = score_path.with_name(f'.{score_path.name}.{os.getpid()}.tmp')
    with gzip.open(temporary, 'wt', encoding='utf8') as stream:
        for record in output_rows:
            stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n')
    os.replace(temporary, score_path)
    write_json(meta_path, dict(contract_sha256=contract['contract_sha256'], raw_sha256=raw_meta['raw_sha256'],
                              scores_sha256=sha256_file(score_path), score_rows=len(output_rows), map_id=map_id))
    return len(output_rows)


_SCORE_WORKER: tuple | None = None


def _score_initialize(args: argparse.Namespace, directory: Path, contract: dict,
                      inputs: dict, multi: set[str], manifest: dict) -> None:
    global _SCORE_WORKER
    torch.set_num_threads(1)
    _SCORE_WORKER = args, directory, contract, inputs, multi, manifest


def _score_worker(map_id: str) -> int:
    args, directory, contract, inputs, multi, manifest = _SCORE_WORKER
    return score_map(args, directory, contract, inputs, map_id, multi, manifest)


def score(args: argparse.Namespace, model_id: str, manifest: dict, inputs: dict) -> None:
    directory, contract = model_contract(args, model_id, manifest)
    maps = sorted(set(map(str, inputs['map_id'])))
    if args.limit_maps:
        maps = maps[:args.limit_maps]
    multi = multimodal_maps(inputs)
    began = time.monotonic(); score_rows = 0
    # Spawn avoids inheriting PyTorch's CPU thread pool. Each worker receives
    # the frozen input cache once, rather than pickling it for every map.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=_score_initialize, initargs=(args,directory,contract,inputs,multi,manifest)) as pool:
        pending = {pool.submit(_score_worker, map_id):map_id for map_id in maps}
        for index, future in enumerate(as_completed(pending)):
            score_rows += future.result()
            if index % 10 == 0 or index == len(maps)-1:
                write_json(directory/'scoring_status.json', dict(status='scoring', complete_maps=index+1, total_maps=len(maps),
                    score_rows=score_rows, elapsed_s=time.monotonic()-began, contract_sha256=contract['contract_sha256']))
                print(f'SCORE {model_id} {index+1}/{len(maps)} maps elapsed={time.monotonic()-began:.1f}s', flush=True)
    write_json(directory/'scoring_status.json', dict(status='pilot_complete' if args.limit_maps else 'complete',
        complete_maps=len(maps), total_maps=len(maps), score_rows=score_rows, contract_sha256=contract['contract_sha256']))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--stage', choices=('generate','score','all'), default='all')
    parser.add_argument('--model', action='append', help='default is Base and every approved adapter')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, choices=(1,2), default=2)
    parser.add_argument('--limit-maps', type=int, default=0, help='pilot only; never marks a full evaluation complete')
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest, inputs = load_benchmark(args.benchmark)
    models = args.model or ['base', *[row['id'] for row in training_matrix()]]
    for model_id in models:
        if args.stage in ('generate', 'all'):
            generate(args, model_id, manifest, inputs)
        if args.stage in ('score', 'all'):
            score(args, model_id, manifest, inputs)


if __name__ == '__main__':
    main()
