"""Score a fixed smooth decoder supplement using only frozen v001 raw paths.

This module never samples, trains, repairs obstacles, or opens oracle artifacts.
The physical edges remain the frozen linear intrinsic/lift edges between 128
returned nodes. Every requested candidate is retained, including decoder errors.
"""
from __future__ import annotations

import os

# Set before importing numerical libraries; spawn workers inherit these limits.
for _thread_variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .contact_adapter import sha256_file
from .contact_benchmark import SCENARIOS, _parse_input_record, _read_gz_jsonl, parse_input_condition
from .contact_experiment import DEFAULT_OUTPUT, training_matrix
from .contact_inference import INFERENCE_VERSION
from .contact_metrics import METRICS_VERSION, _backend, evaluate_contact_path
from .contact_projection import INTRINSIC_INTERPOLATION, PROJECTION_VERSION, ContactProjectionSpec
from .contact_smooth_projection import (SMOOTH_PROJECTION_VERSION,
    project_smooth_contact_pose9, smooth_contact_decoder_parameters)
from .evaluate_contact_experiment import DEFAULT_BENCHMARK, atomic_npz, load_benchmark, multimodal_maps, requests_for_map
from .train_contact_adapter import json_hash, write_json

SUPPLEMENT_VERSION = 'contact_smooth_projection_supplement_v001'
SOURCE_CONTRACT_VERSION = 'contact_smooth_projection_sources_v001'
SCORE_SCHEMA_VERSION = 'contact_smooth_projection_scores_v001'
PROJECTED_SCHEMA_VERSION = 'contact_smooth_projection_paths_v001'
SUPPLEMENT_SOURCES = ('evaluate_contact_smooth_projection.py', 'contact_smooth_projection.py')
INFERENCE_SOURCES = ('contact_inference.py', 'contact_adapter.py', 'model.py', 'geometry.py')
EVALUATION_SOURCES = ('evaluate_contact_experiment.py', 'contact_metrics.py', 'contact_projection.py', 'contact_benchmark.py')
OUTPUT_SCHEMA = dict(score_version=SCORE_SCHEMA_VERSION, projected_version=PROJECTED_SCHEMA_VERSION,
    projection='on_smooth', world_pose9=['N', 128, 9], intrinsic_states=['N', 128, 3],
    request_order='exact_parent_raw_order', failure_paths='all_nan_pose_and_intrinsic_for_decoder_errors',
    flags=['projection_succeeded', 'fit_success', 'geometry_valid'],
    metrics_version=METRICS_VERSION, edge_interpolation=INTRINSIC_INTERPOLATION)


class IntegrityError(RuntimeError):
    """A frozen input, source, or previously committed result changed."""


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise IntegrityError(f'expected a JSON object: {path}')
    return value


def _definition(value: dict, hash_field: str = 'contract_sha256') -> dict:
    return {key: item for key, item in value.items() if key not in (hash_field, 'created_utc')}


def _verified_json(path: Path, hash_field: str = 'contract_sha256') -> dict:
    value = _read(path)
    if value.get(hash_field) != json_hash(_definition(value, hash_field)):
        raise IntegrityError(f'immutable JSON contract checksum mismatch: {path}')
    return value


def _check_hash(path: Path, digest: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise IntegrityError(f'frozen artifact checksum mismatch: {path}')


def _immutable(path: Path, definition: dict, hash_field: str = 'contract_sha256') -> dict:
    digest = json_hash(definition)
    if path.exists():
        prior = _verified_json(path, hash_field)
        if prior[hash_field] != digest:
            raise IntegrityError(f'immutable supplement changed; choose a fresh output: {path}')
        return prior
    result = definition | {hash_field: digest, 'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(path, result)
    return result


def _safe_input_path(root: Path, relative: str) -> Path:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or '..' in parts or any('oracle' in part.lower() or 'route' in part.lower() for part in parts):
        raise IntegrityError('benchmark input path is unsafe or points to oracle/route artifacts')
    path = root/relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise IntegrityError('benchmark input symlink escapes its frozen directory')
    return path


def _validate_protocol(protocol: dict) -> None:
    expected = dict(schema_version='contact_research_protocol_v001', optimizer_updates=8000,
        new_test_conditions=10000, new_source_maps=1000, multimodal_subset_conditions=1000,
        primary_candidates_per_condition=1, multimodal_candidates_per_condition=8,
        sampling_steps=25, latent_seed=20261011, variants=training_matrix(), projection=['off', 'on'])
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise IntegrityError('parent experiment differs from the frozen v001 research protocol')


def _verify_parent_sources(experiment: Path, benchmark: Path, protocol: dict) -> dict:
    path = experiment/'pipeline_source_contract.json'
    contract = _verified_json(path)
    if (Path(contract['experiment']).resolve() != experiment.resolve() or
            Path(contract['benchmark']).resolve() != benchmark.resolve() or
            contract['protocol_sha256'] != protocol['protocol_sha256']):
        raise IntegrityError('parent pipeline source/input contract belongs to another experiment')
    for filename, digest in (contract['source_files'] | contract['input_files']).items():
        if any('oracle' in part.lower() for part in Path(filename).parts):
            raise IntegrityError('parent source contract unexpectedly references an oracle artifact')
        _check_hash(Path(filename), digest)
    code = Path(__file__).parent
    for name in (*INFERENCE_SOURCES, *EVALUATION_SOURCES):
        if contract['source_files'].get(str((code/name).resolve())) != sha256_file(code/name):
            raise IntegrityError(f'parent frozen source contract does not bind {name}')
    if contract['input_files'].get(str((experiment/'protocol.json').resolve())) != sha256_file(experiment/'protocol.json'):
        raise IntegrityError('parent source contract does not bind the experiment protocol')
    if contract['input_files'].get(str(Path(protocol['base_checkpoint']).resolve())) != protocol['base_sha256']:
        raise IntegrityError('parent source contract does not bind the frozen base checkpoint')
    return contract


def _validate_inputs(manifest: dict, inputs: dict, protocol: dict) -> tuple[list[str], set[str], int]:
    required = ('case_id', 'condition_id', 'map_id', 'family', 'kind', 'difficulty', 'scenario',
                'feasible_mask', 'feasibility_status', 'input_sha256', 'input_shard', 'input_row')
    total = 2*protocol['new_test_conditions']
    if (manifest['condition_count'] != protocol['new_test_conditions'] or
            manifest['source_map_count'] != protocol['new_source_maps'] or
            any(name not in inputs or inputs[name].shape != (total,) for name in required) or
            len(set(inputs['case_id'])) != total):
        raise IntegrityError('frozen benchmark has a case/map/shape shortfall')
    if inputs['input_row'].dtype.kind not in 'iu' or inputs['feasible_mask'].dtype.kind != 'b':
        raise IntegrityError('runtime input indices or feasibility flags have an invalid dtype')
    if set(inputs['scenario']) != set(SCENARIOS):
        raise IntegrityError('runtime inputs do not contain the fixed original/shift scenarios')
    for scenario in SCENARIOS:
        rows = inputs['scenario'] == scenario
        if rows.sum() != protocol['new_test_conditions'] or len(set(inputs['condition_id'][rows])) != protocol['new_test_conditions']:
            raise IntegrityError('runtime input scenario repeats or drops a condition')
    maps = sorted(set(map(str, inputs['map_id'])))
    stems = [map_id.rsplit('/', 1)[-1] for map_id in maps]
    if len(maps) != protocol['new_source_maps'] or len(set(stems)) != len(stems) or any(stem in ('', '.', '..') for stem in stems):
        raise IntegrityError('runtime map identities cannot be represented by unique shard names')
    families = set(map(str, inputs['family']))
    if len(families) != 10 or any(np.count_nonzero(inputs['map_id'] == map_id) != 20 for map_id in maps):
        raise IntegrityError('v001 requires ten families and ten paired conditions per source map')
    pairs: dict[str, list[int]] = {}
    for row, identifier in enumerate(inputs['condition_id']):
        pairs.setdefault(str(identifier), []).append(row)
    for pair in pairs.values():
        if len(pair) != 2 or any(inputs[name][pair[0]] != inputs[name][pair[1]] for name in ('map_id', 'family', 'kind', 'difficulty')):
            raise IntegrityError('original/shift runtime conditions are not paired by task identity')
    multi = multimodal_maps(inputs)
    if np.isin(inputs['map_id'], sorted(multi)).sum() != 2*protocol['multimodal_subset_conditions']:
        raise IntegrityError('fixed multimodal subset differs from the v001 protocol')
    count = sum(len(requests_for_map(inputs, map_id, multi)[0]) for map_id in maps)
    if count != 34000:
        raise IntegrityError('supplement requires exactly 34000 paths per model')
    return maps, multi, count


def ensure_supplement(args: argparse.Namespace, manifest: dict, inputs: dict) -> tuple[dict, dict, dict]:
    """Validate the parent without modifying it; commit independent contracts."""
    protocol = _verified_json(args.experiment/'protocol.json', 'protocol_sha256')
    _validate_protocol(protocol)
    parent_sources = _verify_parent_sources(args.experiment, args.benchmark, protocol)
    maps, multi, paths = _validate_inputs(manifest, inputs, protocol)
    if args.output.resolve() == args.experiment.resolve() or args.output.resolve().is_relative_to((args.experiment/'evaluation').resolve()):
        raise IntegrityError('supplement output must be separate from parent evaluation artifacts')
    parameters = smooth_contact_decoder_parameters()
    definition = dict(schema_version=SUPPLEMENT_VERSION, parent_experiment=str(args.experiment.resolve()),
        benchmark=str(args.benchmark.resolve()), parent_protocol_sha256=protocol['protocol_sha256'],
        parent_protocol_file_sha256=sha256_file(args.experiment/'protocol.json'),
        parent_source_contract_sha256=parent_sources['contract_sha256'],
        parent_source_contract_file_sha256=sha256_file(args.experiment/'pipeline_source_contract.json'),
        benchmark_manifest_sha256=sha256_file(args.benchmark/'manifest.json'),
        benchmark_input_files_sha256={name: manifest['input_files_sha256'][name] for name in
            sorted({'runtime_inputs.npz', 'index.jsonl', *map(str, inputs['input_shard'])})},
        planned_models=['base', *[row['id'] for row in protocol['variants']]],
        condition_count=protocol['new_test_conditions'], source_map_count=len(maps),
        multimodal_source_maps=sorted(multi), multimodal_subset_conditions=protocol['multimodal_subset_conditions'],
        primary_candidates=1, multimodal_candidates=8, paths_per_model=paths, score_rows_per_model=paths,
        projection='on_smooth', decoder_version=SMOOTH_PROJECTION_VERSION, decoder_parameters=parameters,
        decoder_parameters_sha256=json_hash(parameters), metrics_version=METRICS_VERSION,
        output_schema=OUTPUT_SCHEMA, output_schema_sha256=json_hash(OUTPUT_SCHEMA),
        input_policy='same_parent_raw_paths_and_whitelisted_frozen_original_shift_conditions',
        failure_policy='retain_every_request; decoder_errors_count_as_failed_candidates',
        parameter_policy='fixed_before_pilot; no_benchmark_or_pilot_tuning',
        interpolation=INTRINSIC_INTERPOLATION, oracle_access=False, generation=False, training=False)
    supplement = _immutable(args.output/'protocol.json', definition, 'protocol_sha256')
    sources = _immutable(args.output/'source_contract.json', dict(schema_version=SOURCE_CONTRACT_VERSION,
        protocol_sha256=supplement['protocol_sha256'], parent_source_contract_sha256=parent_sources['contract_sha256'],
        source_files={str((Path(__file__).parent/name).resolve()): sha256_file(Path(__file__).parent/name) for name in SUPPLEMENT_SOURCES},
        frozen_dependency_sources={name: sha256_file(Path(__file__).parent/name) for name in (*INFERENCE_SOURCES, *EVALUATION_SOURCES)},
        decoder_parameters_sha256=json_hash(parameters), metrics_version=METRICS_VERSION,
        output_schema_sha256=json_hash(OUTPUT_SCHEMA)))
    return protocol, supplement, sources


def _reference_backend() -> dict:
    backend = _backend()
    return dict(urdf_sha256=backend[4], coal_source_sha256=sha256_file(backend[3].parents[2]/'coal_collision.py'),
        coal_version=importlib.metadata.version('coal'), numpy_version=np.__version__, torch_version=torch.__version__)


def model_contract(args: argparse.Namespace, model_id: str, parent_protocol: dict,
                   supplement: dict, sources: dict) -> tuple[Path, dict]:
    if model_id not in supplement['planned_models']:
        raise IntegrityError(f'model is outside the frozen parent matrix: {model_id}')
    parent_path = args.experiment/'evaluation'/model_id/'contract.json'
    parent = _verified_json(parent_path)
    expected = dict(model_id=model_id, base_sha256=parent_protocol['base_sha256'],
        benchmark_manifest_sha256=supplement['benchmark_manifest_sha256'], benchmark_conditions=10000,
        primary_candidates=1, multimodal_maps_per_family=10, multimodal_candidates=8,
        ddim_steps=25, latent_seed=20261011, inference_version=INFERENCE_VERSION,
        projection_version=PROJECTION_VERSION, metrics_version=METRICS_VERSION,
        reference_backend=_reference_backend(),
        inference_sources={name: sha256_file(Path(__file__).parent/name) for name in INFERENCE_SOURCES},
        evaluation_sources={name: sha256_file(Path(__file__).parent/name) for name in EVALUATION_SOURCES})
    if any(parent.get(key) != value for key, value in expected.items()):
        raise IntegrityError('parent evaluation model/benchmark/source/backend contract mismatch')
    checkpoint_binding = dict(base_sha256=parent_protocol['base_sha256'], adapter_sha256=None)
    if model_id != 'base':
        checkpoint = args.experiment/'models'/model_id/'best.pt'
        state = _read(checkpoint.parent/'run_status.json')
        config_path = checkpoint.parent/'resolved_training_config.json'
        config = _read(config_path)
        config_definition = {key: value for key, value in config.items() if key not in ('contract_sha256', 'created_at_utc')}
        if config.get('contract_sha256') != json_hash(config_definition):
            raise IntegrityError('parent selected checkpoint training config checksum mismatch')
        row = next(row for row in parent_protocol['variants'] if row['id'] == model_id)
        if (state.get('status') != 'complete' or state.get('step') != 8000 or
                state.get('contract_sha256') != config['contract_sha256'] or
                config.get('base_sha256') != parent_protocol['base_sha256'] or
                config.get('train_size') != row['train_size'] or config.get('seed') != row['seed'] or
                config.get('training_steps') != 8000 or
                config.get('dataset_freeze_sha256') != parent_protocol['dataset_freeze_sha256'] or
                parent.get('adapter_sha256') != state.get('selected_checkpoint_sha256')):
            raise IntegrityError('parent adapter is not the completed validation-selected checkpoint')
        _check_hash(checkpoint, parent['adapter_sha256'])
        checkpoint_binding.update(adapter_sha256=parent['adapter_sha256'],
            training_config_sha256=sha256_file(config_path), run_status_sha256=sha256_file(checkpoint.parent/'run_status.json'))
    elif parent.get('adapter_sha256') is not None:
        raise IntegrityError('base parent evaluation unexpectedly binds an adapter')
    definition = dict(schema_version=SUPPLEMENT_VERSION, model_id=model_id,
        protocol_sha256=supplement['protocol_sha256'], source_contract_sha256=sources['contract_sha256'],
        parent_contract_sha256=parent['contract_sha256'], parent_contract_file_sha256=sha256_file(parent_path),
        checkpoint_binding=checkpoint_binding, benchmark_manifest_sha256=supplement['benchmark_manifest_sha256'],
        decoder_version=SMOOTH_PROJECTION_VERSION, decoder_parameters_sha256=supplement['decoder_parameters_sha256'],
        metrics_version=METRICS_VERSION, output_schema_sha256=supplement['output_schema_sha256'],
        reference_backend=parent['reference_backend'], score_rows_per_model=34000)
    directory = args.output/'evaluation'/model_id
    return directory, _immutable(directory/'contract.json', definition)


def _raw_requests(args: argparse.Namespace, contract: dict, inputs: dict, map_id: str,
                  multi: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    stem = map_id.rsplit('/', 1)[-1]
    path = args.experiment/'evaluation'/contract['model_id']/'raw'/f'{stem}.npz'
    meta = _read(path.with_suffix('.json'))
    if meta.get('contract_sha256') != contract['parent_contract_sha256'] or meta.get('map_id') != map_id:
        raise IntegrityError('parent raw shard contract/map identity mismatch')
    _check_hash(path, meta['raw_sha256'])
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'world_pose9', 'input_rows', 'candidate_indices'}:
            raise IntegrityError('parent raw shard contains missing or non-protocol arrays')
        paths, rows, candidates = (archive[name].copy() for name in ('world_pose9', 'input_rows', 'candidate_indices'))
    _check_hash(path, meta['raw_sha256'])
    expected_rows, expected_candidates = requests_for_map(inputs, map_id, multi)
    if (paths.shape != (len(expected_rows), 128, 9) or paths.dtype != np.float32 or
            rows.dtype.kind not in 'iu' or candidates.dtype.kind not in 'iu' or
            rows.shape != expected_rows.shape or candidates.shape != expected_candidates.shape or
            not np.array_equal(rows, expected_rows) or not np.array_equal(candidates, expected_candidates) or
            meta.get('candidates') != len(expected_rows) or meta.get('inference_dtype') != 'float32'):
        raise IntegrityError('parent raw path shape/dtype/request list differs from frozen sampling protocol')
    return paths, rows, candidates, meta | dict(raw_meta_sha256=sha256_file(path.with_suffix('.json')))


def _prepare_conditions(args: argparse.Namespace, inputs: dict, rows: np.ndarray, manifest: dict, map_id: str) -> tuple[dict, dict]:
    records, hashes, prepared = {}, {}, {}
    for row in dict.fromkeys(map(int, rows)):
        relative = str(inputs['input_shard'][row])
        if relative not in records:
            path = _safe_input_path(args.benchmark, relative)
            digest = manifest['input_files_sha256'][relative]
            _check_hash(path, digest)
            records[relative] = [_parse_input_record(record) for record in _read_gz_jsonl(path)]
            hashes[relative] = digest
        offset = int(inputs['input_row'][row])
        if not 0 <= offset < len(records[relative]):
            raise IntegrityError('runtime input record index is outside the frozen shard')
        record = records[relative][offset]
        scenario = str(inputs['scenario'][row])
        if any(record[key] != str(inputs[key][row]) for key in ('condition_id', 'family', 'kind', 'difficulty')) or record['map_id'] != map_id:
            raise IntegrityError('runtime case identity differs from its frozen whitelist record')
        condition = parse_input_condition(record['conditions'][scenario])
        if json_hash(condition) != str(inputs['input_sha256'][row]) or record['input_sha256'][scenario] != str(inputs['input_sha256'][row]):
            raise IntegrityError('runtime input checksum differs from whitelist condition')
        prepared[row] = condition
    return prepared, hashes


def _common(contract: dict, inputs: dict, row: int, candidate: int, map_id: str, multi: set[str]) -> dict:
    return dict(model_id=contract['model_id'], case_id=str(inputs['case_id'][row]),
        condition_id=str(inputs['condition_id'][row]), map_id=map_id, family=str(inputs['family'][row]),
        kind=str(inputs['kind'][row]), difficulty=str(inputs['difficulty'][row]), scenario=str(inputs['scenario'][row]),
        candidate_index=int(candidate), multimodal_subset=map_id in multi,
        feasible=bool(inputs['feasible_mask'][row]), feasibility_status=str(inputs['feasibility_status'][row]),
        input_sha256=str(inputs['input_sha256'][row]), projection='on_smooth')


def _scalars(values: dict) -> dict:
    result = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f'projection diagnostic is not scalar: {key}')
            value = value.detach().cpu().item()
        elif isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (bool, int, float)) or isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f'projection diagnostic is not a finite scalar: {key}')
        result[key] = value
    return result


def _numpy(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value, dtype=np.float64)


def _resume(score_path: Path, projected_path: Path, expected_meta: dict, inputs: dict,
            rows: np.ndarray, candidates: np.ndarray, contract: dict, map_id: str, multi: set[str]) -> int | None:
    score_meta_path = score_path.with_name(map_id.rsplit('/', 1)[-1]+'.json')
    projected_meta_path = projected_path.with_suffix('.json')
    paths = (score_path, projected_path, score_meta_path, projected_meta_path)
    if not any(path.exists() for path in paths):
        return None
    # Metadata is the commit record. Orphaned payloads after an interrupted
    # atomic write may be replaced, but any available commit must authenticate.
    for meta_path in (score_meta_path, projected_meta_path):
        if meta_path.exists():
            meta = _read(meta_path)
            if any(meta.get(key) != value for key, value in expected_meta.items()):
                raise IntegrityError('supplement shard metadata/input contract mismatch')
            _check_hash(score_path, meta['scores_sha256'])
            _check_hash(projected_path, meta['projected_sha256'])
    if not all(path.exists() for path in paths):
        return None
    score_meta, projected_meta = _read(score_meta_path), _read(projected_meta_path)
    if score_meta != projected_meta or score_meta.get('score_rows') != len(rows) or score_meta.get('projected_rows') != len(rows):
        raise IntegrityError('supplement committed shard metadata/row count mismatch')
    with np.load(projected_path, allow_pickle=False) as archive:
        expected = {'world_pose9', 'intrinsic_states', 'input_rows', 'candidate_indices', 'projection_succeeded', 'fit_success', 'geometry_valid'}
        if set(archive.files) != expected:
            raise IntegrityError('supplement projected archive schema mismatch')
        data = {name: archive[name] for name in archive.files}
    if (data['world_pose9'].shape != (len(rows), 128, 9) or data['intrinsic_states'].shape != (len(rows), 128, 3) or
            data['world_pose9'].dtype != np.float64 or data['intrinsic_states'].dtype != np.float64 or
            not np.array_equal(data['input_rows'], rows) or not np.array_equal(data['candidate_indices'], candidates) or
            any(data[key].shape != rows.shape or data[key].dtype != np.bool_ for key in ('projection_succeeded', 'fit_success', 'geometry_valid'))):
        raise IntegrityError('supplement projected path shape/request/flag mismatch')
    succeeded = data['projection_succeeded']
    for key in ('world_pose9', 'intrinsic_states'):
        if not np.isfinite(data[key][succeeded]).all() or not np.isnan(data[key][~succeeded]).all():
            raise IntegrityError('supplement failure placeholders or projected poses are malformed')
    if (data['fit_success'][~succeeded].any() or data['geometry_valid'][~succeeded].any()):
        raise IntegrityError('failed decoder paths have successful fit/geometry flags')
    for field, count in (('projection_failed_rows', (~succeeded).sum()), ('fit_success_rows', data['fit_success'].sum()),
                         ('geometry_valid_rows', data['geometry_valid'].sum())):
        if score_meta.get(field) != int(count):
            raise IntegrityError('supplement shard fit/failure/geometry counts disagree with saved paths')
    records = list(_read_gz_jsonl(score_path))
    if len(records) != len(rows):
        raise IntegrityError('supplement score shard has missing/extra candidate rows')
    for offset, (record, row, candidate) in enumerate(zip(records, rows, candidates)):
        common = _common(contract, inputs, int(row), int(candidate), map_id, multi)
        if any(record.get(key) != value for key, value in common.items()) or record.get('schema_version') != SCORE_SCHEMA_VERSION:
            raise IntegrityError('supplement score request identity/schema mismatch')
        metrics = record.get('metrics', {})
        for field, array in (('projection_succeeded', 'projection_succeeded'), ('projection_fit_success', 'fit_success'), ('projection_geometry_valid', 'geometry_valid')):
            if metrics.get(field) is not bool(data[array][offset]):
                raise IntegrityError('supplement score/projected fit/geometry identity mismatch')
        if metrics.get('schema_version') != METRICS_VERSION or not isinstance(metrics.get('candidate_success'), bool) or not isinstance(metrics.get('numeric_candidate_success'), bool):
            raise IntegrityError('supplement metric record schema/success flags malformed')
        if succeeded[offset]:
            if (metrics.get('projection_failed') is not False or
                    not isinstance(metrics.get('projection_before'), dict) or
                    not isinstance(metrics.get('projection_after'), dict) or
                    metrics['projection_after'].get('solver_converged') is not bool(data['fit_success'][offset])):
                raise IntegrityError('supplement fit diagnostics disagree with saved projection status')
            try:
                _scalars(metrics['projection_before']); _scalars(metrics['projection_after'])
            except ValueError as error:
                raise IntegrityError('supplement projection diagnostics are malformed') from error
        if not succeeded[offset] and (metrics.get('candidate_success') or metrics.get('numeric_candidate_success') or metrics.get('projection_failed') is not True):
            raise IntegrityError('decoder error candidate must remain a failed score row')
        try:
            json.dumps(record, allow_nan=False)
        except ValueError as error:
            raise IntegrityError('supplement score contains nonfinite JSON metrics') from error
    return len(records)


def score_map(args: argparse.Namespace, directory: Path, contract: dict, inputs: dict,
              map_id: str, multi: set[str], manifest: dict) -> int:
    paths, rows, candidates, raw_meta = _raw_requests(args, contract, inputs, map_id, multi)
    conditions, input_hashes = _prepare_conditions(args, inputs, rows, manifest, map_id)
    stem = map_id.rsplit('/', 1)[-1]
    score_path, projected_path = directory/'scores'/f'{stem}.jsonl.gz', directory/'projected'/f'{stem}.npz'
    expected_meta = dict(schema_version=SUPPLEMENT_VERSION, contract_sha256=contract['contract_sha256'],
        parent_contract_sha256=contract['parent_contract_sha256'], raw_sha256=raw_meta['raw_sha256'],
        raw_meta_sha256=raw_meta['raw_meta_sha256'], map_id=map_id, input_files_sha256=input_hashes,
        request_sha256=json_hash(dict(input_rows=rows.tolist(), candidate_indices=candidates.tolist())),
        input_sha256=[str(inputs['input_sha256'][row]) for row in rows],
        protocol_sha256=contract['protocol_sha256'], source_contract_sha256=contract['source_contract_sha256'],
        decoder_version=contract['decoder_version'], decoder_parameters_sha256=contract['decoder_parameters_sha256'],
        metrics_version=METRICS_VERSION, output_schema_sha256=contract['output_schema_sha256'])
    resumed = _resume(score_path, projected_path, expected_meta, inputs, rows, candidates, contract, map_id, multi)
    if resumed is not None:
        return resumed
    world = np.full((len(rows), 128, 9), np.nan, dtype=np.float64)
    intrinsic = np.full((len(rows), 128, 3), np.nan, dtype=np.float64)
    succeeded, fit_success, geometry_valid = (np.zeros(len(rows), dtype=bool) for _ in range(3))
    output = []
    for index, (path, row, candidate) in enumerate(zip(paths, rows, candidates)):
        condition = conditions[int(row)]
        before, after = {}, {}
        try:
            projected = project_smooth_contact_pose9(path, ContactProjectionSpec.from_condition(condition))
            if projected.decoder_parameters != smooth_contact_decoder_parameters():
                raise ValueError('decoder returned parameters different from the immutable protocol')
            actual_world, actual_intrinsic = _numpy(projected.world_pose9), _numpy(projected.intrinsic_states)
            if (actual_world.shape != (128, 9) or actual_intrinsic.shape != (128, 3) or
                    not np.isfinite(actual_world).all() or not np.isfinite(actual_intrinsic).all()):
                raise ValueError('decoder must return finite [128,9] poses and [128,3] intrinsic labels')
            before, after = _scalars(projected.before), _scalars(projected.after)
            if not isinstance(after.get('solver_converged'), bool):
                raise ValueError('decoder did not report a boolean solver convergence result')
            geometry = _scalars({'geometry_valid': projected.geometry_valid})['geometry_valid']
            if not isinstance(geometry, bool):
                raise ValueError('decoder did not report boolean geometry validity')
            world[index], intrinsic[index] = actual_world, actual_intrinsic
            succeeded[index], fit_success[index], geometry_valid[index] = True, after['solver_converged'], geometry
            try:
                metrics = evaluate_contact_path(actual_world, condition, projected_intrinsic=actual_intrinsic)
                if (not isinstance(metrics, dict) or metrics.get('schema_version') != METRICS_VERSION or
                        not isinstance(metrics.get('candidate_success'), bool) or
                        not isinstance(metrics.get('numeric_candidate_success'), bool)):
                    raise ValueError('metric evaluator returned a malformed candidate record')
                json.dumps(metrics, allow_nan=False)
            except Exception as error:
                metrics = _failure_metrics(error, 'evaluation_error')
            metrics.update(projection_failed=False, projection_succeeded=True, projection_fit_success=bool(fit_success[index]),
                projection_geometry_valid=bool(geometry_valid[index]), projection_before=before, projection_after=after)
        except Exception as error:
            metrics = _failure_metrics(error, 'projection_error')
            metrics.update(projection_failed=True, projection_succeeded=False, projection_fit_success=False,
                projection_geometry_valid=False, projection_before=before, projection_after=after)
        output.append(_common(contract, inputs, int(row), int(candidate), map_id, multi) |
                      dict(schema_version=SCORE_SCHEMA_VERSION, metrics=metrics))
    atomic_npz(projected_path, world_pose9=world, intrinsic_states=intrinsic, input_rows=rows,
        candidate_indices=candidates, projection_succeeded=succeeded, fit_success=fit_success, geometry_valid=geometry_valid)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = score_path.with_name(f'.{score_path.name}.{os.getpid()}.tmp')
    with gzip.open(temporary, 'wt', encoding='utf8') as stream:
        for record in output:
            stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n')
    os.replace(temporary, score_path)
    meta = expected_meta | dict(scores_sha256=sha256_file(score_path), projected_sha256=sha256_file(projected_path),
        score_rows=len(output), projected_rows=len(rows), projection_failed_rows=int((~succeeded).sum()),
        fit_success_rows=int(fit_success.sum()), geometry_valid_rows=int(geometry_valid.sum()))
    write_json(projected_path.with_suffix('.json'), meta)
    write_json(score_path.with_name(f'{stem}.json'), meta)
    # Fresh writes go through the same complete validation used on resume.
    return _resume(score_path, projected_path, expected_meta, inputs, rows, candidates, contract, map_id, multi)


def _failure_metrics(error: Exception, failure: str) -> dict:
    return dict(schema_version=METRICS_VERSION, candidate_success=False, numeric_candidate_success=False,
        valid=False, physical_collision_free=False, physical_collision_checked=False,
        dense_checked=False, dense_validation_completed=False, failures=[failure],
        evaluation_error=f'{type(error).__name__}: {error}', invalid=True,
        edge_decoder='intrinsic_linear_unwrapped_lift')


_SCORE_WORKER: tuple | None = None


def _limit_threads() -> None:
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _score_initialize(args: argparse.Namespace, directory: Path, contract: dict, inputs: dict,
                      multi: set[str], manifest: dict) -> None:
    global _SCORE_WORKER
    _limit_threads()
    _SCORE_WORKER = args, directory, contract, inputs, multi, manifest


def _score_worker(map_id: str) -> int:
    if _SCORE_WORKER is None:
        raise RuntimeError('smooth score worker was not initialized')
    return score_map(*_SCORE_WORKER[:4], map_id, *_SCORE_WORKER[4:])


def score(args: argparse.Namespace, model_id: str, manifest: dict, inputs: dict,
          parent_protocol: dict, supplement: dict, sources: dict) -> None:
    directory, contract = model_contract(args, model_id, parent_protocol, supplement, sources)
    maps = sorted(set(map(str, inputs['map_id'])))
    if args.limit_maps:
        maps = maps[:args.limit_maps]
    multi = multimodal_maps(inputs)
    expected_rows = sum(len(requests_for_map(inputs, map_id, multi)[0]) for map_id in maps)
    began, score_rows, completed = time.monotonic(), 0, 0
    status = dict(status='scoring', complete_maps=0, total_maps=len(maps), planned_total_maps=supplement['source_map_count'],
        score_rows=0, expected_score_rows=expected_rows, planned_score_rows=supplement['score_rows_per_model'],
        pilot_limit_maps=args.limit_maps, contract_sha256=contract['contract_sha256'])
    write_json(directory/'scoring_status.json', status)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=_score_initialize, initargs=(args, directory, contract, inputs, multi, manifest)) as pool:
            futures = {pool.submit(_score_worker, map_id): map_id for map_id in maps}
            for future in as_completed(futures):
                score_rows += future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(maps):
                    write_json(directory/'scoring_status.json', status | dict(complete_maps=completed, score_rows=score_rows,
                        elapsed_s=time.monotonic()-began))
                    print(f'SCORE_SMOOTH {model_id} {completed}/{len(maps)} maps elapsed={time.monotonic()-began:.1f}s', flush=True)
        if score_rows != expected_rows or (not args.limit_maps and (len(maps) != 1000 or score_rows != 34000)):
            raise IntegrityError('smooth scoring cannot complete with a map/candidate shortfall')
    except Exception as error:
        write_json(directory/'scoring_status.json', status | dict(status='scoring_failed', complete_maps=completed,
            score_rows=score_rows, error=f'{type(error).__name__}: {error}'))
        raise
    write_json(directory/'scoring_status.json', status | dict(status='pilot_complete' if args.limit_maps else 'complete',
        complete_maps=completed, score_rows=score_rows, elapsed_s=time.monotonic()-began))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--output', type=Path, help='default: EXPERIMENT/smooth_projection_supplement')
    parser.add_argument('--model', action='append', help='repeat to select approved models; default all 22')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--limit-maps', type=int, default=0, help='pilot only; never marks full evaluation complete')
    args = parser.parse_args(argv)
    if args.limit_maps < 0:
        parser.error('--limit-maps must be nonnegative')
    args.output = args.output or args.experiment/'smooth_projection_supplement'
    _limit_threads()
    manifest, inputs = load_benchmark(args.benchmark)
    parent, supplement, sources = ensure_supplement(args, manifest, inputs)
    models = list(dict.fromkeys(args.model or supplement['planned_models']))
    if any(model not in supplement['planned_models'] for model in models):
        parser.error('--model must name Base or an approved parent adapter')
    for model_id in models:
        score(args, model_id, manifest, inputs, parent, supplement, sources)


if __name__ == '__main__':
    main()
