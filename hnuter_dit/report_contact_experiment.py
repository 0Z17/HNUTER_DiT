"""Audited Contact research reports; incomplete runs are explicitly previews.

Only frozen input geometry and generated candidates are read. Feasibility comes
from the pre-inference input cache; the isolated oracle directory is never read.
Independent uncertainty units are source maps, not individual trajectories.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


REPORT_VERSION = 'contact_map_cluster_report_v001'
SCENARIOS = ('original', 'obstacle_shift')
PROJECTIONS = ('off', 'on')
SUCCESS_FIELDS = ('candidate_success', 'numeric_candidate_success', 'physical_collision_free',
                  'waypoint_candidate_success', 'practical_contact_gate', 'practical_normal_gate',
                  'attitude_valid', 'workspace_valid', 'shape_quality_gate', 'psi_quality_gate',
                  'endpoint_gate', 'chart_valid', 'tool_point_collision_free', 'dense_checked')
CONTINUOUS_FIELDS = ('contact_error_max_m', 'contact_error_mean_m', 'normal_error_max_rad',
                     'normal_error_mean_rad', 'minimum_clearance_m', 'path_length_m', 'path_efficiency',
                     'maximum_curvature_inv_m', 'p95_turn_angle_deg', 'cumulative_backtracking_m',
                     'jerk_rms_m_per_index3', 'acceleration_rms_m_per_index2', 'psi_total_variation_rad',
                     'psi_excess_reversal_variation_rad', 'maximum_abs_roll_deg', 'maximum_abs_pitch_deg',
                     'evaluation_time_s')
FIELDS = SUCCESS_FIELDS + CONTINUOUS_FIELDS
INDEX = {name: index for index, name in enumerate(FIELDS)}
_TRAINING_IDENTITIES: dict[tuple[str, str], dict[str, np.ndarray]] = {}
PAIRED_FIELDS = ('candidate_success', 'numeric_candidate_success', 'physical_collision_free',
                 'contact_error_max_m', 'normal_error_max_rad', 'path_length_m', 'path_efficiency')


class IntegrityError(RuntimeError):
    """A completed/partial report cannot consume changed or inconsistent artifacts."""


class IncompleteExperiment(RuntimeError):
    """No completed report is emitted when any planned experimental cell is absent."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf8', delete=False) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _check_hash(path: Path, expected: str) -> None:
    if not path.is_file() or _sha(path) != expected:
        raise IntegrityError(f'artifact checksum mismatch: {path}')


def _definition_hash(record: dict, *, timestamp: str) -> str:
    return _hash({key: value for key, value in record.items()
                  if key not in ('contract_sha256', 'protocol_sha256', timestamp)})


def _seed(base: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f'{base}/{label}'.encode()).digest()[:8], 'big')


def cluster_statistics(values: np.ndarray, clusters: np.ndarray, *, replicates: int = 1000,
                       seed: int = 20261011, names: tuple[str, ...] = FIELDS,
                       strata: np.ndarray | None = None) -> dict:
    """Condition-weighted means; resample complete source-map clusters.

    Missing scalar diagnostics have their own observed denominator. They are
    never imputed as zero, while every failed candidate's success is false.
    Percentile CIs describe test-map uncertainty; model-seed std is separate.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    clusters = np.asarray(clusters)
    if values.shape != (len(clusters), len(names)) or replicates < 1:
        raise ValueError('values/cluster/metric shape or bootstrap budget mismatch')
    if strata is not None and len(strata) != len(clusters):
        raise ValueError('bootstrap family strata shape mismatch')
    planned_conditions = len(clusters); planned_maps = len(np.unique(clusters))
    usable = np.isfinite(values).any(axis=1)
    if strata is not None:
        strata = np.asarray(strata)[usable]
    values, clusters = values[usable], clusters[usable]
    unique, inverse = np.unique(clusters, return_inverse=True)
    sums = np.zeros((len(unique), len(names)))
    counts = np.zeros_like(sums)
    present = np.isfinite(values)
    np.add.at(sums, inverse, np.where(present, values, 0.))
    np.add.at(counts, inverse, present.astype(float))
    count = present.sum(axis=0)
    mean = np.divide(sums.sum(axis=0), count, out=np.full(len(names), np.nan), where=count > 0)
    if len(unique):
        random = np.random.default_rng(seed)
        if strata is None:
            weights = random.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=replicates)
        else:
            cluster_families = np.empty(len(unique), dtype=object)
            for index in range(len(unique)):
                families = np.unique(strata[inverse == index])
                if len(families) != 1:
                    raise ValueError('a source map appears in different bootstrap family strata')
                cluster_families[index] = families[0]
            weights = np.zeros((replicates, len(unique)), dtype=np.int64)
            for family in np.unique(cluster_families):
                selected = np.flatnonzero(cluster_families == family)
                weights[:, selected] = random.multinomial(len(selected), np.full(len(selected), 1/len(selected)), size=replicates)
        denominator = weights @ counts
        sampled = np.divide(weights @ sums, denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
    else:
        sampled = np.full((replicates, len(names)), np.nan)
    result = {}
    for index, name in enumerate(names):
        observed = values[present[:, index], index]
        finite = sampled[np.isfinite(sampled[:, index]), index]
        result[name] = {'observations': int(count[index]), 'mean': float(mean[index]) if count[index] else None,
            'std': float(observed.std(ddof=1)) if len(observed) > 1 else 0. if len(observed) else None,
            'median': float(np.median(observed)) if len(observed) else None,
            'p05': float(np.percentile(observed, 5)) if len(observed) else None,
            'p95': float(np.percentile(observed, 95)) if len(observed) else None,
            'ci95': np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [None, None]}
    return {'source_map_count': len(unique), 'condition_count': planned_conditions,
            'planned_source_map_count': planned_maps, 'observed_condition_count': len(clusters),
            'bootstrap_replicates': replicates,
            'bootstrap_policy': 'family-stratified source-map cluster' if strata is not None else 'source-map cluster', 'metrics': result}


def _multimodal_maps(inputs: dict[str, np.ndarray], per_family: int) -> set[str]:
    families = defaultdict(set)
    for family, map_id in zip(inputs['family'], inputs['map_id']):
        families[str(family)].add(str(map_id))
    selected = set()
    for _, maps in sorted(families.items()):
        ranked = sorted(maps, key=lambda key:hashlib.sha256(f'20261011/multimodal/{key}'.encode()).hexdigest())
        selected.update(ranked[:per_family])
    return selected


def _load_inputs(benchmark: Path, protocol: dict) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = _read(benchmark/'manifest.json')
    if manifest['status'] != 'frozen':
        raise IncompleteExperiment('benchmark is not frozen')
    if (manifest['condition_count'] != protocol['new_test_conditions'] or
            manifest['source_map_count'] != protocol['new_source_maps']):
        raise IncompleteExperiment('benchmark has fewer/different tasks or maps than the approved protocol')
    # Do not even open isolated oracle files. Hashes of physical inputs/cache suffice.
    for name, digest in manifest['input_files_sha256'].items():
        if not Path(name).parts or Path(name).is_absolute() or '..' in Path(name).parts:
            raise IntegrityError('unsafe benchmark artifact path')
        if Path(name).parts[0] != 'oracle':
            _check_hash(benchmark/name, digest)
    with np.load(benchmark/manifest['runtime_cache'], allow_pickle=False) as archive:
        inputs = {name: archive[name].copy() for name in archive.files}
    required = ('case_id', 'condition_id', 'map_id', 'family', 'kind', 'difficulty', 'scenario',
                'feasible_mask', 'feasibility_status', 'input_sha256', 'input_shard', 'input_row')
    total = 2*manifest['condition_count']
    if any(len(inputs[name]) != total for name in required) or len(set(inputs['case_id'])) != total:
        raise IntegrityError('frozen runtime cache has repeated/missing scenario cases')
    for scenario in SCENARIOS:
        rows = inputs['scenario'] == scenario
        if rows.sum() != manifest['condition_count'] or len(set(inputs['condition_id'][rows])) != manifest['condition_count']:
            raise IntegrityError('frozen cache scenario is incomplete')
    if set(inputs['scenario']) != set(SCENARIOS) or len(set(inputs['map_id'])) != manifest['source_map_count']:
        raise IntegrityError('frozen cache has inconsistent scene identities')
    return manifest, inputs


def _model_provenance(experiment: Path, row: dict, protocol: dict, contract: dict,
                      problems: list[dict]) -> dict:
    if row['id'] == 'base':
        return {'variant': 'base', 'train_size': 0, 'seed': None, 'frozen_base_sha256': protocol['base_sha256']}
    directory = experiment/'models'/row['id']
    required = ('resolved_training_config.json', 'run_status.json', 'best.pt', 'training_subset.json')
    if any(not (directory/name).is_file() for name in required):
        problems.append({'model_id': row['id'], 'stage': 'training', 'reason': 'missing config/status/selected checkpoint'})
        return row
    config = _read(directory/required[0]); status = _read(directory/required[1])
    if config['contract_sha256'] != _definition_hash(config, timestamp='created_at_utc'):
        raise IntegrityError('training definition hash mismatch')
    if (config['base_sha256'] != protocol['base_sha256'] or config['seed'] != row['seed'] or
            config['train_size'] != row['train_size'] or config['training_steps'] != protocol['optimizer_updates']):
        raise IntegrityError('model config differs from approved matrix')
    if status['status'] != 'complete' or status['step'] != protocol['optimizer_updates']:
        problems.append({'model_id': row['id'], 'stage': 'training', 'reason': 'planned optimizer budget incomplete'})
    else:
        _check_hash(directory/'best.pt', status['selected_checkpoint_sha256'])
        if contract['adapter_sha256'] != status['selected_checkpoint_sha256']:
            raise IntegrityError('scored adapter differs from validation-selected checkpoint')
    if status.get('contract_sha256') != config['contract_sha256']:
        raise IntegrityError('training status/config contract mismatch')
    subset = _read(directory/'training_subset.json')
    if _hash(subset) != config['subset_sha256']:
        raise IntegrityError('training subset differs from selected model contract')
    cache_path = Path(config['dataset_root'])/'prepared_dataset.npz'
    cache_key = (str(cache_path.resolve()), config['prepared_cache_sha256'])
    if cache_key not in _TRAINING_IDENTITIES:
        _check_hash(cache_path, config['prepared_cache_sha256'])
        with np.load(cache_path, allow_pickle=False) as archive:
            _TRAINING_IDENTITIES[cache_key] = {key: archive[key].copy() for key in ('map_ids', 'condition_ids', 'split_codes')}
    identities = _TRAINING_IDENTITIES[cache_key]
    indices = np.asarray(subset['trajectory_indices'])
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(indices) != row['train_size'] or
            len(set(indices)) != len(indices) or (indices < 0).any() or (indices >= len(identities['map_ids'])).any()):
        raise IntegrityError('training subset does not contain the planned distinct expert paths')
    map_ids = identities['map_ids'][indices]; condition_ids = identities['condition_ids'][indices]
    if (np.any(identities['split_codes'][indices] != 0) or set(map_ids) != set(subset['selected_map_ids']) or
            len(set(map_ids)) != subset['map_count'] or subset['map_count']*20 != row['train_size']):
        raise IntegrityError('whole-map training subset identities/splits are inconsistent')
    for map_id in set(map_ids):
        selected = map_ids == map_id
        _, counts = np.unique(condition_ids[selected], return_counts=True)
        if selected.sum() != 20 or len(counts) != 14 or (counts == 2).sum() != 6 or (counts == 1).sum() != 8:
            raise IntegrityError('selected training map does not retain all 14 conditions / 20 paired experts')
    if config.get('dataset_freeze_sha256') != protocol.get('dataset_freeze_sha256'):
        raise IntegrityError('training dataset freeze differs from approved experiment')
    if row['variant'] == 'no_geometry_tokens' and config['adapter_config']['task_conditioning'] != 'none':
        raise IntegrityError('no_geometry control unexpectedly uses geometry tokens')
    if row['variant'] == 'no_contact_loss' and any(config['loss_weights'][key] != 0 for key in ('contact', 'normal', 'smooth')):
        raise IntegrityError('no_contact_loss control retains contact-related auxiliary losses')
    return {'variant': row['variant'], 'train_size': row['train_size'], 'seed': row['seed'],
            'training_config': config, 'training_status': status, 'config_sha256': _sha(directory/required[0]),
            'effective_training_counts': {'expert_paths': len(indices), 'source_maps': len(set(map_ids)),
                'distinct_task_conditions': len(set(condition_ids)), 'conditions_per_source_map': 14,
                'expert_paths_per_source_map': 20, 'subset_file_sha256': _sha(directory/'training_subset.json'),
                'prepared_identity_audit': True}, 'selected_source_maps': subset['selected_map_ids']}


def _read_model(experiment: Path, benchmark: Path, row: dict, protocol: dict,
                inputs: dict, manifest_sha: str, problems: list[dict]) -> dict | None:
    model_id = row['id']; directory = experiment/'evaluation'/model_id
    if not (directory/'contract.json').is_file():
        problems.append({'model_id': model_id, 'stage': 'evaluation', 'reason': 'evaluation contract absent'})
        return None
    contract = _read(directory/'contract.json')
    if contract['contract_sha256'] != _definition_hash(contract, timestamp='created_utc'):
        raise IntegrityError('evaluation definition hash mismatch')
    if (contract['model_id'] != model_id or contract['benchmark_manifest_sha256'] != manifest_sha or
            contract['base_sha256'] != protocol['base_sha256'] or contract['benchmark_conditions'] != protocol['new_test_conditions'] or
            contract['primary_candidates'] != 1 or contract['multimodal_candidates'] != 8 or
            contract['ddim_steps'] != protocol['sampling_steps'] or contract['latent_seed'] != protocol['latent_seed']):
        raise IntegrityError('model evaluation differs from fixed paired inference protocol')
    for section, names in (('evaluation_sources', ('contact_projection.py',)),
                           ('inference_sources', ('model.py', 'geometry.py'))):
        for name in names:
            if name in contract.get(section, {}):
                _check_hash(Path(__file__).parent/name, contract[section][name])
    training = _model_provenance(experiment, row, protocol, contract, problems)
    status_path = directory/'scoring_status.json'
    status = _read(status_path) if status_path.exists() else {'status': 'missing'}
    if status.get('status') != 'complete':
        problems.append({'model_id': model_id, 'stage': 'scoring', 'reason': 'scoring_status is not complete'})
    if 'contract_sha256' in status and status['contract_sha256'] != contract['contract_sha256']:
        raise IntegrityError('scoring status contract mismatch')
    multi = _multimodal_maps(inputs, contract['multimodal_maps_per_family'])
    multi_rows = np.flatnonzero(np.isin(inputs['map_id'], list(multi)))
    if len(multi_rows) != 2*protocol['multimodal_subset_conditions']:
        raise IntegrityError('fixed multimodal subset differs from protocol condition count')
    n = len(inputs['case_id']); primary = np.full((2, n, len(FIELDS)), np.nan)
    multiple = np.full((2, len(multi_rows), 8, len(FIELDS)), np.nan)
    k_lookup = {int(value): index for index, value in enumerate(multi_rows)}
    case_lookup = {str(value): index for index, value in enumerate(inputs['case_id'])}
    seen = np.zeros((2, n, 8), dtype=bool)
    errors = defaultdict(int); failures = defaultdict(int); hashes = {}; raw_index = {}
    maps = sorted(set(map(str, inputs['map_id'])))
    for map_id in maps:
        stem = map_id.rsplit('/', 1)[-1]
        raw_path = directory/'raw'/f'{stem}.npz'; raw_meta_path = raw_path.with_suffix('.json')
        score_path = directory/'scores'/f'{stem}.jsonl.gz'; score_meta_path = score_path.with_name(f'{stem}.json')
        if any(not path.is_file() for path in (raw_path, raw_meta_path, score_path, score_meta_path)):
            problems.append({'model_id': model_id, 'map_id': map_id, 'stage': 'shard', 'reason': 'raw or scored shard absent'})
            continue
        raw_meta = _read(raw_meta_path); score_meta = _read(score_meta_path)
        if (raw_meta['contract_sha256'] != contract['contract_sha256'] or score_meta['contract_sha256'] != contract['contract_sha256'] or
                score_meta['raw_sha256'] != raw_meta['raw_sha256'] or score_meta['map_id'] != map_id or raw_meta['map_id'] != map_id):
            raise IntegrityError('map shard model/raw contract mismatch')
        _check_hash(raw_path, raw_meta['raw_sha256']); _check_hash(score_path, score_meta['scores_sha256'])
        hashes[str(raw_path.relative_to(experiment))] = raw_meta['raw_sha256']
        hashes[str(score_path.relative_to(experiment))] = score_meta['scores_sha256']
        rows = np.flatnonzero(inputs['map_id'] == map_id); count = 8 if map_id in multi else 1
        expected_rows, expected_candidates = np.repeat(rows, count), np.tile(np.arange(count), len(rows))
        with np.load(raw_path, allow_pickle=False) as archive:
            if (not np.array_equal(archive['input_rows'], expected_rows) or
                    not np.array_equal(archive['candidate_indices'], expected_candidates) or
                    archive['world_pose9'].shape != (len(expected_rows), 128, 9)):
                raise IntegrityError('generated raw requests differ from frozen candidates/conditions')
        if map_id in multi:
            raw_index[map_id] = str(raw_path)
        actual_rows = 0
        with gzip.open(score_path, 'rt', encoding='utf8') as stream:
            for line in stream:
                score = json.loads(line); actual_rows += 1
                if score.get('case_id') not in case_lookup or score.get('projection') not in PROJECTIONS:
                    raise IntegrityError('unexpected scored case or projection')
                index = case_lookup[score['case_id']]; candidate = score['candidate_index']; projection = PROJECTIONS.index(score['projection'])
                if not isinstance(candidate, int) or candidate < 0 or candidate >= count or inputs['map_id'][index] != map_id:
                    raise IntegrityError('unexpected scored candidate/map')
                for key in ('condition_id', 'family', 'kind', 'difficulty', 'scenario', 'input_sha256', 'feasibility_status'):
                    if score[key] != str(inputs[key][index]):
                        raise IntegrityError(f'scored physical case identity mismatch: {key}')
                if score['model_id'] != model_id or bool(score['feasible']) != bool(inputs['feasible_mask'][index]) or bool(score['multimodal_subset']) != (map_id in multi):
                    raise IntegrityError('scored model/feasibility/subset identity mismatch')
                if seen[projection, index, candidate]:
                    raise IntegrityError('duplicate scored candidate')
                seen[projection, index, candidate] = True
                metrics = score['metrics']
                if not isinstance(metrics.get('candidate_success'), bool) or not isinstance(metrics.get('numeric_candidate_success'), bool):
                    raise IntegrityError('missing/invalid scored success gates')
                values = np.array([float(metrics[name]) if isinstance(metrics.get(name), (int, float, bool)) and
                                   math.isfinite(float(metrics[name])) else np.nan for name in FIELDS])
                if candidate == 0:
                    primary[projection, index] = values
                if index in k_lookup:
                    multiple[projection, k_lookup[index], candidate] = values
                if metrics.get('evaluation_error'):
                    errors[(score['scenario'], score['projection'])] += 1
                for failure in metrics.get('failures', []):
                    failures[(score['scenario'], score['projection'], failure)] += 1
        if actual_rows != score_meta['score_rows']:
            raise IntegrityError('scored shard row count differs from hashed metadata')
    expected = np.zeros_like(seen)
    expected[:, :, 0] = True; expected[:, multi_rows, :] = True
    missing = expected & ~seen
    if missing.any():
        problems.append({'model_id': model_id, 'stage': 'cells', 'reason': 'missing planned candidate/projection/scenario cells',
                         'missing_score_rows': int(missing.sum()), 'missing_primary_cells': int(missing[:, :, 0].sum())})
    if status.get('status') == 'complete' and (status.get('complete_maps') != len(maps) or status.get('total_maps') != len(maps) or
            status.get('score_rows') != int(expected.sum())):
        problems.append({'model_id': model_id, 'stage': 'status', 'reason': 'complete status counts differ from planned matrix'})
    return {'model_id': model_id, 'training': training, 'contract': contract, 'primary': primary,
            'multiple': multiple, 'multi_rows': multi_rows, 'seen': seen, 'raw_index': raw_index,
            'artifact_sha256': hashes, 'complete': not missing.any() and status.get('status') == 'complete',
            'evaluation_errors': [{'scenario': s, 'projection': p, 'count': v} for (s, p), v in sorted(errors.items())],
            'failure_counts': [{'scenario': s, 'projection': p, 'failure': f, 'count': v} for (s, p, f), v in sorted(failures.items())]}


def _scopes(inputs: dict) -> list[tuple[dict, np.ndarray]]:
    result = [({'scope': 'overall'}, np.ones(len(inputs['case_id']), bool))]
    groups = sorted(set(zip(inputs['family'], inputs['kind'], inputs['difficulty'])))
    for family, kind, difficulty in groups:
        result.append(({'scope': 'family_kind_difficulty', 'family': str(family), 'kind': str(kind), 'difficulty': str(difficulty)},
                       (inputs['family'] == family) & (inputs['kind'] == kind) & (inputs['difficulty'] == difficulty)))
    return result


def _summary_cells(values: np.ndarray, inputs: dict, replicates: int, seed: int,
                   *, names: tuple[str, ...] = FIELDS) -> list[dict]:
    result = []
    for scope, mask in _scopes(inputs):
        for scenario in SCENARIOS:
            for selection in ('all_stress', 'frozen_feasible'):
                selected = mask & (inputs['scenario'] == scenario)
                if selection == 'frozen_feasible':
                    selected &= inputs['feasible_mask']
                for projection, name in enumerate(PROJECTIONS):
                    label = scope | {'scenario': scenario, 'projection': name, 'selection': selection}
                    statistics = cluster_statistics(values[projection, selected], inputs['map_id'][selected],
                        replicates=replicates, seed=_seed(seed, _json(label)), names=names, strata=inputs['family'][selected])
                    result.append(label | statistics)
    return result


def _paired_effects(primary: np.ndarray, inputs: dict, replicates: int, seed: int) -> list[dict]:
    effects = []
    fields = [INDEX[name] for name in PAIRED_FIELDS]
    original = {str(value): index for index, value in enumerate(inputs['condition_id']) if inputs['scenario'][index] == 'original'}
    shifted = {str(value): index for index, value in enumerate(inputs['condition_id']) if inputs['scenario'][index] == 'obstacle_shift'}
    for scope, mask in _scopes(inputs):
        for scenario in SCENARIOS:
            for selection in ('all_stress', 'frozen_feasible'):
                selected = mask & (inputs['scenario'] == scenario)
                if selection == 'frozen_feasible':
                    selected &= inputs['feasible_mask']
                delta = primary[1, selected][:, fields]-primary[0, selected][:, fields]
                label = scope | {'effect': 'projection_on_minus_off', 'scenario': scenario, 'selection': selection}
                effects.append(label | cluster_statistics(delta, inputs['map_id'][selected], names=PAIRED_FIELDS,
                    replicates=replicates, seed=_seed(seed, _json(label)), strata=inputs['family'][selected]))
        a = np.array([index for _, index in sorted(original.items()) if mask[index]], dtype=int)
        b = np.array([shifted[str(inputs['condition_id'][index])] for index in a], dtype=int)
        for selection in ('all_stress', 'both_scenarios_frozen_feasible'):
            choose = np.ones(len(a), bool) if selection == 'all_stress' else inputs['feasible_mask'][a] & inputs['feasible_mask'][b]
            for projection, name in enumerate(PROJECTIONS):
                delta = primary[projection, b[choose]][:, fields]-primary[projection, a[choose]][:, fields]
                label = scope | {'effect': 'shift_minus_original', 'projection': name, 'selection': selection}
                effects.append(label | cluster_statistics(delta, inputs['map_id'][a[choose]], names=PAIRED_FIELDS,
                    replicates=replicates, seed=_seed(seed, _json(label)), strata=inputs['family'][a[choose]]))
    return effects


def _mean_stack(arrays: list[np.ndarray]) -> np.ndarray:
    values = np.stack(arrays); present = np.isfinite(values)
    return np.divide(np.where(present, values, 0.).sum(axis=0), present.sum(axis=0),
                     out=np.full(values.shape[1:], np.nan), where=present.sum(axis=0) > 0)


def _seed_cell_statistics(cells: list[dict], values: list[np.ndarray], inputs: dict,
                          names: tuple[str, ...]) -> None:
    for cell in cells:
        mask = inputs['scenario'] == cell['scenario']
        for key in ('family', 'kind', 'difficulty'):
            if key in cell:
                mask &= inputs[key] == cell[key]
        if cell['selection'] == 'frozen_feasible':
            mask &= inputs['feasible_mask']
        projection = PROJECTIONS.index(cell['projection'])
        for index, name in enumerate(names):
            means = []
            for array in values:
                observed = array[projection, mask, index]
                means.append(float(observed[np.isfinite(observed)].mean()) if np.isfinite(observed).any() else None)
            finite = [value for value in means if value is not None]
            cell['metrics'][name]['seed_means'] = means
            cell['metrics'][name]['seed_std'] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0. if finite else None


def _rotations(path: np.ndarray) -> np.ndarray:
    first = path[..., 3:6]; second = path[..., 6:9]
    first = first/np.linalg.norm(first, axis=-1, keepdims=True)
    second = second-(first*second).sum(axis=-1, keepdims=True)*first
    second = second/np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), -1)


def empirical_direction(uv: np.ndarray, start: np.ndarray, goal: np.ndarray,
                        *, tolerance_m: float = .05) -> str:
    """Mean signed UV deviation from the requested chord, not a homotopy label."""
    delta = np.asarray(goal[:2])-start[:2]; length = np.linalg.norm(delta)
    if length < 1e-10:
        return 'ambiguous'
    relative = uv-np.asarray(start[:2])
    signed = (delta[0]*relative[:, 1]-delta[1]*relative[:, 0])/length
    progress = relative@delta/(length*length)
    interior = (progress > .2) & (progress < .8)
    if not interior.any():
        return 'ambiguous'
    value = float(signed[interior].mean())
    return 'positive' if value > tolerance_m else 'negative' if value < -tolerance_m else 'near_chord'


def diversity_for_paths(paths: np.ndarray, successful: np.ndarray, condition: dict, *, projection: str) -> dict:
    """Successful candidate pairs only; replay just the analytic decoder, no QA/COAL."""
    from .contact_projection import ContactProjectionSpec, lift_contact_intrinsic
    from .contact_projection import _safe_rotations, _uv_from_tip, _contact_basis, _psi_from_tool
    import torch
    success = np.flatnonzero(successful)
    if not len(success):
        return {'successful_candidates': 0, 'successful_pairs': 0, 'position_pair_msd_m2': None,
                'position_pair_rms_m': None, 'uv_pair_msd_m2': None, 'direction_proxy': {}, 'both_direction_signs': False}
    spec = ContactProjectionSpec.from_condition(condition)
    selected = np.asarray(paths[success], dtype=np.float64)
    rotation = _safe_rotations(torch.as_tensor(selected))[0].numpy()
    tip = selected[..., :3] + np.einsum('...ij,j->...i', rotation, spec.tool_offset)
    uv = _uv_from_tip(torch.as_tensor(tip), spec)[0]
    if projection == 'on':
        uv = uv.clamp(torch.as_tensor(spec.bounds[:2, 0]), torch.as_tensor(spec.bounds[:2, 1]))
        _, _, basis = _contact_basis(uv, spec)
        psi = _psi_from_tool(torch.as_tensor(rotation@spec.tool_rotation), basis, spec)[0]
        states = torch.cat((uv, psi[..., None]), -1)
        states[:, 0] = torch.as_tensor(spec.start); states[:, -1] = torch.as_tensor(spec.goal)
        # Match float32 world serialization and immutable original endpoints.
        position = lift_contact_intrinsic(states, spec).numpy()[..., :3].astype(paths.dtype)
        position[:, [0, -1]] = paths[success][:, [0, -1], :3]
        uv = states[..., :2]
    elif projection == 'off':
        position = paths[success, :, :3]
    else:
        raise ValueError('invalid projection')
    uv = uv.numpy(); position_msd = []; uv_msd = []
    for first in range(len(success)):
        for second in range(first+1, len(success)):
            position_msd.append(float(np.mean(np.sum((position[first]-position[second])**2, axis=-1))))
            uv_msd.append(float(np.mean(np.sum((uv[first]-uv[second])**2, axis=-1))))
    directions = defaultdict(int)
    for path_uv in uv:
        directions[empirical_direction(path_uv, spec.start, spec.goal)] += 1
    return {'successful_candidates': len(success), 'successful_pairs': len(position_msd),
            'position_pair_msd_m2': float(np.mean(position_msd)) if position_msd else None,
            'position_pair_rms_m': float(np.mean(np.sqrt(position_msd))) if position_msd else None,
            'uv_pair_msd_m2': float(np.mean(uv_msd)) if uv_msd else None,
            'direction_proxy': dict(directions), 'both_direction_signs': bool(directions['positive'] and directions['negative'])}


def _diversity(model: dict, inputs: dict, benchmark: Path, replicates: int, seed: int,
               condition_cache: dict[str, list[dict]]) -> dict:
    from .contact_benchmark import _read_gz_jsonl, _parse_input_record, parse_input_condition
    multiple = model['multiple']; rows = model['multi_rows']; k_lookup = {int(row): i for i, row in enumerate(rows)}
    any_values = np.full((2, len(rows), 3), np.nan)
    diversity_values = np.full((2, len(rows), 5), np.nan)
    details = []
    for projection in range(2):
        success = multiple[projection, :, :, INDEX['candidate_success']]
        numeric = multiple[projection, :, :, INDEX['numeric_candidate_success']]
        physical = multiple[projection, :, :, INDEX['physical_collision_free']]
        complete = np.isfinite(success).all(axis=1)
        for j, values in enumerate((success, numeric, physical)):
            usable = complete & np.isfinite(values).all(axis=1)
            any_values[projection, usable, j] = np.any(values[usable] > .5, axis=1)
    for map_id, name in sorted(model['raw_index'].items()):
        with np.load(name, allow_pickle=False) as archive:
            paths, input_rows, candidates = archive['world_pose9'], archive['input_rows'], archive['candidate_indices']
        for row in np.unique(input_rows):
            selected = np.flatnonzero(input_rows == row)
            if not np.array_equal(candidates[selected], np.arange(8)):
                raise IntegrityError('diversity raw subset lacks all eight candidates')
            index = k_lookup[int(row)]; relative = str(inputs['input_shard'][row])
            if relative not in condition_cache:
                condition_cache[relative] = [_parse_input_record(record) for record in _read_gz_jsonl(benchmark/relative)]
            record = condition_cache[relative][int(inputs['input_row'][row])]
            condition = parse_input_condition(record['conditions'][str(inputs['scenario'][row])])
            if _hash(condition) != str(inputs['input_sha256'][row]):
                raise IntegrityError('diversity whitelist input checksum mismatch')
            for projection, name in enumerate(PROJECTIONS):
                scored = multiple[projection, index, :, INDEX['candidate_success']]
                if not np.isfinite(scored).all():
                    continue
                data = diversity_for_paths(paths[selected], scored > .5, condition, projection=name)
                label = {'condition_id': str(inputs['condition_id'][row]), 'map_id': map_id,
                         'scenario': str(inputs['scenario'][row]), 'projection': name,
                         'kind': str(inputs['kind'][row]), 'difficulty': str(inputs['difficulty'][row]), 'family': str(inputs['family'][row])}
                details.append(label | data)
                diversity_values[projection, index] = [data['successful_candidates'],
                    data['position_pair_msd_m2'] if data['position_pair_msd_m2'] is not None else np.nan,
                    data['position_pair_rms_m'] if data['position_pair_rms_m'] is not None else np.nan,
                    data['uv_pair_msd_m2'] if data['uv_pair_msd_m2'] is not None else np.nan,
                    float(data['both_direction_signs']) if data['successful_candidates'] >= 2 else np.nan]
    sub_inputs = {key: value[rows] for key, value in inputs.items()}
    return {'_any_values': any_values, '_diversity_values': diversity_values,
            'subset_conditions': len(rows)//2, 'candidates': 8, 'selection': 'preselected source maps, identical across models',
            'any_at_8': _summary_cells(any_values, sub_inputs, replicates, seed,
                names=('any_at_8', 'numeric_any_at_8', 'physical_collision_any_at_8')),
            'diversity': _summary_cells(diversity_values, sub_inputs, replicates, seed,
                names=('successful_candidate_count', 'position_pair_msd_m2', 'position_pair_rms_m',
                       'uv_pair_msd_m2', 'both_empirical_direction_signs')),
            'condition_pair_details': details,
            'interpretation': 'condition-weighted averages of successful-pair statistics; direction is a chord-deviation proxy, not homotopy coverage'}


def _seed_groups(models: dict[str, dict], matrix: list[dict], inputs: dict, replicates: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for row in matrix:
        if row['id'] in models:
            groups[(row['variant'], row['train_size'])].append(row)
    result = []
    for (variant, size), rows in sorted(groups.items()):
        values = [models[row['id']]['primary'] for row in rows]
        average = _mean_stack(values)
        cells = _summary_cells(average, inputs, replicates, seed)
        _seed_cell_statistics(cells, values, inputs, FIELDS)
        for cell in cells:
            cell['candidate_success_seed_means'] = cell['metrics']['candidate_success']['seed_means']
            cell['candidate_success_seed_std'] = cell['metrics']['candidate_success']['seed_std']
        result.append({'variant': variant, 'train_size': size, 'seeds': [row['seed'] for row in rows],
                       'model_ids': [row['id'] for row in rows], 'cells': cells,
                       'paired_effects': _paired_effects(average, inputs, replicates, seed),
                       'seed_statistics': 'per-case average across seeds followed by source-map bootstrap; separate between-seed sample std'})
        if all('_any_values' in models[row['id']] for row in rows):
            selected = models[rows[0]['id']]['multi_rows']
            sub_inputs = {key: value[selected] for key, value in inputs.items()}
            result[-1]['multimodal_any_at_8'] = _summary_cells(_mean_stack([models[row['id']]['_any_values'] for row in rows]),
                sub_inputs, replicates, seed, names=('any_at_8', 'numeric_any_at_8', 'physical_collision_any_at_8'))
            result[-1]['multimodal_diversity'] = _summary_cells(_mean_stack([models[row['id']]['_diversity_values'] for row in rows]),
                sub_inputs, replicates, seed, names=('successful_candidate_count', 'position_pair_msd_m2', 'position_pair_rms_m',
                                                   'uv_pair_msd_m2', 'both_empirical_direction_signs'))
            _seed_cell_statistics(result[-1]['multimodal_any_at_8'], [models[row['id']]['_any_values'] for row in rows],
                sub_inputs, ('any_at_8', 'numeric_any_at_8', 'physical_collision_any_at_8'))
            _seed_cell_statistics(result[-1]['multimodal_diversity'], [models[row['id']]['_diversity_values'] for row in rows],
                sub_inputs, ('successful_candidate_count', 'position_pair_msd_m2', 'position_pair_rms_m',
                             'uv_pair_msd_m2', 'both_empirical_direction_signs'))
    return result


def _ablation_pairs(models: dict[str, dict], matrix: list[dict], inputs: dict, replicates: int, seed: int) -> list[dict]:
    result = []; by_seed = {(row['variant'], row['train_size'], row['seed']): row['id'] for row in matrix}
    for row in matrix:
        if row['variant'] != 'geometry' or row['id'] not in models:
            continue
        controls = ['base']
        if row['train_size'] == 8000:
            controls.extend(by_seed[key] for key in (('no_contact_loss', 8000, row['seed']),
                ('no_geometry_tokens', 8000, row['seed'])) if key in by_seed)
        for control in controls:
            if control not in models:
                continue
            delta = models[row['id']]['primary']-models[control]['primary']
            result.append({'effect': 'geometry_adapter_minus_control', 'model_id': row['id'], 'control_id': control,
                           'train_size': row['train_size'], 'seed': row['seed'],
                           'cells': _summary_cells(delta, inputs, replicates, seed)})
    return result


def _ablation_group_pairs(models: dict[str, dict], matrix: list[dict], inputs: dict, replicates: int, seed: int) -> list[dict]:
    grouped = defaultdict(list)
    lookup = {(row['variant'], row['train_size'], row['seed']): row['id'] for row in matrix}
    for row in matrix:
        if row['variant'] != 'geometry' or row['id'] not in models:
            continue
        controls = [('base', 'base')]
        if row['train_size'] == 8000:
            controls += [(variant, lookup[(variant, 8000, row['seed'])]) for variant in
                ('no_contact_loss', 'no_geometry_tokens') if (variant, 8000, row['seed']) in lookup]
        for variant, model_id in controls:
            if model_id in models:
                grouped[(row['train_size'], variant)].append((row['seed'], models[row['id']]['primary']-models[model_id]['primary']))
    result = []
    for (size, variant), pairs in sorted(grouped.items()):
        arrays = [value for _, value in pairs]
        cells = _summary_cells(_mean_stack(arrays), inputs, replicates, seed)
        _seed_cell_statistics(cells, arrays, inputs, FIELDS)
        result.append({'effect': 'geometry_adapter_minus_control', 'train_size': size, 'control_variant': variant,
                       'seeds': [value for value, _ in pairs], 'cells': cells})
    return result


def _overall(cells: list[dict], *, scenario: str, projection: str, selection: str = 'all_stress') -> dict | None:
    return next((row for row in cells if row['scope'] == 'overall' and row['scenario'] == scenario and
                 row['projection'] == projection and row['selection'] == selection), None)


def _percent(value: float | None) -> str:
    return '—' if value is None else f'{100*value:.2f}%'


def _markdown(report: dict, figures: list[str]) -> str:
    complete = report['status'] == 'complete'
    lines = [f"# {'Contact Adapter 消融与泛化研究报告' if complete else '未完成：Contact Adapter 实验预览'}", '',
        f"状态：{'全部计划实验已完成并通过完整性检查' if complete else '实验未完成，以下仅为已观察数据，不能作为完整结论'}。", '',
        f"冻结测试集含 {report['benchmark']['condition_count']:,} 个不同条件、{report['benchmark']['source_map_count']:,} 张新源地图。"
        f"模型种子和同一地图内的任务并不是额外的独立测试地图。95% CI 在每个场景族内以源地图为聚类单位重采样 {report['bootstrap_replicates']} 次，保留场景族分层数量。", '',
        '## 协议与结果口径', '',
        f"所有模型主结果为 candidate_index=0；DDIM {report['protocol']['sampling_steps']} 步、同一条件/候选 latent seed。"
        f"固定 {report['protocol']['multimodal_subset_conditions']:,} 条条件上的 Any@8 是附加结果，不能替代主评测。", '',
        '原始输出与解析接触投影使用配对候选。Base+投影作为必要基线；投影只读取任务几何、校准工具变换和预测轨迹，不读取专家轨迹或 oracle。', '',
        '移障沿用 Base 协议：选择最接近起终点身体位置线段的非地板 OBB，朝线段移动 0.30 m，只移动该 box。'
        '全部 stress 条件保留，包括 assembly_invalid、endpoint_invalid 和 unknown。frozen feasible 分母只使用推理前已冻结的物理见证，unknown 不表示已证明不可行。', '',
        '成功要求完整 URDF 碰撞、接触误差≤1 mm、法向误差≤1°、姿态/工作空间/端点/图域/形状和 twist 约束，'
        '以及成功候选的密集边检查。numeric success 另要求≤2e-6 m/rad。最小 clearance 仅统计 128 个节点，已扣除 margin；不是密集边最小距离。', '',
        '各训练规模使用相同优化更新数，因此小样本规模具有更多重复训练；种子均值/标准差与地图采样 CI 分别报告。', '',
        '## 全部 stress 条件的主成功率', '',
        '| 变体 | 训练专家数 | 种子数 | Original raw | Original 投影 | Shift raw | Shift 投影 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    groups = ([{'variant': 'base', 'train_size': 0, 'seeds': [], 'cells': report['models']['base']['primary']}]
              if 'base' in report['models'] else [])
    groups += report['seed_groups']
    for group in groups:
        values = []
        for scenario, projection in (('original', 'off'), ('original', 'on'), ('obstacle_shift', 'off'), ('obstacle_shift', 'on')):
            cell = _overall(group['cells'], scenario=scenario, projection=projection)
            if cell is None:
                values.append('—'); continue
            metric = cell['metrics']['candidate_success']; text = _percent(metric['mean'])
            if cell.get('candidate_success_seed_std') is not None:
                text += f" ± {100*cell['candidate_success_seed_std']:.2f} pp"
            text += f" [95% CI {_percent(metric['ci95'][0])}, {_percent(metric['ci95'][1])}]"
            values.append(text)
        lines.append(f"| {group['variant']} | {group['train_size']} | {len(group['seeds']) or 1} | " + ' | '.join(values) + ' |')
    lines += ['', '每个种子、frozen feasible 分母，以及 family×plane/cylinder×easy/detour 的完整标量诊断见 report.json。', '',
        '## 配对比较与负控', '',
        '配对 projection 差值为 on−off；shift 差值为 shifted−original。后者的 feasible 配对比较只用两个场景均有见证的条件，'
        '避免更换分母制造提升。JSON 中还给出同种子 geometry adapter−Base，以及 8k geometry−两类负控的逐条件差值和地图聚类 CI。', '',
        'no_contact_loss 关闭 contact/normal/smooth 辅助项，保留轨迹去噪和实际配置中的 collision 项；'
        'no_geometry_tokens 保留同样 adapter 结构与几何相关损失，但其注意力输入只有起终点 token。因而两者检验不同假设。', '',
        '## K=8 与经验多样性', '',
        'Any@8 仅按固定子集中的八个候选取是否至少一个成功。多样性仅计算成功候选对的同采样进度身体位置均方位移（m²）、RMS（m）和 UV 均方位移。'
        '每条条件先平均候选对，再对条件平均；不足两个成功候选时 pair diversity 为缺失，不以零代替。', '',
        'UV 绕行方向是相对请求起终点弦线的有符号偏移，阈值 0.05 m；它是经验几何代理，不能严格声称同伦类或多模态拓扑覆盖。'
        '圆柱 u 使用弧长，psi 的无意义旋转不计入位置多样性。', '',
        '## 范围与限制', '',
        '这是同十种场景族中的未见地图泛化，不能称为新的语义 OOD。源地图经过容纳完整接触装配和任务的拒绝采样，接受密度/角度与重试记录随 benchmark manifest 冻结。'
        '条件级见证是有限步长的完整 URDF 物理审核；有界求解器未找到见证不构成不可行证明。', '',
        '数据是连续几何接触路径，未测试 approach/retreat、接触力、摩擦、真实工具形状或动力学可执行性。'
        '候选 shape/jerk 是几何节点指标，不能解释成给定控制频率下的真实加速度或 jerk。', '']
    if report['shortfalls']:
        lines += ['## 未完成项', '', f"尚有 {len(report['shortfalls'])} 条完整性/完成状态问题，详见 preview JSON；本文件不声明完成。", '']
    training_lines = ['## 有效训练数据量', '',
        '| 专家轨迹数 | 不同源地图数 | 不同任务条件数 |', '|---:|---:|---:|']
    counts = {}
    for model in report['models'].values():
        value = model['training'].get('effective_training_counts')
        if value:
            counts[value['expert_paths']] = value
    for size, value in sorted(counts.items()):
        training_lines.append(f"| {size} | {value['source_maps']} | {value['distinct_task_conditions']} |")
    training_lines += ['', '每张训练地图含 20 条专家轨迹与 14 个不同任务条件；绕行条件的两个专家属于同一个条件。'
        '上述数量由冻结 prepared 身份数组与实际 training subset 交叉核对；反向增强与重复更新不增加独立地图/条件数量。', '']
    lines[lines.index('## 全部 stress 条件的主成功率'):lines.index('## 全部 stress 条件的主成功率')] = training_lines
    diagnostics = ['## 原始输出的接触与碰撞诊断', '',
        '| 变体 / 专家数 | Original 接触最大误差均值(mm) | Shift 接触最大误差均值(mm) | Original 节点碰撞自由率 | Shift 节点碰撞自由率 |',
        '|---|---:|---:|---:|---:|']
    for group in groups:
        cells = [_overall(group['cells'], scenario=scenario, projection='off') for scenario in SCENARIOS]
        errors = [cell['metrics']['contact_error_max_m']['mean'] if cell else None for cell in cells]
        rates = [cell['metrics']['physical_collision_free']['mean'] if cell else None for cell in cells]
        diagnostics.append(f"| {group['variant']} / {group['train_size']} | " + ' | '.join(
            [f'{1000*value:.3f}' if value is not None else '—' for value in errors] + [_percent(value) for value in rates]) + ' |')
    diagnostics += ['', '接触误差列先对每条轨迹取最大误差，再对轨迹和种子平均；不等同于所有节点的总体平均。'
        '物理碰撞列只要求 128 节点 URDF clearance>0，完整成功率还要求接触、形状及密集边审核。', '',
        '## 8k 负控的配对成功率差', '',
        '| 控制变体 | Original raw Δ(pp) | Original 投影 Δ(pp) | Shift raw Δ(pp) | Shift 投影 Δ(pp) |',
        '|---|---:|---:|---:|---:|']
    for comparison in report.get('paired_adapter_control_groups', []):
        if comparison['train_size'] != 8000:
            continue
        values = []
        for scenario, projection in (('original', 'off'), ('original', 'on'), ('obstacle_shift', 'off'), ('obstacle_shift', 'on')):
            metric = _overall(comparison['cells'], scenario=scenario, projection=projection)['metrics']['candidate_success']
            values.append('—' if metric['mean'] is None else f"{100*metric['mean']:+.2f} [{100*metric['ci95'][0]:+.2f}, {100*metric['ci95'][1]:+.2f}]")
        diagnostics.append(f"| {comparison['control_variant']} | " + ' | '.join(values) + ' |')
    diagnostics += ['', 'Δ 为 geometry adapter−对应控制；使用相同条件、候选 latent 与模型种子，区间以源地图聚类。', '']
    position = lines.index('## 配对比较与负控'); lines[position:position] = diagnostics
    multi_lines = ['## 固定子集 Any@8', '', '| 变体 / 专家数 | Original raw | Original 投影 | Shift raw | Shift 投影 |',
                   '|---|---:|---:|---:|---:|']
    multi_groups = ([{'variant': 'base', 'train_size': 0, 'cells': report['models']['base']['multimodal']['any_at_8']}]
                    if 'base' in report['models'] else [])
    multi_groups += [{'variant': group['variant'], 'train_size': group['train_size'], 'cells': group['multimodal_any_at_8']}
                     for group in report['seed_groups'] if 'multimodal_any_at_8' in group]
    for group in multi_groups:
        values = [_percent(_overall(group['cells'], scenario=scenario, projection=projection)['metrics']['any_at_8']['mean'])
            for scenario, projection in (('original', 'off'), ('original', 'on'), ('obstacle_shift', 'off'), ('obstacle_shift', 'on'))]
        multi_lines.append(f"| {group['variant']} / {group['train_size']} | " + ' | '.join(values) + ' |')
    multi_lines += ['', '固定子集中的成功候选对数量、位置/UV多样性和经验正负方向覆盖率及其地图 CI 见 JSON，不采用专家轨迹进行模式判别。', '']
    position = lines.index('## 范围与限制'); lines[position:position] = multi_lines
    if figures:
        lines += ['## 科学图', '']
        for name in figures:
            lines += [f'![{name}]({name})', '']
    return '\n'.join(lines)


def _figures(report: dict, output: Path, prefix: str) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
    names = []
    geometry = sorted((row for row in report['seed_groups'] if row['variant'] == 'geometry'), key=lambda row:row['train_size'])
    if not geometry:
        return names
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for axis, scenario in zip(axes, SCENARIOS):
        for projection, color in (('off', '#2864b4'), ('on', '#d55e00')):
            sizes, means, low, high = [], [], [], []
            for group in geometry:
                cell = _overall(group['cells'], scenario=scenario, projection=projection)
                metric = cell['metrics']['candidate_success']
                if metric['mean'] is None:
                    continue
                sizes.append(group['train_size']); means.append(metric['mean']*100)
                low.append(metric['ci95'][0]*100); high.append(metric['ci95'][1]*100)
            axis.plot(sizes, means, 'o-', color=color, label='Adapter raw' if projection == 'off' else 'Adapter + projection')
            axis.fill_between(sizes, low, high, color=color, alpha=.15)
            if 'base' in report['models']:
                base = _overall(report['models']['base']['primary'], scenario=scenario, projection=projection)
                if base['metrics']['candidate_success']['mean'] is not None:
                    axis.axhline(100*base['metrics']['candidate_success']['mean'], ls='--', color=color, lw=1, alpha=.65,
                                   label='Base raw' if projection == 'off' else 'Base + projection')
        axis.set_xscale('log', base=2); axis.set_xlabel('Training expert trajectories')
        axis.set_xticks([row['train_size'] for row in geometry], [str(row['train_size']) for row in geometry])
        axis.set_title('Original' if scenario == 'original' else 'Obstacle shift: 0.30 m')
        axis.grid(alpha=.2); axis.set_ylim(-2, 102)
    axes[0].set_ylabel('Candidate-0 success (%)'); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Mean across model seeds; 95% source-map cluster CI' + (' — INCOMPLETE' if prefix else ''), fontsize=10)
    fig.tight_layout(); name = prefix+'learning_curve.png'; fig.savefig(output/name, dpi=220); plt.close(fig); names.append(name)
    controls = [row for row in report['seed_groups'] if row['train_size'] == 8000]
    if controls:
        if 'base' in report['models']:
            controls.append({'variant': 'base', 'cells': report['models']['base']['primary']})
        controls.sort(key=lambda row:row['variant'])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        labels = [row['variant'].replace('_', '\n') for row in controls]
        for axis, scenario in zip(axes, SCENARIOS):
            for offset, projection, color in ((-.18, 'off', '#2864b4'), (.18, 'on', '#d55e00')):
                cells = [_overall(row['cells'], scenario=scenario, projection=projection) for row in controls]
                means = np.array([cell['metrics']['candidate_success']['mean'] or 0 for cell in cells])*100
                bounds = np.array([cell['metrics']['candidate_success']['ci95'] if cell['metrics']['candidate_success']['ci95'][0] is not None else [0., 0.] for cell in cells]).T*100
                axis.bar(np.arange(len(controls))+offset, means, .34, color=color, label=projection,
                         yerr=np.maximum(np.stack((means-bounds[0], bounds[1]-means)), 0), capsize=3)
            axis.set_xticks(np.arange(len(controls)), labels); axis.set_title(scenario); axis.set_ylim(-2, 102)
        axes[0].set_ylabel('Candidate-0 success (%)'); axes[1].legend(frameon=False, title='Projection')
        fig.suptitle('8k negative controls; 95% source-map cluster CI' + (' — INCOMPLETE' if prefix else ''))
        fig.tight_layout(); name = prefix+'ablation_8000.png'; fig.savefig(output/name, dpi=220); plt.close(fig); names.append(name)
    largest = geometry[-1]
    families = sorted({row['family'] for row in largest['cells'] if row['scope'] == 'family_kind_difficulty'})
    if families:
        fig, axes = plt.subplots(2, 2, figsize=(12, max(5., .5*len(families)+2.)), constrained_layout=True)
        categories = [('plane', 'easy'), ('plane', 'detour'), ('cylinder', 'easy'), ('cylinder', 'detour')]
        for axis, (scenario, projection) in zip(axes.ravel(), [(s, p) for s in SCENARIOS for p in PROJECTIONS]):
            matrix = np.full((len(families), 4), np.nan)
            for i, family in enumerate(families):
                for j, (kind, difficulty) in enumerate(categories):
                    cell = next((row for row in largest['cells'] if row['scope'] == 'family_kind_difficulty' and
                        row['family'] == family and row['kind'] == kind and row['difficulty'] == difficulty and
                        row['scenario'] == scenario and row['projection'] == projection and row['selection'] == 'all_stress'), None)
                    if cell and cell['metrics']['candidate_success']['mean'] is not None:
                        matrix[i, j] = cell['metrics']['candidate_success']['mean']*100
            plotted = axis.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=100, cmap='viridis', aspect='auto')
            axis.set_yticks(np.arange(len(families)), [family.replace('_', ' ') for family in families], fontsize=8)
            axis.set_xticks(np.arange(4), [f'{kind} / {difficulty}' for kind, difficulty in categories], rotation=20, ha='right', fontsize=8)
            axis.set_title(f"{scenario.replace('_', ' ')} / {'raw' if projection == 'off' else 'projection'}")
            for i in range(len(families)):
                for j in range(4):
                    axis.text(j, i, f'{matrix[i,j]:.1f}' if np.isfinite(matrix[i,j]) else 'n/a', ha='center', va='center',
                              fontsize=8, color='white' if np.isfinite(matrix[i,j]) and matrix[i,j] < 55 else 'black')
        fig.colorbar(plotted, ax=axes.ravel().tolist(), label='Candidate-0 success (%)', shrink=.8)
        fig.suptitle(f"Geometry adapter ({largest['train_size']} experts): family / kind / difficulty" + (' — INCOMPLETE' if prefix else ''))
        name = prefix+'family_kind_difficulty.png'; fig.savefig(output/name, dpi=220); plt.close(fig); names.append(name)
    return names


def build_report(experiment: str | Path, benchmark: str | Path, output: str | Path, *,
                 allow_partial: bool = False, bootstrap_replicates: int = 1000,
                 bootstrap_seed: int = 20261011, make_figures: bool = True) -> dict:
    """Emit complete report only after every planned model/case/candidate passes."""
    experiment, benchmark, output = Path(experiment).resolve(), Path(benchmark).resolve(), Path(output).resolve()
    if bootstrap_replicates < 1:
        raise ValueError('bootstrap_replicates must be positive')
    protocol = _read(experiment/'protocol.json')
    if protocol['protocol_sha256'] != _definition_hash(protocol, timestamp='created_utc'):
        raise IntegrityError('approved experiment protocol hash mismatch')
    manifest, inputs = _load_inputs(benchmark, protocol)
    matrix = protocol['variants']; planned = [{'id': 'base', 'variant': 'base', 'train_size': 0, 'seed': None}, *matrix]
    if len({row['id'] for row in planned}) != len(planned):
        raise IntegrityError('experiment matrix has duplicate model IDs')
    if protocol.get('schema_version') == 'contact_research_protocol_v001':
        sizes = (500, 1000, 2000, 4000, 8000); seeds = protocol['model_seeds']
        expected = {('geometry', size, seed) for size in sizes for seed in seeds}
        expected.update((variant, 8000, seed) for variant in ('no_contact_loss', 'no_geometry_tokens') for seed in seeds)
        if (len(seeds) != 3 or len(set(seeds)) != 3 or len(matrix) != 21 or
                {(row['variant'], row['train_size'], row['seed']) for row in matrix} != expected or
                protocol['new_test_conditions'] != 10000 or protocol['new_source_maps'] != 1000 or
                protocol['multimodal_subset_conditions'] != 1000):
            raise IntegrityError('official report requires the complete 22-model / 10000-task / 1000-task K8 matrix')
    problems = []; models = {}; summaries = {}; condition_cache = {}
    shared_contract = None
    for row in planned:
        value = _read_model(experiment, benchmark, row, protocol, inputs, _sha(benchmark/'manifest.json'), problems)
        if value is None:
            continue
        shared = {key: value['contract'].get(key) for key in ('base_sha256', 'benchmark_manifest_sha256',
            'primary_candidates', 'multimodal_maps_per_family', 'multimodal_candidates', 'ddim_steps', 'latent_seed',
            'inference_version', 'projection_version', 'metrics_version', 'reference_backend', 'inference_sources', 'evaluation_sources')}
        if shared_contract is not None and shared != shared_contract:
            raise IntegrityError('models use different inference, metric, projection or reference-backend contracts')
        shared_contract = shared
        multimodal = _diversity(value, inputs, benchmark, bootstrap_replicates, bootstrap_seed, condition_cache)
        any_values = multimodal.pop('_any_values'); diversity_values = multimodal.pop('_diversity_values')
        summaries[row['id']] = {'training': value['training'], 'contract': value['contract'], 'complete': value['complete'],
            'primary': _summary_cells(value['primary'], inputs, bootstrap_replicates, bootstrap_seed),
            'paired_effects': _paired_effects(value['primary'], inputs, bootstrap_replicates, bootstrap_seed),
            'multimodal': multimodal,
            'evaluation_errors': value['evaluation_errors'], 'failure_counts': value['failure_counts'],
            'artifact_sha256': value['artifact_sha256']}
        # Only primary scalar arrays are retained across models for paired controls.
        models[row['id']] = {'primary': value['primary'], 'multi_rows': value['multi_rows'],
                             '_any_values': any_values, '_diversity_values': diversity_values}
        print(f'REPORT {row["id"]} complete={value["complete"]}', flush=True)
    selected_by_size = {}
    for name, value in summaries.items():
        training = value['training']
        if 'selected_source_maps' not in training:
            continue
        size = training['train_size']; selected = set(training['selected_source_maps'])
        if size in selected_by_size and selected != selected_by_size[size]:
            raise IntegrityError('same expert budget uses different data across model seeds/controls')
        selected_by_size[size] = selected
    ordered = sorted(selected_by_size)
    if any(not selected_by_size[left] <= selected_by_size[right] for left, right in zip(ordered[:-1], ordered[1:])):
        raise IntegrityError('training expert budgets do not form nested whole-source-map subsets')
    if problems and not allow_partial:
        raise IncompleteExperiment(f'no completed report emitted: {len(problems)} shortfalls; use --allow-partial for an explicit incomplete preview')
    report = {'schema_version': REPORT_VERSION, 'status': 'incomplete_preview' if problems else 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
        'benchmark': manifest, 'benchmark_manifest_sha256': _sha(benchmark/'manifest.json'),
        'protocol_file_sha256': _sha(experiment/'protocol.json'), 'report_source_sha256': _sha(Path(__file__)),
        'bootstrap_replicates': bootstrap_replicates, 'bootstrap_seed': bootstrap_seed,
        'independent_units': 'source maps; model seeds are separate stochastic training replicates',
        'primary_candidates': 1, 'shortfalls': problems, 'models': summaries,
        'seed_groups': _seed_groups(models, matrix, inputs, bootstrap_replicates, bootstrap_seed),
        'paired_adapter_controls': _ablation_pairs(models, matrix, inputs, bootstrap_replicates, bootstrap_seed),
        'paired_adapter_control_groups': _ablation_group_pairs(models, matrix, inputs, bootstrap_replicates, bootstrap_seed),
        'denominator_policy': 'all frozen stress inputs; feasible per-scenario subset; paired shift on the feasibility intersection',
        'diversity_policy': 'successful candidate pairs only, one mean per condition; UV sign is empirical, not homotopy',
        'oracle_access': False}
    output.mkdir(parents=True, exist_ok=True)
    prefix = 'preview_' if problems else ''
    figures = _figures(report, output, prefix) if make_figures else []
    report['figures'] = {name: _sha(output/name) for name in figures}
    json_name = 'report_preview.json' if problems else 'report.json'
    markdown_name = 'research_report_zh_preview.md' if problems else 'research_report_zh.md'
    _write(output/json_name, report)
    (output/markdown_name).write_text(_markdown(report, figures), encoding='utf8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--bootstrap-replicates', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261011)
    args = parser.parse_args()
    report = build_report(args.experiment, args.benchmark, args.output, allow_partial=args.allow_partial,
        bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed)
    print(_json({'status': report['status'], 'models': len(report['models']), 'shortfalls': len(report['shortfalls'])}))


if __name__ == '__main__':
    main()
