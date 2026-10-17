"""Audited three-decoder supplement; previews cannot become completed research.

Read saved predictions and pre-inference physical-input identities only. Never
run a decoder, fit paths, sample, query COAL, or open isolated oracle artifacts.
The frozen parent report supplies its already-audited off/on K8 diversity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import evaluate_contact_smooth_projection as evaluation
from . import report_contact_experiment as parent
from .contact_smooth_projection import SMOOTH_PROJECTION_VERSION, smooth_contact_decoder_parameters

REPORT_VERSION = 'contact_smooth_three_decoder_report_v001'
PROJECTIONS = ('off', 'on', 'on_smooth')
EXTRA_FIELDS = ('projection_succeeded', 'projection_failed', 'projection_fit_success',
                'projection_geometry_valid', 'solver_projected_gradient_max', 'solver_sweeps',
                'fit_mse_normalized', 'smoothing_body_position_correction_max_m',
                'smoothing_body_rotation_correction_max_rad', 'minimum_local_chord_arc_ratio',
                'maximum_turn_angle_deg', 'backtracking_fraction', 'maximum_position_step_m',
                'maximum_rotation_step_rad', 'rotation_total_variation_rad',
                'numeric_contact_gate', 'numeric_normal_gate', 'physical_collision_checked',
                'dense_validation_completed')
FIELDS = parent.FIELDS+EXTRA_FIELDS
INDEX = {name: index for index, name in enumerate(FIELDS)}
PAIRED_FIELDS = parent.PAIRED_FIELDS+('maximum_curvature_inv_m', 'shape_quality_gate')
ANY_FIELDS = ('any_at_8', 'numeric_any_at_8', 'physical_collision_any_at_8')
DIVERSITY_FIELDS = ('successful_candidate_count', 'position_pair_msd_m2',
                    'position_pair_rms_m', 'uv_pair_msd_m2', 'both_empirical_direction_signs')
DECODER_AFTER_FIELDS = frozenset(('solver_projected_gradient_max', 'solver_sweeps', 'fit_mse_normalized',
                                'smoothing_body_position_correction_max_m',
                                'smoothing_body_rotation_correction_max_rad'))
IntegrityError = parent.IntegrityError
IncompleteExperiment = parent.IncompleteExperiment


def _verified(path: Path, field: str = 'contract_sha256') -> dict:
    try:
        return evaluation._verified_json(path, field)
    except evaluation.IntegrityError as error:
        raise IntegrityError(str(error)) from error


def _check(path: Path, digest: str) -> None:
    parent._check_hash(path, digest)


def _safe(root: Path, relative: str) -> Path:
    try:
        return evaluation._safe_input_path(root, relative)
    except evaluation.IntegrityError as error:
        raise IntegrityError(str(error)) from error


def _source_files(contract: dict) -> None:
    for filename, digest in (contract.get('source_files', {}) | contract.get('input_files', {})).items():
        if any('oracle' in part.lower() for part in Path(filename).parts):
            raise IntegrityError('source/input contract references isolated oracle data')
        _check(Path(filename), digest)


def _context(experiment: Path, benchmark: Path, supplement: Path, problems: list,
             *, allow_partial: bool) -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    protocol = _verified(experiment/'protocol.json', 'protocol_sha256')
    official = protocol.get('schema_version') == 'contact_research_protocol_v001'
    if official:
        try:
            evaluation._validate_protocol(protocol)
        except evaluation.IntegrityError as error:
            raise IntegrityError(str(error)) from error
    else:
        problems.append(dict(stage='matrix', reason='preview fixture is not the approved 22-model protocol'))
    manifest, inputs = parent._load_inputs(benchmark, protocol)
    if inputs['input_row'].dtype.kind not in 'iu' or inputs['feasible_mask'].dtype.kind != 'b':
        raise IntegrityError('frozen input index/feasibility dtype is malformed')
    original = {str(value): index for index, value in enumerate(inputs['condition_id']) if inputs['scenario'][index] == 'original'}
    shifted = {str(value): index for index, value in enumerate(inputs['condition_id']) if inputs['scenario'][index] == 'obstacle_shift'}
    if set(original) != set(shifted):
        raise IntegrityError('original/shift inputs do not retain identical condition identities')
    for identifier, a in original.items():
        b = shifted[identifier]
        if any(inputs[key][a] != inputs[key][b] for key in ('map_id', 'family', 'kind', 'difficulty', 'input_shard', 'input_row')):
            raise IntegrityError('paired original/shift input identities are inconsistent')
    if any(str(case) != f'{identifier}/{scenario}' for case, identifier, scenario in
           zip(inputs['case_id'], inputs['condition_id'], inputs['scenario'])):
        raise IntegrityError('frozen case identity is inconsistent with condition/scenario')
    stems = [str(value).rsplit('/', 1)[-1] for value in sorted(set(inputs['map_id']))]
    if len(set(stems)) != len(stems):
        raise IntegrityError('source-map shard names collide')
    if official:
        try:
            evaluation._validate_inputs(manifest, inputs, protocol)
        except evaluation.IntegrityError as error:
            raise IntegrityError(str(error)) from error
    parent_sources = _verified(experiment/'pipeline_source_contract.json')
    _source_files(parent_sources)
    if (parent_sources.get('protocol_sha256') != protocol['protocol_sha256'] or
            Path(parent_sources['experiment']).resolve() != experiment or
            Path(parent_sources['benchmark']).resolve() != benchmark):
        raise IntegrityError('parent source contract belongs to another experiment/benchmark')
    code = Path(__file__).parent
    required_parent = (*evaluation.INFERENCE_SOURCES, *evaluation.EVALUATION_SOURCES,
                       'report_contact_experiment.py')
    for name in required_parent:
        if parent_sources['source_files'].get(str((code/name).resolve())) != parent._sha(code/name):
            raise IntegrityError(f'parent source contract does not bind {name}')
    if (parent_sources['input_files'].get(str((experiment/'protocol.json').resolve())) != parent._sha(experiment/'protocol.json') or
            parent_sources['input_files'].get(str(Path(protocol['base_checkpoint']).resolve())) != protocol['base_sha256']):
        raise IntegrityError('parent source contract does not bind protocol/base checkpoint')
    state_path = experiment/'pipeline_status.json'
    state = parent._read(state_path) if state_path.exists() else {}
    report_path = Path(state.get('report', experiment/'reports/report.json')).resolve()
    if not report_path.is_file():
        raise IncompleteExperiment('completed parent report is absent')
    prior = parent._read(report_path)
    expected_ids = ['base', *[row['id'] for row in protocol['variants']]]
    if len(set(expected_ids)) != len(expected_ids):
        raise IntegrityError('parent protocol contains repeated model identities')
    if (prior.get('status') != 'complete' or prior.get('shortfalls') != [] or
            set(prior.get('models', {})) != set(expected_ids)):
        raise IncompleteExperiment('parent report must be complete before supplementary reporting')
    if (prior.get('protocol') != protocol or prior.get('benchmark') != manifest or
            prior.get('protocol_file_sha256') != parent._sha(experiment/'protocol.json') or
            prior.get('benchmark_manifest_sha256') != parent._sha(benchmark/'manifest.json') or
            prior.get('report_source_sha256') != parent._sha(code/'report_contact_experiment.py') or
            not (report_path.parent/'research_report_zh.md').is_file() or not prior.get('figures')):
        raise IntegrityError('parent report protocol/manifest/source/figure provenance mismatch')
    if official and (prior.get('bootstrap_replicates') != 1000 or prior.get('bootstrap_seed') != 20261011):
        raise IntegrityError('completed parent report changed the fixed bootstrap protocol')
    for name, digest in prior['figures'].items():
        if Path(name).suffix.lower() != '.png':
            raise IntegrityError('parent scientific figure must be a PNG')
        _check(_safe(report_path.parent, name), digest)
    evidence = state.get('stage_evidence', {}).get('report', {})
    if official and (state.get('status') != 'complete' or evidence.get('report_sha256') != parent._sha(report_path) or
                     evidence.get('markdown_sha256') != parent._sha(report_path.parent/'research_report_zh.md') or
                     evidence.get('figures') != prior['figures']):
        raise IntegrityError('parent completed pipeline does not authenticate its report/markdown/figures')
    prior['_report_path'] = str(report_path)
    smooth = _verified(supplement/'protocol.json', 'protocol_sha256')
    parameters = smooth_contact_decoder_parameters()
    multi = parent._multimodal_maps(inputs, 10 if official else int(smooth.get('multimodal_maps_per_family', 1)))
    maps = sorted(set(map(str, inputs['map_id'])))
    expected_paths = sum(len(evaluation.requests_for_map(inputs, map_id, multi)[0]) for map_id in maps)
    expected_input_names = {'runtime_inputs.npz', 'index.jsonl', *map(str, inputs['input_shard'])}
    expected = dict(schema_version=evaluation.SUPPLEMENT_VERSION,
        parent_experiment=str(experiment), benchmark=str(benchmark),
        parent_protocol_sha256=protocol['protocol_sha256'],
        parent_protocol_file_sha256=parent._sha(experiment/'protocol.json'),
        parent_source_contract_sha256=parent_sources['contract_sha256'],
        parent_source_contract_file_sha256=parent._sha(experiment/'pipeline_source_contract.json'),
        benchmark_manifest_sha256=parent._sha(benchmark/'manifest.json'),
        benchmark_input_files_sha256={name: manifest['input_files_sha256'][name] for name in sorted(expected_input_names)},
        planned_models=expected_ids, condition_count=protocol['new_test_conditions'], source_map_count=len(maps),
        multimodal_source_maps=sorted(multi), multimodal_subset_conditions=protocol['multimodal_subset_conditions'],
        primary_candidates=1, multimodal_candidates=8, paths_per_model=expected_paths,
        score_rows_per_model=expected_paths, projection='on_smooth', decoder_version=SMOOTH_PROJECTION_VERSION,
        decoder_parameters=parameters, decoder_parameters_sha256=parent._hash(parameters),
        metrics_version=evaluation.METRICS_VERSION, output_schema=evaluation.OUTPUT_SCHEMA,
        output_schema_sha256=parent._hash(evaluation.OUTPUT_SCHEMA),
        input_policy='same_parent_raw_paths_and_whitelisted_frozen_original_shift_conditions',
        failure_policy='retain_every_request; decoder_errors_count_as_failed_candidates',
        parameter_policy='fixed_before_pilot; no_benchmark_or_pilot_tuning',
        interpolation=evaluation.INTRINSIC_INTERPOLATION, oracle_access=False, generation=False, training=False)
    if any(smooth.get(key) != value for key, value in expected.items()):
        raise IntegrityError('supplement fixed protocol/input/decoder/schema contract mismatch')
    sources = _verified(supplement/'source_contract.json')
    _source_files(sources)
    expected_sources = dict(schema_version=evaluation.SOURCE_CONTRACT_VERSION,
        protocol_sha256=smooth['protocol_sha256'], parent_source_contract_sha256=parent_sources['contract_sha256'],
        decoder_parameters_sha256=smooth['decoder_parameters_sha256'], metrics_version=evaluation.METRICS_VERSION,
        output_schema_sha256=smooth['output_schema_sha256'])
    if any(sources.get(key) != value for key, value in expected_sources.items()):
        raise IntegrityError('supplement source contract protocol/schema mismatch')
    for name in evaluation.SUPPLEMENT_SOURCES:
        if sources.get('source_files', {}).get(str((code/name).resolve())) != parent._sha(code/name):
            raise IntegrityError(f'supplement source contract does not bind {name}')
    dependencies = {name: parent._sha(code/name) for name in (*evaluation.INFERENCE_SOURCES, *evaluation.EVALUATION_SOURCES)}
    if sources.get('frozen_dependency_sources') != dependencies:
        raise IntegrityError('supplement frozen dependency sources differ from v001')
    driver_path = supplement/'pipeline_source_contract.json'
    driver = None
    if driver_path.exists():
        driver = _verified(driver_path)
        _source_files(driver)
        expected_driver = dict(schema_version='contact_smooth_pipeline_sources_v001',
            experiment=str(experiment), benchmark=str(benchmark), supplement=str(supplement),
            parent_source_contract_sha256=parent_sources['contract_sha256'], decoder_parameters=parameters,
            stage_policy='wait_for_strict_parent_complete; score_all22_same_raw; strict_supplement_report')
        if any(driver.get(key) != value for key, value in expected_driver.items()):
            raise IntegrityError('supplement driver contract changed its source/input/parameter policy')
        for name in ('contact_smooth_pipeline.py', 'contact_smooth_projection.py',
                     'evaluate_contact_smooth_projection.py', Path(__file__).name):
            if driver['source_files'].get(str((code/name).resolve())) != parent._sha(code/name):
                raise IntegrityError(f'supplement driver does not bind {name}')
        for key, digest in parent_sources['source_files'].items():
            if driver['source_files'].get(key) != digest:
                raise IntegrityError('supplement driver replaced a frozen parent source')
        for key, digest in (parent_sources['input_files'] | {str(experiment/'pipeline_source_contract.json'): parent._sha(experiment/'pipeline_source_contract.json')}).items():
            if driver['input_files'].get(key) != digest:
                raise IntegrityError('supplement driver replaced a frozen parent input')
    elif allow_partial:
        problems.append(dict(stage='driver', reason='supplement driver source contract absent; preview only'))
    else:
        raise IncompleteExperiment('strict supplementary report requires its driver source contract')
    smooth['_multimodal_maps'] = multi
    smooth['_driver_contract'] = driver
    return protocol, manifest, inputs, prior, smooth, sources, parent_sources


def _values(metrics: dict) -> np.ndarray:
    result = []
    after = metrics.get('projection_after', {})
    for name in FIELDS:
        # Decoder geometry is a separate diagnostic. It cannot fill a missing
        # independent metric after a COAL/evaluation exception.
        value = metrics.get(name, after.get(name)) if name in DECODER_AFTER_FIELDS else metrics.get(name)
        result.append(float(value) if isinstance(value, (int, float, bool)) and math.isfinite(float(value)) else np.nan)
    return np.array(result)


def _parent_values(experiment: Path, baseline: dict, inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extend the frozen parent's summaries using its saved scalar score fields."""
    primary = np.full((2, len(inputs['case_id']), len(FIELDS)), np.nan)
    multiple = np.full((2, len(baseline['multi_rows']), 8, len(FIELDS)), np.nan)
    cases = {str(value): index for index, value in enumerate(inputs['case_id'])}
    subset = {int(value): index for index, value in enumerate(baseline['multi_rows'])}
    for relative, digest in baseline['artifact_sha256'].items():
        if not relative.endswith('.jsonl.gz'):
            continue
        path = experiment/relative; _check(path, digest)
        for record in evaluation._read_gz_jsonl(path):
            index = cases[record['case_id']]; candidate = record['candidate_index']
            projection = ('off', 'on').index(record['projection']); value = _values(record['metrics'])
            if candidate == 0:
                primary[projection, index] = value
            if index in subset:
                multiple[projection, subset[index], candidate] = value
    if (not np.array_equal(primary[..., :len(parent.FIELDS)], baseline['primary'], equal_nan=True) or
            not np.array_equal(multiple[..., :len(parent.FIELDS)], baseline['multiple'], equal_nan=True)):
        raise IntegrityError('parent scalar score matrix changed while extending its saved diagnostics')
    return primary, multiple


def saved_pair_diversity(world: np.ndarray, intrinsic: np.ndarray, successful: np.ndarray) -> dict:
    """Use actual saved body XYZ/UV; absence of successful pairs is missing data."""
    selected = np.flatnonzero(successful)
    position, uv = [], []
    directions = Counter()
    for index in selected:
        directions[parent.empirical_direction(intrinsic[index, :, :2], intrinsic[index, 0], intrinsic[index, -1])] += 1
    for first, a in enumerate(selected):
        for b in selected[first+1:]:
            position.append(float(np.mean(np.sum((world[a, :, :3]-world[b, :, :3])**2, axis=-1))))
            uv.append(float(np.mean(np.sum((intrinsic[a, :, :2]-intrinsic[b, :, :2])**2, axis=-1))))
    return dict(successful_candidates=len(selected), successful_pairs=len(position),
        position_pair_msd_m2=float(np.mean(position)) if position else None,
        position_pair_rms_m=float(np.mean(np.sqrt(position))) if position else None,
        uv_pair_msd_m2=float(np.mean(uv)) if uv else None, direction_proxy=dict(directions),
        both_direction_signs=bool(directions['positive'] and directions['negative']))


def _read_smooth(experiment: Path, benchmark: Path, supplement: Path, row: dict, inputs: dict,
                 manifest: dict, smooth: dict, sources: dict, baseline: dict, problems: list) -> dict:
    model_id = row['id']; directory = supplement/'evaluation'/model_id
    n = len(inputs['case_id']); multi = smooth['_multimodal_maps']
    multi_rows = baseline['multi_rows']; lookup = {int(value): index for index, value in enumerate(multi_rows)}
    primary = np.full((n, len(FIELDS)), np.nan)
    multiple = np.full((len(multi_rows), 8, len(FIELDS)), np.nan)
    diversity = np.full((len(multi_rows), len(DIVERSITY_FIELDS)), np.nan)
    details, failures, errors, artifacts = [], Counter(), Counter(), {}
    path = directory/'contract.json'
    if not path.is_file():
        problems.append(dict(model_id=model_id, stage='contract', reason='supplement model contract absent'))
        args = argparse.Namespace(experiment=experiment)
        raw_contract = dict(model_id=model_id, parent_contract_sha256=baseline['contract']['contract_sha256'])
        try:
            for map_id in sorted(set(map(str, inputs['map_id']))):
                evaluation._raw_requests(args, raw_contract, inputs, map_id, multi)
        except evaluation.IntegrityError as error:
            raise IntegrityError(str(error)) from error
        problems.append(dict(model_id=model_id, stage='requests', missing_score_rows=smooth['score_rows_per_model'],
                             reason='supplement model contract and all its planned scores are absent'))
        return dict(primary=primary, multiple=multiple, diversity=diversity, details=details,
                    artifacts=artifacts, failures=[], errors=[], complete=False, contract=None, scoring_status=None)
    contract = _verified(path)
    parent_contract = baseline['contract']
    binding = dict(base_sha256=parent_contract['base_sha256'], adapter_sha256=parent_contract['adapter_sha256'])
    if model_id != 'base':
        model_dir = experiment/'models'/model_id
        binding.update(training_config_sha256=parent._sha(model_dir/'resolved_training_config.json'),
                       run_status_sha256=parent._sha(model_dir/'run_status.json'))
    expected = dict(schema_version=evaluation.SUPPLEMENT_VERSION, model_id=model_id,
        protocol_sha256=smooth['protocol_sha256'], source_contract_sha256=sources['contract_sha256'],
        parent_contract_sha256=parent_contract['contract_sha256'],
        parent_contract_file_sha256=parent._sha(experiment/'evaluation'/model_id/'contract.json'),
        checkpoint_binding=binding, benchmark_manifest_sha256=smooth['benchmark_manifest_sha256'],
        decoder_version=SMOOTH_PROJECTION_VERSION, decoder_parameters_sha256=smooth['decoder_parameters_sha256'],
        metrics_version=evaluation.METRICS_VERSION, output_schema_sha256=smooth['output_schema_sha256'],
        reference_backend=parent_contract.get('reference_backend'), score_rows_per_model=smooth['score_rows_per_model'])
    if any(contract.get(key) != value for key, value in expected.items()):
        raise IntegrityError('supplement model/checkpoint/backend contract mismatch')
    status_path = directory/'scoring_status.json'
    status = parent._read(status_path) if status_path.exists() else {}
    maps = sorted(set(map(str, inputs['map_id'])))
    status_valid = not (status.get('status') != 'complete' or status.get('contract_sha256') != contract['contract_sha256'] or
            status.get('complete_maps') != len(maps) or status.get('total_maps') != len(maps) or
            status.get('planned_total_maps') != len(maps) or status.get('score_rows') != smooth['score_rows_per_model'] or
            status.get('expected_score_rows') != smooth['score_rows_per_model'] or
            status.get('planned_score_rows') != smooth['score_rows_per_model'] or status.get('pilot_limit_maps') != 0)
    if not status_valid:
        problems.append(dict(model_id=model_id, stage='status', reason='supplement scoring matrix/status is incomplete'))
    artifacts[str(path.relative_to(supplement))] = parent._sha(path)
    if status_path.is_file():
        artifacts[str(status_path.relative_to(supplement))] = parent._sha(status_path)
    args = argparse.Namespace(experiment=experiment, benchmark=benchmark, output=supplement)
    seen = 0
    for map_id in maps:
        stem = map_id.rsplit('/', 1)[-1]
        score = directory/'scores'/f'{stem}.jsonl.gz'
        projected = directory/'projected'/f'{stem}.npz'
        paths = (score, score.with_name(f'{stem}.json'), projected, projected.with_suffix('.json'))
        try:
            raw, requested, candidates, raw_meta = evaluation._raw_requests(args, contract, inputs, map_id, multi)
        except evaluation.IntegrityError as error:
            raise IntegrityError(str(error)) from error
        if not all(path.is_file() for path in paths):
            problems.append(dict(model_id=model_id, map_id=map_id, stage='shard', reason='supplement score/projected shard absent'))
            continue
        try:
            _, input_hashes = evaluation._prepare_conditions(args, inputs, requested, manifest, map_id)
            expected_meta = dict(schema_version=evaluation.SUPPLEMENT_VERSION, contract_sha256=contract['contract_sha256'],
                parent_contract_sha256=contract['parent_contract_sha256'], raw_sha256=raw_meta['raw_sha256'],
                raw_meta_sha256=raw_meta['raw_meta_sha256'], map_id=map_id, input_files_sha256=input_hashes,
                request_sha256=parent._hash(dict(input_rows=requested.tolist(), candidate_indices=candidates.tolist())),
                input_sha256=[str(inputs['input_sha256'][index]) for index in requested],
                protocol_sha256=contract['protocol_sha256'], source_contract_sha256=contract['source_contract_sha256'],
                decoder_version=contract['decoder_version'], decoder_parameters_sha256=contract['decoder_parameters_sha256'],
                metrics_version=evaluation.METRICS_VERSION, output_schema_sha256=contract['output_schema_sha256'])
            evaluation._resume(score, projected, expected_meta, inputs, requested, candidates, contract, map_id, multi)
        except evaluation.IntegrityError as error:
            raise IntegrityError(str(error)) from error
        with np.load(projected, allow_pickle=False) as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        flags = saved['projection_succeeded']
        if (saved['input_rows'].dtype.kind not in 'iu' or saved['candidate_indices'].dtype.kind not in 'iu' or
                not np.array_equal(saved['world_pose9'][flags][:, [0, -1]], raw[flags][:, [0, -1]].astype(np.float64))):
            raise IntegrityError('saved projected endpoints/request dtypes differ from the same parent predictions')
        meta = parent._read(paths[1])
        counts = dict(projection_failed_rows=int((~flags).sum()), fit_success_rows=int(saved['fit_success'].sum()),
                      geometry_valid_rows=int(saved['geometry_valid'].sum()))
        if any(meta.get(key) != value for key, value in counts.items()):
            raise IntegrityError('supplement metadata decoder/fit/geometry counts are inconsistent')
        records = list(evaluation._read_gz_jsonl(score))
        for offset, (record, index, candidate) in enumerate(zip(records, requested, candidates)):
            metrics = record['metrics']
            if flags[offset]:
                after = metrics.get('projection_after', {})
                if after.get('solver_converged') is not bool(saved['fit_success'][offset]):
                    raise IntegrityError('saved fit flag differs from the recorded solver convergence')
                if metrics.get('projection_failed') is not False:
                    raise IntegrityError('successful decoder record has an inconsistent failure flag')
            value = _values(metrics)
            if candidate == 0:
                primary[index] = value
            if index in lookup:
                multiple[lookup[int(index)], candidate] = value
            if metrics.get('evaluation_error'):
                errors[(record['scenario'], 'on_smooth')] += 1
            failures.update((record['scenario'], 'on_smooth', reason) for reason in metrics.get('failures', []))
        if map_id in multi:
            for index in np.unique(requested):
                selected = np.flatnonzero(requested == index)
                successful = np.array([records[offset]['metrics']['candidate_success'] for offset in selected])
                data = saved_pair_diversity(saved['world_pose9'][selected], saved['intrinsic_states'][selected], successful)
                label = {key: str(inputs[key][index]) for key in ('condition_id', 'map_id', 'scenario', 'family', 'kind', 'difficulty')}
                details.append(label | dict(projection='on_smooth') | data)
                diversity[lookup[int(index)]] = [data['successful_candidates'],
                    data['position_pair_msd_m2'] if data['position_pair_msd_m2'] is not None else np.nan,
                    data['position_pair_rms_m'] if data['position_pair_rms_m'] is not None else np.nan,
                    data['uv_pair_msd_m2'] if data['uv_pair_msd_m2'] is not None else np.nan,
                    float(data['both_direction_signs']) if data['successful_candidates'] >= 2 else np.nan]
        seen += len(records)
        for path in paths:
            artifacts[str(path.relative_to(supplement))] = parent._sha(path)
    if seen != smooth['score_rows_per_model']:
        problems.append(dict(model_id=model_id, stage='requests', missing_score_rows=smooth['score_rows_per_model']-seen,
                             reason='not every planned candidate has a committed supplemental score'))
    return dict(primary=primary, multiple=multiple, diversity=diversity, details=details, artifacts=artifacts,
        failures=[dict(scenario=s, projection=p, failure=f, count=c) for (s, p, f), c in sorted(failures.items())],
        errors=[dict(scenario=s, projection=p, count=c) for (s, p), c in sorted(errors.items())],
        complete=seen == smooth['score_rows_per_model'] and status_valid,
        contract=contract, scoring_status=status)


def _scopes(inputs: dict) -> list:
    scopes = parent._scopes(inputs)
    for key in ('family', 'kind', 'difficulty'):
        scopes.extend((dict(scope=key, **{key: str(value)}), inputs[key] == value) for value in sorted(set(inputs[key])))
    return scopes


def _cells(values: np.ndarray, inputs: dict, replicates: int, seed: int, *, names: tuple = FIELDS,
           projections: tuple = PROJECTIONS) -> list[dict]:
    if values.shape != (len(projections), len(inputs['case_id']), len(names)):
        raise ValueError('projection labels do not match the summary matrix')
    cells = []
    for scope, mask in _scopes(inputs):
        for scenario in parent.SCENARIOS:
            for selection in ('all_stress', 'frozen_feasible'):
                selected = mask & (inputs['scenario'] == scenario)
                if selection == 'frozen_feasible':
                    selected &= inputs['feasible_mask']
                for index, projection in enumerate(projections):
                    label = scope | dict(scenario=scenario, selection=selection, projection=projection)
                    cells.append(label | parent.cluster_statistics(values[index, selected], inputs['map_id'][selected],
                        names=names, replicates=replicates, seed=parent._seed(seed, parent._json(label)), strata=inputs['family'][selected]))
    return cells


def _paired_deltas(values: np.ndarray, inputs: dict):
    """Form each within-seed contrast before handling any missing diagnostics."""
    fields = [INDEX[name] for name in PAIRED_FIELDS]
    original = {str(value): index for index, value in enumerate(inputs['condition_id']) if inputs['scenario'][index] == 'original'}
    shifted = {str(value): index for index, value in enumerate(inputs['condition_id']) if inputs['scenario'][index] == 'obstacle_shift'}
    for scope, mask in _scopes(inputs):
        for scenario in parent.SCENARIOS:
            for selection in ('all_stress', 'frozen_feasible'):
                selected = mask & (inputs['scenario'] == scenario)
                if selection == 'frozen_feasible':
                    selected &= inputs['feasible_mask']
                for a, b, effect in ((1, 0, 'projection_on_minus_off'), (2, 1, 'on_smooth_minus_on')):
                    label = scope | dict(effect=effect, scenario=scenario, selection=selection)
                    delta = values[a, selected][:, fields]-values[b, selected][:, fields]
                    yield label, delta, np.flatnonzero(selected)
        a = np.array([index for _, index in sorted(original.items()) if mask[index]], dtype=int)
        b = np.array([shifted[str(inputs['condition_id'][index])] for index in a], dtype=int)
        for selection in ('all_stress', 'both_scenarios_frozen_feasible'):
            choose = np.ones(len(a), bool) if selection == 'all_stress' else inputs['feasible_mask'][a] & inputs['feasible_mask'][b]
            for index, projection in enumerate(PROJECTIONS):
                label = scope | dict(effect='shift_minus_original', projection=projection, selection=selection)
                delta = values[index, b[choose]][:, fields]-values[index, a[choose]][:, fields]
                yield label, delta, a[choose]
            label = scope | dict(effect='smooth_shift_effect_minus_on_shift_effect', selection=selection)
            delta = ((values[2, b[choose]][:, fields]-values[2, a[choose]][:, fields])
                     -(values[1, b[choose]][:, fields]-values[1, a[choose]][:, fields]))
            yield label, delta, a[choose]


def _paired_statistic(label: dict, delta: np.ndarray, rows: np.ndarray, inputs: dict,
                      replicates: int, seed: int) -> dict:
    return label | parent.cluster_statistics(delta, inputs['map_id'][rows], names=PAIRED_FIELDS,
        replicates=replicates, seed=parent._seed(seed, parent._json(label)), strata=inputs['family'][rows])


def _paired(values: np.ndarray, inputs: dict, replicates: int, seed: int) -> list:
    return [_paired_statistic(label, delta, rows, inputs, replicates, seed)
            for label, delta, rows in _paired_deltas(values, inputs)]


def _seed_metrics(cells: list, arrays: list, inputs: dict, names: tuple) -> None:
    for cell in cells:
        selected = inputs['scenario'] == cell['scenario']
        for key in ('family', 'kind', 'difficulty'):
            if key in cell:
                selected &= inputs[key] == cell[key]
        if cell['selection'] == 'frozen_feasible':
            selected &= inputs['feasible_mask']
        projection = PROJECTIONS.index(cell['projection'])
        for index, name in enumerate(names):
            means = []
            for array in arrays:
                value = array[projection, selected, index]; finite = value[np.isfinite(value)]
                means.append(float(finite.mean()) if len(finite) else None)
            observed = [value for value in means if value is not None]
            cell['metrics'][name].update(seed_means=means,
                seed_std=float(np.std(observed, ddof=1)) if len(observed) > 1 else 0. if observed else None)


def _paired_group(arrays: list, inputs: dict, replicates: int, seed: int) -> list:
    result = []
    # Stream one contrast at a time to avoid retaining all scope matrices.
    for group in zip(*(_paired_deltas(array, inputs) for array in arrays), strict=True):
        label, _, rows = group[0]
        if any(other != label or not np.array_equal(indices, rows) for other, _, indices in group):
            raise IntegrityError('training seeds do not share the same paired condition identities')
        matrices = [delta for _, delta, _ in group]
        cell = _paired_statistic(label, parent._mean_stack(matrices), rows, inputs, replicates, seed)
        for index, name in enumerate(PAIRED_FIELDS):
            means = []
            for matrix in matrices:
                observed = matrix[:, index][np.isfinite(matrix[:, index])]
                means.append(float(observed.mean()) if len(observed) else None)
            observed = [value for value in means if value is not None]
            cell['metrics'][name].update(seed_means=means,
                seed_std=float(np.std(observed, ddof=1)) if len(observed) > 1 else 0. if observed else None)
        result.append(cell)
    return result


def _seed_groups(models: dict, matrix: list, inputs: dict, replicates: int, seed: int) -> list:
    groups = defaultdict(list)
    for row in matrix:
        if row['id'] in models:
            groups[(row['variant'], row['train_size'])].append(row)
    result = []
    for (variant, size), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row['seed'])
        values = [models[row['id']]['primary'] for row in rows]
        cells = _cells(parent._mean_stack(values), inputs, replicates, seed)
        _seed_metrics(cells, values, inputs, FIELDS)
        multi_rows = models[rows[0]['id']]['multi_rows']
        subset = {key: value[multi_rows] for key, value in inputs.items()}
        any_values = [models[row['id']]['any_values'] for row in rows]
        any_cells = _cells(parent._mean_stack(any_values), subset, replicates, seed, names=ANY_FIELDS)
        _seed_metrics(any_cells, any_values, subset, ANY_FIELDS)
        diverse_values = [models[row['id']]['diversity_values'] for row in rows]
        diverse_cells = _cells(parent._mean_stack(diverse_values), subset, replicates, seed, names=DIVERSITY_FIELDS)
        _seed_metrics(diverse_cells, diverse_values, subset, DIVERSITY_FIELDS)
        paired = _paired_group(values, inputs, replicates, seed)
        result.append(dict(variant=variant, train_size=size, model_ids=[row['id'] for row in rows],
            seeds=[row['seed'] for row in rows], expected_seed_count=3, observed_seed_count=len(rows), cells=cells,
            paired_effects=paired,
            multimodal=dict(any_at_8=any_cells, diversity=diverse_cells)))
    return result


def _adapter_controls(models: dict, matrix: list, inputs: dict, replicates: int, seed: int) -> tuple[list, list]:
    """Within-seed adapter/control pairs, then seed-averaged paired map CIs."""
    lookup = {(row['variant'], row['train_size'], row['seed']): row['id'] for row in matrix}
    pairs, grouped = [], defaultdict(list)
    for row in matrix:
        if row['variant'] != 'geometry' or row['id'] not in models:
            continue
        controls = [('base', 'base')]
        if row['train_size'] == 8000:
            controls += [(variant, lookup[(variant, 8000, row['seed'])]) for variant in
                ('no_contact_loss', 'no_geometry_tokens') if (variant, 8000, row['seed']) in lookup]
        for variant, identifier in controls:
            if identifier not in models:
                continue
            delta = models[row['id']]['primary']-models[identifier]['primary']
            pairs.append(dict(effect='geometry_adapter_minus_control', model_id=row['id'], control_id=identifier,
                control_variant=variant, train_size=row['train_size'], seed=row['seed'],
                cells=_cells(delta, inputs, replicates, seed)))
            grouped[(row['train_size'], variant)].append((row['seed'], delta))
    groups = []
    for (size, variant), group in sorted(grouped.items()):
        group.sort(key=lambda pair: pair[0]); arrays = [value for _, value in group]
        cells = _cells(parent._mean_stack(arrays), inputs, replicates, seed)
        _seed_metrics(cells, arrays, inputs, FIELDS)
        groups.append(dict(effect='geometry_adapter_minus_control', train_size=size, control_variant=variant,
            seeds=[value for value, _ in group], expected_seed_count=3, observed_seed_count=len(group), cells=cells))
    return pairs, groups


def _baseline_diversity(prior: dict, multi_rows: np.ndarray, inputs: dict) -> np.ndarray:
    values = np.full((3, len(multi_rows), len(DIVERSITY_FIELDS)), np.nan)
    lookup = {(str(inputs['condition_id'][row]), str(inputs['scenario'][row])): index for index, row in enumerate(multi_rows)}
    seen = set()
    for record in prior['multimodal']['condition_pair_details']:
        projection = record.get('projection')
        if projection not in ('off', 'on'):
            raise IntegrityError('parent diversity contains a non-parent projection label')
        identity = (record['condition_id'], record['scenario']); index = lookup.get(identity)
        key = (projection, identity)
        if index is None or key in seen:
            raise IntegrityError('parent diversity condition identity is repeated or outside fixed K8 subset')
        seen.add(key)
        pairs = record['successful_pairs']; successful = record['successful_candidates']
        if pairs != successful*(successful-1)//2 or successful < 0 or successful > 8:
            raise IntegrityError('parent diversity successful-pair denominator is malformed')
        if not pairs and any(record[name] is not None for name in ('position_pair_msd_m2', 'position_pair_rms_m', 'uv_pair_msd_m2')):
            raise IntegrityError('parent condition without successful pairs must have None diversity')
        values[PROJECTIONS.index(projection), index] = [successful,
            record['position_pair_msd_m2'] if pairs else np.nan,
            record['position_pair_rms_m'] if pairs else np.nan,
            record['uv_pair_msd_m2'] if pairs else np.nan,
            float(record['both_direction_signs']) if successful >= 2 else np.nan]
    if len(seen) != 2*len(multi_rows):
        raise IntegrityError('complete parent report lacks fixed-subset K8 diversity conditions')
    return values


def _markdown(report: dict, figures: list) -> str:
    preview = report['status'] != 'complete'
    title = '# 未完成预览：Contact 平滑投影补充消融' if preview else '# Contact 平滑投影补充消融研究报告'
    lines = [title, '', f"状态：`{report['status']}`。计划为22模型、1000个源地图、10000个条件；原始与障碍移动场景各做K=1，固定1000条件做K=8。",
        '', '三个decoder为关闭投影（off）、冻结解析接触投影（on）和固定有界三次拟合后解析接触节点（on_smooth）。'
        '补充decoder固定16控制点、128节点、归一化图域二阶导积分正则λ=1e-6；输出边仍按线性intrinsic/lift解释。'
        '训练预算和模型完全沿用父实验，不读取oracle、不重新拟合保存结果、不修复障碍，曲率与全身碰撞是否达标由独立评分决定。',
        '', '成功率分母保留每个已请求且已评分候选，包括投影异常与不收敛候选；缺失分片只允许预览。'
        '全部压力输入与推理前冻结的物理见证可行子集分别报告；障碍移动的可行配对使用两个场景交集。'
        f"95%区间使用按family分层的源地图cluster bootstrap（本次{report['bootstrap_replicates']}次，种子{report['bootstrap_seed']}；正式固定1000次/20261011）；3个训练种子的均值与标准差单独报告。"
        'minimum_clearance_m是128节点处、已扣安全边界的最小间距，不是密集边最小距离。', '',
        '| 模型 | 规模/seed | 原始 off / on / smooth | 移动 off / on / smooth |',
        '|---|---:|---:|---:|']
    for name, model in report['models'].items():
        training = model['training']; cells = model['primary']; values = []
        for scenario in parent.SCENARIOS:
            values.append(' / '.join(parent._percent(parent._overall(cells, scenario=scenario, projection=projection)['metrics']['candidate_success']['mean']) for projection in PROJECTIONS))
        lines.append(f"| {name} | {training.get('train_size', 0)}/{training.get('seed')} | {values[0]} | {values[1]} |")
    lines.extend(['', 'K=8多样性只使用实际成功候选的保存body XYZ及intrinsic UV，先在单条件成功pair中取均值，再对条件聚合。'
        '单条件少于两个成功候选时pair多样性为None，不填0；K=8任一成功率仍保留全部条件。'
        '方向符号只是相对起终点弦的经验偏离，不能严格声称同伦覆盖。父off/on多样性沿用已校验的完整父报告。', '',
        'JSON包含全部模型、数据规模与3seed聚合、family/type/difficulty交叉及独立分层、接触/法向/曲率/间距/路径质量、'
        '拟合状态与修正量、配对smooth−on、移动效应和移动差分效应、候选失败类型及所有输入/源码/结果校验。'])
    lines.extend(['', 'adapter−Base以及8k geometry−no_contact_loss / geometry−no_geometry_tokens的配对效果，'
        '在同一训练seed与相同测试条件下分别计算；三seed聚合先按条件取seed均值，再以源地图cluster给出区间。'
        '三种decoder都保留这些负控效果，避免将decoder平滑本身的收益归因给adapter。'])
    if report['seed_groups']:
        lines.extend(['', '| variant / expert路径数 | seeds | 原始smooth成功率 mean ± seed std |', '|---|---|---:|'])
        for group in report['seed_groups']:
            metric = parent._overall(group['cells'], scenario='original', projection='on_smooth')['metrics']['candidate_success']
            std = metric.get('seed_std')
            lines.append(f"| {group['variant']} / {group['train_size']} | {group['seeds']} | {parent._percent(metric['mean'])} ± {parent._percent(std)} |")
    if preview:
        lines.extend(['', '本文件仅为明确标注的小样本/缺数据预览，不能替代22模型完整研究报告。', '', '缺项：'])
        lines.extend(f"- {parent._json(problem)}" for problem in report['shortfalls'])
    lines.extend(['', *[f'![{Path(name).stem}]({name})' for name in figures], ''])
    return '\n'.join(lines)


def _figures(report: dict, output: Path, prefix: str) -> list[str]:
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/contact_smooth_report_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    sources = [(name, model['primary'], model['multimodal']) for name, model in report['models'].items()]
    colors = ('#686868', '#2c6eaa', '#ca6b2b'); figures = []
    def save(fig, stem):
        if report['status'] != 'complete':
            fig.suptitle('INCOMPLETE PREVIEW / not a completed research result', fontsize=10)
        name = f'{prefix}{stem}.png'; fig.tight_layout(); fig.savefig(output/name, dpi=240, bbox_inches='tight'); plt.close(fig); figures.append(name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, scenario in zip(axes, parent.SCENARIOS):
        for projection, color in zip(PROJECTIONS, colors):
            x, y = [], []
            for index, (_, cells, _) in enumerate(sources):
                metric = parent._overall(cells, scenario=scenario, projection=projection)['metrics']['candidate_success']
                if metric['mean'] is not None:
                    x.append(index); y.append(metric['mean'])
            axis.plot(x, y, 'o-', color=color, label=projection, markersize=3)
        axis.set(title=scenario, ylabel='K=1 candidate success rate', ylim=(-.02, 1.02))
        axis.set_xticks(range(len(sources)), [name for name, _, _ in sources], rotation=80, fontsize=6)
        axis.legend(frameon=False)
    save(fig, 'decoder_k1_success')
    groups = [group for group in report['seed_groups'] if group['variant'] == 'geometry']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, scenario in zip(axes, parent.SCENARIOS):
        for projection, color in zip(PROJECTIONS, colors):
            data = []
            for group in sorted(groups, key=lambda item: item['train_size']):
                metric = parent._overall(group['cells'], scenario=scenario, projection=projection)['metrics']['candidate_success']
                if metric['mean'] is not None:
                    data.append((group['train_size'], metric['mean'], metric['ci95']))
            if data:
                x = np.array([item[0] for item in data]); y = np.array([item[1] for item in data])
                bounds = np.array([item[2] for item in data], dtype=float)
                axis.plot(x, y, 'o-', color=color, label=projection)
                axis.fill_between(x, bounds[:, 0], bounds[:, 1], color=color, alpha=.14)
        axis.set(title=scenario, xlabel='Training expert trajectories', ylabel='K=1 success (map-cluster 95% CI)', ylim=(-.02, 1.02))
        if groups:
            axis.set_xscale('log', base=2); axis.legend(frameon=False)
        else:
            axis.text(.5, .5, 'No completed three-seed size groups', ha='center', va='center', transform=axis.transAxes)
    save(fig, 'decoder_training_scale_and_shift')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, field in zip(axes, ('any_at_8', 'position_pair_msd_m2')):
        for projection_index, (projection, color) in enumerate(zip(PROJECTIONS, colors)):
            points = []
            for index, (_, _, multimodal) in enumerate(sources):
                cells = multimodal['any_at_8' if field == 'any_at_8' else 'diversity']
                metric = parent._overall(cells, scenario='original', projection=projection)['metrics'][field]
                if metric['mean'] is not None:
                    points.append((index+(projection_index-1)*.2, metric['mean']))
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], 'o', color=color, label=projection, markersize=4)
        axis.set(ylabel='K=8 any-success rate' if field == 'any_at_8' else 'Successful-pair body position MSD (m²)', title='Original / fixed K=8 subset')
        axis.set_xticks(range(len(sources)), [name for name, _, _ in sources], rotation=80, fontsize=6)
        if axis.lines:
            axis.legend(frameon=False)
    save(fig, 'decoder_k8_success_and_saved_pair_diversity')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    fields = ('maximum_curvature_inv_m', 'minimum_local_chord_arc_ratio', 'minimum_clearance_m')
    for axis, field, threshold in zip(axes, fields, (8., .65, 0.)):
        for projection_index, (projection, color) in enumerate(zip(PROJECTIONS, colors)):
            points = []
            for index, (_, cells, _) in enumerate(sources):
                metric = parent._overall(cells, scenario='original', projection=projection)['metrics'][field]
                if metric['mean'] is not None:
                    points.append((index+(projection_index-1)*.2, metric['mean']))
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], 'o', color=color, label=projection, markersize=3)
        axis.axhline(threshold, color='#444444', lw=.8, ls='--')
        axis.set(ylabel=field, title='Saved waypoint diagnostic / original')
        axis.set_xticks(range(len(sources)), [name for name, _, _ in sources], rotation=80, fontsize=6)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False, fontsize=7)
    save(fig, 'decoder_curvature_chord16_clearance')
    selected = next((group for group in report['seed_groups'] if group['variant'] == 'geometry' and group['train_size'] == 8000), None)
    cells = selected['cells'] if selected else report['models'].get('base', {}).get('primary', [])
    families = sorted({cell['family'] for cell in cells if cell['scope'] == 'family'})
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, .35*len(families)+1.5)))
    for axis, selection in zip(axes, ('all_stress', 'frozen_feasible')):
        values = np.full((len(families), 6), np.nan)
        for row, family in enumerate(families):
            for column, (scenario, projection) in enumerate((scenario, projection) for scenario in parent.SCENARIOS for projection in PROJECTIONS):
                cell = next((cell for cell in cells if cell['scope'] == 'family' and cell['family'] == family and
                    cell['scenario'] == scenario and cell['projection'] == projection and cell['selection'] == selection), None)
                if cell is not None and cell['metrics']['candidate_success']['mean'] is not None:
                    values[row, column] = cell['metrics']['candidate_success']['mean']
        if len(families):
            artist = axis.imshow(values, vmin=0., vmax=1., cmap='viridis', aspect='auto')
            fig.colorbar(artist, ax=axis, label='K=1 candidate success')
            axis.set_yticks(range(len(families)), families)
            axis.set_xticks(range(6), [f'{scenario[:4]} / {projection}' for scenario in parent.SCENARIOS for projection in PROJECTIONS], rotation=60, ha='right')
        axis.set_title(f'{selection} / geometry 8k seed mean' if selected else f'{selection} / Base preview')
    save(fig, 'decoder_family_stress_and_frozen_feasible')
    return figures


def build_report(experiment: str | Path, benchmark: str | Path, supplement: str | Path | None = None,
                 output: str | Path | None = None, *, allow_partial: bool = False,
                 bootstrap_replicates: int = 1000, bootstrap_seed: int = 20261011,
                 make_figures: bool = True) -> dict:
    experiment, benchmark = Path(experiment).resolve(), Path(benchmark).resolve()
    supplement = Path(supplement).resolve() if supplement is not None else experiment/'smooth_projection_supplement'
    output = Path(output).resolve() if output is not None else supplement/'reports'
    if (output == experiment or output == supplement or output.is_relative_to(experiment/'evaluation') or
            output.is_relative_to(supplement/'evaluation')):
        raise ValueError('report output must be separate from experimental payloads')
    if bootstrap_replicates < 1:
        raise ValueError('bootstrap_replicates must be positive')
    problems = []
    protocol, manifest, inputs, prior, smooth, sources, parent_sources = _context(
        experiment, benchmark, supplement, problems, allow_partial=allow_partial)
    parent_report_directory = Path(prior['_report_path']).resolve().parent
    if output.is_relative_to(parent_report_directory):
        raise ValueError('supplement report output must be outside the frozen parent report directory')
    if bootstrap_replicates != 1000 or bootstrap_seed != 20261011:
        problems.append(dict(stage='bootstrap', reason='completed research requires 1000 replicates and seed 20261011'))
    planned = [dict(id='base', variant='base', train_size=0, seed=None), *protocol['variants']]
    if (len(planned) != 22 or protocol['new_source_maps'] != 1000 or protocol['new_test_conditions'] != 10000 or
            protocol['multimodal_subset_conditions'] != 1000 or smooth['paths_per_model'] != 34000):
        problems.append(dict(stage='matrix', reason='requires exactly 22 models / 1000 maps / 10000 conditions / fixed 1000-condition K8'))
    summaries, models = {}, {}
    for row in planned:
        parent_problems = []
        baseline = parent._read_model(experiment, benchmark, row, protocol, inputs, parent._sha(benchmark/'manifest.json'), parent_problems)
        if baseline is None or parent_problems or not baseline['complete']:
            raise IntegrityError('parent report no longer matches a complete raw/score/checkpoint matrix')
        prior_model = prior['models'][row['id']]
        if prior_model.get('contract') != baseline['contract'] or prior_model.get('complete') is not True:
            raise IntegrityError('parent report model/contract is not the audited completed model')
        for name, digest in baseline['artifact_sha256'].items():
            if prior_model.get('artifact_sha256', {}).get(name) != digest:
                raise IntegrityError('parent report does not authenticate its current raw/score files')
        expected_parent = dict(inference_version=evaluation.INFERENCE_VERSION, projection_version=evaluation.PROJECTION_VERSION,
            metrics_version=evaluation.METRICS_VERSION, ddim_steps=protocol['sampling_steps'], latent_seed=protocol['latent_seed'],
            primary_candidates=1, multimodal_candidates=8,
            multimodal_maps_per_family=10 if protocol['schema_version'] == 'contact_research_protocol_v001' else
                int(smooth.get('multimodal_maps_per_family', 1)))
        if any(baseline['contract'].get(key) != value for key, value in expected_parent.items()):
            raise IntegrityError('parent inference/decoder/metric/candidate versions differ from the paired protocol')
        if row['id'] == 'base' and baseline['contract'].get('adapter_sha256') is not None:
            raise IntegrityError('parent Base contract unexpectedly binds an adapter')
        # Check all inference/evaluation sources, rather than the parent's subset.
        for section, names in (('inference_sources', evaluation.INFERENCE_SOURCES), ('evaluation_sources', evaluation.EVALUATION_SOURCES)):
            if baseline['contract'].get(section) != {name: parent._sha(Path(__file__).parent/name) for name in names}:
                raise IntegrityError('parent evaluation source provenance differs from frozen sources')
        new = _read_smooth(experiment, benchmark, supplement, row, inputs, manifest, smooth, sources, baseline, problems)
        parent_primary, parent_multiple = _parent_values(experiment, baseline, inputs)
        primary = np.full((3, len(inputs['case_id']), len(FIELDS)), np.nan)
        primary[:2] = parent_primary; primary[2] = new['primary']
        multi_rows = baseline['multi_rows']; subset = {key: value[multi_rows] for key, value in inputs.items()}
        multiple = np.full((3, len(multi_rows), 8, len(FIELDS)), np.nan)
        multiple[:2] = parent_multiple; multiple[2] = new['multiple']
        any_values = np.full((3, len(multi_rows), len(ANY_FIELDS)), np.nan)
        for index, field in enumerate(('candidate_success', 'numeric_candidate_success', 'physical_collision_free')):
            values = multiple[..., INDEX[field]]; complete = np.isfinite(values).all(-1)
            any_values[..., index][complete] = np.any(values[complete] > .5, axis=-1)
        diverse = _baseline_diversity(prior_model, multi_rows, inputs); diverse[2] = new['diversity']
        summaries[row['id']] = dict(training=baseline['training'], parent_contract=baseline['contract'], complete=new['complete'],
            contract=new['contract'], scoring_status=new['scoring_status'],
            primary=_cells(primary, inputs, bootstrap_replicates, bootstrap_seed),
            paired_effects=_paired(primary, inputs, bootstrap_replicates, bootstrap_seed),
            multimodal=dict(subset_conditions=len(multi_rows)//2, candidates=8,
                any_at_8=_cells(any_values, subset, bootstrap_replicates, bootstrap_seed, names=ANY_FIELDS),
                diversity=_cells(diverse, subset, bootstrap_replicates, bootstrap_seed, names=DIVERSITY_FIELDS),
                condition_pair_details=prior_model['multimodal']['condition_pair_details']+new['details']),
            artifact_sha256=new['artifacts'], parent_artifact_sha256=baseline['artifact_sha256'],
            evaluation_errors=baseline['evaluation_errors']+new['errors'], failure_counts=baseline['failure_counts']+new['failures'])
        models[row['id']] = dict(primary=primary, multi_rows=multi_rows, any_values=any_values, diversity_values=diverse)
    if problems and not allow_partial:
        raise IncompleteExperiment(f'no completed supplement report emitted: {len(problems)} shortfalls; use --allow-partial for an explicit preview')
    driver = smooth.pop('_driver_contract'); smooth.pop('_multimodal_maps')
    controls, control_groups = _adapter_controls(models, protocol['variants'], inputs, bootstrap_replicates, bootstrap_seed)
    report = dict(schema_version=REPORT_VERSION, status='incomplete_preview' if problems else 'complete',
        created_utc=datetime.now(timezone.utc).isoformat(), shortfalls=problems, models=summaries,
        seed_groups=_seed_groups(models, protocol['variants'], inputs, bootstrap_replicates, bootstrap_seed),
        paired_adapter_controls=controls, paired_adapter_control_groups=control_groups,
        protocol=smooth, parent_protocol=protocol, benchmark=manifest,
        protocol_file_sha256=parent._sha(supplement/'protocol.json'),
        benchmark_manifest_sha256=parent._sha(benchmark/'manifest.json'), report_source_sha256=parent._sha(Path(__file__)),
        source_contract_file_sha256=parent._sha(supplement/'source_contract.json'),
        source_contract=sources, driver_source_contract=driver, parent_source_contract=parent_sources,
        parent_report_sha256=parent._sha(Path(prior['_report_path'])), parent_report_path=prior['_report_path'],
        bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, projections=list(PROJECTIONS),
        independent_units='family-stratified source-map clusters; separate training-seed mean/std',
        denominator_policy='all requested failures retained; frozen witnessed subsets; paired-shift feasibility intersection',
        diversity_policy='saved actual successful-pair XYZ/UV, condition weighted; missing pair = None; no decoder replay',
        oracle_access=False, decoder_replayed=False, new_sampling=False, primary_candidates=1)
    output.mkdir(parents=True, exist_ok=True)
    prefix = 'preview_' if problems else ''
    figures = _figures(report, output, prefix) if make_figures else []
    if not problems and not figures:
        raise IncompleteExperiment('completed research requires scientific PNG figures')
    report['figures'] = {name: parent._sha(output/name) for name in figures}
    parent._write(output/('report_preview.json' if problems else 'report.json'), report)
    (output/('research_report_zh_preview.md' if problems else 'research_report_zh.md')).write_text(_markdown(report, figures), encoding='utf8')
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    report = build_report(args.experiment, args.benchmark, args.supplement, args.output, allow_partial=args.allow_partial)
    print(parent._json(dict(status=report['status'], models=len(report['models']), shortfalls=len(report['shortfalls']))))


if __name__ == '__main__':
    main()
