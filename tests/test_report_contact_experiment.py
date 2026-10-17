"""Small synthetic score fixtures, never a formal experimental report."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from hnuter_dit import report_contact_experiment as report


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_gzip(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')


def _condition(index=0):
    return {'schema_version': 'surface_task_condition_v001', 'task_type': 'surface',
        'task_version': 'surface_contact_v003', 'surface_geometry': {'type': 'plane',
        'origin': [0., 0., 1.2+index*.01], 'quaternion_wxyz': [1., 0., 0., 0.],
        'thickness_m': .1, 'supports': []}, 'surface_chart': {'bounds': [[-2., 2.], [-2., 2.], [-3.14, 3.14]],
        'forbidden_regions': []}, 'tool_extrinsics': {'position_body': [0., 0., -.75],
        'quaternion_wxyz': [0., 1., 0., 0.]}, 'contact_constraints': {'position_tolerance_m': 1e-7,
        'normal_tolerance_rad': 1e-6, 'collision_margin_m': .01, 'validation_step_m': .02,
        'validation_step_rad': .02, 'attitude_limits_deg': {'roll': [-40., 40.], 'pitch': [-70., 70.]}},
        'start_goal': {'start': [-1., 0., 0.], 'goal': [1., 0., 0.]},
        'environment': {'bounds': {'min': [-4., -4., 0.], 'max': [4., 4., 4.]}, 'obstacles': []}}


def _paths(condition):
    from hnuter_dit.contact_projection import ContactProjectionSpec, lift_contact_intrinsic
    import torch
    progress = np.linspace(0, 1, 128)
    paths = []
    for candidate in range(8):
        states = np.column_stack((-1+2*progress,
            (.4 if candidate%2 else -.4)*np.sin(np.pi*progress), np.zeros(128)))
        states[[0, -1]] = [condition['start_goal']['start'], condition['start_goal']['goal']]
        paths.append(lift_contact_intrinsic(torch.as_tensor(states), ContactProjectionSpec.from_condition(condition)).numpy())
    return np.asarray(paths, dtype=np.float32)


@pytest.fixture
def experiment_fixture(tmp_path):
    experiment = tmp_path/'experiment'; benchmark = tmp_path/'benchmark'; output = tmp_path/'report'
    experiment.mkdir(); benchmark.mkdir()
    protocol = {'schema_version': 'test_protocol', 'base_sha256': 'frozen-base-test-sha',
        'variants': [], 'model_seeds': [17, 29, 43], 'optimizer_updates': 8000,
        'new_test_conditions': 2, 'new_source_maps': 2, 'multimodal_subset_conditions': 1,
        'sampling_steps': 25, 'latent_seed': 20261011}
    protocol['protocol_sha256'] = report._hash(protocol)
    _write_json(experiment/'protocol.json', protocol)
    inputs = {key: [] for key in ('case_id', 'condition_id', 'map_id', 'family', 'kind', 'difficulty',
                                 'scenario', 'feasible_mask', 'feasibility_status', 'input_sha256', 'input_shard', 'input_row')}
    records = {}; files = {}
    for index in range(2):
        map_id = f'fixture/map_{index:06d}'; identifier = map_id+'/condition_000'
        condition = _condition(index)
        record = {'schema_version': 'contact_benchmark_input_v001', 'condition_id': identifier,
            'map_id': map_id, 'source_map_index': index, 'family': 'central_block', 'kind': 'plane',
            'difficulty': 'detour', 'angle_band': 0, 'body_endpoints_pose7': [],
            'conditions': {name: copy.deepcopy(condition) for name in report.SCENARIOS},
            'input_sha256': {name: report._hash(condition) for name in report.SCENARIOS},
            'shift': {'moved_obstacle_id': 'fixture', 'displacement_xyz': [0., .3, 0.], 'shift_distance_m': .3,
                      'shift_policy': 'test-fixture', 'structural_fixture_members_moved_together': False}}
        relative = f'maps/map_{index:06d}/inputs.jsonl.gz'
        _write_gzip(benchmark/relative, [record]); files[relative] = report._sha(benchmark/relative)
        records[map_id] = record
        for scenario in report.SCENARIOS:
            feasible = not (index == 0 and scenario == 'obstacle_shift')
            values = {'case_id': identifier+'/'+scenario, 'condition_id': identifier, 'map_id': map_id,
                'family': 'central_block', 'kind': 'plane', 'difficulty': 'detour', 'scenario': scenario,
                'feasible_mask': feasible, 'feasibility_status': 'expert_witness_valid' if feasible else 'unknown',
                'input_sha256': report._hash(condition), 'input_shard': relative, 'input_row': 0}
            for key, value in values.items():
                inputs[key].append(value)
    inputs = {key: np.asarray(value) for key, value in inputs.items()}
    np.savez_compressed(benchmark/'runtime_inputs.npz', **inputs)
    files['runtime_inputs.npz'] = report._sha(benchmark/'runtime_inputs.npz')
    # No oracle file exists: a completed report must never try to read this hash.
    files['oracle/map_000000.jsonl.gz'] = 'unreadable-isolated-oracle'
    manifest = {'status': 'frozen', 'condition_count': 2, 'source_map_count': 2,
                'input_files_sha256': files, 'runtime_cache': 'runtime_inputs.npz', 'source_parameters': []}
    _write_json(benchmark/'manifest.json', manifest)
    contract = {'model_id': 'base', 'base_sha256': protocol['base_sha256'], 'adapter_sha256': None,
        'benchmark_manifest_sha256': report._sha(benchmark/'manifest.json'), 'benchmark_conditions': 2,
        'primary_candidates': 1, 'multimodal_maps_per_family': 1, 'multimodal_candidates': 8,
        'ddim_steps': 25, 'latent_seed': 20261011}
    contract['contract_sha256'] = report._hash(contract)
    directory = experiment/'evaluation/base'; _write_json(directory/'contract.json', contract)
    multi = report._multimodal_maps(inputs, 1); score_count = 0
    for map_id, record in records.items():
        stem = map_id.rsplit('/', 1)[-1]
        rows = np.flatnonzero(inputs['map_id'] == map_id); count = 8 if map_id in multi else 1
        requested_rows = np.repeat(rows, count); candidates = np.tile(np.arange(count), len(rows))
        paths = _paths(record['conditions']['original'])
        raw_path = directory/'raw'/f'{stem}.npz'; raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, world_pose9=paths[candidates], input_rows=requested_rows, candidate_indices=candidates)
        raw_sha = report._sha(raw_path)
        _write_json(raw_path.with_suffix('.json'), {'contract_sha256': contract['contract_sha256'],
                    'raw_sha256': raw_sha, 'map_id': map_id})
        scores = []
        for row, candidate in zip(requested_rows, candidates):
            for projection in report.PROJECTIONS:
                success = bool(inputs['feasible_mask'][row] and (projection == 'on' or candidate > 0))
                metrics = {'candidate_success': success, 'numeric_candidate_success': success,
                    'physical_collision_free': True, 'contact_error_max_m': 0. if success else .01,
                    'contact_error_mean_m': 0. if success else .005, 'normal_error_max_rad': 0.,
                    'normal_error_mean_rad': 0., 'path_length_m': 2.4, 'path_efficiency': 1.2,
                    'minimum_clearance_m': .5, 'failures': [] if success else ['contact']}
                scores.append({key: str(inputs[key][row]) for key in ('case_id', 'condition_id', 'map_id', 'family',
                    'kind', 'difficulty', 'scenario', 'input_sha256', 'feasibility_status')} |
                    {'model_id': 'base', 'candidate_index': int(candidate), 'projection': projection,
                     'multimodal_subset': map_id in multi, 'feasible': bool(inputs['feasible_mask'][row]), 'metrics': metrics})
        score_path = directory/'scores'/f'{stem}.jsonl.gz'; _write_gzip(score_path, scores)
        _write_json(score_path.with_name(f'{stem}.json'), {'contract_sha256': contract['contract_sha256'],
            'raw_sha256': raw_sha, 'scores_sha256': report._sha(score_path), 'score_rows': len(scores), 'map_id': map_id})
        score_count += len(scores)
    _write_json(directory/'scoring_status.json', {'status': 'complete', 'complete_maps': 2, 'total_maps': 2,
        'score_rows': score_count, 'contract_sha256': contract['contract_sha256']})
    return experiment, benchmark, output, inputs


def test_cluster_ci_resamples_maps_and_keeps_missing_scalar_denominators():
    values = np.r_[np.zeros(10), np.ones(10)]
    stats = report.cluster_statistics(values, np.repeat(['a', 'b'], 10), names=('success',), replicates=1000)
    assert stats['source_map_count'] == 2 and stats['condition_count'] == 20
    assert stats['metrics']['success']['mean'] == .5
    assert stats['metrics']['success']['ci95'] == [0., 1.]
    missing = report.cluster_statistics(np.array([1., np.nan]), np.array(['seen', 'missing']),
        names=('error',), replicates=32)
    assert missing['source_map_count'] == 1 and missing['planned_source_map_count'] == 2
    assert missing['metrics']['error']['observations'] == 1


def test_complete_report_checks_full_matrix_and_never_reads_oracle(experiment_fixture):
    experiment, benchmark, output, _ = experiment_fixture
    result = report.build_report(experiment, benchmark, output, bootstrap_replicates=32, make_figures=False)
    assert result['status'] == 'complete' and result['shortfalls'] == [] and result['oracle_access'] is False
    assert (output/'report.json').is_file() and (output/'research_report_zh.md').is_file()
    primary = result['models']['base']['primary']
    original = report._overall(primary, scenario='original', projection='on')
    assert original['metrics']['candidate_success']['mean'] == 1.
    assert original['metrics']['candidate_success']['observations'] == 2
    shift = report._overall(primary, scenario='obstacle_shift', projection='on')
    assert shift['metrics']['candidate_success']['mean'] == .5
    filtered = report._overall(primary, scenario='obstacle_shift', projection='on', selection='frozen_feasible')
    assert filtered['metrics']['candidate_success']['mean'] == 1.
    assert filtered['condition_count'] == 1
    assert result['models']['base']['multimodal']['subset_conditions'] == 1
    text = (output/'research_report_zh.md').read_text()
    assert '不能严格声称同伦' in text and '不是密集边最小距离' in text


def test_missing_projected_k8_candidate_cannot_be_called_complete(experiment_fixture):
    experiment, benchmark, output, _ = experiment_fixture
    directory = experiment/'evaluation/base'; contract = report._read(directory/'contract.json')
    for meta_path in sorted((directory/'scores').glob('map_*.json')):
        meta = report._read(meta_path)
        if meta['score_rows'] > 4:
            score_path = meta_path.with_name(meta_path.stem+'.jsonl.gz')
            with gzip.open(score_path, 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            rows = [row for row in rows if not (row['candidate_index'] == 7 and row['projection'] == 'on' and row['scenario'] == 'original')]
            _write_gzip(score_path, rows)
            meta['scores_sha256'] = report._sha(score_path); meta['score_rows'] = len(rows); _write_json(meta_path, meta)
            break
    with pytest.raises(report.IncompleteExperiment, match='no completed report'):
        report.build_report(experiment, benchmark, output, bootstrap_replicates=16, make_figures=False)
    assert not (output/'report.json').exists()
    result = report.build_report(experiment, benchmark, output, allow_partial=True, bootstrap_replicates=16, make_figures=False)
    assert result['status'] == 'incomplete_preview'
    assert any(row.get('missing_score_rows') == 1 for row in result['shortfalls'])
    assert not (output/'report.json').exists()
    assert (output/'research_report_zh_preview.md').read_text().startswith('# 未完成')


def test_hash_corruption_fails_even_partial_preview(experiment_fixture):
    experiment, benchmark, output, _ = experiment_fixture
    score_path = next((experiment/'evaluation/base/scores').glob('*.jsonl.gz'))
    score_path.write_bytes(score_path.read_bytes()+b'corrupted')
    with pytest.raises(report.IntegrityError, match='checksum'):
        report.build_report(experiment, benchmark, output, allow_partial=True, bootstrap_replicates=16, make_figures=False)
    assert not output.exists()


def test_shift_pair_uses_both_feasibility_intersection(experiment_fixture):
    _, _, _, inputs = experiment_fixture
    values = np.zeros((2, 4, len(report.FIELDS)))
    values[:, inputs['scenario'] == 'original', report.INDEX['candidate_success']] = 1.
    values[:, inputs['scenario'] == 'obstacle_shift', report.INDEX['candidate_success']] = [0., 1.]
    effects = report._paired_effects(values, inputs, 32, 17)
    raw = next(row for row in effects if row['scope'] == 'overall' and row.get('effect') == 'shift_minus_original'
               and row['projection'] == 'off' and row['selection'] == 'all_stress')
    feasible = next(row for row in effects if row['scope'] == 'overall' and row.get('effect') == 'shift_minus_original'
               and row['projection'] == 'off' and row['selection'] == 'both_scenarios_frozen_feasible')
    assert raw['metrics']['candidate_success']['mean'] == -.5
    assert feasible['condition_count'] == 1 and feasible['metrics']['candidate_success']['mean'] == 0.


def test_diversity_successful_pairs_and_empirical_direction():
    condition = _condition(); paths = _paths(condition)
    result = report.diversity_for_paths(paths, np.array([True, True, False, False, False, False, False, False]),
        condition, projection='on')
    assert result['successful_candidates'] == 2 and result['successful_pairs'] == 1
    assert result['both_direction_signs'] and result['position_pair_msd_m2'] > .1
    assert result['uv_pair_msd_m2'] == pytest.approx(result['position_pair_msd_m2'], abs=1e-7)
    single = report.diversity_for_paths(paths, np.array([True, False, False, False, False, False, False, False]),
        condition, projection='off')
    assert single['position_pair_msd_m2'] is None and single['successful_pairs'] == 0
    # Yaw-only changes have no body/tip translation here and do not inflate diversity.
    identical = paths.copy(); identical[1] = identical[0]
    identical[1, :, 3:9] = [0., 1., 0., -1., 0., 0.]
    same_position = report.diversity_for_paths(identical, np.array([True, True, False, False, False, False, False, False]),
        condition, projection='off')
    assert same_position['position_pair_msd_m2'] == 0.


def test_three_seed_mean_std_and_cluster_ci(experiment_fixture):
    _, _, _, inputs = experiment_fixture
    matrix = [{'id': f'model{seed}', 'variant': 'geometry', 'train_size': 500, 'seed': seed} for seed in (17, 29, 43)]
    models = {}
    for row, successes in zip(matrix, ([0., 0., 0., 0.], [1., 1., 1., 1.], [0., 0., 1., 1.])):
        primary = np.zeros((2, len(inputs['case_id']), len(report.FIELDS)))
        primary[:, :, report.INDEX['candidate_success']] = successes
        models[row['id']] = {'primary': primary}
    group = report._seed_groups(models, matrix, inputs, 32, 17)[0]
    cell = report._overall(group['cells'], scenario='original', projection='off')
    assert cell['metrics']['candidate_success']['mean'] == .5
    assert cell['candidate_success_seed_means'] == [0., 1., .5]
    assert cell['candidate_success_seed_std'] == .5


def test_training_identity_audit_counts_paths_conditions_and_maps(tmp_path):
    experiment = tmp_path/'experiment'; dataset = tmp_path/'dataset'; dataset.mkdir()
    # One complete map: six paired detours plus eight singleton easy conditions.
    identifiers = np.array([f'condition-{index}' for index in range(14)] + [f'condition-{index}' for index in range(6)])
    np.savez_compressed(dataset/'prepared_dataset.npz', map_ids=np.repeat('train-map', 20),
                        condition_ids=identifiers, split_codes=np.zeros(20, dtype=np.int8))
    row = {'id': 'geometry_fixture', 'variant': 'geometry', 'train_size': 20, 'seed': 17}
    directory = experiment/'models'/row['id']; directory.mkdir(parents=True)
    best = directory/'best.pt'; best.write_bytes(b'test selected checkpoint')
    subset = {'trajectory_indices': list(range(20)), 'map_count': 1, 'selected_map_ids': ['train-map']}
    _write_json(directory/'training_subset.json', subset)
    config = {'base_sha256': 'frozen-base', 'seed': 17, 'train_size': 20, 'training_steps': 8000,
        'dataset_root': str(dataset), 'prepared_cache_sha256': report._sha(dataset/'prepared_dataset.npz'),
        'subset_sha256': report._hash(subset), 'dataset_freeze_sha256': 'frozen-dataset',
        'adapter_config': {'task_conditioning': 'geometry'}, 'loss_weights': {'contact': 5., 'normal': .5, 'smooth': .1}}
    config['contract_sha256'] = report._hash(config)
    _write_json(directory/'resolved_training_config.json', config)
    status = {'status': 'complete', 'step': 8000, 'selected_checkpoint_sha256': report._sha(best),
              'contract_sha256': config['contract_sha256']}
    _write_json(directory/'run_status.json', status)
    protocol = {'base_sha256': 'frozen-base', 'optimizer_updates': 8000, 'dataset_freeze_sha256': 'frozen-dataset'}
    problems = []
    result = report._model_provenance(experiment, row, protocol, {'adapter_sha256': report._sha(best)}, problems)
    assert problems == []
    assert result['effective_training_counts']['expert_paths'] == 20
    assert result['effective_training_counts']['distinct_task_conditions'] == 14
    assert result['effective_training_counts']['source_maps'] == 1
    # A tampered selection cannot silently report a full paired map.
    subset['trajectory_indices'][-1] = 0
    _write_json(directory/'training_subset.json', subset)
    with pytest.raises(report.IntegrityError, match='subset'):
        report._model_provenance(experiment, row, protocol, {'adapter_sha256': report._sha(best)}, [])


def test_png_scientific_artifact(experiment_fixture):
    experiment, benchmark, output, _ = experiment_fixture
    result = report.build_report(experiment, benchmark, output, bootstrap_replicates=16, make_figures=False)
    cells = result['models']['base']['primary']
    result['seed_groups'] = [{'variant': 'geometry', 'train_size': size, 'cells': cells} for size in (500, 8000)]
    names = report._figures(result, output, '')
    assert names == ['learning_curve.png', 'ablation_8000.png', 'family_kind_difficulty.png']
    from PIL import Image
    for name in names:
        with Image.open(output/name) as image:
            assert image.format == 'PNG' and image.width >= 1500 and image.height >= 600
