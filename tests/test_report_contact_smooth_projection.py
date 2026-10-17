"""Tiny synthetic artifacts only; no inference, decoder fitting, or COAL."""
from __future__ import annotations

import argparse
import base64
import copy
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hnuter_dit import evaluate_contact_smooth_projection as evaluation
from hnuter_dit import report_contact_experiment as parent
from hnuter_dit import report_contact_smooth_projection as report
from hnuter_dit.contact_smooth_projection import smooth_contact_decoder_parameters

torch.set_num_threads(1)


def write(path, value):
    parent._write(path, value)


def contract(path, value, field='contract_sha256'):
    result = value | {field: parent._hash(value), 'created_utc': 'fixture'}
    write(path, result)
    return result


def rewrite_scores(path, rows):
    with gzip.open(path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')


def records(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def commit_meta(supplement, stem, change):
    score = supplement/'evaluation/base/scores'/f'{stem}.json'
    meta = parent._read(score) | change
    write(score, meta)
    write(supplement/'evaluation/base/projected'/f'{stem}.json', meta)


@pytest.fixture
def fixture(tmp_path):
    # Reuse the frozen parent's small synthetic physical-input fixture, not its
    # real experiment. Add the complete extra bindings the supplement requires.
    spec = importlib.util.spec_from_file_location('parent_report_fixture',
        Path(__file__).with_name('test_report_contact_experiment.py'))
    fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    experiment, benchmark, old_output, inputs = fixtures.experiment_fixture.__wrapped__(tmp_path)
    supplement = experiment/'smooth_projection_supplement'
    code = Path(report.__file__).parent
    protocol = parent._read(experiment/'protocol.json')
    protocol.pop('protocol_sha256'); protocol['base_checkpoint'] = str(tmp_path/'base.pt')
    Path(protocol['base_checkpoint']).write_bytes(b'fixture base checkpoint')
    protocol['base_sha256'] = parent._sha(Path(protocol['base_checkpoint']))
    protocol['primary_candidates_per_condition'] = 1
    protocol['multimodal_candidates_per_condition'] = 8
    protocol['projection'] = ['off', 'on']
    protocol = contract(experiment/'protocol.json', protocol, 'protocol_sha256')
    (benchmark/'index.jsonl').write_text('fixture-index\n')
    manifest = parent._read(benchmark/'manifest.json')
    manifest['input_files_sha256']['index.jsonl'] = parent._sha(benchmark/'index.jsonl')
    write(benchmark/'manifest.json', manifest)
    path = experiment/'evaluation/base/contract.json'
    definition = parent._read(path); definition.pop('contract_sha256')
    definition.update(base_sha256=protocol['base_sha256'], benchmark_manifest_sha256=parent._sha(benchmark/'manifest.json'),
        inference_version=evaluation.INFERENCE_VERSION, projection_version=evaluation.PROJECTION_VERSION,
        metrics_version=evaluation.METRICS_VERSION,
        reference_backend={'urdf_sha256': 'mock', 'coal_source_sha256': 'mock'},
        inference_sources={name: parent._sha(code/name) for name in evaluation.INFERENCE_SOURCES},
        evaluation_sources={name: parent._sha(code/name) for name in evaluation.EVALUATION_SOURCES})
    model = contract(path, definition)
    directory = path.parent
    for raw_meta_path in (directory/'raw').glob('*.json'):
        meta = parent._read(raw_meta_path)
        with np.load(raw_meta_path.with_suffix('.npz'), allow_pickle=False) as archive:
            count = len(archive['input_rows'])
        meta.update(contract_sha256=model['contract_sha256'], candidates=count, inference_dtype='float32')
        write(raw_meta_path, meta)
    for score_meta_path in (directory/'scores').glob('*.json'):
        meta = parent._read(score_meta_path); meta['contract_sha256'] = model['contract_sha256']; write(score_meta_path, meta)
    status = parent._read(directory/'scoring_status.json'); status['contract_sha256'] = model['contract_sha256']
    write(directory/'scoring_status.json', status)
    parent_sources = contract(experiment/'pipeline_source_contract.json', dict(experiment=str(experiment), benchmark=str(benchmark),
        protocol_sha256=protocol['protocol_sha256'],
        source_files={str((code/name).resolve()): parent._sha(code/name) for name in
            (*evaluation.INFERENCE_SOURCES, *evaluation.EVALUATION_SOURCES, 'report_contact_experiment.py')},
        input_files={str(experiment/'protocol.json'): parent._sha(experiment/'protocol.json'),
                     protocol['base_checkpoint']: protocol['base_sha256']}))
    prior = parent.build_report(experiment, benchmark, experiment/'reports', bootstrap_replicates=8, make_figures=False)
    # The parent artifact is a complete *mock*, authenticated with a real PNG.
    pixel = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jBw0AAAAASUVORK5CYII=')
    figure = experiment/'reports/fixture.png'; figure.write_bytes(pixel)
    prior['figures'] = {'fixture.png': parent._sha(figure)}
    write(experiment/'reports/report.json', prior)
    write(experiment/'pipeline_status.json', dict(status='complete', report=str(experiment/'reports/report.json'),
        stage_evidence={'report': dict(report_sha256=parent._sha(experiment/'reports/report.json'),
            markdown_sha256=parent._sha(experiment/'reports/research_report_zh.md'), figures=prior['figures'])}))
    multi = parent._multimodal_maps(inputs, 1)
    maps = sorted(set(map(str, inputs['map_id'])))
    path_count = sum(len(evaluation.requests_for_map(inputs, map_id, multi)[0]) for map_id in maps)
    params = smooth_contact_decoder_parameters()
    smooth = contract(supplement/'protocol.json', dict(schema_version=evaluation.SUPPLEMENT_VERSION,
        parent_experiment=str(experiment), benchmark=str(benchmark), parent_protocol_sha256=protocol['protocol_sha256'],
        parent_protocol_file_sha256=parent._sha(experiment/'protocol.json'), parent_source_contract_sha256=parent_sources['contract_sha256'],
        parent_source_contract_file_sha256=parent._sha(experiment/'pipeline_source_contract.json'),
        benchmark_manifest_sha256=parent._sha(benchmark/'manifest.json'),
        benchmark_input_files_sha256={name: manifest['input_files_sha256'][name] for name in sorted({'runtime_inputs.npz', 'index.jsonl', *map(str, inputs['input_shard'])})},
        planned_models=['base'], condition_count=2, source_map_count=2, multimodal_source_maps=sorted(multi),
        multimodal_maps_per_family=1, multimodal_subset_conditions=1, primary_candidates=1, multimodal_candidates=8,
        paths_per_model=path_count, score_rows_per_model=path_count, projection='on_smooth',
        decoder_version=evaluation.SMOOTH_PROJECTION_VERSION, decoder_parameters=params, decoder_parameters_sha256=parent._hash(params),
        metrics_version=evaluation.METRICS_VERSION, output_schema=evaluation.OUTPUT_SCHEMA,
        output_schema_sha256=parent._hash(evaluation.OUTPUT_SCHEMA),
        input_policy='same_parent_raw_paths_and_whitelisted_frozen_original_shift_conditions',
        failure_policy='retain_every_request; decoder_errors_count_as_failed_candidates',
        parameter_policy='fixed_before_pilot; no_benchmark_or_pilot_tuning',
        interpolation=evaluation.INTRINSIC_INTERPOLATION, oracle_access=False, generation=False, training=False), 'protocol_sha256')
    sources = contract(supplement/'source_contract.json', dict(schema_version=evaluation.SOURCE_CONTRACT_VERSION,
        protocol_sha256=smooth['protocol_sha256'], parent_source_contract_sha256=parent_sources['contract_sha256'],
        source_files={str((code/name).resolve()): parent._sha(code/name) for name in evaluation.SUPPLEMENT_SOURCES},
        frozen_dependency_sources={name: parent._sha(code/name) for name in (*evaluation.INFERENCE_SOURCES, *evaluation.EVALUATION_SOURCES)},
        decoder_parameters_sha256=smooth['decoder_parameters_sha256'], metrics_version=evaluation.METRICS_VERSION,
        output_schema_sha256=smooth['output_schema_sha256']))
    contract(supplement/'pipeline_source_contract.json', dict(schema_version='contact_smooth_pipeline_sources_v001',
        experiment=str(experiment), benchmark=str(benchmark), supplement=str(supplement),
        parent_source_contract_sha256=parent_sources['contract_sha256'], decoder_parameters=params,
        source_files=parent_sources['source_files'] | {str((code/name).resolve()): parent._sha(code/name) for name in
            ('contact_smooth_pipeline.py', 'contact_smooth_projection.py', 'evaluate_contact_smooth_projection.py', 'report_contact_smooth_projection.py')},
        input_files=parent_sources['input_files'] | {str(experiment/'pipeline_source_contract.json'): parent._sha(experiment/'pipeline_source_contract.json')},
        stage_policy='wait_for_strict_parent_complete; score_all22_same_raw; strict_supplement_report'))
    new_dir = supplement/'evaluation/base'
    new_contract = contract(new_dir/'contract.json', dict(schema_version=evaluation.SUPPLEMENT_VERSION, model_id='base',
        protocol_sha256=smooth['protocol_sha256'], source_contract_sha256=sources['contract_sha256'],
        parent_contract_sha256=model['contract_sha256'], parent_contract_file_sha256=parent._sha(directory/'contract.json'),
        checkpoint_binding={'base_sha256': protocol['base_sha256'], 'adapter_sha256': None},
        benchmark_manifest_sha256=smooth['benchmark_manifest_sha256'], decoder_version=evaluation.SMOOTH_PROJECTION_VERSION,
        decoder_parameters_sha256=smooth['decoder_parameters_sha256'], metrics_version=evaluation.METRICS_VERSION,
        output_schema_sha256=smooth['output_schema_sha256'], reference_backend=model['reference_backend'], score_rows_per_model=path_count))
    args = argparse.Namespace(experiment=experiment, benchmark=benchmark, output=supplement)
    for map_id in maps:
        raw, rows, candidates, raw_meta = evaluation._raw_requests(args, new_contract, inputs, map_id, multi)
        _, input_hashes = evaluation._prepare_conditions(args, inputs, rows, manifest, map_id)
        stem = map_id.rsplit('/', 1)[-1]
        world = raw.astype(np.float64)
        t = np.linspace(0., 1., 128)
        intrinsic = np.stack([np.column_stack((-1+2*t, (.4 if candidate%2 else -.4)*np.sin(np.pi*t), np.zeros(128))) for candidate in candidates])
        succeeded = np.ones(len(rows), dtype=bool)
        if map_id in multi:
            succeeded[(candidates == 7) | ((candidates == 0) & (inputs['scenario'][rows] == 'original'))] = False
        world[~succeeded] = np.nan; intrinsic[~succeeded] = np.nan
        output = []
        for offset, (row, candidate) in enumerate(zip(rows, candidates)):
            success = bool(succeeded[offset] and inputs['feasible_mask'][row] and candidate < 2)
            if succeeded[offset]:
                metrics = {name: True for name in parent.SUCCESS_FIELDS}
                metrics.update({name: 0. for name in parent.CONTINUOUS_FIELDS})
                metrics.update(schema_version=evaluation.METRICS_VERSION, candidate_success=success,
                    numeric_candidate_success=success, physical_collision_free=True, minimum_clearance_m=.5,
                    path_length_m=2.4, path_efficiency=1.2, maximum_curvature_inv_m=2., failures=[] if success else ['mock_quality'],
                    projection_succeeded=True, projection_failed=False, projection_fit_success=True,
                    projection_geometry_valid=True, projection_before={},
                    projection_after={'solver_converged': True, 'solver_sweeps': 0, 'solver_projected_gradient_max': 0., 'fit_mse_normalized': .002})
            else:
                metrics = evaluation._failure_metrics(RuntimeError('mock decoder exception'), 'projection_error')
                metrics.update(projection_succeeded=False, projection_failed=True, projection_fit_success=False,
                    projection_geometry_valid=False, projection_before={}, projection_after={})
            output.append(evaluation._common(new_contract, inputs, int(row), int(candidate), map_id, multi) |
                          {'schema_version': evaluation.SCORE_SCHEMA_VERSION, 'metrics': metrics})
        projected = new_dir/'projected'/f'{stem}.npz'; projected.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(projected, world_pose9=world, intrinsic_states=intrinsic, input_rows=rows,
                            candidate_indices=candidates, projection_succeeded=succeeded, fit_success=succeeded, geometry_valid=succeeded)
        score = new_dir/'scores'/f'{stem}.jsonl.gz'; score.parent.mkdir(parents=True, exist_ok=True)
        rewrite_scores(score, output)
        meta = dict(schema_version=evaluation.SUPPLEMENT_VERSION, contract_sha256=new_contract['contract_sha256'],
            parent_contract_sha256=new_contract['parent_contract_sha256'], raw_sha256=raw_meta['raw_sha256'],
            raw_meta_sha256=raw_meta['raw_meta_sha256'], map_id=map_id, input_files_sha256=input_hashes,
            request_sha256=parent._hash(dict(input_rows=rows.tolist(), candidate_indices=candidates.tolist())),
            input_sha256=[str(inputs['input_sha256'][row]) for row in rows], protocol_sha256=new_contract['protocol_sha256'],
            source_contract_sha256=new_contract['source_contract_sha256'], decoder_version=new_contract['decoder_version'],
            decoder_parameters_sha256=new_contract['decoder_parameters_sha256'], metrics_version=evaluation.METRICS_VERSION,
            output_schema_sha256=new_contract['output_schema_sha256'], scores_sha256=parent._sha(score), projected_sha256=parent._sha(projected),
            score_rows=len(rows), projected_rows=len(rows), projection_failed_rows=int((~succeeded).sum()),
            fit_success_rows=int(succeeded.sum()), geometry_valid_rows=int(succeeded.sum()))
        write(projected.with_suffix('.json'), meta); write(score.with_name(f'{stem}.json'), meta)
    write(new_dir/'scoring_status.json', dict(status='complete', complete_maps=2, total_maps=2, score_rows=path_count,
        planned_total_maps=2, expected_score_rows=path_count, planned_score_rows=path_count,
        pilot_limit_maps=0, contract_sha256=new_contract['contract_sha256']))
    return experiment, benchmark, supplement, inputs


def preview(fixture, **kwargs):
    experiment, benchmark, supplement, _ = fixture
    return report.build_report(experiment, benchmark, supplement, allow_partial=True,
                               bootstrap_replicates=16, make_figures=False, **kwargs)


def test_small_complete_mock_can_only_produce_explicit_preview_and_failure_denominators(fixture):
    experiment, benchmark, supplement, _ = fixture
    with pytest.raises(report.IncompleteExperiment, match='22|shortfalls'):
        report.build_report(experiment, benchmark, supplement, make_figures=False)
    assert not (supplement/'reports/report.json').exists()
    result = preview(fixture)
    assert result['status'] == 'incomplete_preview' and result['projections'] == ['off', 'on', 'on_smooth']
    assert result['oracle_access'] is False and result['decoder_replayed'] is False
    cell = parent._overall(result['models']['base']['primary'], scenario='original', projection='on_smooth')
    assert cell['metrics']['candidate_success']['observations'] == 2
    assert cell['metrics']['candidate_success']['mean'] == .5
    assert cell['metrics']['projection_fit_success']['mean'] == .5
    assert (supplement/'reports/report_preview.json').is_file()
    text = (supplement/'reports/research_report_zh_preview.md').read_text()
    assert text.startswith('# 未完成预览') and '不能严格声称同伦' in text and '不是密集边最小距离' in text


def test_missing_shard_preserves_planned_denominator_and_never_emits_complete(fixture):
    _, _, supplement, _ = fixture
    next((supplement/'evaluation/base/projected').glob('*.npz')).unlink()
    result = preview(fixture)
    assert result['status'] == 'incomplete_preview'
    assert any(problem.get('missing_score_rows', 0) > 0 for problem in result['shortfalls'])
    assert not (supplement/'reports/report.json').exists()


@pytest.mark.parametrize('relative', ['', 'nested_preview'])
def test_frozen_parent_report_directory_cannot_be_used_as_supplement_output(fixture, relative):
    experiment, benchmark, supplement, _ = fixture
    parent_directory = Path(parent._read(experiment/'pipeline_status.json')['report']).parent
    originals = {path: parent._sha(path) for path in parent_directory.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='frozen parent report directory'):
        report.build_report(experiment, benchmark, supplement, output=parent_directory/relative,
                            allow_partial=True, bootstrap_replicates=16, make_figures=False)
    assert {path: parent._sha(path) for path in parent_directory.rglob('*') if path.is_file()} == originals


def test_saved_diversity_uses_actual_positions_and_none_without_pairs():
    world = np.zeros((8, 128, 9)); intrinsic = np.zeros((8, 128, 3))
    world[1, :, 0] = 2.; intrinsic[1, :, 1] = 3.
    result = report.saved_pair_diversity(world, intrinsic, np.array([True, True]+[False]*6))
    assert result['position_pair_msd_m2'] == 4. and result['uv_pair_msd_m2'] == 9.
    single = report.saved_pair_diversity(world, intrinsic, np.array([True]+[False]*7))
    assert single['successful_pairs'] == 0 and single['position_pair_msd_m2'] is None
    assert single['uv_pair_msd_m2'] is None


def test_k8_none_pair_is_not_imputed_and_any_success_keeps_the_condition(fixture):
    result = preview(fixture)
    multiple = result['models']['base']['multimodal']
    pair = parent._overall(multiple['diversity'], scenario='original', projection='on_smooth')
    assert pair['condition_count'] == 1
    assert pair['metrics']['position_pair_msd_m2']['observations'] == 0
    assert pair['metrics']['position_pair_msd_m2']['mean'] is None
    any_cell = parent._overall(multiple['any_at_8'], scenario='original', projection='on_smooth')
    assert any_cell['metrics']['any_at_8']['observations'] == 1
    assert any_cell['metrics']['any_at_8']['mean'] == 1.


def test_projection_label_mismatch_is_rejected_even_after_payload_hash_is_updated(fixture):
    _, _, supplement, _ = fixture
    score = next((supplement/'evaluation/base/scores').glob('*.jsonl.gz'))
    data = records(score); data[0]['projection'] = 'on'; rewrite_scores(score, data)
    commit_meta(supplement, score.name.split('.')[0], {'scores_sha256': parent._sha(score)})
    with pytest.raises(report.IntegrityError, match='identity|projection'):
        preview(fixture)


def test_projected_endpoint_corruption_is_rejected_after_hash_update(fixture):
    _, _, supplement, _ = fixture
    path = next((supplement/'evaluation/base/projected').glob('*.npz'))
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    index = np.flatnonzero(data['projection_succeeded'])[0]
    data['world_pose9'][index, 0, 0] += .01
    np.savez_compressed(path, **data)
    commit_meta(supplement, path.stem, {'projected_sha256': parent._sha(path)})
    with pytest.raises(report.IntegrityError, match='endpoints'):
        preview(fixture)


@pytest.mark.parametrize('artifact', ['projected', 'parent_figure', 'parent_raw'])
def test_payload_hash_corruption_always_fails(fixture, artifact):
    experiment, _, supplement, _ = fixture
    if artifact == 'projected':
        path = next((supplement/'evaluation/base/projected').glob('*.npz'))
    elif artifact == 'parent_raw':
        path = next((experiment/'evaluation/base/raw').glob('*.npz'))
    else:
        path = experiment/'reports/fixture.png'
    path.write_bytes(path.read_bytes()+b'corrupted')
    with pytest.raises(report.IntegrityError, match='checksum'):
        preview(fixture)


def test_missing_driver_is_a_preview_shortfall_and_not_an_approval_bypass(fixture):
    _, _, supplement, _ = fixture
    (supplement/'pipeline_source_contract.json').unlink()
    result = preview(fixture)
    assert any(problem['stage'] == 'driver' for problem in result['shortfalls'])
    assert result['status'] == 'incomplete_preview'


def test_report_source_binding_is_required(fixture):
    _, _, supplement, _ = fixture
    path = supplement/'pipeline_source_contract.json'
    value = parent._read(path); value.pop('contract_sha256'); value.pop('created_utc')
    value['source_files'].pop(str(Path(report.__file__).resolve()))
    contract(path, value)
    with pytest.raises(report.IntegrityError, match='does not bind'):
        preview(fixture)


def test_generalized_projection_labels_and_shift_feasibility_pairing(fixture):
    _, _, _, inputs = fixture
    values = np.zeros((3, len(inputs['case_id']), len(report.FIELDS)))
    values[1, :, report.INDEX['candidate_success']] = 1.
    values[2, inputs['scenario'] == 'original', report.INDEX['candidate_success']] = 1.
    values[2, inputs['scenario'] == 'obstacle_shift', report.INDEX['candidate_success']] = [0., 1.]
    effects = report._paired(values, inputs, 16, 20261011)
    delta = next(cell for cell in effects if cell['scope'] == 'overall' and cell['effect'] == 'on_smooth_minus_on'
                 and cell['scenario'] == 'obstacle_shift' and cell['selection'] == 'all_stress')
    assert delta['metrics']['candidate_success']['mean'] == -.5
    shifted = next(cell for cell in effects if cell['scope'] == 'overall' and cell['effect'] == 'shift_minus_original'
                   and cell['projection'] == 'on_smooth' and cell['selection'] == 'both_scenarios_frozen_feasible')
    assert shifted['condition_count'] == 1 and shifted['metrics']['candidate_success']['mean'] == 0.
    with pytest.raises(ValueError, match='labels'):
        report._cells(values[:2], inputs, 16, 20261011)


def test_three_seed_statistics_retain_individual_seeds(fixture):
    _, _, _, inputs = fixture
    matrix = [dict(id=f'fixture{seed}', variant='geometry', train_size=500, seed=seed) for seed in (17, 29, 43)]
    models = {}
    for row, value in zip(matrix, (0., 1., .5)):
        models[row['id']] = dict(primary=np.full((3, len(inputs['case_id']), len(report.FIELDS)), value),
            multi_rows=np.arange(len(inputs['case_id'])), any_values=np.full((3, len(inputs['case_id']), 3), value),
            diversity_values=np.full((3, len(inputs['case_id']), 5), value))
    group = report._seed_groups(models, matrix, inputs, 16, 20261011)[0]
    cell = parent._overall(group['cells'], scenario='original', projection='on_smooth')
    assert cell['metrics']['candidate_success']['seed_means'] == [0., 1., .5]
    assert cell['metrics']['candidate_success']['seed_std'] == .5
    assert group['observed_seed_count'] == 3


def test_scientific_pngs_are_generated_and_authenticated_for_preview(fixture):
    experiment, benchmark, supplement, _ = fixture
    result = report.build_report(experiment, benchmark, supplement, allow_partial=True,
                                 bootstrap_replicates=16, make_figures=True)
    assert len(result['figures']) == 5
    for name, digest in result['figures'].items():
        path = supplement/'reports'/name
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert parent._sha(path) == digest
        from PIL import Image
        with Image.open(path) as image:
            assert image.width >= 1800 and image.height >= 700


def test_adapter_base_and_negative_controls_are_paired_for_all_three_decoders(fixture):
    _, _, _, inputs = fixture
    matrix, models = [], {'base': {'primary': np.zeros((3, len(inputs['case_id']), len(report.FIELDS)))}}
    for seed in (17, 29, 43):
        for variant, value in (('geometry', .8), ('no_contact_loss', .4), ('no_geometry_tokens', .2)):
            identifier = f'{variant}_{seed}'
            matrix.append(dict(id=identifier, variant=variant, train_size=8000, seed=seed))
            models[identifier] = {'primary': np.full((3, len(inputs['case_id']), len(report.FIELDS)), value)}
    pairs, groups = report._adapter_controls(models, matrix, inputs, 16, 20261011)
    assert len(pairs) == 9 and len(groups) == 3
    group = next(group for group in groups if group['control_variant'] == 'no_contact_loss')
    assert group['seeds'] == [17, 29, 43]
    for projection in report.PROJECTIONS:
        metric = parent._overall(group['cells'], scenario='original', projection=projection)['metrics']['candidate_success']
        assert metric['mean'] == pytest.approx(.4)
        assert metric['seed_means'] == pytest.approx([.4, .4, .4])
        assert metric['ci95'] == pytest.approx([.4, .4])


@pytest.mark.parametrize('contrast', ['decoder', 'shift', 'difference_in_differences'])
def test_missing_continuous_diagnostics_are_paired_within_seed_before_seed_average(fixture, contrast):
    _, _, _, inputs = fixture
    arrays = [np.zeros((3, len(inputs['case_id']), len(report.FIELDS))) for _ in range(3)]
    field = report.INDEX['contact_error_max_m']
    original, shifted = inputs['scenario'] == 'original', inputs['scenario'] == 'obstacle_shift'
    for array, before, after in zip(arrays, (.1, .6, .3), (np.nan, .4, .7)):
        if contrast == 'decoder':
            array[1, :, field], array[2, :, field] = before, after
        else:
            array[2, original, field], array[2, shifted, field] = before, after
            if contrast == 'difference_in_differences':
                array[1, shifted, field] = .05
    cells = report._paired_group(arrays, inputs, 16, 20261011)
    if contrast == 'decoder':
        selected = next(cell for cell in cells if cell['scope'] == 'overall' and cell['effect'] == 'on_smooth_minus_on'
                        and cell['scenario'] == 'original' and cell['selection'] == 'all_stress')
    elif contrast == 'shift':
        selected = next(cell for cell in cells if cell['scope'] == 'overall' and cell['effect'] == 'shift_minus_original'
                        and cell['projection'] == 'on_smooth' and cell['selection'] == 'all_stress')
    else:
        selected = next(cell for cell in cells if cell['scope'] == 'overall' and cell['effect'] == 'smooth_shift_effect_minus_on_shift_effect'
                        and cell['selection'] == 'all_stress')
    expected = .05 if contrast == 'difference_in_differences' else .1
    metric = selected['metrics']['contact_error_max_m']
    assert metric['mean'] == pytest.approx(expected)
    assert metric['seed_means'][0] is None
    assert metric['seed_means'][1:] == pytest.approx([-.25, .35] if contrast == 'difference_in_differences' else [-.2, .4])


def test_independent_evaluation_error_does_not_use_decoder_geometry_as_metric_observations(fixture):
    experiment, _, _, inputs = fixture
    relative = 'evaluation/base/scores/legal_evaluation_error.jsonl.gz'
    path = experiment/relative
    metrics = dict(candidate_success=False, numeric_candidate_success=False, physical_collision_free=False,
        evaluation_error='mock COAL error before metric fields were written',
        projection_after={'contact_error_max_m': 0., 'normal_error_max_rad': 0.,
                          'psi_total_variation_rad': .1, 'solver_sweeps': 4})
    rewrite_scores(path, [dict(case_id=str(inputs['case_id'][0]), candidate_index=0, projection='on', metrics=metrics)])
    baseline = dict(primary=np.full((2, len(inputs['case_id']), len(parent.FIELDS)), np.nan),
        multiple=np.full((2, 0, 8, len(parent.FIELDS)), np.nan), multi_rows=np.array([], dtype=int),
        artifact_sha256={relative: parent._sha(path)})
    for name in ('candidate_success', 'numeric_candidate_success', 'physical_collision_free'):
        baseline['primary'][1, 0, parent.INDEX[name]] = 0.
    values, _ = report._parent_values(experiment, baseline, inputs)
    assert np.isnan(values[1, 0, report.INDEX['contact_error_max_m']])
    assert np.isnan(values[1, 0, report.INDEX['normal_error_max_rad']])
    assert np.isnan(values[1, 0, report.INDEX['psi_total_variation_rad']])
    assert values[1, 0, report.INDEX['solver_sweeps']] == 4.
    stats = parent.cluster_statistics(values[1], inputs['map_id'], names=report.FIELDS,
                                      replicates=16, strata=inputs['family'])
    assert stats['metrics']['candidate_success']['observations'] == 1
    assert stats['metrics']['candidate_success']['mean'] == 0.
    assert stats['metrics']['contact_error_max_m']['observations'] == 0
