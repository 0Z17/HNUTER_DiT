"""Synthetic scalar fixtures only: no actual complete research, GPU, or COAL.

The writer test deliberately substitutes an already-audited synthetic payload
and emits SYNTHETIC FIXTURE / synthetic_test_only. Production has no bypass.
"""
from __future__ import annotations

import copy
import csv
import gzip
import json

import numpy as np
import pytest
import torch

from hnuter_dit import report_contact_ablation_appendix as appendix
from hnuter_dit import report_contact_experiment as parent
from hnuter_dit import report_contact_smooth_projection as smooth

torch.set_num_threads(1)


def metric(*, mean=.5, observations=2, grouped=False):
    result = dict(observations=observations, mean=mean, std=.2 if observations else None,
        median=mean, p05=mean, p95=mean, ci95=[max(-1., mean-.1), min(1., mean+.1)] if observations else [None, None])
    if grouped:
        means = [mean-.1, mean, mean+.1] if observations else [None, None, None]
        result.update(seed_means=means, seed_std=float(np.std(means, ddof=1)) if observations else None)
    return result


def cells(expected, names, *, grouped=False):
    result = []
    for identity, (conditions, maps) in expected.items():
        row = dict(identity) | dict(condition_count=conditions, planned_source_map_count=maps,
            observed_condition_count=conditions, source_map_count=maps, bootstrap_replicates=1000,
            metrics={name: metric(observations=conditions, mean=.5 if conditions else None, grouped=grouped) for name in names})
        result.append(row)
    return result


@pytest.fixture
def synthetic(tmp_path):
    experiment = tmp_path/'synthetic_experiment'; benchmark = tmp_path/'synthetic_benchmark'
    supplement = experiment/'smooth_projection_supplement'
    parent_path, new_path = experiment/'reports/report.json', supplement/'reports/report.json'
    for path in (parent_path, new_path):
        path.parent.mkdir(parents=True); path.write_text('{"synthetic_fixture":true}\n')
    inputs = dict(condition_id=np.array(['a', 'b', 'a', 'b']),
        scenario=np.array(['original', 'original', 'obstacle_shift', 'obstacle_shift']),
        case_id=np.array(['a/original', 'b/original', 'a/obstacle_shift', 'b/obstacle_shift']),
        map_id=np.array(['map0', 'map1', 'map0', 'map1']), family=np.array(['fixture']*4),
        kind=np.array(['plane', 'cylinder', 'plane', 'cylinder']), difficulty=np.array(['easy', 'detour', 'easy', 'detour']),
        feasible_mask=np.array([True, False, True, False]))
    matrix = [dict(id=f'geometry_n{size:04d}_seed{seed}', variant='geometry', train_size=size, seed=seed)
              for size in appendix.BUDGETS for seed in appendix.SEEDS]
    matrix += [dict(id=f'{variant}_n8000_seed{seed}', variant=variant, train_size=8000, seed=seed)
               for seed in appendix.SEEDS for variant in ('no_contact_loss', 'no_geometry_tokens')]
    protocol = dict(variants=matrix, sampling_steps=25, latent_seed=20261011, base_sha256='b'*64)
    matrix = appendix._matrix(protocol)
    expected, pairs = appendix._expected_cells(inputs), appendix._expected_paired(inputs)
    primary = cells(expected, smooth.FIELDS); paired_cells = cells(pairs, smooth.PAIRED_FIELDS)
    multiple = dict(any_at_8=cells(expected, smooth.ANY_FIELDS), diversity=cells(expected, smooth.DIVERSITY_FIELDS))
    grouped_primary = cells(expected, smooth.FIELDS, grouped=True)
    grouped_pairs = cells(pairs, smooth.PAIRED_FIELDS, grouped=True)
    grouped_multiple = dict(any_at_8=cells(expected, smooth.ANY_FIELDS, grouped=True), diversity=cells(expected, smooth.DIVERSITY_FIELDS, grouped=True))
    models = {}
    for row in matrix:
        training = {key: row[key] for key in ('variant', 'train_size', 'seed')}
        if row['id'] != 'base':
            training['training_status'] = dict(step=8000, successful_sample_presentations=512000,
                best_step=7800, best_validation_score=.12345, selected_checkpoint_sha256='c'*64)
        models[row['id']] = dict(training=training, parent_contract={}, contract={}, complete=True,
            primary=primary, paired_effects=paired_cells, multimodal=multiple,
            failure_counts=[dict(reason='synthetic failure', count=1)], evaluation_errors=[])
    groups = []
    for variant, size in sorted({(row['variant'], row['train_size']) for row in matrix[1:]}):
        groups.append(dict(variant=variant, train_size=size, seeds=list(appendix.SEEDS),
            cells=grouped_primary, paired_effects=grouped_pairs, multimodal=grouped_multiple))
    controls = []
    for size, variant in [(size, 'base') for size in appendix.BUDGETS]+[(8000, variant) for variant in ('no_contact_loss', 'no_geometry_tokens')]:
        controls.append(dict(train_size=size, control_variant=variant, seeds=list(appendix.SEEDS), effect='geometry_adapter_minus_control',
            cells=grouped_primary))
    paired = []
    for seed in appendix.SEEDS:
        paired.append(dict(model_id=f'geometry_n8000_seed{seed}', control_id=f'no_contact_loss_n8000_seed{seed}',
            train_size=8000, seed=seed, cells=primary))
    report = dict(models=models, seed_groups=groups, paired_adapter_controls=paired,
                  paired_adapter_control_groups=controls, benchmark_manifest_sha256='d'*64, figures={})
    audit = appendix._Audit(); audit.check(parent_path); audit.check(new_path)
    return dict(experiment=experiment, benchmark=benchmark, supplement=supplement,
        parent_report_path=parent_path, supplement_report_path=new_path, protocol=protocol, inputs=inputs,
        matrix=matrix, parent_report=dict(figures={}), supplement_report=report,
        verified_input_files_sha256=audit.hashes, _audit=audit, synthetic_fixture=True)


def test_exact_five_budget_three_seed_matrix_includes_both_controls(synthetic):
    rows = appendix._matrix(synthetic['protocol'])
    assert len(rows) == 22 and rows[0]['id'] == 'base'
    bad = copy.deepcopy(synthetic['protocol']); bad['variants'][0]['seed'] = 99
    with pytest.raises(appendix.IntegrityError, match='matrix differs'):
        appendix._matrix(bad)


@pytest.mark.parametrize('value', [dict(status='incomplete_preview', shortfalls=[]),
    dict(status='complete', shortfalls=[{'missing': 1}]), dict(status='complete', shortfalls=[], synthetic_fixture=True)])
def test_no_partial_or_synthetic_artifact_can_pass_complete(value):
    with pytest.raises((appendix.IntegrityError, appendix.IncompleteExperiment)):
        appendix._complete(value, 'synthetic test artifact')


def test_expected_shift_pairs_use_both_feasibility_masks(synthetic):
    inputs = synthetic['inputs']; inputs['feasible_mask'][2] = False
    expected = appendix._expected_paired(inputs)
    selected = [counts for identity, counts in expected.items() if dict(identity).get('scope') == 'overall'
                and dict(identity).get('selection') == 'both_scenarios_frozen_feasible']
    assert selected and all(conditions == 0 for conditions, _ in selected)


def test_full_failure_denominator_and_three_seed_sd_are_validated(synthetic):
    expected = appendix._expected_cells(synthetic['inputs'])
    rows = cells(expected, smooth.FIELDS, grouped=True)
    appendix._validate_cells(rows, expected, smooth.FIELDS, seed_group=True, full_metrics=('candidate_success',))
    first = next(row for row in rows if row['condition_count'] == 2)
    first['metrics']['candidate_success']['observations'] = 1
    with pytest.raises(appendix.IntegrityError, match='full denominator'):
        appendix._validate_cells(rows, expected, smooth.FIELDS, seed_group=True, full_metrics=('candidate_success',))
    first['metrics']['candidate_success']['observations'] = 2
    first['metrics']['candidate_success']['seed_std'] = .4
    with pytest.raises(appendix.IntegrityError, match='seed SD'):
        appendix._validate_cells(rows, expected, smooth.FIELDS, seed_group=True)


def test_missing_strata_or_decoder_cell_fails_even_when_retained_cells_are_valid(synthetic):
    expected = appendix._expected_cells(synthetic['inputs'])
    rows = cells(expected, smooth.FIELDS)
    with pytest.raises(appendix.IntegrityError, match='omit planned'):
        appendix._validate_cells(rows[:-1], expected, smooth.FIELDS)
    rows[0]['projection'] = 'smoothed_unbound'
    with pytest.raises(appendix.IntegrityError, match='unexpected'):
        appendix._validate_cells(rows, expected, smooth.FIELDS)


def test_missing_diagnostics_remain_null_without_removing_candidate_denominator(synthetic):
    expected = appendix._expected_cells(synthetic['inputs'])
    rows = cells(expected, smooth.FIELDS, grouped=True)
    for row in rows:
        row['metrics']['contact_error_max_m'] = metric(observations=0, mean=None, grouped=True)
    appendix._validate_cells(rows, expected, smooth.FIELDS, seed_group=True, full_metrics=('candidate_success',))
    synthetic['supplement_report']['seed_groups'][0]['cells'] = rows
    exported = list(appendix._rows(synthetic['supplement_report']))
    missing = [row for row in exported if row['aggregation'] == 'three_seed_mean' and row['metric'] == 'contact_error_max_m'
               and row.get('train_size') == synthetic['supplement_report']['seed_groups'][0]['train_size']
               and row.get('variant') == synthetic['supplement_report']['seed_groups'][0]['variant'] and row['table'] == 'K1']
    assert missing and all(row['mean'] is None and row['ci95_low'] is None for row in missing)
    assert appendix._metric(rows[0]['metrics']['contact_error_max_m']) == '缺诊断（n=0）'


def test_map_ci_and_between_seed_sd_are_separate_in_rendering():
    value = metric(grouped=True); value['seed_std'] = .25; value['ci95'] = [.4, .6]
    assert appendix._metric(value) == '50.00 ± 25.00% [40.00, 60.00]'
    assert appendix._metric(value, difference=True) == '+50.00 ± 25.00 pp [40.00, 60.00]'


def test_artifact_coverage_and_bytes_are_both_required(tmp_path):
    path = tmp_path/'candidate.npz'; path.write_bytes(b'synthetic payload')
    audit = appendix._Audit(); digest = parent._sha(path)
    with pytest.raises(appendix.IntegrityError, match='omits or adds'):
        appendix._artifacts(audit, tmp_path, {'candidate.npz': digest}, {'candidate.npz', 'missing.meta'})
    with pytest.raises(appendix.IntegrityError, match='checksum'):
        audit.check(path, '0'*64)
    audit = appendix._Audit(); audit.check(path, digest); path.write_bytes(b'mutated synthetic')
    with pytest.raises(appendix.IntegrityError, match='changed'):
        audit.unchanged()


def test_nonfinite_json_and_oracle_paths_are_rejected(tmp_path):
    path = tmp_path/'bad.json'; path.write_text('{"mean":NaN}')
    with pytest.raises(appendix.IntegrityError, match='non-finite'):
        appendix._read(path)
    path = tmp_path/'oracle'/'private.json'; path.parent.mkdir(); path.write_text('{}')
    with pytest.raises(appendix.IntegrityError, match='oracle'):
        appendix._Audit().check(path)


def test_missing_or_changed_scientific_figure_and_markdown_are_rejected(tmp_path):
    renderer = lambda *args: 'SYNTHETIC FIXTURE\n'
    (tmp_path/'research_report_zh.md').write_text(renderer())
    with pytest.raises(appendix.IntegrityError, match='no scientific PNGs'):
        appendix._report_files(appendix._Audit(), tmp_path, dict(figures={}), renderer)
    image = tmp_path/'fixture.png'; image.write_bytes(b'synthetic PNG checksum fixture')
    record = dict(figures={'fixture.png': parent._sha(image)})
    appendix._report_files(appendix._Audit(), tmp_path, record, renderer)
    image.unlink()
    with pytest.raises(appendix.IntegrityError, match='absent'):
        appendix._report_files(appendix._Audit(), tmp_path, record, renderer)
    image.write_bytes(b'synthetic PNG checksum fixture')
    (tmp_path/'research_report_zh.md').write_text('tampered')
    with pytest.raises(appendix.IntegrityError, match='Markdown differs'):
        appendix._report_files(appendix._Audit(), tmp_path, record, renderer)


def test_missing_training_checkpoint_and_unfinished_budget_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(appendix, '_training_inputs', lambda *args: None)
    row = dict(id='geometry_n0500_seed17', variant='geometry', train_size=500, seed=17)
    directory = tmp_path/'models'/row['id']; directory.mkdir(parents=True)
    config = dict(base_sha256='b'*64, seed=17, train_size=500, training_steps=8000, batch_size=64,
                  validation_count=1000, reverse_augmentation=True)
    config['contract_sha256'] = parent._hash(config)
    (directory/'resolved_training_config.json').write_text(parent._json(config))
    (directory/'best.pt').write_bytes(b'synthetic selected checkpoint')
    (directory/'training_subset.json').write_text('{}')
    status = dict(status='complete', step=8000, successful_sample_presentations=512000, frozen_base_unchanged=True,
        planned_steps=8000, train_experts=500, validation_experts=1000,
        contract_sha256=config['contract_sha256'], best_step=7900, best_validation_score=.2,
        selected_checkpoint_sha256=parent._sha(directory/'best.pt'))
    (directory/'run_status.json').write_text(parent._json(status))
    counts = dict(expert_paths=500, source_maps=25, distinct_task_conditions=350, conditions_per_source_map=14,
        expert_paths_per_source_map=20, prepared_identity_audit=True, subset_file_sha256=parent._sha(directory/'training_subset.json'))
    training = {key: row[key] for key in ('variant', 'train_size', 'seed')} | dict(training_config=config, training_status=status,
        effective_training_counts=counts, config_sha256=parent._sha(directory/'resolved_training_config.json'))
    model = dict(training=training, parent_contract=dict(base_sha256='b'*64, adapter_sha256=status['selected_checkpoint_sha256']),
        contract=dict(checkpoint_binding=dict(training_config_sha256=training['config_sha256'], run_status_sha256=parent._sha(directory/'run_status.json'))))
    appendix._audit_training(appendix._Audit(), tmp_path, row, model, dict(base_sha256='b'*64))
    (directory/'best.pt').unlink()
    with pytest.raises(appendix.IntegrityError, match='absent'):
        appendix._audit_training(appendix._Audit(), tmp_path, row, model, dict(base_sha256='b'*64))
    status['step'] = 7999; (directory/'run_status.json').write_text(parent._json(status))
    with pytest.raises(appendix.IntegrityError, match='completion'):
        appendix._audit_training(appendix._Audit(), tmp_path, row, model, dict(base_sha256='b'*64))


def test_training_freeze_hashes_prepared_bytes_and_current_auxiliary_metadata(tmp_path):
    root = tmp_path/'synthetic_training_data'; root.mkdir()
    prepared = root/'prepared_dataset.npz'; prepared.write_bytes(b'synthetic prepared bytes, no trajectory load')
    manifest = dict(status='frozen', config_sha256='configuration')
    (root/'manifest.json').write_text(parent._json(manifest))
    freeze = dict(status='frozen', integrity_passed=True, trajectory_count=10000,
        manifest_sha256=parent._sha(root/'manifest.json'), prepared_cache_sha256=parent._sha(prepared))
    (root/'dataset_freeze.json').write_text(parent._json(freeze))
    normalization = tmp_path/'normalization.json'; normalization.write_text('{}')
    provenance = dict(config_sha256='configuration', base_checkpoint_sha256='b'*64,
        base_normalization_reference_sha256=parent._sha(normalization), source_snapshot={})
    (root/'provenance.json').write_text(parent._json(provenance))
    for split in ('train', 'validation', 'test'):
        (root/split).mkdir(); (root/split/'index.json').write_text('{"synthetic_fixture":true}')
    config = dict(dataset_root=str(root), dataset_freeze_sha256=parent._sha(root/'dataset_freeze.json'),
        dataset_manifest_sha256=freeze['manifest_sha256'], prepared_cache_sha256=freeze['prepared_cache_sha256'],
        normalization_reference=str(normalization), normalization_sha256=parent._sha(normalization))
    protocol = dict(training_dataset=str(root), dataset_freeze_sha256=config['dataset_freeze_sha256'], base_sha256='b'*64)
    audit = appendix._Audit(); appendix._training_inputs(audit, config, protocol)
    assert str(prepared) in audit.hashes and str(root/'train/index.json') in audit.hashes and str(root/'provenance.json') in audit.hashes
    prepared.write_bytes(b'changed synthetic cache')
    with pytest.raises(appendix.IntegrityError, match='checksum'):
        appendix._training_inputs(appendix._Audit(), config, protocol)


def test_failed_audit_creates_no_report_or_formal_placeholder(monkeypatch, tmp_path):
    def fail(*args):
        raise appendix.IncompleteExperiment('synthetic incomplete upstream')
    monkeypatch.setattr(appendix, 'audit_inputs', fail)
    with pytest.raises(appendix.IncompleteExperiment):
        appendix.build_consolidated_report(tmp_path/'experiment', tmp_path/'benchmark')
    assert not (tmp_path/'experiment/consolidated_reports').exists()


def test_interpretation_does_not_claim_direction_when_ci_crosses_zero(synthetic):
    lines = '\n'.join(appendix._interpretation(synthetic['supplement_report']))
    assert '结果解读' in lines and '五档' in lines and '同条件 K8−K1' in lines
    assert appendix._interval_direction(dict(mean=.1, ci95=[-.2, .3])) == '地图区间包含0，不能据此断言差值方向确定'
    assert '不是未见训练 seed' in lines and '不同' in lines


def test_writer_exports_complete_synthetic_tables_but_never_claims_research_complete(monkeypatch, synthetic):
    monkeypatch.setattr(appendix, 'audit_inputs', lambda *args: synthetic)
    result = appendix.build_consolidated_report(synthetic['experiment'], synthetic['benchmark'])
    output = synthetic['experiment']/'consolidated_reports'
    assert result['status'] == 'synthetic_test_only' and len(result['model_ids']) == 22
    assert set(result['output_files_sha256']) == {'research_report_zh.md', 'full_statistics.csv.gz', 'full_statistics.jsonl.gz'}
    for relative, digest in result['output_files_sha256'].items():
        assert parent._sha(output/relative) == digest
    with gzip.open(output/'full_statistics.csv.gz', 'rt') as stream:
        rows = list(csv.DictReader(stream))
    with gzip.open(output/'full_statistics.jsonl.gz', 'rt') as stream:
        json_rows = [json.loads(line) for line in stream]
    assert len(rows) == len(json_rows) == result['statistics_rows']
    assert {row['projection'] for row in rows if row['projection']} == set(appendix.PROJECTIONS)
    assert {'K1', 'K8_any', 'K8_diversity', 'K1_paired', 'K1_adapter_control'} == {row['table'] for row in rows}
    assert {'family', 'kind', 'difficulty', 'family_kind_difficulty', 'overall'} == {row['scope'] for row in rows}
    assert any(row['aggregation'] == 'within_seed' and row['control_id'] for row in rows)
    text = (output/'research_report_zh.md').read_text()
    assert text.startswith('# SYNTHETIC FIXTURE')
    for phrase in ('三种子负控', '512000', '1024/512/256/128/64', '不是任务成功率', '源地图',
                   'K8 Any@8', 'on_smooth−on', 'contact、normal、smooth', '两场景见证交集',
                   'family×plane/cylinder×easy/detour', 'Inspection', '飞行动力学', 'flight/control latency'):
        assert phrase in text


def test_output_cannot_overwrite_frozen_inputs(monkeypatch, synthetic):
    monkeypatch.setattr(appendix, 'audit_inputs', lambda *args: synthetic)
    original = parent._sha(synthetic['parent_report_path'])
    with pytest.raises(ValueError, match='consolidated_reports'):
        appendix.build_consolidated_report(synthetic['experiment'], synthetic['benchmark'], output=synthetic['parent_report_path'].parent)
    assert parent._sha(synthetic['parent_report_path']) == original


def test_cli_has_no_partial_bypass(monkeypatch, synthetic):
    calls = []
    monkeypatch.setattr(appendix, 'build_consolidated_report', lambda *args: calls.append(args) or
                        dict(status='synthetic_test_only', model_ids=['synthetic'], statistics_rows=1))
    appendix.main(['--experiment', str(synthetic['experiment']), '--benchmark', str(synthetic['benchmark'])])
    assert calls[0][2:] == (None, None)
    with pytest.raises(SystemExit):
        appendix.main(['--experiment', str(synthetic['experiment']), '--benchmark', str(synthetic['benchmark']), '--allow-partial'])
