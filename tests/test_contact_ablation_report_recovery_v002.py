"""Report recovery integrity regressions; no models, scoring, or physics."""
import json
from pathlib import Path

import pytest

from hnuter_dit import contact_ablation_report_recovery_v002 as recovery


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def failure(tmp_path, monkeypatch):
    directory = tmp_path/'consolidated_reports'
    child = dict(pid=111, uid=1000, pgid=111, sid=111, start_ticks=100, state='R')
    guardian = dict(pid=222, uid=1000, pgid=222, sid=222, start_ticks=90, state='S')
    source = tmp_path/'original.py'; source.write_text('frozen')
    contract = dict(source_files={str(source): recovery.file_sha256(source)}, input_files={})
    contract['contract_sha256'] = recovery.json_hash(contract)
    save(directory/'source_contract.json', contract)
    save(directory/'pipeline_status.json', dict(status='failed', error_type='KeyError', error="'input_files'",
        completed_stages=[], pid=child['pid'], source_contract_sha256=contract['contract_sha256']))
    save(directory/'thermal/pipeline/status.json', dict(status='failed', exit_code=1, child=child))
    save(directory/'pipeline_launch.json', dict(child_identity=child, guardian_identity=guardian))
    monkeypatch.setattr(recovery, 'matching_identity', lambda _: False)
    monkeypatch.setattr(recovery, 'remaining_owned_members', lambda _: [])
    return directory, source


def test_failure_evidence_preserves_bytes_and_ignores_own_recovery(tmp_path, monkeypatch):
    directory, _ = failure(tmp_path, monkeypatch)
    first = recovery.failure_evidence(directory)
    save(directory/'recovery_v002/pipeline_status.json', dict(status='initializing'))
    assert recovery.failure_evidence(directory) == first
    for path, digest in first['preserved_files_sha256'].items():
        assert recovery.file_sha256(Path(path)) == digest


def test_modified_frozen_original_is_rejected(tmp_path, monkeypatch):
    directory, source = failure(tmp_path, monkeypatch)
    source.write_text('different')
    with pytest.raises(recovery.PipelineError, match='immutable'):
        recovery.failure_evidence(directory)


@pytest.mark.parametrize('function', ['matching_identity', 'remaining_owned_members'])
def test_live_original_identity_prevents_duplicate_recovery(tmp_path, monkeypatch, function):
    directory, _ = failure(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, function, lambda _: True)
    with pytest.raises(recovery.PipelineError, match='owned processes'):
        recovery.failure_evidence(directory)


def report(tmp_path):
    output = tmp_path/'recovery'; output.mkdir()
    paths = dict(parent_report=tmp_path/'parent.json', supplement_report=tmp_path/'smooth.json')
    for path in paths.values():
        path.write_text('{}')
    bindings = {key+'_path': str(path) for key, path in paths.items()}
    bindings.update({key+'_sha256': recovery.file_sha256(path) for key, path in paths.items()})
    bindings['benchmark_manifest_sha256'] = 'manifest'
    ids = ['base', *[row['id'] for row in recovery.original.approved_rows()]]
    value = dict(schema_version=recovery.reporter.REPORT_VERSION, status='complete', shortfalls=[],
        model_ids=ids, models={name:{} for name in ids}, planned_conditions=10000, planned_source_maps=1000,
        multimodal_subset_conditions=1000, primary_candidates=1, multimodal_candidates=8,
        projections=['off','on','on_smooth'], bootstrap_replicates=1000, bootstrap_seed=20261011,
        report_source_sha256=recovery.file_sha256(Path(recovery.reporter.__file__)),
        recovery_schema_version=recovery.reporter.RECOVERY_SCHEMA_VERSION,
        original_reporter_path=str(recovery.reporter.ORIGINAL_REPORTER.resolve()),
        original_reporter_sha256=recovery.reporter.ORIGINAL_REPORTER_SHA256,
        oracle_access=False, decoder_replayed=False, new_sampling=False, statistics_rows=33,
        verified_input_files_sha256={str(path): recovery.file_sha256(path) for path in paths.values()}, **bindings)
    for name in recovery.original.OUTPUT_FILES:
        (output/name).write_text('saved artifact')
    value['output_files_sha256'] = {name: recovery.file_sha256(output/name) for name in recovery.original.OUTPUT_FILES}
    save(output/'report.json', value)
    return output, bindings, value


@pytest.mark.parametrize('change', [dict(projections=['off','on']), dict(status='partial'),
    dict(shortfalls=['missing']), dict(multimodal_subset_conditions=10000), dict(oracle_access=True),
    dict(report_source_sha256='wrong'), dict(original_reporter_sha256='wrong')])
def test_recovered_matrix_and_policies_remain_strict(tmp_path, change):
    output, bindings, value = report(tmp_path)
    assert recovery.verify_output(output, bindings)['statistics_rows'] == 33
    save(output/'report.json', value | change)
    with pytest.raises(recovery.PipelineError, match='complete frozen'):
        recovery.verify_output(output, bindings)


def test_changed_upstream_or_appendix_rejected_after_report(tmp_path):
    output, bindings, _ = report(tmp_path)
    (output/'full_statistics.csv.gz').write_text('changed')
    with pytest.raises(recovery.PipelineError, match='checksum'):
        recovery.verify_output(output, bindings)
    (output/'full_statistics.csv.gz').write_text('saved artifact')
    Path(bindings['parent_report_path']).write_text('changed')
    with pytest.raises(recovery.PipelineError, match='upstream report changed'):
        recovery.verify_output(output, bindings)
