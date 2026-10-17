"""Schema recovery regressions: source metadata and synthetic output only.

No benchmark audit, inference, CUDA, COAL, or original artifact mutation.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from hnuter_dit import report_contact_ablation_appendix as original
from hnuter_dit import report_contact_ablation_appendix_v002 as recovery

torch.set_num_threads(1)


@pytest.fixture
def code_contract():
    code = Path(recovery.parent.__file__).parent
    evaluator = recovery.smooth.evaluation
    return dict(schema_version=evaluator.SOURCE_CONTRACT_VERSION,
        source_files={str((code/name).resolve()): recovery.parent._sha(code/name)
                      for name in evaluator.SUPPLEMENT_SOURCES},
        frozen_dependency_sources={name: recovery.parent._sha(code/name)
            for name in (*evaluator.INFERENCE_SOURCES, *evaluator.EVALUATION_SOURCES)})


def test_code_only_contract_without_input_files_reproduces_v001_failure_and_is_fully_audited(code_contract):
    with pytest.raises(KeyError, match='input_files'):
        code_contract['source_files'] | code_contract['input_files']
    bindings = recovery._contract_file_bindings(code_contract)
    code = Path(recovery.parent.__file__).parent
    expected = code_contract['source_files'] | {
        str((code/name).resolve()): digest for name, digest in code_contract['frozen_dependency_sources'].items()}
    assert bindings == expected
    audit = recovery._Audit()
    for filename, digest in bindings.items():
        audit.check(Path(filename), digest)
    assert audit.hashes == expected
    assert 'input_files' not in code_contract  # original metadata remains intact


@pytest.mark.parametrize('schema', ['contact_pipeline_sources_v001', 'contact_smooth_pipeline_sources_v001',
                                  'contact_ablation_report_pipeline_sources_v001', 'unknown_schema'])
def test_only_code_only_schema_allows_missing_input_files(schema):
    with pytest.raises(KeyError, match='input_files'):
        recovery._contract_file_bindings(dict(schema_version=schema, source_files={}))


def test_pipeline_schema_keeps_declared_source_and_input_checksums(tmp_path):
    source, physical_input = tmp_path/'synthetic_source.py', tmp_path/'synthetic_input.json'
    source.write_text('# synthetic metadata test\n'); physical_input.write_text('{}')
    contract = dict(schema_version='contact_smooth_pipeline_sources_v001',
        source_files={str(source): recovery.parent._sha(source)},
        input_files={str(physical_input): recovery.parent._sha(physical_input)})
    bindings = recovery._contract_file_bindings(contract)
    assert bindings == contract['source_files'] | contract['input_files']
    audit = recovery._Audit()
    for filename, digest in bindings.items():
        audit.check(Path(filename), digest)
    assert len(audit.hashes) == 2


@pytest.mark.parametrize('field', ['source_files', 'frozen_dependency_sources'])
def test_code_only_schema_missing_mandatory_sources_or_dependencies_is_rejected(code_contract, field):
    del code_contract[field]
    with pytest.raises(KeyError, match=field):
        recovery._contract_file_bindings(code_contract)


@pytest.mark.parametrize('field', ['source_files', 'frozen_dependency_sources'])
def test_code_only_schema_bad_checksum_is_not_hidden_by_optional_input_compatibility(code_contract, field):
    key = next(iter(code_contract[field])); code_contract[field][key] = '0'*64
    with pytest.raises(recovery.IntegrityError, match='source/dependency'):
        recovery._contract_file_bindings(code_contract)


def test_unknown_dependency_names_are_rejected(code_contract):
    code_contract['frozen_dependency_sources']['../oracle/private.py'] = '0'*64
    with pytest.raises(recovery.IntegrityError, match='source/dependency'):
        recovery._contract_file_bindings(code_contract)


def test_declared_code_only_inputs_are_audited_and_conflicts_or_null_are_rejected(code_contract, tmp_path):
    physical_input = tmp_path/'synthetic_input.json'; physical_input.write_text('{}')
    code_contract['input_files'] = {str(physical_input): recovery.parent._sha(physical_input)}
    bindings = recovery._contract_file_bindings(code_contract)
    assert bindings[str(physical_input)] == recovery.parent._sha(physical_input)
    filename = next(iter(code_contract['source_files']))
    code_contract['input_files'][filename] = '0'*64
    with pytest.raises(recovery.IntegrityError, match='conflicting'):
        recovery._contract_file_bindings(code_contract)
    code_contract['input_files'] = None
    with pytest.raises(recovery.IntegrityError, match='mapping'):
        recovery._contract_file_bindings(code_contract)


def test_frozen_original_and_all_statistical_rendering_functions_are_unchanged():
    assert recovery.parent._sha(recovery.ORIGINAL_REPORTER) == recovery.ORIGINAL_REPORTER_SHA256
    before = ast.parse(Path(original.__file__).read_text())
    after = ast.parse(Path(recovery.__file__).read_text())
    functions_before = {node.name: node for node in before.body if isinstance(node, ast.FunctionDef)}
    functions_after = {node.name: node for node in after.body if isinstance(node, ast.FunctionDef)}
    for name, node in functions_before.items():
        if name not in {'audit_inputs', 'build_consolidated_report'}:
            assert ast.dump(node, include_attributes=False) == ast.dump(functions_after[name], include_attributes=False), name


def test_recovery_output_records_own_and_original_source_versions_without_claiming_synthetic_complete(monkeypatch, tmp_path):
    experiment = tmp_path/'synthetic_experiment'; supplement = experiment/'smooth_projection_supplement'
    parent_path = experiment/'reports/report.json'; smooth_path = supplement/'reports/report.json'
    for path in (parent_path, smooth_path):
        path.parent.mkdir(parents=True); path.write_text('{"synthetic_fixture":true}\n')
    audit = recovery._Audit(); audit.check(parent_path); audit.check(smooth_path)
    data = dict(experiment=experiment, benchmark=tmp_path/'synthetic_benchmark', supplement=supplement,
        parent_report_path=parent_path, supplement_report_path=smooth_path,
        matrix=[dict(id='base')], supplement_report=dict(benchmark_manifest_sha256='0'*64,
            models=dict(base=dict(training=dict(variant='base', train_size=0, seed=None), parent_contract={}, contract={}))),
        verified_input_files_sha256=audit.hashes, _audit=audit, synthetic_fixture=True)
    monkeypatch.setattr(recovery, 'audit_inputs', lambda *args: data)
    monkeypatch.setattr(recovery, '_rows', lambda *args: iter([dict(table='SYNTHETIC_FIXTURE', metric='synthetic_only', mean=0., observations=1)]))
    monkeypatch.setattr(recovery, '_markdown', lambda *args: '# SYNTHETIC FIXTURE：非科研结果\n')
    output = experiment/'consolidated_reports/recovery_v002'
    result = recovery.build_consolidated_report(experiment, data['benchmark'], output=output)
    assert result['status'] == 'synthetic_test_only'
    assert result['schema_version'] == 'contact_ablation_consolidated_report_v002'
    assert result['recovery_schema_version'] == 'contact_ablation_schema_recovery_v002'
    assert result['original_reporter_path'] == str(recovery.ORIGINAL_REPORTER.resolve())
    assert result['original_reporter_sha256'] == recovery.parent._sha(recovery.ORIGINAL_REPORTER)
    assert result['report_source_sha256'] == recovery.parent._sha(Path(recovery.__file__))
    assert result['report_source_sha256'] != result['original_reporter_sha256']
    assert 'all pipeline inputs remain mandatory' in result['recovery_policy']
    for filename, digest in result['output_files_sha256'].items():
        assert recovery.parent._sha(output/filename) == digest
