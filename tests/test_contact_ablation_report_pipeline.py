"""Small mocked status/artifact tests; never launch models or physics workers."""
import argparse
import json
import os
from pathlib import Path

import pytest

from hnuter_dit import contact_ablation_report_pipeline as driver


@pytest.fixture(params=['parent', 'smooth'])
def upstream(tmp_path, monkeypatch, request):
    experiment, benchmark = tmp_path/'experiment', tmp_path/'benchmark'
    supplement = experiment/'smooth_projection_supplement'
    directory = experiment if request.param == 'parent' else supplement
    (directory/'thermal/pipeline').mkdir(parents=True)
    benchmark.mkdir()
    child = dict(pid=601, uid=os.getuid(), pgid=601, sid=601, start_ticks=1000)
    guardian = dict(pid=501, uid=os.getuid(), pgid=501, sid=501, start_ticks=900, state='S')
    module = 'hnuter_dit.contact_pipeline' if request.param == 'parent' else 'hnuter_dit.contact_smooth_pipeline'
    command = ['python', '-m', module, '--experiment', str(experiment), '--benchmark', str(benchmark)]
    state = dict(status='running', pid=601, source_contract_sha256=request.param+'_sources', protocol_sha256='protocol')
    guard = dict(status='paused_cooling', guard_pid=501, child=child, command=command)
    contract = dict(parent_source_contract_sha256='parent_sources', smooth_source_contract_sha256='smooth_sources',
                    parent_protocol_sha256='protocol')
    monkeypatch.setattr(driver, 'process_identity', lambda _: guardian)
    monkeypatch.setattr(driver, '_guardian_command', lambda *args: None)
    monkeypatch.setattr(driver, '_fresh_guard', lambda *args: None)
    monkeypatch.setattr(driver, 'verify_identity', lambda *args: None)
    monkeypatch.setattr(driver, 'remaining_owned_members', lambda _: [])
    monkeypatch.setattr(driver.time, 'sleep', lambda _: None)
    def save():
        (directory/'pipeline_status.json').write_text(json.dumps(state))
        (directory/'thermal/pipeline/status.json').write_text(json.dumps(guard))
    save()
    return request.param, experiment, benchmark, supplement, contract, state, guard, guardian, save


def health(fixture, binding=None):
    kind, experiment, benchmark, supplement, contract, *_ = fixture
    return driver.upstream_health(kind, experiment, benchmark, supplement, contract, binding)


def test_waiting_is_lightweight_and_pins_full_identity(upstream, monkeypatch):
    monkeypatch.setattr(driver, 'file_sha256', lambda _: (_ for _ in ()).throw(AssertionError('hash during wait')))
    result = health(upstream)
    assert result['complete'] is False and result['guard_status'] == 'paused_cooling'
    assert result['binding']['guardian']['start_ticks'] == 900
    assert health(upstream, result['binding']) == result


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('pid', 602), ('source_contract_sha256', 'changed')])
def test_failed_or_changed_upstream_is_rejected(upstream, field, value):
    upstream[5][field] = value
    upstream[-1]()
    with pytest.raises(driver.PipelineError):
        health(upstream)


def test_changed_guardian_identity_is_rejected(upstream):
    binding = health(upstream)['binding']
    upstream[7]['start_ticks'] += 1
    with pytest.raises(driver.PipelineError, match='changed after binding'):
        health(upstream, binding)


def test_complete_status_still_waits_for_original_guardian_exit(upstream, monkeypatch):
    binding = health(upstream)['binding']
    upstream[5]['status'] = 'complete'
    upstream[6].update(status='complete', exit_code=0)
    upstream[-1]()
    assert health(upstream, binding)['complete'] is False
    monkeypatch.setattr(driver, 'process_identity', lambda _: None)
    assert health(upstream, binding)['complete'] is True


def test_completed_historical_guardian_pid_reuse_is_not_adopted(upstream, monkeypatch):
    binding = health(upstream)['binding']
    upstream[5]['status'] = 'complete'
    upstream[6].update(status='complete', exit_code=0)
    upstream[-1]()
    monkeypatch.setattr(driver, 'process_identity', lambda _: dict(upstream[7], start_ticks=2000))
    monkeypatch.setattr(driver, '_guardian_command', lambda *args: (_ for _ in ()).throw(AssertionError('reused process inspected')))
    assert health(upstream, binding)['complete'] is True


@pytest.mark.parametrize('exit_code,members', [(1, []), (0, [dict(pid=601)])])
def test_complete_requires_exit0_and_no_owned_members(upstream, monkeypatch, exit_code, members):
    upstream[5]['status'] = 'complete'
    upstream[6].update(status='complete', exit_code=exit_code)
    upstream[-1]()
    monkeypatch.setattr(driver, 'remaining_owned_members', lambda _: members)
    with pytest.raises(driver.PipelineError, match='exit0'):
        health(upstream)


@pytest.mark.parametrize('old_guard_status', ['running', 'complete'])
def test_normal_cross_file_exit_snapshot_is_retried(upstream, monkeypatch, old_guard_status):
    binding = health(upstream)['binding']
    snapshots = iter([dict(upstream[5]), dict(upstream[6], status=old_guard_status, exit_code=0),
                      dict(upstream[5], status='complete'), dict(upstream[6], status='complete', exit_code=0)])
    monkeypatch.setattr(driver, 'read_json', lambda _: next(snapshots))
    monkeypatch.setattr(driver, 'process_identity', lambda _: None)
    assert health(upstream, binding)['complete'] is True


def test_command_inspection_exit_race_is_retried(upstream, monkeypatch):
    binding = health(upstream)['binding']
    identities = iter([upstream[7], None, None])
    monkeypatch.setattr(driver, 'process_identity', lambda _: next(identities))
    def race(*args):
        upstream[5]['status'] = 'complete'
        upstream[6].update(status='complete', exit_code=0)
        upstream[-1]()
        raise driver.PipelineError('command disappeared')
    monkeypatch.setattr(driver, '_guardian_command', race)
    assert health(upstream, binding)['complete'] is True


def test_disappeared_guardian_does_not_complete(upstream, monkeypatch):
    monkeypatch.setattr(driver, 'process_identity', lambda _: None)
    with pytest.raises(driver.PipelineError, match='disappeared'):
        health(upstream)


def _contract_fixture(tmp_path, monkeypatch):
    experiment, benchmark = tmp_path/'experiment', tmp_path/'benchmark'
    supplement, output = experiment/'smooth_projection_supplement', experiment/'consolidated_reports'
    output.mkdir(parents=True); supplement.mkdir(); benchmark.mkdir()
    root = tmp_path/'root'
    (root/'hnuter_dit').mkdir(parents=True)
    for name in driver.SOURCES:
        (root/'hnuter_dit'/name).write_text('fixture source')
    source = root/'parent_source.py'; source.write_text('parent')
    input_path = benchmark/'input.json'; input_path.write_text('{}')
    parent = dict(experiment=str(experiment), benchmark=str(benchmark),
                  source_files={str(source): driver.file_sha256(source)},
                  input_files={str(input_path): driver.file_sha256(input_path)})
    parent['contract_sha256'] = driver.json_hash(parent)
    smooth = {key: value for key, value in parent.items() if key != 'contract_sha256'} | dict(
        supplement=str(supplement), parent_source_contract_sha256=parent['contract_sha256'])
    smooth['contract_sha256'] = driver.json_hash(smooth)
    (experiment/'pipeline_source_contract.json').write_text(json.dumps(parent))
    (supplement/'pipeline_source_contract.json').write_text(json.dumps(smooth))
    monkeypatch.setattr(driver, 'ROOT', root)
    monkeypatch.setattr(driver, 'load_protocol', lambda _: dict(protocol_sha256='protocol'))
    return experiment, benchmark, supplement, output, root


def test_separate_source_contract_is_immutable(tmp_path, monkeypatch):
    experiment, benchmark, supplement, output, root = _contract_fixture(tmp_path, monkeypatch)
    contract = driver.ensure_contract(experiment, benchmark, supplement, output)
    assert len(contract['source_files']) == 3
    assert len(contract['input_files']) == 3
    assert driver.ensure_contract(experiment, benchmark, supplement, output) == contract
    (root/'hnuter_dit/report_contact_ablation_appendix.py').write_text('changed')
    with pytest.raises(driver.PipelineError, match='changed'):
        driver.ensure_contract(experiment, benchmark, supplement, output)


def test_changed_upstream_input_is_rejected(tmp_path, monkeypatch):
    experiment, benchmark, supplement, output, _ = _contract_fixture(tmp_path, monkeypatch)
    (benchmark/'input.json').write_text('changed')
    with pytest.raises(driver.PipelineError, match='immutable'):
        driver.ensure_contract(experiment, benchmark, supplement, output)


def _report_fixture(output, root):
    output.mkdir(parents=True, exist_ok=True)
    parent_report, smooth_report = root/'parent_report.json', root/'smooth_report.json'
    parent_report.write_text('{}'); smooth_report.write_text('{}')
    inputs = dict(parent_report_path=str(parent_report), supplement_report_path=str(smooth_report),
                  parent_report_sha256=driver.file_sha256(parent_report),
                  supplement_report_sha256=driver.file_sha256(smooth_report),
                  benchmark_manifest_sha256='manifest', verified_input_files_sha256={str(parent_report):driver.file_sha256(parent_report)})
    ids = ['base', *[row['id'] for row in driver.approved_rows()]]
    report = dict(schema_version='contact_ablation_consolidated_report_v001', status='complete', shortfalls=[],
                  model_ids=ids, models={key:{} for key in ids}, planned_conditions=10000, planned_source_maps=1000,
                  multimodal_subset_conditions=1000, primary_candidates=1, multimodal_candidates=8,
                  bootstrap_replicates=1000, bootstrap_seed=20261011, statistics_rows=22,
                  report_source_sha256=driver.file_sha256(root/'hnuter_dit/report_contact_ablation_appendix.py'), **inputs)
    def write():
        for name in driver.OUTPUT_FILES:
            (output/name).write_text('fixture')
        report['output_files_sha256'] = {name:driver.file_sha256(output/name) for name in driver.OUTPUT_FILES}
        (output/'report.json').write_text(json.dumps(report))
    write()
    return inputs, report, write


@pytest.mark.parametrize('field,value', [('status','partial'), ('shortfalls',['missing']), ('planned_conditions',9999),
                                        ('model_ids',['base']), ('bootstrap_seed',1)])
def test_report_completeness_is_strict(tmp_path, monkeypatch, field, value):
    *_, output, root = _contract_fixture(tmp_path, monkeypatch)
    inputs, report, write = _report_fixture(output, root)
    assert driver.verify_output(output, inputs)['statistics_rows'] == 22
    report[field] = value; write()
    with pytest.raises(driver.PipelineError, match='complete fixed'):
        driver.verify_output(output, inputs)


def test_saved_statistics_and_upstream_report_hashes_are_verified(tmp_path, monkeypatch):
    *_, output, root = _contract_fixture(tmp_path, monkeypatch)
    inputs, _, write = _report_fixture(output, root)
    (output/'full_statistics.csv.gz').write_text('changed')
    with pytest.raises(driver.PipelineError, match='artifact checksum'):
        driver.verify_output(output, inputs)
    write()
    Path(inputs['parent_report_path']).write_text('changed')
    with pytest.raises(driver.PipelineError, match='changed after'):
        driver.verify_output(output, inputs)


def _run_fixture(tmp_path, monkeypatch):
    experiment, benchmark, supplement, output, root = _contract_fixture(tmp_path, monkeypatch)
    guard_dir = output/'thermal/pipeline'; guard_dir.mkdir(parents=True)
    (guard_dir/'status.json').write_text(json.dumps(dict(child=dict(pid=os.getpid()))))
    inputs, report, write = _report_fixture(output, root)
    (output/'report.json').unlink()
    monkeypatch.setattr(driver, 'verify_pipeline_guard', lambda *args: dict(pid=701))
    monkeypatch.setattr(driver, 'ensure_contract', lambda *args: dict(contract_sha256='sources'))
    monkeypatch.setattr(driver, 'verify_source_contract', lambda *args: None)
    monkeypatch.setattr(driver, 'audit_ready_inputs', lambda *args: inputs)
    monkeypatch.setattr(driver, 'upstream_health', lambda kind,*args: dict(complete=True, binding=dict(kind=kind)))
    calls = []
    def run_stage(stage, logs, env):
        calls.append((stage, env)); write()
        return dict(exit_code=0, elapsed_s=.1, log=str(logs/'report.log'))
    monkeypatch.setattr(driver, 'run_stage', run_stage)
    args = argparse.Namespace(experiment=experiment, benchmark=benchmark, supplement=supplement, poll_seconds=30.)
    return args, output, inputs, report, write, calls


def test_guarded_report_runs_once_and_recovery_authenticates_saved_result(tmp_path, monkeypatch):
    args, output, _, _, _, calls = _run_fixture(tmp_path, monkeypatch)
    result = driver.run(args)
    assert result['status'] == 'complete' and result['completed_stages'] == ['report']
    assert len(calls) == 1 and calls[0][0].name == 'report'
    assert '--allow-partial' not in calls[0][0].command
    assert all(calls[0][1][key]=='1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    assert driver.run(args)['status'] == 'complete'
    assert len(calls) == 1
    value = json.loads((output/'report.json').read_text());value['statistics_rows'] += 1
    (output/'report.json').write_text(json.dumps(value))
    with pytest.raises(driver.PipelineError, match='saved evidence'):
        driver.run(args)
    assert json.loads((output/'pipeline_status.json').read_text())['status'] == 'failed'
    assert len(calls) == 1
    # Failed status must not erase the earlier completion's immutable hashes.
    with pytest.raises(driver.PipelineError, match='saved evidence'):
        driver.run(args)
    assert len(calls) == 1


def test_committed_report_survives_interrupted_final_status_write(tmp_path, monkeypatch):
    args, output, _, _, _, calls = _run_fixture(tmp_path, monkeypatch)
    assert driver.run(args)['status'] == 'complete'
    assert (output/'report_commit.json').is_file()
    status = json.loads((output/'pipeline_status.json').read_text())
    status.update(status='running', completed_stages=[])
    status['stage_evidence'] = {}
    (output/'pipeline_status.json').write_text(json.dumps(status))
    assert driver.run(args)['status'] == 'complete'
    assert len(calls) == 1


def test_upstream_failure_does_not_launch_report_or_signal_upstream(tmp_path, monkeypatch):
    args, output, _, _, _, calls = _run_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, 'upstream_health', lambda *args: (_ for _ in ()).throw(driver.PipelineError('parent failed')))
    with pytest.raises(driver.PipelineError, match='parent failed'):
        driver.run(args)
    state = json.loads((output/'pipeline_status.json').read_text())
    assert state['status'] == 'failed' and state['completed_stages'] == [] and calls == []


def test_waits_for_both_groups_before_full_audit(tmp_path, monkeypatch):
    args, _, inputs, _, _, calls = _run_fixture(tmp_path, monkeypatch)
    round_number, audit_rounds = [0], []
    def health(kind, *rest):
        if kind=='parent': round_number[0] += 1
        return dict(complete=kind=='parent' or round_number[0]>1, binding=dict(kind=kind))
    monkeypatch.setattr(driver, 'upstream_health', health)
    monkeypatch.setattr(driver.time, 'sleep', lambda _: None)
    monkeypatch.setattr(driver, 'audit_ready_inputs', lambda *args: audit_rounds.append(round_number[0]) or inputs)
    assert driver.run(args)['status'] == 'complete'
    assert audit_rounds == [2] and len(calls) == 1


def test_does_not_write_into_overlapping_supplement(tmp_path):
    args = argparse.Namespace(experiment=tmp_path, benchmark=tmp_path/'benchmark',
                              supplement=tmp_path/'consolidated_reports', poll_seconds=30.)
    with pytest.raises(driver.PipelineError, match='overlap'):
        driver.run(args)
