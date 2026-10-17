import json
import os
from pathlib import Path

import pytest

from hnuter_dit import contact_smooth_pipeline as driver


@pytest.fixture
def parent(tmp_path, monkeypatch):
    experiment, benchmark = tmp_path/'experiment', tmp_path/'benchmark'
    (experiment/'thermal/pipeline').mkdir(parents=True)
    benchmark.mkdir()
    child = dict(pid=601, uid=os.getuid(), pgid=601, sid=601, start_ticks=1000)
    guardian = dict(pid=501, uid=os.getuid(), pgid=501, sid=501, start_ticks=900, state='S')
    command = ['python', '-m', 'hnuter_dit.contact_pipeline', '--experiment', str(experiment),
               '--benchmark', str(benchmark)]
    state = dict(status='running', current_stage='training', pid=601, source_contract_sha256='sources',
                 protocol_sha256='protocol')
    guard = dict(status='paused_cooling', guard_pid=501, child=child, command=command)
    monkeypatch.setattr(driver, 'load_protocol', lambda _: dict(protocol_sha256='protocol'))
    monkeypatch.setattr(driver, 'process_identity', lambda _: guardian)
    monkeypatch.setattr(driver, '_guardian_command', lambda *args: None)
    monkeypatch.setattr(driver, '_fresh_guard', lambda *args: None)
    monkeypatch.setattr(driver, 'verify_identity', lambda *args: None)
    monkeypatch.setattr(driver, 'remaining_owned_members', lambda _: [])
    def save():
        (experiment/'pipeline_status.json').write_text(json.dumps(state))
        (experiment/'thermal/pipeline/status.json').write_text(json.dumps(guard))
    save()
    return experiment, benchmark, state, guard, guardian, save


def health(parent, binding=None):
    experiment, benchmark, *_ = parent
    return driver.parent_health(experiment, benchmark, {'parent_source_contract_sha256': 'sources'}, binding)


def test_waits_for_parent_and_pins_identity(parent):
    result = health(parent)
    assert result['complete'] is False
    assert result['parent_guard_status'] == 'paused_cooling'
    assert result['binding']['child']['start_ticks'] == 1000
    assert health(parent, result['binding'])['complete'] is False


def test_fails_if_parent_guardian_disappears(parent, monkeypatch):
    monkeypatch.setattr(driver, 'process_identity', lambda _: None)
    with pytest.raises(driver.PipelineError, match='disappeared'):
        health(parent)


def test_rejects_changed_parent_identity(parent):
    binding = health(parent)['binding']
    parent[3]['child']['start_ticks'] += 1
    parent[-1]()
    with pytest.raises(driver.PipelineError, match='changed after binding'):
        health(parent, binding)


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('source_contract_sha256', 'changed'),
                                        ('protocol_sha256', 'changed'), ('pid', 602)])
def test_rejects_failed_or_changed_parent(parent, field, value):
    parent[2][field] = value
    parent[-1]()
    with pytest.raises(driver.PipelineError):
        health(parent)


def test_complete_status_waits_for_guardian_exit(parent):
    parent[2]['status'] = 'complete'
    parent[-1]()
    assert health(parent)['complete'] is False


def test_guardian_complete_does_not_mean_study_complete(parent):
    parent[3].update(status='complete', exit_code=0)
    parent[-1]()
    with pytest.raises(driver.PipelineError, match='without successful full study'):
        health(parent)


def test_completed_parent_requires_full_report_verification(parent, monkeypatch):
    binding = health(parent)['binding']
    experiment = parent[0]
    report = experiment/'reports/report.json'
    report.parent.mkdir()
    report.write_text('{}')
    parent[2].update(status='complete', report=str(report))
    parent[3].update(status='complete', exit_code=0)
    parent[-1]()
    monkeypatch.setattr(driver, 'process_identity', lambda _: None)
    calls = []
    monkeypatch.setattr(driver, 'verify_stage_outputs', lambda *args: calls.append(args) or {'full': True})
    result = health(parent, binding)
    assert result['complete'] is True
    assert result['report_evidence'] == {'full': True}
    assert len(calls) == 1 and calls[0][0].name == 'report'
    monkeypatch.setattr(driver, 'verify_stage_outputs', lambda *args: (_ for _ in ()).throw(driver.PipelineError('partial report')))
    with pytest.raises(driver.PipelineError, match='partial report'):
        health(parent, binding)


def test_complete_parent_rejects_remaining_owned_members(parent, monkeypatch):
    parent[2]['status'] = 'complete'
    parent[3].update(status='complete', exit_code=0)
    parent[-1]()
    monkeypatch.setattr(driver, 'remaining_owned_members', lambda _: [parent[3]['child']])
    with pytest.raises(driver.PipelineError, match='without successful full study'):
        health(parent)


def test_completed_parent_ignores_reused_guardian_pid(parent, monkeypatch):
    binding = health(parent)['binding']
    report = parent[0]/'reports/report.json'
    report.parent.mkdir()
    report.write_text('{}')
    parent[2].update(status='complete', report=str(report))
    parent[3].update(status='complete', exit_code=0)
    parent[-1]()
    reused = dict(parent[4], start_ticks=2000)
    monkeypatch.setattr(driver, 'process_identity', lambda _: reused)
    monkeypatch.setattr(driver, '_guardian_command', lambda *args: (_ for _ in ()).throw(AssertionError('reused process inspected')))
    monkeypatch.setattr(driver, 'verify_stage_outputs', lambda *args: {'full': True})
    assert health(parent, binding)['complete'] is True


@pytest.mark.parametrize('old_guard_status', ['running', 'complete'])
def test_exit_file_snapshot_race_is_retried(parent, monkeypatch, old_guard_status):
    binding = health(parent)['binding']
    experiment = parent[0]
    report = experiment/'reports/report.json'
    report.parent.mkdir()
    report.write_text('{}')
    old_state = dict(parent[2])
    complete_state = old_state | dict(status='complete', report=str(report))
    old_guard = dict(parent[3], status=old_guard_status, exit_code=0)
    complete_guard = old_guard | dict(status='complete')
    snapshots = iter([old_state, old_guard, complete_state, complete_guard])
    monkeypatch.setattr(driver, 'read_json', lambda _: next(snapshots))
    monkeypatch.setattr(driver, 'process_identity', lambda _: None)
    monkeypatch.setattr(driver, 'time', type('Clock', (), {'sleep': staticmethod(lambda _: None)}))
    monkeypatch.setattr(driver, 'verify_stage_outputs', lambda *args: {'full': True})
    assert health(parent, binding)['complete'] is True


def test_guardian_exit_during_command_read_is_retried(parent, monkeypatch):
    binding = health(parent)['binding']
    report = parent[0]/'reports/report.json'
    report.parent.mkdir()
    report.write_text('{}')
    identities = iter([parent[4], None, None])
    monkeypatch.setattr(driver, 'process_identity', lambda _: next(identities))
    def command_race(*args):
        parent[2].update(status='complete', report=str(report))
        parent[3].update(status='complete', exit_code=0)
        parent[-1]()
        raise driver.PipelineError('process vanished before cmdline read')
    monkeypatch.setattr(driver, '_guardian_command', command_race)
    monkeypatch.setattr(driver, 'verify_stage_outputs', lambda *args: {'full': True})
    assert health(parent, binding)['complete'] is True


def test_stages_always_request_full_matrix(tmp_path):
    stages = driver.build_stages(tmp_path/'parent', tmp_path/'benchmark', tmp_path/'supplement', 'python')
    assert [stage.name for stage in stages] == ['score', 'report']
    assert '--limit-maps' not in stages[0].command and '--model' not in stages[0].command
    assert stages[0].command[-2:] == ['--workers', '2']
    assert '--allow-partial' not in stages[1].command


def _complete_score_fixture(tmp_path):
    supplement = tmp_path/'supplement'
    ids = ['base', *[row['id'] for row in driver.approved_rows()]]
    supplement.mkdir()
    benchmark = tmp_path/'benchmark'
    benchmark.mkdir()
    (benchmark/'manifest.json').write_text('{}')
    manifest_hash = driver.file_sha256(benchmark/'manifest.json')
    protocol = dict(planned_models=ids, condition_count=10000, source_map_count=1000,
                    projection='on_smooth', decoder_parameters=driver.smooth_contact_decoder_parameters(),
                    benchmark_manifest_sha256=manifest_hash)
    protocol['protocol_sha256'] = driver.json_hash(protocol)
    (supplement/'protocol.json').write_text(json.dumps(protocol))
    for model in ids:
        directory = supplement/'evaluation'/model
        directory.mkdir(parents=True)
        contract = dict(model_id=model, protocol_sha256=protocol['protocol_sha256'],
                        benchmark_manifest_sha256=manifest_hash, score_rows_per_model=34000)
        contract['contract_sha256'] = driver.json_hash(contract)
        (directory/'contract.json').write_text(json.dumps(contract))
        status = dict(status='complete', complete_maps=1000, total_maps=1000,
                      planned_total_maps=1000, pilot_limit_maps=0, score_rows=34000,
                      expected_score_rows=34000, planned_score_rows=34000,
                      contract_sha256=contract['contract_sha256'])
        (directory/'scoring_status.json').write_text(json.dumps(status))
    return benchmark, supplement, ids


def test_partial_smooth_model_is_never_complete(tmp_path):
    benchmark, supplement, ids = _complete_score_fixture(tmp_path)
    assert driver.verify_output(driver.Stage('score', []), benchmark, supplement)['score_rows'] == 22*34000
    path = supplement/'evaluation'/ids[-1]/'scoring_status.json'
    status = json.loads(path.read_text()) | dict(status='pilot_complete', complete_maps=1)
    path.write_text(json.dumps(status))
    with pytest.raises(driver.PipelineError, match=ids[-1]):
        driver.verify_output(driver.Stage('score', []), benchmark, supplement)


@pytest.mark.parametrize('field,value', [('score_rows', 33999), ('expected_score_rows', 33999),
                                        ('planned_score_rows', 33999), ('pilot_limit_maps', 1)])
def test_full_map_count_cannot_hide_short_or_pilot_scores(tmp_path, field, value):
    benchmark, supplement, ids = _complete_score_fixture(tmp_path)
    path = supplement/'evaluation'/ids[-1]/'scoring_status.json'
    value = json.loads(path.read_text()) | {field: value}
    path.write_text(json.dumps(value))
    with pytest.raises(driver.PipelineError, match=ids[-1]):
        driver.verify_output(driver.Stage('score', []), benchmark, supplement)


@pytest.mark.parametrize('target', ['protocol', 'model'])
def test_stage_rejects_changed_json_contract(tmp_path, target):
    benchmark, supplement, ids = _complete_score_fixture(tmp_path)
    path = supplement/'protocol.json' if target == 'protocol' else supplement/'evaluation'/ids[-1]/'contract.json'
    value = json.loads(path.read_text()) | dict(changed=True)
    path.write_text(json.dumps(value))
    with pytest.raises(driver.PipelineError):
        driver.verify_output(driver.Stage('score', []), benchmark, supplement)
