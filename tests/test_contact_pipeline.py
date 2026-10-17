"""Isolated controller checks: no training, benchmark jobs or real signals."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hnuter_dit import contact_pipeline as pipeline


def put(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def identity(pid: int, ticks: int = 100, state: str = 'S') -> dict:
    return dict(pid=pid, uid=os.getuid(), pgid=pid, sid=pid,
                start_ticks=ticks, state=state)


@pytest.fixture(autouse=True)
def no_real_process_actions(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('pipeline regression tests must not signal or launch real processes')
    monkeypatch.setattr(pipeline.os, 'kill', forbidden)
    monkeypatch.setattr(pipeline.os, 'killpg', forbidden)
    monkeypatch.setattr(pipeline.subprocess, 'run', forbidden)


@pytest.fixture
def training(tmp_path, monkeypatch):
    experiment = tmp_path / 'experiment'
    state_dir = experiment / 'thermal/training'
    child, guardian = identity(510101, 200), identity(510102, 100)
    processes = {child['pid']: child, guardian['pid']: guardian}
    guard = dict(status='running', guard_pid=guardian['pid'], child=child,
                 command=['/fake/python', '-m', 'hnuter_dit.contact_experiment',
                          '--output', str(experiment)],
                 updated_utc=datetime.now(timezone.utc).isoformat())
    matrix = dict(status='training', total_models=21, complete_models=[])
    monkeypatch.setattr(pipeline, 'process_identity', lambda pid: processes.get(pid))
    monkeypatch.setattr(pipeline, 'process_command', lambda _pid: [
        '/fake/python', '-m', 'hnuter_dit.thermal_run', '--state-dir', str(state_dir), '--gpu', '0'])
    monkeypatch.setattr(pipeline, 'remaining_owned_members',
                        lambda saved: [processes[saved['pid']]] if saved['pid'] in processes else [])
    def save():
        put(state_dir / 'status.json', guard)
        put(experiment / 'experiment_status.json', matrix)
    save()
    return SimpleNamespace(experiment=experiment, state_dir=state_dir, child=child,
                           guardian=guardian, processes=processes, guard=guard,
                           matrix=matrix, save=save)


@pytest.mark.parametrize('field', pipeline.IDENTITY_FIELDS)
def test_identity_reuse_or_mismatch_is_rejected(monkeypatch, field):
    saved = identity(510101)
    current = saved | {field: saved[field] + 1}
    monkeypatch.setattr(pipeline, 'process_identity', lambda _pid: current)
    with pytest.raises(pipeline.PipelineError, match='changed'):
        pipeline.verify_identity(saved, 'owned test process')


@pytest.mark.parametrize('current', [None, identity(510101, state='Z'), identity(510101, state='X')])
def test_dead_or_zombie_identity_is_rejected(monkeypatch, current):
    monkeypatch.setattr(pipeline, 'process_identity', lambda _pid: current)
    with pytest.raises(pipeline.PipelineError):
        pipeline.verify_identity(identity(510101), 'owned test process')


def test_active_training_pins_guard_and_accepts_thermal_pause(training):
    first = pipeline.inspect_training(training.experiment)
    assert not first['complete']
    assert first['binding']['guardian']['pid'] == training.guardian['pid']
    training.guard['status'] = 'paused_cooling'
    training.child['state'] = 'T'
    training.save()
    paused = pipeline.inspect_training(training.experiment, first['binding'])
    assert not paused['complete'] and paused['guard_status'] == 'paused_cooling'


def test_changed_guard_pid_is_rejected_even_if_original_guard_is_alive(training):
    binding = pipeline.inspect_training(training.experiment)['binding']
    replacement = identity(510103, 300)
    training.processes[replacement['pid']] = replacement
    training.guard['guard_pid'] = replacement['pid']
    training.save()
    with pytest.raises(pipeline.PipelineError, match='guardian'):
        pipeline.inspect_training(training.experiment, binding)


@pytest.mark.parametrize('failure', ['dead_guard', 'stopped_guard', 'stale', 'failed_matrix',
                                     'failed_guard', 'foreign_session', 'wrong_output'])
def test_unhealthy_training_never_counts_as_complete(training, failure):
    if failure == 'dead_guard':
        training.processes.pop(training.guardian['pid'])
    elif failure == 'stopped_guard':
        training.guardian['state'] = 'T'
    elif failure == 'stale':
        training.guard['updated_utc'] = (datetime.now(timezone.utc) - timedelta(seconds=121)).isoformat()
    elif failure == 'failed_matrix':
        training.matrix['status'] = 'training_failed'
    elif failure == 'failed_guard':
        training.guard['status'] = 'monitor_failed_child_paused'
    elif failure == 'foreign_session':
        training.child['sid'] += 1
    elif failure == 'wrong_output':
        training.guard['command'][-1] = str(training.experiment / 'another_matrix')
    training.save()
    with pytest.raises(pipeline.PipelineError):
        pipeline.inspect_training(training.experiment)


def test_terminal_matrix_waits_for_guard_and_all_members_to_exit(training):
    training.matrix.update(status='training_complete_evaluation_pending',
                           complete_models=[row['id'] for row in pipeline.approved_rows()])
    training.processes.pop(training.child['pid'])
    training.save()
    assert not pipeline.inspect_training(training.experiment)['complete']
    training.guard.update(status='complete', exit_code=0)
    training.processes.pop(training.guardian['pid'])
    training.save()
    assert pipeline.inspect_training(training.experiment)['complete']
    training.matrix['complete_models'].pop()
    training.save()
    with pytest.raises(pipeline.PipelineError, match='21'):
        pipeline.inspect_training(training.experiment)


def test_complete_guard_with_remaining_children_is_rejected(training):
    training.guard.update(status='complete', exit_code=0)
    training.matrix.update(status='training_complete_evaluation_pending',
                           complete_models=[row['id'] for row in pipeline.approved_rows()])
    training.save()
    with pytest.raises(pipeline.PipelineError, match='fully exited'):
        pipeline.inspect_training(training.experiment)


def test_guard_command_must_belong_to_this_state_directory(training, monkeypatch):
    monkeypatch.setattr(pipeline, 'process_command', lambda _pid: [
        '/fake/python', '-m', 'hnuter_dit.thermal_run', '--state-dir', '/tmp/another_guard', '--gpu', '0'])
    with pytest.raises(pipeline.PipelineError, match='different state directory'):
        pipeline.inspect_training(training.experiment)


@pytest.mark.parametrize('policy', ['missing_gpu', 'hot_cpu_limit', 'gpu_limit', 'cooldown', 'spoofed_child_argv'])
def test_training_guard_thermal_policy_cannot_be_weakened(training, monkeypatch, policy):
    command = ['/fake/python', '-m', 'hnuter_dit.thermal_run', '--state-dir', str(training.state_dir)]
    if policy != 'missing_gpu':
        command += ['--gpu', '0']
    if policy == 'hot_cpu_limit':
        command += ['--cpu-pause-c', '100']
    elif policy == 'gpu_limit':
        command += ['--gpu-pause-c', '95']
    elif policy == 'cooldown':
        command += ['--cool-seconds', '0']
    elif policy == 'spoofed_child_argv':
        command = ['/fake/python', '-m', 'unrelated.wrapper', '--', *command]
    monkeypatch.setattr(pipeline, 'process_command', lambda _pid: command)
    with pytest.raises(pipeline.PipelineError):
        pipeline.inspect_training(training.experiment)


@pytest.mark.parametrize('failure', ['dead', 'stopped', 'identity_changed', 'wrong_child', 'shared_group', 'stale'])
def test_outer_guard_identity_and_independence_are_required(tmp_path, monkeypatch, failure):
    state_dir = tmp_path / 'outer_guard'
    child, guardian = identity(os.getpid(), 200), identity(510102, 100)
    processes = {child['pid']: child, guardian['pid']: guardian}
    state = dict(status='running', child=child.copy(), guard_pid=guardian['pid'],
                 updated_utc=datetime.now(timezone.utc).isoformat())
    monkeypatch.setattr(pipeline, 'process_identity', lambda pid: processes.get(pid))
    monkeypatch.setattr(pipeline, 'process_command', lambda _pid: [
        '/fake/python', '-m', 'hnuter_dit.thermal_run', '--state-dir', str(state_dir), '--gpu', '0'])
    put(state_dir / 'status.json', state)
    binding = pipeline.verify_pipeline_guard(state_dir)
    if failure == 'dead':
        processes.pop(guardian['pid'])
    elif failure == 'stopped':
        guardian['state'] = 'T'
    elif failure == 'identity_changed':
        guardian['start_ticks'] += 1
    elif failure == 'wrong_child':
        state['child']['start_ticks'] += 1
    elif failure == 'shared_group':
        guardian['pgid'] = child['pgid']
    else:
        state['updated_utc'] = (datetime.now(timezone.utc) - timedelta(seconds=121)).isoformat()
    put(state_dir / 'status.json', state)
    with pytest.raises(pipeline.PipelineError):
        pipeline.verify_pipeline_guard(state_dir, binding if failure != 'shared_group' else None)


@pytest.fixture
def frozen_models(tmp_path):
    experiment = tmp_path / 'experiment'
    base = tmp_path / 'base/models/dit_cross/best.pt'
    base.parent.mkdir(parents=True)
    base.write_bytes(b'fake frozen base')
    statistics = base.parents[2] / 'canonical_statistics.json'
    put(statistics, {'fake': 'canonical statistics'})
    dataset = tmp_path / 'dataset'
    put(dataset / 'dataset_freeze.json', {'frozen': True})
    protocol = dict(base_checkpoint=str(base), base_sha256=pipeline.file_sha256(base),
                    training_dataset=str(dataset),
                    dataset_freeze_sha256=pipeline.file_sha256(dataset / 'dataset_freeze.json'))
    sources = {name: pipeline.file_sha256(pipeline.ROOT / 'hnuter_dit' / name)
               for name in pipeline.TRAINER_SOURCES}
    for row in pipeline.approved_rows():
        directory = experiment / 'models' / row['id']
        subset = dict(map_count=row['train_size'] // 20,
                      selected_map_ids=[f'map_{i:04d}' for i in range(row['train_size'] // 20)])
        config = dict(base_path=str(base), base_sha256=protocol['base_sha256'],
                      dataset_root=str(dataset), dataset_freeze_sha256=protocol['dataset_freeze_sha256'],
                      normalization_reference=str(statistics), normalization_sha256=pipeline.file_sha256(statistics),
                      train_size=row['train_size'], seed=row['seed'], training_steps=8000, batch_size=64,
                      data_subset_seed=20261011, source_hashes=sources, subset_sha256=pipeline.json_hash(subset),
                      loss_weights=dict(contact=0. if row['variant'] == 'no_contact_loss' else 5.,
                                        normal=0. if row['variant'] == 'no_contact_loss' else .5,
                                        smooth=0. if row['variant'] == 'no_contact_loss' else .1, collision=.002),
                      validation_weights=dict(contact=5., normal=.5, smooth=.1, collision=.002),
                      validation_noise_seed=202610110077, validation_count=1000,
                      adapter_config=dict(task_conditioning='none' if row['variant'] == 'no_geometry_tokens' else 'geometry',
                                          blocks=[0, 3, 7], bottleneck=64, physical_length_scale_m=2.))
        config['contract_sha256'] = pipeline.json_hash(config)
        put(directory / 'resolved_training_config.json', config)
        put(directory / 'training_subset.json', subset)
        (directory / 'best.pt').write_bytes(('fake selected checkpoint ' + row['id']).encode())
        put(directory / 'run_status.json', dict(status='complete', step=8000, planned_steps=8000,
                                               train_experts=row['train_size'], validation_experts=1000,
                                               successful_sample_presentations=512000,
                                               contract_sha256=config['contract_sha256'], frozen_base_unchanged=True,
                                               selected_checkpoint_sha256=pipeline.file_sha256(directory / 'best.pt'),
                                               best_step=7000))
    return experiment, protocol


def resign_config(directory: Path, config: dict) -> None:
    config.pop('contract_sha256', None)
    config['contract_sha256'] = pipeline.json_hash(config)
    put(directory / 'resolved_training_config.json', config)
    status = pipeline.read_json(directory / 'run_status.json')
    status['contract_sha256'] = config['contract_sha256']
    put(directory / 'run_status.json', status)


def test_all_21_selected_model_contracts_are_verified(frozen_models):
    experiment, protocol = frozen_models
    evidence = pipeline.verify_training_models(experiment, protocol)
    assert set(evidence) == {row['id'] for row in pipeline.approved_rows()}
    assert len(evidence) == 21
    assert all(value['best_step'] == 7000 for value in evidence.values())


@pytest.mark.parametrize('tamper', ['checkpoint', 'seed', 'budget', 'conditioning', 'source', 'missing'])
def test_model_tampering_cannot_advance_pipeline(frozen_models, tamper):
    experiment, protocol = frozen_models
    directory = experiment / 'models/geometry_n0500_seed17'
    if tamper == 'checkpoint':
        (directory / 'best.pt').write_bytes(b'different weights')
    elif tamper == 'missing':
        (directory / 'best.pt').unlink()
    elif tamper == 'budget':
        status = pipeline.read_json(directory / 'run_status.json')
        status['successful_sample_presentations'] -= 64
        put(directory / 'run_status.json', status)
    else:
        config = pipeline.read_json(directory / 'resolved_training_config.json')
        if tamper == 'seed':
            config['seed'] = 29
        elif tamper == 'conditioning':
            config['adapter_config']['task_conditioning'] = 'none'
        else:
            config['source_hashes']['model.py'] = '0' * 64
        resign_config(directory, config)
    with pytest.raises(pipeline.PipelineError):
        pipeline.verify_training_models(experiment, protocol)


def test_whole_map_subsets_must_remain_nested(frozen_models):
    experiment, protocol = frozen_models
    for seed in (17, 29, 43):
        directory = experiment / 'models' / f'geometry_n1000_seed{seed}'
        subset = dict(map_count=50, selected_map_ids=[f'map_{i:04d}' for i in range(25, 75)])
        put(directory / 'training_subset.json', subset)
        config = pipeline.read_json(directory / 'resolved_training_config.json')
        config['subset_sha256'] = pipeline.json_hash(subset)
        resign_config(directory, config)
    with pytest.raises(pipeline.PipelineError, match='not nested'):
        pipeline.verify_training_models(experiment, protocol)


def benchmark_definition(tmp_path) -> dict:
    return dict(source_map_count=1000, conditions_per_source=10,
                families=[f'family_{i}' for i in range(10)], benchmark_id='fake_benchmark', base_seed=23,
                maximum_map_replacements=8, maximum_condition_retries=24,
                oracle_solve_time_s=.75, oracle_attempts=24, collector_root=str(tmp_path / 'collector'),
                exclusion_datasets=[str(tmp_path / 'dataset')])


def test_stages_keep_the_approved_full_workload_and_worker_budgets(tmp_path):
    experiment, benchmark = tmp_path / 'experiment', tmp_path / 'benchmark'
    put(benchmark / 'provenance.json', {'definition': benchmark_definition(tmp_path)})
    stages = pipeline.build_stages(experiment, benchmark, {'base_checkpoint': '/tmp/fake_base.pt'},
                                   '/tmp/fake_python', tmp_path / 'reports')
    assert [stage.name for stage in stages] == ['benchmark', 'generate', 'score', 'report']
    assert stages[0].command[stages[0].command.index('--workers') + 1] == '1'
    assert stages[0].command[stages[0].command.index('--source-map-count') + 1] == '1000'
    assert '--resume' in stages[0].command and stages[0].command.count('--family') == 10
    assert stages[2].command[stages[2].command.index('--workers') + 1] == '2'
    assert all('--pilot' not in stage.command and '--limit-maps' not in stage.command for stage in stages)


@pytest.fixture
def stage_artifacts(frozen_models, tmp_path):
    experiment, protocol = frozen_models
    benchmark, reports = tmp_path / 'benchmark', tmp_path / 'reports'
    put(experiment / 'protocol.json', protocol)
    put(benchmark / 'cases.json', {'fake': 'all fixed requests'})
    put(benchmark / 'manifest.json', dict(status='frozen', source_map_count=1000, condition_count=10000,
                                         scenario_case_count=20000,
                                         input_files_sha256={'cases.json': pipeline.file_sha256(benchmark / 'cases.json')}))
    manifest_hash = pipeline.file_sha256(benchmark / 'manifest.json')
    ids = ['base', *[row['id'] for row in pipeline.approved_rows()]]
    for model_id in ids:
        directory = experiment / 'evaluation' / model_id
        contract = dict(model_id=model_id, benchmark_manifest_sha256=manifest_hash,
                        base_sha256=protocol['base_sha256'])
        contract['contract_sha256'] = pipeline.json_hash(contract)
        put(directory / 'contract.json', contract)
        for filename in ('generation_status.json', 'scoring_status.json'):
            put(directory / filename, dict(status='complete', complete_maps=1000, total_maps=1000,
                                           contract_sha256=contract['contract_sha256']))
    reports.mkdir()
    (reports / 'research_report_zh.md').write_text('fake complete research report\n')
    (reports / 'figure.svg').write_text('<svg>fake scientific figure</svg>')
    put(reports / 'report.json', dict(status='complete', shortfalls=[], models={key: {} for key in ids},
                                     protocol_file_sha256=pipeline.file_sha256(experiment / 'protocol.json'),
                                     benchmark_manifest_sha256=manifest_hash,
                                     report_source_sha256=pipeline.file_sha256(pipeline.ROOT / 'hnuter_dit/report_contact_experiment.py'),
                                     figures={'figure.svg': pipeline.file_sha256(reports / 'figure.svg')}))
    return SimpleNamespace(experiment=experiment, benchmark=benchmark, protocol=protocol, reports=reports, ids=ids)


@pytest.mark.parametrize('name', ['benchmark', 'generate', 'score', 'report'])
def test_stage_acceptance_requires_full_fixed_artifact_set(stage_artifacts, name):
    fixture = stage_artifacts
    evidence = pipeline.verify_stage_outputs(pipeline.Stage(name, []), fixture.experiment,
                                              fixture.benchmark, fixture.protocol, fixture.reports)
    if name in ('generate', 'score'):
        assert len(evidence['complete_models']) == 22
        assert evidence['complete_models'] == fixture.ids
    elif name == 'report':
        assert evidence['models'] == 22
    else:
        assert evidence['source_maps'] == 1000 and evidence['conditions'] == 10000


@pytest.mark.parametrize('name', ['generate', 'score'])
def test_last_of_22_model_artifacts_cannot_be_missing_or_partial(stage_artifacts, name):
    fixture = stage_artifacts
    filename = 'generation_status.json' if name == 'generate' else 'scoring_status.json'
    path = fixture.experiment / 'evaluation' / fixture.ids[-1] / filename
    status = pipeline.read_json(path)
    status['complete_maps'] = 999
    put(path, status)
    with pytest.raises(pipeline.PipelineError, match=fixture.ids[-1]):
        pipeline.verify_stage_outputs(pipeline.Stage(name, []), fixture.experiment,
                                      fixture.benchmark, fixture.protocol, fixture.reports)


def test_report_figure_tampering_is_rejected(stage_artifacts):
    fixture = stage_artifacts
    (fixture.reports / 'figure.svg').write_text('changed figure')
    with pytest.raises(pipeline.PipelineError, match='figure hash mismatch'):
        pipeline.verify_stage_outputs(pipeline.Stage('report', []), fixture.experiment,
                                      fixture.benchmark, fixture.protocol, fixture.reports)


@pytest.mark.parametrize('tamper', ['semantic_hash', 'wrong_model'])
def test_generation_contract_cannot_be_relabelled(stage_artifacts, tamper):
    fixture = stage_artifacts
    directory = fixture.experiment / 'evaluation/base'
    contract = pipeline.read_json(directory / 'contract.json')
    if tamper == 'semantic_hash':
        contract['unrecorded_setting'] = 'changed'
    else:
        contract['model_id'] = fixture.ids[-1]
        contract.pop('contract_sha256')
        contract['contract_sha256'] = pipeline.json_hash(contract)
        status = pipeline.read_json(directory / 'generation_status.json')
        status['contract_sha256'] = contract['contract_sha256']
        put(directory / 'generation_status.json', status)
    put(directory / 'contract.json', contract)
    with pytest.raises(pipeline.PipelineError, match='mismatched model base'):
        pipeline.verify_stage_outputs(pipeline.Stage('generate', []), fixture.experiment,
                                      fixture.benchmark, fixture.protocol, fixture.reports)


def test_frozen_benchmark_input_tampering_is_rejected(stage_artifacts):
    fixture = stage_artifacts
    put(fixture.benchmark / 'cases.json', {'changed': 'requests'})
    with pytest.raises(pipeline.PipelineError, match='input hash mismatch'):
        pipeline.verify_stage_outputs(pipeline.Stage('benchmark', []), fixture.experiment,
                                      fixture.benchmark, fixture.protocol, fixture.reports)


def test_source_contract_refuses_mixed_resume_inputs(frozen_models, tmp_path, monkeypatch):
    experiment, protocol = frozen_models
    root, benchmark = tmp_path / 'source_root', tmp_path / 'benchmark'
    source = root / 'hnuter_dit/fake.py'
    source.parent.mkdir(parents=True)
    source.write_text('version_one = 1\n')
    monkeypatch.setattr(pipeline, 'ROOT', root)
    monkeypatch.setattr(pipeline, 'PIPELINE_SOURCES', ('fake.py',))
    put(experiment / 'protocol.json', protocol)
    protocol = protocol | {'protocol_sha256': 'fake approved protocol hash'}
    put(benchmark / 'provenance.json', {'source_sha256': {}, 'definition': benchmark_definition(tmp_path)})
    initial = pipeline.ensure_source_contract(experiment, benchmark, protocol)
    assert pipeline.ensure_source_contract(experiment, benchmark, protocol) == initial
    source.write_text('version_two = 2\n')
    with pytest.raises(pipeline.PipelineError, match='mix source/input'):
        pipeline.ensure_source_contract(experiment, benchmark, protocol)
    with pytest.raises(pipeline.PipelineError, match='changed'):
        pipeline.verify_source_contract(initial)


def test_pipeline_lock_excludes_a_second_driver(tmp_path):
    lock = tmp_path / 'pipeline.lock'
    with pipeline.pipeline_lock(lock):
        with pytest.raises(pipeline.PipelineError, match='another driver'):
            with pipeline.pipeline_lock(lock):
                pytest.fail('second driver acquired an occupied lock')
    with pipeline.pipeline_lock(lock):
        assert int(lock.read_text()) == os.getpid()


def test_atomic_status_never_replaces_valid_json_with_partial_write(tmp_path, monkeypatch):
    path = tmp_path / 'state/status.json'
    pipeline.atomic_json(path, {'status': 'old'})
    seen = []
    replace = pipeline.os.replace
    def inspect_replace(source, destination):
        seen.append(json.loads(Path(source).read_text()))
        assert json.loads(path.read_text()) == {'status': 'old'}
        replace(source, destination)
    monkeypatch.setattr(pipeline.os, 'replace', inspect_replace)
    pipeline.atomic_json(path, {'status': 'complete', 'models': 22})
    assert seen == [{'status': 'complete', 'models': 22}]
    assert pipeline.read_json(path) == seen[0]
    assert not list(path.parent.glob('*.tmp'))


def test_run_stage_inherits_guarded_group_and_logs_nonzero_exit(tmp_path, monkeypatch):
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs['stdout'].write('fake child output\n')
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_run)
    with pytest.raises(pipeline.PipelineError, match='exited 7'):
        pipeline.run_stage(pipeline.Stage('score', ['/tmp/fake_command']), tmp_path / 'logs', {'MOCK': '1'})
    assert len(calls) == 1
    assert 'start_new_session' not in calls[0][1] and 'preexec_fn' not in calls[0][1]
    assert calls[0][1]['env'] == {'MOCK': '1'}
    assert 'fake child output' in (tmp_path / 'logs/score.log').read_text()


@pytest.fixture
def driver(tmp_path, monkeypatch):
    experiment, benchmark = tmp_path / 'experiment', tmp_path / 'benchmark'
    args = pipeline.parser().parse_args(['--experiment', str(experiment), '--benchmark', str(benchmark),
                                         '--training-already-running', '--poll-seconds', '.001'])
    put(experiment / 'thermal/pipeline/status.json', {'child': {'pid': os.getpid()}})
    binding = dict(child=identity(510101), guardian=identity(510102), command=['fake matrix'])
    healths = iter([dict(complete=False, binding=binding, guard_status='paused_cooling', matrix_status='training'),
                    dict(complete=True, binding=binding, guard_status='complete',
                         matrix_status='training_complete_evaluation_pending')])
    stages = [pipeline.Stage(name, ['/tmp/fake_command', name])
              for name in ('benchmark', 'generate', 'score', 'report')]
    launched, guard_checks, source_checks = [], [], []
    monkeypatch.setattr(pipeline.time, 'sleep', lambda _seconds: None)
    monkeypatch.setattr(pipeline, 'load_protocol', lambda _experiment: {'protocol_sha256': 'approved'})
    monkeypatch.setattr(pipeline, 'ensure_source_contract', lambda *_args: {'contract_sha256': 'sources'})
    monkeypatch.setattr(pipeline, 'verify_source_contract', lambda contract: source_checks.append(contract))
    monkeypatch.setattr(pipeline, 'verify_pipeline_guard',
                        lambda state_dir, binding=None: guard_checks.append((state_dir, binding)) or {'pid': 510102})
    monkeypatch.setattr(pipeline, 'inspect_training', lambda *_args: next(healths))
    monkeypatch.setattr(pipeline, 'verify_training_models', lambda *_args: {'verified_models': 21})
    monkeypatch.setattr(pipeline, 'build_stages', lambda *_args: stages)
    monkeypatch.setattr(pipeline, 'verify_stage_outputs', lambda stage, *_args: {'verified_stage': stage.name})
    def fake_stage(stage, _logs, env):
        launched.append(stage.name)
        assert env['OMP_NUM_THREADS'] == env['MKL_NUM_THREADS'] == env['OPENBLAS_NUM_THREADS'] == '1'
        return dict(log=str(tmp_path / f'{stage.name}.log'), elapsed_s=.01, exit_code=0)
    monkeypatch.setattr(pipeline, 'run_stage', fake_stage)
    return SimpleNamespace(args=args, experiment=experiment, launched=launched, binding=binding,
                           guard_checks=guard_checks, source_checks=source_checks, stages=stages,
                           fake_stage=fake_stage)


def test_driver_waits_for_guarded_training_and_runs_all_stages_in_order(driver):
    state = pipeline.run_pipeline(driver.args)
    assert state['status'] == 'complete'
    assert driver.launched == ['benchmark', 'generate', 'score', 'report']
    assert state['completed_stages'] == driver.launched
    assert state['training_models'] == {'verified_models': 21}
    assert set(state['stage_evidence']) == set(driver.launched)
    assert pipeline.read_json(driver.experiment / 'pipeline_training_binding.json') == driver.binding
    assert pipeline.read_json(driver.experiment / 'pipeline_status.json')['status'] == 'complete'
    assert len(driver.guard_checks) >= 11 and len(driver.source_checks) >= 10


@pytest.mark.parametrize('failure', ['training_guard', 'model_contract', 'stage_exit', 'stage_output', 'source_change'])
def test_driver_fails_closed_without_starting_later_stages(driver, monkeypatch, failure):
    def fail(*_args, **_kwargs):
        raise pipeline.PipelineError(f'isolation fixture {failure}')
    if failure == 'training_guard':
        monkeypatch.setattr(pipeline, 'inspect_training', fail)
    elif failure == 'model_contract':
        monkeypatch.setattr(pipeline, 'verify_training_models', fail)
    elif failure == 'stage_exit':
        def exit_stage(stage, logs, env):
            if stage.name == 'generate':
                driver.launched.append(stage.name)
                fail()
            return driver.fake_stage(stage, logs, env)
        monkeypatch.setattr(pipeline, 'run_stage', exit_stage)
    elif failure == 'stage_output':
        def invalid_output(stage, *_args):
            if stage.name == 'generate':
                fail()
            return {'verified_stage': stage.name}
        monkeypatch.setattr(pipeline, 'verify_stage_outputs', invalid_output)
    else:
        def reject_changed_source(_contract):
            if driver.launched:
                fail()
        monkeypatch.setattr(pipeline, 'verify_source_contract', reject_changed_source)
    with pytest.raises(pipeline.PipelineError, match=failure):
        pipeline.run_pipeline(driver.args)
    status = pipeline.read_json(driver.experiment / 'pipeline_status.json')
    assert status['status'] == 'failed' and status['error_type'] == 'PipelineError'
    if failure in ('training_guard', 'model_contract'):
        assert not driver.launched
    elif failure == 'source_change':
        assert driver.launched == ['benchmark'] and not status['completed_stages']
    else:
        assert driver.launched == ['benchmark', 'generate']
        assert status['completed_stages'] == ['benchmark']


def test_resume_rechecks_artifacts_instead_of_trusting_old_complete_status(driver):
    put(driver.experiment / 'pipeline_status.json', {'status': 'complete', 'completed_stages': ['report']})
    state = pipeline.run_pipeline(driver.args)
    assert state['completed_stages'] == ['benchmark', 'generate', 'score', 'report']
    assert driver.launched == state['completed_stages']


def test_training_wait_timeout_is_failure_and_does_not_start_evaluation(driver, monkeypatch):
    driver.args.wait_timeout = .001
    ticks = itertools.count()
    monkeypatch.setattr(pipeline.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(pipeline, 'inspect_training', lambda *_args: dict(
        complete=False, binding=driver.binding, guard_status='paused_cooling', matrix_status='training'))
    with pytest.raises(pipeline.PipelineError, match='deadline elapsed'):
        pipeline.run_pipeline(driver.args)
    assert not driver.launched
    assert pipeline.read_json(driver.experiment / 'pipeline_status.json')['status'] == 'failed'
