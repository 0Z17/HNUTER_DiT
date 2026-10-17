"""Resume the guarded Contact matrix, benchmark, evaluation and full report.

Launch through ``hnuter_dit.thermal_run`` with state directory
``EXPERIMENT/thermal/pipeline``. This driver creates no sessions and sends no
signals. Its subprocesses inherit its guarded process group. A live training
matrix is observed through its independent guardian; failure never starts the
next stage. Every invocation rechecks immutable sources and resume artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterator, NamedTuple

from .thermal_run import process_identity, remaining_owned_members

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / 'artifacts/contact_adapter_multifamily_v001'
DEFAULT_BENCHMARK = Path('/home/z017/research/HNUTER_datasets/surface_contact_adapter_benchmark_10000cond_v001')
IDENTITY_FIELDS = ('pid', 'start_ticks', 'uid', 'pgid', 'sid')
TRAINER_SOURCES = ('contact_adapter.py', 'train_contact_adapter.py', 'model.py',
                   'geometry.py', 'collision_loss.py', 'training.py')
PIPELINE_SOURCES = ('__init__.py', *TRAINER_SOURCES, 'contact_experiment.py', 'contact_pipeline.py',
                    'thermal_run.py', 'contact_benchmark.py', 'evaluate_obstacle_shift.py',
                    'evaluate_contact_experiment.py', 'contact_inference.py',
                    'contact_metrics.py', 'contact_projection.py', 'report_contact_experiment.py')


class PipelineError(RuntimeError):
    pass


class Stage(NamedTuple):
    name: str
    command: list[str]


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise PipelineError(f'cannot read {path}: {error}') from error
    if not isinstance(value, dict):
        raise PipelineError(f'expected JSON object: {path}')
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    except OSError as error:
        raise PipelineError(f'cannot hash {path}: {error}') from error
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def approved_rows() -> list[dict]:
    rows = [dict(id=f'geometry_n{size:04d}_seed{seed}', variant='geometry', train_size=size,
                 seed=seed, extra_args=[]) for size in (500, 1000, 2000, 4000, 8000) for seed in (17, 29, 43)]
    for seed in (17, 29, 43):
        rows += [dict(id=f'no_contact_loss_n8000_seed{seed}', variant='no_contact_loss', train_size=8000,
                      seed=seed, extra_args=['--contact-weight', '0', '--normal-weight', '0', '--smooth-weight', '0']),
                 dict(id=f'no_geometry_tokens_n8000_seed{seed}', variant='no_geometry_tokens', train_size=8000,
                      seed=seed, extra_args=['--task-conditioning', 'none'])]
    return rows


def load_protocol(experiment: Path) -> dict:
    value = read_json(experiment / 'protocol.json')
    definition = {key: item for key, item in value.items() if key not in ('protocol_sha256', 'created_utc')}
    if value.get('protocol_sha256') != json_hash(definition):
        raise PipelineError('experiment protocol hash mismatch')
    expected = dict(variants=approved_rows(), optimizer_updates=8000, batch_size=64,
                    data_subset_seed=20261011, new_source_maps=1000, new_test_conditions=10000,
                    sampling_steps=25, diffusion_steps=100, multimodal_subset_conditions=1000,
                    multimodal_candidates_per_condition=8, primary_candidates_per_condition=1)
    if any(value.get(key) != item for key, item in expected.items()):
        raise PipelineError('experiment is not the approved full Contact matrix')
    return value


def _saved_identity(saved: dict, label: str, *, dedicated: bool = False) -> dict:
    if any(not isinstance(saved.get(key), int) for key in IDENTITY_FIELDS):
        raise PipelineError(f'{label}: incomplete process identity')
    if saved['uid'] != os.getuid() or saved['pid'] <= 1 or saved['start_ticks'] < 1:
        raise PipelineError(f'{label}: process is not owned by the current user')
    if dedicated and not saved['pid'] == saved['pgid'] == saved['sid']:
        raise PipelineError(f'{label}: not a dedicated process group/session')
    return {key: saved[key] for key in IDENTITY_FIELDS}


def verify_identity(saved: dict, label: str) -> dict:
    expected = _saved_identity(saved, label)
    current = process_identity(expected['pid'])
    if current is None or current.get('state') in ('Z', 'X') or any(current.get(key) != item for key, item in expected.items()):
        raise PipelineError(f'{label}: process died or its PID/start ticks/UID/PGID/SID changed')
    return current


def process_command(pid: int) -> list[str]:
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    except (OSError, UnicodeDecodeError) as error:
        raise PipelineError(f'cannot inspect command of PID {pid}') from error


def _guardian_command(pid: int, state_dir: Path) -> None:
    command = process_command(pid)
    command = command[:command.index('--')] if '--' in command else command
    if 'hnuter_dit.thermal_run' not in command or '--state-dir' not in command:
        raise PipelineError('recorded guardian PID is not the thermal wrapper')
    try:
        recorded = Path(command[command.index('--state-dir') + 1]).resolve()
    except (IndexError, ValueError) as error:
        raise PipelineError('guardian command has no valid state directory') from error
    if recorded != state_dir.resolve():
        raise PipelineError('guardian command belongs to a different state directory')
    expected = {'--cpu-pause-c': 90., '--cpu-resume-c': 80., '--gpu-pause-c': 83.,
                '--gpu-resume-c': 75., '--cool-seconds': 10., '--poll-seconds': 1.}
    for option, default in expected.items():
        try:
            value = float(command[command.index(option) + 1]) if option in command else default
        except (ValueError, IndexError) as error:
            raise PipelineError('guardian has invalid thermal policy') from error
        if value != default:
            raise PipelineError(f'guardian thermal policy differs: {option}')
    if '--gpu' not in command:
        raise PipelineError('Contact pipeline requires a CPU and GPU guardian')


def _fresh_guard(status: dict, stale_seconds: float) -> None:
    try:
        updated = datetime.fromisoformat(status['updated_utc'])
        age = (datetime.now(timezone.utc) - updated).total_seconds()
    except (KeyError, TypeError, ValueError) as error:
        raise PipelineError('guardian has no valid heartbeat timestamp') from error
    if age < -30 or age > stale_seconds:
        raise PipelineError(f'guardian heartbeat is stale ({age:.1f}s)')


def _matrix_command(command: list[str], experiment: Path) -> None:
    if 'hnuter_dit.contact_experiment' not in command:
        raise PipelineError('training guardian does not own the matrix command')
    try:
        output = Path(command[command.index('--output') + 1]) if '--output' in command else DEFAULT_EXPERIMENT
    except IndexError as error:
        raise PipelineError('training matrix command has no valid output') from error
    if not output.is_absolute():
        output = ROOT / output
    if output.resolve() != experiment.resolve():
        raise PipelineError('training guardian owns a different experiment directory')


def inspect_training(experiment: Path, binding: dict | None = None, stale_seconds: float = 120.) -> dict:
    """Return waiting/completed health, pinning the guardian's full /proc identity.

    A fresh driver may accept an already terminated *successful* group after all
    21 checkpoint contracts pass. An active group always requires a live guard.
    The completed matrix/guardian race is a waiting state, never completion.
    """
    state_dir = experiment / 'thermal/training'
    guard = read_json(state_dir / 'status.json')
    matrix = read_json(experiment / 'experiment_status.json')
    terminal = matrix.get('status') == 'training_complete_evaluation_pending'
    if matrix.get('status') not in ('training', 'training_complete_evaluation_pending'):
        raise PipelineError(f"training matrix failed or has unexpected state: {matrix.get('status')}")
    if guard.get('status') not in ('running', 'paused_cooling', 'complete'):
        raise PipelineError(f"training guardian failed or stopped: {guard.get('status')}")
    child = _saved_identity(guard.get('child', {}), 'training child', dedicated=True)
    command = guard.get('command', [])
    _matrix_command(command, experiment)
    if binding is not None and child != binding.get('child'):
        raise PipelineError('training child identity changed since this pipeline bound it')
    current_guard = process_identity(guard.get('guard_pid', -1))
    guard_identity = binding.get('guardian') if binding is not None else None
    if guard_identity is not None and guard.get('guard_pid') != guard_identity['pid']:
        raise PipelineError('training guardian PID changed since pipeline binding')
    if current_guard is not None and current_guard.get('state') not in ('Z', 'X'):
        if guard_identity is None:
            guard_identity = _saved_identity(current_guard, 'training guardian')
        verify_identity(guard_identity, 'training guardian')
        if current_guard.get('state') in ('T', 't'):
            raise PipelineError('training guardian itself is stopped')
        if current_guard['pgid'] == child['pgid'] or current_guard['sid'] == child['sid']:
            raise PipelineError('training guardian is inside the guarded child group')
        _guardian_command(guard['guard_pid'], state_dir)
    elif guard.get('status') != 'complete':
        raise PipelineError('training guardian is no longer alive')
    actual_binding = dict(child=child, guardian=guard_identity, command=command)
    if binding is not None and binding.get('command') != command:
        raise PipelineError('training command changed since pipeline binding')
    if guard.get('status') == 'complete':
        if guard.get('exit_code') != 0 or not terminal or remaining_owned_members(child):
            raise PipelineError('training guardian ended without a successful, fully exited matrix')
        if (len(matrix.get('complete_models', [])) != 21 or
                set(matrix.get('complete_models', [])) != {row['id'] for row in approved_rows()} or matrix.get('total_models') != 21):
            raise PipelineError('training matrix did not complete all 21 approved models')
        return dict(complete=True, binding=actual_binding, guard_status='complete', matrix_status=matrix['status'])
    _fresh_guard(guard, stale_seconds)
    current_child = process_identity(child['pid'])
    if current_child is not None:
        if terminal and current_child.get('state') == 'Z':
            if any(current_child.get(key) != value for key, value in child.items()):
                raise PipelineError('completed training child identity changed')
        else:
            verify_identity(child, 'training child')
    elif not terminal:
        raise PipelineError('training leader exited before the matrix completed')
    return dict(complete=False, binding=actual_binding, guard_status=guard['status'], matrix_status=matrix['status'])


def verify_training_models(experiment: Path, protocol: dict) -> dict:
    expected_sources = {name: file_sha256(ROOT / 'hnuter_dit' / name) for name in TRAINER_SOURCES}
    statistics = Path(protocol['base_checkpoint']).parents[2] / 'canonical_statistics.json'
    statistics_hash = file_sha256(statistics)
    evidence, subsets = {}, {}
    for row in approved_rows():
        directory = experiment / 'models' / row['id']
        status = read_json(directory / 'run_status.json')
        config = read_json(directory / 'resolved_training_config.json')
        subset = read_json(directory / 'training_subset.json')
        weights = dict(contact=0. if row['variant'] == 'no_contact_loss' else 5.,
                       normal=0. if row['variant'] == 'no_contact_loss' else .5,
                       smooth=0. if row['variant'] == 'no_contact_loss' else .1, collision=.002)
        expected = dict(base_sha256=protocol['base_sha256'], dataset_freeze_sha256=protocol['dataset_freeze_sha256'],
                        base_path=protocol['base_checkpoint'], dataset_root=protocol['training_dataset'],
                        normalization_reference=str(statistics), normalization_sha256=statistics_hash,
                        train_size=row['train_size'], seed=row['seed'], training_steps=8000, batch_size=64,
                        data_subset_seed=20261011, source_hashes=expected_sources, loss_weights=weights,
                        validation_weights=dict(contact=5., normal=.5, smooth=.1, collision=.002),
                        validation_noise_seed=202610110077, validation_count=1000)
        definition = {key: item for key, item in config.items() if key not in ('contract_sha256', 'created_at_utc')}
        if config.get('contract_sha256') != json_hash(definition) or any(config.get(key) != item for key, item in expected.items()):
            raise PipelineError(f"{row['id']}: training configuration differs from approved row")
        if config.get('adapter_config', {}).get('task_conditioning') != ('none' if row['variant'] == 'no_geometry_tokens' else 'geometry'):
            raise PipelineError(f"{row['id']}: wrong task-conditioning control")
        adapter = config['adapter_config']
        if adapter.get('blocks') != [0, 3, 7] or adapter.get('bottleneck') != 64 or adapter.get('physical_length_scale_m') != 2.:
            raise PipelineError(f"{row['id']}: wrong task adapter architecture")
        if (status.get('status') != 'complete' or status.get('step') != 8000 or
                status.get('planned_steps') != 8000 or status.get('train_experts') != row['train_size'] or
                status.get('validation_experts') != 1000 or status.get('successful_sample_presentations') != 512000 or
                status.get('contract_sha256') != config['contract_sha256'] or not status.get('frozen_base_unchanged')):
            raise PipelineError(f"{row['id']}: incomplete budget or unfrozen base")
        checkpoint_hash = file_sha256(directory / 'best.pt')
        if checkpoint_hash != status.get('selected_checkpoint_sha256'):
            raise PipelineError(f"{row['id']}: selected checkpoint hash mismatch")
        if (json_hash(subset) != config.get('subset_sha256') or not isinstance(subset.get('map_count'), int) or
                subset['map_count'] * 20 != row['train_size']):
            raise PipelineError(f"{row['id']}: subset hash or whole-map count mismatch")
        map_ids = subset.get('selected_map_ids', [])
        if len(map_ids) != len(set(map_ids)) or len(map_ids) != subset['map_count']:
            raise PipelineError(f"{row['id']}: invalid selected source maps")
        previous = subsets.setdefault(row['train_size'], config['subset_sha256'])
        if previous != config['subset_sha256']:
            raise PipelineError('model seeds or controls used different training subsets')
        evidence[row['id']] = dict(checkpoint_sha256=checkpoint_hash, config_sha256=file_sha256(directory / 'resolved_training_config.json'),
                                   subset_sha256=config['subset_sha256'], best_step=status.get('best_step'))
    sets = {size: set(read_json(experiment / 'models' / f'geometry_n{size:04d}_seed17' / 'training_subset.json')['selected_map_ids'])
            for size in (500, 1000, 2000, 4000, 8000)}
    if any(not sets[small].issubset(sets[large]) for small, large in zip(sets, list(sets)[1:])):
        raise PipelineError('training map subsets are not nested')
    return evidence


def verify_source_contract(contract: dict) -> None:
    definition = {key: value for key, value in contract.items() if key not in ('contract_sha256', 'created_utc')}
    if contract.get('contract_sha256') != json_hash(definition):
        raise PipelineError('pipeline source contract hash mismatch')
    for path, digest in (contract['source_files'] | contract['input_files']).items():
        if file_sha256(Path(path)) != digest:
            raise PipelineError(f'immutable pipeline source/input changed: {path}')


def ensure_source_contract(experiment: Path, benchmark: Path, protocol: dict) -> dict:
    path = experiment / 'pipeline_source_contract.json'
    sources = {str((ROOT / 'hnuter_dit' / name).resolve()): file_sha256(ROOT / 'hnuter_dit' / name) for name in PIPELINE_SOURCES}
    inputs = {str((experiment / 'protocol.json').resolve()): file_sha256(experiment / 'protocol.json'),
              protocol['base_checkpoint']: protocol['base_sha256'],
              str(Path(protocol['training_dataset']) / 'dataset_freeze.json'): protocol['dataset_freeze_sha256']}
    statistics = Path(protocol['base_checkpoint']).parents[2] / 'canonical_statistics.json'
    inputs[str(statistics)] = file_sha256(statistics)
    provenance_path = benchmark / 'provenance.json'
    if not provenance_path.is_file():
        raise PipelineError('pipeline requires the initialized resumable benchmark provenance')
    provenance = read_json(provenance_path)
    sources.update(provenance['source_sha256'])
    collector = Path(provenance['definition']['collector_root'])
    native = collector / 'native/ompl_region_sampler'
    # The existing benchmark provenance covers Python physics/scene sources;
    # also pin its locally compiled deterministic sampling implementation.
    native_files = [collector / 'native/__init__.py', native / '__init__.py',
                    native / 'region_sampler.cpp', native / 'build_extension.py', native / 'CMakeLists.txt',
                    *native.glob('*.so')]
    for source in native_files:
        if source.is_file():
            sources[str(source.resolve())] = file_sha256(source)
    inputs[str(provenance_path.resolve())] = file_sha256(provenance_path)
    definition = dict(schema_version='contact_pipeline_sources_v001', experiment=str(experiment), benchmark=str(benchmark),
                      protocol_sha256=protocol['protocol_sha256'], source_files=sources, input_files=inputs,
                      stage_policy='benchmark1000maps1worker; generate22; score22with2workers; strict_full_report')
    digest = json_hash(definition)
    if path.exists():
        result = read_json(path)
        if result.get('contract_sha256') != digest:
            raise PipelineError('pipeline resume would mix source/input contracts; use a fresh version')
    else:
        result = definition | dict(contract_sha256=digest, created_utc=datetime.now(timezone.utc).isoformat())
        verify_source_contract(result)
        atomic_json(path, result)
    verify_source_contract(result)
    return result


def build_stages(experiment: Path, benchmark: Path, protocol: dict, python: str, report_output: Path) -> list[Stage]:
    definition = read_json(benchmark / 'provenance.json')['definition']
    if definition.get('source_map_count') != 1000 or len(definition.get('families', [])) != 10 or definition.get('conditions_per_source') != 10:
        raise PipelineError('benchmark is not the planned 1000-map/10000-condition contract')
    command = [python, '-m', 'hnuter_dit.contact_benchmark', '--output', str(benchmark), '--resume',
               '--benchmark-id', definition['benchmark_id'], '--source-map-count', '1000', '--workers', '1',
               '--seed', str(definition['base_seed']), '--maximum-map-replacements', str(definition['maximum_map_replacements']),
               '--maximum-condition-retries', str(definition['maximum_condition_retries']),
               '--oracle-solve-time', str(definition['oracle_solve_time_s']), '--oracle-attempts', str(definition['oracle_attempts']),
               '--collector-root', definition['collector_root']]
    for family in definition['families']:
        command += ['--family', family]
    for dataset in definition['exclusion_datasets']:
        command += ['--exclude-dataset', dataset]
    evaluation = [python, '-m', 'hnuter_dit.evaluate_contact_experiment', '--experiment', str(experiment),
                  '--benchmark', str(benchmark), '--base', protocol['base_checkpoint']]
    return [Stage('benchmark', command), Stage('generate', evaluation + ['--stage', 'generate']),
            Stage('score', evaluation + ['--stage', 'score', '--workers', '2']),
            Stage('report', [python, '-m', 'hnuter_dit.report_contact_experiment', '--experiment', str(experiment),
                             '--benchmark', str(benchmark), '--output', str(report_output)])]


def verify_stage_outputs(stage: Stage, experiment: Path, benchmark: Path, protocol: dict, report_output: Path) -> dict:
    manifest = read_json(benchmark / 'manifest.json')
    if any(manifest.get(key) != value for key, value in dict(status='frozen', source_map_count=1000,
                                                           condition_count=10000, scenario_case_count=20000).items()):
        raise PipelineError('benchmark ended without all frozen cases')
    manifest_hash = file_sha256(benchmark / 'manifest.json')
    if stage.name == 'benchmark':
        for relative, digest in manifest['input_files_sha256'].items():
            if file_sha256(benchmark / relative) != digest:
                raise PipelineError(f'benchmark input hash mismatch: {relative}')
        return dict(manifest_sha256=manifest_hash, source_maps=1000, conditions=10000)
    ids = ['base', *[row['id'] for row in approved_rows()]]
    if stage.name in ('generate', 'score'):
        filename = 'generation_status.json' if stage.name == 'generate' else 'scoring_status.json'
        for model_id in ids:
            directory = experiment / 'evaluation' / model_id
            status, contract = read_json(directory / filename), read_json(directory / 'contract.json')
            definition = {key: value for key, value in contract.items() if key not in ('contract_sha256', 'created_utc')}
            if (status.get('status') != 'complete' or status.get('complete_maps') != 1000 or status.get('total_maps') != 1000 or
                    contract.get('contract_sha256') != json_hash(definition) or contract.get('model_id') != model_id or
                    status.get('contract_sha256') != contract.get('contract_sha256') or
                    contract.get('benchmark_manifest_sha256') != manifest_hash or contract.get('base_sha256') != protocol['base_sha256']):
                raise PipelineError(f'{stage.name}: incomplete or mismatched model {model_id}')
        return dict(complete_models=ids, manifest_sha256=manifest_hash)
    report = read_json(report_output / 'report.json')
    if (report.get('status') != 'complete' or report.get('shortfalls') != [] or set(report.get('models', {})) != set(ids) or
            report.get('protocol_file_sha256') != file_sha256(experiment / 'protocol.json') or
            report.get('benchmark_manifest_sha256') != manifest_hash or
            report.get('report_source_sha256') != file_sha256(ROOT / 'hnuter_dit/report_contact_experiment.py') or
            not (report_output / 'research_report_zh.md').is_file() or not report.get('figures')):
        raise PipelineError('full 22-model research report is incomplete or has mismatched provenance')
    for relative, digest in report['figures'].items():
        if file_sha256(report_output / relative) != digest:
            raise PipelineError(f'report figure hash mismatch: {relative}')
    return dict(report_sha256=file_sha256(report_output / 'report.json'), models=22,
                markdown_sha256=file_sha256(report_output / 'research_report_zh.md'), figures=report['figures'])


@contextmanager
def pipeline_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PipelineError('another driver owns this pipeline lock') from error
        stream.seek(0); stream.truncate(); stream.write(str(os.getpid()) + '\n'); stream.flush()
        yield


def verify_pipeline_guard(state_dir: Path, binding: dict | None = None) -> dict:
    current = process_identity(os.getpid())
    if current is None:
        raise PipelineError('cannot identify pipeline process')
    child = _saved_identity(current, 'pipeline', dedicated=True)
    status = read_json(state_dir / 'status.json')
    if status.get('status') not in ('running', 'paused_cooling'):
        raise PipelineError(f"pipeline guardian is not active: {status.get('status')}")
    if _saved_identity(status.get('child', {}), 'pipeline guardian child', dedicated=True) != child:
        raise PipelineError('outer guardian does not own this pipeline identity')
    guardian = process_identity(status.get('guard_pid', -1))
    if guardian is None or guardian.get('state') in ('T', 't', 'Z', 'X'):
        raise PipelineError('pipeline guardian is dead or stopped')
    actual = _saved_identity(guardian, 'pipeline guardian')
    if actual['pgid'] == child['pgid'] or actual['sid'] == child['sid']:
        raise PipelineError('pipeline guardian is inside the guarded child group')
    if binding is not None and binding != actual:
        raise PipelineError('pipeline guardian identity changed')
    _guardian_command(guardian['pid'], state_dir)
    _fresh_guard(status, 120.)
    return actual


def run_stage(stage: Stage, logs: Path, env: dict) -> dict:
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f'{stage.name}.log'
    began = time.monotonic()
    with path.open('a', buffering=1) as stream:
        stream.write(f'\nPIPELINE {datetime.now(timezone.utc).isoformat()} {json.dumps(stage.command)}\n')
        # No setsid: every helper and its worker children remain in our guarded group.
        result = subprocess.run(stage.command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise PipelineError(f'{stage.name} exited {result.returncode}; inspect {path}')
    return dict(log=str(path), elapsed_s=time.monotonic() - began, exit_code=0)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--experiment', type=Path, default=DEFAULT_EXPERIMENT)
    result.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    result.add_argument('--training-already-running', action='store_true')
    result.add_argument('--guardian-state-dir', type=Path, help='outer thermal wrapper directory; default EXPERIMENT/thermal/pipeline')
    result.add_argument('--report-output', type=Path, help='default EXPERIMENT/reports')
    result.add_argument('--poll-seconds', type=float, default=30.)
    result.add_argument('--wait-timeout', type=float, default=0., help='seconds to wait for training; 0 has no deadline; timeout is failed, never complete')
    return result


def run_pipeline(args: argparse.Namespace) -> dict:
    experiment, benchmark = args.experiment.resolve(), args.benchmark.resolve()
    report_output = (args.report_output or experiment / 'reports').resolve()
    guardian_dir = (args.guardian_state_dir or experiment / 'thermal/pipeline').resolve()
    if args.poll_seconds <= 0 or args.poll_seconds > 60 or args.wait_timeout < 0:
        raise PipelineError('poll seconds must be in (0,60] and wait timeout nonnegative')
    status_path, logs = experiment / 'pipeline_status.json', experiment / 'pipeline_logs'
    began = time.monotonic()
    with pipeline_lock(experiment / 'pipeline.lock'):
        state = dict(schema_version='contact_pipeline_status_v001', status='initializing', current_stage='initialize',
                     pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(), completed_stages=[], stage_evidence={})
        def update(**extra: Any) -> None:
            state.update(extra, updated_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic() - began)
            atomic_json(status_path, state)
        try:
            # The wrapper starts its child before writing the first heartbeat.
            startup_deadline = time.monotonic() + 20.
            while time.monotonic() < startup_deadline:
                if (guardian_dir / 'status.json').is_file() and read_json(guardian_dir / 'status.json').get('child', {}).get('pid') == os.getpid():
                    break
                time.sleep(.1)
            guardian = verify_pipeline_guard(guardian_dir)
            protocol = load_protocol(experiment)
            contract = ensure_source_contract(experiment, benchmark, protocol)
            update(source_contract_sha256=contract['contract_sha256'], protocol_sha256=protocol['protocol_sha256'])
            env = os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MPLCONFIGDIR=str(Path('/tmp') / f'contact_pipeline_mpl_{os.getuid()}'))
            if args.training_already_running:
                binding_path = experiment / 'pipeline_training_binding.json'
                binding = read_json(binding_path) if binding_path.exists() else None
                while True:
                    verify_source_contract(contract)
                    verify_pipeline_guard(guardian_dir, guardian)
                    health = inspect_training(experiment, binding)
                    binding = health['binding']
                    if not binding_path.exists():
                        atomic_json(binding_path, binding)
                    if health['complete']:
                        break
                    update(status='waiting_training', current_stage='training', training_guard_status=health['guard_status'],
                           training_matrix_status=health['matrix_status'], training_binding=binding)
                    if args.wait_timeout and time.monotonic() - began >= args.wait_timeout:
                        raise PipelineError('training wait deadline elapsed; evaluation was not started')
                    time.sleep(args.poll_seconds)
            else:
                old_guard_path = experiment / 'thermal/training/status.json'
                if old_guard_path.exists():
                    old_guard = read_json(old_guard_path)
                    if old_guard.get('child') and remaining_owned_members(old_guard['child']):
                        raise PipelineError('a training group is already alive; use --training-already-running')
                update(status='running', current_stage='training')
                run_stage(Stage('training', [sys.executable, '-m', 'hnuter_dit.contact_experiment', '--output', str(experiment),
                                            '--base', protocol['base_checkpoint'], '--dataset', protocol['training_dataset']]), logs, env)
            update(status='validating_training', current_stage='training', training_models=verify_training_models(experiment, protocol))
            for stage in build_stages(experiment, benchmark, protocol, sys.executable, report_output):
                verify_source_contract(contract)
                verify_pipeline_guard(guardian_dir, guardian)
                update(status='running', current_stage=stage.name, command=stage.command)
                print(f'STAGE {stage.name}', flush=True)
                runtime = run_stage(stage, logs, env)
                verify_source_contract(contract)
                verify_pipeline_guard(guardian_dir, guardian)
                evidence = verify_stage_outputs(stage, experiment, benchmark, protocol, report_output)
                state['stage_evidence'][stage.name] = runtime | evidence
                state['completed_stages'].append(stage.name)
                update()
            update(status='complete', current_stage='complete', report=str(report_output / 'report.json'))
            print(f'COMPLETE models=22 report={report_output / "report.json"}', flush=True)
            return state
        except BaseException as error:
            update(status='failed', error_type=type(error).__name__, error=str(error))
            raise


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        run_pipeline(args)
        return 0
    except (PipelineError, OSError, ValueError) as error:
        print(f'PIPELINE FAILED: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
