"""One-shot smooth-decoder supplement, after the complete frozen parent study.

Every subprocess stays inside the enclosing temperature guardian's group. The
parent study and its original projection-off/on outputs are read-only inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from .contact_pipeline import (
    PipelineError, Stage, _fresh_guard, _guardian_command, _saved_identity,
    approved_rows, file_sha256, load_protocol, pipeline_lock, read_json,
    run_stage, verify_identity, verify_pipeline_guard, verify_source_contract,
    verify_stage_outputs,
)
from .contact_smooth_projection import smooth_contact_decoder_parameters
from .thermal_run import atomic_json, process_identity, remaining_owned_members
from .train_contact_adapter import json_hash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT/'artifacts/contact_adapter_multifamily_v001'
DEFAULT_BENCHMARK = Path('/home/z017/research/HNUTER_datasets/surface_contact_adapter_benchmark_10000cond_v001')
SOURCES = ('contact_smooth_pipeline.py', 'contact_smooth_projection.py',
           'evaluate_contact_smooth_projection.py', 'report_contact_smooth_projection.py')


def ensure_contract(experiment: Path, benchmark: Path, supplement: Path) -> dict:
    parent_path = experiment/'pipeline_source_contract.json'
    parent = read_json(parent_path)
    verify_source_contract(parent)
    if Path(parent['experiment']).resolve() != experiment or Path(parent['benchmark']).resolve() != benchmark:
        raise PipelineError('smooth driver parent source contract belongs to another experiment')
    definition = dict(schema_version='contact_smooth_pipeline_sources_v001',
        experiment=str(experiment), benchmark=str(benchmark), supplement=str(supplement),
        parent_source_contract_sha256=parent['contract_sha256'],
        source_files=parent['source_files'] | {
            str((ROOT/'hnuter_dit'/name).resolve()): file_sha256(ROOT/'hnuter_dit'/name) for name in SOURCES},
        input_files=parent['input_files'] | {str(parent_path): file_sha256(parent_path)},
        decoder_parameters=smooth_contact_decoder_parameters(),
        stage_policy='wait_for_strict_parent_complete; score_all22_same_raw; strict_supplement_report')
    path = supplement/'pipeline_source_contract.json'
    if path.exists():
        value = read_json(path)
        if value.get('contract_sha256') != json_hash(definition):
            raise PipelineError('smooth driver immutable sources/parameters changed; use a fresh supplement version')
    else:
        value = definition | dict(contract_sha256=json_hash(definition), created_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(path, value)
    verify_source_contract(value)
    return value


class _ParentTransition(PipelineError):
    """Separate status files may straddle the parent's normal exit writes."""


def parent_health(experiment: Path, benchmark: Path, contract: dict, binding: dict | None) -> dict:
    """Pin the active parent group; fail if it vanishes before full completion."""
    for attempt in range(3):
        try:
            return _parent_health_once(experiment, benchmark, contract, binding)
        except _ParentTransition:
            if attempt == 2:
                raise
            time.sleep(.1)
    raise AssertionError('bounded parent snapshot loop did not return')


def _parent_health_once(experiment: Path, benchmark: Path, contract: dict, binding: dict | None) -> dict:
    state = read_json(experiment/'pipeline_status.json')
    guard_dir = experiment/'thermal/pipeline'
    guard = read_json(guard_dir/'status.json')
    if state.get('status') == 'failed':
        raise PipelineError(f"parent pipeline failed: {state.get('error', 'unknown error')}")
    if state.get('status') not in ('initializing', 'running', 'validating_training', 'waiting_training', 'complete'):
        raise PipelineError('parent pipeline has an unexpected state')
    if guard.get('status') not in ('running', 'paused_cooling', 'complete'):
        raise PipelineError(f"parent guardian is not healthy: {guard.get('status')}")
    child = _saved_identity(guard.get('child', {}), 'parent pipeline child', dedicated=True)
    command = guard.get('command', [])
    expected = {'--experiment': experiment, '--benchmark': benchmark}
    if 'hnuter_dit.contact_pipeline' not in command:
        raise PipelineError('parent guardian owns a different command')
    for option, path in expected.items():
        try:
            actual = Path(command[command.index(option)+1]).resolve()
        except (ValueError, IndexError) as error:
            raise PipelineError('parent guardian command lacks an input directory') from error
        if actual != path:
            raise PipelineError('parent guardian owns a different experiment or benchmark')
    if state.get('pid') != child['pid']:
        raise PipelineError('parent pipeline status does not belong to its guarded child')
    if (state.get('source_contract_sha256') != contract['parent_source_contract_sha256'] or
            state.get('protocol_sha256') != load_protocol(experiment)['protocol_sha256']):
        raise PipelineError('parent pipeline changed its frozen source or protocol binding')
    guardian = process_identity(guard.get('guard_pid', -1))
    if guard.get('status') == 'complete':
        # A completed historical group is authenticated by its saved binding,
        # full report and absence of owned members, never a reused current PID.
        identity = binding.get('guardian') if binding else None
    elif guardian is not None and guardian.get('state') not in ('Z', 'X'):
        identity = _saved_identity(guardian, 'parent guardian')
        if guardian.get('state') in ('T', 't'):
            raise PipelineError('parent guardian itself is stopped')
        if identity['pgid'] == child['pgid'] or identity['sid'] == child['sid']:
            raise PipelineError('parent guardian is inside its child group')
        try:
            _guardian_command(identity['pid'], guard_dir)
        except PipelineError as error:
            current = process_identity(identity['pid'])
            if current is None or current.get('state') in ('Z', 'X'):
                raise _ParentTransition('parent guardian exited during command inspection') from error
            raise
    else:
        raise _ParentTransition('parent guardian disappeared before study completion')
    actual_binding = dict(child=child, guardian=identity, command=command)
    if binding is not None and binding != actual_binding:
        raise PipelineError('parent PID/start ticks/UID/group/command changed after binding')
    if guard.get('status') == 'complete':
        if state.get('status') != 'complete':
            raise _ParentTransition('parent group ended without successful full study completion')
        if guard.get('exit_code') != 0 or remaining_owned_members(child):
            raise PipelineError('parent group ended without successful full study completion')
        protocol = load_protocol(experiment)
        report_output = Path(state['report']).parent
        evidence = verify_stage_outputs(Stage('report', []), experiment, benchmark, protocol, report_output)
        return dict(complete=True, binding=actual_binding, report_evidence=evidence,
                    parent_report_sha256=file_sha256(report_output/'report.json'))
    _fresh_guard(guard, 120.)
    if state.get('status') != 'complete':
        try:
            verify_identity(child, 'parent pipeline child')
        except PipelineError as error:
            current = process_identity(child['pid'])
            if current is None or current.get('state') in ('Z', 'X'):
                raise _ParentTransition('parent child exited during identity inspection') from error
            raise
    return dict(complete=False, binding=actual_binding, parent_stage=state.get('current_stage'),
                parent_status=state['status'], parent_guard_status=guard['status'])


def build_stages(experiment: Path, benchmark: Path, supplement: Path, python: str) -> list[Stage]:
    return [
        Stage('score', [python, '-m', 'hnuter_dit.evaluate_contact_smooth_projection',
                       '--experiment', str(experiment), '--benchmark', str(benchmark),
                       '--output', str(supplement), '--workers', '2']),
        Stage('report', [python, '-m', 'hnuter_dit.report_contact_smooth_projection',
                        '--experiment', str(experiment), '--benchmark', str(benchmark),
                        '--supplement', str(supplement), '--output', str(supplement/'reports')]),
    ]


def verify_output(stage: Stage, benchmark: Path, supplement: Path) -> dict:
    ids = ['base', *[row['id'] for row in approved_rows()]]
    protocol = read_json(supplement/'protocol.json')
    definition = {key: value for key, value in protocol.items() if key not in ('protocol_sha256', 'created_utc')}
    manifest_hash = file_sha256(benchmark/'manifest.json')
    if (protocol.get('protocol_sha256') != json_hash(definition) or
            protocol.get('benchmark_manifest_sha256') != manifest_hash or
            protocol.get('planned_models') != ids or protocol.get('condition_count') != 10000 or
            protocol.get('source_map_count') != 1000 or protocol.get('projection') != 'on_smooth' or
            protocol.get('decoder_parameters') != smooth_contact_decoder_parameters()):
        raise PipelineError('smooth supplement is not the fixed full research protocol')
    if stage.name == 'score':
        for model_id in ids:
            directory = supplement/'evaluation'/model_id
            status = read_json(directory/'scoring_status.json')
            model = read_json(directory/'contract.json')
            definition = {key: value for key, value in model.items() if key not in ('contract_sha256', 'created_utc')}
            if (status.get('status') != 'complete' or status.get('complete_maps') != 1000 or
                    status.get('total_maps') != 1000 or status.get('contract_sha256') != model.get('contract_sha256') or
                    status.get('planned_total_maps') != 1000 or status.get('pilot_limit_maps') != 0 or
                    any(status.get(key) != 34000 for key in ('score_rows', 'expected_score_rows', 'planned_score_rows')) or
                    model.get('contract_sha256') != json_hash(definition) or model.get('model_id') != model_id or
                    model.get('benchmark_manifest_sha256') != manifest_hash or
                    model.get('score_rows_per_model') != 34000 or
                    model.get('protocol_sha256') != protocol.get('protocol_sha256')):
                raise PipelineError(f'incomplete smooth score matrix: {model_id}')
        return dict(complete_models=ids, score_rows=22*34000)
    reports = supplement/'reports'
    report = read_json(reports/'report.json')
    if (report.get('status') != 'complete' or report.get('shortfalls') != [] or
            set(report.get('models', {})) != set(ids) or not report.get('figures') or
            report.get('protocol_file_sha256') != file_sha256(supplement/'protocol.json') or
            report.get('benchmark_manifest_sha256') != file_sha256(benchmark/'manifest.json') or
            report.get('report_source_sha256') != file_sha256(ROOT/'hnuter_dit/report_contact_smooth_projection.py') or
            not (reports/'research_report_zh.md').is_file()):
        raise PipelineError('full smooth-decoder research report is incomplete or changed')
    for relative, digest in report['figures'].items():
        path = reports/relative
        if not path.resolve().is_relative_to(reports.resolve()) or file_sha256(path) != digest:
            raise PipelineError('smooth report figure checksum/path mismatch')
    return dict(report_sha256=file_sha256(reports/'report.json'), report=str(reports/'research_report_zh.md'))


def run(args: argparse.Namespace) -> dict:
    experiment, benchmark = args.experiment.resolve(), args.benchmark.resolve()
    supplement = (args.supplement or experiment/'smooth_projection_supplement').resolve()
    if supplement == experiment or supplement.is_relative_to(experiment/'evaluation'):
        raise PipelineError('supplement must not overwrite parent artifacts')
    supplement.mkdir(parents=True, exist_ok=True)
    guardian_dir = supplement/'thermal/pipeline'
    began = time.monotonic()
    with pipeline_lock(supplement/'pipeline.lock'):
        state = dict(schema_version='contact_smooth_pipeline_status_v001', status='initializing',
                     current_stage='initialize', pid=os.getpid(), completed_stages=[], stage_evidence={},
                     started_utc=datetime.now(timezone.utc).isoformat())
        def update(**extra: Any) -> None:
            state.update(extra, elapsed_s=time.monotonic()-began, updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(supplement/'pipeline_status.json', state)
        try:
            deadline = time.monotonic()+20.
            while time.monotonic() < deadline:
                if (guardian_dir/'status.json').is_file() and read_json(guardian_dir/'status.json').get('child', {}).get('pid') == os.getpid():
                    break
                time.sleep(.1)
            guardian = verify_pipeline_guard(guardian_dir)
            contract = ensure_contract(experiment, benchmark, supplement)
            update(source_contract_sha256=contract['contract_sha256'])
            binding_path = supplement/'parent_pipeline_binding.json'
            binding = read_json(binding_path) if binding_path.exists() else None
            while True:
                verify_source_contract(contract)
                verify_pipeline_guard(guardian_dir, guardian)
                health = parent_health(experiment, benchmark, contract, binding)
                binding = health['binding']
                if not binding_path.exists():
                    atomic_json(binding_path, binding)
                if health['complete']:
                    state['stage_evidence']['parent'] = health
                    break
                update(status='waiting_parent', current_stage='parent',
                       **{key: value for key, value in health.items() if key != 'complete'})
                time.sleep(args.poll_seconds)
            env = os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MPLCONFIGDIR=str(Path('/tmp')/f'contact_smooth_mpl_{os.getuid()}'))
            for stage in build_stages(experiment, benchmark, supplement, sys.executable):
                verify_source_contract(contract)
                verify_pipeline_guard(guardian_dir, guardian)
                update(status='running', current_stage=stage.name, command=stage.command)
                runtime = run_stage(stage, supplement/'pipeline_logs', env)
                verify_source_contract(contract)
                verify_pipeline_guard(guardian_dir, guardian)
                state['stage_evidence'][stage.name] = runtime | verify_output(stage, benchmark, supplement)
                state['completed_stages'].append(stage.name)
                update()
            update(status='complete', current_stage='complete', report=str(supplement/'reports/report.json'))
            return state
        except BaseException as error:
            update(status='failed', error_type=type(error).__name__, error=str(error))
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--poll-seconds', type=float, default=30.)
    args = parser.parse_args(argv)
    if not 0 < args.poll_seconds <= 60:
        parser.error('--poll-seconds must be in (0,60]')
    try:
        run(args)
        return 0
    except (PipelineError, OSError, ValueError) as error:
        print(f'SMOOTH PIPELINE FAILED: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
