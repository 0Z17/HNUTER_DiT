"""Guarded, read-only wait for both full studies, then write their appendix.

Launch with ``thermal_run --state-dir EXP/consolidated_reports/thermal/pipeline``.
Only the report subprocess belongs to this driver's group. Upstream process
groups are observed and never signalled. Waiting reads small status files and
process identities; full artifact audits run only after both groups have exited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import sys
import time
from typing import Any

from .contact_pipeline import (
    DEFAULT_BENCHMARK, DEFAULT_EXPERIMENT, ROOT, PipelineError, Stage,
    _fresh_guard, _guardian_command, _saved_identity, approved_rows,
    atomic_json, file_sha256, json_hash, load_protocol, pipeline_lock,
    read_json, run_stage, verify_identity, verify_pipeline_guard,
    verify_source_contract,
)
from .thermal_run import process_identity, remaining_owned_members

SOURCES = ('contact_ablation_report_pipeline.py', 'report_contact_ablation_appendix.py')
REPORT_MODULE = 'hnuter_dit.report_contact_ablation_appendix'
OUTPUT_FILES = ('research_report_zh.md', 'full_statistics.csv.gz', 'full_statistics.jsonl.gz')


def ensure_contract(experiment: Path, benchmark: Path, supplement: Path, output: Path) -> dict:
    """Freeze the new sources and the already pinned upstream contracts."""
    paths = {'parent': experiment/'pipeline_source_contract.json',
             'smooth': supplement/'pipeline_source_contract.json'}
    upstream = {name: read_json(path) for name, path in paths.items()}
    for name, contract in upstream.items():
        verify_source_contract(contract)
        if (Path(contract['experiment']).resolve() != experiment or
                Path(contract['benchmark']).resolve() != benchmark):
            raise PipelineError(f'{name} source contract belongs to different inputs')
    if (Path(upstream['smooth']['supplement']).resolve() != supplement or
            upstream['smooth']['parent_source_contract_sha256'] != upstream['parent']['contract_sha256']):
        raise PipelineError('smooth source contract is not bound to this parent')
    sources, inputs = {}, {}
    for contract in upstream.values():
        for destination, key in ((sources, 'source_files'), (inputs, 'input_files')):
            for path, digest in contract[key].items():
                if path in destination and destination[path] != digest:
                    raise PipelineError(f'upstream contracts disagree on {path}')
                destination[path] = digest
    sources.update({str((ROOT/'hnuter_dit'/name).resolve()): file_sha256(ROOT/'hnuter_dit'/name)
                    for name in SOURCES})
    inputs.update({str(path): file_sha256(path) for path in paths.values()})
    definition = dict(schema_version='contact_ablation_report_pipeline_sources_v001',
        experiment=str(experiment), benchmark=str(benchmark), supplement=str(supplement), output=str(output),
        parent_source_contract_sha256=upstream['parent']['contract_sha256'],
        smooth_source_contract_sha256=upstream['smooth']['contract_sha256'],
        parent_protocol_sha256=load_protocol(experiment)['protocol_sha256'],
        source_files=sources, input_files=inputs,
        stage_policy='wait_both_full_studies_and_guardian_exit0; audit_saved_inputs; report_only')
    path = output/'source_contract.json'
    if path.exists():
        value = read_json(path)
        if value.get('contract_sha256') != json_hash(definition):
            raise PipelineError('consolidated report sources/inputs changed; use a fresh version')
    else:
        value = definition | dict(contract_sha256=json_hash(definition),
                                  created_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(path, value)
    verify_source_contract(value)
    return value


class _ExitTransition(PipelineError):
    """Two atomic upstream status writes may straddle normal termination."""


def _command(command: list[str], kind: str, experiment: Path, benchmark: Path, supplement: Path) -> None:
    module = 'hnuter_dit.contact_pipeline' if kind == 'parent' else 'hnuter_dit.contact_smooth_pipeline'
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command) or module not in command:
        raise PipelineError(f'{kind} guardian owns a different command')
    expected = {'--experiment': (experiment, DEFAULT_EXPERIMENT), '--benchmark': (benchmark, DEFAULT_BENCHMARK)}
    if kind == 'smooth':
        expected['--supplement'] = (supplement, experiment/'smooth_projection_supplement')
    for option, (target, default) in expected.items():
        try:
            actual = Path(command[command.index(option)+1]) if option in command else default
        except IndexError as error:
            raise PipelineError(f'{kind} command has an invalid {option}') from error
        if not actual.is_absolute():
            actual = ROOT/actual
        if actual.resolve() != target:
            raise PipelineError(f'{kind} command belongs to different inputs: {option}')


def upstream_health(kind: str, experiment: Path, benchmark: Path, supplement: Path,
                    contract: dict, binding: dict | None) -> dict:
    for attempt in range(3):
        try:
            return _upstream_health_once(kind, experiment, benchmark, supplement, contract, binding)
        except _ExitTransition:
            if attempt == 2:
                raise
            time.sleep(.1)
    raise AssertionError('bounded upstream snapshot loop did not return')


def _upstream_health_once(kind: str, experiment: Path, benchmark: Path, supplement: Path,
                          contract: dict, binding: dict | None) -> dict:
    directory = experiment if kind == 'parent' else supplement
    state, guard_dir = read_json(directory/'pipeline_status.json'), directory/'thermal/pipeline'
    guard = read_json(guard_dir/'status.json')
    allowed = ('initializing', 'running', 'validating_training', 'waiting_training', 'complete') if kind == 'parent' else (
        'initializing', 'waiting_parent', 'running', 'complete')
    if state.get('status') not in allowed:
        raise PipelineError(f'{kind} pipeline failed or has unexpected state: {state.get("status")}: {state.get("error", "")}')
    if guard.get('status') not in ('running', 'paused_cooling', 'complete'):
        raise PipelineError(f'{kind} guardian failed or has unexpected state: {guard.get("status")}')
    child = _saved_identity(guard.get('child', {}), f'{kind} child', dedicated=True)
    command = guard.get('command', [])
    _command(command, kind, experiment, benchmark, supplement)
    expected_source = contract['parent_source_contract_sha256' if kind == 'parent' else 'smooth_source_contract_sha256']
    if state.get('pid') != child['pid'] or state.get('source_contract_sha256') != expected_source:
        raise PipelineError(f'{kind} pipeline changed its child or source binding')
    if kind == 'parent' and state.get('protocol_sha256') != contract['parent_protocol_sha256']:
        raise PipelineError('parent pipeline changed its protocol binding')
    if binding is not None and (binding.get('child') != child or binding.get('command') != command or
                               (binding.get('guardian') is not None and
                                binding['guardian']['pid'] != guard.get('guard_pid'))):
        raise PipelineError(f'{kind} pipeline identity or command changed after binding')
    guardian = process_identity(guard.get('guard_pid', -1))
    identity = binding.get('guardian') if binding else None
    if guard['status'] == 'complete':
        if state['status'] != 'complete':
            raise _ExitTransition(f'{kind} group ended without full study completion')
        if guard.get('exit_code') != 0 or remaining_owned_members(child):
            raise PipelineError(f'{kind} group ended without exit0 and full child termination')
        # A fresh historical completion has no saved guardian start ticks. Do
        # not adopt an unrelated process that subsequently reused its PID.
        if identity is None and guardian is not None and guardian.get('state') not in ('Z', 'X'):
            try:
                _guardian_command(guardian['pid'], guard_dir)
            except PipelineError:
                guardian = None
            else:
                identity = _saved_identity(guardian, f'{kind} guardian')
        original_alive = (identity is not None and guardian is not None and
                          all(guardian.get(key) == value for key, value in identity.items()) and
                          guardian.get('state') not in ('Z', 'X'))
        if original_alive and guardian.get('state') in ('T', 't'):
            raise PipelineError(f'{kind} guardian itself is stopped')
        return dict(complete=not original_alive, binding=dict(child=child, guardian=identity, command=command),
                    pipeline_status=state['status'], guard_status=guard['status'])
    if guardian is None or guardian.get('state') in ('Z', 'X'):
        raise _ExitTransition(f'{kind} guardian disappeared before full study completion')
    current = _saved_identity(guardian, f'{kind} guardian')
    if identity is not None and current != identity:
        raise PipelineError(f'{kind} guardian PID/start ticks/UID/group changed after binding')
    if guardian.get('state') in ('T', 't') or current['pgid'] == child['pgid'] or current['sid'] == child['sid']:
        raise PipelineError(f'{kind} guardian is stopped or inside the child group')
    try:
        _guardian_command(current['pid'], guard_dir)
    except PipelineError as error:
        reread = process_identity(current['pid'])
        if reread is None or reread.get('state') in ('Z', 'X'):
            raise _ExitTransition(f'{kind} guardian exited during command inspection') from error
        raise
    _fresh_guard(guard, 120.)
    if state['status'] != 'complete':
        try:
            verify_identity(child, f'{kind} child')
        except PipelineError as error:
            reread = process_identity(child['pid'])
            if reread is None or reread.get('state') in ('Z', 'X'):
                raise _ExitTransition(f'{kind} child exited during identity inspection') from error
            raise
    return dict(complete=False, binding=dict(child=child, guardian=current, command=command),
                pipeline_status=state['status'], guard_status=guard['status'])


def build_stage(experiment: Path, benchmark: Path, supplement: Path, output: Path, python: str) -> Stage:
    return Stage('report', [python, '-m', REPORT_MODULE, '--experiment', str(experiment),
                           '--benchmark', str(benchmark), '--supplement', str(supplement), '--output', str(output)])


def audit_ready_inputs(experiment: Path, benchmark: Path, supplement: Path) -> dict:
    """Only called after both guardians have exited; this hashes saved data."""
    audit = importlib.import_module(REPORT_MODULE).audit_inputs(experiment, benchmark, supplement)
    return dict(parent_report_path=str(audit['parent_report_path']),
                supplement_report_path=str(audit['supplement_report_path']),
                parent_report_sha256=file_sha256(audit['parent_report_path']),
                supplement_report_sha256=file_sha256(audit['supplement_report_path']),
                benchmark_manifest_sha256=file_sha256(benchmark/'manifest.json'),
                verified_input_files_sha256=audit['verified_input_files_sha256'])


def verify_output(output: Path, inputs: dict) -> dict:
    report = read_json(output/'report.json')
    ids = ['base', *[row['id'] for row in approved_rows()]]
    expected = dict(schema_version='contact_ablation_consolidated_report_v001', status='complete', shortfalls=[],
                    model_ids=ids, planned_conditions=10000, planned_source_maps=1000,
                    multimodal_subset_conditions=1000, primary_candidates=1, multimodal_candidates=8,
                    bootstrap_replicates=1000, bootstrap_seed=20261011,
                    report_source_sha256=file_sha256(ROOT/'hnuter_dit/report_contact_ablation_appendix.py'))
    if any(report.get(key) != value for key, value in expected.items()) or set(report.get('models', {})) != set(ids):
        raise PipelineError('consolidated report is not the complete fixed ablation matrix')
    if any(report.get(key) != value for key, value in inputs.items()):
        raise PipelineError('consolidated report changed its audited upstream inputs')
    files = report.get('output_files_sha256', {})
    if set(files) != set(OUTPUT_FILES) or not isinstance(report.get('statistics_rows'), int) or report['statistics_rows'] <= 0:
        raise PipelineError('consolidated report lacks full statistics and Markdown artifacts')
    for name, digest in files.items():
        path = output/name
        if not path.resolve().is_relative_to(output.resolve()) or file_sha256(path) != digest:
            raise PipelineError(f'consolidated artifact checksum/path mismatch: {name}')
    for key in ('parent_report', 'supplement_report'):
        if file_sha256(Path(inputs[f'{key}_path'])) != inputs[f'{key}_sha256']:
            raise PipelineError(f'{key} changed after its completed audit')
    return dict(report_sha256=file_sha256(output/'report.json'), output_files_sha256=files,
                statistics_rows=report['statistics_rows'], report=str(output/'research_report_zh.md'))


def run(args: argparse.Namespace) -> dict:
    experiment, benchmark = args.experiment.resolve(), args.benchmark.resolve()
    supplement = (args.supplement or experiment/'smooth_projection_supplement').resolve()
    output = experiment/'consolidated_reports'
    if supplement == experiment or supplement == output or supplement.is_relative_to(output):
        raise PipelineError('supplement directory would overlap the consolidated output')
    output.mkdir(parents=True, exist_ok=True)
    guard_dir, began = output/'thermal/pipeline', time.monotonic()
    with pipeline_lock(output/'pipeline.lock'):
        old_state = read_json(output/'pipeline_status.json') if (output/'pipeline_status.json').exists() else {}
        state = dict(schema_version='contact_ablation_report_pipeline_status_v001', status='initializing',
                     current_stage='initialize', pid=os.getpid(), completed_stages=[], stage_evidence={},
                     started_utc=datetime.now(timezone.utc).isoformat())
        def update(**extra: Any) -> None:
            state.update(extra, elapsed_s=time.monotonic()-began, updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(output/'pipeline_status.json', state)
        try:
            deadline = time.monotonic()+20.
            while time.monotonic() < deadline:
                if ((guard_dir/'status.json').is_file() and
                        read_json(guard_dir/'status.json').get('child', {}).get('pid') == os.getpid()):
                    break
                time.sleep(.1)
            guardian = verify_pipeline_guard(guard_dir)
            contract = ensure_contract(experiment, benchmark, supplement, output)
            update(source_contract_sha256=contract['contract_sha256'])
            bindings_path = output/'upstream_pipeline_bindings.json'
            bindings = read_json(bindings_path) if bindings_path.exists() else {}
            while True:
                verify_pipeline_guard(guard_dir, guardian)
                health = {kind: upstream_health(kind, experiment, benchmark, supplement, contract, bindings.get(kind))
                          for kind in ('parent', 'smooth')}
                new_bindings = {kind: item['binding'] for kind, item in health.items()}
                if bindings and bindings != new_bindings:
                    raise PipelineError('upstream persisted identity binding changed')
                bindings = new_bindings
                if not bindings_path.exists():
                    atomic_json(bindings_path, bindings)
                if all(item['complete'] for item in health.values()):
                    break
                update(status='waiting_upstream', current_stage='upstream', upstream_health=health)
                time.sleep(args.poll_seconds)
            verify_source_contract(contract)
            inputs = audit_ready_inputs(experiment, benchmark, supplement)
            inputs_path = output/'upstream_reports_binding.json'
            if inputs_path.exists() and read_json(inputs_path) != inputs:
                raise PipelineError('upstream completed report/input hashes changed after binding')
            if not inputs_path.exists():
                atomic_json(inputs_path, inputs)
            state['stage_evidence']['inputs'] = inputs
            commit_path = output/'report_commit.json'
            if commit_path.exists() or old_state.get('status') == 'complete':
                evidence = verify_output(output, inputs)
                saved = read_json(commit_path) if commit_path.exists() else old_state.get('stage_evidence', {}).get('report', {})
                source_hash = saved.get('source_contract_sha256') if commit_path.exists() else old_state.get('source_contract_sha256')
                if (source_hash != contract['contract_sha256'] or
                        (not commit_path.exists() and old_state.get('completed_stages') != ['report']) or
                        (commit_path.exists() and (saved.get('schema_version') != 'contact_ablation_report_commit_v001' or
                         saved.get('upstream_reports_binding_sha256') != file_sha256(inputs_path))) or
                        any(saved.get(key) != value for key, value in evidence.items())):
                    raise PipelineError('completed consolidated report no longer matches its saved evidence')
                state['stage_evidence']['report'] = evidence
            else:
                stage = build_stage(experiment, benchmark, supplement, output, sys.executable)
                env = os.environ.copy()
                env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           MPLCONFIGDIR=str(Path('/tmp')/f'contact_appendix_mpl_{os.getuid()}'))
                verify_pipeline_guard(guard_dir, guardian)
                update(status='running', current_stage='report', command=stage.command)
                runtime = run_stage(stage, output/'pipeline_logs', env)
                verify_source_contract(contract)
                verify_pipeline_guard(guard_dir, guardian)
                evidence = verify_output(output, inputs)
                state['stage_evidence']['report'] = runtime | evidence
            if not commit_path.exists():
                atomic_json(commit_path, dict(schema_version='contact_ablation_report_commit_v001',
                    source_contract_sha256=contract['contract_sha256'],
                    upstream_reports_binding_sha256=file_sha256(inputs_path),
                    **evidence))
            state['completed_stages'] = ['report']
            update(status='complete', current_stage='complete', report=str(output/'report.json'))
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
    except (RuntimeError, OSError, ValueError) as error:
        print(f'CONSOLIDATED REPORT PIPELINE FAILED: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
