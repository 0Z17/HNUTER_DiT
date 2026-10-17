"""Report-only recovery of the documented v001 source-contract schema error.

The failed v001 directory and every frozen training/evaluation source remain
untouched. Run under thermal_run in EXP/consolidated_reports/recovery_v002.
The v002 reporter performs the full audit exactly once; no inference, scoring,
decoder execution, bootstrap, or upstream process signalling is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time

from . import contact_ablation_report_pipeline as original
from . import report_contact_ablation_appendix_v002 as reporter
from .contact_pipeline import (DEFAULT_BENCHMARK, DEFAULT_EXPERIMENT, ROOT,
    PipelineError, atomic_json, file_sha256, json_hash, pipeline_lock, read_json,
    verify_pipeline_guard, verify_source_contract)
from .thermal_run import matching_identity, remaining_owned_members

OUTPUT_NAME = 'recovery_v002'
SOURCES = ('contact_ablation_report_recovery_v002.py', 'report_contact_ablation_appendix_v002.py')


def failure_evidence(directory: Path) -> dict:
    """Authenticate the exact failed v001 incident, without changing its files."""
    state = read_json(directory/'pipeline_status.json')
    guard = read_json(directory/'thermal/pipeline/status.json')
    launch = read_json(directory/'pipeline_launch.json')
    if (state.get('status') != 'failed' or state.get('error_type') != 'KeyError'
            or state.get('error') != "'input_files'" or state.get('completed_stages') != []
            or guard.get('status') != 'failed' or guard.get('exit_code') != 1
            or guard.get('child') != launch.get('child_identity')
            or state.get('pid') != guard.get('child', {}).get('pid')):
        raise PipelineError('recovery requires the preserved, documented v001 schema failure')
    for key in ('guardian_identity', 'child_identity'):
        identity = launch[key]
        if matching_identity(identity) or remaining_owned_members(identity):
            raise PipelineError('failed v001 still has owned processes; do not launch a duplicate')
    frozen = read_json(directory/'source_contract.json')
    verify_source_contract(frozen)
    if state.get('source_contract_sha256') != frozen['contract_sha256']:
        raise PipelineError('failed v001 changed its source binding')
    files = {str(path.resolve()): file_sha256(path) for path in sorted(directory.rglob('*'))
             if path.is_file() and OUTPUT_NAME not in path.relative_to(directory).parts}
    return dict(schema_version='contact_ablation_report_failure_evidence_v002',
        failure_type='v001_reporter_assumed_input_files_in_code_only_smooth_contract',
        failed_source_contract_sha256=frozen['contract_sha256'], preserved_files_sha256=files,
        identity_policy='bound_PID_start_ticks_UID_PGID_SID; original_groups_exited')


def ensure_contract(experiment: Path, benchmark: Path, supplement: Path, output: Path) -> dict:
    directory = experiment/'consolidated_reports'
    old = read_json(directory/'source_contract.json')
    if (Path(old['experiment']).resolve() != experiment or Path(old['benchmark']).resolve() != benchmark
            or Path(old['supplement']).resolve() != supplement):
        raise PipelineError('failed report belongs to different inputs')
    evidence = failure_evidence(directory)
    evidence_path = output/'failure_evidence.json'
    if evidence_path.exists():
        if read_json(evidence_path) != evidence:
            raise PipelineError('preserved failed v001 evidence changed')
    else:
        atomic_json(evidence_path, evidence)
    sources = dict(old['source_files'])
    sources.update({str(ROOT/'hnuter_dit'/name): file_sha256(ROOT/'hnuter_dit'/name) for name in SOURCES})
    for name in ('test_contact_ablation_report_recovery_v002.py', 'test_report_contact_ablation_appendix_v002.py'):
        path = ROOT/'tests'/name
        sources[str(path)] = file_sha256(path)
    inputs = old['input_files'] | evidence['preserved_files_sha256'] | {str(evidence_path): file_sha256(evidence_path)}
    snapshot_path = output/'source_snapshot/manifest.json'
    if not snapshot_path.is_file():
        raise PipelineError('recovery requires its prelaunch source/test/environment snapshot')
    snapshot = read_json(snapshot_path)
    snapshot_sources = {str(ROOT/'hnuter_dit'/name) for name in SOURCES} | {
        str(ROOT/'tests'/name) for name in
        ('test_contact_ablation_report_recovery_v002.py', 'test_report_contact_ablation_appendix_v002.py')}
    if set(snapshot['files']) != snapshot_sources:
        raise PipelineError('recovery snapshot omits a required new source/test')
    for filename, entry in snapshot['files'].items():
        copy = (output/'source_snapshot'/entry['snapshot_path']).resolve()
        if (not copy.is_relative_to(output/'source_snapshot') or file_sha256(copy) != entry['sha256']
                or sources.get(filename) != entry['sha256']):
            raise PipelineError('recovery snapshot does not match its immutable sources/tests')
        inputs[str(copy)] = entry['sha256']
    inputs[str(snapshot_path)] = file_sha256(snapshot_path)
    environment_path = Path(snapshot['environment_origin'])
    if file_sha256(environment_path) != snapshot['environment_sha256']:
        raise PipelineError('recovery snapshot environment binding changed')
    inputs[str(environment_path)] = snapshot['environment_sha256']
    definition = dict(schema_version='contact_ablation_report_recovery_sources_v002',
        experiment=str(experiment), benchmark=str(benchmark), supplement=str(supplement), output=str(output),
        parent_source_contract_sha256=old['parent_source_contract_sha256'],
        smooth_source_contract_sha256=old['smooth_source_contract_sha256'],
        parent_protocol_sha256=old['parent_protocol_sha256'],
        failed_report_source_contract_sha256=old['contract_sha256'],
        source_files=sources, input_files=inputs,
        stage_policy='both_frozen_studies_complete_and_guardian_exit0; schema_aware_full_audit_once; report_only',
        numerical_policy='copy_frozen_statistics; no_new_bootstrap; unchanged_thresholds_and_denominators')
    path = output/'source_contract.json'
    if path.exists():
        contract = read_json(path)
        if contract.get('contract_sha256') != json_hash(definition):
            raise PipelineError('recovery source/evidence changed; use a fresh version')
    else:
        contract = definition | dict(contract_sha256=json_hash(definition), created_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(path, contract)
    verify_source_contract(contract)
    return contract


def verify_output(output: Path, bindings: dict) -> dict:
    report = read_json(output/'report.json')
    ids = ['base', *[row['id'] for row in original.approved_rows()]]
    expected = dict(schema_version=reporter.REPORT_VERSION, status='complete', shortfalls=[],
        model_ids=ids, planned_conditions=10000, planned_source_maps=1000, multimodal_subset_conditions=1000,
        primary_candidates=1, multimodal_candidates=8, projections=['off', 'on', 'on_smooth'],
        bootstrap_replicates=1000, bootstrap_seed=20261011,
        report_source_sha256=file_sha256(Path(reporter.__file__)),
        recovery_schema_version=reporter.RECOVERY_SCHEMA_VERSION,
        original_reporter_path=str(reporter.ORIGINAL_REPORTER.resolve()),
        original_reporter_sha256=reporter.ORIGINAL_REPORTER_SHA256,
        oracle_access=False, decoder_replayed=False, new_sampling=False)
    if any(report.get(key) != value for key, value in expected.items()) or set(report.get('models', {})) != set(ids):
        raise PipelineError('recovered report lacks the complete frozen 22-model/three-decoder matrix')
    hashes = report.get('verified_input_files_sha256', {})
    if not isinstance(hashes, dict) or not hashes:
        raise PipelineError('recovered report lacks full authenticated input coverage')
    for key, value in bindings.items():
        if report.get(key) != value:
            raise PipelineError('recovered report changed its upstream report binding')
    for key in ('parent_report', 'supplement_report'):
        filename, digest = bindings[key+'_path'], bindings[key+'_sha256']
        if hashes.get(filename) != digest or file_sha256(Path(filename)) != digest:
            raise PipelineError('upstream report changed after recovered audit')
    files = report.get('output_files_sha256', {})
    if (set(files) != set(original.OUTPUT_FILES) or not isinstance(report.get('statistics_rows'), int)
            or report['statistics_rows'] <= 0):
        raise PipelineError('recovered report lacks full statistics/Markdown artifacts')
    for name, digest in files.items():
        path = output/name
        if not path.resolve().is_relative_to(output) or file_sha256(path) != digest:
            raise PipelineError('recovered output checksum/path mismatch')
    return dict(report_sha256=file_sha256(output/'report.json'), output_files_sha256=files,
        statistics_rows=report['statistics_rows'], report=str(output/'research_report_zh.md'),
        audited_input_files=len(hashes))


def run(args: argparse.Namespace) -> dict:
    experiment, benchmark = args.experiment.resolve(), args.benchmark.resolve()
    supplement = (args.supplement or experiment/'smooth_projection_supplement').resolve()
    output = experiment/'consolidated_reports'/OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    guard_dir, began = output/'thermal/pipeline', time.monotonic()
    with pipeline_lock(output/'pipeline.lock'):
        if (output/'report_commit.json').exists():
            raise PipelineError('recovery already committed; inspect its evidence rather than rerun')
        state = dict(schema_version='contact_ablation_report_recovery_status_v002', status='initializing',
            current_stage='initialize', pid=os.getpid(), completed_stages=[], stage_evidence={},
            started_utc=datetime.now(timezone.utc).isoformat())
        def update(**extra):
            state.update(extra, elapsed_s=time.monotonic()-began, updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(output/'pipeline_status.json', state)
        try:
            deadline = time.monotonic()+20.
            while time.monotonic() < deadline:
                if ((guard_dir/'status.json').is_file() and read_json(guard_dir/'status.json').get('child', {}).get('pid') == os.getpid()):
                    break
                time.sleep(.1)
            guardian = verify_pipeline_guard(guard_dir)
            contract = ensure_contract(experiment, benchmark, supplement, output)
            update(source_contract_sha256=contract['contract_sha256'])
            bindings = read_json(experiment/'consolidated_reports/upstream_pipeline_bindings.json')
            health = {kind: original.upstream_health(kind, experiment, benchmark, supplement, contract, bindings[kind])
                      for kind in ('parent', 'smooth')}
            if not all(item['complete'] for item in health.values()):
                raise PipelineError('recovery requires both upstream guardians to have exited normally')
            state['stage_evidence']['upstream'] = health
            paths = dict(parent_report=experiment/'reports/report.json', supplement_report=supplement/'reports/report.json')
            inputs = {key+'_path': str(path) for key, path in paths.items()}
            inputs.update({key+'_sha256': file_sha256(path) for key, path in paths.items()})
            inputs['benchmark_manifest_sha256'] = file_sha256(benchmark/'manifest.json')
            binding_path = output/'upstream_reports_binding.json'
            if binding_path.exists() and read_json(binding_path) != inputs:
                raise PipelineError('completed upstream reports changed across recovery')
            atomic_json(binding_path, inputs)
            verify_pipeline_guard(guard_dir, guardian)
            update(status='running', current_stage='audit_and_report')
            report_start = time.monotonic()
            reporter.build_consolidated_report(experiment, benchmark, supplement, output)
            verify_pipeline_guard(guard_dir, guardian)
            verify_source_contract(contract)
            evidence = verify_output(output, inputs)
            atomic_json(output/'report_commit.json', dict(schema_version='contact_ablation_report_recovery_commit_v002',
                source_contract_sha256=contract['contract_sha256'],
                upstream_reports_binding_sha256=file_sha256(binding_path),
                failure_evidence_sha256=file_sha256(output/'failure_evidence.json'), **evidence))
            state['stage_evidence']['report'] = evidence | dict(exit_code=0, elapsed_s=time.monotonic()-report_start)
            state['completed_stages'] = ['report']
            update(status='complete', current_stage='complete', report=str(output/'report.json'))
            return state
        except BaseException as error:
            update(status='failed', error_type=type(error).__name__, error=str(error))
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--supplement', type=Path)
    run(parser.parse_args())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
