"""Strict, read-only consolidation of the completed Contact/Surface ablation.

No inference, path fitting, collision queries, oracle reads, or new bootstrap.
The frozen reports supply all statistics; this module verifies their current
provenance and exports every recorded metric, scope, seed and paired contrast.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from . import report_contact_experiment as parent
from . import report_contact_smooth_projection as smooth

REPORT_VERSION = 'contact_ablation_consolidated_report_v001'
BUDGETS = (500, 1000, 2000, 4000, 8000)
SEEDS = (17, 29, 43)
PROJECTIONS = smooth.PROJECTIONS
IntegrityError = parent.IntegrityError
IncompleteExperiment = parent.IncompleteExperiment
LABELS = ('scope', 'family', 'kind', 'difficulty', 'scenario', 'projection', 'selection', 'effect')
CSV_FIELDS = ('table', 'aggregation', 'model_id', 'variant', 'train_size', 'seed', 'seeds',
              'control_id', 'control_variant', *LABELS, 'metric', 'condition_count',
              'planned_source_map_count', 'observed_condition_count', 'source_map_count',
              'observations', 'mean', 'sample_std', 'median', 'p05', 'p95',
              'ci95_low', 'ci95_high', 'seed_means', 'seed_std')


def _read(path: Path) -> dict:
    def invalid(value: str):
        raise IntegrityError(f'non-finite JSON literal {value} in {path}')
    try:
        value = json.loads(path.read_text(encoding='utf8'), parse_constant=invalid)
    except (OSError, ValueError) as error:
        raise IntegrityError(f'unreadable JSON: {path}') from error
    if not isinstance(value, dict):
        raise IntegrityError(f'expected JSON object: {path}')
    return value


def _complete(value: dict, label: str) -> None:
    if value.get('status') != 'complete' or value.get('shortfalls') != []:
        raise IncompleteExperiment(f'{label} is not strictly complete')
    if value.get('synthetic_fixture') or value.get('preview'):
        raise IntegrityError(f'{label} is synthetic/preview, not a completed research artifact')


class _Audit:
    """Hash each distinct file once; detect mutations before committing output."""
    def __init__(self):
        self.hashes: dict[str, str] = {}
        self.signatures: dict[str, tuple] = {}

    @staticmethod
    def signature(path: Path) -> tuple:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def check(self, path: Path, digest: str | None = None) -> str:
        path = path.resolve()
        if any('oracle' in part.lower() for part in path.parts):
            raise IntegrityError('consolidation must not read isolated oracle artifacts')
        key = str(path)
        try:
            before = self.signature(path)
            if key not in self.hashes:
                actual = parent._sha(path)
                if self.signature(path) != before:
                    raise IntegrityError(f'input changed while hashing: {path}')
                self.hashes[key], self.signatures[key] = actual, before
            if self.signatures[key] != before or (digest is not None and self.hashes[key] != digest):
                raise IntegrityError(f'input checksum/identity changed: {path}')
        except OSError as error:
            raise IntegrityError(f'required input absent: {path}') from error
        return self.hashes[key]

    def unchanged(self) -> None:
        for filename, signature in self.signatures.items():
            if self.signature(Path(filename)) != signature:
                raise IntegrityError(f'input changed before report commit: {filename}')


def _identity(cell: dict) -> tuple:
    return tuple((name, cell[name]) for name in LABELS if name in cell)


def _expected_cells(inputs: dict, projections: tuple = PROJECTIONS, *, marginal: bool = True) -> dict:
    scopes = smooth._scopes(inputs) if marginal else parent._scopes(inputs)
    result = {}
    for scope, mask in scopes:
        for scenario in parent.SCENARIOS:
            for selection in ('all_stress', 'frozen_feasible'):
                selected = mask & (inputs['scenario'] == scenario)
                if selection == 'frozen_feasible':
                    selected &= inputs['feasible_mask']
                for projection in projections:
                    label = scope | dict(scenario=scenario, selection=selection, projection=projection)
                    result[_identity(label)] = (int(selected.sum()), len(set(map(str, inputs['map_id'][selected]))))
    return result


def _expected_paired(inputs: dict) -> dict:
    result = {}
    original = {str(value): index for index, value in enumerate(inputs['condition_id'])
                if inputs['scenario'][index] == 'original'}
    shifted = {str(value): index for index, value in enumerate(inputs['condition_id'])
               if inputs['scenario'][index] == 'obstacle_shift'}
    for scope, mask in smooth._scopes(inputs):
        for scenario in parent.SCENARIOS:
            for selection in ('all_stress', 'frozen_feasible'):
                selected = mask & (inputs['scenario'] == scenario)
                if selection == 'frozen_feasible':
                    selected &= inputs['feasible_mask']
                counts = int(selected.sum()), len(set(map(str, inputs['map_id'][selected])))
                for effect in ('projection_on_minus_off', 'on_smooth_minus_on'):
                    result[_identity(scope | dict(scenario=scenario, selection=selection, effect=effect))] = counts
        a = np.array([index for _, index in sorted(original.items()) if mask[index]], dtype=int)
        b = np.array([shifted[str(inputs['condition_id'][index])] for index in a], dtype=int)
        for selection in ('all_stress', 'both_scenarios_frozen_feasible'):
            choose = np.ones(len(a), bool) if selection == 'all_stress' else inputs['feasible_mask'][a] & inputs['feasible_mask'][b]
            counts = int(choose.sum()), len(set(map(str, inputs['map_id'][a[choose]])))
            for projection in PROJECTIONS:
                result[_identity(scope | dict(selection=selection, effect='shift_minus_original', projection=projection))] = counts
            result[_identity(scope | dict(selection=selection, effect='smooth_shift_effect_minus_on_shift_effect'))] = counts
    return result


def _finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_cells(cells: list, expected: dict, names: tuple, *, seed_group: bool = False,
                    full_metrics: tuple = ()) -> None:
    if not isinstance(cells, list):
        raise IntegrityError('missing complete statistics table')
    seen = set()
    for cell in cells:
        identity = _identity(cell)
        if identity not in expected or identity in seen:
            raise IntegrityError('unexpected or duplicate statistics scope/condition/projection')
        seen.add(identity); conditions, maps = expected[identity]
        if (cell.get('condition_count') != conditions or cell.get('planned_source_map_count') != maps or
                cell.get('bootstrap_replicates') != 1000 or cell.get('observed_condition_count') != conditions or
                cell.get('source_map_count') != maps or set(cell.get('metrics', {})) != set(names)):
            raise IntegrityError('statistics denominator/bootstrap/metric schema differs from frozen inputs')
        for name, metric in cell['metrics'].items():
            count = metric.get('observations')
            if type(count) is not int or not 0 <= count <= conditions or (name in full_metrics and count != conditions):
                raise IntegrityError('failed/missing candidates were removed from a required full denominator')
            interval = metric.get('ci95')
            if (not isinstance(interval, list) or len(interval) != 2 or
                    (count == 0 and (metric.get('mean') is not None or interval != [None, None])) or
                    (count > 0 and (not _finite(metric.get('mean')) or not all(_finite(v) for v in interval) or interval[0] > interval[1]))):
                raise IntegrityError('missing or non-finite statistics/CI')
            for field in ('std', 'median', 'p05', 'p95'):
                if field not in metric or (count and not _finite(metric[field])) or (not count and metric[field] is not None):
                    raise IntegrityError('incomplete recorded scalar diagnostics')
            if seed_group:
                means = metric.get('seed_means')
                if not isinstance(means, list) or len(means) != 3 or any(value is not None and not _finite(value) for value in means):
                    raise IntegrityError('three training-seed statistics are absent')
                observed = [value for value in means if value is not None]
                wanted = float(np.std(observed, ddof=1)) if len(observed) > 1 else 0. if observed else None
                actual = metric.get('seed_std')
                if (wanted is None and actual is not None) or (wanted is not None and
                        (not _finite(actual) or not math.isclose(actual, wanted, rel_tol=1e-12, abs_tol=1e-12))):
                    raise IntegrityError('seed SD does not match recorded per-seed means')
    if seen != set(expected):
        raise IntegrityError('complete statistics omit planned strata/decoder/scenario cells')


def _matrix(protocol: dict) -> list:
    wanted = [(f'geometry_n{size:04d}_seed{seed}', 'geometry', size, seed) for size in BUDGETS for seed in SEEDS]
    wanted += [(f'{variant}_n8000_seed{seed}', variant, 8000, seed) for seed in SEEDS
               for variant in ('no_contact_loss', 'no_geometry_tokens')]
    actual = [(row.get('id'), row.get('variant'), row.get('train_size'), row.get('seed')) for row in protocol.get('variants', [])]
    if actual != wanted:
        raise IntegrityError('approved five-budget / three-seed / two-negative-control matrix differs')
    return [dict(id='base', variant='base', train_size=0, seed=None), *protocol['variants']]


def _artifacts(audit: _Audit, root: Path, records: dict, expected: set[str]) -> None:
    if set(records) != expected:
        raise IntegrityError('complete model artifact list omits or adds a planned shard')
    for relative, digest in records.items():
        audit.check(smooth._safe(root, relative), digest)


def _report_files(audit: _Audit, directory: Path, report: dict, renderer) -> None:
    if not report.get('figures'):
        raise IntegrityError('completed research report has no scientific PNGs')
    markdown = directory/'research_report_zh.md'
    audit.check(markdown)
    if markdown.read_text(encoding='utf8') != renderer(report, list(report['figures'])):
        raise IntegrityError('upstream Markdown differs from the frozen report renderer/statistics')
    for relative, digest in report['figures'].items():
        if Path(relative).suffix.lower() != '.png':
            raise IntegrityError('scientific figure is not a PNG')
        audit.check(smooth._safe(directory, relative), digest)


def _training_inputs(audit: _Audit, config: dict, protocol: dict) -> None:
    """Hash the frozen prepared cache and metadata without loading trajectories.

    Dataset freeze authenticates manifest/prepared bytes. Legacy split indexes
    and provenance have no hash in that freeze, so their *current* bytes are
    additionally pinned in this consolidation, not represented as older pins.
    """
    root = Path(config['dataset_root']).resolve()
    if root != Path(protocol['training_dataset']).resolve() or config.get('dataset_freeze_sha256') != protocol['dataset_freeze_sha256']:
        raise IntegrityError('training dataset root/freeze differs from approved protocol')
    audit.check(root/'dataset_freeze.json', protocol['dataset_freeze_sha256'])
    freeze, manifest = _read(root/'dataset_freeze.json'), _read(root/'manifest.json')
    if (freeze.get('status') != 'frozen' or freeze.get('integrity_passed') is not True or
            freeze.get('trajectory_count') != 10000 or freeze.get('manifest_sha256') != config.get('dataset_manifest_sha256') or
            freeze.get('prepared_cache_sha256') != config.get('prepared_cache_sha256') or manifest.get('status') != 'frozen'):
        raise IntegrityError('training manifest/prepared provenance differs from dataset freeze')
    audit.check(root/'manifest.json', freeze['manifest_sha256'])
    audit.check(root/'prepared_dataset.npz', freeze['prepared_cache_sha256'])
    provenance = _read(root/'provenance.json'); audit.check(root/'provenance.json')
    if (provenance.get('config_sha256') != manifest.get('config_sha256') or
            provenance.get('base_checkpoint_sha256') != protocol['base_sha256'] or
            provenance.get('base_normalization_reference_sha256') != config.get('normalization_sha256')):
        raise IntegrityError('training provenance changed Base/normalization/dataset config')
    audit.check(Path(config['normalization_reference']), config['normalization_sha256'])
    for split in ('train', 'validation', 'test'):
        audit.check(root/split/'index.json')
    for record in provenance.get('source_snapshot', {}).values():
        audit.check(smooth._safe(root, record['path']), record['sha256'])


def _audit_training(audit: _Audit, experiment: Path, row: dict, model: dict, protocol: dict) -> None:
    training = model['training']; contract = model['parent_contract']
    if any(training.get(key) != row[key] for key in ('variant', 'train_size', 'seed')):
        raise IntegrityError('training identity differs from approved model matrix')
    if contract.get('base_sha256') != protocol['base_sha256']:
        raise IntegrityError('model changed the frozen Base checkpoint')
    if row['id'] == 'base':
        if training.get('frozen_base_sha256') != protocol['base_sha256'] or contract.get('adapter_sha256') is not None:
            raise IntegrityError('Base is not the frozen adapter-free baseline')
        return
    directory = experiment/'models'/row['id']
    config_path, status_path = directory/'resolved_training_config.json', directory/'run_status.json'
    config, status = _read(config_path), _read(status_path)
    if (config != training.get('training_config') or status != training.get('training_status') or
            config.get('contract_sha256') != parent._definition_hash(config, timestamp='created_at_utc') or
            status.get('contract_sha256') != config.get('contract_sha256') or status.get('status') != 'complete' or
            status.get('step') != 8000 or status.get('successful_sample_presentations') != 512000 or
            status.get('planned_steps') != 8000 or status.get('train_experts') != row['train_size'] or
            status.get('validation_experts') != 1000 or config.get('validation_count') != 1000 or
            config.get('reverse_augmentation') is not True or
            status.get('frozen_base_unchanged') is not True or config.get('training_steps') != 8000 or
            config.get('batch_size') != 64 or config.get('base_sha256') != protocol['base_sha256'] or
            config.get('seed') != row['seed'] or config.get('train_size') != row['train_size'] or
            type(status.get('best_step')) is not int or not 1 <= status['best_step'] <= 8000 or
            not _finite(status.get('best_validation_score'))):
        raise IntegrityError('training/checkpoint completion or common validation evidence is incomplete')
    _training_inputs(audit, config, protocol)
    counts = training.get('effective_training_counts', {})
    expected = dict(expert_paths=row['train_size'], source_maps=row['train_size']//20,
        distinct_task_conditions=row['train_size']//20*14, conditions_per_source_map=14,
        expert_paths_per_source_map=20, prepared_identity_audit=True)
    if any(counts.get(key) != value for key, value in expected.items()):
        raise IntegrityError('effective independent training maps/conditions differ from approved whole-map budget')
    audit.check(config_path, training['config_sha256'])
    audit.check(config_path, model['contract']['checkpoint_binding']['training_config_sha256'])
    audit.check(status_path, model['contract']['checkpoint_binding']['run_status_sha256'])
    audit.check(directory/'best.pt', status['selected_checkpoint_sha256'])
    audit.check(directory/'training_subset.json', counts['subset_file_sha256'])
    if contract.get('adapter_sha256') != status['selected_checkpoint_sha256']:
        raise IntegrityError('evaluation did not use validation-selected adapter')
    if row['variant'] == 'no_contact_loss' and any(config['loss_weights'][key] != 0 for key in ('contact', 'normal', 'smooth')):
        raise IntegrityError('auxiliary-loss negative control retained contact/normal/smooth')
    if row['variant'] == 'no_geometry_tokens' and config['adapter_config']['task_conditioning'] != 'none':
        raise IntegrityError('geometry-token negative control retained geometry attention tokens')


def audit_inputs(experiment: str | Path, benchmark: str | Path, supplement: str | Path | None = None) -> dict:
    """Strict source/report/shard audit, reusable by the separately owned runner.

    Only file hashes and saved scalar summaries are read. Raw/projected NPZs
    are hashed, never decoded, refitted or passed through physical metrics.
    """
    experiment, benchmark = Path(experiment).resolve(), Path(benchmark).resolve()
    supplement = Path(supplement).resolve() if supplement is not None else experiment/'smooth_projection_supplement'
    problems = []
    protocol, manifest, inputs, prior, smooth_protocol, sources, parent_sources = smooth._context(
        experiment, benchmark, supplement, problems, allow_partial=False)
    if problems or protocol.get('schema_version') != 'contact_research_protocol_v001':
        raise IncompleteExperiment('consolidation requires the completed approved research protocol, never pilots')
    matrix = _matrix(protocol); ids = [row['id'] for row in matrix]
    if (protocol.get('new_source_maps') != 1000 or protocol.get('new_test_conditions') != 10000 or
            protocol.get('multimodal_subset_conditions') != 1000 or len(set(map(str, inputs['map_id']))) != 1000 or
            len(inputs['case_id']) != 20000):
        raise IntegrityError('planned source-map / condition / paired-scenario matrix is incomplete')
    audit = _Audit()
    parent_path = Path(prior.pop('_report_path'))
    state_path = supplement/'pipeline_status.json'; state = _read(state_path)
    if state.get('status') != 'complete' or state.get('current_stage') != 'complete':
        raise IncompleteExperiment('supplement pipeline is not strictly complete')
    report_path = Path(state.get('report', supplement/'reports/report.json')).resolve()
    if not report_path.is_relative_to(supplement):
        raise IntegrityError('supplement report path escapes its experiment')
    new = _read(report_path); _complete(prior, 'parent report'); _complete(new, 'supplement report')
    evidence = state.get('stage_evidence', {}).get('report', {})
    audit.check(report_path, evidence.get('report_sha256'))
    if not evidence.get('report_sha256'):
        raise IntegrityError('supplement pipeline does not authenticate its completed report bytes')
    parent_digest = audit.check(parent_path)
    smooth_protocol.pop('_multimodal_maps'); driver = smooth_protocol.pop('_driver_contract')
    expected = dict(schema_version=smooth.REPORT_VERSION, parent_protocol=protocol, protocol=smooth_protocol,
        benchmark=manifest, protocol_file_sha256=parent._sha(supplement/'protocol.json'),
        benchmark_manifest_sha256=parent._sha(benchmark/'manifest.json'),
        report_source_sha256=parent._sha(Path(smooth.__file__)),
        source_contract_file_sha256=parent._sha(supplement/'source_contract.json'), source_contract=sources,
        driver_source_contract=driver, parent_source_contract=parent_sources,
        parent_report_sha256=parent_digest, parent_report_path=str(parent_path),
        bootstrap_replicates=1000, bootstrap_seed=20261011, projections=list(PROJECTIONS),
        oracle_access=False, decoder_replayed=False, new_sampling=False, primary_candidates=1)
    if any(new.get(key) != value for key, value in expected.items()) or set(new.get('models', {})) != set(ids):
        raise IntegrityError('completed supplement report source/input/bootstrap/model contract differs')
    if prior.get('schema_version') != parent.REPORT_VERSION or set(prior.get('models', {})) != set(ids):
        raise IntegrityError('parent report schema/model matrix differs')
    for contract in (parent_sources, sources, driver):
        for filename, digest in (contract['source_files'] | contract['input_files']).items():
            audit.check(Path(filename), digest)
    for filename, digest in manifest['input_files_sha256'].items():
        if filename in {'runtime_inputs.npz', 'index.jsonl', *map(str, inputs['input_shard'])}:
            audit.check(smooth._safe(benchmark, filename), digest)
    for path in (experiment/'protocol.json', benchmark/'manifest.json', experiment/'pipeline_status.json',
                 experiment/'pipeline_source_contract.json', supplement/'protocol.json', supplement/'source_contract.json',
                 supplement/'pipeline_source_contract.json', state_path, Path(__file__)):
        audit.check(path)
    _report_files(audit, parent_path.parent, prior, parent._markdown)
    _report_files(audit, report_path.parent, new, smooth._markdown)
    maps = sorted(set(map(str, inputs['map_id'])))
    shared_validation = None
    multi_maps = parent._multimodal_maps(inputs, 10)
    multi_rows = np.flatnonzero(np.isin(inputs['map_id'], list(multi_maps)))
    subset = {key: value[multi_rows] for key, value in inputs.items()}
    cells, multi_cells, pairs = _expected_cells(inputs), _expected_cells(subset), _expected_paired(inputs)
    parent_cells = _expected_cells(inputs, ('off', 'on'), marginal=False)
    parent_multi = _expected_cells(subset, ('off', 'on'), marginal=False)
    for row in matrix:
        model, old = new['models'][row['id']], prior['models'][row['id']]
        if model.get('complete') is not True or old.get('complete') is not True or model.get('training') != old.get('training'):
            raise IncompleteExperiment('completed report omits a trained/scored model')
        if model.get('parent_contract') != old.get('contract') or model.get('parent_artifact_sha256') != old.get('artifact_sha256'):
            raise IntegrityError('supplement replaced parent model/raw/score binding')
        directory = f'evaluation/{row["id"]}'
        parent_files = {f'{directory}/{kind}/{map_id.rsplit("/", 1)[-1]}{extension}' for map_id in maps
                        for kind, extension in (('raw', '.npz'), ('scores', '.jsonl.gz'))}
        new_files = {f'{directory}/{kind}/{map_id.rsplit("/", 1)[-1]}{extension}' for map_id in maps
                     for kind, extension in (('scores', '.jsonl.gz'), ('scores', '.json'), ('projected', '.npz'), ('projected', '.json'))}
        new_files |= {f'{directory}/contract.json', f'{directory}/scoring_status.json'}
        _artifacts(audit, experiment, old['artifact_sha256'], parent_files)
        _artifacts(audit, supplement, model['artifact_sha256'], new_files)
        parent_contract_path = experiment/directory/'contract.json'
        current_parent = smooth._verified(parent_contract_path)
        current_new = smooth._verified(supplement/directory/'contract.json')
        if current_parent != old['contract'] or current_new != model['contract']:
            raise IntegrityError('saved model contract bytes differ from completed report')
        expected_parent = dict(model_id=row['id'], base_sha256=protocol['base_sha256'],
            benchmark_manifest_sha256=new['benchmark_manifest_sha256'], benchmark_conditions=10000,
            primary_candidates=1, multimodal_candidates=8, multimodal_maps_per_family=10,
            ddim_steps=25, latent_seed=20261011, inference_version=smooth.evaluation.INFERENCE_VERSION,
            projection_version=smooth.evaluation.PROJECTION_VERSION, metrics_version=smooth.evaluation.METRICS_VERSION)
        expected_new = dict(schema_version=smooth.evaluation.SUPPLEMENT_VERSION, model_id=row['id'],
            protocol_sha256=smooth_protocol['protocol_sha256'], source_contract_sha256=sources['contract_sha256'],
            parent_contract_sha256=current_parent['contract_sha256'], benchmark_manifest_sha256=new['benchmark_manifest_sha256'],
            decoder_version=smooth_protocol['decoder_version'], decoder_parameters_sha256=smooth_protocol['decoder_parameters_sha256'],
            metrics_version=smooth_protocol['metrics_version'], output_schema_sha256=smooth_protocol['output_schema_sha256'],
            reference_backend=current_parent.get('reference_backend'), score_rows_per_model=34000)
        if any(current_parent.get(key) != value for key, value in expected_parent.items()) or any(
                current_new.get(key) != value for key, value in expected_new.items()):
            raise IntegrityError('model inference/decoder/schema/checkpoint contract differs from frozen protocol')
        for section, names in (('inference_sources', smooth.evaluation.INFERENCE_SOURCES),
                               ('evaluation_sources', smooth.evaluation.EVALUATION_SOURCES)):
            if current_parent.get(section) != {name: parent._sha(Path(parent.__file__).parent/name) for name in names}:
                raise IntegrityError('model inference/evaluation source contract changed')
        audit.check(parent_contract_path, current_new['parent_contract_file_sha256'])
        _audit_training(audit, experiment, row, model, protocol)
        if row['id'] != 'base':
            config = model['training']['training_config']
            validation = {key: config[key] for key in ('validation_count', 'validation_noise_seed', 'validation_weights')}
            if shared_validation is not None and validation != shared_validation:
                raise IntegrityError('models used different common fixed-noise validation objectives')
            shared_validation = validation
        status = _read(supplement/directory/'scoring_status.json')
        old_status = _read(experiment/directory/'scoring_status.json'); audit.check(experiment/directory/'scoring_status.json')
        if (status != model.get('scoring_status') or status.get('status') != 'complete' or
                status.get('contract_sha256') != current_new['contract_sha256'] or
                any(status.get(key) != 34000 for key in ('score_rows', 'expected_score_rows', 'planned_score_rows')) or
                any(status.get(key) != 1000 for key in ('complete_maps', 'total_maps', 'planned_total_maps')) or
                status.get('pilot_limit_maps') != 0 or old_status.get('status') != 'complete' or
                old_status.get('complete_maps') != 1000 or old_status.get('total_maps') != 1000 or
                old_status.get('score_rows') != 68000):
            raise IntegrityError('scoring completion counters differ from full requested candidates')
        for map_id in maps:
            stem = map_id.rsplit('/', 1)[-1]
            raw_meta_path = experiment/directory/'raw'/f'{stem}.json'
            raw_meta = _read(raw_meta_path)
            old_meta_path = experiment/directory/'scores'/f'{stem}.json'; old_meta = _read(old_meta_path)
            meta = _read(supplement/directory/'scores'/f'{stem}.json')
            requested, candidates = smooth.evaluation.requests_for_map(inputs, map_id, multi_maps)
            expected_meta = dict(schema_version=smooth.evaluation.SUPPLEMENT_VERSION,
                protocol_sha256=smooth_protocol['protocol_sha256'], source_contract_sha256=sources['contract_sha256'],
                decoder_version=smooth_protocol['decoder_version'], decoder_parameters_sha256=smooth_protocol['decoder_parameters_sha256'],
                metrics_version=smooth_protocol['metrics_version'], output_schema_sha256=smooth_protocol['output_schema_sha256'],
                input_files_sha256={name: manifest['input_files_sha256'][name] for name in set(map(str, inputs['input_shard'][requested]))})
            if (meta != _read(supplement/directory/'projected'/f'{stem}.json') or
                    any(meta.get(key) != value for key, value in expected_meta.items()) or
                    meta.get('raw_sha256') != old['artifact_sha256'][f'{directory}/raw/{stem}.npz'] or
                    meta.get('scores_sha256') != model['artifact_sha256'][f'{directory}/scores/{stem}.jsonl.gz'] or
                    meta.get('projected_sha256') != model['artifact_sha256'][f'{directory}/projected/{stem}.npz'] or
                    meta.get('contract_sha256') != current_new['contract_sha256'] or
                    meta.get('parent_contract_sha256') != current_parent['contract_sha256'] or
                    meta.get('map_id') != map_id or meta.get('score_rows') != len(requested) or
                    meta.get('projected_rows') != len(requested) or
                    not meta.get('raw_meta_sha256') or
                    meta.get('request_sha256') != parent._hash(dict(input_rows=requested.tolist(), candidate_indices=candidates.tolist())) or
                    meta.get('input_sha256') != [str(inputs['input_sha256'][index]) for index in requested] or
                    raw_meta.get('contract_sha256') != current_parent['contract_sha256'] or
                    raw_meta.get('map_id') != map_id or raw_meta.get('raw_sha256') != meta['raw_sha256'] or
                    old_meta.get('contract_sha256') != current_parent['contract_sha256'] or
                    old_meta.get('raw_sha256') != meta['raw_sha256'] or old_meta.get('map_id') != map_id or
                    old_meta.get('score_rows') != 2*len(requested) or
                    old_meta.get('scores_sha256') != old['artifact_sha256'][f'{directory}/scores/{stem}.jsonl.gz']):
                raise IntegrityError('current parent/supplement metadata no longer authenticate requests/results')
            audit.check(raw_meta_path, meta.get('raw_meta_sha256')); audit.check(old_meta_path)
        _validate_cells(old['primary'], parent_cells, parent.FIELDS, full_metrics=('candidate_success', 'numeric_candidate_success'))
        _validate_cells(old['multimodal']['any_at_8'], parent_multi, smooth.ANY_FIELDS, full_metrics=('any_at_8', 'numeric_any_at_8'))
        _validate_cells(model['primary'], cells, smooth.FIELDS, full_metrics=('candidate_success', 'numeric_candidate_success'))
        _validate_cells(model['paired_effects'], pairs, smooth.PAIRED_FIELDS, full_metrics=('candidate_success', 'numeric_candidate_success'))
        if model['multimodal'].get('subset_conditions') != 1000 or model['multimodal'].get('candidates') != 8:
            raise IntegrityError('K8 uses a different fixed subset/candidate budget')
        _validate_cells(model['multimodal']['any_at_8'], multi_cells, smooth.ANY_FIELDS, full_metrics=('any_at_8', 'numeric_any_at_8'))
        _validate_cells(model['multimodal']['diversity'], multi_cells, smooth.DIVERSITY_FIELDS, full_metrics=('successful_candidate_count',))
    expected_groups = {(row['variant'], row['train_size']) for row in matrix[1:]}
    seen = set()
    for group in new.get('seed_groups', []):
        key = group['variant'], group['train_size']
        expected_ids = [row['id'] for row in matrix[1:] if (row['variant'], row['train_size']) == key]
        if (key not in expected_groups or key in seen or group.get('model_ids') != expected_ids or
                group.get('seeds') != list(SEEDS) or group.get('expected_seed_count') != 3 or group.get('observed_seed_count') != 3):
            raise IntegrityError('completed three-seed aggregation is malformed')
        seen.add(key)
        _validate_cells(group['cells'], cells, smooth.FIELDS, seed_group=True, full_metrics=('candidate_success', 'numeric_candidate_success'))
        _validate_cells(group['paired_effects'], pairs, smooth.PAIRED_FIELDS, seed_group=True, full_metrics=('candidate_success', 'numeric_candidate_success'))
        _validate_cells(group['multimodal']['any_at_8'], multi_cells, smooth.ANY_FIELDS, seed_group=True, full_metrics=('any_at_8', 'numeric_any_at_8'))
        _validate_cells(group['multimodal']['diversity'], multi_cells, smooth.DIVERSITY_FIELDS, seed_group=True, full_metrics=('successful_candidate_count',))
    if seen != expected_groups:
        raise IntegrityError('missing complete budget/negative-control seed group')
    wanted_pairs = {(row['id'], 'base') for row in matrix if row['variant'] == 'geometry'}
    wanted_pairs |= {(f'geometry_n8000_seed{seed}', f'{variant}_n8000_seed{seed}') for seed in SEEDS
                     for variant in ('no_contact_loss', 'no_geometry_tokens')}
    seen = set()
    for pair in new.get('paired_adapter_controls', []):
        key = pair['model_id'], pair['control_id']
        if key not in wanted_pairs or key in seen:
            raise IntegrityError('unexpected/duplicate adapter-control contrast')
        seen.add(key); _validate_cells(pair['cells'], cells, smooth.FIELDS,
                                     full_metrics=('candidate_success', 'numeric_candidate_success'))
    if seen != wanted_pairs:
        raise IntegrityError('adapter−Base or a negative-control contrast is missing')
    wanted_groups = {(size, 'base') for size in BUDGETS} | {(8000, name) for name in ('no_contact_loss', 'no_geometry_tokens')}
    seen = set()
    for group in new.get('paired_adapter_control_groups', []):
        key = group['train_size'], group['control_variant']
        if key not in wanted_groups or key in seen or group.get('seeds') != list(SEEDS):
            raise IntegrityError('adapter-control three-seed aggregation is malformed')
        seen.add(key); _validate_cells(group['cells'], cells, smooth.FIELDS, seed_group=True,
                                     full_metrics=('candidate_success', 'numeric_candidate_success'))
    if seen != wanted_groups:
        raise IntegrityError('adapter-control seed-mean contrast is missing')
    audit.unchanged()
    return dict(experiment=experiment, benchmark=benchmark, supplement=supplement,
        parent_report_path=parent_path, supplement_report_path=report_path,
        parent_report=prior, supplement_report=new, protocol=protocol, manifest=manifest, inputs=inputs,
        matrix=matrix, verified_input_files_sha256=audit.hashes, _audit=audit)


def _rows(report: dict):
    """Lossless long-form scalar export; null diagnostics stay null."""
    def table(kind: str, cells: list, identity: dict):
        for cell in cells:
            labels = {name: cell[name] for name in LABELS if name in cell}
            counts = {name: cell[name] for name in ('condition_count', 'planned_source_map_count', 'observed_condition_count', 'source_map_count')}
            for name, metric in cell['metrics'].items():
                yield identity | labels | counts | dict(table=kind, metric=name,
                    observations=metric['observations'], mean=metric['mean'], sample_std=metric['std'],
                    median=metric['median'], p05=metric['p05'], p95=metric['p95'],
                    ci95_low=metric['ci95'][0], ci95_high=metric['ci95'][1],
                    seed_means=metric.get('seed_means'), seed_std=metric.get('seed_std'))
    for name, model in report['models'].items():
        training = model['training']
        identity = dict(aggregation='individual_model', model_id=name,
                       **{key: training.get(key) for key in ('variant', 'train_size', 'seed')})
        yield from table('K1', model['primary'], identity)
        yield from table('K1_paired', model['paired_effects'], identity)
        yield from table('K8_any', model['multimodal']['any_at_8'], identity)
        yield from table('K8_diversity', model['multimodal']['diversity'], identity)
    for group in report['seed_groups']:
        identity = dict(aggregation='three_seed_mean', **{key: group[key] for key in ('variant', 'train_size', 'seeds')})
        yield from table('K1', group['cells'], identity)
        yield from table('K1_paired', group['paired_effects'], identity)
        yield from table('K8_any', group['multimodal']['any_at_8'], identity)
        yield from table('K8_diversity', group['multimodal']['diversity'], identity)
    for pair in report['paired_adapter_controls']:
        identity = {key: value for key, value in pair.items() if key != 'cells'}
        yield from table('K1_adapter_control', pair['cells'], identity | dict(aggregation='within_seed'))
    for group in report['paired_adapter_control_groups']:
        identity = {key: value for key, value in group.items() if key not in ('cells', 'expected_seed_count', 'observed_seed_count')}
        yield from table('K1_adapter_control', group['cells'], identity | dict(aggregation='three_seed_mean'))


def _metric(metric: dict, *, percent: bool = True, difference: bool = False, sd: bool = True) -> str:
    if metric['mean'] is None:
        return f"缺诊断（n={metric['observations']}）"
    scale = 100. if percent else 1.
    sign = '+' if difference else ''
    text = format(scale*metric['mean'], sign+'.2f')
    if sd and metric.get('seed_std') is not None:
        text += f" ± {scale*metric['seed_std']:.2f}"
    text += (' pp' if difference else '%') if percent else ''
    text += f" [{scale*metric['ci95'][0]:.2f}, {scale*metric['ci95'][1]:.2f}]"
    return text


def _groups(report: dict) -> list:
    base = report['models']['base']
    return [dict(variant='base', train_size=0, seeds=[], cells=base['primary'],
                 multimodal=base['multimodal'], paired_effects=base['paired_effects']), *report['seed_groups']]


def _overall(cells: list, scenario: str, projection: str, selection: str) -> dict:
    cell = parent._overall(cells, scenario=scenario, projection=projection, selection=selection)
    if cell is None:
        raise IntegrityError('required readable primary table cell is absent')
    return cell


def _interval_direction(metric: dict) -> str:
    if metric['mean'] is None:
        return '缺少可用诊断'
    low, high = metric['ci95']
    if low > 0:
        return '地图区间完全高于0'
    if high < 0:
        return '地图区间完全低于0'
    return '地图区间包含0，不能据此断言差值方向确定'


def _interpretation(report: dict) -> list[str]:
    """Describe recorded numbers without model selection or causal claims."""
    largest = next(group for group in report['seed_groups'] if group['variant'] == 'geometry' and group['train_size'] == 8000)
    control_groups = {group['control_variant']: group for group in report['paired_adapter_control_groups'] if group['train_size'] == 8000}
    lines = ['## 结果解读（仅依据冻结统计）', '',
        '以下事实均以全部压力条件为分母；冻结物理见证子集见正文和完整附录。'
        '区间的正负只描述这些源地图上的不确定性，不是未见训练 seed 的泛化证明，也未做多重比较校正。', '']
    for scenario in parent.SCENARIOS:
        geometry = [_overall(largest['cells'], scenario, projection, 'all_stress')['metrics']['candidate_success'] for projection in PROJECTIONS]
        base = [_overall(report['models']['base']['primary'], scenario, projection, 'all_stress')['metrics']['candidate_success'] for projection in PROJECTIONS]
        differences = [_overall(control_groups['base']['cells'], scenario, projection, 'all_stress')['metrics']['candidate_success'] for projection in PROJECTIONS]
        lines.append(f"8k geometry / {scenario}：按 off/on/smooth，K1 为 " + ' / '.join(parent._percent(value['mean']) for value in geometry)
            + '，冻结 Base 为 ' + ' / '.join(parent._percent(value['mean']) for value in base)
            + '；同条件 adapter−Base 配对 Δ 为 ' + ' / '.join(_metric(value, difference=True) for value in differences)
            + '。各 decoder 的区间解释依次为：' + '；'.join(_interval_direction(value) for value in differences)+'。')
        for variant in ('no_contact_loss', 'no_geometry_tokens'):
            control = next(group for group in report['seed_groups'] if group['variant'] == variant and group['train_size'] == 8000)
            rates = [_overall(control['cells'], scenario, projection, 'all_stress')['metrics']['candidate_success'] for projection in PROJECTIONS]
            differences = [_overall(control_groups[variant]['cells'], scenario, projection, 'all_stress')['metrics']['candidate_success'] for projection in PROJECTIONS]
            lines.append(f"{variant} / {scenario}：off/on/smooth 的实际 K1 为 " + ' / '.join(parent._percent(value['mean']) for value in rates)
                + '；8k geometry−该控制的 matched-seed Δ 为 ' + ' / '.join(_metric(value, difference=True) for value in differences)
                + '。' + '；'.join(_interval_direction(value) for value in differences)+'。')
    lines.append('')
    for cell in largest['paired_effects']:
        if cell['scope'] != 'overall' or cell['selection'] != 'all_stress' or cell['effect'] == 'projection_on_minus_off':
            continue
        label = {'on_smooth_minus_on': 'smooth−on', 'shift_minus_original': '移动−原始',
                 'smooth_shift_effect_minus_on_shift_effect': 'smooth移动效应−on移动效应（DID）'}[cell['effect']]
        value = cell['metrics']['candidate_success']
        lines.append(f"8k geometry，{cell.get('scenario', cell.get('projection', '两场景'))} 的 {label}："
            + _metric(value, difference=True)+'；'+_interval_direction(value)+'。')
    lines.extend(['', '五档 geometry 学习曲线按 500→1000→2000→4000→8000 的实际 K1 均值如下；'
        '不强制单调，也不从 validation loss 推断任务成功，不据此选择 checkpoint 或调 decoder 参数。', ''])
    learning = sorted((group for group in report['seed_groups'] if group['variant'] == 'geometry'), key=lambda group: group['train_size'])
    for scenario in parent.SCENARIOS:
        for projection in PROJECTIONS:
            values = [_overall(group['cells'], scenario, projection, 'all_stress')['metrics']['candidate_success']['mean'] for group in learning]
            lines.append(f"{scenario} / {projection}："+' → '.join(parent._percent(value) for value in values)+'。')
    lines.extend(['', 'K8 使用额外候选预算；固定子集 Any@8 与主 K1 的全部10000条件分母不同，'
        '不能把两表的差直接称为同条件 K8−K1 收益。冻结 JSON 未提供固定子集 candidate0 的汇总 CI，'
        '本报告不补造配对增益区间。', ''])
    for scenario in parent.SCENARIOS:
        for projection in PROJECTIONS:
            success = _overall(largest['multimodal']['any_at_8'], scenario, projection, 'all_stress')['metrics']['any_at_8']
            cell = _overall(largest['multimodal']['diversity'], scenario, projection, 'all_stress')
            metric = cell['metrics']['position_pair_msd_m2']; missing = cell['condition_count']-metric['observations']
            lines.append(f"8k geometry / {scenario} / {projection}：Any@8 {_metric(success)}，"
                f"{metric['observations']}/{cell['condition_count']} 个条件至少在一个训练 seed 中有成功候选对，"
                f"其余 {missing} 个条件的 pair 诊断缺失；XYZ MSD {_metric(metric, percent=False)}。")
    lines.extend(['', '三seed聚合的 pair 观测计数是至少一个seed可计算的条件数；各seed有效pair条件可能不同。'
        '成功pair太少时，多样性只描述成功条件的条件分布，不能解释为全部请求的覆盖改善。', ''])
    return lines


def _markdown(data: dict, output: Path, rows: int) -> str:
    report, protocol = data['supplement_report'], data['protocol']
    title = '# Contact/Surface 完整科研消融报告'
    if data.get('synthetic_fixture'):
        title = '# SYNTHETIC FIXTURE：仅用于生成器单元测试，非科研结果'
    lines = [title, '', '本报告整合已经严格完成且文件校验通过的父实验与平滑投影补充实验。'
        '范围是 plane/cylinder 曲面上的连续几何接触路径；不属于 Inspection，也未验证接触力或飞行动力学。', '',
        '## 实验矩阵与公平性', '',
        '冻结 Base 加 21 个 adapter：500、1000、2000、4000、8000 条专家路径各三个种子（17/29/43），'
        '另有 8k 的两个三种子负控。Base 权重未更新。每个 adapter 为 8000 次 optimizer 更新、batch 64，共 512000 次样本呈现；'
        '每条专家近似呈现次数随五档数据量分别为 1024/512/256/128/64；小数据预算获得更多重复呈现，这是固定 compute 的数据量比较，并非等 epoch 实验。', '',
        '主 K1：1000 张未见源地图、10000 个条件，各在 original 与 obstacle_shift 场景评测 candidate_index=0。'
        'K8：推理前固定的同一 1000 条条件，两场景均保留八个候选。'
        f"统一 DDIM {protocol['sampling_steps']} 步、同一 latent seed={protocol['latent_seed']}、同一条件/候选与相同冻结 Base。", '',
        '本次 Base 是 frozen v003 的统一 DDIM、无额外碰撞引导先验；adapter 也使用相同统一采样。'
        '旧 Base free-flight 报告中的 pointwise/smooth collision-guided 成功率不能直接与此 Contact 成功率比较。'
        '沿用同一个障碍 shift function 只保证障碍移动定义一致，不表示旧 guided 协议的整个采样与评测一致。', '',
        '三 decoder：off 为原始输出，on 为冻结解析接触投影，on_smooth 为固定 16 控制点的有界三次拟合后 128 个接触节点。'
        '正则 λ=1e-6，边仍按 v001 线性 intrinsic/lift 解释；没有障碍修复、专家或 oracle 模式判别，也没有重新拟合保存轨迹。', '',
        'no_contact_loss 的准确含义是同时移除 contact、normal、smooth 三个辅助损失，保留轨迹去噪与实际配置中的 collision 项；'
        'no_geometry_tokens 保留 adapter 与损失，但注意力输入仅起终点 token。两项控制检验不同假设。', '',
        '95% CI 是按 family 分层、以源地图为 cluster 的 1000 次 percentile bootstrap（seed 20261011）；'
        '± 后数值是三个训练种子的样本 SD，不能将地图 CI 当作训练 seed 方差的 CI。所有区间沿用已审计报告，不重新 bootstrap。'
        '单模型/Base 不展示不存在的三 seed SD。', '',
        '失败、投影异常和不收敛候选留在请求分母内。诊断缺失保留空值，不能填 0。frozen_feasible 只表示推理前物理见证；'
        'unknown 不等于不可行，见证也不保证本实验 128 节点的 shape-quality gate 可满足。移动差值的可行子集采用两场景见证交集。', '',
        '## 有效训练数据与完成证据', '',
        '| 专家路径 | 独立源地图 | 不同任务条件 |', '|---:|---:|---:|']
    for size in BUDGETS:
        lines.append(f'| {size} | {size//20} | {size//20*14} |')
    lines.extend(['', '每张地图 20 条专家路径、14 个不同任务条件；成对绕行专家、反向增强与重复更新不增加独立地图或条件。', '',
        '| 模型 | updates / presentations | best_step | common validation_total | selected adapter SHA256 |',
        '|---|---:|---:|---:|---|'])
    for name, model in report['models'].items():
        if name == 'base':
            continue
        status = model['training']['training_status']
        lines.append(f"| {name} | {status['step']} / {status['successful_sample_presentations']} | {status['best_step']} | "
                     f"{status['best_validation_score']:.8g} | `{status['selected_checkpoint_sha256']}` |")
    lines.extend(['', f"冻结 Base SHA256：`{protocol['base_sha256']}`。上述 validation_total 是同一固定噪声验证目标的模型选择损失，**不是任务成功率**。", ''])
    for table, field, heading in (('K1', 'candidate_success', '主 K1 成功率'), ('K8_any', 'any_at_8', '固定子集 K8 Any@8')):
        lines.extend([f'## {heading}', '', '表格为 mean ± seed SD [95% map-cluster CI]，单位 %。', ''])
        for selection, label in (('all_stress', '全部压力条件'), ('frozen_feasible', '冻结物理见证子集')):
            lines.extend([f'### {label}', '', '| 变体 / 专家数 | 场景 | 条件数 | off | on | on_smooth |', '|---|---|---:|---:|---:|---:|'])
            for group in _groups(report):
                cells = group['cells'] if table == 'K1' else group['multimodal']['any_at_8']
                for scenario in parent.SCENARIOS:
                    selected = [_overall(cells, scenario, projection, selection) for projection in PROJECTIONS]
                    lines.append(f"| {group['variant']} / {group['train_size']} | {scenario} | {selected[0]['condition_count']} | "
                        + ' | '.join(_metric(cell['metrics'][field]) for cell in selected)+' |')
            lines.append('')
    lines.extend(['## 22 模型逐种子 K1 / K8', '', '以下逐模型区间均为源地图 CI；完整 frozen-feasible 与全部分层逐种子指标见机器可读附录。', '',
        '| 模型 / seed | 场景 | K1 off / on / smooth | K8 off / on / smooth |', '|---|---|---|---|'])
    for name, model in report['models'].items():
        for scenario in parent.SCENARIOS:
            values = []
            for cells, field in ((model['primary'], 'candidate_success'), (model['multimodal']['any_at_8'], 'any_at_8')):
                values.append(' / '.join(_metric(_overall(cells, scenario, projection, 'all_stress')['metrics'][field], sd=False) for projection in PROJECTIONS))
            lines.append(f"| {name} / {model['training']['seed']} | {scenario} | {values[0]} | {values[1]} |")
    lines.extend(['', '## K1 decoder、移动与差分配对效果', '',
        'Δ 为相同条件/候选/训练 seed 内的差，再按条件聚合三个 seed。表为成功率差（pp），± seed SD [95% map CI]。'
        'K8 不在此表混用；其固定子集汇总 CI 已在上表报告。', '',
        '| 变体 / 专家数 | 分母 | 场景/decoder | 对比 | Δ成功率 |', '|---|---|---|---|---:|'])
    effect_names = dict(projection_on_minus_off='on−off', on_smooth_minus_on='on_smooth−on',
        shift_minus_original='shift−original', smooth_shift_effect_minus_on_shift_effect='(smooth shift效应)−(on shift效应)')
    for group in _groups(report):
        for cell in group['paired_effects']:
            if cell['scope'] != 'overall':
                continue
            lines.append(f"| {group['variant']} / {group['train_size']} | {cell['selection']} (n={cell['condition_count']}) | "
                f"{cell.get('scenario', cell.get('projection', '两场景'))} | {effect_names[cell['effect']]} | "
                + _metric(cell['metrics']['candidate_success'], difference=True)+' |')
    lines.extend(['', '## K1 adapter−Base 与 8k 负控配对效果', '',
        '| geometry 专家数 | 对照 | 分母 | 场景 | off Δ(pp) | on Δ(pp) | smooth Δ(pp) |', '|---:|---|---|---|---:|---:|---:|'])
    for group in report['paired_adapter_control_groups']:
        for selection in ('all_stress', 'frozen_feasible'):
            for scenario in parent.SCENARIOS:
                values = [_metric(_overall(group['cells'], scenario, projection, selection)['metrics']['candidate_success'], difference=True) for projection in PROJECTIONS]
                lines.append(f"| {group['train_size']} | {group['control_variant']} | {selection} | {scenario} | " + ' | '.join(values)+' |')
    lines.extend(['', '逐 seed 配对结果、所有质量指标的差值及分层 CI 同样完整导出，不能仅比较未配对点估计。', '',
        '## 成功 gate 与关键失败分解', '',
        '完整成功同时要求完整 URDF 物理碰撞通过、接触误差≤1 mm、法向误差≤1°、有效姿态/工作空间/两端/接触图域/twist、'
        'shape-quality 通过，并对成功候选做密集边审核。shape gate 含最大曲率≤8/m、累计回退≤0.25 m、chord16 ratio≥0.65。'
        'numeric success 另要求≤2e-6 m/rad，不能替代物理成功。单独 gate 的通过率不代表完整候选成功，各失败原因也可以同时发生。', '',
        '| 变体 / 专家数 | 场景 / decoder | URDF节点碰撞自由 | contact≤1mm | normal≤1° | shape gate | dense_checked |',
        '|---|---|---:|---:|---:|---:|---:|'])
    gate_names = ('physical_collision_free', 'practical_contact_gate', 'practical_normal_gate', 'shape_quality_gate', 'dense_checked')
    for group in _groups(report):
        for scenario in parent.SCENARIOS:
            for projection in PROJECTIONS:
                metrics = _overall(group['cells'], scenario, projection, 'all_stress')['metrics']
                lines.append(f"| {group['variant']} / {group['train_size']} | {scenario} / {projection} | "
                    + ' | '.join(_metric(metrics[name]) for name in gate_names)+' |')
    lines.extend(['', 'dense_checked 是已进行密集检查的比例，评分流程仅对满足前置条件的候选进入密集审核。'
        '节点 clearance 是 128 节点处扣除 margin 后的最小值，不能解释成密集边最小 clearance。'
        'contact/normal/curvature、回退、chord16、clearance、path efficiency、图域/twist、拟合收敛与修正量以及失败类型均见完整附录和输入报告。', '',
        '## 8k geometry 的曲面、难度与场景族分层', '',
        '以下为三个种子聚合的 K1 成功率；全部预算、Base、负控、逐 seed、K8 和 family×plane/cylinder×easy/detour 交叉表完整保存于附录。', '',
        '| 分层 | 场景 | 分母 | 条件数 | off | on | smooth |', '|---|---|---|---:|---:|---:|---:|'])
    largest = next(group for group in report['seed_groups'] if group['variant'] == 'geometry' and group['train_size'] == 8000)
    lookup = {_identity(cell): cell for cell in largest['cells']}
    for cell in largest['cells']:
        if cell['scope'] not in ('family', 'kind', 'difficulty') or cell['projection'] != 'off':
            continue
        key = cell['scope']; label = f"{key}={cell[key]}"
        values = []
        for projection in PROJECTIONS:
            match = dict(cell, projection=projection)
            values.append(_metric(lookup[_identity(match)]['metrics']['candidate_success']))
        lines.append(f"| {label} | {cell['scenario']} | {cell['selection']} | {cell['condition_count']} | " + ' | '.join(values)+' |')
    lines.extend(['', '## K8 成功候选对的几何多样性', '',
        '只使用实际保存的成功 body XYZ / intrinsic UV，按相同节点进度计算每对候选的 MSD，再先对单条件候选对平均、后对条件平均。'
        '少于两个成功候选的条件为 null，不能填零；Any@8 仍包含这些条件。方向符号是相对端点弦的经验几何代理，不是同伦类证明。', '',
        '| 变体 / 专家数 | 场景 / decoder | 可计算pair的条件数 | XYZ MSD(m²) | XYZ RMS(m) | UV MSD(m²) |',
        '|---|---|---:|---:|---:|---:|'])
    for group in _groups(report):
        for scenario in parent.SCENARIOS:
            for projection in PROJECTIONS:
                metrics = _overall(group['multimodal']['diversity'], scenario, projection, 'all_stress')['metrics']
                lines.append(f"| {group['variant']} / {group['train_size']} | {scenario} / {projection} | "
                    f"{metrics['position_pair_msd_m2']['observations']} | "
                    + ' | '.join(_metric(metrics[name], percent=False) for name in ('position_pair_msd_m2', 'position_pair_rms_m', 'uv_pair_msd_m2'))+' |')
    lines.extend(['', *_interpretation(report)])
    lines.extend(['', '## 泛化范围、物理见证与计时限制', '',
        '测试地图来自相同十种场景族的新地图；障碍移动是在原条件中将最近非地板 OBB 朝端点身体位置弦线移动 0.30 m。'
        '这支持地图/几何 shift 泛化评估，不证明新的工厂语义 OOD。完整接触装配可容纳的拒绝采样、角度/密度接受规则和重试记录随 manifest 冻结，不能假定等同任意实际场景分布。', '',
        '条件物理见证是有限步长完整 URDF 审核的可行证据；求解器未找到见证不构成不可行证明，见证也不保证严格 shape gate。'
        '圆柱只覆盖有限上部接触带，并非全周或任意法向覆盖；forbidden 图域字段虽保留，但当前冻结输入没有启用案例，不声称该功能已经数据验证。'
        '未测试 approach/retreat、摩擦、接触力、真实工具形状或动力学可执行性；几何 jerk/acceleration 不能当作真实时间控制指标。'
        '端到端训练/守护 wall time 含温控暂停；保存的 evaluation_time_s 是评分调用 wall 计时，也可能包含暂停，均不是 flight/control latency。'
        '正式温控是 reactive 暂停：CPU≥90°C 或 GPU≥83°C，恢复需 CPU≤80°C 且 GPU≤75°C 稳定10秒；'
        '它不保证 CPU 从不越过90°C，已观察的97.125°C峰值与这一控制语义并不矛盾。', '',
        '## 完整机器可读附录与复核', '',
        f'全部 {rows:,} 个指标行： [CSV.gz](full_statistics.csv.gz) / [JSONL.gz](full_statistics.jsonl.gz)。'
        '每行包含表类、模型/预算/seed或三seed组、场景、decoder、分母、family/kind/difficulty/交叉scope、指标、观测数量、mean/sample SD/分位数/CI和seed均值/SD。'
        'CSV 缺值为空；JSONL 缺值为 null，seed_means 与 seeds 保留数组。按 table=K1/K8_any/K8_diversity/K1_paired/K1_adapter_control 筛选可复核所有表。', '',
        f"父报告 SHA256：`{data['verified_input_files_sha256'][str(data['parent_report_path'])]}`。",
        f"补充报告 SHA256：`{data['verified_input_files_sha256'][str(data['supplement_report_path'])]}`。", '',
        'report.json 记录所有 22 模型、冻结参数、源文件、上游报告/输入/artifact哈希和本次三个输出文件 SHA256。'
        '生成前逐项验证完整状态、当前源码/输入/模型/raw/score/projected/meta/figure checksum与精确条件分母；缺项直接失败，CLI没有 partial/preview 模式。', '',
        '训练 dataset_freeze 校验其 manifest 与 prepared_cache 字节，不加载训练轨迹。旧 freeze 没有保存 split-index/provenance 的单独 hash；'
        '本次额外记录其当前文件 hash，并交叉核对 Base、归一化与采集配置身份，不将这些新快照伪称为旧 freeze 中已有的 hash。', '',
        '## 已审计的科研图', ''])
    for path_key, report_key in (('parent_report_path', 'parent_report'), ('supplement_report_path', 'supplement_report')):
        for relative in data[report_key]['figures']:
            path = data[path_key].parent/relative
            link = Path(os.path.relpath(path, output)).as_posix()
            lines.extend([f'![{Path(relative).stem}]({link})', ''])
    return '\n'.join(lines)


def build_consolidated_report(experiment: str | Path, benchmark: str | Path,
                              supplement: str | Path | None = None, output: str | Path | None = None) -> dict:
    data = audit_inputs(experiment, benchmark, supplement)
    experiment = data['experiment']
    output = Path(output).resolve() if output is not None else experiment/'consolidated_reports'
    if not output.is_relative_to(experiment/'consolidated_reports'):
        raise ValueError('output must be inside EXP/consolidated_reports, outside frozen upstream outputs')
    if output.is_relative_to(data['parent_report_path'].parent) or output.is_relative_to(data['supplement_report_path'].parent):
        raise ValueError('output collides with a frozen upstream report directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.contact_appendix_', dir=output.parent) as temporary:
        temporary = Path(temporary); count = 0
        csv_path, json_path = temporary/'full_statistics.csv.gz', temporary/'full_statistics.jsonl.gz'
        with gzip.open(csv_path, 'wt', encoding='utf8', newline='') as csv_stream, gzip.open(json_path, 'wt', encoding='utf8') as json_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS, extrasaction='ignore'); writer.writeheader()
            for row in _rows(data['supplement_report']):
                json_stream.write(parent._json(row)+'\n')
                writer.writerow({key: parent._json(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
                count += 1
        markdown = temporary/'research_report_zh.md'; markdown.write_text(_markdown(data, output, count), encoding='utf8')
        report = dict(schema_version=REPORT_VERSION, status='synthetic_test_only' if data.get('synthetic_fixture') else 'complete',
            shortfalls=[], created_utc=datetime.now(timezone.utc).isoformat(),
            model_ids=[row['id'] for row in data['matrix']], models={name: dict(training=model['training'],
                parent_contract=model['parent_contract'], contract=model['contract'], failure_counts=model.get('failure_counts', []),
                evaluation_errors=model.get('evaluation_errors', [])) for name, model in data['supplement_report']['models'].items()},
            planned_conditions=10000, planned_source_maps=1000, multimodal_subset_conditions=1000,
            primary_candidates=1, multimodal_candidates=8, projections=list(PROJECTIONS),
            bootstrap_replicates=1000, bootstrap_seed=20261011,
            parent_report_path=str(data['parent_report_path']), supplement_report_path=str(data['supplement_report_path']),
            parent_report_sha256=data['verified_input_files_sha256'][str(data['parent_report_path'])],
            supplement_report_sha256=data['verified_input_files_sha256'][str(data['supplement_report_path'])],
            benchmark_manifest_sha256=data['supplement_report']['benchmark_manifest_sha256'],
            report_source_sha256=parent._sha(Path(__file__)), verified_input_files_sha256=data['verified_input_files_sha256'],
            output_files_sha256={path.name: parent._sha(path) for path in (markdown, csv_path, json_path)}, statistics_rows=count,
            definition_policy='copy_frozen_statistics; no_new_bootstrap; all_failures_retained; paired_feasible_intersection',
            training_auxiliary_inputs_policy='dataset_freeze_binds_manifest_and_prepared; current_index_and_provenance_byte_snapshots',
            oracle_access=False, decoder_replayed=False, new_sampling=False)
        parent._write(temporary/'report.json', report)
        data['_audit'].unchanged()
        output.mkdir(parents=True, exist_ok=True)
        for name in ('full_statistics.csv.gz', 'full_statistics.jsonl.gz', 'research_report_zh.md', 'report.json'):
            os.replace(temporary/name, output/name)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = build_consolidated_report(args.experiment, args.benchmark, args.supplement, args.output)
    print(parent._json(dict(status=report['status'], models=len(report['model_ids']), statistics_rows=report['statistics_rows'])))


if __name__ == '__main__':
    main()
