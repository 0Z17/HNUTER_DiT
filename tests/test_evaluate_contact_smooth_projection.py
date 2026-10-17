"""Mock-only supplement integrity tests; no sampling, training, or URDF backend."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hnuter_dit import evaluate_contact_smooth_projection as evaluation

evaluation._limit_threads()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def _gzip(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False)+'\n')


def _contract(path: Path, definition: dict, key: str = 'contract_sha256') -> dict:
    value = definition | {key: evaluation.json_hash(definition)}
    _write(path, value)
    return value


def _condition() -> dict:
    return dict(schema_version='surface_task_condition_v001', task_type='surface',
        task_version='surface_contact_v003', surface_geometry=dict(type='plane', origin=[0., 0., 1.2],
            quaternion_wxyz=[1., 0., 0., 0.], thickness_m=.1, supports=[]),
        surface_chart=dict(bounds=[[-2., 2.], [-2., 2.], [-3.14, 3.14]], forbidden_regions=[]),
        tool_extrinsics=dict(position_body=[0., 0., -.75], quaternion_wxyz=[0., 1., 0., 0.]),
        contact_constraints=dict(position_tolerance_m=1e-7, normal_tolerance_rad=1e-6,
            collision_margin_m=.01, validation_step_m=.02, validation_step_rad=.02,
            attitude_limits_deg=dict(roll=[-40., 40.], pitch=[-70., 70.])),
        start_goal=dict(start=[-1., 0., 0.], goal=[1., 0., 0.]),
        environment=dict(bounds=dict(min=[-4., -4., 0.], max=[4., 4., 4.]), obstacles=[]))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    experiment, benchmark, output = (tmp_path/name for name in ('experiment', 'benchmark', 'supplement'))
    experiment.mkdir(); benchmark.mkdir()
    args = argparse.Namespace(experiment=experiment, benchmark=benchmark, output=output, workers=1, limit_maps=0)
    keys = ('case_id', 'condition_id', 'map_id', 'family', 'kind', 'difficulty', 'scenario', 'feasible_mask',
            'feasibility_status', 'input_sha256', 'input_shard', 'input_row')
    inputs = {key: [] for key in keys}
    files, records = {}, {}
    maps = [f'fixture/map_{index:06d}' for index in range(2)]
    for index, map_id in enumerate(maps):
        condition = _condition()
        condition_id = map_id+'/condition_000'
        record = dict(schema_version='contact_benchmark_input_v001', condition_id=condition_id,
            map_id=map_id, source_map_index=index, family='central_block', kind='plane', difficulty='detour',
            angle_band=0, body_endpoints_pose7=[],
            conditions={scenario: copy.deepcopy(condition) for scenario in evaluation.SCENARIOS},
            input_sha256={scenario: evaluation.json_hash(condition) for scenario in evaluation.SCENARIOS},
            shift=dict(moved_obstacle_id='fixture', displacement_xyz=[0., .3, 0.], shift_distance_m=.3,
                shift_policy='mock_fixture', structural_fixture_members_moved_together=False))
        relative = f'maps/map_{index:06d}/inputs.jsonl.gz'
        _gzip(benchmark/relative, [record]); files[relative] = evaluation.sha256_file(benchmark/relative)
        records[map_id] = record
        for scenario in evaluation.SCENARIOS:
            value = dict(case_id=condition_id+'/'+scenario, condition_id=condition_id, map_id=map_id,
                family='central_block', kind='plane', difficulty='detour', scenario=scenario,
                feasible_mask=not (index == 0 and scenario == 'obstacle_shift'),
                feasibility_status='expert_witness_valid' if scenario == 'original' else 'unknown',
                input_sha256=evaluation.json_hash(condition), input_shard=relative, input_row=0)
            for key in keys:
                inputs[key].append(value[key])
    inputs = {key: np.asarray(value) for key, value in inputs.items()}
    inputs['conditions'] = np.zeros((4, 18), dtype=np.float32)
    np.savez_compressed(benchmark/'runtime_inputs.npz', **inputs)
    (benchmark/'index.jsonl').write_text('{}\n')
    for name in ('runtime_inputs.npz', 'index.jsonl'):
        files[name] = evaluation.sha256_file(benchmark/name)
    # This isolated artifact must never be opened, even to verify its hash.
    files['oracle/unreadable.jsonl.gz'] = 'unreadable_oracle_fixture'
    manifest = dict(status='frozen', condition_count=2, source_map_count=2, input_files_sha256=files)
    _write(benchmark/'manifest.json', manifest)
    base = experiment/'base.pt'; base.write_bytes(b'frozen-base-fixture')
    protocol_definition = dict(schema_version='contact_research_protocol_v001', variants=evaluation.training_matrix(),
        optimizer_updates=8000, new_test_conditions=10000, new_source_maps=1000,
        multimodal_subset_conditions=1000, primary_candidates_per_condition=1, multimodal_candidates_per_condition=8,
        sampling_steps=25, latent_seed=20261011, projection=['off', 'on'],
        base_checkpoint=str(base), base_sha256=evaluation.sha256_file(base), dataset_freeze_sha256='fixture-freeze')
    protocol = _contract(experiment/'protocol.json', protocol_definition, 'protocol_sha256')
    code = Path(evaluation.__file__).parent
    parent_sources = _contract(experiment/'pipeline_source_contract.json', dict(experiment=str(experiment), benchmark=str(benchmark),
        protocol_sha256=protocol['protocol_sha256'],
        source_files={str((code/name).resolve()): evaluation.sha256_file(code/name) for name in (*evaluation.INFERENCE_SOURCES, *evaluation.EVALUATION_SOURCES)},
        input_files={str((experiment/'protocol.json').resolve()): evaluation.sha256_file(experiment/'protocol.json'),
                     str(base.resolve()): evaluation.sha256_file(base)}))
    backend = dict(urdf_sha256='mock-urdf', coal_source_sha256='mock-coal-source', coal_version='mock',
        numpy_version=np.__version__, torch_version=torch.__version__)
    monkeypatch.setattr(evaluation, '_reference_backend', lambda: backend)
    parent = _contract(experiment/'evaluation/base/contract.json', dict(model_id='base', base_sha256=protocol['base_sha256'], adapter_sha256=None,
        benchmark_manifest_sha256=evaluation.sha256_file(benchmark/'manifest.json'), benchmark_conditions=10000,
        primary_candidates=1, multimodal_maps_per_family=10, multimodal_candidates=8, ddim_steps=25, latent_seed=20261011,
        inference_version=evaluation.INFERENCE_VERSION, projection_version=evaluation.PROJECTION_VERSION,
        metrics_version=evaluation.METRICS_VERSION, reference_backend=backend,
        inference_sources={name: evaluation.sha256_file(code/name) for name in evaluation.INFERENCE_SOURCES},
        evaluation_sources={name: evaluation.sha256_file(code/name) for name in evaluation.EVALUATION_SOURCES}))
    multi = {maps[0]}
    for map_id in maps:
        rows, candidates = evaluation.requests_for_map(inputs, map_id, multi)
        paths = np.zeros((len(rows), 128, 9), dtype=np.float32)
        paths[:, :, 0] = candidates[:, None]
        paths[:, :, 1] = rows[:, None]
        paths[:, :, 3] = 1.; paths[:, :, 7] = 1.
        raw = experiment/'evaluation/base/raw'/f'{map_id.rsplit("/", 1)[-1]}.npz'
        evaluation.atomic_npz(raw, world_pose9=paths, input_rows=rows, candidate_indices=candidates)
        _write(raw.with_suffix('.json'), dict(contract_sha256=parent['contract_sha256'], raw_sha256=evaluation.sha256_file(raw),
            map_id=map_id, candidates=len(rows), inference_dtype='float32', batch_size=128))
    contract = dict(model_id='base', parent_contract_sha256=parent['contract_sha256'], contract_sha256='fixture-supplement-model',
        protocol_sha256='fixture-supplement-protocol', source_contract_sha256='fixture-supplement-sources',
        decoder_version=evaluation.SMOOTH_PROJECTION_VERSION,
        decoder_parameters_sha256=evaluation.json_hash(evaluation.smooth_contact_decoder_parameters()),
        output_schema_sha256=evaluation.json_hash(evaluation.OUTPUT_SCHEMA))
    directory = output/'evaluation/base'
    decoder_calls, metric_calls = [], []

    def project(path, spec):
        decoder_calls.append(path.copy())
        candidate = int(path[0, 0])
        world = path.astype(np.float64)+.001
        intrinsic = np.full((128, 3), candidate, dtype=np.float64)
        return SimpleNamespace(world_pose9=world, intrinsic_states=intrinsic,
            before={'contact_error_max_m': torch.tensor(.1)},
            after={'solver_converged': torch.tensor(candidate != 1), 'solver_sweeps': torch.tensor(2),
                   'solver_projected_gradient_max': torch.tensor(1e-11 if candidate != 1 else .01)},
            geometry_valid=torch.tensor(candidate != 2), decoder_parameters=evaluation.smooth_contact_decoder_parameters())

    def metrics(world, condition, *, projected_intrinsic):
        metric_calls.append((world.copy(), copy.deepcopy(condition), projected_intrinsic.copy()))
        return dict(schema_version=evaluation.METRICS_VERSION, candidate_success=True, numeric_candidate_success=True,
            physical_collision_free=True, physical_collision_checked=True, edge_decoder='intrinsic_linear_unwrapped_lift')

    monkeypatch.setattr(evaluation, 'project_smooth_contact_pose9', project)
    monkeypatch.setattr(evaluation, 'evaluate_contact_path', metrics)
    return SimpleNamespace(args=args, manifest=manifest, inputs=inputs, maps=maps, multi=multi,
        directory=directory, contract=contract, protocol=protocol, parent=parent, parent_sources=parent_sources,
        decoder_calls=decoder_calls, metric_calls=metric_calls, records=records)


def _score(fixture, index=0):
    return evaluation.score_map(fixture.args, fixture.directory, fixture.contract, fixture.inputs,
        fixture.maps[index], fixture.multi, fixture.manifest)


def _artifacts(fixture, index=0):
    stem = fixture.maps[index].rsplit('/', 1)[-1]
    return (fixture.directory/'scores'/f'{stem}.jsonl.gz', fixture.directory/'projected'/f'{stem}.npz')


def _rehash(fixture, index=0):
    score, projected = _artifacts(fixture, index)
    meta_path = score.with_name(fixture.maps[index].rsplit('/', 1)[-1]+'.json')
    meta = evaluation._read(meta_path)
    meta.update(scores_sha256=evaluation.sha256_file(score), projected_sha256=evaluation.sha256_file(projected))
    _write(meta_path, meta); _write(projected.with_suffix('.json'), meta)


def test_same_raw_requests_and_full_metric_arguments_are_saved(fixture):
    assert _score(fixture) == 16
    score, projected = _artifacts(fixture)
    records = list(evaluation._read_gz_jsonl(score))
    assert len(fixture.decoder_calls) == len(fixture.metric_calls) == len(records) == 16
    assert all(record['projection'] == 'on_smooth' for record in records)
    assert [record['candidate_index'] for record in records] == list(range(8))*2
    assert all(record['metrics']['candidate_success'] for record in records)
    # Nonconvergence and geometry diagnostics do not promise/gate collision success.
    assert records[1]['metrics']['projection_fit_success'] is False
    assert records[2]['metrics']['projection_geometry_valid'] is False
    assert not records[8]['feasible']  # Keep the shifted stress candidate denominator.
    with np.load(projected, allow_pickle=False) as archive:
        assert archive['world_pose9'].shape == (16, 128, 9)
        assert archive['intrinsic_states'].shape == (16, 128, 3)
        assert archive['fit_success'].tolist() == [True, False, True, True, True, True, True, True]*2
        for index, (world, condition, intrinsic) in enumerate(fixture.metric_calls):
            np.testing.assert_array_equal(world, archive['world_pose9'][index])
            np.testing.assert_array_equal(intrinsic, archive['intrinsic_states'][index])
            assert condition == fixture.records[fixture.maps[0]]['conditions'][records[index]['scenario']]
    assert _score(fixture, 1) == 2
    assert _score(fixture) == 16  # Resume must not fit or score again.
    assert len(fixture.decoder_calls) == 18
    assert score.with_name('map_000000.json').read_bytes() == projected.with_suffix('.json').read_bytes()


def test_decoder_and_metric_errors_keep_all_cases_and_nan_placeholders(fixture, monkeypatch):
    original_project = evaluation.project_smooth_contact_pose9
    original_metrics = evaluation.evaluate_contact_path

    def project(path, spec):
        if int(path[0, 0]) == 3:
            raise ValueError('mock invalid endpoint')
        return original_project(path, spec)

    def metrics(world, condition, *, projected_intrinsic):
        if int(projected_intrinsic[0, 0]) == 4:
            raise RuntimeError('mock backend exception')
        return original_metrics(world, condition, projected_intrinsic=projected_intrinsic)

    monkeypatch.setattr(evaluation, 'project_smooth_contact_pose9', project)
    monkeypatch.setattr(evaluation, 'evaluate_contact_path', metrics)
    assert _score(fixture) == 16
    score, projected = _artifacts(fixture)
    records = list(evaluation._read_gz_jsonl(score))
    for offset in (3, 11):
        metrics = records[offset]['metrics']
        assert not metrics['candidate_success'] and metrics['projection_failed']
        assert metrics['projection_before'] == {} and metrics['projection_after'] == {}
        assert 'invalid endpoint' in metrics['evaluation_error']
    for offset in (4, 12):
        assert records[offset]['metrics']['projection_succeeded']
        assert records[offset]['metrics']['failures'] == ['evaluation_error']
        assert not records[offset]['metrics']['candidate_success']
    with np.load(projected, allow_pickle=False) as archive:
        assert np.isnan(archive['world_pose9'][[3, 11]]).all()
        assert np.isnan(archive['intrinsic_states'][[3, 11]]).all()
        assert np.isfinite(archive['world_pose9'][[4, 12]]).all()
        assert not archive['projection_succeeded'][[3, 11]].any()
    assert _score(fixture) == 16


@pytest.mark.parametrize('change', ['shape', 'rows', 'candidates', 'dtype', 'sha', 'meta'])
def test_parent_raw_corruption_or_rewritten_requests_rejected(fixture, change):
    path = fixture.args.experiment/'evaluation/base/raw/map_000000.npz'
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if change == 'shape':
        arrays['world_pose9'] = arrays['world_pose9'][:, :127]
    elif change == 'rows':
        arrays['input_rows'][0] = 2
    elif change == 'candidates':
        arrays['candidate_indices'][-1] = 6
    elif change == 'dtype':
        arrays['world_pose9'] = arrays['world_pose9'].astype(np.float64)
    evaluation.atomic_npz(path, **arrays)
    meta = evaluation._read(path.with_suffix('.json'))
    if change != 'sha':
        meta['raw_sha256'] = evaluation.sha256_file(path)
    if change == 'meta':
        meta['map_id'] = fixture.maps[1]
    _write(path.with_suffix('.json'), meta)
    # Compressing the same raw arrays is deterministic; explicitly corrupt SHA fixture.
    if change == 'sha':
        path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(evaluation.IntegrityError):
        _score(fixture)
    assert not fixture.decoder_calls


@pytest.mark.parametrize('artifact', ['score', 'projected'])
def test_committed_payload_sha_corruption_rejected_before_refit(fixture, artifact):
    _score(fixture)
    score, projected = _artifacts(fixture)
    path = score if artifact == 'score' else projected
    path.write_bytes(path.read_bytes()+b'corrupted')
    with pytest.raises(evaluation.IntegrityError, match='checksum'):
        _score(fixture)
    assert len(fixture.decoder_calls) == 16


@pytest.mark.parametrize('change', ['missing_score', 'identity', 'projected_shape', 'projected_candidates', 'flags'])
def test_rehashed_resume_still_checks_complete_rows_shape_and_identity(fixture, change):
    _score(fixture)
    score, projected = _artifacts(fixture)
    if change in ('missing_score', 'identity'):
        rows = list(evaluation._read_gz_jsonl(score))
        if change == 'missing_score':
            rows.pop()
        else:
            rows[0]['candidate_index'] = 7
        _gzip(score, rows)
    else:
        with np.load(projected, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        if change == 'projected_shape':
            arrays['world_pose9'] = arrays['world_pose9'][:, :127]
        elif change == 'projected_candidates':
            arrays['candidate_indices'][0] = 7
        else:
            arrays['fit_success'][1] = True
        evaluation.atomic_npz(projected, **arrays)
    _rehash(fixture)
    with pytest.raises(evaluation.IntegrityError):
        _score(fixture)
    assert len(fixture.decoder_calls) == 16


def test_whitelist_oracle_fields_and_frozen_condition_hash_rejected(fixture):
    relative = str(fixture.inputs['input_shard'][0]); path = fixture.args.benchmark/relative
    record = copy.deepcopy(fixture.records[fixture.maps[0]])
    record['conditions']['original']['oracle_route'] = [[1., 2., 3.]]
    _gzip(path, [record]); fixture.manifest['input_files_sha256'][relative] = evaluation.sha256_file(path)
    with pytest.raises(ValueError, match='non-whitelisted'):
        _score(fixture)
    record = copy.deepcopy(fixture.records[fixture.maps[0]])
    record['conditions']['original']['start_goal']['start'][0] += .1
    _gzip(path, [record]); fixture.manifest['input_files_sha256'][relative] = evaluation.sha256_file(path)
    with pytest.raises(evaluation.IntegrityError, match='checksum'):
        _score(fixture)
    assert not fixture.decoder_calls


def test_frozen_shard_identity_and_hash_are_checked_on_resume(fixture):
    _score(fixture)
    relative = str(fixture.inputs['input_shard'][0]); path = fixture.args.benchmark/relative
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(evaluation.IntegrityError, match='checksum'):
        _score(fixture)
    assert len(fixture.decoder_calls) == 16


def test_independent_contract_immutability_and_parent_sources(fixture, monkeypatch):
    # Production still enforces exact 10000 conditions; this fixture only mocks
    # that cardinality gate, never decoder parameters or source/input bindings.
    monkeypatch.setattr(evaluation, '_validate_inputs', lambda *_: (fixture.maps, fixture.multi, 34000))
    parent, protocol, sources = evaluation.ensure_supplement(fixture.args, fixture.manifest, fixture.inputs)
    assert protocol['projection'] == 'on_smooth' and protocol['oracle_access'] is False
    assert protocol['decoder_parameters'] == evaluation.smooth_contact_decoder_parameters()
    assert set(sources['source_files']) == {str((Path(evaluation.__file__).parent/name).resolve()) for name in evaluation.SUPPLEMENT_SOURCES}
    assert evaluation.ensure_supplement(fixture.args, fixture.manifest, fixture.inputs)[1] == protocol
    parent_path = fixture.args.experiment/'protocol.json'
    original_parent = parent_path.read_bytes()
    directory, model = evaluation.model_contract(fixture.args, 'base', parent, protocol, sources)
    assert directory == fixture.directory and model['parent_contract_sha256'] == fixture.parent['contract_sha256']
    assert parent_path.read_bytes() == original_parent
    definition = evaluation._definition(protocol, 'protocol_sha256')
    definition['decoder_parameters']['bending_weight'] *= 2
    _contract(fixture.args.output/'protocol.json', definition, 'protocol_sha256')
    with pytest.raises(evaluation.IntegrityError, match='immutable supplement changed'):
        evaluation.ensure_supplement(fixture.args, fixture.manifest, fixture.inputs)


def test_parent_source_and_parent_model_hash_binding_rejected(fixture):
    path = fixture.args.experiment/'pipeline_source_contract.json'
    definition = evaluation._definition(evaluation._read(path))
    source = next(iter(definition['source_files']))
    definition['source_files'][source] = 'wrong-source-sha'
    _contract(path, definition)
    with pytest.raises(evaluation.IntegrityError, match='checksum'):
        evaluation._verify_parent_sources(fixture.args.experiment, fixture.args.benchmark, fixture.protocol)
    supplement = dict(planned_models=['base'], benchmark_manifest_sha256=evaluation.sha256_file(fixture.args.benchmark/'manifest.json'),
        protocol_sha256='mock', decoder_parameters_sha256='mock', output_schema_sha256='mock')
    parent_path = fixture.args.experiment/'evaluation/base/contract.json'
    definition = evaluation._definition(evaluation._read(parent_path)); definition['latent_seed'] = 12
    _contract(parent_path, definition)
    with pytest.raises(evaluation.IntegrityError, match='parent evaluation'):
        evaluation.model_contract(fixture.args, 'base', fixture.protocol, supplement, dict(contract_sha256='mock'))


def test_production_cardinality_requires_frozen_v001_matrix(fixture):
    evaluation._validate_protocol(fixture.protocol)
    protocol = copy.deepcopy(fixture.protocol); protocol['new_test_conditions'] = 2
    with pytest.raises(evaluation.IntegrityError, match='v001 research protocol'):
        evaluation._validate_protocol(protocol)
    with pytest.raises(evaluation.IntegrityError, match='shortfall'):
        evaluation._validate_inputs(fixture.manifest, fixture.inputs, fixture.protocol)


def test_exact_formal_request_cardinality_without_any_model_or_geometry_execution(fixture):
    rows = np.arange(20000)
    map_index, condition_index = rows//20, rows//2
    map_ids = np.asarray([f'family{index//100}/map_{index:06d}' for index in range(1000)])
    identifiers = np.asarray([f'condition-{index:05d}' for index in range(10000)])
    inputs = dict(case_id=np.asarray([f'case-{index:05d}' for index in rows]), condition_id=identifiers[condition_index],
        map_id=map_ids[map_index], family=np.asarray([f'family{index//100}' for index in map_index]),
        kind=np.repeat('plane', 20000), difficulty=np.repeat('easy', 20000),
        scenario=np.asarray(evaluation.SCENARIOS)[rows%2], feasible_mask=np.ones(20000, dtype=bool),
        feasibility_status=np.repeat('unknown', 20000), input_sha256=np.repeat('mock-sha', 20000),
        input_shard=np.repeat('maps/mock/inputs.jsonl.gz', 20000), input_row=(rows%20)//2)
    maps, multi, paths = evaluation._validate_inputs(dict(condition_count=10000, source_map_count=1000), inputs, fixture.protocol)
    assert len(maps) == 1000 and len(multi) == 100 and paths == 34000
    assert sum(len(evaluation.requests_for_map(inputs, map_id, multi)[0]) for map_id in multi) == 16000
    inputs['condition_id'][1] = inputs['condition_id'][2]
    with pytest.raises(evaluation.IntegrityError):
        evaluation._validate_inputs(dict(condition_count=10000, source_map_count=1000), inputs, fixture.protocol)
    assert not fixture.decoder_calls and not fixture.metric_calls


def test_adapter_contract_binds_completed_selected_checkpoint_and_config(fixture):
    row = fixture.protocol['variants'][0]; model_id = row['id']
    directory = fixture.args.experiment/'models'/model_id
    directory.mkdir(parents=True)
    checkpoint = directory/'best.pt'; checkpoint.write_bytes(b'mock-selected-checkpoint')
    digest = evaluation.sha256_file(checkpoint)
    config = _contract(directory/'resolved_training_config.json', dict(base_sha256=fixture.protocol['base_sha256'],
        train_size=row['train_size'], seed=row['seed'], training_steps=8000,
        dataset_freeze_sha256=fixture.protocol['dataset_freeze_sha256']))
    _write(directory/'run_status.json', dict(status='complete', step=8000,
        contract_sha256=config['contract_sha256'], selected_checkpoint_sha256=digest))
    parent_definition = evaluation._definition(fixture.parent)
    parent_definition.update(model_id=model_id, adapter_sha256=digest)
    _contract(fixture.args.experiment/'evaluation'/model_id/'contract.json', parent_definition)
    supplement = dict(planned_models=[model_id], benchmark_manifest_sha256=fixture.parent['benchmark_manifest_sha256'],
        protocol_sha256='mock', decoder_parameters_sha256='mock', output_schema_sha256='mock')
    _, contract = evaluation.model_contract(fixture.args, model_id, fixture.protocol, supplement, dict(contract_sha256='mock'))
    assert contract['checkpoint_binding']['adapter_sha256'] == digest
    assert contract['checkpoint_binding']['training_config_sha256'] == evaluation.sha256_file(directory/'resolved_training_config.json')
    checkpoint.write_bytes(b'mock-other-checkpoint')
    with pytest.raises(evaluation.IntegrityError, match='checksum'):
        evaluation.model_contract(fixture.args, model_id, fixture.protocol, supplement, dict(contract_sha256='mock'))
    checkpoint.write_bytes(b'mock-selected-checkpoint')
    state = evaluation._read(directory/'run_status.json'); state['status'] = 'running'
    _write(directory/'run_status.json', state)
    with pytest.raises(evaluation.IntegrityError, match='completed'):
        evaluation.model_contract(fixture.args, model_id, fixture.protocol, supplement, dict(contract_sha256='mock'))


def test_safe_paths_never_read_oracle_route_or_escape(tmp_path):
    for relative in ('oracle/paths.jsonl.gz', 'maps/routes/data.gz', '../inputs.gz', '/tmp/inputs.gz'):
        with pytest.raises(evaluation.IntegrityError):
            evaluation._safe_input_path(tmp_path, relative)


def test_pilot_status_and_spawn_workers_are_strict(fixture, monkeypatch):
    class Future:
        def result(self):
            return 16

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs['max_workers'] == 2
            assert kwargs['mp_context'].get_start_method() == 'spawn'
            assert kwargs['initializer'] is evaluation._score_initialize

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def submit(self, worker, map_id):
            assert worker is evaluation._score_worker and map_id == fixture.maps[0]
            return Future()

    fixture.args.limit_maps = 1; fixture.args.workers = 2
    monkeypatch.setattr(evaluation, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(evaluation, 'as_completed', lambda futures: list(futures))
    monkeypatch.setattr(evaluation, 'model_contract', lambda *_: (fixture.directory, fixture.contract))
    monkeypatch.setattr(evaluation, 'multimodal_maps', lambda *_: fixture.multi)
    evaluation.score(fixture.args, 'base', fixture.manifest, fixture.inputs, fixture.protocol,
        dict(source_map_count=1000, score_rows_per_model=34000), {})
    status = evaluation._read(fixture.directory/'scoring_status.json')
    assert status['status'] == 'pilot_complete' and status['score_rows'] == 16
    assert status['planned_score_rows'] == 34000 and status['pilot_limit_maps'] == 1
    # A count shortfall cannot produce even a completed pilot status.
    monkeypatch.setattr(Future, 'result', lambda _: 15)
    with pytest.raises(evaluation.IntegrityError, match='shortfall'):
        evaluation.score(fixture.args, 'base', fixture.manifest, fixture.inputs, fixture.protocol,
            dict(source_map_count=1000, score_rows_per_model=34000), {})
    assert evaluation._read(fixture.directory/'scoring_status.json')['status'] == 'scoring_failed'


def test_cli_defaults_and_thread_bounds_are_explicit(fixture, monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation, 'load_benchmark', lambda _: (fixture.manifest, fixture.inputs))
    monkeypatch.setattr(evaluation, 'ensure_supplement', lambda args, *_: (fixture.protocol,
        dict(planned_models=['base', *[row['id'] for row in evaluation.training_matrix()]]), {}))
    monkeypatch.setattr(evaluation, 'score', lambda args, model, *_: calls.append((args, model)))
    evaluation.main(['--experiment', str(fixture.args.experiment), '--benchmark', str(fixture.args.benchmark)])
    assert len(calls) == 22 and calls[0][1] == 'base'
    assert calls[0][0].output == fixture.args.experiment/'smooth_projection_supplement'
    assert calls[0][0].workers == 2
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    assert all(evaluation.os.environ[name] == '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
    calls.clear()
    evaluation.main(['--experiment', str(fixture.args.experiment), '--model', 'base', '--model', 'base', '--workers', '1', '--limit-maps', '1'])
    assert len(calls) == 1 and calls[0][0].limit_maps == 1
    with pytest.raises(SystemExit):
        evaluation.main(['--limit-maps', '-1'])
