"""Benchmark leakage, perturbation, witness semantics and inference-cache contract."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hnuter_dit import contact_benchmark as benchmark
from hnuter_dit.evaluate_obstacle_shift import shift_relevant_obstacle


@pytest.fixture(scope='module', autouse=True)
def collector_imports():
    benchmark._load_collector(benchmark.COLLECTOR)


@pytest.fixture
def condition():
    from expert_trajectory_collector.surface.geometry import default_condition
    result = default_condition('plane')
    result['surface_geometry']['origin'] = [0., 0., 1.]
    result['surface_chart']['forbidden_regions'] = []
    result['environment']['bounds'] = {'min': [-4., -4., 0.], 'max': [4., 4., 4.]}
    result['environment']['obstacles'] = [
        {'id': 'floor', 'type': 'box', 'role': 'floor', 'size': [8., 8., .1],
         'position': [0., 0., -.05], 'quaternion_wxyz': [1., 0., 0., 0.]},
        {'id': 'background', 'type': 'box', 'role': 'obstacle', 'size': [.2, .2, .2],
         'position': [3., 3., 1.], 'quaternion_wxyz': [1., 0., 0., 0.]},
    ]
    return benchmark.clean_input_condition(result)


def source_scene(condition):
    return {'schema_version': 'obstacle_scene_v001', 'environment_id': 'test-map',
            'bounds': copy.deepcopy(condition['environment']['bounds']),
            'generation_parameters': {'global_yaw_deg': 0., 'translate_x': 0., 'translate_y': 0.},
            'obstacles': [{'id': box['id'], 'type': 'box', 'role': box['role'], 'collision': True,
                           'size_xyz': box['size'], 'pose': {'position': box['position'],
                           'quaternion_wxyz': box['quaternion_wxyz']}}
                          for box in condition['environment']['obstacles']]}


def test_schedule_and_thermal_budget():
    rows = benchmark.condition_schedule(0)
    assert len(rows) == 10
    for kind in ('plane', 'cylinder'):
        selected = [row for row in rows if row['kind'] == kind]
        assert len(selected) == 5
        assert sum(row['difficulty'] == 'detour' for row in selected) == 3
    assert {row['angle_band'] for row in rows} == {0, 1, 2}
    assert benchmark.condition_schedule(0) != benchmark.condition_schedule(1)
    with pytest.raises(ValueError, match='thermal'):
        benchmark.ContactBenchmarkConfig(workers=3).validate()
    with pytest.raises(ValueError, match='divisible'):
        benchmark.ContactBenchmarkConfig(source_map_count=11).validate()


def test_nested_whitelist_rejects_oracle_leakage(condition):
    raw = copy.deepcopy(condition)
    raw['campaign_metadata'] = {'route_certificates': [{'intrinsic_states': [[0, 0, 0]]}]}
    raw['surface_geometry']['expert_twist'] = .1
    raw['environment']['obstacles'][1]['route_guide'] = 'expert'
    raw['contact_constraints']['attitude_limits_deg']['expert_yaw'] = 1
    assert benchmark.clean_input_condition(raw) == condition
    with pytest.raises(ValueError, match='non-whitelisted'):
        benchmark.parse_input_condition(raw)
    for section in ('surface_geometry', 'start_goal', 'environment'):
        contaminated = copy.deepcopy(condition)
        contaminated[section]['route_certificates'] = []
        with pytest.raises(ValueError, match='non-whitelisted'):
            benchmark.parse_input_condition(contaminated)


def test_fingerprints_ignore_ids_order_sign_and_authored_rigid_transform(condition):
    scene = source_scene(condition)
    changed = copy.deepcopy(scene)
    changed['obstacles'].reverse()
    for index, box in enumerate(changed['obstacles']):
        box['id'] = f'changed-{index}'
        box['pose']['quaternion_wxyz'] = [-v for v in box['pose']['quaternion_wxyz']]
    assert benchmark.geometry_fingerprint(scene) == benchmark.geometry_fingerprint(changed)
    yaw = math.radians(27)
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                         [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    transformed = copy.deepcopy(scene)
    transformed['generation_parameters'] = {'global_yaw_deg': 27., 'translate_x': .3, 'translate_y': -.2}
    for box in transformed['obstacles']:
        box['pose']['position'] = (rotation @ box['pose']['position'] + [.3, -.2, 0]).tolist()
        box['pose']['quaternion_wxyz'] = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
    assert benchmark.geometry_fingerprint(scene) != benchmark.geometry_fingerprint(transformed)
    assert benchmark.geometry_fingerprint(scene, canonical=True) == benchmark.geometry_fingerprint(transformed, canonical=True)
    transformed['obstacles'][1]['size_xyz'] = [.21, .2, .2]
    assert benchmark.geometry_fingerprint(scene, canonical=True) != benchmark.geometry_fingerprint(transformed, canonical=True)


def test_shift_matches_base_exactly_and_preserves_task_and_original(condition):
    from expert_trajectory_collector.surface.geometry import SurfaceGeometry
    saved = copy.deepcopy(condition)
    geometry = SurfaceGeometry(condition)
    expected, box_id, delta = shift_relevant_obstacle(source_scene(condition),
        geometry.lift(geometry.start), geometry.lift(geometry.goal), .30)
    shifted, metadata = benchmark.shift_contact_condition(condition)
    assert condition == saved
    assert shifted['start_goal'] == condition['start_goal']
    assert shifted['surface_geometry'] == condition['surface_geometry']
    assert metadata['moved_obstacle_id'] == box_id
    assert metadata['displacement_xyz'] == delta
    assert np.linalg.norm(delta) == pytest.approx(.30)
    assert not metadata['structural_fixture_members_moved_together']
    assert [box['position'] for box in shifted['environment']['obstacles']] == [box['pose']['position'] for box in expected['obstacles']]


def test_physical_witness_classification_and_independent_solver_audit(condition, monkeypatch):
    from expert_trajectory_collector.surface.geometry import SurfaceGeometry
    from expert_trajectory_collector.surface import campaign_planner
    config = benchmark.ContactBenchmarkConfig(oracle_attempts=0)
    cert = {'mode': 'direct', 'intrinsic_states': [condition['start_goal']['start'], condition['start_goal']['goal']]}
    feasible = benchmark.certify_feasibility(condition, [cert], 10, config)
    assert feasible['status'] == 'expert_witness_valid'
    assert feasible['certified_feasible']
    unknown = benchmark.certify_feasibility(condition, [], 10, config)
    assert unknown['status'] == 'unknown' and not unknown['certified_feasible']
    assert 'does not prove' in unknown['reason']
    blocked = copy.deepcopy(condition)
    start_pose = SurfaceGeometry(condition).lift(condition['start_goal']['start'])
    blocked['environment']['obstacles'].append({'id': 'endpoint-blocker', 'type': 'box',
        'size': [.2, .2, .2], 'position': start_pose[:3].tolist(), 'quaternion_wxyz': [1, 0, 0, 0]})
    assert benchmark.certify_feasibility(blocked, [cert], 10, config)['status'] == 'endpoint_invalid'
    obstructed = copy.deepcopy(condition)
    obstructed['environment']['obstacles'].append({'id': 'target-overlap', 'type': 'box',
        'size': [.2, .2, .2], 'position': [0, 0, 1.], 'quaternion_wxyz': [1, 0, 0, 0]})
    assert benchmark.certify_feasibility(obstructed, [cert], 10, config)['status'] == 'assembly_invalid'
    config = benchmark.ContactBenchmarkConfig(oracle_attempts=1)
    monkeypatch.setattr(campaign_planner, 'global_query', lambda *args: (np.asarray(cert['intrinsic_states']), .001, {}))
    assert benchmark.certify_feasibility(condition, [], 10, config)['status'] == 'solver_witness_valid'
    invalid = np.asarray(cert['intrinsic_states']); invalid[-1, 0] += .2
    monkeypatch.setattr(campaign_planner, 'global_query', lambda *args: (invalid, .001, {}))
    assert benchmark.certify_feasibility(condition, [], 10, config)['status'] == 'unknown'


def test_frozen_cache_preserves_prepared_masks_and_never_reads_oracle(condition, tmp_path):
    from expert_trajectory_collector.surface.geometry import SurfaceGeometry
    from expert_trajectory_collector.surface.dataset_v1 import surface_features
    config = benchmark.ContactBenchmarkConfig(benchmark_id='unit-fixture', source_map_count=1,
        families=('central_block',), exclusion_datasets=())
    root = tmp_path / 'benchmark'; root.mkdir(); (root / 'maps').mkdir()
    inputs = []; oracles = []
    for index in range(10):
        original = copy.deepcopy(condition)
        original['start_goal']['start'][2] = index * .01
        original['start_goal']['goal'][2] = index * .01
        original['surface_chart']['forbidden_regions'] = [[-.2, .2, 1.5, 1.8]]
        shifted, shift = benchmark.shift_contact_condition(original)
        geometry = SurfaceGeometry(original)
        identifier = f'unit-fixture/map_000000/condition_{index:03d}'
        inputs.append({'schema_version': 'contact_benchmark_input_v001', 'condition_id': identifier,
            'map_id': 'unit-fixture/map_000000', 'source_map_index': 0, 'family': 'central_block',
            'kind': 'plane', 'difficulty': 'easy', 'angle_band': 0,
            'body_endpoints_pose7': [geometry.lift(geometry.start).tolist(), geometry.lift(geometry.goal).tolist()],
            'conditions': {'original': original, 'obstacle_shift': shifted},
            'input_sha256': {'original': benchmark._hash(original), 'obstacle_shift': benchmark._hash(shifted)}, 'shift': shift})
        cert = {'mode': 'direct', 'intrinsic_states': [original['start_goal']['start'], original['start_goal']['goal']]}
        assessment = benchmark.certify_feasibility(original, [cert], index+1, config)
        assert assessment['certified_feasible']
        oracles.append({'condition_id': identifier, 'original': assessment, 'obstacle_shift': copy.deepcopy(assessment)})
    scene = source_scene(condition)
    benchmark._commit_map(root, {'index': 0, 'family': 'central_block', 'map_id': 'unit-fixture/map_000000',
        'replacement': 0, 'source': benchmark._safe_source(scene), 'source_fingerprints': [benchmark.geometry_fingerprint(scene)],
        'generation_parameters': scene['generation_parameters'], 'inputs': inputs, 'oracles': oracles,
        'failures': [], 'elapsed_s': 0.})
    manifest = benchmark._freeze(root, config, {})
    assert manifest['condition_count'] == 10 and manifest['scenario_case_count'] == 20
    with np.load(root / 'runtime_inputs.npz', allow_pickle=False) as arrays:
        assert arrays['conditions'].shape == (20, 18)
        assert arrays['obstacles'].shape == (20, 32, 10)
        assert arrays['surface_features'].shape == (20, 32)
        assert arrays['surface_mask'].shape == (20, 32)
        assert arrays['forbidden_regions'].shape == (20, 8, 4)
        assert arrays['forbidden_mask'].shape == (20, 8)
        assert arrays['support_tokens'].shape == (20, 8, 10)
        prepared = surface_features(inputs[0]['conditions']['original'])
        np.testing.assert_array_equal(arrays['surface_mask'][0], prepared['surface_feature_mask'])
        np.testing.assert_array_equal(arrays['forbidden_mask'][0], prepared['forbidden_region_mask'])
        assert not arrays['surface_mask'][:, 14].any() and arrays['surface_mask'][:, 15].all()
        assert not any('intrinsic_states' in key or 'certificate' in key for key in arrays.files)
        assert all(value.dtype != object for value in arrays.values())
    # Oracle access is unnecessary after freeze: simulate withholding it from inference.
    (root / 'oracle' / 'map_000000.jsonl.gz').unlink()
    examples = list(benchmark.iter_benchmark_inputs(root, 'obstacle_shift', feasible_only=True))
    assert len(examples) == 10
    assert all('certificate' not in json.dumps(example) for example in examples)
    contaminated = copy.deepcopy(inputs[0]); contaminated['route_certificates'] = []
    with pytest.raises(ValueError, match='non-whitelisted'):
        benchmark._parse_input_record(contaminated)
    rows = benchmark._read_gz_jsonl(root / 'maps/map_000000/inputs.jsonl.gz')
    rows[0]['conditions']['original']['start_goal']['start'][0] += .01
    benchmark._write_jsonl_gz(root / 'maps/map_000000/inputs.jsonl.gz', rows)
    with pytest.raises(ValueError, match='checksum'):
        next(benchmark.iter_benchmark_inputs(root))
