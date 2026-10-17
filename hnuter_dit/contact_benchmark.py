"""Frozen, new-map contact inputs with physically audited oracles kept separate.

The inference reader rejects provenance/route fields.  Oracle polylines are
feasibility witnesses only, never targets for a model, projection, or guidance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict, dataclass
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from collections import Counter
from typing import Any, Iterator

import numpy as np

from .evaluate_obstacle_shift import shift_relevant_obstacle

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = Path('/home/z017/research/HNUTER_trajectory_collector')
DATASETS = Path('/home/z017/research/HNUTER_datasets')
FAMILIES = ('sparse_obb_clutter', 'central_block', 'multi_homotopy', 'narrow_passage',
            'orientation_sensitive_passage', 'wall_protrusion_bracket', 'frame_doorway',
            'pillar_wall', 'staggered_corridor', 'mixed_industrial')
VERSION = 'contact_new_map_benchmark_v001'
FEASIBLE = {'expert_witness_valid', 'solver_witness_valid'}
SCENARIOS = ('original', 'obstacle_shift')
_INITIAL_EXCLUSIONS: set[str] = set()
_BUILD_ROOT: Path | None = None
BOX_FIELDS = ('id', 'type', 'role', 'size', 'position', 'quaternion_wxyz')
SURFACE_FIELDS = ('type', 'origin', 'quaternion_wxyz', 'radius_m', 'thickness_m',
                  'assembly_type', 'target_obstacle_id')
LIMIT_FIELDS = ('position_tolerance_m', 'normal_tolerance_rad', 'collision_margin_m',
                'validation_step_m', 'validation_step_rad', 'attitude_limits_deg')


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(base: int, *parts: Any) -> int:
    return max(1, int.from_bytes(hashlib.sha256('/'.join(map(str, (base, VERSION, *parts))).encode()).digest()[:4], 'big') & 0x7fffffff)


def _load_collector(path: str | Path) -> None:
    root = Path(path).resolve()
    if not (root / 'obstacle_scene_builder.py').is_file():
        raise FileNotFoundError(f'collector root not found: {root}')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


@dataclass(frozen=True)
class ContactBenchmarkConfig:
    benchmark_id: str = VERSION
    source_map_count: int = 1000
    base_seed: int = 20261011
    families: tuple[str, ...] = FAMILIES
    workers: int = 1
    maximum_map_replacements: int = 50
    maximum_condition_retries: int = 8
    oracle_solve_time_s: float = .25
    oracle_attempts: int = 2
    collector_root: str = str(COLLECTOR)
    exclusion_datasets: tuple[str, ...] = (
        str(DATASETS / 'surface_multifamily_contact_10000traj_v001'),
        str(DATASETS / 'free_flight_159103traj_dit_v001'),
    )

    def validate(self) -> None:
        if not self.benchmark_id or Path(self.benchmark_id).name != self.benchmark_id:
            raise ValueError('benchmark_id must be a single directory name')
        if not self.families or len(set(self.families)) != len(self.families) or not set(self.families) <= set(FAMILIES):
            raise ValueError('families must be distinct supported scene families')
        if self.source_map_count <= 0 or self.source_map_count % len(self.families):
            raise ValueError('source_map_count must be positive and divisible by family count')
        if not 1 <= self.workers <= 2:
            raise ValueError('thermal resource budget permits only one or two workers')
        if not 0 <= self.maximum_map_replacements <= 100 or not 1 <= self.maximum_condition_retries <= 100:
            raise ValueError('invalid rejection sampling budget')
        if not 0 <= self.oracle_attempts <= 20 or not 0 < self.oracle_solve_time_s <= 10:
            raise ValueError('invalid oracle query budget')
        if self.base_seed <= 0:
            raise ValueError('base_seed must be positive')

    def definition(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop('workers')
        return {'schema_version': VERSION, **value, 'conditions_per_source': 10,
                'shift_distance_m': .30, 'input_schema': 'contact_benchmark_input_v001',
                'feasibility_policy': 'dense_full_urdf_witness; unknown_is_not_proven_infeasible'}


def condition_schedule(map_index: int) -> list[dict[str, Any]]:
    """Six detours and four direct tasks; easy angle bins rotate across maps."""
    rows = []
    for kind in ('plane', 'cylinder'):
        for band in ((0, 1, 0) if kind == 'plane' else (0, 1, 2)):
            rows.append({'index': len(rows), 'kind': kind, 'difficulty': 'detour', 'angle_band': band})
        easy = (map_index % 2, 2) if kind == 'plane' else (0, 1 + map_index % 2)
        for band in easy:
            rows.append({'index': len(rows), 'kind': kind, 'difficulty': 'easy', 'angle_band': band})
    return rows


def _selected(value: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {k: copy.deepcopy(value[k]) for k in fields if k in value}


def clean_input_condition(condition: dict[str, Any]) -> dict[str, Any]:
    """Whitelist nested fields rather than trying to remove known expert labels."""
    result = _selected(condition, ('schema_version', 'task_type', 'task_version'))
    geometry = condition['surface_geometry']
    result['surface_geometry'] = _selected(geometry, SURFACE_FIELDS)
    result['surface_geometry']['supports'] = [_selected(box, BOX_FIELDS) for box in geometry.get('supports', [])]
    result['surface_chart'] = _selected(condition['surface_chart'], ('bounds', 'forbidden_regions'))
    result['tool_extrinsics'] = _selected(condition['tool_extrinsics'], ('position_body', 'quaternion_wxyz'))
    result['contact_constraints'] = _selected(condition['contact_constraints'], LIMIT_FIELDS)
    attitude = result['contact_constraints'].get('attitude_limits_deg')
    if attitude is not None:
        result['contact_constraints']['attitude_limits_deg'] = _selected(attitude, ('roll', 'pitch'))
    result['start_goal'] = _selected(condition['start_goal'], ('start', 'goal'))
    result['environment'] = {'bounds': _selected(condition['environment']['bounds'], ('min', 'max')),
                             'obstacles': [_selected(box, BOX_FIELDS) for box in condition['environment']['obstacles']]}
    _json(result)
    return result


def parse_input_condition(condition: dict[str, Any]) -> dict[str, Any]:
    """Reject an input containing any field outside the inference whitelist."""
    clean = clean_input_condition(condition)
    if _json(clean) != _json(condition):
        raise ValueError('input contains non-whitelisted fields; route/oracle metadata is forbidden')
    if clean.get('task_type') != 'surface':
        raise ValueError('input must be a Surface task')
    return clean


def _parse_input_record(record: dict[str, Any]) -> dict[str, Any]:
    fields = {'schema_version', 'condition_id', 'map_id', 'source_map_index', 'family',
              'kind', 'difficulty', 'angle_band', 'body_endpoints_pose7', 'conditions',
              'input_sha256', 'shift'}
    if set(record) != fields or record['schema_version'] != 'contact_benchmark_input_v001':
        raise ValueError('input record contains non-whitelisted fields; oracle/route fields are forbidden')
    if set(record['conditions']) != set(SCENARIOS) or set(record['input_sha256']) != set(SCENARIOS):
        raise ValueError('input record must contain exactly original and obstacle_shift scenarios')
    shift_fields = {'moved_obstacle_id', 'displacement_xyz', 'shift_distance_m', 'shift_policy',
                    'structural_fixture_members_moved_together'}
    if set(record['shift']) != shift_fields:
        raise ValueError('shift metadata contains non-whitelisted fields')
    for condition in record['conditions'].values():
        parse_input_condition(condition)
    return record


def geometry_fingerprint(scene: dict[str, Any], *, canonical: bool = False) -> str:
    """Ignore IDs, order and quaternion sign; optionally remove authored yaw/XY shift."""
    params = scene.get('generation_parameters', {})
    yaw = math.radians(float(params.get('global_yaw_deg', 0))) if canonical else 0.
    ca, sa = math.cos(yaw), math.sin(yaw)
    inverse = np.array([[ca, sa, 0], [-sa, ca, 0], [0, 0, 1.]])
    translation = np.array([params.get('translate_x', 0), params.get('translate_y', 0), 0.]) if canonical else np.zeros(3)
    boxes = []
    for box in scene.get('obstacles', []):
        if not box.get('collision', True) or box.get('type', 'box') != 'box' or (canonical and box.get('role') == 'floor'):
            continue
        pose = box.get('pose', box)
        q = np.asarray(pose['quaternion_wxyz'], dtype=float)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        matrix = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                           [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                           [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        digits = 5 if canonical else 6
        position = inverse @ (np.asarray(pose.get('position', box.get('position'))) - translation)
        shape = np.asarray(box.get('size_xyz', box.get('size')), dtype=float)
        boxes.append({'size': np.round(shape, digits).tolist(), 'position': np.round(position, digits).tolist(),
                      'rotation': np.round(inverse @ matrix, digits).tolist()})
    # Normalize -0.0 before sorting/hashing floating point rigid transforms.
    for box in boxes:
        for key, value in box.items():
            array = np.asarray(value); array[array == 0] = 0.; box[key] = array.tolist()
    return _hash({'boxes': sorted(boxes, key=_json),
                  'workspace_extent': (np.asarray(scene['bounds']['max']) - scene['bounds']['min']).tolist()})


def _exclusions(config: ContactBenchmarkConfig) -> tuple[set[str], dict[str, Any]]:
    blocked = set(); provenance = {}
    for name in config.exclusion_datasets:
        root = Path(name).resolve()
        index = root / 'environments/index.jsonl'
        if not index.is_file():
            raise FileNotFoundError(f'exclusion dataset has no source-map index: {root}')
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        for row in rows:
            scene = json.loads((root / row['path']).read_text())
            blocked.update((geometry_fingerprint(scene), geometry_fingerprint(scene, canonical=True)))
        provenance[str(root)] = {'source_maps': len(rows), 'index_sha256': _file_hash(index),
                                 'all_splits_excluded': True}
    return blocked, provenance


def shift_contact_condition(condition: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The original Base one-box 0.30m displacement, without validity filtering."""
    from expert_trajectory_collector.surface.geometry import SurfaceGeometry
    clean = parse_input_condition(condition)
    geometry = SurfaceGeometry(clean)
    start, goal = geometry.lift(geometry.start), geometry.lift(geometry.goal)
    scene = {'obstacles': [{'id': box['id'], 'type': 'box', 'collision': True,
                           'role': box.get('role', 'obstacle'), 'size_xyz': box['size'],
                           'pose': {'position': box['position'], 'quaternion_wxyz': box['quaternion_wxyz']}}
                          for box in clean['environment']['obstacles']]}
    changed, box_id, displacement = shift_relevant_obstacle(scene, start, goal, .30)
    for box, updated in zip(clean['environment']['obstacles'], changed['obstacles']):
        box['position'] = updated['pose']['position']
    return clean, {'moved_obstacle_id': box_id, 'displacement_xyz': displacement,
                   'shift_distance_m': .30, 'shift_policy': 'base_nearest_box_toward_body_endpoint_segment',
                   'structural_fixture_members_moved_together': False}


def certify_feasibility(condition: dict[str, Any], certificates: list[dict[str, Any]],
                        seed: int, config: ContactBenchmarkConfig) -> dict[str, Any]:
    """Validate witnesses; failure to find one remains unknown, never infeasible."""
    from expert_trajectory_collector.surface.geometry import SurfaceGeometry
    from expert_trajectory_collector.surface.random_placement import assembly_valid
    from expert_trajectory_collector.surface.campaign_planner import CachedValidityChecker, global_query
    from expert_trajectory_collector.surface import planner
    clean = parse_input_condition(condition)
    geometry = SurfaceGeometry(clean)
    if not assembly_valid(geometry):
        return {'status': 'assembly_invalid', 'certified_feasible': False, 'certificates': [],
                'reason': 'contact target/support assembly intersects an obstacle or leaves workspace'}
    checker = CachedValidityChecker(geometry)
    if not checker.check(geometry.start) or not checker.check(geometry.goal):
        return {'status': 'endpoint_invalid', 'certified_feasible': False, 'certificates': [],
                'reason': 'full contact endpoint validity failed'}
    valid = []
    for cert in certificates:
        states = np.asarray(cert['intrinsic_states'], dtype=float)
        if (states.ndim == 2 and states.shape[1] == 3 and len(states) >= 2
                and np.allclose(states[[0, -1]], [geometry.start, geometry.goal], rtol=0, atol=1e-10)
                and planner.audit(checker, states)):
            valid.append({'mode': cert.get('mode', 'unclassified'), 'intrinsic_states': states.tolist(),
                          'witness_origin': 'generator_feasibility_certificate'})
    if valid:
        return {'status': 'expert_witness_valid', 'certified_feasible': True, 'certificates': valid,
                'witness_semantics': 'dense finite-subdivision physical feasibility; not an expert training label'}
    attempts = []
    for attempt in range(config.oracle_attempts):
        query_seed = _seed(seed, 'oracle', attempt)
        try:
            # No certificate corridor or fixed waypoint is supplied to this oracle query.
            with planner._LOCK:
                states, elapsed, statistics = global_query(checker, query_seed, config.oracle_solve_time_s)
            if (not np.allclose(states[[0, -1]], [geometry.start, geometry.goal], rtol=0, atol=1e-10)
                    or not planner.audit(checker, states)):
                raise RuntimeError('oracle solution failed independent endpoint/edge audit')
            return {'status': 'solver_witness_valid', 'certified_feasible': True,
                    'certificates': [{'mode': 'unclassified', 'intrinsic_states': states.tolist(),
                                      'witness_origin': 'global_ompl_query'}],
                    'query_seed': query_seed, 'elapsed_s': elapsed, 'sampler_statistics': statistics,
                    'rng_policy': 'seeded_sampler; planner_process_global_rng', 'attempts': attempts}
        except RuntimeError as error:
            attempts.append({'seed': query_seed, 'reason': str(error)})
    return {'status': 'unknown', 'certified_feasible': False, 'certificates': [], 'attempts': attempts,
            'reason': 'bounded oracle found no witness; this does not prove infeasibility'}


def _safe_source(scene: dict[str, Any]) -> dict[str, Any]:
    # Physical source scene for reference metrics; no precheck pairs/route guides.
    result = _selected(scene, ('schema_version', 'environment_id', 'bounds', 'sampling_space'))
    result['obstacles'] = [
        {**_selected(box, ('id', 'type', 'collision', 'role', 'size_xyz')),
         'pose': _selected(box['pose'], ('position', 'quaternion_wxyz'))}
        for box in scene['obstacles']]
    return result


def _initialize_worker(blocked: set[str], root: str) -> None:
    global _INITIAL_EXCLUSIONS, _BUILD_ROOT
    _INITIAL_EXCLUSIONS = blocked
    _BUILD_ROOT = Path(root)


def _map_status(index: int, **fields: Any) -> None:
    if _BUILD_ROOT is not None:
        _atomic_json(_BUILD_ROOT / 'progress' / f'map_{index:06d}.json',
                     {'source_map_index': index, 'updated_at_unix': time.time(), **fields})


def _build_map(config: ContactBenchmarkConfig, index: int, first_replacement: int,
               blocked: set[str] | None) -> dict[str, Any]:
    blocked = _INITIAL_EXCLUSIONS if blocked is None else blocked
    _load_collector(config.collector_root)
    from expert_trajectory_collector.surface.campaign_sampling import generate_source_environment, generate_contact_condition
    from expert_trajectory_collector.surface.geometry import SurfaceGeometry
    family = config.families[index % len(config.families)]
    failures = []; began = time.monotonic()
    for replacement in range(first_replacement, config.maximum_map_replacements + 1):
        try:
            _map_status(index, state='source_generation', family=family, replacement=replacement)
            source = generate_source_environment(index, family, _seed(config.base_seed, config.benchmark_id, index, replacement), 'test')
            fingerprints = list(dict.fromkeys((geometry_fingerprint(source), geometry_fingerprint(source, canonical=True))))
            if any(value in blocked for value in fingerprints):
                raise ValueError('source geometry overlaps an excluded/new source map')
            inputs = []; oracles = []
            condition_hashes = set()
            map_id = f'{config.benchmark_id}/map_{index:06d}'
            for slot in condition_schedule(index // len(config.families)):
                for retry in range(config.maximum_condition_retries):
                    seed = _seed(config.base_seed, config.benchmark_id, index, replacement, slot['index'], retry)
                    try:
                        _map_status(index, state='contact_generation', family=family, replacement=replacement,
                                    condition_index=slot['index'], retry=retry, accepted_conditions=len(inputs))
                        raw = generate_contact_condition(source, slot['kind'], slot['difficulty'], seed, slot['angle_band'])
                        original = clean_input_condition(raw)
                        if _hash(original) in condition_hashes:
                            raise ValueError('same-map contact conditions are geometrically duplicated')
                        certificates = copy.deepcopy(raw['campaign_metadata']['route_certificates'])
                        if slot['difficulty'] == 'easy':
                            certificates = [{'mode': 'direct', 'intrinsic_states': [original['start_goal']['start'], original['start_goal']['goal']]}]
                        original_oracle = certify_feasibility(original, certificates, seed, config)
                        if not original_oracle['certified_feasible']:
                            raise ValueError('unperturbed condition lacks an audited feasible witness')
                        break
                    except (RuntimeError, ValueError) as error:
                        failures.append({'replacement': replacement, 'condition_index': slot['index'],
                                         'retry': retry, 'seed': seed, 'stage': 'condition', 'reason': str(error)})
                else:
                    raise ValueError(f'condition {slot["index"]} exhausted task seeds')
                shifted, shift = shift_contact_condition(original)
                condition_hashes.add(_hash(original))
                _map_status(index, state='shift_oracle', family=family, replacement=replacement,
                            condition_index=slot['index'], accepted_conditions=len(inputs))
                shifted_oracle = certify_feasibility(shifted, certificates, _seed(seed, 'shifted'), config)
                condition_id = f'{map_id}/condition_{slot["index"]:03d}'
                geometry = SurfaceGeometry(original)
                endpoints = [geometry.lift(geometry.start).tolist(), geometry.lift(geometry.goal).tolist()]
                inputs.append({'schema_version': 'contact_benchmark_input_v001', 'condition_id': condition_id,
                               'map_id': map_id, 'source_map_index': index, 'family': family,
                               'kind': slot['kind'], 'difficulty': slot['difficulty'], 'angle_band': slot['angle_band'],
                               'body_endpoints_pose7': endpoints, 'conditions': {'original': original, 'obstacle_shift': shifted},
                               'input_sha256': {name: _hash(value) for name, value in (('original', original), ('obstacle_shift', shifted))},
                               'shift': shift})
                oracles.append({'condition_id': condition_id, 'original': original_oracle, 'obstacle_shift': shifted_oracle,
                                'generator_orientation': raw['campaign_metadata']['sampled_orientation'],
                                'condition_seed': seed, 'generator_rejection_counts': raw['campaign_metadata']['rejection_counts']})
            return {'index': index, 'family': family, 'map_id': map_id, 'replacement': replacement,
                    'source': _safe_source(source), 'source_fingerprints': fingerprints,
                    'generation_parameters': source['generation_parameters'], 'inputs': inputs, 'oracles': oracles,
                    'failures': failures, 'elapsed_s': time.monotonic() - began}
        except (RuntimeError, ValueError) as error:
            failures.append({'replacement': replacement, 'stage': 'source_map', 'reason': str(error)})
            _map_status(index, state='source_rejected', family=family, replacement=replacement, reason=str(error))
    _map_status(index, state='capacity_exhausted', family=family, failures=failures)
    raise RuntimeError(f'map {index}/{family} exhausted source capacity: {_json(failures)}')


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf8') as stream:
        stream.write(_json(value) + '\n'); temporary = Path(stream.name)
    os.replace(temporary, path)


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            for row in rows:
                compressed.write((_json(row) + '\n').encode())


def _commit_map(root: Path, result: dict[str, Any]) -> None:
    staging = Path(tempfile.mkdtemp(prefix='.map-', dir=root))
    try:
        _write_jsonl_gz(staging / 'inputs.jsonl.gz', result['inputs'])
        _atomic_json(staging / 'source.json', result['source'])
        metadata = {key: value for key, value in result.items() if key not in ('inputs', 'oracles', 'source')}
        metadata['file_sha256'] = {name: _file_hash(staging / name) for name in ('inputs.jsonl.gz', 'source.json')}
        oracle = root / 'oracle' / f'map_{result["index"]:06d}.jsonl.gz'
        # Isolated directory is never consulted by inference/parsing/cache code.
        temporary_oracle = staging / 'oracle.jsonl.gz'
        _write_jsonl_gz(temporary_oracle, result['oracles'])
        metadata['oracle_sha256'] = _file_hash(temporary_oracle)
        oracle.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_oracle, oracle)
        _atomic_json(staging / 'metadata.json', metadata)
        destination = root / 'maps' / f'map_{result["index"]:06d}'
        if destination.exists():
            raise ValueError('refusing to overwrite an accepted source map')
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _read_gz_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def _snapshot_hashes(config: ContactBenchmarkConfig) -> dict[str, str]:
    files = [Path(__file__), ROOT / 'hnuter_dit/evaluate_obstacle_shift.py']
    collector = Path(config.collector_root)
    files.extend(collector / name for name in ('obstacle_scene_builder.py', 'coal_collision.py',
        'expert_trajectory_collector/surface/campaign_sampling.py', 'expert_trajectory_collector/surface/geometry.py',
        'expert_trajectory_collector/surface/random_placement.py', 'expert_trajectory_collector/surface/campaign_planner.py',
        'expert_trajectory_collector/surface/planner.py', 'expert_trajectory_collector/surface/scenes.py',
        'expert_trajectory_collector/surface/dataset_v1.py', 'expert_trajectory_collector/tasks/surface.py',
        'expert_trajectory_collector/assets/HDJQR-0102-0055.SLDASM.urdf'))
    return {str(path.resolve()): _file_hash(path) for path in files}


def benchmark_status(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    status = json.loads((root / 'status.json').read_text())
    active = []
    for path in (root / 'progress').glob('map_*.json'):
        row = json.loads(path.read_text())
        if row['state'] != 'complete':
            active.append(row)
    return {**status, 'active_maps': active}


def iter_benchmark_inputs(root: str | Path, scenario: str = 'original',
                          feasible_only: bool = False) -> Iterator[dict[str, Any]]:
    """Return physical inputs only. Feasibility selection uses booleans in index."""
    if scenario not in SCENARIOS:
        raise ValueError('unsupported scenario')
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'frozen':
        raise ValueError('benchmark must be frozen before model evaluation')
    cached_path = None; rows = []
    for line in (root / 'index.jsonl').read_text().splitlines():
        ref = json.loads(line)
        if feasible_only and not ref['certified_feasible'][scenario]:
            continue
        path = root / ref['input_shard']
        if path != cached_path:
            rows = _read_gz_jsonl(path); cached_path = path
        record = _parse_input_record(rows[ref['row_index']])
        condition = parse_input_condition(record['conditions'][scenario])
        if _hash(condition) != record['input_sha256'][scenario] or _hash(condition) != ref['input_sha256'][scenario]:
            raise ValueError('input checksum mismatch')
        yield {key: copy.deepcopy(record[key]) for key in ('condition_id', 'map_id', 'source_map_index',
               'family', 'kind', 'difficulty', 'angle_band', 'body_endpoints_pose7')} | {
               'case_id': record['condition_id'] + '/' + scenario, 'scenario': scenario, 'condition': condition,
               'input_sha256': record['input_sha256'][scenario]}


def _freeze(root: Path, config: ContactBenchmarkConfig, exclusion_info: dict[str, Any]) -> dict[str, Any]:
    _load_collector(config.collector_root)
    from expert_trajectory_collector.surface.dataset_v1 import surface_features
    from expert_trajectory_collector.tasks.surface import SurfaceTaskPlugin
    from expert_trajectory_collector.surface.geometry import SurfaceGeometry
    index = []; matrices = {key: [] for key in ('conditions', 'obstacles', 'obstacle_mask', 'surface_features',
        'surface_mask', 'forbidden_regions', 'forbidden_mask', 'support_tokens', 'support_mask', 'body_endpoints_pose7', 'case_id', 'condition_id',
        'map_id', 'family', 'kind', 'difficulty', 'scenario', 'feasible_mask', 'feasibility_status',
        'input_sha256', 'input_shard', 'input_row', 'source_json')}
    counts = Counter(); status = Counter(); used = set(); seen_conditions = set(); source_parameters = []; checksum_files = {}
    plugin = SurfaceTaskPlugin()
    for map_index in range(config.source_map_count):
        directory = root / 'maps' / f'map_{map_index:06d}'
        metadata = json.loads((directory / 'metadata.json').read_text())
        for value in set(metadata['source_fingerprints']):
            if value in used:
                raise ValueError('new benchmark source maps are geometrically duplicated')
            used.add(value)
        for name, expected in metadata['file_sha256'].items():
            if _file_hash(directory / name) != expected:
                raise ValueError('source map shard checksum mismatch')
            checksum_files[str((directory / name).relative_to(root))] = expected
        oracle_path = root / 'oracle' / f'map_{map_index:06d}.jsonl.gz'
        if _file_hash(oracle_path) != metadata['oracle_sha256']:
            raise ValueError('isolated oracle checksum mismatch')
        checksum_files[str(oracle_path.relative_to(root))] = metadata['oracle_sha256']
        inputs = _read_gz_jsonl(directory / 'inputs.jsonl.gz'); oracles = _read_gz_jsonl(oracle_path)
        if len(inputs) != 10 or len(oracles) != 10:
            raise ValueError('a source map must have ten distinct task conditions')
        source_parameters.append({'family': metadata['family'], **metadata['generation_parameters'],
                                  'replacement': metadata['replacement']})
        for row, (record, oracle) in enumerate(zip(inputs, oracles)):
            record = _parse_input_record(record)
            if record['input_sha256']['original'] in seen_conditions:
                raise ValueError('benchmark contains a repeated original contact condition')
            seen_conditions.add(record['input_sha256']['original'])
            if record['condition_id'] != oracle['condition_id']:
                raise ValueError('isolated oracle belongs to a different condition')
            ref = {'condition_id': record['condition_id'], 'map_id': record['map_id'], 'family': record['family'],
                   'kind': record['kind'], 'difficulty': record['difficulty'], 'angle_band': record['angle_band'],
                   'input_shard': str((directory / 'inputs.jsonl.gz').relative_to(root)), 'row_index': row,
                   'input_sha256': record['input_sha256'],
                   'certified_feasible': {s: oracle[s]['certified_feasible'] for s in SCENARIOS},
                   'feasibility_status': {s: oracle[s]['status'] for s in SCENARIOS}}
            index.append(ref)
            counts[(record['family'], record['kind'], record['difficulty'])] += 1
            for scenario in SCENARIOS:
                condition = parse_input_condition(record['conditions'][scenario])
                if _hash(condition) != record['input_sha256'][scenario]:
                    raise ValueError('per-condition whitelist SHA mismatch')
                payload = plugin.conditioning_payload(condition)
                geometry = SurfaceGeometry(condition)
                endpoints = np.asarray([geometry.lift(geometry.start), geometry.lift(geometry.goal)])
                if not np.allclose(endpoints, record['body_endpoints_pose7'], atol=1e-10, rtol=0):
                    raise ValueError('body endpoints disagree with physical contact input')
                task = surface_features(condition)
                values = {'conditions': payload['condition_pose9'], 'obstacles': payload['base_environment']['tokens'],
                          'obstacle_mask': payload['base_environment']['mask'], 'surface_features': task['surface_features_world'],
                          'surface_mask': task['surface_feature_mask'],
                          'forbidden_regions': task['forbidden_regions_intrinsic'], 'forbidden_mask': task['forbidden_region_mask'],
                          'support_tokens': task['task_support_tokens_world'], 'support_mask': task['task_support_mask'],
                          'body_endpoints_pose7': endpoints, 'case_id': record['condition_id'] + '/' + scenario,
                          'condition_id': record['condition_id'], 'map_id': record['map_id'], 'family': record['family'],
                          'kind': record['kind'], 'difficulty': record['difficulty'], 'scenario': scenario,
                          'feasible_mask': oracle[scenario]['certified_feasible'], 'feasibility_status': oracle[scenario]['status'],
                          'input_sha256': record['input_sha256'][scenario], 'input_shard': ref['input_shard'],
                          'input_row': row, 'source_json': str((directory / 'source.json').relative_to(root))}
                for key, value in values.items():
                    matrices[key].append(value)
                status[(scenario, oracle[scenario]['status'])] += 1
    with (root / 'index.jsonl').open('w') as stream:
        for row in index:
            stream.write(_json(row) + '\n')
    bool_keys = {'obstacle_mask', 'surface_mask', 'forbidden_mask', 'support_mask', 'feasible_mask'}
    float_keys = {'conditions', 'obstacles', 'surface_features', 'forbidden_regions', 'support_tokens', 'body_endpoints_pose7'}
    arrays = {key: np.asarray(value, dtype=bool if key in bool_keys else np.float32 if key in float_keys else None)
              for key, value in matrices.items()}
    np.savez_compressed(root / 'runtime_inputs.npz', **arrays)
    checksum_files['runtime_inputs.npz'] = _file_hash(root / 'runtime_inputs.npz')
    checksum_files['index.jsonl'] = _file_hash(root / 'index.jsonl')
    manifest = {'schema_version': VERSION, 'benchmark_id': config.benchmark_id, 'status': 'frozen',
                'source_map_count': config.source_map_count, 'condition_count': len(index), 'scenario_case_count': 2*len(index),
                'counts': [{'family': f, 'kind': k, 'difficulty': d, 'conditions': n} for (f, k, d), n in sorted(counts.items())],
                'feasibility_counts': [{'scenario': s, 'status': state, 'count': n} for (s, state), n in sorted(status.items())],
                'source_parameters': source_parameters, 'exclusion_datasets': exclusion_info,
                'input_files_sha256': checksum_files, 'config_sha256': _hash(config.definition()),
                'runtime_cache': 'runtime_inputs.npz', 'shift_distance_m': .30,
                'shift_semantics': 'Base one-box move; fixture support is not moved with selected box',
                'stress_selection': 'all frozen conditions, including assembly_invalid/endpoint_invalid/unknown',
                'feasible_selection': 'only audited generator or global-solver witnesses; fixed across all models',
                'oracle_storage': 'oracle/; never a model/projection/guidance input',
                'limitations': ['new-map IID, shared scene families; no semantic OOD claim',
                                '10000 distinct tasks are grouped by 1000 independent source maps; bootstrap by map',
                                'finite dense contact/URDF edge audit, not a continuous-time proof',
                                'bounded oracle failure is unknown, not proven infeasibility',
                                'one-box stress may break a fixture assembly; report raw and feasible subsets separately']}
    _atomic_json(root / 'manifest.json', manifest)
    return manifest


def build_benchmark(config: ContactBenchmarkConfig, output: str | Path, *, resume: bool = False) -> dict[str, Any]:
    """Explicit construction; output must be outside the code repository."""
    config.validate(); root = Path(output).resolve()
    if root.is_relative_to(ROOT):
        raise ValueError('generated benchmark must be outside the Git repository')
    _load_collector(config.collector_root)
    blocked, exclusions = _exclusions(config)
    hashes = _snapshot_hashes(config)
    provenance = {'definition': config.definition(), 'definition_sha256': _hash(config.definition()),
                  'source_sha256': hashes, 'exclusions': exclusions, 'exclusion_fingerprint_sha256': _hash(sorted(blocked)),
                  'python': sys.version, 'worker_budget': config.workers,
                  'numpy': importlib.metadata.version('numpy')}
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise ValueError('output already contains data; use resume with identical definition')
        saved = json.loads((root / 'provenance.json').read_text())
        for key in ('definition_sha256', 'source_sha256', 'exclusion_fingerprint_sha256'):
            if saved[key] != provenance[key]:
                raise ValueError(f'resume rejected: changed {key}')
        if (root / 'manifest.json').is_file():
            manifest = json.loads((root / 'manifest.json').read_text())
            for name, expected in manifest['input_files_sha256'].items():
                if _file_hash(root / name) != expected:
                    raise ValueError(f'frozen benchmark checksum mismatch: {name}')
            return manifest
    root.mkdir(parents=True, exist_ok=True); (root / 'maps').mkdir(exist_ok=True)
    if not (root / 'provenance.json').exists():
        _atomic_json(root / 'provenance.json', provenance)
    pending = []
    for index in range(config.source_map_count):
        path = root / 'maps' / f'map_{index:06d}' / 'metadata.json'
        if path.exists():
            metadata = json.loads(path.read_text()); blocked.update(metadata['source_fingerprints'])
        else:
            pending.append(index)
    began = time.monotonic()
    complete = config.source_map_count - len(pending)
    _atomic_json(root / 'status.json', {'state': 'collecting', 'complete_maps': complete,
        'condition_count': complete*10, 'target_conditions': config.source_map_count*10,
        'workers': config.workers, 'started_at_unix': time.time()})
    with ProcessPoolExecutor(max_workers=config.workers, initializer=_initialize_worker, initargs=(blocked, str(root))) as executor:
        futures = {executor.submit(_build_map, config, index, 0, None): index for index in pending}
        while futures:
            for future in as_completed(tuple(futures)):
                index = futures.pop(future)
                try:
                    result = future.result()
                except BaseException as error:
                    for waiting in futures:
                        waiting.cancel()
                    _atomic_json(root / 'status.json', {'state': 'failed', 'complete_maps': complete,
                        'condition_count': complete*10, 'target_conditions': config.source_map_count*10,
                        'workers': config.workers, 'failed_map': index, 'reason': str(error),
                        'elapsed_s': time.monotonic() - began})
                    raise
                if any(value in blocked for value in result['source_fingerprints']):
                    first = result['replacement'] + 1
                    if first > config.maximum_map_replacements:
                        raise RuntimeError('source duplicate rejection exhausted map replacement budget')
                    futures[executor.submit(_build_map, config, index, first, blocked)] = index
                    continue
                _commit_map(root, result); blocked.update(result['source_fingerprints'])
                _atomic_json(root / 'progress' / f'map_{index:06d}.json',
                             {'state': 'complete', 'source_map_index': index, 'replacement': result['replacement']})
                complete += 1
                _atomic_json(root / 'status.json', {'state': 'collecting', 'complete_maps': complete,
                    'condition_count': complete*10, 'target_conditions': config.source_map_count*10,
                    'workers': config.workers, 'last_map': index, 'elapsed_s': time.monotonic() - began})
                print(f'contact benchmark {complete}/{config.source_map_count} maps', flush=True)
    manifest = _freeze(root, config, exclusions)
    _atomic_json(root / 'status.json', {'state': 'frozen', 'complete_maps': config.source_map_count,
                  'condition_count': manifest['condition_count'], 'elapsed_s': time.monotonic() - began})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark-id')
    parser.add_argument('--source-map-count', type=int)
    parser.add_argument('--workers', type=int, choices=(1, 2))
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--family', choices=FAMILIES, action='append')
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--maximum-map-replacements', type=int, default=50)
    parser.add_argument('--maximum-condition-retries', type=int, default=8)
    parser.add_argument('--oracle-solve-time', type=float, default=.25)
    parser.add_argument('--oracle-attempts', type=int, default=2)
    parser.add_argument('--collector-root', type=Path, default=COLLECTOR)
    parser.add_argument('--exclude-dataset', type=Path, action='append')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(_json(benchmark_status(args.output)))
        return
    if args.benchmark_id is None or args.source_map_count is None or args.workers is None:
        parser.error('construction requires --benchmark-id --source-map-count --workers')
    defaults = ContactBenchmarkConfig()
    config = ContactBenchmarkConfig(benchmark_id=args.benchmark_id, source_map_count=args.source_map_count,
        workers=args.workers, base_seed=args.seed, maximum_map_replacements=args.maximum_map_replacements,
        families=tuple(args.family) if args.family else FAMILIES,
        maximum_condition_retries=args.maximum_condition_retries, oracle_solve_time_s=args.oracle_solve_time,
        oracle_attempts=args.oracle_attempts, collector_root=str(args.collector_root.resolve()),
        exclusion_datasets=tuple(str(p.resolve()) for p in args.exclude_dataset) if args.exclude_dataset else defaults.exclusion_datasets)
    result = build_benchmark(config, args.output, resume=args.resume)
    print(_json({key: result[key] for key in ('status', 'source_map_count', 'condition_count', 'feasibility_counts')}))


if __name__ == '__main__':
    main()
