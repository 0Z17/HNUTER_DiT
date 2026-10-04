"""Causal history and frozen-dataset integrity tests; no GPU or simulation."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hnuter_dit.temporal_data import _snapshot_sources, load_dataset, sample_from_dataset
from hnuter_dit.temporal_observations import (FUTURE_TIMES, HISTORY_TIMES, TEMPORAL_SCHEMA,
                                            encode_history)


def history(origin=(0., 0., 0.)):
    origin = np.asarray(origin)
    frames = []
    fixed_wall = origin + np.array([[3., -.2, .2], [3., .2, .5]])
    for index, t in enumerate(HISTORY_TIMES + 1.):
        sensor = np.eye(4)
        sensor[:3, 3] = origin + [t-1., 0., .13]
        frames.append(SimpleNamespace(points=fixed_wall-sensor[:3, 3], world_T_sensor=sensor,
                                      sim_time=t, frame_index=index,
                                      point_timestamps=np.full(2, t)))
    current = dict(position=origin.copy(), velocity=np.array([.1, .2, 0.]), quaternion=np.array([1., 0., 0., 0.]))
    return frames, current


def test_history_ego_motion_is_compensated_before_token_encoding():
    frames, current = history()
    result = encode_history(frames, current, [4., 0., 0.])
    for tokens in result['history_tokens'][1:]:
        np.testing.assert_allclose(tokens, result['history_tokens'][0], atol=1e-6)
    np.testing.assert_allclose(result['condition'], [4, 0, 0, .1, .2, 0, 1, 0, 0, 0, 1, 0])
    assert result['metadata']['causal']


def test_one_current_canonical_frame_and_world_translation_equivariance():
    frames, current = history()
    first = encode_history(frames, current, [0., 4., 0.])
    moved_frames, moved_state = history((-25., -35., 2.2))
    moved = encode_history(moved_frames, moved_state, [-25., -31., 2.2])
    np.testing.assert_allclose(first['history_tokens'], moved['history_tokens'], atol=3e-6)
    np.testing.assert_allclose(first['condition'], moved['condition'], atol=1e-6)
    np.testing.assert_allclose(first['condition'][3:], [.2, -.1, 0., 0., -1., 0., 1., 0., 0.], atol=1e-6)
    active = first['history_tokens'][first['history_mask']]
    np.testing.assert_allclose(active[:, 6:], np.tile([2**-.5, 0, 0, -2**-.5], (len(active), 1)), atol=1e-6)


def test_missing_stale_or_future_history_is_rejected_without_padding():
    frames, current = history()
    with pytest.raises(ValueError, match='four'):
        encode_history(frames[:-1], current, [4, 0, 0])
    with pytest.raises(ValueError, match='timestamps'):
        encode_history(frames, current, [4, 0, 0], query_time=1.1)
    frames[0].sim_time = 1.2
    with pytest.raises(ValueError, match='timestamps'):
        encode_history(frames, current, [4, 0, 0])


def test_teacher_future_attributes_cannot_change_student_features():
    frames, current = history()
    before = encode_history(frames, current, [4, 0, 0])
    for frame in frames:
        frame.gt_boxes = np.full((65, 4, 10), np.nan)
        frame.teacher_velocity = [999, 999, 999]
    after = encode_history(frames, current, [4, 0, 0])
    for key in ('history_tokens', 'history_mask', 'history_times', 'condition'):
        np.testing.assert_array_equal(before[key], after[key])


def dataset_fixture(directory, change=None):
    n, rays = 4, 2
    tokens = np.zeros((n, 4, 32, 10), dtype=np.float32)
    tokens[..., 6] = 1
    tokens[:, :, 0, 0] = 3
    tokens[:, :, 0, 3:6] = .5
    mask = np.zeros((n, 4, 32), dtype=bool)
    mask[:, :, 0] = True
    boxes = np.zeros((n, 65, 4, 10), dtype=np.float32)
    boxes[..., 0], boxes[..., 3:6], boxes[..., 6] = 3, .5, 1
    data = dict(history_tokens=tokens, history_mask=mask, condition=np.zeros((n, 12)),
                positions=np.zeros((n, 65, 3)), history_times=HISTORY_TIMES, future_times=FUTURE_TIMES,
                group_id=np.array([0, 0, 1, 1]), episode_id=np.arange(n), pair_variant=np.array([0, 1, 0, 1]),
                split=np.array([0, 0, 1, 1]), sample_id=np.array(['a', 'b', 'c', 'd']),
                origin_world=np.zeros((n, 3)), world_to_canonical=np.tile(np.eye(3), (n, 1, 1)),
                gt_boxes=boxes, gt_mask=np.ones((n, 65, 4), dtype=bool),
                history_ray_ranges=np.full((n, 4, rays), 3.),
                ray_directions=np.tile(np.array([[1., 0, 0], [0, 1., 0]]), (4, 1, 1)),
                history_sensor_poses=np.tile(np.eye(4), (n, 4, 1, 1)),
                teacher_scenario_json=np.array(['{}']*n))
    if change:
        change(data)
    np.savez_compressed(directory/'dataset.npz', **data)
    digest = hashlib.sha256((directory/'dataset.npz').read_bytes()).hexdigest()
    freeze = dict(schema=TEMPORAL_SCHEMA, status='frozen', dataset_file='dataset.npz', dataset_sha256=digest,
                  samples=n, independent_groups=2, episodes=4, split_samples=dict(train=2, validation=2, test=0),
                  sensor=dict(rays_per_frame=rays))
    (directory/'freeze.json').write_text(json.dumps(freeze))


def test_frozen_dataset_roundtrip_and_sampling_preserve_time_constants(tmp_path):
    dataset_fixture(tmp_path)
    data = load_dataset(tmp_path)
    sample = sample_from_dataset(data, 2)
    assert sample['sample_id'] == 'c' and int(sample['group_id']) == 1
    assert sample['positions'].shape == (65, 3)
    np.testing.assert_array_equal(sample['future_times'], FUTURE_TIMES)
    assert sample['teacher_scenario'] == {}


def test_hash_change_is_rejected(tmp_path):
    dataset_fixture(tmp_path)
    with (tmp_path/'dataset.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        load_dataset(tmp_path)


@pytest.mark.parametrize('change', [
    lambda d: d['split'].__setitem__(1, 1),
    lambda d: d['history_tokens'].__setitem__((1, -1, 0, 0), 4),
    lambda d: d.__setitem__('history_mask', d['history_mask'].astype(float)),
    lambda d: d['gt_boxes'].__setitem__((0, 0, 0, 3), -1),
    lambda d: d['gt_boxes'].__setitem__((0, 0, 0, 0), np.nan),
    lambda d: d['gt_boxes'].__setitem__((0, 0, 0, 6), 0),
    lambda d: d['history_sensor_poses'].__setitem__((0, 0, 0, 0), 2),
    lambda d: d['history_ray_ranges'].__setitem__((1, -1, 0), 4),
])
def test_semantic_corruption_is_rejected_even_with_updated_hash(tmp_path, change):
    dataset_fixture(tmp_path, change)
    with pytest.raises(ValueError):
        load_dataset(tmp_path)


def test_source_snapshot_remains_verifiable_after_original_source_changes(tmp_path):
    dataset_fixture(tmp_path)
    source = tmp_path/'collector.py'
    source.write_text('version = 1\n')
    hashes, snapshots = _snapshot_sources(tmp_path, [source])
    freeze = json.loads((tmp_path/'freeze.json').read_text())
    freeze.update(source_sha256=hashes, source_snapshots=snapshots)
    (tmp_path/'freeze.json').write_text(json.dumps(freeze))
    source.write_text('version = 2\n')
    load_dataset(tmp_path)
    (tmp_path/next(iter(snapshots.values()))).write_text('tampered\n')
    with pytest.raises(ValueError, match='source snapshot hash'):
        load_dataset(tmp_path)


def test_reference_targets_require_separate_executed_trajectory_evidence(tmp_path):
    dataset_fixture(tmp_path)
    freeze = json.loads((tmp_path/'freeze.json').read_text())
    freeze['target_semantics'] = 'physically_validated_teacher_reference'
    (tmp_path/'freeze.json').write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match='actual physical trajectory'):
        load_dataset(tmp_path)
