"""Comparison integrity and behavior tests; no GPU or native simulation."""
import copy

import numpy as np
import pytest

from hnuter_dit.geometry_evaluation import (_native_summary, config_hash, evaluate_variant,
                                           select_new_encoder, split_data, validate_comparison,
                                           validate_protocol_dataset, native_results)


def descriptions():
    return {name: dict(input_representation=name, history_mode='temporal', training_state='trained',
        training_step=1000, source_dataset={'dataset_sha256': 'frozen'},
        training_budget=dict(steps=2000, batch_size=32, seed=7, input_representation=name, learning_rate=.0002,
                             ema_decay=.99, validation_interval=100, sampling_steps=20),
        architecture=dict(dimension=128, depth=4, heads=4, history_depth=2), control_points=16,
        future_times=(np.arange(65)*.125).tolist(), geometry_config={'representation': name},
        encoder_parameter_count=100, parameter_count=1000) for name in ('box', 'point', 'voxel')}


def test_comparison_allows_encoder_intervention_but_rejects_budget_or_head_change():
    original = descriptions()
    validate_comparison(original, 'frozen')
    for section, key, value in [('training_budget', 'seed', 8), ('architecture', 'depth', 5),
                                ('source_dataset', 'dataset_sha256', 'other')]:
        modified = copy.deepcopy(original)
        modified['point'][section][key] = value
        with pytest.raises(ValueError):
            validate_comparison(modified, 'frozen')
    with pytest.raises(ValueError):
        validate_comparison({k: v for k, v in original.items() if k != 'box'}, 'frozen')


def selection_variants():
    return {name: dict(split='validation', summary={
        'group_balanced_feasible_safe_progress_rate': rate, 'position_rmse_m': error})
        for name, rate, error in [('box', .9, .1), ('point', .7, .2), ('voxel', .7, .15)]}


def test_validation_selection_uses_frozen_tiebreak_and_does_not_claim_box_improvement():
    variants = selection_variants()
    selected = select_new_encoder(variants)
    assert selected['representation'] == 'voxel'
    assert not selected['improvement_over_box_observed']
    variants['point']['summary']['position_rmse_m'] = .15
    assert select_new_encoder(variants)['representation'] == 'point'
    variants['box']['split'] = 'test'
    with pytest.raises(ValueError, match='validation'):
        select_new_encoder(variants)


def test_partition_happens_before_preprocessing_and_keeps_constant_ray_patterns():
    data = dict(positions=np.arange(4*65*3).reshape(4,65,3),
                history_ray_ranges=np.arange(4*4*2).reshape(4,4,2),
                ray_directions=np.zeros((4,2,3)), history_times=np.arange(4),
                future_times=np.arange(65), _freeze={'dataset_sha256':'frozen'})
    partition = split_data(data, np.array([2,3]))
    assert partition['positions'].shape == (2,65,3)
    np.testing.assert_array_equal(partition['history_ray_ranges'], data['history_ray_ranges'][[2,3]])
    assert partition['ray_directions'].shape == (4,2,3)
    partition['positions'][0] = -1
    assert data['positions'][2,0,0] != -1


def encoded_pair():
    tokens = np.zeros((2,4,5,3), dtype=np.float32)
    tokens[:, :, :, 0] = np.arange(4)[None,:,None]
    masks = np.zeros((2,4,5), bool)
    masks[:, :, 0] = True
    boxes = np.zeros((2,65,1,10))
    boxes[...,0], boxes[...,3:6], boxes[...,6] = 10., .5, 1.
    return dict(history_tokens=tokens, history_mask=masks, history_times=np.array([-.6,-.4,-.2,0]),
        condition=np.tile([1,0,0,0,0,0,1,0,0,0,1,0],(2,1)), positions=np.zeros((2,65,3)),
        future_times=np.arange(65)*.125, group_id=np.array([9,9]), episode_id=np.array([18,19]),
        pair_variant=np.array([0,1]), sample_id=np.array(['a','b']),
        gt_boxes=boxes, gt_mask=np.ones((2,65,1),bool), teacher_scenario_json=np.array(['{}','{}']),
        _input_representation='point', _preprocessing={'representation':'point'})


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def sample(self, tokens, masks, ages, condition, *, seed, steps):
        self.calls.append(dict(tokens=tokens.copy(), masks=masks.copy(), ages=ages.copy(), seed=seed, steps=steps))
        return dict(positions=np.zeros((65,3)), control_points=np.zeros((16,3)),
                    future_times=np.arange(65)*.125, inference_seconds=.001)


def test_each_condition_gets_one_candidate_same_pair_seed_and_ablation_preserves_unknown_mask():
    runtime = FakeRuntime()
    encoded = encoded_pair()
    original = encoded['history_tokens'].copy()
    result, positions, controls = evaluate_variant(runtime, encoded, [12,13], frozen_history=True)
    assert len(runtime.calls) == 2 and len(result['rows']) == 2
    assert [x['seed'] for x in runtime.calls] == [29404,29404]
    assert positions.shape == (2,65,3) and controls.shape == (2,16,3)
    for call in runtime.calls:
        np.testing.assert_array_equal(call['tokens'], np.broadcast_to(original[0,-1], (4,5,3)))
        assert not call['masks'][:,1:].any()  # Unknown remains unknown, not filled as valid.
        np.testing.assert_array_equal(call['ages'], encoded['history_times'])
    np.testing.assert_array_equal(encoded['history_tokens'], original)


def test_native_missing_contact_evidence_is_not_passed():
    complete = dict(completed=True,steps=8000,simulation_seconds=8.,no_retiming=True,
                    collision_detected=False,position_tracking_max_error_m=.02,goal_reached=False)
    summary = _native_summary([complete]*8)
    assert summary['complete'] and summary['collision_rate'] == 0
    assert summary['physical_validation_passed']
    assert summary['goal_reached_rate'] == 0
    partial = {key:value for key,value in complete.items() if key != 'steps'}
    assert not _native_summary([partial])['complete']
    no_contact_field = {key:value for key,value in complete.items() if key != 'collision_detected'}
    assert _native_summary([no_contact_field])['collision_rate'] is None
    assert _native_summary([no_contact_field])['contact_unknown_episodes'] == 1
    assert not _native_summary([complete])['complete']


def test_native_failure_is_retained_unknown_and_does_not_skip_remaining_fixed_samples(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('hnuter_dit.geometry_evaluation.sample_from_dataset', lambda data, i:
        dict(group_id=i//2, future_times=np.arange(65)*.125, sample_index=i))
    def execute(sample, positions, times, **kwargs):
        calls.append(sample['sample_index'])
        if sample['sample_index'] == 2:
            raise RuntimeError('nonfinite physics state')
        return dict(completed=True,steps=8000,simulation_seconds=8.,no_retiming=True,
                    collision_detected=False,position_tracking_max_error_m=.02,goal_reached=False)
    monkeypatch.setattr('hnuter_dit.geometry_evaluation.native_rollout', execute)
    rows = list(native_results({}, np.arange(8), np.zeros((8,65,3)), np.zeros((8,16,3)), tmp_path))
    assert calls == list(range(8)) and len(rows) == 8
    assert rows[2]['collision_detected'] is None and 'nonfinite physics' in rows[2]['error']
    assert (tmp_path/'sample_0002/evaluation_failure.json').is_file()
    summary = _native_summary(rows)
    assert not summary['physical_validation_passed'] and not summary['complete']
    assert summary['contact_unknown_episodes'] == 1 and summary['collision_rate'] == 0


def test_preprocessing_binding_is_order_independent_but_sensitive_to_geometry():
    assert config_hash({'resolution':.25,'range':8}) == config_hash({'range':8,'resolution':.25})
    assert config_hash({'resolution':.25}) != config_hash({'resolution':.5})


def test_previous_pilot_or_incomplete_new_collection_cannot_be_relabelled_as_fresh():
    protocol = dict(dataset={'collection_seed':107,'groups_requested':96})
    freeze = dict(seed=107,requested_groups=96,samples=192,independent_groups=96,
                  split_samples=dict(train=128,validation=32,test=32))
    validate_protocol_dataset(freeze, protocol)
    with pytest.raises(ValueError, match='fresh'):
        validate_protocol_dataset({**freeze,'seed':73}, protocol)
    with pytest.raises(ValueError, match='all 96'):
        validate_protocol_dataset({**freeze,'samples':190}, protocol)
