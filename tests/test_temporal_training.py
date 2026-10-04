import copy
import json

import numpy as np
import pytest
import torch

from hnuter_dit.temporal_model import FUTURE_TIMES, HISTORY_TIMES, decode_control_points
from hnuter_dit.temporal_training import train_arrays
from hnuter_dit.temporal_inference import TemporalRuntime, validate_inputs


def synthetic_data():
    tokens = np.zeros((6, 4, 32, 10), np.float32)
    tokens[:, :, 0, 0] = 1
    tokens[:, :, 0, 1] = np.arange(4)[None]*.2
    tokens[:, :, 0, 3:6] = .5
    tokens[..., 6] = 1
    mask = np.zeros((6, 4, 32), bool); mask[:, :, 0] = True
    condition = np.tile([2., 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], (6, 1)).astype(np.float32)
    controls = np.zeros((6, 16, 3))
    controls[1::2, 2:, 0] = np.linspace(0, 2, 14)
    return dict(history_tokens=tokens, history_mask=mask, history_times=HISTORY_TIMES,
        condition=condition, positions=decode_control_points(controls)['positions'].astype(np.float32),
        future_times=FUTURE_TIMES, split=np.array([0, 0, 1, 1, 2, 2]),
        group_id=np.repeat(np.arange(3), 2), teacher_future_secret=np.full((6,), 12345),
        _freeze=dict(dataset_file='synthetic.npz', dataset_sha256='synthetic-fixture', stationary_start_only=True))


def test_cpu_real_training_checkpoint_sampling_and_schema(tmp_path):
    old = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        data = synthetic_data()
        summary = train_arrays(data, tmp_path, steps=2, batch_size=2, device='cpu',
            dimension=32, depth=1, history_depth=1, validation_interval=1, sampling_steps=3)
        assert summary['weights_changed'] and not summary['test_rows_used']
        runtime = TemporalRuntime(tmp_path/'best.pt', device='cpu')
        initial = TemporalRuntime(tmp_path/'initial.pt', device='cpu')
        assert initial.describe()['training_state'] == 'untrained_initial'
        assert runtime.describe()['start_velocity_support']['training'] == 'stationary anchors only'
        args = (data['history_tokens'][:2], data['history_mask'][:2], data['history_times'], data['condition'][:2])
        first = runtime.sample(*args, seed=4, steps=3)
        second = runtime.sample(*args, seed=4, steps=3)
        np.testing.assert_array_equal(first['positions'], second['positions'])
        assert first['positions'].shape == (2, 65, 3) and first['control_points'].shape == (2, 16, 3)
        assert first['metadata']['model_evaluations'] == 3 and not first['metadata']['retimed']
        np.testing.assert_allclose(first['positions'][:, 0], 0, atol=1e-7)
        np.testing.assert_allclose(first['velocities'][:, 0], data['condition'][:2, 3:6], atol=1e-7)
        assert not np.array_equal(first['positions'], initial.sample(*args, seed=4, steps=3)['positions'])
        one = runtime.sample(data['history_tokens'][0], data['history_mask'][0], data['history_times'], data['condition'][0], seed=4, steps=3)
        assert one['positions'].shape == (65, 3)
        # Privileged evaluation fields are absent from the checkpoint input contract.
        checkpoint = torch.load(tmp_path/'best.pt', weights_only=False)
        assert checkpoint['input_allowlist'] == ['history_tokens', 'history_mask', 'history_times', 'condition']
        assert checkpoint['normalization']['train_row_count'] == 2
        assert json.loads((tmp_path/'representation_audit.json').read_text())['test_rows_used'] is False
        checkpoint['schema'] = 'old-static-weights'
        torch.save(checkpoint, tmp_path/'wrong.pt')
        with pytest.raises(ValueError, match='schema'):
            TemporalRuntime(tmp_path/'wrong.pt', device='cpu')
        with pytest.raises(FileExistsError):
            train_arrays(data, tmp_path, device='cpu', steps=1)
    finally:
        torch.set_num_threads(old)


def test_rotation_age_and_group_leakage_rejected(tmp_path):
    data = synthetic_data()
    args = [data[key] for key in ('history_tokens', 'history_mask', 'history_times', 'condition')]
    bad = copy.deepcopy(args); bad[3][:, 9:12] = bad[3][:, 6:9]
    with pytest.raises(ValueError, match='orthonormal'):
        validate_inputs(*bad)
    bad = copy.deepcopy(args); bad[0][:, :, 0, 6] = 2
    with pytest.raises(ValueError, match='unit'):
        validate_inputs(*bad)
    bad = copy.deepcopy(args); bad[2] = np.array([-.6, -.4, -.2, .2])
    with pytest.raises(ValueError, match='history_times'):
        validate_inputs(*bad)
    data['group_id'][2] = data['group_id'][0]
    with pytest.raises(ValueError, match='group'):
        train_arrays(data, tmp_path, device='cpu', steps=1)
