import copy
import numpy as np
import pytest
import torch

from hnuter_dit.geometry_observations import reencode_dataset, preprocessing_configuration
from hnuter_dit.temporal_training import train_arrays
from hnuter_dit.temporal_inference import TemporalRuntime, validate_inputs
from hnuter_dit.temporal_model import TemporalDiT, HISTORY_TIMES
from test_temporal_training import synthetic_data


def raw_data():
    data = synthetic_data()
    data.update(history_ray_ranges=np.tile([1.,1.5,.2,np.nan],(6,4,1)),
        ray_directions=np.tile([[1.,0,0],[0,1.,0],[0,0,-1.],[0,-1.,0]],(4,1,1)),
        history_sensor_poses=np.tile(np.eye(4),(6,4,1,1)),
        origin_world=np.zeros((6,3)), world_to_canonical=np.tile(np.eye(3),(6,1,1)))
    data['history_ray_ranges'][:,0,0] += .35
    return data


@pytest.mark.parametrize('representation', ['point','voxel'])
def test_geometry_real_training_and_checkpoint_contract(tmp_path, representation):
    old_threads = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        data = raw_data()
        summary = train_arrays(data, tmp_path, input_representation=representation,
            steps=2,batch_size=2,device='cpu',dimension=32,depth=1,history_depth=1,
            validation_interval=1,sampling_steps=2)
        assert summary['weights_changed']
        runtime = TemporalRuntime(tmp_path/'best.pt', 'cpu')
        description = runtime.describe()
        assert description['input_representation'] == representation
        assert description['geometry_config'] == preprocessing_configuration(representation)
        assert description['encoder_parameter_count'] > 0
        encoded = reencode_dataset(data, representation)
        args = [encoded[k][0] if k != 'history_times' else encoded[k] for k in ('history_tokens','history_mask','history_times','condition')]
        result = runtime.sample(*args,seed=7,steps=2)
        assert result['positions'].shape == (65,3) and np.isfinite(result['positions']).all()
        assert result['metadata']['input_representation'] == representation
        np.testing.assert_allclose(result['positions'][0], 0, atol=1e-7)
        checkpoint = torch.load(tmp_path/'best.pt', weights_only=False)
        checkpoint['geometry_config']['roi_radius_m'] = 7.
        torch.save(checkpoint, tmp_path/'wrong.pt')
        with pytest.raises(ValueError, match='preprocessing'):
            TemporalRuntime(tmp_path/'wrong.pt', 'cpu')
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('representation', ['point','voxel'])
def test_history_conditioning_and_padding_are_used(representation):
    old_threads = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        data = reencode_dataset(raw_data(), representation)
        model = TemporalDiT(dimension=32,depth=1,history_depth=1,input_representation=representation)
        tokens = torch.from_numpy(data['history_tokens'][:1])
        mask = torch.from_numpy(data['history_mask'][:1])
        ages = torch.from_numpy(HISTORY_TIMES.astype(np.float32))
        first, valid = model.encode_history(tokens, mask, ages)
        reverse, _ = model.encode_history(tokens.flip(1),mask.flip(1),ages)
        assert not torch.allclose(first, reverse)
        model.history_mode = 'latest'
        _, valid_latest = model.encode_history(tokens,mask,ages)
        assert not valid_latest[:,1:193].any()
        assert valid[:,0].all() and valid_latest[:,0].all()
        empty, empty_valid = model.encode_history(tokens, torch.zeros_like(mask), ages)
        assert torch.isfinite(empty).all() and empty_valid.sum() == 1
    finally:
        torch.set_num_threads(old_threads)


def test_voxel_channel_consistency_rejected():
    data = reencode_dataset(raw_data(), 'voxel')
    args = [data[k][:1] if k != 'history_times' else data[k] for k in ('history_tokens','history_mask','history_times','condition')]
    args[0] = args[0].copy(); args[0][0,0,0,0,0,0] = 1
    with pytest.raises(ValueError, match='disjoint'):
        validate_inputs(*args, input_representation='voxel')


def test_cached_preprocessing_configuration_must_match(tmp_path):
    data = reencode_dataset(raw_data(), 'point')
    data['_preprocessing']['roi_radius_m'] = 7.
    with pytest.raises(ValueError, match='cached geometry preprocessing'):
        train_arrays(data,tmp_path,input_representation='point',steps=1,device='cpu')
