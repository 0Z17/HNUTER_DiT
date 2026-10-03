from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation, Slerp

from hnuter_dit.geometry import CANONICAL_FRAME_VERSION
from hnuter_dit.model import DiffusionSchedule
from hnuter_dit.online_diffusion import sample_online, _basis


class RecordingVelocityModel:
    def __init__(self, gain=1.):
        self.gain, self.calls = gain, []

    def __call__(self, latent, timestep, conditions, obstacles, mask):
        self.calls.append((timestep.clone(), conditions.clone(), obstacles.clone(), mask.clone()))
        # A deterministic conditional test double. Its changed weights and its
        # token input both affect output, so bypassing model calls fails tests.
        result = torch.zeros_like(latent)
        signal = (obstacles[:, :, 1] * mask).sum(dim=1)
        phase = torch.linspace(0, torch.pi, latent.shape[1], device=latent.device)
        result[..., 1] = self.gain * (1 + signal[:, None]) * torch.sin(phase)[None]
        return result


def runtime(gain=1.):
    return SimpleNamespace(canonical_frame=CANONICAL_FRAME_VERSION, sequence_length=128,
        path_mean=np.zeros(9, np.float32), path_std=np.ones(9, np.float32),
        device=torch.device('cpu'), schedule=DiffusionSchedule.cosine(100, 'cpu'),
        model=RecordingVelocityModel(gain), lock=threading.Lock(), checkpoint_sha256='test-double')


def condition():
    tokens = np.zeros((32, 10), np.float32)
    tokens[:, 6] = 1
    mask = np.zeros(32, bool)
    return tokens, mask, np.array([0., 0., 2., 1., 0., 0., 0.]), np.array([3., 0., 2., 1., 0., 0., 0.])


def test_real_model_calls_seed_endpoints_smoothness_and_bounded_correction():
    tokens, mask, start, goal = condition()
    first = runtime()
    paths, metadata = sample_online(first, tokens, mask, start, goal, seed=19, count=2)
    repeat, _ = sample_online(runtime(), tokens, mask, start, goal, seed=19, count=2)
    changed, _ = sample_online(runtime(), tokens, mask, start, goal, seed=20, count=2)
    assert len(first.model.calls) == metadata['model_evaluations'] == 6
    assert [int(call[0][0]) for call in first.model.calls] == [10, 8, 6, 4, 2, 0]
    np.testing.assert_array_equal(paths, repeat)
    assert not np.allclose(paths, changed)
    np.testing.assert_allclose(paths[:, 0], np.broadcast_to(start, (2, 7)))
    np.testing.assert_allclose(paths[:, -1], np.broadcast_to(goal, (2, 7)))
    assert metadata['final_max_prior_correction_m'] <= .250001
    assert np.max(np.linalg.norm(np.diff(paths[..., :3], n=2, axis=1), axis=-1)) < .01
    np.testing.assert_allclose(np.linalg.norm(paths[..., 3:], axis=-1), 1, atol=1e-6)
    assert metadata['collision_certified'] is False
    assert metadata['correction_envelope'] == 'sin_squared_uniform_phase'


def test_network_changes_affect_paths_and_current_tokens_reach_every_call():
    tokens, mask, start, goal = condition()
    tokens[0] = [2., 10., 2., .5, .5, .5, 1., 0., 0., 0.]
    mask[0] = True
    original = runtime(.2)
    positive, metadata = sample_online(original, tokens, mask, start, goal, seed=2, count=1)
    negative, _ = sample_online(runtime(-.2), tokens, mask, start, goal, seed=2, count=1)
    assert np.max(np.abs(positive[..., :3] - negative[..., :3])) > .01
    assert any(row['network_velocity_prediction_rms'] > 0 for row in metadata['updates'])
    for _, _, seen, seen_mask in original.model.calls:
        np.testing.assert_allclose(seen[0, 0, :3], [2., 10., 0.])
        assert seen_mask[0, 0] and int(seen_mask.sum()) == 1


def test_global_yaw_translation_equivariance_and_fixed_attitude():
    tokens, mask, start, goal = condition()
    start[3:] = Rotation.from_euler('xyz', [.1, .05, .1]).as_quat()[[3, 0, 1, 2]]
    goal[3:] = Rotation.from_euler('xyz', [.12, .02, .3]).as_quat()[[3, 0, 1, 2]]
    paths, _ = sample_online(runtime(), tokens, mask, start, goal, seed=6, count=1)
    rotation = Rotation.from_euler('z', 1.2)
    translation = np.array([-25., -35., .7])
    moved_start, moved_goal = start.copy(), goal.copy()
    for moved in (moved_start, moved_goal):
        moved[:3] = rotation.apply(moved[:3]) + translation
        moved[3:] = (rotation * Rotation.from_quat(moved[[4, 5, 6, 3]])).as_quat()[[3, 0, 1, 2]]
    moved_paths, _ = sample_online(runtime(), tokens, mask, moved_start, moved_goal, seed=6, count=1)
    np.testing.assert_allclose(moved_paths[0, :, :3], rotation.apply(paths[0, :, :3])+translation, atol=3e-6)
    expected = Slerp([0, 1], Rotation.from_quat(np.stack((start, goal))[:, [4, 5, 6, 3]]))(np.linspace(0, 1, 128))
    actual = Rotation.from_quat(paths[0, :, [4, 5, 6, 3]].T)
    np.testing.assert_allclose((expected.inv()*actual).magnitude(), 0, atol=2e-7)


def test_bad_attitude_or_tokens_fail_explicitly():
    tokens, mask, start, goal = condition()
    goal[3:] = [0., 1., 0., 0.]
    with pytest.raises(ValueError, match='attitude'):
        sample_online(runtime(), tokens, mask, start, goal, seed=1)
    goal[3:] = [1., 0., 0., 0.]
    mask[0] = True
    with pytest.raises(ValueError, match='extents'):
        sample_online(runtime(), tokens, mask, start, goal, seed=1)


def test_measured_obstacle_activates_actual_guidance_without_safety_claim():
    tokens, mask, start, goal = condition()
    tokens[0] = [1.5, 0., 2., .4, .4, .4, 1., 0., 0., 0.]
    mask[0] = True
    paths, metadata = sample_online(runtime(), tokens, mask, start, goal, seed=4, count=1)
    assert np.isfinite(paths).all()
    assert metadata['guidance_enabled'] is True
    assert max(row['guidance_displacement_max_m'] for row in metadata['updates']) > 1e-3
    assert metadata['final_max_prior_correction_m'] <= .250001
    # The local .25 m refinement cannot certify clearing this central obstacle.
    assert metadata['collision_certified'] is False


@pytest.mark.parametrize('delta', [[2., 0., 0.], [1., 1., .8], [0., 0., 2.]])
def test_longitudinal_progress_is_uniform_even_for_sloped_and_vertical_goals(delta):
    tokens, mask, start, goal = condition()
    goal[:3] = start[:3] + delta
    model = runtime()
    original = model.model

    class LongitudinalAndTransverseVelocity:
        def __call__(self, latent, timestep, conditions, obstacles, mask):
            result = original(latent, timestep, conditions, obstacles, mask)
            # Deliberately huge longitudinal predictions must not create
            # reversals, while transverse network predictions still survive.
            result[..., 0] = 500 * torch.sin(torch.linspace(0, 4*torch.pi, 128))[None]
            result[..., 2] += 100 * torch.cos(torch.linspace(0, 3*torch.pi, 128))[None]
            return result

    model.model = LongitudinalAndTransverseVelocity()
    paths, metadata = sample_online(model, tokens, mask, start, goal, seed=9, count=8)
    direction = np.asarray(delta) / np.linalg.norm(delta)
    progress = (paths[..., :3].astype(float) - start[:3]) @ direction
    expected = np.linspace(0, np.linalg.norm(delta), 128)
    np.testing.assert_allclose(progress, np.broadcast_to(expected, progress.shape), atol=1e-6)
    assert np.min(np.diff(progress, axis=1)) > 0
    assert metadata['minimum_chord_progress_step_m'] > 0
    assert metadata['model_evaluations'] == len(original.calls) == 6
    assert metadata['final_max_prior_correction_m'] <= .250001


def test_correction_basis_preserves_endpoint_tangents_and_convex_norm_bound():
    basis, _ = _basis(4097, 10)
    assert np.min(basis) >= 0
    assert np.max(basis.sum(axis=1)) <= 1 + 1e-7
    np.testing.assert_array_equal(basis[[0, -1]], 0.)
    # The analytic product is O(s^3) at each end. Halving the finite-difference
    # interval reduces its apparent first derivative by approximately four.
    h = 1 / 4096
    near_start = np.linalg.norm(basis[1] / h)
    wider_start = np.linalg.norm(basis[2] / (2*h))
    near_end = np.linalg.norm(basis[-2] / h)
    wider_end = np.linalg.norm(basis[-3] / (2*h))
    assert near_start < 2e-5 and near_end < 2e-5
    assert .24 < near_start / wider_start < .26
    assert .24 < near_end / wider_end < .26
