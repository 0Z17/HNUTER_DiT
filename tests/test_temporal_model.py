import numpy as np
import pytest
import torch

from hnuter_dit.temporal_model import (TemporalDiT, FUTURE_TIMES, HISTORY_TIMES,
    KNOT_SPACING, fit_control_points, decode_control_points, normalization_from_training)


@pytest.fixture(autouse=True)
def limited_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_fixed_time_spline_wait_and_exact_initial_velocity():
    controls = np.zeros((2, 16, 3))
    controls[1, 1, 0] = .2 * KNOT_SPACING / 3
    controls[1, 6:, 0] = np.linspace(0, 2, 10)
    decoded = decode_control_points(controls)
    np.testing.assert_allclose(decoded['positions'][:, 0], 0, atol=1e-12)
    np.testing.assert_allclose(decoded['velocities'][:, 0], [[0, 0, 0], [.2, 0, 0]], atol=1e-12)
    fitted = fit_control_points(decoded['positions'], decoded['velocities'][:, 0])
    np.testing.assert_allclose(fitted, controls, atol=1e-6)
    np.testing.assert_allclose(decode_control_points(fitted)['positions'][0], 0)
    # Independent central differences validate physical seconds, not array index.
    t, epsilon = np.linspace(.01, 7.99, 500), 1e-4
    plus = decode_control_points(controls, t+epsilon)
    minus = decode_control_points(controls, t-epsilon)
    analytic = decode_control_points(controls, t)
    np.testing.assert_allclose((plus['positions']-minus['positions'])/(2*epsilon), analytic['velocities'], atol=1e-6)
    np.testing.assert_allclose((plus['velocities']-minus['velocities'])/(2*epsilon), analytic['accelerations'], atol=1e-6)


def test_normalization_does_not_read_heldout_values():
    controls = np.zeros((3, 16, 3)); controls[1] = 1
    condition = np.zeros((3, 12))
    tokens = np.zeros((3, 4, 32, 10)); mask = np.ones((3, 4, 32), bool)
    before = normalization_from_training(controls, condition, tokens, mask, [0, 1])
    controls[2] = 1e9; condition[2] = -1e8; tokens[2] = 1e7
    after = normalization_from_training(controls, condition, tokens, mask, [0, 1])
    assert before == after
    assert before['fit_split'] == 'train'


def _inputs():
    torch.manual_seed(11)
    tokens = torch.randn(2, 4, 32, 10)
    mask = torch.ones(2, 4, 32, dtype=torch.bool)
    return torch.randn(2, 14, 3), torch.tensor([5, 18]), torch.randn(2, 12), tokens, mask, torch.tensor(HISTORY_TIMES, dtype=torch.float32)


def test_observation_set_permutation_and_history_are_real_inputs():
    model = TemporalDiT(32, 1, 4, 1).eval()
    # The production head is zero initialized; a nonzero head is required to
    # test the actual condition pathway before learning changes those weights.
    torch.nn.init.normal_(model.output.weight, std=.1)
    inputs = _inputs()
    result = model(*inputs)
    permuted = list(inputs)
    permutation = torch.randperm(32)
    permuted[3] = inputs[3][:, :, permutation]
    permuted[4] = inputs[4][:, :, permutation]
    torch.testing.assert_close(model(*permuted), result, atol=1e-6, rtol=1e-5)
    changed = list(inputs); changed[3] = inputs[3].clone(); changed[3][:, :-1, :, 0] += 10
    assert not torch.allclose(model(*changed), result, atol=1e-5)
    ages_changed = list(inputs); ages_changed[-1] = -inputs[-1]
    assert not torch.allclose(model(*ages_changed), result, atol=1e-5)
    model.history_mode = 'latest'
    torch.testing.assert_close(model(*changed), model(*inputs), atol=1e-6, rtol=1e-5)
    empty = list(inputs); empty[4] = torch.zeros_like(inputs[4])
    assert torch.isfinite(model(*empty)).all()


def test_torch_decode_has_exact_boundary_and_gradients():
    model = TemporalDiT(32, 1, 4, 1)
    free = torch.randn(2, 14, 3, requires_grad=True)
    condition = torch.randn(2, 12)
    full = model.full_controls(free, condition)
    decoded = model.decode(full)
    torch.testing.assert_close(decoded['velocities'][:, 0], condition[:, 3:6])
    decoded['positions'].square().mean().backward()
    assert free.grad is not None and torch.isfinite(free.grad).all() and free.grad.abs().sum() > 0
    np.testing.assert_allclose(decoded['positions'].detach(), decode_control_points(full.detach())['positions'], atol=1e-6)
