"""Actual motion diagnostics use velocity samples and real time spacing."""
import numpy as np
import pytest

from hnuter_dit.mujoco_execution import _SampledMotionDiagnostics


def state(time):
    value = np.zeros(13)
    value[6] = 1.
    value[3] = time**2
    value[10] = 2*time**2
    return value


def test_motion_metrics_handle_irregular_intervals_and_measure_actual_velocity():
    tracker = _SampledMotionDiagnostics(0., state(0.))
    for time in (.05, .13, .2, .4):
        metrics = tracker.update(time, state(time))
    assert metrics['max_linear_speed_mps'] == pytest.approx(.16)
    assert metrics['max_angular_speed_rps'] == pytest.approx(.32)
    assert metrics['max_linear_acceleration_mps2'] == pytest.approx(.6)
    assert metrics['max_angular_acceleration_rps2'] == pytest.approx(1.2)
    assert metrics['max_linear_jerk_mps3'] == pytest.approx(2.)
    assert metrics['max_angular_jerk_rps3'] == pytest.approx(4.)
    assert metrics['sample_count'] == 5
    assert 'not 1 ms' in metrics['sampling_scope']


def test_stationary_stop_before_physics_step_keeps_pending_values_unmeasured():
    tracker = _SampledMotionDiagnostics(0., state(0.))
    metrics = tracker.update(0., state(0.))
    assert metrics['sample_count'] == 1
    assert metrics['max_linear_acceleration_mps2'] is None
    assert metrics['max_linear_jerk_mps3'] is None
    with pytest.raises(ValueError):
        tracker.update(-.01, state(-.01))
