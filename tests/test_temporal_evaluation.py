import numpy as np
import pytest

from hnuter_dit.temporal_evaluation import (timed_proxy_clearance, expansion_decision, rotation6_matrix,
                                          validate_runtime_role, industrial_evidence_valid)


def test_timed_crossing_differs_from_snapshot():
    t = np.linspace(0, 8, 65)
    p = np.column_stack((t * .25, np.zeros_like(t), np.zeros_like(t)))
    boxes = np.zeros((65, 1, 10))
    boxes[:, 0, 0] = 1.
    boxes[:, 0, 1] = 2. - .5 * t
    boxes[:, 0, 3:6] = [.5, .5, 2.]
    boxes[:, 0, 6] = 1
    mask = np.ones((65, 1), bool)
    assert timed_proxy_clearance(p, t, boxes, mask, np.eye(3)) < 0
    frozen = np.broadcast_to(boxes[:1], boxes.shape).copy()
    assert timed_proxy_clearance(p, t, frozen, mask, np.eye(3)) > 1


def test_missing_measurements_prevent_expansion():
    r = {"groups": 16, "group_balanced_feasible_safe_progress_rate": 1., "sampling_seconds_p95": .1}
    comparisons = {"temporal": r, **{n: {**r, "group_balanced_feasible_safe_progress_rate": .5}
                                  for n in ("latest_only", "frozen_history", "untrained")}}
    result = expansion_decision(comparisons)
    assert not result["expand"]
    assert not result["checks"]["native_contact_validation"]
    native = {"episodes": 8, "collision_rate": 0, "complete": True, "maximum_tracking_error_m": .02}
    assert not expansion_decision(comparisons, native=native, industrial={"passed": True})["expand"]
    run = {"passed": True, "completed": True, "mode": "temporal", "goal_reached": True,
           "dynamic_obstacle": True, "steps": 8000,
           "collision": {"complete": True, "collision_detected": False, "samples": 8000},
           "planner": {"failed": False, "temporal_checkpoint_used": True, "accepted_plans": 1,
                       "guard_interventions": 0, "guard_rejections": 0,
                       "checkpoint": {"checkpoint_sha256": "abc"}, "time_in_state_seconds": {"model_reference": 4.}}}
    industrial = {"runs": [{**run, "dynamic_profile": profile} for profile in ('constant_incoming', 'constant_outgoing')]}
    assert expansion_decision(comparisons, native=native, industrial=industrial, checkpoint_sha256='abc')["expand"]
    assert not industrial_evidence_valid(industrial, 'wrong-weights')
    assert not expansion_decision(comparisons, native={**native, "maximum_tracking_error_m": 1.},
                                  industrial=industrial, checkpoint_sha256='abc')["expand"]


def test_checkpoint_roles_cannot_be_substituted_or_use_a_different_dataset():
    trained = {"history_mode": "temporal", "training_state": "trained", "training_step": 500,
               "source_dataset": {"dataset_sha256": "data"}, "architecture": {"dimension": 128}}
    validate_runtime_role(trained, 'temporal', 'data')
    for role in ('latest_only', 'untrained'):
        with pytest.raises(ValueError):
            validate_runtime_role(trained, role, 'data')
    with pytest.raises(ValueError):
        validate_runtime_role(trained, 'temporal', 'another-data')


def test_rotation6_and_unknown_truth_rejected():
    np.testing.assert_allclose(rotation6_matrix([1, 0, 0, 0, 1, 0]), np.eye(3))
    with pytest.raises(ValueError):
        rotation6_matrix([1, 0, 0, 1, 0, 0])
    with pytest.raises(ValueError, match="missing"):
        timed_proxy_clearance(np.zeros((3, 3)), np.arange(3.), np.empty((3, 0, 10)),
                              np.empty((3, 0), bool), np.eye(3))
