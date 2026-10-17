from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from hnuter_dit import contact_metrics as metrics
from hnuter_dit.contact_metrics import evaluate_contact_path
from hnuter_dit.contact_projection import ContactProjectionSpec, lift_contact_intrinsic, project_contact_pose9


def condition(kind="plane"):
    result = {
        "surface_geometry": {"type": kind, "origin": [0., 0., .5], "quaternion_wxyz": [1., 0., 0., 0.],
                             "radius_m": 1.5, "thickness_m": .2, "supports": []},
        "surface_chart": {"bounds": [[-2., 2.], [-2., 2.], [-math.pi, math.pi]], "forbidden_regions": []},
        "tool_extrinsics": {"position_body": [0., 0., -.8], "quaternion_wxyz": [0., 1., 0., 0.]},
        "contact_constraints": {"position_tolerance_m": 1e-7, "normal_tolerance_rad": 1e-6,
                                "collision_margin_m": .01, "validation_step_m": .02, "validation_step_rad": .02},
        "start_goal": {"start": [-1., 0., 0.], "goal": [1., 0., 0.]},
        "environment": {"obstacles": [], "bounds": {"min": [-4., -4., 0.], "max": [4., 4., 5.]}},
    }
    if kind == "cylinder":
        r = 1.5
        result["surface_geometry"].update(origin=[0., 0., r], quaternion_wxyz=[2**-.5, 2**-.5, 0., 0.])
        result["surface_chart"]["bounds"][0] = [r*(math.pi/2-.3), r*(math.pi/2+.3)]
        result["start_goal"] = {"start": [r*(math.pi/2-.25), -.8, 0.], "goal": [r*(math.pi/2+.25), .8, 0.]}
    return result


def path(c):
    spec = ContactProjectionSpec.from_condition(c)
    states = np.linspace(spec.start, spec.goal, 128)
    return states, lift_contact_intrinsic(states, spec).numpy()


@pytest.fixture
def backend():
    pytest.importorskip("coal")
    return metrics._backend()


@pytest.mark.parametrize("kind", ["plane", "cylinder"])
def test_exact_contact_success_requires_real_full_body_and_dense_checks(kind, backend):
    c = condition(kind)
    states, world = path(c)
    result = evaluate_contact_path(world.astype(np.float32), c, projected_intrinsic=states)
    assert result["evaluation_error"] is None
    assert result["candidate_success"] and result["numeric_candidate_success"]
    assert result["physical_collision_checked"] and result["physical_collision_free"]
    assert len(result["clearance_m"]) == 128
    assert result["urdf_collision_geometry_count"] == len(backend[0].geometries)
    assert result["urdf_collision_geometry_count"] > 1
    assert result["dense_checked"] and result["dense_validation_completed"]
    assert result["dense_physical_checked_samples"] == result["dense_planned_samples"]
    assert not result["dense_clearance_sampled"]
    assert result["minimum_clearance_dense_m"] is None
    assert result["minimum_clearance_scope"] == "all_128_waypoints_only"


def test_raw_body_poses_are_not_lifted_before_physical_collision_check(backend):
    c = condition()
    states, world = path(c)
    raw = world.copy()
    raw[1:-1, 2] += 3.
    # Guaranteed overlap with a real active URDF primitive at a RAW body pose.
    shape = backend[0].geometries[0]
    obstacle = raw[64, :3]+shape.local_pose.position
    c["environment"]["obstacles"] = [{"id": "raw_only_collision", "type": "box", "size": [.04]*3,
                                       "position": obstacle.tolist(), "quaternion_wxyz": [1., 0., 0., 0.]}]
    off = evaluate_contact_path(raw, c)
    assert off["physical_collision_checked"]
    assert not off["physical_collision_free"]
    assert off["minimum_clearance_m"] < 0
    assert not off["dense_checked"]
    assert "physical_collision" in off["failures"] and "contact" in off["failures"]
    projected = project_contact_pose9(raw, ContactProjectionSpec.from_condition(c))
    on = evaluate_contact_path(projected.world_pose9, c, projected_intrinsic=projected.intrinsic_states)
    assert on["physical_collision_free"] and on["candidate_success"]
    np.testing.assert_array_equal(raw[[0, -1]], world[[0, -1]])


def test_target_is_never_exempted_for_any_body_primitive(backend):
    c = condition()
    c["tool_extrinsics"]["position_body"] = [0., 0., -.01]
    states, world = path(c)
    result = evaluate_contact_path(world, c, projected_intrinsic=states)
    assert result["numeric_contact_gate"] and result["numeric_normal_gate"]
    assert not result["physical_collision_free"]
    assert not result["candidate_success"]
    assert result["environment_collision_object_count"] == 1
    assert result["tool_point_collision_free"]  # ideal point may touch the target


def test_workspace_includes_whole_body_aabb_not_just_origin(backend):
    c = condition()
    c["environment"]["bounds"]["max"][0] = 1.001
    states, world = path(c)
    assert np.all(world[:, 0] <= 1.001)
    result = evaluate_contact_path(world, c, projected_intrinsic=states)
    assert result["physical_collision_free"]
    assert not result["workspace_valid"]
    assert result["workspace_violation_sample_count"] > 0
    assert "workspace" in result["failures"]


def test_clearance_margin_is_subtracted_exactly_once(backend):
    c = condition()
    states, world = path(c)
    c["contact_constraints"]["collision_margin_m"] = 0.
    zero = evaluate_contact_path(world, c, projected_intrinsic=states)
    c["contact_constraints"]["collision_margin_m"] = .02
    margin = evaluate_contact_path(world, c, projected_intrinsic=states)
    np.testing.assert_allclose(margin["clearance_m"], np.asarray(zero["clearance_m"])-.02, rtol=0., atol=2e-14)
    assert margin["clearance_margin_already_deducted"]


def test_supplied_intrinsic_cannot_launder_off_surface_predictions(backend):
    c = condition()
    states, world = path(c)
    world[1:-1, 2] += .05
    result = evaluate_contact_path(world, c, projected_intrinsic=states)
    assert result["physical_collision_checked"]
    assert not result["projection_binding_valid"]
    assert "projection_binding" in result["failures"]
    assert not result["dense_checked"]


def test_off_slerp_and_on_intrinsic_edges_have_distinct_numeric_contact_errors(backend):
    c = condition("cylinder")
    c["contact_constraints"]["validation_step_m"] = .004
    states, world = path(c)
    off = evaluate_contact_path(world, c)
    on = evaluate_contact_path(world, c, projected_intrinsic=states)
    assert off["candidate_success"] and on["candidate_success"]
    assert off["numeric_contact_gate"] and on["numeric_contact_gate"]  # identical nodes
    assert off["dense_contact_error_max_m"] > 2e-6
    assert not off["numeric_candidate_success"]
    assert on["dense_contact_error_max_m"] < 1e-13 and on["numeric_candidate_success"]
    assert off["edge_decoder"] == "world_xyz_linear_shortest_so3_slerp"
    assert on["edge_decoder"] == "intrinsic_linear_unwrapped_lift"


def test_thin_forbidden_crossing_is_caught_analytically_between_nodes(backend):
    c = condition()
    c["surface_chart"]["forbidden_regions"] = [[0., .0001, -.001, .001]]
    states, world = path(c)
    result = evaluate_contact_path(world, c, projected_intrinsic=states)
    assert result["chart_valid"]
    assert result["forbidden_edge_violation_count"] == 1
    assert "forbidden_edge" in result["failures"]
    assert not result["dense_checked"]


def test_exact_pose_cache_reuses_only_identical_queries_and_model_is_memoized(backend, monkeypatch):
    c = condition()
    states, world = path(c)
    metrics._exact_pose_clearance.cache_clear()
    first = evaluate_contact_path(world, c, projected_intrinsic=states)
    second = evaluate_contact_path(world.copy(), c, projected_intrinsic=states.copy())
    assert first["waypoint_clearance_exact_cache_hits"] == 0
    assert second["waypoint_clearance_exact_cache_hits"] == 128
    changed = world.copy(); changed[64, 2] += 1e-8
    third = evaluate_contact_path(changed, c, projected_intrinsic=states)
    assert third["waypoint_clearance_exact_cache_hits"] == 127
    assert metrics._backend() is backend
    assert metrics._scene_checker.cache_info().maxsize == 8


def test_oracle_fields_are_ignored_and_pose_quality_remains_measurable(backend):
    c = condition()
    states, world = path(c)
    c.update(route_certificates=[float("nan")], generation_policy="unreadable", campaign_metadata="unreadable", experts="unreadable")
    result = evaluate_contact_path(world, c, projected_intrinsic=states)
    assert result["candidate_success"]
    assert result["path_length_m"] == pytest.approx(2.)
    assert result["maximum_curvature_inv_m"] == pytest.approx(0., abs=1e-10)
    assert result["minimum_local_chord_arc_ratio"] == pytest.approx(1.)
    assert result["cumulative_backtracking_m"] == 0.
    assert result["jerk_rms_m_per_index3"] < 1e-15


@pytest.mark.parametrize("bad", ["nan", "degenerate_rotation", "wrong_length", "bad_condition"])
def test_invalid_candidate_fails_without_crashing_the_evaluation(bad):
    c = condition()
    states, world = path(c)
    if bad == "nan":
        world[5, 0] = float("nan")
    elif bad == "degenerate_rotation":
        world[5, 3:9] = 0
    elif bad == "wrong_length":
        world = world[:-1]
    else:
        c = copy.deepcopy(c); c["surface_geometry"]["quaternion_wxyz"] = [0., 0., 0., 0.]
    result = evaluate_contact_path(world, c, projected_intrinsic=states)
    assert not result["candidate_success"]
    assert result["evaluation_error"] is not None
    assert "invalid_input_or_backend_error" in result["failures"]
