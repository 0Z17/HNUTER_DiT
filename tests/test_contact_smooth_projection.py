from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from hnuter_dit.contact_projection import (
    INTRINSIC_INTERPOLATION, ContactProjectionSpec, lift_contact_intrinsic,
    project_contact_pose9,
)
from hnuter_dit.contact_smooth_projection import (
    BENDING_WEIGHT, CONTROL_POINT_COUNT, OUTPUT_NODE_COUNT, SOLVER_TOLERANCE,
    contact_cubic_basis, evaluate_contact_cubic, project_smooth_contact_pose9,
    smooth_contact_decoder_parameters,
)
from hnuter_dit.geometry import canonicalize_pose9_batch


torch.set_num_threads(1)


def condition(kind="plane"):
    c = {
        "surface_geometry": {"type": kind, "origin": [1.2, -.4, .7],
                             "quaternion_wxyz": [1., 0., 0., 0.], "radius_m": 1.5},
        "surface_chart": {"bounds": [[-2., 2.], [-2., 2.], [-math.pi, math.pi]],
                          "forbidden_regions": []},
        "tool_extrinsics": {"position_body": [.15, -.07, -.8],
                            "quaternion_wxyz": [0., 1., 0., 0.]},
        "contact_constraints": {"position_tolerance_m": 1e-7, "normal_tolerance_rad": 1e-6,
                                "collision_margin_m": .01, "validation_step_m": .02,
                                "validation_step_rad": .02,
                                "attitude_limits_deg": {"roll": [-40., 40.], "pitch": [-70., 70.]}},
        "start_goal": {"start": [-1., -.3, .2], "goal": [1., .4, .25]},
        "environment": {"obstacles": []},
    }
    if kind == "cylinder":
        c["surface_geometry"].update(quaternion_wxyz=[2**-.5, 2**-.5, 0., 0.],
                                     origin=[1.2, -.4, 1.5])
        c["surface_chart"]["bounds"][0] = [1.5*(math.pi/2-.3), 1.5*(math.pi/2+.3)]
        c["start_goal"] = {"start": [1.5*(math.pi/2-.2), -.8, .2],
                           "goal": [1.5*(math.pi/2+.2), .8, .25]}
    return c


def states_for(spec, n=128):
    t = torch.linspace(0., 1., n, dtype=torch.float64)
    return torch.tensor(spec.start)+t[:, None]*torch.tensor(spec.goal-spec.start)


def jittered_path(spec, n=128):
    states = states_for(spec, n)
    t = torch.linspace(0., 1., n, dtype=torch.float64)
    states[:, 1] += .12*torch.sin(46*math.pi*t)
    states[:, 2] += .06*torch.sin(38*math.pi*t)
    states[[0, -1]] = torch.tensor(np.stack((spec.start, spec.goal)))
    path = lift_contact_intrinsic(states, spec)
    path[1:-1, :3] += torch.tensor([.03, -.02, .015])
    path[1:-1, 3:9] += torch.tensor([.01, -.015, .02, .015, .01, -.02])
    return path


def test_fixed_basis_is_nonnegative_partition_of_unity_and_reproduces_affine_curves():
    params = smooth_contact_decoder_parameters()
    knots = np.array(params["knots"])
    t = torch.cat((torch.linspace(0., 1., 1001, dtype=torch.float64),
                   torch.tensor(np.unique(knots))))
    basis = contact_cubic_basis(t)
    assert basis.shape == (len(t), 16)
    assert float(basis.min()) >= 0.
    torch.testing.assert_close(basis.sum(-1), torch.ones_like(t), rtol=0., atol=5e-16)
    torch.testing.assert_close(contact_cubic_basis(torch.tensor([0., 1.], dtype=torch.float64)),
                               torch.eye(16, dtype=torch.float64)[[0, -1]], rtol=0., atol=0.)
    greville = np.array([knots[index+1:index+4].mean() for index in range(16)])
    slope = torch.tensor([2., -.8, .6], dtype=torch.float64)
    offset = torch.tensor([-.5, 1., 2.], dtype=torch.float64)
    controls = torch.tensor(greville)[:, None]*slope+offset
    expected = t[:, None]*slope+offset
    torch.testing.assert_close(evaluate_contact_cubic(controls, t), expected, rtol=0., atol=2e-15)
    torch.testing.assert_close(evaluate_contact_cubic(controls, t, derivative=1),
                               torch.tensor([2., -.8, .6], dtype=torch.float64).expand(len(t), 3),
                               rtol=0., atol=2e-14)
    assert float(evaluate_contact_cubic(controls, t, derivative=2).abs().max()) < 2e-12
    assert params["edge_interpolation"] == INTRINSIC_INTERPOLATION
    assert params["bending_weight"] == BENDING_WEIGHT
    params["control_points"] = 999
    assert smooth_contact_decoder_parameters()["control_points"] == CONTROL_POINT_COUNT


@pytest.mark.parametrize("kind", ["plane", "cylinder"])
def test_smoothing_uses_predictions_preserves_endpoints_and_returns_valid_contact_nodes(kind):
    spec = ContactProjectionSpec.from_condition(condition(kind))
    path = jittered_path(spec)
    original = path.clone()
    analytic = project_contact_pose9(path, spec)
    result = project_smooth_contact_pose9(path, spec)
    assert result.projected_pose9.shape == (OUTPUT_NODE_COUNT, 9)
    torch.testing.assert_close(path, original, rtol=0., atol=0.)
    torch.testing.assert_close(result.unprojected_pose9, original, rtol=0., atol=0.)
    torch.testing.assert_close(result.projected_pose9[[0, -1]], path[[0, -1]], rtol=0., atol=0.)
    torch.testing.assert_close(result.source_intrinsic_states, analytic.intrinsic_states, rtol=0., atol=0.)
    torch.testing.assert_close(result.intrinsic_states,
                               evaluate_contact_cubic(result.control_points, torch.linspace(0., 1., 128, dtype=torch.float64)),
                               rtol=0., atol=1e-14)
    assert result.geometry_valid
    assert float(result.after["contact_error_max_m"]) < 2e-14
    assert float(result.after["normal_error_max_rad"]) < 2e-14
    assert result.after["solver_converged"]
    assert float(result.after["solver_projected_gradient_max"]) <= SOLVER_TOLERANCE
    assert float(result.after["smoothing_intrinsic_uv_correction_max_m"]) > .05
    assert torch.diff(result.intrinsic_states[:, 1], n=2).square().sum() < torch.diff(analytic.intrinsic_states[:, 1], n=2).square().sum()/100
    assert not result.full_body_collision_checked
    assert not result.curvature_gate_guaranteed
    assert not result.continuous_spline_collision_checked


@pytest.mark.parametrize("n", [2, 17, 128])
def test_affine_intrinsic_path_is_exact_fixed_point_even_for_short_sources(n):
    spec = ContactProjectionSpec.from_condition(condition())
    states = states_for(spec, n)
    result = project_smooth_contact_pose9(lift_contact_intrinsic(states, spec), spec)
    torch.testing.assert_close(result.intrinsic_states, states_for(spec), rtol=0., atol=3e-12)
    assert result.source_intrinsic_states.shape == (n, 3)
    assert result.after["solver_sweeps"] == 0
    assert float(result.after["bending_energy_normalized"]) < 1e-20


def test_active_bounds_are_solved_and_whole_cubic_and_final_linear_edges_remain_in_chart():
    c = condition()
    c["surface_chart"]["bounds"] = [[-1.1, 1.1], [-.6, .7], [-.3, .35]]
    spec = ContactProjectionSpec.from_condition(c)
    states = states_for(spec)
    # Blocks near alternating opposite faces require genuinely active bounds.
    blocks = (torch.arange(128)//20) % 2
    for axis in range(3):
        states[1:-1, axis] = torch.tensor(spec.bounds[axis])[blocks[1:-1]]
    result = project_smooth_contact_pose9(lift_contact_intrinsic(states, spec), spec)
    bounds = torch.tensor(spec.bounds)
    assert result.after["solver_sweeps"] > 0
    assert result.after["solver_converged"]
    assert float(result.after["solver_projected_gradient_max"]) <= SOLVER_TOLERANCE
    assert int(result.after["control_chart_violation_count"]) == 0
    assert bool(((result.control_points == bounds[:, 0]) | (result.control_points == bounds[:, 1])).any())
    dense = evaluate_contact_cubic(result.control_points, torch.linspace(0., 1., 4097, dtype=torch.float64))
    assert bool((dense >= bounds[:, 0]-1e-14).all())
    assert bool((dense <= bounds[:, 1]+1e-14).all())
    linear_midpoints = (result.intrinsic_states[1:]+result.intrinsic_states[:-1])/2
    assert bool((linear_midpoints >= bounds[:, 0]).all())
    assert bool((linear_midpoints <= bounds[:, 1]).all())
    # Violent twist reversal can remain invalid after a bounded fit; no repair claim.
    assert int(result.after["chart_violation_count"]) == 0


def test_unwrapped_cylinder_chart_and_twist_do_not_take_periodic_shortcuts():
    c = condition("cylinder")
    c["surface_chart"]["bounds"][0] = [1.5*2.8, 1.5*3.5]
    c["surface_chart"]["bounds"][2] = [2.7, 3.5]
    c["start_goal"] = {"start": [1.5*2.9, -.5, 2.9], "goal": [1.5*3.4, .5, 3.4]}
    spec = ContactProjectionSpec.from_condition(c)
    result = project_smooth_contact_pose9(lift_contact_intrinsic(states_for(spec), spec), spec)
    torch.testing.assert_close(result.intrinsic_states, states_for(spec), rtol=0., atol=3e-13)
    assert float(torch.diff(result.intrinsic_states[:, 0]).min()) > 0.
    assert float(result.intrinsic_states[:, 2].min()) >= 2.9-1e-13
    assert float(result.intrinsic_states[:, 2].max()) <= 3.4+1e-13
    c = condition()
    c["start_goal"]["start"][2], c["start_goal"]["goal"][2] = 3.1, -3.1
    spec = ContactProjectionSpec.from_condition(c)
    result = project_smooth_contact_pose9(lift_contact_intrinsic(states_for(spec), spec), spec)
    torch.testing.assert_close(result.intrinsic_states[:, 2], states_for(spec)[:, 2], rtol=0., atol=3e-12)
    assert float(result.after["psi_total_variation_rad"]) == pytest.approx(6.2, abs=1e-12)


def test_cubic_second_derivative_is_continuous_at_simple_interior_knots():
    spec = ContactProjectionSpec.from_condition(condition())
    result = project_smooth_contact_pose9(jittered_path(spec), spec)
    knot = torch.tensor(np.unique(result.decoder_parameters["knots"])[1:-1])
    for derivative, tolerance in ((0, 1e-7), (1, 1e-6), (2, 1e-4)):
        left = evaluate_contact_cubic(result.control_points, knot-1e-9, derivative=derivative)
        right = evaluate_contact_cubic(result.control_points, knot+1e-9, derivative=derivative)
        torch.testing.assert_close(left, right, rtol=0., atol=tolerance)


def test_final_cylinder_edges_use_linear_intrinsic_lift_and_are_contact_preserving():
    spec = ContactProjectionSpec.from_condition(condition("cylinder"))
    result = project_smooth_contact_pose9(jittered_path(spec), spec)
    fraction = torch.tensor([.13, .5, .91], dtype=torch.float64)
    states = result.intrinsic_states[:-1, None]+fraction[None, :, None]*torch.diff(result.intrinsic_states, dim=0)[:, None]
    poses = lift_contact_intrinsic(states, spec)
    body = torch.stack((poses[..., 3:6], poses[..., 6:9],
                        torch.cross(poses[..., 3:6], poses[..., 6:9], dim=-1)), -1)
    tip = poses[..., :3]+(body @ torch.tensor(spec.tool_offset)[:, None]).squeeze(-1)
    local = (tip-torch.tensor(spec.origin)) @ torch.tensor(spec.frame)
    torch.testing.assert_close(torch.linalg.vector_norm(local[..., :2], dim=-1),
                               torch.full(states.shape[:-1], spec.radius, dtype=torch.float64), rtol=0., atol=2e-14)
    assert result.decoder_parameters["spline_role"] == "fit_and_resample_only"
    # The physical midpoint differs from the fitted mathematical spline midpoint.
    cubic = evaluate_contact_cubic(result.control_points,
                                   (torch.arange(127, dtype=torch.float64)+.5)/127)
    assert float((cubic-states[:, 1]).abs().max()) > 1e-8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_normalized_canonical_output_matches_world_and_preserves_exact_endpoints(dtype):
    spec = ContactProjectionSpec.from_condition(condition("cylinder"))
    world = torch.stack((jittered_path(spec), jittered_path(spec))).to(dtype)
    conditions = torch.cat((world[:, 0], world[:, -1]), -1)
    mean = torch.tensor([.1, -.2, .4, .03, -.07, .09, .2, .1, -.1], dtype=dtype)
    std = torch.tensor([1.2, 1.7, .7, .5, .4, .6, .7, .3, .8], dtype=dtype)
    normalized = (canonicalize_pose9_batch(world, conditions)-mean)/std
    result = project_smooth_contact_pose9(normalized, spec, coordinate_frame="canonical",
                                        world_condition_pose9=conditions,
                                        normalization_mean=mean, normalization_std=std)
    direct = project_smooth_contact_pose9(world, spec)
    torch.testing.assert_close(result.projected_pose9[:, [0, -1]], normalized[:, [0, -1]], rtol=0., atol=0.)
    torch.testing.assert_close(result.world_pose9, direct.world_pose9,
                               rtol=0., atol=8e-7 if dtype == torch.float32 else 1e-13)
    assert result.geometry_valid.shape == (2,)
    assert bool(result.geometry_valid.all())
    assert result.projected_pose9.dtype == dtype


def test_degenerate_input_rotation_and_cylinder_axis_are_finite_and_audited():
    spec = ContactProjectionSpec.from_condition(condition("cylinder"))
    path = jittered_path(spec)
    path[40, 3:9] = 0.
    path[40, :3] = torch.tensor(spec.origin-spec.tool_offset)
    result = project_smooth_contact_pose9(path, spec)
    assert bool(torch.isfinite(result.projected_pose9).all())
    assert int(result.before["degenerate_rotation_count"]) == 1
    assert int(result.after["input_cylinder_axis_degeneracy_count"]) == 1
    assert float(result.after["contact_error_max_m"]) < 2e-14


def test_coincident_task_endpoints_give_a_finite_constant_fit():
    c = condition()
    c["start_goal"]["goal"] = list(c["start_goal"]["start"])
    spec = ContactProjectionSpec.from_condition(c)
    path = lift_contact_intrinsic(states_for(spec, 2), spec)
    result = project_smooth_contact_pose9(path, spec)
    assert result.geometry_valid
    torch.testing.assert_close(result.intrinsic_states,
                               torch.tensor(spec.start).expand(128, 3), rtol=0., atol=3e-12)
    torch.testing.assert_close(result.projected_pose9[[0, -1]], path, rtol=0., atol=0.)
    assert bool(torch.isfinite(result.control_points).all())
    assert result.after["solver_converged"]


def test_solver_nonconvergence_is_explicit_and_prevents_a_success_claim(monkeypatch):
    import hnuter_dit.contact_smooth_projection as decoder
    c = condition()
    c["surface_chart"]["bounds"] = [[-1.1, 1.1], [-.6, .7], [-.3, .35]]
    spec = ContactProjectionSpec.from_condition(c)
    states = states_for(spec)
    blocks = (torch.arange(128)//20) % 2
    for axis in range(3):
        states[1:-1, axis] = torch.tensor(spec.bounds[axis])[blocks[1:-1]]
    path = lift_contact_intrinsic(states, spec)
    monkeypatch.setattr(decoder, "SOLVER_MAX_SWEEPS", 0)
    result = project_smooth_contact_pose9(path, spec)
    assert not result.after["solver_converged"]
    assert float(result.after["solver_projected_gradient_max"]) > SOLVER_TOLERANCE
    assert not result.geometry_valid
    assert bool(torch.isfinite(result.projected_pose9).all())
    torch.testing.assert_close(result.projected_pose9[[0, -1]], path[[0, -1]], rtol=0., atol=0.)


def test_forbidden_and_support_crossing_remain_invalid_without_obstacle_planning():
    c = condition()
    c["start_goal"] = {"start": [-1., 0., 0.], "goal": [1., 0., 0.]}
    c["surface_chart"]["forbidden_regions"] = [[-.005, .005, -.005, .005]]
    c["surface_geometry"]["supports"] = [{"type": "box", "id": "support", "size": [.08]*3,
                                          "position": c["surface_geometry"]["origin"],
                                          "quaternion_wxyz": [1., 0., 0., 0.]}]
    spec = ContactProjectionSpec.from_condition(c)
    result = project_smooth_contact_pose9(lift_contact_intrinsic(states_for(spec), spec), spec)
    assert not result.geometry_valid
    assert int(result.after["forbidden_edge_violation_count"]) > 0
    assert int(result.after["dense_tool_box_violation_count"]) > 0
    changed = copy.deepcopy(c)
    changed["surface_chart"]["forbidden_regions"] = []
    changed["surface_geometry"]["supports"] = []
    clear_spec = ContactProjectionSpec.from_condition(changed)
    clear = project_smooth_contact_pose9(lift_contact_intrinsic(states_for(clear_spec), clear_spec), clear_spec)
    torch.testing.assert_close(result.intrinsic_states, clear.intrinsic_states, rtol=0., atol=0.)
    torch.testing.assert_close(result.control_points, clear.control_points, rtol=0., atol=0.)


def test_expert_and_oracle_metadata_do_not_affect_decoder():
    c = condition()
    spec = ContactProjectionSpec.from_condition(c)
    path = jittered_path(spec)
    first = project_smooth_contact_pose9(path, spec)
    c.update(experts="invalid", route_certificates=[float("nan")],
             generation_policy={"routes": "invalid"}, oracle="invalid")
    second = project_smooth_contact_pose9(path, ContactProjectionSpec.from_condition(c))
    torch.testing.assert_close(first.projected_pose9, second.projected_pose9, rtol=0., atol=0.)
    torch.testing.assert_close(first.control_points, second.control_points, rtol=0., atol=0.)


def test_invalid_inputs_use_frozen_endpoint_contract_and_helpers_reject_bad_domains():
    spec = ContactProjectionSpec.from_condition(condition())
    path = jittered_path(spec)
    changed = path.clone(); changed[0, 0] += .001
    with pytest.raises(ValueError, match="endpoints"):
        project_smooth_contact_pose9(changed, spec)
    changed = path.clone(); changed[5, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        project_smooth_contact_pose9(changed, spec)
    with pytest.raises(ValueError, match="world_condition"):
        project_smooth_contact_pose9(path, spec, coordinate_frame="canonical")
    with pytest.raises(ValueError, match="normalization"):
        project_smooth_contact_pose9(path, spec, normalization_mean=np.zeros(9), normalization_std=np.zeros(9))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        contact_cubic_basis(torch.tensor([-.1, 1.]))
    with pytest.raises(ValueError, match="finite"):
        evaluate_contact_cubic(torch.zeros(15, 3), torch.tensor([.5]))
