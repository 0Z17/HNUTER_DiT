from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from hnuter_dit.contact_projection import (
    ContactProjectionSpec, lift_contact_intrinsic, project_contact_pose9,
)
from hnuter_dit.geometry import CanonicalFrame


def condition(kind="plane"):
    result = {
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
        r = 1.5
        result["surface_geometry"].update(quaternion_wxyz=[2**-.5, 2**-.5, 0., 0.], origin=[1.2, -.4, 1.5])
        result["surface_chart"]["bounds"] = [[r*(math.pi/2-.3), r*(math.pi/2+.3)], [-2., 2.], [-math.pi, math.pi]]
        result["start_goal"] = {"start": [r*(math.pi/2-.2), -.8, .2], "goal": [r*(math.pi/2+.2), .8, .25]}
    return result


def path_for(spec, n=17):
    states = torch.linspace(0., 1., n, dtype=torch.float64)[:, None]
    states = torch.as_tensor(spec.start) + states*torch.as_tensor(spec.goal-spec.start)
    return states, lift_contact_intrinsic(states, spec)


def noisy_path(spec):
    states, path = path_for(spec)
    path = path.clone()
    path[1:-1, :3] += torch.tensor([.12, -.08, .2])
    # Perturb both 6D columns instead of assuming the tool axis is body Z.
    path[1:-1, 3:9] += torch.tensor([.1, -.06, .09, -.04, .12, -.07])
    return states, path


@pytest.mark.parametrize("kind", ["plane", "cylinder"])
def test_contact_normal_projection_is_exact_and_preserves_same_sample_endpoints(kind):
    spec = ContactProjectionSpec.from_condition(condition(kind))
    _, path = noisy_path(spec)
    original = path.clone()
    result = project_contact_pose9(path, spec)
    torch.testing.assert_close(path, original, rtol=0., atol=0.)
    torch.testing.assert_close(result.unprojected_pose9, original, rtol=0., atol=0.)
    torch.testing.assert_close(result.projected_pose9[[0, -1]], original[[0, -1]], rtol=0., atol=0.)
    assert float(result.before["contact_error_max_m"]) > .05
    assert float(result.before["normal_error_max_rad"]) > .01
    assert float(result.after["contact_error_max_m"]) < 2e-14
    assert float(result.after["normal_error_max_rad"]) < 2e-14
    assert result.geometry_valid
    assert result.full_body_collision_checked is False
    # Continuous cylinder lift stays on the surface; its XYZ chord does not.
    interpolated = (result.intrinsic_states[:-1] + result.intrinsic_states[1:])/2
    dense = lift_contact_intrinsic(interpolated, spec)
    assert torch.isfinite(dense).all()
    if kind == "cylinder":
        body = torch.stack((dense[:, 3:6], dense[:, 6:9], torch.cross(dense[:, 3:6], dense[:, 6:9], dim=-1)), -1)
        tip = dense[:, :3] + (body @ torch.tensor(spec.tool_offset)[:, None]).squeeze(-1)
        local = (tip-torch.tensor(spec.origin)) @ torch.tensor(spec.frame)
        torch.testing.assert_close(torch.linalg.vector_norm(local[:, :2], dim=-1), torch.full((len(tip),), spec.radius, dtype=torch.float64), rtol=0, atol=1e-14)


def test_nontrivial_tool_rotation_is_applied_in_correct_direction():
    c = condition()
    # R_body_tool = Rx(pi) Rz(.37), with a nonzero lateral tool offset.
    c["tool_extrinsics"]["quaternion_wxyz"] = [0., math.cos(.37/2), -math.sin(.37/2), 0.]
    angle = .2
    c["surface_geometry"]["quaternion_wxyz"] = [math.cos(angle/2), 0., math.sin(angle/2), 0.]
    spec = ContactProjectionSpec.from_condition(c)
    _, path = noisy_path(spec)
    result = project_contact_pose9(path, spec)
    s, p = result.intrinsic_states.numpy(), result.world_pose9.numpy()
    for state, pose in zip(s, p):
        rotation = np.column_stack((pose[3:6], pose[6:9], np.cross(pose[3:6], pose[6:9])))
        tip = pose[:3] + rotation @ spec.tool_offset
        np.testing.assert_allclose(tip, spec.origin+spec.frame @ [state[0], state[1], 0.], atol=2e-14)
        np.testing.assert_allclose(rotation @ spec.tool_rotation[:, 2], -spec.frame[:, 2], atol=2e-14)
        assert np.linalg.det(rotation) == pytest.approx(1., abs=2e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_world_and_normalized_canonical_decoders_agree(dtype):
    spec = ContactProjectionSpec.from_condition(condition("cylinder"))
    _, path64 = noisy_path(spec)
    path = path64.to(dtype)
    world_condition = torch.cat((path[0], path[-1]))
    frame = CanonicalFrame.from_pose9_condition(world_condition.numpy())
    canonical = torch.tensor(frame.pose9_to_canonical(path.numpy()), dtype=dtype)
    mean = torch.tensor([.1, -.2, .4, .03, -.07, .09, .2, .1, -.1], dtype=dtype)
    std = torch.tensor([1.2, 1.7, .7, .5, .4, .6, .7, .3, .8], dtype=dtype)
    normalized = (canonical-mean)/std
    result = project_contact_pose9(normalized, spec, coordinate_frame="canonical",
                                   world_condition_pose9=world_condition,
                                   normalization_mean=mean, normalization_std=std)
    world = project_contact_pose9(path, spec)
    tolerance = 8e-7 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(result.world_pose9, world.world_pose9, rtol=0., atol=tolerance)
    torch.testing.assert_close(result.projected_pose9[[0, -1]], normalized[[0, -1]], rtol=0., atol=0.)
    assert result.geometry_valid
    assert float(result.after["contact_error_max_m"]) < (2e-6 if dtype == torch.float32 else 2e-14)


def test_differentiable_batch_projection_keeps_finite_gradients():
    spec = ContactProjectionSpec.from_condition(condition())
    _, path = noisy_path(spec)
    batch = torch.stack((path, path)).requires_grad_()
    result = project_contact_pose9(batch, spec)
    result.projected_pose9[:, 1:-1].square().mean().backward()
    assert batch.grad is not None and torch.isfinite(batch.grad).all()
    assert float(batch.grad[:, 1:-1, :3].abs().sum()) > 0
    assert float(batch.grad[:, 1:-1, 3:].abs().sum()) > 0
    assert result.geometry_valid.shape == (2,)


def test_plane_chart_clamping_does_not_claim_obstacle_repair_or_hide_supports():
    c = condition()
    c["surface_geometry"]["supports"] = [{"type": "box", "id": "support", "size": [.08, .08, .08],
                                          "position": [1.2, -.4, .7], "quaternion_wxyz": [1., 0., 0., 0.]}]
    c["surface_chart"]["forbidden_regions"] = [[-.005, .005, -.005, .005]]
    c["start_goal"] = {"start": [-1., 0., 0.], "goal": [1., 0., 0.]}
    spec = ContactProjectionSpec.from_condition(c)
    states = torch.tensor([[-1., 0., 0.], [-.1, 0., 0.], [.1, 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    result = project_contact_pose9(lift_contact_intrinsic(states, spec), spec)
    assert not result.geometry_valid
    assert int(result.after["forbidden_edge_violation_count"]) == 1
    assert int(result.after["tool_box_violation_count"]) == 0
    assert int(result.after["dense_tool_box_violation_count"]) > 0
    _, path = path_for(spec)
    path[3, 0] += 20
    clamped = project_contact_pose9(path, spec)
    assert float(clamped.intrinsic_states[3, 0]) == spec.bounds[0, 1]
    assert int(clamped.before["chart_violation_count"]) > 0


def test_projection_reports_incompatible_attitude_without_breaking_normal_constraint():
    c = condition()
    spec = ContactProjectionSpec.from_condition(c)
    _, path = path_for(spec)
    # Changing the requested flight attitude limits can make exact contact infeasible.
    c["contact_constraints"]["attitude_limits_deg"]["roll"] = [5., 35.]
    spec = ContactProjectionSpec.from_condition(c)
    result = project_contact_pose9(path, spec)
    assert not result.geometry_valid
    assert int(result.after["dense_attitude_violation_count"]) > 0
    assert float(result.after["normal_error_max_rad"]) < 1e-14


def test_twist_range_is_clamped_and_reported_without_changing_endpoints():
    c = condition()
    c["surface_chart"]["bounds"][2] = [-.3, .3]
    spec = ContactProjectionSpec.from_condition(c)
    states, _ = path_for(spec)
    states[3:8, 2] = 1.0
    result = project_contact_pose9(lift_contact_intrinsic(states, spec), spec)
    assert float(result.intrinsic_states[:, 2].max()) == pytest.approx(.3)
    assert int(result.before["chart_violation_count"]) == 5
    assert int(result.after["chart_violation_count"]) == 0
    assert result.geometry_valid
    assert float(result.after["effective_position_tolerance_m"]) == 2e-6


def test_necessary_90_degree_turn_is_retained_but_full_spin_and_jitter_are_rejected():
    c = condition()
    c["start_goal"] = {"start": [-1., 0., 0.], "goal": [1., 0., 0.]}
    spec = ContactProjectionSpec.from_condition(c)
    states, _ = path_for(spec, 17)
    states[:, 2] = torch.tensor([0., .3, .6, .9, 1.2, math.pi/2, math.pi/2, math.pi/2, math.pi/2,
                                math.pi/2, math.pi/2, 1.2, .9, .6, .3, 0., 0.])
    maneuver = project_contact_pose9(lift_contact_intrinsic(states, spec), spec)
    assert maneuver.geometry_valid
    assert float(maneuver.after["psi_total_variation_rad"]) == pytest.approx(math.pi, abs=1e-6)
    # A full rotation returns to the same endpoint rotation; endpoint clamping
    # does not excuse the spin. Use enough nodes to make the unwrap unambiguous.
    spin, _ = path_for(spec, 33)
    spin[:, 2] = torch.linspace(0., 2*math.pi, 33, dtype=torch.float64)
    spinning = project_contact_pose9(lift_contact_intrinsic(spin, spec), spec)
    assert not spinning.geometry_valid
    jitter, _ = path_for(spec, 17)
    jitter[1:-1, 2] = torch.tensor([.25, -.25]*7+[.25])
    rejected = project_contact_pose9(lift_contact_intrinsic(jitter, spec), spec)
    assert not rejected.geometry_valid
    assert float(rejected.after["psi_excess_reversal_variation_rad"]) > math.pi/6


def test_cylinder_chart_branch_across_atan2_seam_is_not_wrapped():
    c = condition("cylinder")
    c["surface_chart"]["bounds"][0] = [1.5*2.8, 1.5*3.5]
    c["start_goal"] = {"start": [1.5*2.9, -.5, 0.], "goal": [1.5*3.4, .5, 0.]}
    spec = ContactProjectionSpec.from_condition(c)
    states, path = path_for(spec)
    result = project_contact_pose9(path, spec)
    torch.testing.assert_close(result.intrinsic_states, states, rtol=0., atol=2e-14)
    assert torch.diff(result.intrinsic_states[:, 0]).min() > 0
    assert float(result.after["normal_error_max_rad"]) < 1e-14


def test_degenerate_rotation_and_cylinder_axis_are_finite_and_counted():
    spec = ContactProjectionSpec.from_condition(condition("cylinder"))
    _, path = path_for(spec)
    path[5, 3:9] = 0
    # _safe_rotations chooses identity for zero 6D. Place the resulting tip on axis.
    path[5, :3] = torch.tensor(spec.origin-spec.tool_offset)
    result = project_contact_pose9(path, spec)
    assert torch.isfinite(result.projected_pose9).all()
    assert int(result.before["degenerate_rotation_count"]) == 1
    assert int(result.after["input_cylinder_axis_degeneracy_count"]) == 1
    assert float(result.after["contact_error_max_m"]) < 2e-14


def test_projection_ignores_expert_and_route_fields():
    c = condition()
    spec = ContactProjectionSpec.from_condition(c)
    _, path = noisy_path(spec)
    c.update(generation_policy={"route_certificates": "invalid"}, campaign_metadata="invalid",
             route_certificates=[float("nan")], experts="invalid")
    ignored = ContactProjectionSpec.from_condition(c)
    first, second = project_contact_pose9(path, spec), project_contact_pose9(path, ignored)
    torch.testing.assert_close(first.projected_pose9, second.projected_pose9, rtol=0., atol=0.)


def test_invalid_endpoints_nonfinite_input_and_bad_specs_are_rejected():
    c = condition()
    spec = ContactProjectionSpec.from_condition(c)
    _, path = path_for(spec)
    changed = path.clone(); changed[0, 0] += .001
    with pytest.raises(ValueError, match="endpoints"):
        project_contact_pose9(changed, spec)
    changed = path.clone(); changed[4, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        project_contact_pose9(changed, spec)
    with pytest.raises(ValueError, match="world_condition"):
        project_contact_pose9(path, spec, coordinate_frame="canonical")
    with pytest.raises(ValueError, match="normalization"):
        project_contact_pose9(path, spec, normalization_mean=np.zeros(9), normalization_std=np.zeros(9))
    changed_condition = copy.deepcopy(c)
    changed_condition["tool_extrinsics"]["quaternion_wxyz"] = [0., 0., 0., 0.]
    with pytest.raises(ValueError, match="unit"):
        ContactProjectionSpec.from_condition(changed_condition)
