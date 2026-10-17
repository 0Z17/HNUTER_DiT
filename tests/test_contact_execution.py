from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hnuter_dit.contact_execution import (
    _IntrinsicContactCurve, _IntrinsicContactPolyline, audit_contact_reference, build_contact_scene,
    contact_execution_metrics, prepare_contact_reference, prepare_contact_polyline_reference,
)
from hnuter_dit.contact_projection import ContactProjectionSpec, lift_contact_intrinsic
from hnuter_dit.mujoco_execution import (
    MODEL_FILENAME, MujocoExecutionManager, ScenePoseTrackingCost, build_scene_xml,
)


def condition(kind="plane"):
    result = {
        "surface_geometry": {"type": kind, "origin": [0., 0., .5], "quaternion_wxyz": [1., 0., 0., 0.],
                             "radius_m": 1.5, "thickness_m": .2, "supports": []},
        "surface_chart": {"bounds": [[-2., 2.], [-2., 2.], [-math.pi, math.pi]], "forbidden_regions": []},
        "tool_extrinsics": {"position_body": [.1, -.04, -.8], "quaternion_wxyz": [0., 1., 0., 0.]},
        "contact_constraints": {"position_tolerance_m": 1e-7, "normal_tolerance_rad": 1e-6,
                                "collision_margin_m": .01, "validation_step_m": .02, "validation_step_rad": .02},
        "start_goal": {"start": [-1., 0., .1], "goal": [1., .2, .3]},
        "environment": {"obstacles": [], "bounds": {"min": [-4., -4., 0.], "max": [4., 4., 5.]}},
    }
    if kind == "cylinder":
        radius = 1.5
        result["surface_geometry"].update(origin=[0., 0., radius], quaternion_wxyz=[2**-.5, 2**-.5, 0., 0.])
        result["surface_chart"]["bounds"][0] = [radius*(math.pi/2-.3), radius*(math.pi/2+.3)]
        result["start_goal"] = {"start": [radius*(math.pi/2-.25), -.8, .1],
                                "goal": [radius*(math.pi/2+.25), .8, .3]}
    return result


def controls(c):
    values = np.linspace(c["start_goal"]["start"], c["start_goal"]["goal"], 16)
    values[1:-1, 2] += .04*np.sin(np.linspace(0., np.pi, 16))[1:-1]
    return values


@pytest.mark.parametrize("kind", ["plane", "cylinder"])
def test_contact_curve_has_exact_lift_and_true_body_angular_derivatives(kind):
    c = condition(kind)
    curve = _IntrinsicContactCurve(c, controls(c))
    s = np.linspace(.01, .99, 1001)
    eps = 1e-5
    pose = curve.geometry(s)
    expected = lift_contact_intrinsic(curve.intrinsic(s), ContactProjectionSpec.from_condition(c)).numpy()
    r = Rotation.from_quat(pose[:, [4, 5, 6, 3]])
    np.testing.assert_allclose(pose[:, :3], expected[:, :3], atol=1e-12)
    np.testing.assert_allclose(r.as_matrix()[:, :, :2], np.stack((expected[:, 3:6], expected[:, 6:9]), -1), atol=1e-12)
    first, second = curve.derivatives(s)
    plus, minus = curve.geometry(s+eps), curve.geometry(s-eps)
    rp, rm = Rotation.from_quat(plus[:, [4, 5, 6, 3]]), Rotation.from_quat(minus[:, [4, 5, 6, 3]])
    numerical_omega = ((r.inv()*rp).as_rotvec()-(r.inv()*rm).as_rotvec())/(2*eps)
    np.testing.assert_allclose(first[:, :3], (plus[:, :3]-minus[:, :3])/(2*eps), atol=2e-6)
    np.testing.assert_allclose(first[:, 3:], numerical_omega, atol=2e-6)
    np.testing.assert_allclose(second, (curve.derivatives(s+eps)[0]-curve.derivatives(s-eps)[0])/(2*eps), atol=2e-5)
    np.testing.assert_allclose(curve.third_derivatives(s),
                               (curve.derivatives(s+eps)[1]-curve.derivatives(s-eps)[1])/(2*eps), atol=5e-4)


def test_task_retiming_preserves_cylinder_contact_at_every_time_sample():
    pytest.importorskip("toppra")
    pytest.importorskip("coal")
    c = condition("cylinder")
    reference = prepare_contact_reference(c, controls(c))
    state, acceleration = reference.sample(np.linspace(0., reference.duration+.1, 3001))
    measured = contact_execution_metrics(c, np.column_stack((state[:, :3], state[:, 6:10])))
    assert measured["contact_error_max_m"] < 1e-12
    assert measured["normal_error_max_rad"] < 1e-12
    assert measured["task_success"]
    assert np.linalg.norm(state[:, 3:6], axis=1).max() <= reference.config.speed_mps
    assert np.linalg.norm(acceleration[:, :3], axis=1).max() <= reference.config.max_acceleration_mps2
    assert reference.describe()["differs_from_frozen_benchmark_edges"]
    assert not reference.describe()["contact_force_control"]


@pytest.mark.parametrize("kind", ["plane", "cylinder"])
def test_exact_execution_curve_has_independent_full_urdf_audit(kind):
    pytest.importorskip("coal")
    pytest.importorskip("toppra")
    c = condition(kind)
    reference = prepare_contact_reference(c, controls(c))
    audit = audit_contact_reference(c, reference)
    assert audit["checked"] and audit["physical_collision_checked"]
    assert audit["sample_count"] > 128
    assert audit["analytic_curve_sampled_maximum_curvature_inv_m"] <= 8
    changed = copy.deepcopy(c)
    changed["tool_extrinsics"]["position_body"][0] += .1
    with pytest.raises(ValueError, match="tool_offset"):
        audit_contact_reference(changed, reference)
    blocked = copy.deepcopy(c)
    pose = reference.curve.geometry([.5])[0]
    blocked["environment"]["obstacles"].append({"id": "blocker", "type": "box", "size": [1, 1, 1],
                                                "position": pose[:3].tolist(), "quaternion_wxyz": [1, 0, 0, 0]})
    blocked_reference = prepare_contact_reference(blocked, controls(blocked))
    with pytest.raises(ValueError, match="full_urdf_collision"):
        audit_contact_reference(blocked, blocked_reference)


def test_actual_tool_drift_fails_task_even_if_flight_endpoint_is_close():
    pytest.importorskip("coal")
    c = condition()
    curve = _IntrinsicContactCurve(c, controls(c))
    poses = curve.geometry(np.linspace(0., 1., 101))
    poses[:, 2] += .002
    measured = contact_execution_metrics(c, poses)
    assert not measured["task_success"]
    assert measured["contact_error_max_m"] == pytest.approx(.002)
    assert not measured["physical_collision_checked"]
    assert not measured["contact_force_control"]


def test_actual_exact_contact_twist_reversals_still_fail_task_rules():
    pytest.importorskip("coal")
    c = condition()
    spec = ContactProjectionSpec.from_condition(c)
    states = np.linspace(spec.start, spec.goal, 128)
    states[1:-1, 2] = np.tile([-.1, .1], 63)
    world = lift_contact_intrinsic(states, spec).numpy()
    rotation = np.stack((world[:, 3:6], world[:, 6:9],
                         np.cross(world[:, 3:6], world[:, 6:9])), axis=-1)
    quaternion = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
    measured = contact_execution_metrics(c, np.column_stack((world[:, :3], quaternion)))
    assert measured["practical_contact_gate"] and measured["practical_normal_gate"]
    assert measured["chart_valid"] and measured["attitude_valid"]
    assert not measured["psi_quality_gate"]
    assert not measured["task_success"]


def test_bounded_control_and_endpoint_binding_reject_invalid_reference():
    c = condition()
    outside = controls(c)
    outside[5, 0] = 3.
    with pytest.raises(ValueError, match="bounded"):
        _IntrinsicContactCurve(c, outside)
    invalid = controls(c)
    invalid[0, 0] += .01
    with pytest.raises(ValueError, match="endpoints"):
        _IntrinsicContactCurve(c, invalid)


def test_contact_scene_xml_has_exact_cylinder_and_nonphysical_tool_sites(tmp_path):
    (tmp_path/MODEL_FILENAME).write_text('<mujoco><worldbody><body name="drone"><freejoint/></body></worldbody></mujoco>')
    c = condition("cylinder")
    c["surface_geometry"]["supports"] = [{"id": "support", "type": "box", "size": [.2, .2, .3],
        "position": [0., 0., -.2], "quaternion_wxyz": [1, 0, 0, 0]}]
    scene = build_contact_scene(c)
    curve = _IntrinsicContactCurve(c, controls(c))
    endpoints = curve.geometry([0., 1.])
    xml = ET.fromstring(build_scene_xml(scene, tmp_path, *endpoints))
    cylinder = xml.find("worldbody/geom[@type='cylinder']")
    assert cylinder is not None
    np.testing.assert_allclose(np.fromstring(cylinder.get("size"), sep=" "), [1.5, 2.])
    assert len(scene["obstacles"]) == 2
    assert xml.find("worldbody/body/site[@name='ideal_contact_tool_tip']") is not None
    # Sites add no collision geoms to the robot or physical tool mass.
    assert not xml.findall("worldbody/body/geom")


def test_finite_cylinder_proxy_does_not_penalize_free_aabb_corner():
    class ZeroCost:
        def trajectory_cost(self, states, controls, reference):
            return np.zeros(len(states))
    scene = {"obstacles": [{"type": "cylinder", "collision": True, "radius_m": 2., "height_m": 6.,
        "size_xyz": [4., 4., 6.], "pose": {"position": [0., 0., 0.], "quaternion_wxyz": [1., 0, 0, 0]}}]}
    cost = ScenePoseTrackingCost(ZeroCost(), scene)
    states = np.zeros((2, 5, 13))
    states[..., 6] = 1.
    states[0, :, :3] = [0., 0., 0.]
    states[1, :, :3] = [2.8, 2.8, 0.]
    penalties = cost.trajectory_cost(states, np.zeros((2, 4, 6)), states[1])
    assert penalties[0] > 100
    assert penalties[1] == 0


def test_manager_rejects_an_injected_reference_with_different_endpoints(monkeypatch):
    pytest.importorskip("toppra")
    c = condition()
    reference = prepare_contact_reference(c, controls(c))
    path = reference.poses.copy()
    path[-1, 0] += .1
    manager = MujocoExecutionManager(viewer=False)
    monkeypatch.setattr(manager, "_ensure_available", lambda: None)
    with pytest.raises(ValueError, match="endpoints"):
        manager.start(build_contact_scene(c), path, reference=reference)
    manager.close()


def noisy_intrinsic_states(c, amplitude=.003):
    values = np.linspace(c["start_goal"]["start"], c["start_goal"]["goal"], 128)
    values[1:-1, 1] += amplitude*np.tile([-1., 1.], 63)
    return values


@pytest.mark.parametrize("kind", ["plane", "cylinder"])
def test_polyline_keeps_all_original_nodes_and_analytic_lift_derivatives(kind):
    c = condition(kind)
    states = noisy_intrinsic_states(c)
    curve = _IntrinsicContactPolyline(c, states)
    np.testing.assert_allclose(curve.intrinsic(curve.source_parameters), states, atol=1e-15)
    s = (np.arange(127)+.4)/127
    eps = 1e-7
    pose = curve.geometry(s)
    expected = lift_contact_intrinsic(curve.intrinsic(s), ContactProjectionSpec.from_condition(c)).numpy()
    rotation = Rotation.from_quat(pose[:, [4, 5, 6, 3]])
    np.testing.assert_allclose(pose[:, :3], expected[:, :3], atol=1e-12)
    np.testing.assert_allclose(rotation.as_matrix()[:, :, :2],
                               np.stack((expected[:, 3:6], expected[:, 6:9]), -1), atol=1e-12)
    plus, minus = curve.geometry(s+eps), curve.geometry(s-eps)
    rp, rm = Rotation.from_quat(plus[:, [4, 5, 6, 3]]), Rotation.from_quat(minus[:, [4, 5, 6, 3]])
    first, second = curve.derivatives(s)
    numerical_omega = ((rotation.inv()*rp).as_rotvec()-(rotation.inv()*rm).as_rotvec())/(2*eps)
    np.testing.assert_allclose(first[:, :3], (plus[:, :3]-minus[:, :3])/(2*eps), atol=1e-8)
    np.testing.assert_allclose(first[:, 3:], numerical_omega, atol=1e-8)
    np.testing.assert_allclose(second, (curve.derivatives(s+eps)[0]-curve.derivatives(s-eps)[0])/(2*eps), atol=1e-8)
    np.testing.assert_allclose(curve.third_derivatives(s),
                               (curve.derivatives(s+eps)[1]-curve.derivatives(s-eps)[1])/(2*eps), atol=2e-8)


@pytest.mark.parametrize("kind", ["plane", "cylinder"])
def test_polyline_quintic_timing_stops_at_original_nodes_and_bounds_kinematics(kind):
    c = condition(kind)
    states = noisy_intrinsic_states(c)
    reference = prepare_contact_polyline_reference(c, states, {"initial_hold_s": .2}, allow_curvature_failure=True)
    knot_state, knot_acceleration = reference.sample(reference.knot_times+reference.config.initial_hold_s)
    np.testing.assert_allclose(knot_state[:, :3], reference.poses[:, :3], atol=1e-12)
    np.testing.assert_allclose(knot_state[:, 6:10], reference.poses[:, 3:], atol=1e-12)
    np.testing.assert_allclose(knot_state[:, 3:6], 0., atol=1e-12)
    np.testing.assert_allclose(knot_state[:, 10:13], 0., atol=1e-12)
    np.testing.assert_allclose(knot_acceleration, 0., atol=1e-12)
    description = reference.describe()
    assert description["stops_at_waypoints"] and description["source_node_count"] == 128
    assert not description["geometry_smoothing"] and not description["differs_from_frozen_benchmark_edges"]
    # Check independent temporal differences inside every original edge.
    times = reference.config.initial_hold_s+reference.knot_times[:-1]+.4*reference.segment_durations
    state, acceleration = reference.sample(times)
    eps = 1e-5
    plus, plus_acceleration = reference.sample(times+eps)
    minus, minus_acceleration = reference.sample(times-eps)
    rotation = Rotation.from_quat(state[:, [7, 8, 9, 6]])
    rp, rm = Rotation.from_quat(plus[:, [7, 8, 9, 6]]), Rotation.from_quat(minus[:, [7, 8, 9, 6]])
    np.testing.assert_allclose(state[:, 3:6], (plus[:, :3]-minus[:, :3])/(2*eps), atol=1e-8)
    np.testing.assert_allclose(state[:, 10:13], ((rotation.inv()*rp).as_rotvec()-(rotation.inv()*rm).as_rotvec())/(2*eps), atol=1e-8)
    np.testing.assert_allclose(acceleration, (plus[:, [3, 4, 5, 10, 11, 12]]-minus[:, [3, 4, 5, 10, 11, 12]])/(2*eps), atol=1e-8)
    np.testing.assert_allclose(reference._sample_active_jerk(times-reference.config.initial_hold_s),
                               (plus_acceleration-minus_acceleration)/(2*eps), atol=1e-7)
    # Check each incoming one-sided jerk even when quintic progress rounds
    # to the next node. Its derivative must still belong to the incoming edge.
    before_arrival = np.nextafter(reference.knot_times[1:], reference.knot_times[:-1])
    incoming_first = reference.curve.edge_kinematics(np.arange(127), np.ones(127))[2]
    expected_incoming_jerk = incoming_first*(60/127/reference.segment_durations**3)[:, None]
    np.testing.assert_allclose(reference._sample_active_jerk(before_arrival), expected_incoming_jerk, atol=1e-10)
    probes = reference.knot_times[:-1, None]+reference.segment_durations[:, None]*np.linspace(0., 1., 257)
    all_state, all_acceleration = reference.sample(probes.ravel()+reference.config.initial_hold_s)
    jerk = reference._sample_active_jerk(probes.ravel())
    assert np.linalg.norm(all_state[:, 3:6], axis=1).max() <= reference.config.speed_mps
    assert np.linalg.norm(all_state[:, 10:13], axis=1).max() <= reference.config.max_angular_speed_rps
    assert np.linalg.norm(all_acceleration[:, :3], axis=1).max() <= reference.config.max_acceleration_mps2
    assert np.linalg.norm(all_acceleration[:, 3:], axis=1).max() <= reference.config.max_angular_acceleration_rps2
    assert np.linalg.norm(jerk[:, :3], axis=1).max() <= reference.max_linear_jerk_mps3
    assert np.linalg.norm(jerk[:, 3:], axis=1).max() <= reference.max_angular_jerk_rps3


@pytest.mark.parametrize("decoder", ["on", "on_smooth"])
def test_diagnostic_curvature_warning_preserves_formal_failure_and_collision_gate(decoder):
    pytest.importorskip("coal")
    if decoder == "on_smooth":
        pytest.importorskip("toppra")
    c = condition()
    if decoder == "on":
        values = noisy_intrinsic_states(c)
        prepare = prepare_contact_polyline_reference
    else:
        values = controls(c)
        values[1:-1, 1] += .05*np.tile([-1., 1.], 7)
        prepare = prepare_contact_reference
    strict = prepare(c, values)
    with pytest.raises(ValueError, match="shape"):
        audit_contact_reference(c, strict)
    reference = prepare(c, values, allow_curvature_failure=True)
    audit = audit_contact_reference(c, reference)
    assert audit["valid"] and audit["physical_collision_checked"]
    assert audit["curvature_warning"] and audit["execution_warnings"] and audit["curvature_failure_allowed"]
    assert not audit["candidate_success"] and not audit["formal_shape_quality_gate"]
    assert not audit["source_shape_metrics"]["shape_quality_gate"]
    assert audit["source_node_maximum_curvature_inv_m"] > 8.
    blocked = copy.deepcopy(c)
    pose = reference.curve.geometry([.5])[0]
    blocked["environment"]["obstacles"].append({"id": "blocker", "type": "box", "size": [1., 1., 1.],
                                                "position": pose[:3].tolist(), "quaternion_wxyz": [1., 0., 0., 0.]})
    collision_reference = prepare(blocked, values, allow_curvature_failure=True)
    with pytest.raises(ValueError, match="full_urdf_collision"):
        audit_contact_reference(blocked, collision_reference)


def test_diagnostic_mode_never_waives_other_shape_constraints():
    pytest.importorskip("coal")
    c = condition()
    values = noisy_intrinsic_states(c, .03)
    reference = prepare_contact_polyline_reference(c, values, allow_curvature_failure=True)
    with pytest.raises(ValueError, match="shape"):
        audit_contact_reference(c, reference)
