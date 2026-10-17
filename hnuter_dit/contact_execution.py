"""Contact-preserving time references and independently audited physical trials.

Execution supports both the mathematical intrinsic cubic and the exact frozen
benchmark's 128-node intrinsic-linear edges. The latter stops at every original
node so its corners never imply instantaneous velocity changes. Both actual
curves must pass a full-URDF audit. The calibrated tool is an ideal point and normal only:
this module does not implement a material tool, force control, or friction.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from .contact_projection import ContactProjectionSpec, lift_contact_intrinsic
from .contact_metrics import (
    _COAL_LOCK, _backend, _constraint_metrics, _infer_states, _quaternions,
    _rotations, _rotation_angles, _scene_checker, _scene_key, _shape_metrics,
)
from .retiming import TimedPoseReference, _config


EXECUTION_CONTACT_VERSION = "contact_intrinsic_cubic_toppra_v001"
EXECUTION_POLYLINE_VERSION = "contact_intrinsic_linear_quintic_stops_v001"
CONTACT_LIMITATION = (
    "Ideal calibrated tool point/normal tracking; no physical tool geometry, "
    "contact force controller, friction, or approach/retreat model."
)


def build_contact_scene(condition: Mapping[str, Any]) -> dict[str, Any]:
    """Build exactly the obstacles/supports/target used by contact URDF auditing."""
    spec = ContactProjectionSpec.from_condition(condition)
    geometry = condition["surface_geometry"]
    items = []
    for index, box in enumerate([*condition.get("environment", {}).get("obstacles", []),
                                 *geometry.get("supports", [])]):
        items.append({"id": str(box.get("id", f"contact_box_{index}")), "type": "box",
                      "role": box.get("role", "obstacle"), "collision": True, "visual": True,
                      "size_xyz": copy.deepcopy(box["size"]),
                      "pose": {"position": copy.deepcopy(box["position"]),
                               "quaternion_wxyz": copy.deepcopy(box["quaternion_wxyz"])}})
    q = _quaternions(spec.frame).tolist()
    if spec.kind == "plane":
        thickness = float(geometry.get("thickness_m", .2))
        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError("surface thickness must be finite and positive")
        local = np.r_[spec.bounds[:2].mean(axis=1), -thickness/2]
        size = [*np.ptp(spec.bounds[:2], axis=1), thickness]
        target = {"type": "box", "size_xyz": size}
    else:
        local = np.array([0., 0., spec.bounds[1].mean()])
        height = float(np.ptp(spec.bounds[1]))
        target = {"type": "cylinder", "radius_m": spec.radius, "height_m": height,
                  "size_xyz": [2*spec.radius, 2*spec.radius, height]}
    target.update(id="contact_surface", role="contact_target", collision=True, visual=True,
                  pose={"position": (spec.origin+spec.frame@local).tolist(), "quaternion_wxyz": q})
    items.append(target)
    result = {"environment_id": "contact_task", "obstacles": items,
              "contact_model": "ideal_tool_point_and_normal", "contact_limitations": CONTACT_LIMITATION,
              "contact_condition": copy.deepcopy(dict(condition))}
    if spec.workspace is not None:
        result["sampling_space"] = {"position_bounds": {"min": spec.workspace[0].tolist(),
                                                        "max": spec.workspace[1].tolist()}}
    return result


class _IntrinsicContactCurve:
    """Analytic SE(3) lift of a clamped cubic with true body angular derivatives."""

    position_deviation_m = 0.0
    rotation_deviation_rad = 0.0

    def __init__(self, condition: Mapping[str, Any], controls: Any) -> None:
        from scipy.interpolate import BSpline
        self.spec = ContactProjectionSpec.from_condition(condition)
        self.controls = np.asarray(controls, dtype=np.float64).copy()
        if self.controls.shape != (16, 3) or not np.isfinite(self.controls).all():
            raise ValueError("contact control_points must be finite [16,3]")
        if (np.any(self.controls < self.spec.bounds[:, 0]-1e-10)
                or np.any(self.controls > self.spec.bounds[:, 1]+1e-10)):
            raise ValueError("contact control points exceed bounded intrinsic chart")
        if not np.allclose(self.controls[[0, -1]], [self.spec.start, self.spec.goal], atol=2e-6, rtol=0):
            raise ValueError("contact controls do not bind to task endpoints")
        self.parameters = np.linspace(0., 1., 14)
        self.source_parameters = np.linspace(0., 1., 128)
        knots = np.r_[np.zeros(4), np.arange(1, 13)/13, np.ones(4)]
        self.spline = BSpline(knots, self.controls, 3, extrapolate=False)
        self.poses = self.geometry(self.source_parameters)
        if np.max(np.linalg.norm(np.diff(self.poses[:, :3], axis=0), axis=1)) < 1e-10 and np.max(
                _rotation_angles(self._kinematics(self.source_parameters)[1][:-1],
                                 self._kinematics(self.source_parameters)[1][1:])) < 1e-10:
            raise ValueError("contact reference has no motion")

    def intrinsic(self, parameters: Any, order: int = 0) -> np.ndarray:
        values = np.atleast_1d(np.asarray(parameters, dtype=np.float64))
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("contact parameters must be a finite vector")
        return np.asarray(self.spline(np.clip(values, 0., 1.), nu=order))

    def _kinematics(self, parameters: Any, *, intrinsic: tuple[np.ndarray, ...] | None = None) -> tuple[np.ndarray, ...]:
        s, d, dd, ddd = ([self.intrinsic(parameters, order) for order in range(4)]
                         if intrinsic is None else intrinsic)
        spec = self.spec
        n = len(s)
        zero = np.zeros(n)
        ez = np.tile([0., 0., 1.], (n, 1))
        if spec.kind == "plane":
            normal = ez.copy()
            tangent = np.tile([1., 0., 0.], (n, 1))
            theta_d = theta_dd = theta_ddd = zero
            point = np.column_stack((s[:, :2], zero))
            p1, p2, p3 = [np.column_stack((v[:, :2], zero)) for v in (d, dd, ddd)]
        else:
            theta = s[:, 0]/spec.radius
            theta_d, theta_dd, theta_ddd = [v[:, 0]/spec.radius for v in (d, dd, ddd)]
            normal = np.column_stack((np.cos(theta), np.sin(theta), zero))
            tangent = np.column_stack((-np.sin(theta), np.cos(theta), zero))
            point = spec.radius*normal+ez*s[:, 1, None]
            p1 = spec.radius*tangent*theta_d[:, None]+ez*d[:, 1, None]
            p2 = spec.radius*(-normal*theta_d[:, None]**2+tangent*theta_dd[:, None])+ez*dd[:, 1, None]
            p3 = spec.radius*(-tangent*theta_d[:, None]**3-3*normal*(theta_d*theta_dd)[:, None]
                              +tangent*theta_ddd[:, None])+ez*ddd[:, 1, None]
        inward = -normal
        basis = np.stack((tangent, np.cross(inward, tangent), inward), axis=-1)
        c, sn = np.cos(s[:, 2]), np.sin(s[:, 2])
        twist = np.zeros((n, 3, 3))
        twist[:, 0, 0] = twist[:, 1, 1] = c
        twist[:, 1, 0], twist[:, 0, 1], twist[:, 2, 2] = sn, -sn, 1.
        rotation = spec.frame@basis@twist@spec.tool_rotation.T
        inward_d = -tangent*theta_d[:, None]
        inward_dd = normal*theta_d[:, None]**2-tangent*theta_dd[:, None]
        w = (ez*theta_d[:, None]+inward*d[:, 2, None])@spec.frame.T
        w1 = (ez*theta_dd[:, None]+inward_d*d[:, 2, None]+inward*dd[:, 2, None])@spec.frame.T
        w2 = (ez*theta_ddd[:, None]+inward_dd*d[:, 2, None]
              +2*inward_d*dd[:, 2, None]+inward*ddd[:, 2, None])@spec.frame.T
        arm = rotation@spec.tool_offset
        arm1 = np.cross(w, arm)
        arm2 = np.cross(w1, arm)+np.cross(w, arm1)
        position = point@spec.frame.T+spec.origin-arm
        position_d = p1@spec.frame.T-arm1
        position_dd = p2@spec.frame.T-arm2
        position_ddd = p3@spec.frame.T-np.cross(w2, arm)-2*np.cross(w1, arm1)-np.cross(w, arm2)
        rt = np.swapaxes(rotation, -1, -2)
        omega = np.einsum("nij,nj->ni", rt, w)
        omega_d = np.einsum("nij,nj->ni", rt, w1)
        omega_dd = np.einsum("nij,nj->ni", rt, w2-np.cross(w, w1))
        return position, rotation, np.column_stack((position_d, omega)), np.column_stack((position_dd, omega_d)), np.column_stack((position_ddd, omega_dd))

    def geometry(self, parameters: Any) -> np.ndarray:
        position, rotation, *_ = self._kinematics(parameters)
        return np.column_stack((position, _quaternions(rotation)))

    def derivatives(self, parameters: Any) -> tuple[np.ndarray, np.ndarray]:
        _, _, first, second, _ = self._kinematics(parameters)
        return first, second

    def third_derivatives(self, parameters: Any) -> np.ndarray:
        return self._kinematics(parameters)[4]


class TimedContactReference(TimedPoseReference):
    def __init__(self, condition: Mapping[str, Any], controls: Any, config: Any = None,
                 *, allow_curvature_failure: bool = False) -> None:
        self.condition = copy.deepcopy(dict(condition))
        self.allow_curvature_failure = bool(allow_curvature_failure)
        self.contact_curve = _IntrinsicContactCurve(condition, controls)
        super().__init__(self.contact_curve.poses, config, curve=self.contact_curve)

    def describe(self) -> dict[str, Any]:
        result = super().describe()
        result.update(method=EXECUTION_CONTACT_VERSION, geometry="analytic_intrinsic_cubic_contact_lift",
                      differs_from_frozen_benchmark_edges=True, contact_limitations=CONTACT_LIMITATION,
                      contact_force_control=False, diagnostic_execution=self.allow_curvature_failure,
                      geometry_smoothing=False)
        return result


def prepare_contact_reference(condition: Mapping[str, Any], control_points: Any,
                              config: Any = None, *, allow_curvature_failure: bool = False) -> TimedContactReference:
    return TimedContactReference(condition, control_points, config,
                                 allow_curvature_failure=allow_curvature_failure)


class _IntrinsicContactPolyline(_IntrinsicContactCurve):
    """Lift each original intrinsic-linear edge without any geometric smoothing.

    Spatial derivatives at a node use its outgoing edge. The time reference
    has zero speed and acceleration at every node, so the composed temporal
    position and orientation remain C2 across these spatial corners.
    """

    def __init__(self, condition: Mapping[str, Any], states: Any) -> None:
        self.spec = ContactProjectionSpec.from_condition(condition)
        self.states = np.asarray(states, dtype=np.float64).copy()
        if self.states.shape != (128, 3) or not np.isfinite(self.states).all():
            raise ValueError("contact intrinsic_states must be finite [128,3]")
        if (np.any(self.states < self.spec.bounds[:, 0]-1e-10)
                or np.any(self.states > self.spec.bounds[:, 1]+1e-10)):
            raise ValueError("contact intrinsic states exceed bounded intrinsic chart")
        if not np.allclose(self.states[[0, -1]], [self.spec.start, self.spec.goal], atol=2e-6, rtol=0):
            raise ValueError("contact intrinsic states do not bind to task endpoints")
        self.source_parameters = self.parameters = np.linspace(0., 1., len(self.states))
        self.poses = self.geometry(self.source_parameters)
        if np.max(np.linalg.norm(np.diff(self.poses[:, :3], axis=0), axis=1)) < 1e-10 and np.max(
                _rotation_angles(self._kinematics(self.source_parameters)[1][:-1],
                                 self._kinematics(self.source_parameters)[1][1:])) < 1e-10:
            raise ValueError("contact reference has no motion")

    def intrinsic(self, parameters: Any, order: int = 0) -> np.ndarray:
        values = np.atleast_1d(np.asarray(parameters, dtype=np.float64))
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("contact parameters must be a finite vector")
        progress = np.clip(values, 0., 1.)*(len(self.states)-1)
        index = np.minimum(np.floor(progress).astype(int), len(self.states)-2)
        delta = self.states[index+1]-self.states[index]
        if order == 0:
            return self.states[index]+delta*(progress-index)[:, None]
        if order == 1:
            return delta*(len(self.states)-1)
        if order in (2, 3):
            return np.zeros_like(delta)
        raise ValueError("contact intrinsic derivative order must be 0 through 3")

    def edge_derivative_bounds(self) -> tuple[np.ndarray, ...]:
        """Conservative norm bounds for unit progress along each lifted edge."""
        delta = np.diff(self.states, axis=0)
        spec = self.spec
        theta = np.zeros(len(delta)) if spec.kind == "plane" else np.abs(delta[:, 0])/spec.radius
        psi = np.abs(delta[:, 2])
        surface1 = np.linalg.norm(delta[:, :2], axis=1)
        surface2 = np.zeros(len(delta)) if spec.kind == "plane" else np.abs(delta[:, 0])**2/spec.radius
        surface3 = np.zeros(len(delta)) if spec.kind == "plane" else np.abs(delta[:, 0])**3/spec.radius**2
        # World angular derivatives: w = theta' ez + psi' inward,
        # w' = psi' inward', w'' = psi' inward''. Convert the third
        # angular derivative into body coordinates with the transport term.
        w, w1, w2 = theta+psi, theta*psi, theta**2*psi
        arm = float(np.linalg.norm(spec.tool_offset))
        linear1 = surface1+arm*w
        linear2 = surface2+arm*(w1+w**2)
        linear3 = surface3+arm*(w2+3*w*w1+w**3)
        return linear1, linear2, linear3, w, w1, w2+w*w1

    def edge_kinematics(self, indices: np.ndarray, fractions: np.ndarray) -> tuple[np.ndarray, ...]:
        # Near an arrival, quintic progress can round to the exact next node.
        # Bind derivatives to the actual active edge rather than reselecting
        # the outgoing edge from that rounded geometric progress value.
        delta = self.states[indices+1]-self.states[indices]
        states = self.states[indices]+delta*fractions[:, None]
        first = delta*(len(self.states)-1)
        zero = np.zeros_like(delta)
        return self._kinematics(fractions, intrinsic=(states, first, zero, zero))


class TimedContactPolylineReference(TimedPoseReference):
    """Exact 128-node intrinsic path with bounded quintic motion on each edge.

    Every original node is retained. Stopping at corners is necessary to keep
    acceleration finite without changing the selected path. Diagnostic flight
    therefore measures this stop-and-go reference and its tracking errors; it
    cannot demonstrate what traversing the same sharp corner at speed would do.
    """

    def __init__(self, condition: Mapping[str, Any], intrinsic_states: Any, config: Any = None,
                 *, allow_curvature_failure: bool = False) -> None:
        self.condition = copy.deepcopy(dict(condition))
        self.allow_curvature_failure = bool(allow_curvature_failure)
        self.config = _config(config)
        self.contact_curve = self.curve = _IntrinsicContactPolyline(condition, intrinsic_states)
        self.poses = self.curve.poses.copy()
        self.max_linear_jerk_mps3, self.max_angular_jerk_rps3 = 15., 8.
        first, second, third, omega, alpha, angular_third = self.curve.edge_derivative_bounds()
        # Quintic smoothstep maxima over 0 <= u <= 1. Applying the chain
        # rule and triangle inequality bounds all times, including nodes.
        q1, q2, q3 = 1.875, 10/np.sqrt(3), 60.
        linear_speed, angular_speed = first*q1, omega*q1
        linear_accel, angular_accel = second*q1**2+first*q2, alpha*q1**2+omega*q2
        linear_jerk = third*q1**3+3*second*q1*q2+first*q3
        angular_jerk = angular_third*q1**3+3*alpha*q1*q2+omega*q3
        c = self.config
        self.segment_durations = np.maximum.reduce((
            np.full(len(first), c.control_dt),
            linear_speed/(.98*c.speed_mps), angular_speed/(.98*c.max_angular_speed_rps),
            np.sqrt(linear_accel/(.98*c.max_acceleration_mps2)),
            np.sqrt(angular_accel/(.98*c.max_angular_acceleration_rps2)),
            np.cbrt(linear_jerk/(.98*self.max_linear_jerk_mps3)),
            np.cbrt(angular_jerk/(.98*self.max_angular_jerk_rps3)),
        ))
        self.knot_times = np.r_[0., np.cumsum(self.segment_durations)]
        self.motion_duration = float(self.knot_times[-1])
        self.duration = c.initial_hold_s+self.motion_duration
        self.gridpoints = np.unique((self.curve.parameters[:-1, None]
                                     + np.diff(self.curve.parameters)[:, None]*np.linspace(0., 1., 17)).ravel())
        probes = np.r_[(self.knot_times[:-1, None]+self.segment_durations[:, None]*np.linspace(0., 1., 65)).ravel(),
                       np.nextafter(self.knot_times[1:], self.knot_times[:-1])]
        state, acceleration, _, _ = self._sample_active(probes)
        jerk = self._sample_active_jerk(probes)
        self.validation = dict(
            valid=bool(np.isfinite(state).all() and np.isfinite(acceleration).all() and np.isfinite(jerk).all()),
            validation_point_count=len(probes), conservative_derivative_bounds=True,
            max_linear_speed_mps=float(np.linalg.norm(state[:, 3:6], axis=1).max()),
            max_angular_speed_rps=float(np.linalg.norm(state[:, 10:13], axis=1).max()),
            max_linear_acceleration_mps2=float(np.linalg.norm(acceleration[:, :3], axis=1).max()),
            max_angular_acceleration_rps2=float(np.linalg.norm(acceleration[:, 3:], axis=1).max()),
            estimated_max_linear_jerk_mps3=float(np.linalg.norm(jerk[:, :3], axis=1).max()),
            estimated_rms_linear_jerk_mps3=float(np.sqrt(np.mean(np.sum(jerk[:, :3]**2, axis=1)))),
            estimated_max_angular_jerk_rps3=float(np.linalg.norm(jerk[:, 3:], axis=1).max()),
            estimated_rms_angular_jerk_rps3=float(np.sqrt(np.mean(np.sum(jerk[:, 3:]**2, axis=1)))),
            progress_monotonic=True,
        )
        if not self.validation["valid"]:
            raise RuntimeError("intrinsic polyline time reference has non-finite kinematics")

    def _progress(self, times: Any) -> tuple[np.ndarray, ...]:
        times = np.clip(np.asarray(times, dtype=float), 0., self.motion_duration)
        index = np.minimum(np.searchsorted(self.knot_times, times, side="right")-1, len(self.segment_durations)-1)
        duration = self.segment_durations[index]
        u = np.clip((times-self.knot_times[index])/duration, 0., 1.)
        progress = u**3*(10+u*(-15+6*u))
        first = 30*u**2*(1-u)**2/duration
        second = 60*u*(1-u)*(1-2*u)/duration**2
        third = 60*(1-6*u+6*u**2)/duration**3
        divisor = len(self.poses)-1
        return (index+progress)/divisor, first/divisor, second/divisor, third/divisor, index, progress

    def _sample_active(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        parameter, speed, acceleration, _, index, progress = self._progress(times)
        position, rotation, first, second, _ = self.curve.edge_kinematics(index, progress)
        velocity = first*speed[:, None]
        feedforward = first*acceleration[:, None]+second*speed[:, None]**2
        state = np.column_stack((position, velocity[:, :3], _quaternions(rotation), velocity[:, 3:]))
        return state, feedforward, parameter, speed

    def _sample_active_jerk(self, times: np.ndarray) -> np.ndarray:
        _, speed, acceleration, jerk, index, progress = self._progress(times)
        _, _, first, second, third = self.curve.edge_kinematics(index, progress)
        return (third*speed[:, None]**3+3*second*(speed*acceleration)[:, None]+first*jerk[:, None])

    def audit_poses(self, max_spacing_m: float = .01, max_rotation_rad: float = .01) -> np.ndarray:
        if (not np.isfinite(max_spacing_m) or not np.isfinite(max_rotation_rad)
                or min(max_spacing_m, max_rotation_rad) <= 0):
            raise ValueError("碰撞审计采样间隔必须为正的有限数值")
        linear, _, _, angular, _, _ = self.curve.edge_derivative_bounds()
        counts = np.maximum(2, np.ceil(np.maximum(linear/(.8*max_spacing_m), angular/(.8*max_rotation_rad))).astype(int))
        parameters = np.concatenate([
            (index+np.linspace(0., 1., count, endpoint=False))/(len(self.poses)-1)
            for index, count in enumerate(counts)
        ]+[np.ones(1)])
        return self.curve.geometry(parameters)

    def describe(self) -> dict[str, Any]:
        c = self.config
        return dict(method=EXECUTION_POLYLINE_VERSION, geometry="analytic_intrinsic_linear_contact_lift",
                    waypoint_count=len(self.poses), source_node_count=len(self.poses), gridpoint_count=len(self.gridpoints),
                    duration_s=self.duration, motion_duration_s=self.motion_duration, initial_hold_s=c.initial_hold_s,
                    speed_mps=c.speed_mps, max_acceleration_mps2=c.max_acceleration_mps2,
                    max_angular_speed_rps=c.max_angular_speed_rps,
                    max_angular_acceleration_rps2=c.max_angular_acceleration_rps2,
                    max_linear_jerk_mps3=self.max_linear_jerk_mps3, max_angular_jerk_rps3=self.max_angular_jerk_rps3,
                    stops_at_waypoints=True, acceleration_continuous_in_motion=True,
                    geometry_smoothing=False, position_deviation_m=0., orientation_deviation_rad=0.,
                    differs_from_frozen_benchmark_edges=False, requires_smoothed_geometry_audit=False,
                    diagnostic_execution=self.allow_curvature_failure, contact_force_control=False,
                    contact_limitations=CONTACT_LIMITATION, kinematic_validation=self.validation.copy(),
                    corner_timing_policy="quintic rest-to-rest at every original intrinsic node",
                    timing_limitation="Stops at original nodes; does not traverse sharp corners at nonzero speed.")


def prepare_contact_polyline_reference(condition: Mapping[str, Any], intrinsic_states: Any,
                                       config: Any = None, *, allow_curvature_failure: bool = False) -> TimedContactPolylineReference:
    return TimedContactPolylineReference(condition, intrinsic_states, config,
                                         allow_curvature_failure=allow_curvature_failure)


def audit_contact_reference(condition: Mapping[str, Any], reference: TimedContactReference | TimedContactPolylineReference) -> dict[str, Any]:
    """Audit the selected exact geometry, including full URDF and task gates.

    No projection/refinement occurs here. This remains a finite sampling audit,
    not a continuous collision certificate or a claim of physical task success.
    Diagnostic execution can waive the maximum-curvature threshold only. Shape
    metrics and the formal quality result continue reporting their strict values.
    """
    spec = ContactProjectionSpec.from_condition(condition)
    if not isinstance(reference, (TimedContactReference, TimedContactPolylineReference)):
        raise ValueError("contact execution requires a task-bound contact reference")
    if _scene_key(condition, spec) != _scene_key(reference.condition, reference.contact_curve.spec):
        raise ValueError("contact reference scene differs from task condition")
    bound_spec = reference.contact_curve.spec
    for key in ("origin", "frame", "bounds", "tool_offset", "tool_rotation", "start", "goal",
                "forbidden", "attitude_limits"):
        expected_value, actual_value = getattr(spec, key), getattr(bound_spec, key)
        if expected_value.shape != actual_value.shape or not np.allclose(expected_value, actual_value, atol=1e-12, rtol=0):
            raise ValueError(f"contact reference {key} differs from task condition")
    if ((spec.workspace is None) != (bound_spec.workspace is None)
            or (spec.workspace is not None and not np.allclose(spec.workspace, bound_spec.workspace, atol=1e-12, rtol=0))):
        raise ValueError("contact reference workspace differs from task condition")
    # Audit the actual selected lift, including every original/spline node.
    poses = reference.audit_poses(min(.01, spec.validation_step_m), min(.01, spec.validation_step_rad))
    # Derive intrinsic coordinates from actual geometry independently of labels.
    from .mujoco_execution import _quaternion_matrix
    rotation = _quaternion_matrix(poses[:, 3:])
    path = np.column_stack((poses[:, :3], rotation[:, :, 0], rotation[:, :, 1]))
    states = _infer_states(path, rotation, spec)
    constraints = _constraint_metrics(path, rotation, states, spec)
    source = reference.contact_curve.geometry(np.linspace(0., 1., 128))
    source_r = _quaternion_matrix(source[:, 3:])
    shape = _shape_metrics(source[:, :3], source_r)
    source_intrinsic = reference.contact_curve.intrinsic(np.linspace(0., 1., 128))
    source_psi_large_steps = int((np.abs(np.diff(source_intrinsic[:, 2])) > np.pi).sum())
    source_cylinder_large_steps = (int((np.abs(np.diff(source_intrinsic[:, 0]))/spec.radius > np.pi).sum())
                                    if spec.kind == "cylinder" else 0)
    first, second = reference.contact_curve.derivatives(np.linspace(0., 1., 4097))
    speed = np.linalg.norm(first[:, :3], axis=1)
    curvature = np.divide(np.linalg.norm(np.cross(first[:, :3], second[:, :3]), axis=1), speed**3,
                          out=np.zeros_like(speed), where=speed > 1e-8)
    analytic_curvature = float(curvature.max(initial=0.))
    polyline = isinstance(reference, TimedContactPolylineReference)
    curvature_valid = (shape["maximum_curvature_inv_m"] <= 8.+1e-8 and analytic_curvature <= 8.+1e-8)
    other_shape_valid = (shape["cumulative_backtracking_m"] <= .25+1e-8
                         and shape["minimum_local_chord_arc_ratio"] >= .65-1e-8)
    formal_shape_valid = bool(curvature_valid and other_shape_valid)
    diagnostic = bool(reference.allow_curvature_failure)
    curvature_warning = bool(not curvature_valid)
    warnings = (["Maximum curvature exceeds 8/m; diagnostic flight preserves the failed quality result."]
                if curvature_warning else [])
    expected = lift_contact_intrinsic(np.stack((spec.start, spec.goal)), spec).numpy()
    endpoints_position = float(np.linalg.norm(path[[0, -1], :3]-expected[:, :3], axis=1).max())
    endpoints_angle = float(_rotation_angles(rotation[[0, -1]], _rotations(expected)).max())
    with _COAL_LOCK:
        checker = _scene_checker(_scene_key(condition, spec))
        collision_free = all(checker.is_collision_free(p[:3], p[3:]) for p in poses)
        backend = _backend()
    gates = {"full_urdf_collision": collision_free, "contact": constraints["practical_contact_gate"],
             "normal": constraints["practical_normal_gate"], "numeric_contact": constraints["numeric_contact_gate"],
             "numeric_normal": constraints["numeric_normal_gate"], "workspace": constraints["workspace_valid"],
             "attitude": constraints["attitude_valid"], "chart": constraints["chart_valid"],
             "tool_point": constraints["tool_point_collision_free"], "psi": constraints["psi_quality_gate"],
             "forbidden": constraints["forbidden_edge_violation_count"] == 0,
             "intrinsic_continuity": (constraints["psi_large_step_count"] == constraints["cylinder_large_step_count"]
                                      == source_psi_large_steps == source_cylinder_large_steps == 0),
             "shape": other_shape_valid and (curvature_valid or diagnostic),
             "endpoints": endpoints_position <= 2e-6 and endpoints_angle <= 2e-6}
    failures = [name for name, passed in gates.items() if not passed]
    if failures:
        raise ValueError("Contact execution reference failed: "+", ".join(failures))
    return dict(checked=True, valid=True, candidate_success=formal_shape_valid, physical_collision_checked=True,
                collision_backend="coal_full_urdf_contact_security_margin", urdf_sha256=backend[4],
                sample_count=len(poses), audit_type=("sampled exact intrinsic-linear edges, not continuous certification"
                                                   if polyline else "sampled exact intrinsic cubic, not continuous certification"),
                translation_step_m=min(.01, spec.validation_step_m), rotation_step_rad=min(.01, spec.validation_step_rad),
                endpoint_position_error_max_m=endpoints_position, endpoint_rotation_error_max_rad=endpoints_angle,
                analytic_curve_sampled_maximum_curvature_inv_m=analytic_curvature,
                analytic_curvature_scope=("edge interiors only; spatial corner curvature is undefined"
                                          if polyline else "sampled smooth curve"),
                continuous_corner_curvature_defined=not polyline,
                source_psi_large_step_count=source_psi_large_steps,
                source_cylinder_large_step_count=source_cylinder_large_steps,
                source_node_count=len(reference.poses), source_node_maximum_curvature_inv_m=shape["maximum_curvature_inv_m"],
                source_shape_metrics=copy.deepcopy(shape), execution_curve_shape_metrics=copy.deepcopy(shape),
                curvature_warning=curvature_warning, execution_warnings=warnings, warnings=warnings,
                diagnostic_execution=diagnostic, curvature_failure_allowed=diagnostic and curvature_warning,
                formal_shape_quality_gate=formal_shape_valid, execution_shape_gate=gates["shape"],
                gates=gates, shape_metrics=shape, contact_constraints=constraints, contact_limitations=CONTACT_LIMITATION)


def contact_execution_metrics(condition: Mapping[str, Any], executed_poses: Any) -> dict[str, Any]:
    """Measure actual sampled ideal-tool geometry independently of flight success."""
    result = dict(task_success=False, geometric_task_success=False, actual_sample_count=0,
                  contact_force_control=False, contact_limitations=CONTACT_LIMITATION,
                  physical_collision_checked=False, task_success_scope="sampled actual ideal-tool geometry",
                  sample_scope="control-update actual poses; environment collision checked every physics step",
                  max_contact_error_m=None, max_normal_error_deg=None,
                  tip_points=[], contact_error_m=[], normal_error_deg=[])
    poses = np.asarray(executed_poses, dtype=np.float64)
    if poses.size == 0:
        return result
    if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
        raise ValueError("actual contact poses must be finite [N,7]")
    norms = np.linalg.norm(poses[:, 3:], axis=1)
    if np.any(norms <= 1e-10):
        raise ValueError("actual contact pose has a zero quaternion")
    from .mujoco_execution import _quaternion_matrix
    rotation = _quaternion_matrix(poses[:, 3:]/norms[:, None])
    path = np.column_stack((poses[:, :3], rotation[:, :, 0], rotation[:, :, 1]))
    spec = ContactProjectionSpec.from_condition(condition)
    states = _infer_states(path, rotation, spec)
    constraints = _constraint_metrics(path, rotation, states, spec)
    tip = poses[:, :3]+rotation@spec.tool_offset
    uv = np.clip(states[:, :2], spec.bounds[:2, 0], spec.bounds[:2, 1])
    if spec.kind == "plane":
        local_point = np.column_stack((uv, np.zeros(len(uv))))
        normal = np.tile([0., 0., 1.], (len(uv), 1))
    else:
        angle = uv[:, 0]/spec.radius
        normal = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(uv))))
        local_point = spec.radius*normal+np.column_stack((np.zeros((len(uv), 2)), uv[:, 1]))
    nearest = local_point@spec.frame.T+spec.origin
    inward = -normal@spec.frame.T
    axis = rotation@spec.tool_rotation[:, 2]
    error = np.linalg.norm(tip-nearest, axis=1)
    angle_error = np.degrees(np.arctan2(np.linalg.norm(np.cross(axis, inward), axis=1),
                                      np.sum(axis*inward, axis=1)))
    success = bool(all(constraints[key] for key in ("practical_contact_gate", "practical_normal_gate",
                   "workspace_valid", "attitude_valid", "chart_valid", "tool_point_collision_free", "psi_quality_gate"))
                   and constraints["forbidden_edge_violation_count"] == 0
                   and constraints["psi_large_step_count"] == 0
                   and constraints["cylinder_large_step_count"] == 0)
    result.update(constraints, task_success=success, geometric_task_success=success,
                  actual_sample_count=len(poses), last_tool_position=tip[-1].tolist(),
                  max_contact_error_m=float(error.max()), max_normal_error_deg=float(angle_error.max()),
                  tip_points=tip.tolist(), contact_error_m=error.tolist(), normal_error_deg=angle_error.tolist())
    return result
