"""Independent contact evaluation on the actual predicted full-URDF poses.

Waypoint clearance is the Base-compatible 128-node COAL signed distance with
the configured margin already deducted. Successful waypoint candidates receive
dense collision/constraint checks, using <=2 cm / .02 rad steps and the same
COAL security-margin collision queries as the reference collector. Dense signed
distance is intentionally NOT measured: the reported minimum is waypoint-only.

Projection-off edges mean world XYZ linear interpolation plus shortest SO(3)
SLERP. Projection-on edges mean linear unwrapped intrinsic interpolation/lift;
the supplied intrinsic labels must first agree with the actual world prediction.
There is no expert route input, projection, or obstacle repair in this evaluator.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np
import torch

from .contact_projection import ContactProjectionSpec, lift_contact_intrinsic


METRICS_VERSION = "full_urdf_contact_actual_pose_v001"
PRACTICAL_POSITION_TOLERANCE_M = .001
PRACTICAL_NORMAL_TOLERANCE_RAD = math.pi/180
NUMERIC_POSITION_TOLERANCE_M = 2e-6
NUMERIC_NORMAL_TOLERANCE_RAD = 2e-6
_COAL_LOCK = threading.RLock()


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _rotations(path: np.ndarray) -> np.ndarray:
    a, b = path[:, 3:6], path[:, 6:9]
    na = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(na <= 1e-10) or not np.isfinite(na).all():
        raise ValueError("degenerate first rotation-6D column")
    x = a/na
    y = b-np.sum(x*b, axis=1, keepdims=True)*x
    ny = np.linalg.norm(y, axis=1, keepdims=True)
    if np.any(ny <= 1e-10) or not np.isfinite(ny).all():
        raise ValueError("degenerate second rotation-6D column")
    y = y/ny
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def _quaternions(r: np.ndarray) -> np.ndarray:
    """Vectorized stable matrix-to-wxyz conversion, including rotations of pi."""
    r = np.asarray(r)
    k = np.empty(r.shape[:-2]+(4, 4), dtype=np.float64)
    k[..., 0, 0] = r[..., 0, 0]-r[..., 1, 1]-r[..., 2, 2]
    k[..., 1, 1] = r[..., 1, 1]-r[..., 0, 0]-r[..., 2, 2]
    k[..., 2, 2] = r[..., 2, 2]-r[..., 0, 0]-r[..., 1, 1]
    k[..., 3, 3] = np.trace(r, axis1=-2, axis2=-1)
    k[..., 0, 1] = k[..., 1, 0] = r[..., 0, 1]+r[..., 1, 0]
    k[..., 0, 2] = k[..., 2, 0] = r[..., 0, 2]+r[..., 2, 0]
    k[..., 1, 2] = k[..., 2, 1] = r[..., 1, 2]+r[..., 2, 1]
    k[..., 0, 3] = k[..., 3, 0] = r[..., 2, 1]-r[..., 1, 2]
    k[..., 1, 3] = k[..., 3, 1] = r[..., 0, 2]-r[..., 2, 0]
    k[..., 2, 3] = k[..., 3, 2] = r[..., 1, 0]-r[..., 0, 1]
    _, vectors = np.linalg.eigh(k/3)
    q = vectors[..., :, -1][..., [3, 0, 1, 2]]
    return q*np.where(q[..., :1] < 0, -1., 1.)


def _rotation_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(a, -1, -2) @ b
    skew = np.stack((relative[..., 2, 1]-relative[..., 1, 2],
                     relative[..., 0, 2]-relative[..., 2, 0],
                     relative[..., 1, 0]-relative[..., 0, 1]), -1)
    return np.arctan2(np.linalg.norm(skew, axis=-1)/2,
                      (np.trace(relative, axis1=-2, axis2=-1)-1)/2)


@lru_cache(maxsize=1)
def _backend() -> tuple[Any, Any, Any, Path, str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    root = Path(os.environ.get("HNUTER_COLLECTOR_ROOT", str(Path(__file__).resolve().parents[2]/"HNUTER_trajectory_collector"))).resolve()
    if not (root/"coal_collision.py").is_file():
        raise FileNotFoundError(f"collector collision backend not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from coal_collision import CoalCollisionChecker, RigidURDFCollisionModel, StaticCollisionObject
    urdf = root/"expert_trajectory_collector/assets/HDJQR-0102-0055.SLDASM.urdf"
    model = RigidURDFCollisionModel.from_urdf(urdf)
    positions, rotations, half_extents = [], [], []
    from hnuter_dit.model import quaternion_wxyz_to_matrix_numpy
    for shape in model.geometries:
        parameters, kind = shape.geometry.parameters, shape.geometry.kind
        if kind == "box":
            half = np.asarray(parameters)/2
        elif kind == "sphere":
            half = np.repeat(parameters[0], 3)
        elif kind == "cylinder":
            half = np.array([parameters[0], parameters[0], parameters[1]/2])
        elif kind == "capsule":
            half = np.array([parameters[0], parameters[0], parameters[1]/2+parameters[0]])
        else:
            raise ValueError("workspace audit needs primitive URDF collision geometry")
        positions.append(shape.local_pose.position)
        rotations.append(quaternion_wxyz_to_matrix_numpy(shape.local_pose.quaternion))
        half_extents.append(half)
    return (model, CoalCollisionChecker, StaticCollisionObject, urdf,
            hashlib.sha256(urdf.read_bytes()).hexdigest(),
            (np.array(positions), np.array(rotations), np.array(half_extents)))


def _scene_key(condition: Mapping[str, Any], spec: ContactProjectionSpec) -> str:
    # Explicit input whitelist: no condition serialization or oracle metadata.
    g = condition["surface_geometry"]
    objects = []
    for index, box in enumerate([*condition.get("environment", {}).get("obstacles", []), *g.get("supports", [])]):
        objects.append({"id": str(box.get("id", f"box_{index}")), "size": _numpy(box["size"]).tolist(),
                        "position": _numpy(box["position"]).tolist(),
                        "quaternion_wxyz": _numpy(box["quaternion_wxyz"]).tolist()})
    thickness = float(g.get("thickness_m", .2))
    if not math.isfinite(thickness) or thickness <= 0:
        raise ValueError("surface thickness must be finite and positive")
    return json.dumps({"kind": spec.kind, "origin": spec.origin.tolist(), "frame": spec.frame.tolist(),
                       "bounds_uv": spec.bounds[:2].tolist(), "radius": spec.radius,
                       "thickness": thickness, "margin": spec.collision_margin, "objects": objects},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


@lru_cache(maxsize=8)
def _scene_checker(key: str) -> Any:
    model, Checker, Object, _, _, _ = _backend()
    value = json.loads(key)
    origin, frame, bounds = np.array(value["origin"]), np.array(value["frame"]), np.array(value["bounds_uv"])
    q = _quaternions(frame)
    if value["kind"] == "plane":
        local = np.r_[bounds.mean(axis=1), -value["thickness"]/2]
        target = Object.box("contact_surface", [*np.ptp(bounds, axis=1), value["thickness"]], origin+frame@local, q)
    else:
        local = np.array([0., 0., bounds[1].mean()])
        target = Object.cylinder("contact_surface", value["radius"], float(np.ptp(bounds[1])), origin+frame@local, q)
    boxes = [Object.box(box["id"], box["size"], box["position"], box["quaternion_wxyz"]) for box in value["objects"]]
    # Every active body primitive collides with the target, attached target box,
    # supports and original environment; the ideal point exemption is separate.
    return Checker(model, [target, *boxes], safety_margin=value["margin"])


@lru_cache(maxsize=8192)
def _exact_pose_clearance(scene_key: str, position_bytes: bytes, quaternion_bytes: bytes) -> float:
    """Reuse only bit-identical physical poses in an identical immutable scene."""
    checker = _scene_checker(scene_key)
    position = np.frombuffer(position_bytes, dtype=np.float64)
    quaternion = np.frombuffer(quaternion_bytes, dtype=np.float64)
    return float(checker.check_pose(position, quaternion).signed_clearance)


def _shape_metrics(position: np.ndarray, rotation: np.ndarray) -> dict[str, Any]:
    step = np.diff(position, axis=0)
    length = np.linalg.norm(step, axis=1)
    usable = (length[:-1] > 1e-8) & (length[1:] > 1e-8)
    angles = np.arccos(np.clip(np.sum(step[:-1][usable]*step[1:][usable], axis=1)/(length[:-1][usable]*length[1:][usable]), -1., 1.))
    curvature = 2*angles/(length[:-1][usable]+length[1:][usable])
    direction = position[-1]-position[0]
    separation = float(np.linalg.norm(direction))
    forward = step@(direction/max(separation, 1e-9))
    cumulative = np.r_[0., np.cumsum(length)]
    chord = np.linalg.norm(position[16:]-position[:-16], axis=1)
    arc = cumulative[16:]-cumulative[:-16]
    ratios = np.divide(chord, arc, out=np.ones_like(chord), where=arc > 1e-8)
    turns_deg = np.degrees(angles)
    rotations = _rotation_angles(rotation[:-1], rotation[1:])
    result = {"path_length_m": float(length.sum()), "path_efficiency": float(length.sum()/max(separation, 1e-9)),
              "p95_turn_angle_deg": float(np.percentile(turns_deg, 95)) if len(turns_deg) else 0.,
              "maximum_turn_angle_deg": float(turns_deg.max(initial=0.)),
              "maximum_curvature_inv_m": float(curvature.max(initial=0.)),
              "minimum_local_chord_arc_ratio": float(ratios.min(initial=1.)), "local_chord_window_intervals": 16,
              "cumulative_backtracking_m": float(-np.minimum(forward, 0).sum()),
              "backtracking_fraction": float(np.mean(forward < -1e-10)),
              "acceleration_rms_m_per_index2": float(np.sqrt(np.mean(np.diff(position, n=2, axis=0)**2))),
              "jerk_rms_m_per_index3": float(np.sqrt(np.mean(np.diff(position, n=3, axis=0)**2))),
              "rotation_total_variation_rad": float(rotations.sum()), "maximum_rotation_step_rad": float(rotations.max(initial=0.)),
              "maximum_position_step_m": float(length.max(initial=0.)), "turn_angle_policy": "ignore_zero_translation_segments"}
    result["shape_quality_gate"] = bool(result["maximum_curvature_inv_m"] <= 8.+1e-8 and
                                        result["cumulative_backtracking_m"] <= .25+1e-8 and
                                        result["minimum_local_chord_arc_ratio"] >= .65-1e-8)
    return result


def _infer_states(path: np.ndarray, rotation: np.ndarray, spec: ContactProjectionSpec) -> np.ndarray:
    tip = path[:, :3] + rotation@spec.tool_offset
    local = (tip-spec.origin)@spec.frame
    if spec.kind == "plane":
        u, v = local[:, 0], local[:, 1]
        tangent = np.broadcast_to([1., 0., 0.], local.shape)
        normal = np.broadcast_to([0., 0., 1.], local.shape)
    else:
        radial = np.linalg.norm(local[:, :2], axis=1)
        if np.any(radial <= 1e-10):
            raise ValueError("predicted tool point is on the cylinder axis; chart undefined")
        angle = np.arctan2(local[:, 1], local[:, 0])
        angle += 2*np.pi*np.round((spec.start[0]/spec.radius-angle[0])/(2*np.pi))
        angle = np.unwrap(angle)
        u, v = spec.radius*angle, local[:, 2]
        normal = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(path))))
        tangent = np.column_stack((-np.sin(angle), np.cos(angle), np.zeros(len(path))))
    x, z = tangent@spec.frame.T, -normal@spec.frame.T
    y = np.cross(z, x)
    tool = rotation@spec.tool_rotation
    a = np.sum(x*tool[:, :, 0], axis=1)+np.sum(y*tool[:, :, 1], axis=1)
    b = np.sum(y*tool[:, :, 0], axis=1)-np.sum(x*tool[:, :, 1], axis=1)
    if np.any(a*a+b*b <= 1e-20):
        raise ValueError("predicted tool normal is antipodal; contact twist undefined")
    psi = np.arctan2(b, a)
    psi += 2*np.pi*np.round((spec.start[2]-psi[0])/(2*np.pi))
    psi = np.unwrap(psi)
    return np.column_stack((u, v, psi))


def _forbidden_edges(states: np.ndarray, spec: ContactProjectionSpec) -> int:
    a, delta = states[:-1, :2], np.diff(states[:, :2], axis=0)
    hit = np.zeros(len(a), dtype=bool)
    for f in spec.forbidden:
        low, high = np.zeros(len(a)), np.ones(len(a))
        possible = np.ones(len(a), dtype=bool)
        for axis, lo, hi in ((0, f[0], f[1]), (1, f[2], f[3])):
            parallel = np.abs(delta[:, axis]) < 1e-14
            possible &= ~parallel | ((a[:, axis] >= lo) & (a[:, axis] <= hi))
            safe = np.where(parallel, 1., delta[:, axis])
            t0, t1 = (lo-a[:, axis])/safe, (hi-a[:, axis])/safe
            low = np.maximum(low, np.where(parallel, -np.inf, np.minimum(t0, t1)))
            high = np.minimum(high, np.where(parallel, np.inf, np.maximum(t0, t1)))
        hit |= possible & (low <= high)
    return int(hit.sum())


def _constraint_metrics(path: np.ndarray, rotation: np.ndarray, states: np.ndarray,
                        spec: ContactProjectionSpec) -> dict[str, Any]:
    tip = path[:, :3]+rotation@spec.tool_offset
    uv = np.clip(states[:, :2], spec.bounds[:2, 0], spec.bounds[:2, 1])
    if spec.kind == "plane":
        local_point = np.column_stack((uv, np.zeros(len(path))))
        local_normal = np.broadcast_to([0., 0., 1.], local_point.shape)
    else:
        angle = uv[:, 0]/spec.radius
        local_normal = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(path))))
        local_point = spec.radius*local_normal + np.column_stack((np.zeros((len(path), 2)), uv[:, 1]))
    point, normal = local_point@spec.frame.T+spec.origin, local_normal@spec.frame.T
    axis = rotation@spec.tool_rotation[:, 2]
    contact_error = np.linalg.norm(tip-point, axis=1)
    normal_error = np.arctan2(np.linalg.norm(np.cross(axis, -normal), axis=1), np.sum(axis*-normal, axis=1))
    roll = np.degrees(np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2]))
    pitch = np.degrees(np.arctan2(-rotation[:, 2, 0], np.hypot(rotation[:, 0, 0], rotation[:, 1, 0])))
    attitude = np.column_stack((roll, pitch))
    attitude_valid = np.all((attitude >= spec.attitude_limits[:, 0]-1e-6) & (attitude <= spec.attitude_limits[:, 1]+1e-6), axis=1)
    tool_safe = np.ones(len(path), dtype=bool)
    for center, frame, half in spec.boxes:
        tool_safe &= ~np.all(np.abs((tip-center)@frame) <= half+spec.collision_margin, axis=1)
    chart_valid = np.all((states >= spec.bounds[:, 0]-1e-9) & (states <= spec.bounds[:, 1]+1e-9), axis=1)
    for f in spec.forbidden:
        chart_valid &= ~((states[:, 0] >= f[0]) & (states[:, 0] <= f[1]) & (states[:, 1] >= f[2]) & (states[:, 1] <= f[3]))
    _, _, _, _, _, (local_positions, local_rotations, half) = _backend()
    workspace_valid = np.ones(len(path), dtype=bool)
    if spec.workspace is not None:
        centers = path[:, None, :3] + np.einsum("nij,gj->ngi", rotation, local_positions)
        shape_rotation = np.einsum("nij,gjk->ngik", rotation, local_rotations)
        extent = np.einsum("ngij,gj->ngi", np.abs(shape_rotation), half)
        workspace_valid = np.all((centers-extent >= spec.workspace[0]-1e-8) & (centers+extent <= spec.workspace[1]+1e-8), axis=(1, 2))
        workspace_valid &= np.all((tip >= spec.workspace[0]-1e-8) & (tip <= spec.workspace[1]+1e-8), axis=1)
    psi = states[:, 2]
    tv = float(np.abs(np.diff(psi)).sum())
    endpoint_delta = float(abs(psi[-1]-psi[0]))
    lo, hi = float(psi.min()), float(psi.max())
    minimum_tv = hi-lo+min(abs(psi[0]-lo)+abs(psi[-1]-hi), abs(psi[0]-hi)+abs(psi[-1]-lo))
    psi_excess = max(0., tv-minimum_tv)
    result = {"contact_error_max_m": float(contact_error.max()), "contact_error_mean_m": float(contact_error.mean()),
              "normal_error_max_rad": float(normal_error.max()), "normal_error_mean_rad": float(normal_error.mean()),
              "practical_contact_gate": bool(np.all(contact_error <= PRACTICAL_POSITION_TOLERANCE_M)),
              "practical_normal_gate": bool(np.all(normal_error <= PRACTICAL_NORMAL_TOLERANCE_RAD)),
              "numeric_contact_gate": bool(np.all(contact_error <= NUMERIC_POSITION_TOLERANCE_M)),
              "numeric_normal_gate": bool(np.all(normal_error <= NUMERIC_NORMAL_TOLERANCE_RAD)),
              "attitude_valid": bool(attitude_valid.all()), "attitude_violation_sample_count": int((~attitude_valid).sum()),
              "maximum_abs_roll_deg": float(np.abs(roll).max()), "maximum_abs_pitch_deg": float(np.abs(pitch).max()),
              "workspace_valid": bool(workspace_valid.all()), "workspace_violation_sample_count": int((~workspace_valid).sum()),
              "tool_point_collision_free": bool(tool_safe.all()), "tool_point_violation_sample_count": int((~tool_safe).sum()),
              "chart_valid": bool(chart_valid.all()), "chart_violation_sample_count": int((~chart_valid).sum()),
              "forbidden_edge_violation_count": _forbidden_edges(states, spec),
              "psi_total_variation_rad": tv, "psi_endpoint_delta_rad": endpoint_delta,
              "psi_excess_reversal_variation_rad": psi_excess,
              "psi_quality_gate": bool(tv <= endpoint_delta+7*np.pi/6+1e-9 and psi_excess <= np.pi/6+1e-9),
              "psi_large_step_count": int((np.abs(np.diff(psi)) > np.pi).sum()),
              "cylinder_large_step_count": int((np.abs(np.diff(states[:, 0]))/spec.radius > np.pi).sum()) if spec.kind == "cylinder" else 0}
    return result


def _dense_path(path: np.ndarray, rotation: np.ndarray, states: np.ndarray,
                spec: ContactProjectionSpec, projected: bool) -> tuple[np.ndarray, np.ndarray]:
    step_m, step_rad = min(.02, spec.validation_step_m), min(.02, spec.validation_step_rad)
    q = _quaternions(rotation)
    angle = _rotation_angles(rotation[:-1], rotation[1:])
    _, _, _, _, _, (local_positions, _, half) = _backend()
    body_radius = float(np.max(np.linalg.norm(local_positions, axis=1)+np.linalg.norm(half, axis=1)))
    paths, intrinsic = [], []
    for index in range(len(path)-1):
        if projected:
            d = np.abs(states[index+1]-states[index])
            turning = d[2]+(d[0]/spec.radius if spec.kind == "cylinder" else 0.)
            travel = np.linalg.norm(d[:2])+(np.linalg.norm(spec.tool_offset)+body_radius)*turning
        else:
            turning = angle[index]
            travel = np.linalg.norm(path[index+1, :3]-path[index, :3])+max(body_radius, np.linalg.norm(spec.tool_offset))*turning
        count = max(1, int(np.ceil(travel/step_m)), int(np.ceil(turning/step_rad)))
        if count > 100000:
            raise ValueError("edge exceeds dense validation budget")
        t = np.arange(count)/count
        if projected:
            interpolated = states[index]+t[:, None]*(states[index+1]-states[index])
            pose = lift_contact_intrinsic(interpolated, spec).numpy()
        else:
            a, b = q[index], q[index+1]
            dot = float(np.dot(a, b))
            if dot < 0:
                b, dot = -b, -dot
            theta = np.arccos(np.clip(dot, -1., 1.))
            if theta < 1e-8:
                quaternion = np.repeat(a[None], count, axis=0)
            else:
                quaternion = (np.sin((1-t)*theta)[:, None]*a+np.sin(t*theta)[:, None]*b)/np.sin(theta)
                quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
            from .model import quaternion_wxyz_to_matrix_numpy
            r = quaternion_wxyz_to_matrix_numpy(quaternion)
            position = path[index, :3]+t[:, None]*(path[index+1, :3]-path[index, :3])
            pose = np.column_stack((position, r[:, :, 0], r[:, :, 1]))
            interpolated = _infer_states(pose, r, spec)
        paths.append(pose)
        intrinsic.append(interpolated)
    # On mode uses exact intrinsic lift even at the last point; actual serialized
    # nodes were separately checked, and binding errors cannot bypass the gate.
    paths.append(lift_contact_intrinsic(states[-1:], spec).numpy() if projected else path[-1:])
    intrinsic.append(states[-1:])
    return np.concatenate(paths), np.concatenate(intrinsic)


def evaluate_contact_path(world_pose9: Any, input_condition: Mapping[str, Any], *,
                          projected_intrinsic: Any | None = None) -> dict[str, Any]:
    """Evaluate exactly 128 actual world pose9 nodes; return failures, never repair.

    ``candidate_success`` uses practical 1 mm / 1 degree contact constraints.
    ``numeric_candidate_success`` additionally requires 2e-6 m/rad, including
    dense edges. Endpoint identity and projection-label binding use 2e-6.
    Shape gates mirror the contact campaign's discrete 8/m, .25m backtracking
    and .65 sixteen-interval chord/arc thresholds; jerk has no invented cutoff.
    Backend/invalid-input failures return a failed record instead of aborting a
    batch. Callers should distinguish ``evaluation_error`` from model failure.
    """
    started = time.perf_counter()
    result: dict[str, Any] = {"schema_version": METRICS_VERSION, "candidate_success": False,
        "numeric_candidate_success": False, "valid": False, "physical_collision_free": False,
        "physical_collision_checked": False, "waypoint_count": 0, "minimum_clearance_m": None,
        "dense_checked": False, "dense_validation_completed": False, "dense_physical_collision_free": None,
        "dense_clearance_sampled": False, "clearance_dense_sampled": False, "minimum_clearance_dense_m": None,
        "minimum_clearance_scope": "all_128_waypoints_only", "clearance_margin_already_deducted": True,
        "dense_physical_checked_samples": 0, "dense_planned_samples": 0,
        "edge_decoder": "intrinsic_linear_unwrapped_lift" if projected_intrinsic is not None else "world_xyz_linear_shortest_so3_slerp",
        "contact_tolerances": {"practical_position_m": .001, "practical_normal_rad": math.pi/180,
                               "numeric_position_m": 2e-6, "numeric_normal_rad": 2e-6},
        "failures": [], "evaluation_error": None}
    try:
        path = _numpy(world_pose9)
        if path.shape != (128, 9) or not np.isfinite(path).all():
            raise ValueError("world_pose9 must be finite [128,9]; no implicit resampling")
        if np.abs(path[:, :3]).max() > 1e6:
            raise ValueError("world coordinate magnitude exceeds safe collision-query domain")
        result["waypoint_count"] = 128
        rotation = _rotations(path)
        spec = ContactProjectionSpec.from_condition(input_condition)
        with _COAL_LOCK:
            scene_key = _scene_key(input_condition, spec)
            checker = _scene_checker(scene_key)
            model, _, _, urdf, sha, _ = _backend()
            # Always inspect all physical poses, even when contact or shape fails.
            cached_before = _exact_pose_clearance.cache_info().hits
            clearance = np.array([_exact_pose_clearance(scene_key, position.tobytes(), quaternion.tobytes())
                                  for position, quaternion in zip(path[:, :3], _quaternions(rotation))])
            result["waypoint_clearance_exact_cache_hits"] = _exact_pose_clearance.cache_info().hits-cached_before
            result["waypoint_clearance_requested_samples"] = 128
        if clearance.shape != (128,) or not np.isfinite(clearance).all():
            raise ValueError("COAL returned invalid waypoint clearances")
        result.update(physical_collision_checked=True, physical_collision_free=bool(np.all(clearance > 0)),
                      minimum_clearance_m=float(clearance.min()), collision_violation_sample_count=int(np.count_nonzero(clearance <= 0)),
                      clearance_m=clearance.tolist(), urdf_path=str(urdf), urdf_sha256=sha,
                      urdf_collision_geometry_count=len(model.geometries), environment_collision_object_count=checker.environment_object_count,
                      collision_margin_m=spec.collision_margin)
        if projected_intrinsic is None:
            states = _infer_states(path, rotation, spec)
            binding_valid = True
            result.update(projection_binding_position_error_max_m=None, projection_binding_rotation_error_max_rad=None)
        else:
            states = _numpy(projected_intrinsic)
            if states.shape != (128, 3) or not np.isfinite(states).all():
                raise ValueError("projected_intrinsic must be finite [128,3]")
            lifted = lift_contact_intrinsic(states, spec).numpy()
            binding_position = float(np.linalg.norm(path[:, :3]-lifted[:, :3], axis=1).max())
            binding_rotation = float(_rotation_angles(rotation, _rotations(lifted)).max())
            intrinsic_endpoint = float(np.max(np.abs(states[[0, -1]]-np.stack((spec.start, spec.goal)))))
            binding_valid = binding_position <= 2e-6 and binding_rotation <= 2e-6 and intrinsic_endpoint <= 2e-6
            result.update(projection_binding_position_error_max_m=binding_position,
                          projection_binding_rotation_error_max_rad=binding_rotation,
                          projection_intrinsic_endpoint_error_max=intrinsic_endpoint)
        result["projection_binding_valid"] = bool(binding_valid)
        result.update(_constraint_metrics(path, rotation, states, spec))
        result.update(_shape_metrics(path[:, :3], rotation))
        expected = lift_contact_intrinsic(np.stack((spec.start, spec.goal)), spec).numpy()
        endpoint_position = float(np.linalg.norm(path[[0, -1], :3]-expected[:, :3], axis=1).max())
        endpoint_rotation = float(_rotation_angles(rotation[[0, -1]], _rotations(expected)).max())
        result.update(endpoint_position_error_max_m=endpoint_position, endpoint_rotation_error_max_rad=endpoint_rotation,
                      endpoint_gate=bool(endpoint_position <= 2e-6 and endpoint_rotation <= 2e-6))
        gates = {"physical_collision": result["physical_collision_free"], "contact": result["practical_contact_gate"],
                 "normal": result["practical_normal_gate"], "attitude": result["attitude_valid"],
                 "workspace": result["workspace_valid"], "tool_point_collision": result["tool_point_collision_free"],
                 "chart": result["chart_valid"], "forbidden_edge": result["forbidden_edge_violation_count"] == 0,
                 "psi_quality": result["psi_quality_gate"], "psi_continuity": result["psi_large_step_count"] == 0,
                 "cylinder_chart_continuity": result["cylinder_large_step_count"] == 0,
                 "endpoint": result["endpoint_gate"], "shape_quality": result["shape_quality_gate"], "projection_binding": binding_valid}
        result["failures"] = [name for name, passed in gates.items() if not passed]
        result["waypoint_candidate_success"] = bool(all(gates.values()))
        result["waypoint_numeric_candidate_success"] = bool(result["waypoint_candidate_success"] and result["numeric_contact_gate"] and result["numeric_normal_gate"])
        if result["waypoint_candidate_success"]:
            dense_path, dense_states = _dense_path(path, rotation, states, spec, projected_intrinsic is not None)
            dense_rotation = _rotations(dense_path)
            dense = _constraint_metrics(dense_path, dense_rotation, dense_states, spec)
            result.update({"dense_"+name: value for name, value in dense.items()})
            result.update(dense_checked=True, dense_planned_samples=len(dense_path),
                          dense_step_m=min(.02, spec.validation_step_m), dense_step_rad=min(.02, spec.validation_step_rad))
            dense_free = True
            with _COAL_LOCK:
                for position, quaternion in zip(dense_path[:, :3], _quaternions(dense_rotation)):
                    result["dense_physical_checked_samples"] += 1
                    if not checker.is_collision_free(position, quaternion):
                        dense_free = False
                        break
            result.update(dense_physical_collision_free=dense_free,
                          dense_validation_completed=bool(result["dense_physical_checked_samples"] == len(dense_path)))
            dense_gates = {"dense_physical_collision": dense_free, "dense_contact": dense["practical_contact_gate"],
                           "dense_normal": dense["practical_normal_gate"], "dense_attitude": dense["attitude_valid"],
                           "dense_workspace": dense["workspace_valid"], "dense_tool_point_collision": dense["tool_point_collision_free"],
                           "dense_chart": dense["chart_valid"], "dense_forbidden_edge": dense["forbidden_edge_violation_count"] == 0}
            result["failures"].extend(name for name, passed in dense_gates.items() if not passed)
            result["candidate_success"] = bool(all(dense_gates.values()))
            result["numeric_candidate_success"] = bool(result["candidate_success"] and result["waypoint_numeric_candidate_success"] and dense["numeric_contact_gate"] and dense["numeric_normal_gate"])
        else:
            result["dense_skip_reason"] = "a_128_waypoint_success_gate_already_failed"
        result["valid"] = result["candidate_success"]
    except Exception as error:
        result["evaluation_error"] = f"{type(error).__name__}: {error}"
        result["failures"].append("invalid_input_or_backend_error")
        result["candidate_success"] = result["numeric_candidate_success"] = result["valid"] = False
    result["evaluation_time_s"] = time.perf_counter()-started
    return result
