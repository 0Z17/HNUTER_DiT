"""Analytic, differentiable contact decoding for metric/normalized v003 pose9.

This decoder uses only task geometry and calibrated body-to-tool extrinsics.
It does not plan, smooth toward an expert, or repair obstacle detours. A batch
contains different samples of ONE task. Use one spec per task for mixed batches.

The continuous output is ``lift_contact_intrinsic(linear_interpolation(states))``.
Linear interpolation of world XYZ/pose9 is NOT a contact-preserving decoder on a
cylinder. ``geometry_valid`` covers chart, equality, attitude, ideal-tool-point
OBB and workspace checks; full URDF collisions, dynamics, real tool geometry,
forces, friction, and reachability still require independent validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .geometry import canonicalize_pose9_batch, decanonicalize_pose9_batch
from .model import quaternion_wxyz_to_matrix_numpy


PROJECTION_VERSION = "analytic_tool_contact_pose9_v001"
INTRINSIC_INTERPOLATION = "linear_unwrapped_nonperiodic_chart_v001"
SERIALIZATION_POSITION_TOLERANCE_M = 2e-6
SERIALIZATION_NORMAL_TOLERANCE_RAD = 2e-6


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return result.copy()


def _rotation(value: Any, name: str) -> np.ndarray:
    q = _array(value, (4,), name)
    if abs(np.linalg.norm(q) - 1.0) > 1e-6:
        raise ValueError(f"{name} must be a unit wxyz quaternion")
    return quaternion_wxyz_to_matrix_numpy(q)


@dataclass(frozen=True)
class ContactProjectionSpec:
    kind: str
    origin: np.ndarray
    frame: np.ndarray
    bounds: np.ndarray
    tool_offset: np.ndarray
    tool_rotation: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    radius: float
    forbidden: np.ndarray
    attitude_limits: np.ndarray
    boxes: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    workspace: np.ndarray | None
    position_tolerance: float
    normal_tolerance: float
    collision_margin: float
    validation_step_m: float
    validation_step_rad: float

    @classmethod
    def from_condition(cls, condition: Mapping[str, Any]) -> "ContactProjectionSpec":
        """Read geometry/limits only; route certificates and labels are ignored."""
        g, chart = condition["surface_geometry"], condition["surface_chart"]
        kind = g["type"]
        if kind not in ("plane", "cylinder"):
            raise ValueError("contact projection supports plane and cylinder")
        bounds = _array(chart["bounds"], (3, 2), "chart bounds")
        radius = float(g.get("radius_m", 1.0))
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be positive and finite")
        if np.any(bounds[:, 0] >= bounds[:, 1]) or np.ptp(bounds[2]) > 2*np.pi+1e-12:
            raise ValueError("chart intervals must increase; psi span must be <=2*pi")
        if kind == "cylinder" and np.ptp(bounds[0]) >= 2*np.pi*radius:
            raise ValueError("cylinder chart must be open and shorter than a circumference")
        forbidden = np.asarray(chart.get("forbidden_regions", []), dtype=np.float64).reshape(-1, 4)
        if not np.isfinite(forbidden).all() or np.any(forbidden[:, [0, 2]] >= forbidden[:, [1, 3]]):
            raise ValueError("forbidden rectangles must have finite increasing bounds")
        tool, limits = condition["tool_extrinsics"], condition["contact_constraints"]
        angles = limits.get("attitude_limits_deg", {"roll": [-40, 40], "pitch": [-70, 70]})
        attitude = _array([angles["roll"], angles["pitch"]], (2, 2), "attitude limits")
        if (np.any(attitude[:, 0] >= attitude[:, 1]) or
                np.any(attitude[:, 0] < [-40, -70]) or np.any(attitude[:, 1] > [40, 70])):
            raise ValueError("attitude limits must remain within roll +/-40 and pitch +/-70 degrees")
        policy = [float(limits.get(k, default)) for k, default in (
            ("position_tolerance_m", 1e-7), ("normal_tolerance_rad", 1e-6),
            ("collision_margin_m", .01), ("validation_step_m", .02), ("validation_step_rad", .02))]
        if not np.isfinite(policy).all() or min(policy[:2]) <= 0 or policy[2] < 0 or not all(.001 <= x <= .05 for x in policy[3:]):
            raise ValueError("invalid contact tolerances, margin, or dense-validation steps")
        start = _array(condition["start_goal"]["start"], (3,), "start intrinsic")
        goal = _array(condition["start_goal"]["goal"], (3,), "goal intrinsic")
        for state in (start, goal):
            if np.any(state < bounds[:, 0]) or np.any(state > bounds[:, 1]):
                raise ValueError("task endpoints must lie in the chart")
            if any(f[0] <= state[0] <= f[1] and f[2] <= state[1] <= f[3] for f in forbidden):
                raise ValueError("task endpoints lie in a forbidden rectangle")
        environment = condition.get("environment", {})
        boxes = []
        for box in [*environment.get("obstacles", []), *g.get("supports", [])]:
            if box.get("type", "box") != "box":
                raise ValueError("ideal-tool-point checks support OBB boxes only")
            # Only the designated ideal point is exempt from its attached target.
            if box.get("id") is not None and box.get("id") == g.get("target_obstacle_id"):
                continue
            size = _array(box["size"], (3,), "box size")
            if np.any(size <= 0):
                raise ValueError("box sizes must be positive")
            boxes.append((_array(box["position"], (3,), "box center"),
                          _rotation(box["quaternion_wxyz"], "box rotation"), size/2))
        workspace = environment.get("bounds")
        if workspace is not None:
            workspace = _array([workspace["min"], workspace["max"]], (2, 3), "workspace")
            if np.any(workspace[0] >= workspace[1]):
                raise ValueError("workspace intervals must increase")
        return cls(kind, _array(g["origin"], (3,), "surface origin"),
                   _rotation(g["quaternion_wxyz"], "surface rotation"), bounds,
                   _array(tool["position_body"], (3,), "tool offset"),
                   _rotation(tool["quaternion_wxyz"], "tool rotation"), start, goal,
                   radius, forbidden.copy(), attitude, tuple(boxes), workspace, *policy)


@dataclass
class ContactProjectionResult:
    unprojected_pose9: Tensor
    projected_pose9: Tensor
    world_pose9: Tensor
    intrinsic_states: Tensor
    before: dict[str, Tensor]
    after: dict[str, Tensor]
    geometry_valid: Tensor
    full_body_collision_checked: bool = False


def _constant(value: Any, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _safe_rotations(pose9: Tensor) -> tuple[Tensor, Tensor]:
    """Gram-Schmidt with deterministic finite fallbacks for degenerate 6D input."""
    a, b = pose9[..., 3:6], pose9[..., 6:9]
    na = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    first = torch.where(na > 1e-10, a/na.clamp_min(1e-10), _constant([1., 0., 0.], a))
    second_raw = b - (b*first).sum(-1, keepdim=True)*first
    nb = torch.linalg.vector_norm(second_raw, dim=-1, keepdim=True)
    axis = torch.nn.functional.one_hot(first.abs().argmin(-1), 3).to(first.dtype)
    fallback = axis - (axis*first).sum(-1, keepdim=True)*first
    fallback = fallback/torch.linalg.vector_norm(fallback, dim=-1, keepdim=True).clamp_min(1e-10)
    second = torch.where(nb > 1e-10, second_raw/nb.clamp_min(1e-10), fallback)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), -1), ((na[..., 0] <= 1e-10) | (nb[..., 0] <= 1e-10))


def _contact_basis(uv: Tensor, spec: ContactProjectionSpec) -> tuple[Tensor, Tensor, Tensor]:
    u, v = uv.unbind(-1)
    zero, one = torch.zeros_like(u), torch.ones_like(u)
    if spec.kind == "plane":
        point = torch.stack((u, v, zero), -1)
        normal = torch.stack((zero, zero, one), -1)
        tangent = torch.stack((one, zero, zero), -1)
    else:
        angle = u/spec.radius
        normal = torch.stack((torch.cos(angle), torch.sin(angle), zero), -1)
        point = spec.radius*normal + torch.stack((zero, zero, v), -1)
        tangent = torch.stack((-torch.sin(angle), torch.cos(angle), zero), -1)
    frame, origin = _constant(spec.frame, uv), _constant(spec.origin, uv)
    point = point @ frame.T + origin
    normal, tangent = normal @ frame.T, tangent @ frame.T
    inward = -normal
    basis = torch.stack((tangent, torch.cross(inward, tangent, dim=-1), inward), -1)
    return point, normal, basis


def lift_contact_intrinsic(states: Tensor | np.ndarray, spec: ContactProjectionSpec) -> Tensor:
    """Lift [...,3] unwrapped intrinsic states, including interpolated edge states."""
    states = torch.as_tensor(states)
    if not states.is_floating_point():
        states = states.to(torch.float64)
    if states.shape[-1:] != (3,) or not bool(torch.isfinite(states).all()):
        raise ValueError("intrinsic states must be finite [...,3]")
    point, _, basis = _contact_basis(states[..., :2], spec)
    psi = states[..., 2]
    x = basis[..., :, 0]*torch.cos(psi)[..., None] + basis[..., :, 1]*torch.sin(psi)[..., None]
    z = basis[..., :, 2]
    tool = torch.stack((x, torch.cross(z, x, dim=-1), z), -1)
    body = tool @ _constant(spec.tool_rotation.T, states)
    position = point - (body @ _constant(spec.tool_offset, states)[..., None]).squeeze(-1)
    return torch.cat((position, body[..., :, 0], body[..., :, 1]), -1)


def _uv_from_tip(tip: Tensor, spec: ContactProjectionSpec) -> tuple[Tensor, Tensor]:
    local = (tip-_constant(spec.origin, tip)) @ _constant(spec.frame, tip)
    if spec.kind == "plane":
        uv, degenerate = local[..., :2], torch.zeros_like(local[..., 0], dtype=torch.bool)
    else:
        radial = torch.linalg.vector_norm(local[..., :2], dim=-1)
        degenerate = radial <= 1e-10
        safe_x = torch.where(degenerate, torch.ones_like(radial), local[..., 0])
        safe_y = torch.where(degenerate, torch.zeros_like(radial), local[..., 1])
        angle = torch.atan2(safe_y, safe_x)
        midpoint = float(np.mean(spec.bounds[0])/spec.radius)
        angle = angle + 2*np.pi*torch.round((midpoint-angle)/(2*np.pi))
        u = torch.where(degenerate, torch.full_like(angle, spec.start[0]), spec.radius*angle)
        uv = torch.stack((u, local[..., 2]), -1)
    return uv, degenerate


def _psi_from_tool(tool: Tensor, basis: Tensor, spec: ContactProjectionSpec) -> tuple[Tensor, Tensor, Tensor]:
    # The 2D orthogonal-Procrustes solution minimizes SO(3) orientation change
    # with the inward tool axis fixed. At antipodal normals twist is undefined.
    x, y = basis[..., :, 0], basis[..., :, 1]
    a = (x*tool[..., :, 0]).sum(-1) + (y*tool[..., :, 1]).sum(-1)
    b = (y*tool[..., :, 0]).sum(-1) - (x*tool[..., :, 1]).sum(-1)
    undefined = a.square()+b.square() <= 1e-20
    angle = torch.atan2(torch.where(undefined, torch.zeros_like(b), b),
                        torch.where(undefined, torch.ones_like(a), a))
    previous = torch.full_like(angle[..., 0], spec.start[2])
    values, unconstrained_values = [], []
    for index in range(angle.shape[-1]):
        current = angle[..., index] + 2*np.pi*torch.round((previous-angle[..., index])/(2*np.pi))
        current = torch.where(undefined[..., index], previous, current)
        unconstrained_values.append(current)
        current = current.clamp(float(spec.bounds[2, 0]), float(spec.bounds[2, 1]))
        values.append(current)
        previous = current
    return torch.stack(values, -1), undefined, torch.stack(unconstrained_values, -1)


def _dense_states(states: Tensor, spec: ContactProjectionSpec) -> Tensor:
    chunks = []
    for a, b in zip(states[:-1], states[1:]):
        delta = (b-a).detach().abs()
        angle = delta[2] + (delta[0]/spec.radius if spec.kind == "cylinder" else 0)
        travel = torch.linalg.vector_norm(delta[:2]) + np.linalg.norm(spec.tool_offset)*angle
        count = max(1, int(torch.ceil(travel/spec.validation_step_m)), int(torch.ceil(angle/spec.validation_step_rad)))
        if count > 10000:
            raise ValueError("a contact edge requires too many validation points")
        t = torch.arange(count, device=a.device, dtype=a.dtype)/count
        chunks.append(a[None]+t[:, None]*(b-a)[None])
    return torch.cat([*chunks, states[-1:]], 0)


def _forbidden_edges(states: Tensor, spec: ContactProjectionSpec) -> Tensor:
    a, delta = states[..., :-1, :2], states[..., 1:, :2]-states[..., :-1, :2]
    hits = torch.zeros_like(a[..., 0], dtype=torch.bool)
    for f in spec.forbidden:
        lo, hi = _constant(f[[0, 2]], states), _constant(f[[1, 3]], states)
        parallel = delta.abs() < 1e-14
        safe = torch.where(parallel, torch.ones_like(delta), delta)
        t0, t1 = (lo-a)/safe, (hi-a)/safe
        enter = torch.where(parallel, torch.full_like(t0, -torch.inf), torch.minimum(t0, t1))
        leave = torch.where(parallel, torch.full_like(t1, torch.inf), torch.maximum(t0, t1))
        possible = (~parallel | ((a >= lo) & (a <= hi))).all(-1)
        hits |= possible & (enter.amax(-1).clamp_min(0) <= leave.amin(-1).clamp_max(1))
    return hits.sum(-1)


def _point_checks(world: Tensor, states: Tensor, spec: ContactProjectionSpec) -> dict[str, Tensor]:
    body, degenerate = _safe_rotations(world)
    tip = world[..., :3] + (body @ _constant(spec.tool_offset, world)[..., None]).squeeze(-1)
    point, normal, _ = _contact_basis(states[..., :2], spec)
    tool_axis = (body @ _constant(spec.tool_rotation[:, 2], world)[..., None]).squeeze(-1)
    normal_error = torch.atan2(torch.linalg.vector_norm(torch.cross(tool_axis, -normal, dim=-1), dim=-1), (tool_axis*-normal).sum(-1))
    roll = torch.atan2(body[..., 2, 1], body[..., 2, 2])*180/np.pi
    pitch = torch.atan2(-body[..., 2, 0], torch.hypot(body[..., 0, 0], body[..., 1, 0]))*180/np.pi
    limits = _constant(spec.attitude_limits, world)
    attitude_bad = ((torch.stack((roll, pitch), -1) < limits[:, 0]-1e-6) | (torch.stack((roll, pitch), -1) > limits[:, 1]+1e-6)).any(-1)
    box_bad = torch.zeros_like(roll, dtype=torch.bool)
    for center, rotation, half in spec.boxes:
        local = (tip-_constant(center, world)) @ _constant(rotation, world)
        box_bad |= (local.abs() <= _constant(half+spec.collision_margin, world)).all(-1)
    workspace_bad = torch.zeros_like(roll, dtype=torch.bool)
    if spec.workspace is not None:
        limits_ws = _constant(spec.workspace, world)
        workspace_bad = ((world[..., :3] < limits_ws[0]) | (world[..., :3] > limits_ws[1]) | (tip < limits_ws[0]) | (tip > limits_ws[1])).any(-1)
    bounds = _constant(spec.bounds, states)
    return {"contact_error_max_m": torch.linalg.vector_norm(tip-point, dim=-1).amax(-1),
            "normal_error_max_rad": normal_error.amax(-1),
            "roll_abs_max_deg": roll.abs().amax(-1), "pitch_abs_max_deg": pitch.abs().amax(-1),
            "attitude_violation_count": attitude_bad.sum(-1), "tool_box_violation_count": box_bad.sum(-1),
            "body_origin_or_tool_workspace_violation_count": workspace_bad.sum(-1),
            "chart_violation_count": ((states < bounds[:, 0]-1e-9) | (states > bounds[:, 1]+1e-9)).any(-1).sum(-1),
            "degenerate_rotation_count": degenerate.sum(-1)}


def _metrics(world: Tensor, states: Tensor, spec: ContactProjectionSpec, *, dense: bool) -> dict[str, Tensor]:
    values = _point_checks(world, states, spec)
    psi = states[..., 2]
    tv = torch.diff(psi, dim=-1).abs().sum(-1)
    endpoint_delta = (psi[..., -1]-psi[..., 0]).abs()
    lo, hi = psi.amin(-1), psi.amax(-1)
    minimum_tv = hi-lo + torch.minimum((psi[..., 0]-lo).abs()+(psi[..., -1]-hi).abs(),
                                      (psi[..., 0]-hi).abs()+(psi[..., -1]-lo).abs())
    values.update(psi_total_variation_rad=tv, psi_endpoint_delta_rad=endpoint_delta,
                  psi_excess_reversal_variation_rad=(tv-minimum_tv).clamp_min(0),
                  psi_large_step_count=(torch.diff(psi, dim=-1).abs() > np.pi).sum(-1),
                  forbidden_edge_violation_count=_forbidden_edges(states, spec))
    values["cylinder_large_step_count"] = ((torch.diff(states[..., 0], dim=-1).abs()/spec.radius > np.pi).sum(-1)
                                            if spec.kind == "cylinder" else torch.zeros_like(tv, dtype=torch.long))
    if dense:
        checks = [_point_checks(lift_contact_intrinsic(s, spec), s, spec)
                  for s in (_dense_states(path, spec) for path in states)]
        for key in ("attitude_violation_count", "tool_box_violation_count", "body_origin_or_tool_workspace_violation_count"):
            values["dense_"+key] = torch.stack([c[key] for c in checks])
    return values


def project_contact_pose9(
    paths: Tensor | np.ndarray, spec: ContactProjectionSpec, *,
    coordinate_frame: str = "world", world_condition_pose9: Tensor | np.ndarray | None = None,
    normalization_mean: Tensor | np.ndarray | None = None,
    normalization_std: Tensor | np.ndarray | None = None,
) -> ContactProjectionResult:
    """Project samples without changing endpoints; malformed endpoints raise.

    Output has the input frame, normalization, dtype and shape. Geometric math
    uses float64; diagnostics inspect the actually returned values. Canonical
    mode requires the ORIGINAL metric world [18] condition, not a normalized
    or canonical condition. Endpoints must already match the task body poses
    (2 micrometres/radians serialization allowance); no silent endpoint repair.
    Rotational/axis degeneracies receive deterministic finite fallbacks and
    are counted. Nonfinite inputs are rejected. Bounds/branch decisions are
    piecewise differentiable; this is a decoder, not an orthogonal pose metric
    projection or an obstacle/attitude feasibility solver.
    """
    original = torch.as_tensor(paths)
    if not original.is_floating_point() or original.ndim not in (2, 3) or original.shape[-1] != 9 or original.shape[-2] < 2:
        raise ValueError("paths must be floating [L,9] or [B,L,9], L>=2")
    if not bool(torch.isfinite(original).all()):
        raise ValueError("paths must be finite")
    if coordinate_frame not in ("world", "canonical"):
        raise ValueError("coordinate_frame must be world or canonical")
    squeeze = original.ndim == 2
    batch = original[None] if squeeze else original
    work = batch.to(torch.float64)
    if (normalization_mean is None) != (normalization_std is None):
        raise ValueError("normalization mean and std must be supplied together")
    mean, std = torch.zeros(9, device=work.device, dtype=work.dtype), torch.ones(9, device=work.device, dtype=work.dtype)
    if normalization_mean is not None:
        mean, std = _constant(normalization_mean, work), _constant(normalization_std, work)
        if mean.shape != (9,) or std.shape != (9,) or not bool(torch.isfinite(mean).all() & torch.isfinite(std).all() & (std > 0).all()):
            raise ValueError("normalization must be finite [9] with positive std")
    condition = None
    if coordinate_frame == "canonical":
        if world_condition_pose9 is None:
            raise ValueError("canonical projection requires world_condition_pose9")
        condition = _constant(world_condition_pose9, work)
        if condition.shape == (18,):
            condition = condition[None].expand(len(work), -1)
        if condition.shape != (len(work), 18) or not bool(torch.isfinite(condition).all()):
            raise ValueError("world_condition_pose9 must be finite [18] or [B,18]")
    metric = work*std+mean
    world = decanonicalize_pose9_batch(metric, condition) if condition is not None else metric
    expected = lift_contact_intrinsic(_constant(np.stack((spec.start, spec.goal)), work), spec)
    endpoints = world[:, [0, -1]]
    endpoint_rotations, endpoint_degenerate = _safe_rotations(endpoints)
    expected_rotations, _ = _safe_rotations(expected)
    relative = endpoint_rotations.transpose(-1, -2) @ expected_rotations
    endpoint_rotation_error = torch.atan2(torch.linalg.vector_norm(torch.stack((relative[..., 2, 1]-relative[..., 1, 2], relative[..., 0, 2]-relative[..., 2, 0], relative[..., 1, 0]-relative[..., 0, 1]), -1), dim=-1)/2, (relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2)
    endpoint_position_error = torch.linalg.vector_norm(endpoints[..., :3]-expected[..., :3], dim=-1)
    if bool(endpoint_degenerate.any() | (endpoint_position_error > SERIALIZATION_POSITION_TOLERANCE_M).any() | (endpoint_rotation_error > SERIALIZATION_NORMAL_TOLERANCE_RAD).any()):
        raise ValueError("input endpoints must already equal the contact task body start/goal")
    body, degenerate_rotation = _safe_rotations(world)
    tip = world[..., :3] + (body @ _constant(spec.tool_offset, world)[..., None]).squeeze(-1)
    raw_uv, degenerate_axis = _uv_from_tip(tip, spec)
    bounds = _constant(spec.bounds, work)
    uv = torch.maximum(torch.minimum(raw_uv, bounds[:2, 1]), bounds[:2, 0])
    _, _, basis = _contact_basis(uv, spec)
    psi, undefined_twist, unconstrained_psi = _psi_from_tool(body @ _constant(spec.tool_rotation, world), basis, spec)
    raw_states = torch.cat((raw_uv, unconstrained_psi[..., None]), -1)
    states = torch.cat((uv, psi[..., None]), -1)
    states = torch.cat((_constant(spec.start, work)[None, None].expand(len(work), 1, 3), states[:, 1:-1],
                        _constant(spec.goal, work)[None, None].expand(len(work), 1, 3)), 1)
    projected_world = lift_contact_intrinsic(states, spec)
    projected_metric = canonicalize_pose9_batch(projected_world, condition) if condition is not None else projected_world
    encoded = ((projected_metric-mean)/std).to(batch.dtype)
    encoded = torch.cat((batch[:, :1], encoded[:, 1:-1], batch[:, -1:]), 1)
    returned_metric = encoded.to(torch.float64)*std+mean
    returned_world = decanonicalize_pose9_batch(returned_metric, condition) if condition is not None else returned_metric
    before, after = _metrics(world, raw_states, spec, dense=False), _metrics(returned_world, states, spec, dense=True)
    before["degenerate_rotation_count"] = degenerate_rotation.sum(-1)
    after["input_cylinder_axis_degeneracy_count"] = degenerate_axis.sum(-1)
    after["input_undefined_twist_count"] = undefined_twist.sum(-1)
    after["position_correction_max_m"] = torch.linalg.vector_norm(returned_world[..., :3]-world[..., :3], dim=-1).amax(-1)
    rotation_after, _ = _safe_rotations(returned_world)
    relative = body.transpose(-1, -2) @ rotation_after
    after["rotation_correction_max_rad"] = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)).amax(-1)
    position_tol = max(spec.position_tolerance, SERIALIZATION_POSITION_TOLERANCE_M)
    normal_tol = max(spec.normal_tolerance, SERIALIZATION_NORMAL_TOLERANCE_RAD)
    after["effective_position_tolerance_m"] = torch.full_like(after["contact_error_max_m"], position_tol)
    after["effective_normal_tolerance_rad"] = torch.full_like(after["normal_error_max_rad"], normal_tol)
    after["endpoint_position_error_max_m"] = endpoint_position_error.amax(-1)
    after["endpoint_rotation_error_max_rad"] = endpoint_rotation_error.amax(-1)
    valid = (after["contact_error_max_m"] <= position_tol) & (after["normal_error_max_rad"] <= normal_tol)
    for key in ("chart_violation_count", "forbidden_edge_violation_count", "psi_large_step_count", "cylinder_large_step_count",
                "dense_attitude_violation_count", "dense_tool_box_violation_count", "dense_body_origin_or_tool_workspace_violation_count"):
        valid &= after[key] == 0
    valid &= (after["psi_total_variation_rad"] <= after["psi_endpoint_delta_rad"] + 7*np.pi/6 + 1e-9)
    valid &= after["psi_excess_reversal_variation_rad"] <= np.pi/6 + 1e-9
    def restored(value: Tensor) -> Tensor:
        return value[0] if squeeze else value
    return ContactProjectionResult(original.clone(), restored(encoded), restored(returned_world), restored(states),
                                   {k: restored(v) for k, v in before.items()},
                                   {k: restored(v) for k, v in after.items()}, restored(valid))
