"""Fixed contact decoder supplement: bounded cubic fitting, then 128 nodes.

The source is the predicted path after the frozen analytic v001 projection.
Neither fitting nor its objective reads experts, routes, obstacles, or oracle
labels. Diagnostics inspect task geometry, including its forbidden regions.

The physical output contract is exactly the v001 *linear intrinsic/lift* edge
contract between the returned 128 nodes. The cubic spline is only a fitting
and resampling device. Its nonnegative partition-of-unity basis and bounded
control points keep its entire mathematical curve inside the rectangular,
nonperiodic [u,v,psi] chart; this is not a continuous spline collision audit.
No full-body collision or curvature-quality guarantee is made. This module
does not change the existing projection-off/projection-on ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .contact_projection import (
    INTRINSIC_INTERPOLATION,
    PROJECTION_VERSION,
    ContactProjectionSpec,
    _metrics,
    _safe_rotations,
    lift_contact_intrinsic,
    project_contact_pose9,
)
from .geometry import canonicalize_pose9_batch, decanonicalize_pose9_batch


SMOOTH_PROJECTION_VERSION = "bounded_cubic_contact_nodes_v001"
CONTROL_POINT_COUNT = 16
OUTPUT_NODE_COUNT = 128
SPLINE_DEGREE = 3
BENDING_WEIGHT = 1e-6
SOLVER_TOLERANCE = 1e-10
SOLVER_MAX_SWEEPS = 512
_SPAN_COUNT = CONTROL_POINT_COUNT - SPLINE_DEGREE
_KNOTS = np.r_[np.zeros(4), np.arange(1, _SPAN_COUNT)/_SPAN_COUNT,
               np.ones(4)]


def smooth_contact_decoder_parameters() -> dict[str, Any]:
    """Return the fixed, independently versioned supplementary protocol."""
    return dict(version=SMOOTH_PROJECTION_VERSION, control_points=CONTROL_POINT_COUNT,
                output_nodes=OUTPUT_NODE_COUNT, degree=SPLINE_DEGREE,
                knots=_KNOTS.tolist(), parameterization="uniform_source_node_index",
                coordinate_scaling="each_intrinsic_chart_interval_to_[0,1]",
                objective="mean_squared_fit_plus_integrated_squared_second_derivative",
                bending_weight=BENDING_WEIGHT, bending_quadrature="two_point_Gauss_per_knot_span",
                solver="unconstrained_warm_start_then_box_coordinate_descent",
                solver_tolerance=SOLVER_TOLERANCE, solver_max_sweeps=SOLVER_MAX_SWEEPS,
                solver_residual_definition="max_abs(x-clip(x-(A.T@A@x-A.T@b),0,1))",
                endpoint_policy="fixed_task_controls_and_bitwise_preserved_input_pose9",
                source_projection_version=PROJECTION_VERSION,
                edge_interpolation=INTRINSIC_INTERPOLATION,
                spline_role="fit_and_resample_only", expert_or_oracle_used=False,
                continuous_mathematical_spline_chart_domain_guaranteed=True,
                obstacle_repair=False, continuous_spline_collision_checked=False,
                full_body_collision_checked=False, curvature_gate_guaranteed=False)


@dataclass
class SmoothContactProjectionResult:
    unprojected_pose9: Tensor
    projected_pose9: Tensor
    world_pose9: Tensor
    intrinsic_states: Tensor
    source_intrinsic_states: Tensor
    control_points: Tensor
    before: dict[str, Tensor]
    after: dict[str, Tensor]
    geometry_valid: Tensor
    decoder_parameters: dict[str, Any]
    full_body_collision_checked: bool = False
    curvature_gate_guaranteed: bool = False
    continuous_spline_collision_checked: bool = False


def contact_cubic_basis(parameters: Tensor | np.ndarray, *, derivative: int = 0) -> Tensor:
    """Evaluate the fixed clamped basis (or derivatives in normalized time).

    Returns [T,16]. Endpoint derivatives are one-sided. The degree-zero basis
    at t=1 uses the last nonempty interval, avoiding an epsilon perturbation.
    """
    t = torch.as_tensor(parameters)
    if not t.is_floating_point():
        t = t.to(torch.float64)
    if t.ndim != 1 or not len(t) or not bool(torch.isfinite(t).all()):
        raise ValueError("parameters must be a nonempty finite vector")
    if bool(((t < 0) | (t > 1)).any()) or derivative not in (0, 1, 2):
        raise ValueError("parameters must lie in [0,1]; derivative must be 0, 1 or 2")
    knots = torch.as_tensor(_KNOTS, dtype=t.dtype, device=t.device)
    value = ((t[:, None] >= knots[:-1]) & (t[:, None] < knots[1:])).to(t.dtype)
    at_end = torch.nn.functional.one_hot(
        torch.full((len(t),), CONTROL_POINT_COUNT-1, device=t.device), len(knots)-1).to(t.dtype)
    value = torch.where((t == 1)[:, None], at_end, value)
    first, second = torch.zeros_like(value), torch.zeros_like(value)
    for degree in range(1, SPLINE_DEGREE+1):
        count = len(knots)-degree-1
        left = knots[degree:degree+count]-knots[:count]
        right = knots[degree+1:degree+count+1]-knots[1:count+1]
        inv_left = torch.where(left > 0, 1/left.clamp_min(torch.finfo(t.dtype).tiny), 0.)
        inv_right = torch.where(right > 0, 1/right.clamp_min(torch.finfo(t.dtype).tiny), 0.)
        second = degree*(first[:, :count]*inv_left-first[:, 1:count+1]*inv_right)
        first = degree*(value[:, :count]*inv_left-value[:, 1:count+1]*inv_right)
        value = ((t[:, None]-knots[:count])*inv_left*value[:, :count]
                 + (knots[degree+1:degree+count+1]-t[:, None])*inv_right*value[:, 1:count+1])
    return (value, first, second)[derivative]


def evaluate_contact_cubic(control_points: Tensor | np.ndarray,
                           parameters: Tensor | np.ndarray, *, derivative: int = 0) -> Tensor:
    """Evaluate mathematical intrinsic fit [...,16,3] at parameters [T].

    This helper does not define physical output edges or certify collisions.
    Psi remains an ordinary, unwrapped coordinate; it is never reduced modulo
    2*pi. Its derivatives are with respect to t in [0,1].
    """
    controls = torch.as_tensor(control_points)
    if not controls.is_floating_point():
        controls = controls.to(torch.float64)
    if controls.shape[-2:] != (CONTROL_POINT_COUNT, 3) or not bool(torch.isfinite(controls).all()):
        raise ValueError("control_points must be finite [...,16,3]")
    basis = contact_cubic_basis(torch.as_tensor(parameters, dtype=controls.dtype,
                                               device=controls.device), derivative=derivative)
    return torch.einsum("tc,...cj->...tj", basis, controls)


@lru_cache(maxsize=16)
def _design(source_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = contact_cubic_basis(torch.linspace(0., 1., source_length, dtype=torch.float64)).numpy()
    # A cubic's second derivative is piecewise linear; two Gauss nodes integrate
    # its squared derivative exactly on each of the 13 simple knot intervals.
    span = np.arange(_SPAN_COUNT, dtype=np.float64)[:, None]
    abscissa = np.array([(1-1/np.sqrt(3))/2, (1+1/np.sqrt(3))/2])
    nodes = ((span+abscissa)/_SPAN_COUNT).reshape(-1)
    bending = contact_cubic_basis(nodes, derivative=2).numpy()/np.sqrt(2*_SPAN_COUNT)
    design = np.concatenate((fit/np.sqrt(source_length), np.sqrt(BENDING_WEIGHT)*bending))
    for matrix in (fit, bending, design):
        matrix.setflags(write=False)
    return fit, bending, design


def _fit_controls(states: np.ndarray, spec: ContactProjectionSpec) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    low, width = spec.bounds[:, 0], np.ptp(spec.bounds, axis=1)
    target = (states-low)/width
    endpoints = (np.stack((spec.start, spec.goal))-low)/width
    fit, bending, design = _design(len(states))
    fixed = design[:, [0, -1]] @ endpoints
    right = np.concatenate((target/np.sqrt(len(states)), np.zeros((len(bending), 3))))-fixed
    free = design[:, 1:-1]
    hessian, rhs = free.T @ free, free.T @ right
    controls = np.clip(np.linalg.solve(hessian, rhs), 0., 1.)
    residual = rhs-hessian @ controls
    kkt = float(np.abs(controls-np.clip(controls+residual, 0., 1.)).max())
    sweeps = 0
    while kkt > SOLVER_TOLERANCE and sweeps < SOLVER_MAX_SWEEPS:
        for index in range(CONTROL_POINT_COUNT-2):
            updated = np.clip(controls[index]+residual[index]/hessian[index, index], 0., 1.)
            delta = updated-controls[index]
            controls[index] = updated
            residual -= hessian[:, index, None]*delta[None]
        # Recompute after each sweep to prevent accumulated update roundoff.
        residual = rhs-hessian @ controls
        kkt = float(np.abs(controls-np.clip(controls+residual, 0., 1.)).max())
        sweeps += 1
    normalized = np.concatenate((endpoints[:1], controls, endpoints[1:]))
    fit_mse = float(np.square(fit @ normalized-target).mean())
    bending_energy = float(np.square(bending @ normalized).sum()/3)
    metric_controls = np.clip(low+width*normalized, spec.bounds[:, 0], spec.bounds[:, 1])
    metric_controls[[0, -1]] = np.stack((spec.start, spec.goal))
    return metric_controls, dict(fit_mse_normalized=fit_mse,
                                bending_energy_normalized=bending_energy,
                                objective_normalized=fit_mse+BENDING_WEIGHT*bending_energy,
                                solver_projected_gradient_max=kkt,
                                solver_sweeps=sweeps, solver_converged=kkt <= SOLVER_TOLERANCE)


def _linear_resample(states: Tensor, node_count: int) -> Tensor:
    parameter = torch.linspace(0., states.shape[-2]-1, node_count, dtype=states.dtype, device=states.device)
    index = parameter.floor().to(torch.long).clamp_max(states.shape[-2]-2)
    weight = parameter-index
    return states[:, index]*(1-weight[None, :, None])+states[:, index+1]*weight[None, :, None]


def _rotation_correction(before: Tensor, after: Tensor) -> Tensor:
    first, _ = _safe_rotations(before)
    second, _ = _safe_rotations(after)
    relative = first.transpose(-1, -2) @ second
    skew = torch.stack((relative[..., 2, 1]-relative[..., 1, 2],
                        relative[..., 0, 2]-relative[..., 2, 0],
                        relative[..., 1, 0]-relative[..., 0, 1]), -1)
    return torch.atan2(torch.linalg.vector_norm(skew, dim=-1)/2,
                       (relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).amax(-1)


@torch.no_grad()
def project_smooth_contact_pose9(
    paths: Tensor | np.ndarray, spec: ContactProjectionSpec, *,
    coordinate_frame: str = "world", world_condition_pose9: Tensor | np.ndarray | None = None,
    normalization_mean: Tensor | np.ndarray | None = None,
    normalization_std: Tensor | np.ndarray | None = None,
) -> SmoothContactProjectionResult:
    """Inference-only, fixed decoder; accepts the frozen projection's inputs.

    Returns 128 nodes in the input frame, normalization, device and dtype.
    Both endpoint pose9 values are preserved bit-for-bit. Their validity and
    all malformed-input checks use the frozen v001 endpoint contract. The
    source may have any L>=2; fitting uses uniform node-index parameterization.
    Batch entries must share one task spec. Fitting is detached and CPU float64,
    and introduces no dependency beyond the existing NumPy/PyTorch runtime.

    ``geometry_valid`` uses v001 diagnostics on the actually returned poses
    and their linear intrinsic/lift edges, plus solver convergence. It does
    not include full URDF collisions or trajectory quality gates. No obstacle,
    attitude, or route feasibility terms enter the fitting objective.
    """
    analytic = project_contact_pose9(paths, spec, coordinate_frame=coordinate_frame,
                                    world_condition_pose9=world_condition_pose9,
                                    normalization_mean=normalization_mean,
                                    normalization_std=normalization_std)
    original = analytic.unprojected_pose9
    squeeze = original.ndim == 2
    batch = original[None] if squeeze else original
    source = analytic.intrinsic_states[None] if squeeze else analytic.intrinsic_states
    if not len(batch):
        raise ValueError("batch must contain at least one path")
    fitted = [_fit_controls(path, spec) for path in source.detach().cpu().numpy()]
    controls = torch.as_tensor(np.stack([item[0] for item in fitted]), dtype=torch.float64, device=batch.device)
    parameter = torch.linspace(0., 1., OUTPUT_NODE_COUNT, dtype=torch.float64, device=batch.device)
    states = evaluate_contact_cubic(controls, parameter)
    # Explicit endpoints also remove summation roundoff in basis evaluation.
    states[:, 0], states[:, -1] = controls[:, 0], controls[:, -1]
    lifted = lift_contact_intrinsic(states, spec)
    mean = torch.zeros(9, dtype=torch.float64, device=batch.device)
    std = torch.ones_like(mean)
    if normalization_mean is not None:
        mean = torch.as_tensor(normalization_mean, dtype=torch.float64, device=batch.device)
        std = torch.as_tensor(normalization_std, dtype=torch.float64, device=batch.device)
    condition = None
    if coordinate_frame == "canonical":
        condition = torch.as_tensor(world_condition_pose9, dtype=torch.float64, device=batch.device)
        if condition.shape == (18,):
            condition = condition[None].expand(len(batch), -1)
    metric = canonicalize_pose9_batch(lifted, condition) if condition is not None else lifted
    encoded = ((metric-mean)/std).to(batch.dtype)
    encoded[:, 0], encoded[:, -1] = batch[:, 0], batch[:, -1]
    returned_metric = encoded.to(torch.float64)*std+mean
    world = decanonicalize_pose9_batch(returned_metric, condition) if condition is not None else returned_metric
    # Frozen v001 diagnostics deliberately inspect linear intrinsic/lift edges.
    after = _metrics(world, states, spec, dense=True)
    for name in ("input_cylinder_axis_degeneracy_count", "input_undefined_twist_count",
                 "effective_position_tolerance_m", "effective_normal_tolerance_rad",
                 "endpoint_position_error_max_m", "endpoint_rotation_error_max_rad"):
        value = analytic.after[name]
        after[name] = value[None] if squeeze else value
    reference_states = _linear_resample(source, OUTPUT_NODE_COUNT)
    reference_world = lift_contact_intrinsic(reference_states, spec)
    after["smoothing_body_position_correction_max_m"] = torch.linalg.vector_norm(world[..., :3]-reference_world[..., :3], dim=-1).amax(-1)
    after["smoothing_body_rotation_correction_max_rad"] = _rotation_correction(reference_world, world)
    after["smoothing_intrinsic_uv_correction_max_m"] = torch.linalg.vector_norm(states[..., :2]-reference_states[..., :2], dim=-1).amax(-1)
    after["smoothing_intrinsic_psi_correction_max_rad"] = (states[..., 2]-reference_states[..., 2]).abs().amax(-1)
    for name in ("position_correction_max_m", "rotation_correction_max_rad"):
        value = analytic.after[name]
        after["pointwise_"+name] = value[None] if squeeze else value
    bounds = torch.as_tensor(spec.bounds, dtype=controls.dtype, device=controls.device)
    after["control_chart_violation_count"] = ((controls < bounds[:, 0]-1e-12) | (controls > bounds[:, 1]+1e-12)).any(-1).sum(-1)
    for name, value in fitted[0][1].items():
        dtype = torch.bool if isinstance(value, bool) else torch.long if isinstance(value, int) else torch.float64
        after[name] = torch.tensor([item[1][name] for item in fitted], dtype=dtype, device=batch.device)
    valid = ((after["contact_error_max_m"] <= after["effective_position_tolerance_m"])
             & (after["normal_error_max_rad"] <= after["effective_normal_tolerance_rad"])
             & after["solver_converged"])
    for name in ("control_chart_violation_count", "chart_violation_count", "forbidden_edge_violation_count",
                 "psi_large_step_count", "cylinder_large_step_count", "dense_attitude_violation_count",
                 "dense_tool_box_violation_count", "dense_body_origin_or_tool_workspace_violation_count"):
        valid &= after[name] == 0
    valid &= after["psi_total_variation_rad"] <= after["psi_endpoint_delta_rad"]+7*np.pi/6+1e-9
    valid &= after["psi_excess_reversal_variation_rad"] <= np.pi/6+1e-9
    def restored(value: Tensor) -> Tensor:
        return value[0] if squeeze else value
    return SmoothContactProjectionResult(
        original.clone(), restored(encoded), restored(world), restored(states), restored(source),
        restored(controls), analytic.before, {name: restored(value) for name, value in after.items()},
        restored(valid), smooth_contact_decoder_parameters())
