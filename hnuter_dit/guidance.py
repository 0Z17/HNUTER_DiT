"""Late-step differentiable collision guidance for trajectory diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .collision_loss import sphere_obb_signed_distances
from .model import hard_clamp_endpoints


@dataclass(frozen=True)
class CollisionGuidanceConfig:
    """Configuration for position-only clean-trajectory collision guidance."""

    mode: str = "smooth"
    steps: int = 0
    step_size_m: float = 0.08
    safe_distance_m: float = 0.04
    sigma_m: float = 0.05
    smoothing_passes: int = 1
    control_points: int = 16
    optimization_steps: int = 3
    softmin_temperature_m: float = 0.02
    reference_weight: float = 0.10
    smoothness_weight: float = 0.25
    jerk_weight: float = 0.05
    stretch_weight: float = 0.25
    tangent_weight: float = 1.00

    def validate(self, sampling_steps: int) -> None:
        if not 0 <= self.steps <= sampling_steps:
            raise ValueError("guidance steps 必须在 [0, DDIM steps]")
        if self.mode not in {"smooth", "pointwise"}:
            raise ValueError("guidance mode 必须是 smooth 或 pointwise")
        if not 0 < self.step_size_m <= 0.20:
            raise ValueError("guidance step size 必须在 (0, 0.20] m")
        if self.safe_distance_m < 0 or self.sigma_m <= 0:
            raise ValueError("guidance safe distance 必须非负且 sigma 必须为正")
        if not 0 <= self.smoothing_passes <= 8:
            raise ValueError("guidance smoothing passes 必须在 [0, 8]")
        if not 4 <= self.control_points <= 64:
            raise ValueError("guidance control points 必须在 [4, 64]")
        if not 1 <= self.optimization_steps <= 12:
            raise ValueError("guidance optimization steps 必须在 [1, 12]")
        if self.softmin_temperature_m <= 0:
            raise ValueError("guidance softmin temperature 必须为正")
        if min(
            self.reference_weight, self.smoothness_weight, self.jerk_weight,
            self.stretch_weight, self.tangent_weight,
        ) < 0:
            raise ValueError("guidance regularization weights 必须非负")


def _smooth_displacement(displacement: Tensor, passes: int) -> Tensor:
    if passes == 0:
        return displacement
    value = displacement.transpose(1, 2)
    kernel = torch.tensor(
        [0.25, 0.5, 0.25], dtype=value.dtype, device=value.device
    ).reshape(1, 1, 3).expand(3, 1, 3)
    for _ in range(passes):
        value = F.conv1d(F.pad(value, (1, 1), mode="replicate"), kernel, groups=3)
    result = value.transpose(1, 2)
    result[:, 0] = 0
    result[:, -1] = 0
    return result


def _pointwise_guidance(
    predicted_x0_normalized: Tensor,
    conditions_normalized: Tensor,
    path_mean: Tensor,
    path_std: Tensor,
    obstacles: Tensor,
    obstacle_mask: Tensor,
    config: CollisionGuidanceConfig, mean: Tensor, std: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Legacy independent-waypoint guidance retained for controlled ablation."""

    with torch.enable_grad():
        metric = (
            predicted_x0_normalized.detach().float() * std + mean
        ).requires_grad_(True)
        minimum = sphere_obb_signed_distances(
            metric, obstacles.float(), obstacle_mask.bool(), waypoint_stride=1
        )
        penalty = F.softplus(
            (config.safe_distance_m - minimum) / config.sigma_m
        ).square()
        objective = penalty.mean()
        gradient = torch.autograd.grad(objective, metric, only_inputs=True)[0]

    position_gradient = gradient[..., :3]
    gradient_norm = torch.linalg.vector_norm(
        position_gradient, dim=-1, keepdim=True
    )
    severity = torch.sigmoid(
        (config.safe_distance_m - minimum.detach()) / config.sigma_m
    )
    full_severity = torch.zeros(
        metric.shape[:2], dtype=metric.dtype, device=metric.device
    )
    full_severity[:, 1:-1] = severity
    displacement = -config.step_size_m * full_severity[..., None] * (
        position_gradient / gradient_norm.clamp_min(1e-12)
    )
    displacement = torch.where(
        gradient_norm > 1e-12, displacement, torch.zeros_like(displacement)
    )
    displacement = _smooth_displacement(displacement, config.smoothing_passes)
    displacement_norm = torch.linalg.vector_norm(
        displacement, dim=-1, keepdim=True
    )
    displacement = displacement * torch.clamp(
        config.step_size_m / displacement_norm.clamp_min(1e-12), max=1.0
    )
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)

    guided_metric = metric.detach().clone()
    guided_metric[..., :3] += displacement
    guided = (guided_metric - mean) / std
    guided = hard_clamp_endpoints(guided, conditions_normalized)
    return guided.to(predicted_x0_normalized.dtype), {
        "mode": "pointwise",
        "active_fraction": float(
            (minimum.detach() < config.safe_distance_m).any(dim=1).float().mean()
        ),
        "minimum_clearance_m": float(minimum.detach().amin()),
        "mean_displacement_m": float(
            torch.linalg.vector_norm(displacement, dim=-1).mean()
        ),
        "maximum_displacement_m": float(
            torch.linalg.vector_norm(displacement, dim=-1).amax()
        ),
    }


def _control_displacement(control: Tensor, length: int) -> Tensor:
    displacement = F.interpolate(
        control.transpose(1, 2), size=length,
        mode="linear", align_corners=True,
    ).transpose(1, 2)
    phase = torch.linspace(
        0, torch.pi, length, dtype=control.dtype, device=control.device
    )
    return displacement * torch.sin(phase).square()[None, :, None]


@torch.enable_grad()
def _smooth_guidance(
    predicted_x0_normalized: Tensor,
    conditions_normalized: Tensor,
    obstacles: Tensor,
    obstacle_mask: Tensor,
    config: CollisionGuidanceConfig,
    mean: Tensor,
    std: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Optimize a low-dimensional smooth displacement field around the DiT prior."""

    base_metric = predicted_x0_normalized.detach().float() * std + mean
    base_position = base_metric[..., :3]
    base_segments = base_position[:, 1:] - base_position[:, :-1]
    base_segment_length = torch.linalg.vector_norm(
        base_segments, dim=-1
    ).clamp_min(1e-5)
    length_scale = base_segment_length.mean(dim=1).clamp_min(0.02)
    control = torch.zeros(
        (len(base_metric), config.control_points, 3),
        dtype=base_metric.dtype, device=base_metric.device,
        requires_grad=True,
    )
    first_moment = torch.zeros_like(control)
    second_moment = torch.zeros_like(control)
    component_values: dict[str, Tensor] = {}
    minimum = None
    learning_rate = config.step_size_m / config.optimization_steps
    for optimization_index in range(config.optimization_steps):
        displacement = _control_displacement(control, base_metric.shape[1])
        candidate = base_metric.clone()
        candidate[..., :3] = base_position + displacement
        minimum = sphere_obb_signed_distances(
            candidate, obstacles.float(), obstacle_mask.bool(),
            waypoint_stride=1,
            softmin_temperature=config.softmin_temperature_m,
        )
        collision = F.softplus(
            (config.safe_distance_m - minimum) / config.sigma_m
        ).square().mean(dim=1)
        reference = (
            displacement / max(config.step_size_m, 1e-4)
        ).square().mean(dim=(1, 2))
        acceleration = (
            displacement[:, 2:] - 2 * displacement[:, 1:-1]
            + displacement[:, :-2]
        )
        jerk = acceleration[:, 1:] - acceleration[:, :-1]
        acceleration_scale = (0.25 * length_scale).clamp_min(0.005)
        jerk_scale = (0.125 * length_scale).clamp_min(0.0025)
        smoothness = (
            acceleration / acceleration_scale[:, None, None]
        ).square().mean(dim=(1, 2))
        jerk_cost = (
            jerk / jerk_scale[:, None, None]
        ).square().mean(dim=(1, 2))
        candidate_segments = candidate[:, 1:, :3] - candidate[:, :-1, :3]
        candidate_length = torch.linalg.vector_norm(
            candidate_segments, dim=-1
        ).clamp_min(1e-6)
        stretch = (
            (candidate_length - base_segment_length)
            / length_scale[:, None]
        ).square().mean(dim=1)
        base_tangent = F.normalize(base_segments, dim=-1, eps=1e-8)
        candidate_tangent = F.normalize(candidate_segments, dim=-1, eps=1e-8)
        alignment = (base_tangent * candidate_tangent).sum(dim=-1)
        tangent = F.relu(0.5 - alignment).square().mean(dim=1)
        components = {
            "collision": collision,
            "reference": reference,
            "smoothness": smoothness,
            "jerk": jerk_cost,
            "stretch": stretch,
            "tangent": tangent,
        }
        objective = (
            collision
            + config.reference_weight * reference
            + config.smoothness_weight * smoothness
            + config.jerk_weight * jerk_cost
            + config.stretch_weight * stretch
            + config.tangent_weight * tangent
        ).sum()
        gradient = torch.autograd.grad(objective, control, only_inputs=True)[0]
        with torch.no_grad():
            first_moment.mul_(0.9).add_(gradient, alpha=0.1)
            second_moment.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            step = optimization_index + 1
            direction = (
                first_moment / (1 - 0.9 ** step)
            ) / (
                torch.sqrt(second_moment / (1 - 0.999 ** step)) + 1e-8
            )
            control.add_(direction, alpha=-learning_rate)
            magnitude = torch.linalg.vector_norm(control, dim=-1, keepdim=True)
            control.mul_(torch.clamp(
                config.step_size_m / magnitude.clamp_min(1e-12), max=1.0
            ))
        control = control.detach().requires_grad_(True)
        component_values = {name: value.detach() for name, value in components.items()}

    displacement = _control_displacement(control.detach(), base_metric.shape[1])
    displacement = _smooth_displacement(displacement, config.smoothing_passes)
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    displacement = displacement * torch.clamp(
        config.step_size_m / displacement_norm.clamp_min(1e-12), max=1.0
    )
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    guided_metric = base_metric.clone()
    guided_metric[..., :3] = base_position + displacement
    after_minimum = sphere_obb_signed_distances(
        guided_metric, obstacles.float(), obstacle_mask.bool(), waypoint_stride=1
    )
    guided = hard_clamp_endpoints(
        (guided_metric - mean) / std, conditions_normalized
    )
    return guided.to(predicted_x0_normalized.dtype), {
        "mode": "smooth",
        "active_fraction": float(
            (minimum.detach() < config.safe_distance_m).any(dim=1).float().mean()
        ) if minimum is not None else 0.0,
        "minimum_clearance_m": float(after_minimum.detach().amin()),
        "mean_displacement_m": float(displacement_norm.mean()),
        "maximum_displacement_m": float(displacement_norm.amax()),
        **{
            f"cost_{name}": float(value.mean())
            for name, value in component_values.items()
        },
    }


def guide_clean_trajectory(
    predicted_x0_normalized: Tensor,
    conditions_normalized: Tensor,
    path_mean: Tensor,
    path_std: Tensor,
    obstacles: Tensor,
    obstacle_mask: Tensor,
    config: CollisionGuidanceConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Apply legacy pointwise or regularized smooth collision guidance."""

    if config.steps <= 0:
        return predicted_x0_normalized, {
            "mode": config.mode,
            "active_fraction": 0.0,
            "minimum_clearance_m": float("inf"),
        }
    mean = path_mean.reshape(1, 1, -1).float()
    std = path_std.reshape(1, 1, -1).float()
    if config.mode == "pointwise":
        return _pointwise_guidance(
            predicted_x0_normalized, conditions_normalized,
            path_mean, path_std, obstacles, obstacle_mask, config, mean, std,
        )
    return _smooth_guidance(
        predicted_x0_normalized, conditions_normalized,
        obstacles, obstacle_mask, config, mean, std,
    )
