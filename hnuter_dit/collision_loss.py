"""Differentiable UAM collision-sphere loss against oriented boxes."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import quaternion_wxyz_to_matrix_torch, rotation_6d_to_matrix


# Exact contents of the verified cuRobo base_link collision-sphere model used
# by HNUTER-diffusion.  Values are metres in the vehicle body frame.
ROBOT_SPHERE_CENTERS = (
    (-0.162177, -0.059880, -0.091330), (-0.162177, -0.059880, 0.023090),
    (-0.162177, 0.059720, -0.091330), (-0.162177, 0.059720, 0.023090),
    (0.003180, -0.059880, -0.091330), (0.003180, -0.059880, 0.023090),
    (0.003180, 0.059720, -0.091330), (0.003180, 0.059720, 0.023090),
    (0.168537, -0.059880, -0.091330), (0.168537, -0.059880, 0.023090),
    (0.168537, 0.059720, -0.091330), (0.168537, 0.059720, 0.023090),
    (0.140869, 0.216609, -0.287840), (0.132563, 0.204606, -0.256944),
    (0.124256, 0.192603, -0.226047), (0.115950, 0.180600, -0.195150),
    (0.107644, 0.168597, -0.164253), (0.099337, 0.156594, -0.133356),
    (0.091031, 0.144591, -0.102460), (0.140201, -0.217177, -0.287844),
    (0.132057, -0.205075, -0.256943), (0.123914, -0.192972, -0.226041),
    (0.115770, -0.180870, -0.195140), (0.107626, -0.168768, -0.164239),
    (0.099483, -0.156665, -0.133337), (0.091339, -0.144563, -0.102436),
    (-0.389541, -0.000080, -0.287977), (-0.374984, -0.000080, -0.257061),
    (-0.360427, -0.000080, -0.226146), (-0.345870, -0.000080, -0.195230),
    (-0.331313, -0.000080, -0.164314), (-0.316756, -0.000080, -0.133399),
    (-0.302199, -0.000080, -0.102483), (0.003430, 0.408270, -0.005720),
    (0.003430, -0.408270, -0.005720), (-0.768000, 0.000000, 0.003400),
    (-0.668000, 0.000000, 0.003400), (-0.718000, 0.086603, 0.003400),
    (-0.818000, 0.086603, 0.003400), (-0.868000, 0.000000, 0.003400),
    (-0.818000, -0.086603, 0.003400), (-0.718000, -0.086603, 0.003400),
)
ROBOT_SPHERE_RADII = (
    *((0.116982,) * 12), *((0.027802,) * 21),
    0.239200, 0.239200, *((0.126901,) * 7),
)


def sphere_obb_signed_distances(
    path_metric: Tensor, obstacles: Tensor, obstacle_mask: Tensor,
    *, waypoint_stride: int = 4,
    softmin_temperature: float | None = None,
) -> Tensor:
    """Return the minimum robot-sphere/OBB distance at sampled waypoints."""

    if waypoint_stride <= 0:
        raise ValueError("waypoint_stride must be positive")
    path = path_metric[:, 1:-1:waypoint_stride].float()
    position = path[..., :3]
    rotation = rotation_6d_to_matrix(path[..., 3:9])
    centers = torch.as_tensor(
        ROBOT_SPHERE_CENTERS, dtype=path.dtype, device=path.device
    )
    radii = torch.as_tensor(
        ROBOT_SPHERE_RADII, dtype=path.dtype, device=path.device
    )
    sphere_centers = position[:, :, None] + torch.einsum(
        "blij,sj->blsi", rotation, centers
    )
    obstacle_rotation = quaternion_wxyz_to_matrix_torch(
        obstacles[..., 6:10].float()
    )
    relative = sphere_centers[:, :, :, None] - obstacles[:, None, None, :, :3]
    local = torch.einsum("blsmi,bmij->blsmj", relative, obstacle_rotation)
    half_size = 0.5 * obstacles[:, None, None, :, 3:6]
    delta = torch.abs(local) - half_size
    outside = torch.linalg.vector_norm(F.relu(delta), dim=-1)
    inside = torch.minimum(delta.amax(dim=-1), torch.zeros_like(outside))
    signed = outside + inside - radii[None, None, :, None]
    signed = signed.masked_fill(
        ~obstacle_mask[:, None, None], torch.finfo(signed.dtype).max
    )
    if softmin_temperature is None:
        return signed.amin(dim=(2, 3))
    if softmin_temperature <= 0:
        raise ValueError("softmin_temperature must be positive")
    return -softmin_temperature * torch.logsumexp(
        -signed / softmin_temperature, dim=(2, 3)
    )


def collision_auxiliary_loss(
    predicted_x0_normalized: Tensor,
    path_mean: Tensor,
    path_std: Tensor,
    obstacles: Tensor,
    obstacle_mask: Tensor,
    alpha_bar: Tensor,
    *,
    safe_distance: float = 0.04,
    sigma: float = 0.05,
    timestep_power: float = 1.0,
    waypoint_stride: int = 4,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Collision loss on recovered clean trajectories with low-noise weighting."""

    if sigma <= 0 or timestep_power < 0:
        raise ValueError("sigma must be positive and timestep_power non-negative")
    path_metric = predicted_x0_normalized.float() * path_std.float() + path_mean.float()
    minimum = sphere_obb_signed_distances(
        path_metric, obstacles.float(), obstacle_mask.bool(),
        waypoint_stride=waypoint_stride,
    )
    penalty = F.softplus((safe_distance - minimum) / sigma).square().mean(dim=1)
    weights = alpha_bar.reshape(len(path_metric), -1)[:, 0].float().pow(timestep_power)
    loss = (weights * penalty).mean()
    return loss, {
        "minimum_clearance": minimum.detach().amin(),
        "mean_minimum_clearance": minimum.detach().amin(dim=1).mean(),
        "collision_fraction": (minimum.detach() <= 0).any(dim=1).float().mean(),
        "mean_timestep_weight": weights.detach().mean(),
    }
