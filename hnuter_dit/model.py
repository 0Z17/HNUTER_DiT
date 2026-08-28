"""Checkpoint-compatible HNUTER environment-conditioned DiT inference.

This is the inference-only subset of ``HNUTER-diffusion/se3_diffusion.py`` at
commit ``c7485fef14b1e1036a9ce529093fe8218b57132e``.  It intentionally preserves
module and parameter names so the trained Base checkpoint loads strictly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


POSE_DIM = 9
CONDITION_DIM = 18
OBSTACLE_DIM = 10


def quaternion_wxyz_to_matrix_numpy(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
        2 * (x * z + y * w), 2 * (x * y + z * w),
        1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ), axis=-1).reshape(quaternion.shape[:-1] + (3, 3))


def pose7_to_pose9(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    rotation = quaternion_wxyz_to_matrix_numpy(pose[..., 3:7])
    rotation_6d = np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)
    return np.concatenate((pose[..., :3], rotation_6d), axis=-1)


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    first = F.normalize(rotation_6d[..., :3], dim=-1, eps=1e-8)
    second_raw = rotation_6d[..., 3:6]
    second = F.normalize(
        second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first,
        dim=-1, eps=1e-8,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def matrix_to_quaternion_wxyz(matrix: Tensor) -> Tensor:
    batch_shape = matrix.shape[:-2]
    flat = matrix.reshape(-1, 3, 3)
    output = torch.empty((len(flat), 4), dtype=matrix.dtype, device=matrix.device)
    for index, item in enumerate(flat):
        trace = item.trace()
        if trace > 0:
            scale = torch.sqrt(trace + 1.0) * 2.0
            output[index] = torch.stack((
                0.25 * scale,
                (item[2, 1] - item[1, 2]) / scale,
                (item[0, 2] - item[2, 0]) / scale,
                (item[1, 0] - item[0, 1]) / scale,
            ))
        else:
            diagonal = torch.diagonal(item)
            largest = int(torch.argmax(diagonal).item())
            if largest == 0:
                scale = torch.sqrt(1 + item[0, 0] - item[1, 1] - item[2, 2]) * 2
                output[index] = torch.stack(((item[2, 1] - item[1, 2]) / scale,
                    0.25 * scale, (item[0, 1] + item[1, 0]) / scale,
                    (item[0, 2] + item[2, 0]) / scale))
            elif largest == 1:
                scale = torch.sqrt(1 + item[1, 1] - item[0, 0] - item[2, 2]) * 2
                output[index] = torch.stack(((item[0, 2] - item[2, 0]) / scale,
                    (item[0, 1] + item[1, 0]) / scale, 0.25 * scale,
                    (item[1, 2] + item[2, 1]) / scale))
            else:
                scale = torch.sqrt(1 + item[2, 2] - item[0, 0] - item[1, 1]) * 2
                output[index] = torch.stack(((item[1, 0] - item[0, 1]) / scale,
                    (item[0, 2] + item[2, 0]) / scale,
                    (item[1, 2] + item[2, 1]) / scale, 0.25 * scale))
    return F.normalize(output.reshape(batch_shape + (4,)), dim=-1)


def pose9_to_pose7_numpy(pose: np.ndarray) -> np.ndarray:
    value = torch.as_tensor(pose, dtype=torch.float64)
    rotation = rotation_6d_to_matrix(value[..., 3:9])
    quaternion = matrix_to_quaternion_wxyz(rotation)
    return torch.cat((value[..., :3], quaternion), dim=-1).cpu().numpy()


def timestep_embedding(
    timesteps: Tensor, dimension: int, maximum_period: int = 10000,
) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(maximum_period)
        * torch.arange(half, device=timesteps.device) / half
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    return F.pad(embedding, (0, 1)) if dimension % 2 else embedding


class DiTBlock(nn.Module):
    def __init__(
        self, dimension: int, heads: int, condition_dim: int,
        cross_attention: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.cross_attention = (
            nn.MultiheadAttention(dimension, heads, batch_first=True)
            if cross_attention else None
        )
        self.norm_cross = nn.LayerNorm(dimension, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dimension, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, 4 * dimension), nn.GELU(),
            nn.Linear(4 * dimension, dimension),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(condition_dim, 6 * dimension)
        )

    def forward(
        self, value: Tensor, condition: Tensor,
        environment: Tensor | None, environment_mask: Tensor | None,
    ) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(
            condition
        ).chunk(6, dim=-1)
        query = self.norm1(value) * (1 + scale1[:, None]) + shift1[:, None]
        attended = self.self_attention(query, query, query, need_weights=False)[0]
        value = value + gate1[:, None] * attended
        if self.cross_attention is not None:
            if environment is None:
                raise ValueError("cross-attention DiT requires environment tokens")
            attended = self.cross_attention(
                self.norm_cross(value), environment, environment,
                key_padding_mask=(
                    ~environment_mask if environment_mask is not None else None
                ),
                need_weights=False,
            )[0]
            value = value + attended
        hidden = self.norm2(value) * (1 + scale2[:, None]) + shift2[:, None]
        return value + gate2[:, None] * self.mlp(hidden)


class ConditionalDiT(nn.Module):
    def __init__(
        self, sequence_length: int = 128, path_dim: int = POSE_DIM,
        context_dim: int = CONDITION_DIM, dimension: int = 128,
        depth: int = 4, heads: int = 4, cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.model_type = "dit_cross" if cross_attention else "dit"
        self.cross_attention = cross_attention
        self.dimension = dimension
        self.register_buffer("environment_geometry_mean", torch.zeros(6), persistent=False)
        self.register_buffer("environment_geometry_std", torch.ones(6), persistent=False)
        self.input = nn.Linear(path_dim, dimension)
        self.position_embedding = nn.Parameter(
            torch.randn(1, sequence_length, dimension) * 0.02
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(64, dimension), nn.SiLU(), nn.Linear(dimension, dimension)
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, dimension), nn.SiLU(),
            nn.Linear(dimension, dimension),
        )
        self.environment_mlp = (
            nn.Sequential(
                nn.Linear(OBSTACLE_DIM, dimension), nn.SiLU(),
                nn.Linear(dimension, dimension),
            ) if cross_attention else None
        )
        self.blocks = nn.ModuleList([
            DiTBlock(dimension, heads, dimension, cross_attention)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dimension, 2 * dimension)
        )
        self.output = nn.Linear(dimension, path_dim)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def configure_environment_normalization(
        self, mean: np.ndarray | Tensor, std: np.ndarray | Tensor,
    ) -> None:
        mean_tensor = torch.as_tensor(
            mean, dtype=self.environment_geometry_mean.dtype,
            device=self.environment_geometry_mean.device,
        )
        std_tensor = torch.as_tensor(
            std, dtype=self.environment_geometry_std.dtype,
            device=self.environment_geometry_std.device,
        )
        if mean_tensor.shape != (6,) or std_tensor.shape != (6,):
            raise ValueError("environment geometry normalization must have shape [6]")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
            raise ValueError("environment normalization must be finite")
        if not (std_tensor > 0).all():
            raise ValueError("environment geometry std must be positive")
        self.environment_geometry_mean.copy_(mean_tensor)
        self.environment_geometry_std.copy_(std_tensor)

    def encode_environment(
        self, path: Tensor, obstacles: Tensor | None, obstacle_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.environment_mlp is None:
            return None, obstacle_mask
        if obstacles is None:
            obstacles = torch.zeros(
                (len(path), 1, OBSTACLE_DIM), dtype=path.dtype, device=path.device
            )
            obstacle_mask = torch.ones(
                (len(path), 1), dtype=torch.bool, device=path.device
            )
        elif obstacles.ndim == 2:
            obstacles = obstacles[None].expand(len(path), -1, -1)
            if obstacle_mask is not None and obstacle_mask.ndim == 1:
                obstacle_mask = obstacle_mask[None].expand(len(path), -1)
        if obstacle_mask is None:
            obstacle_mask = torch.ones(
                obstacles.shape[:2], dtype=torch.bool, device=obstacles.device
            )
        else:
            obstacle_mask = obstacle_mask.bool()
            empty = ~obstacle_mask.any(dim=1)
            if empty.any():
                obstacles = obstacles.clone()
                obstacle_mask = obstacle_mask.clone()
                obstacles[empty, 0] = 0
                obstacle_mask[empty, 0] = True
        normalized = obstacles.clone()
        geometry = (
            normalized[..., :6] - self.environment_geometry_mean
        ) / self.environment_geometry_std
        normalized[..., :6] = torch.where(
            obstacle_mask[..., None], geometry, torch.zeros_like(geometry)
        )
        normalized = torch.where(
            obstacle_mask[..., None], normalized, torch.zeros_like(normalized)
        )
        return self.environment_mlp(normalized), obstacle_mask

    def forward(
        self, path: Tensor, timestep: Tensor, context: Tensor,
        obstacles: Tensor | None = None, obstacle_mask: Tensor | None = None,
    ) -> Tensor:
        condition = (
            self.time_mlp(timestep_embedding(timestep, 64))
            + self.context_mlp(context)
        )
        value = self.input(path) + self.position_embedding[:, :path.shape[1]]
        environment, environment_mask = self.encode_environment(
            path, obstacles, obstacle_mask
        )
        for block in self.blocks:
            value = block(value, condition, environment, environment_mask)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        value = self.final_norm(value) * (1 + scale[:, None]) + shift[:, None]
        return self.output(value)


@dataclass
class DiffusionSchedule:
    betas: Tensor
    alphas: Tensor
    alpha_bars: Tensor

    @classmethod
    def cosine(cls, steps: int, device: torch.device) -> "DiffusionSchedule":
        offset = 0.008
        points = torch.linspace(0, steps, steps + 1, device=device)
        alpha_bar = torch.cos(
            ((points / steps + offset) / (1 + offset)) * math.pi / 2
        ) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], 1e-5, 0.999)
        alphas = 1 - betas
        return cls(betas, alphas, torch.cumprod(alphas, dim=0))


def hard_clamp_endpoints(path: Tensor, context: Tensor) -> Tensor:
    result = path.clone()
    result[:, 0] = context[:, :POSE_DIM]
    result[:, -1] = context[:, POSE_DIM:]
    return result


@torch.inference_mode()
def ddim_sample(
    model: nn.Module, conditions: Tensor, schedule: DiffusionSchedule,
    sequence_length: int, sampling_steps: int, obstacles: Tensor,
    obstacle_mask: Tensor, generator: torch.Generator,
    clip_x0: float = 4.0,
) -> Tensor:
    path = torch.randn(
        (len(conditions), sequence_length, POSE_DIM),
        device=conditions.device, generator=generator,
    )
    path = hard_clamp_endpoints(path, conditions)
    if obstacles.ndim == 2:
        obstacles = obstacles[None].expand(len(path), -1, -1)
        obstacle_mask = obstacle_mask[None].expand(len(path), -1)
    indices = torch.linspace(
        len(schedule.betas) - 1, 0, sampling_steps, device=conditions.device
    ).round().long().unique_consecutive()
    for sample_index, timestep_value in enumerate(indices):
        timestep = torch.full(
            (len(path),), int(timestep_value),
            device=conditions.device, dtype=torch.long,
        )
        velocity = model(path, timestep, conditions, obstacles, obstacle_mask)
        alpha_bar = schedule.alpha_bars[timestep_value]
        predicted_x0 = (
            torch.sqrt(alpha_bar) * path
            - torch.sqrt(1 - alpha_bar) * velocity
        )
        predicted_x0 = hard_clamp_endpoints(
            torch.clamp(predicted_x0, -clip_x0, clip_x0), conditions
        )
        predicted_noise = (
            path - torch.sqrt(alpha_bar) * predicted_x0
        ) / torch.clamp(torch.sqrt(1 - alpha_bar), min=1e-6)
        if sample_index == len(indices) - 1:
            path = predicted_x0
        else:
            previous_alpha_bar = schedule.alpha_bars[indices[sample_index + 1]]
            path = (
                torch.sqrt(previous_alpha_bar) * predicted_x0
                + torch.sqrt(1 - previous_alpha_bar) * predicted_noise
            )
            path = hard_clamp_endpoints(path, conditions)
    return hard_clamp_endpoints(path, conditions)


def build_model(
    model_type: str, sequence_length: int = 128, unet_channels: int = 64,
    dit_dimension: int = 128, dit_depth: int = 4, dit_heads: int = 4,
) -> nn.Module:
    del unet_channels
    if model_type not in {"dit", "dit_cross"}:
        raise ValueError(f"unsupported inference model type: {model_type}")
    return ConditionalDiT(
        sequence_length=sequence_length, dimension=dit_dimension,
        depth=dit_depth, heads=dit_heads,
        cross_attention=model_type == "dit_cross",
    )
