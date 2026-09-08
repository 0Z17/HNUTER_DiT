"""Checkpoint-compatible HNUTER environment-conditioned DiT inference.

This is the inference-only subset of ``HNUTER-diffusion/se3_diffusion.py`` at
commit ``c7485fef14b1e1036a9ce529093fe8218b57132e``.  It intentionally preserves
module and parameter names so the trained Base checkpoint loads strictly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


POSE_DIM = 9
CONDITION_DIM = 18
OBSTACLE_DIM = 10
RELATIVE_GEOMETRY_DIM = 13


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


def quaternion_wxyz_to_matrix_torch(quaternion: Tensor) -> Tensor:
    quaternion = F.normalize(quaternion, dim=-1, eps=1e-8)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
        2 * (x * z + y * w), 2 * (x * y + z * w),
        1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(quaternion.shape[:-1] + (3, 3))


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


class PairwiseGeometryCrossAttention(nn.Module):
    """Cross-attention with pairwise SE(3) geometry in logits and values."""

    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("attention dimension must be divisible by heads")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.relative_bias = nn.Sequential(
            nn.Linear(RELATIVE_GEOMETRY_DIM, 64), nn.SiLU(),
            nn.Linear(64, heads),
        )
        # Linear value injection can be contracted after attention weighting,
        # avoiding a [B,H,L,M,Dh] pairwise value tensor.
        self.relative_value = nn.Parameter(torch.zeros(
            heads, RELATIVE_GEOMETRY_DIM, self.head_dimension
        ))
        nn.init.zeros_(self.relative_bias[-1].weight)
        nn.init.zeros_(self.relative_bias[-1].bias)

    def _heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.heads, self.head_dimension).transpose(1, 2)

    def forward(
        self, query: Tensor, environment: Tensor, relative_geometry: Tensor,
        environment_mask: Tensor | None,
    ) -> Tensor:
        q = self._heads(self.query(query))
        k = self._heads(self.key(environment))
        v = self._heads(self.value(environment))
        logits = torch.einsum("bhld,bhmd->bhlm", q, k) / math.sqrt(self.head_dimension)
        logits = logits + self.relative_bias(relative_geometry).permute(0, 3, 1, 2)
        if environment_mask is not None:
            logits = logits.masked_fill(
                ~environment_mask[:, None, None], -torch.finfo(logits.dtype).max
            )
        weights = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bhlm,bhmd->bhld", weights, v)
        attended = attended + torch.einsum(
            "bhlm,blmr,hrd->bhld", weights, relative_geometry,
            self.relative_value,
        )
        attended = attended.transpose(1, 2).reshape(query.shape)
        return self.output(attended)


class DiTBlock(nn.Module):
    def __init__(
        self, dimension: int, heads: int, condition_dim: int,
        cross_attention: bool, relative_geometry_attention: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.cross_attention = (
            nn.MultiheadAttention(dimension, heads, batch_first=True)
            if cross_attention and not relative_geometry_attention else None
        )
        self.relative_cross_attention = (
            PairwiseGeometryCrossAttention(dimension, heads)
            if relative_geometry_attention else None
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
        relative_geometry: Tensor | None = None,
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
        elif self.relative_cross_attention is not None:
            if environment is None or relative_geometry is None:
                raise ValueError("relative cross-attention requires environment and geometry")
            attended = self.relative_cross_attention(
                self.norm_cross(value), environment, relative_geometry,
                environment_mask,
            )
            value = value + attended
        hidden = self.norm2(value) * (1 + scale2[:, None]) + shift2[:, None]
        return value + gate2[:, None] * self.mlp(hidden)


class ConditionalDiT(nn.Module):
    def __init__(
        self, sequence_length: int = 128, path_dim: int = POSE_DIM,
        context_dim: int = CONDITION_DIM, dimension: int = 128,
        depth: int = 4, heads: int = 4, cross_attention: bool = False,
        relative_geometry_attention: bool = False,
    ) -> None:
        super().__init__()
        if relative_geometry_attention and not cross_attention:
            raise ValueError("relative geometry attention requires cross-attention")
        self.model_type = (
            "dit_rel_cross" if relative_geometry_attention
            else "dit_cross" if cross_attention else "dit"
        )
        self.cross_attention = cross_attention
        self.relative_geometry_attention = relative_geometry_attention
        self.dimension = dimension
        self.register_buffer("environment_geometry_mean", torch.zeros(6), persistent=False)
        self.register_buffer("environment_geometry_std", torch.ones(6), persistent=False)
        self.register_buffer("path_geometry_mean", torch.zeros(POSE_DIM), persistent=False)
        self.register_buffer("path_geometry_std", torch.ones(POSE_DIM), persistent=False)
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
            DiTBlock(
                dimension, heads, dimension, cross_attention,
                relative_geometry_attention,
            )
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

    def configure_path_normalization(
        self, mean: np.ndarray | Tensor, std: np.ndarray | Tensor,
    ) -> None:
        mean_tensor = torch.as_tensor(
            mean, dtype=self.path_geometry_mean.dtype,
            device=self.path_geometry_mean.device,
        )
        std_tensor = torch.as_tensor(
            std, dtype=self.path_geometry_std.dtype,
            device=self.path_geometry_std.device,
        )
        if mean_tensor.shape != (POSE_DIM,) or std_tensor.shape != (POSE_DIM,):
            raise ValueError("path geometry normalization must have shape [9]")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
            raise ValueError("path normalization must be finite")
        if not (std_tensor > 0).all():
            raise ValueError("path geometry std must be positive")
        self.path_geometry_mean.copy_(mean_tensor)
        self.path_geometry_std.copy_(std_tensor)

    def pairwise_relative_geometry(
        self, path: Tensor, obstacles: Tensor,
    ) -> Tensor:
        """Build body-frame trajectory/OBB pair features ``[B,L,M,13]``."""

        metric_path = path * self.path_geometry_std + self.path_geometry_mean
        path_rotation = rotation_6d_to_matrix(metric_path[..., 3:9].float()).to(path.dtype)
        obstacle_rotation = quaternion_wxyz_to_matrix_torch(
            obstacles[..., 6:10].float()
        ).to(path.dtype)
        delta = obstacles[:, None, :, :3] - metric_path[:, :, None, :3]
        body_delta = torch.einsum("blmi,blij->blmj", delta, path_rotation)
        distance = torch.linalg.vector_norm(body_delta, dim=-1, keepdim=True)
        relative_rotation = torch.matmul(
            path_rotation.transpose(-1, -2)[:, :, None],
            obstacle_rotation[:, None],
        )
        rotation6 = torch.cat((
            relative_rotation[..., :, 0], relative_rotation[..., :, 1]
        ), dim=-1)
        extents = obstacles[:, None, :, 3:6].expand(
            -1, path.shape[1], -1, -1
        )
        # Metric features are scaled to a numerically compact range; rotation
        # columns are already bounded by [-1, 1].
        return torch.cat((
            body_delta / 3.0, distance / 3.0, extents / 2.0, rotation6
        ), dim=-1)

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
        relative_geometry = None
        if self.relative_geometry_attention:
            if obstacles is None:
                obstacles = torch.zeros(
                    (len(path), 1, OBSTACLE_DIM), dtype=path.dtype,
                    device=path.device,
                )
            elif obstacles.ndim == 2:
                obstacles = obstacles[None].expand(len(path), -1, -1)
            relative_geometry = self.pairwise_relative_geometry(path, obstacles)
        for block in self.blocks:
            value = block(
                value, condition, environment, environment_mask,
                relative_geometry,
            )
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


@torch.no_grad()
def ddim_sample(
    model: nn.Module, conditions: Tensor, schedule: DiffusionSchedule,
    sequence_length: int, sampling_steps: int, obstacles: Tensor,
    obstacle_mask: Tensor, generator: torch.Generator,
    clip_x0: float = 4.0,
    clean_guidance: Callable[[Tensor, int, int, int], Tensor] | None = None,
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
        if clean_guidance is not None:
            predicted_x0 = clean_guidance(
                predicted_x0, sample_index, len(indices), int(timestep_value)
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
    if model_type not in {"dit", "dit_cross", "dit_rel_cross"}:
        raise ValueError(f"unsupported inference model type: {model_type}")
    return ConditionalDiT(
        sequence_length=sequence_length, dimension=dit_dimension,
        depth=dit_depth, heads=dit_heads,
        cross_attention=model_type in {"dit_cross", "dit_rel_cross"},
        relative_geometry_attention=model_type == "dit_rel_cross",
    )
