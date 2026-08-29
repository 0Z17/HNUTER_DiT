"""Canonical-frame transforms shared by training and inference.

The canonical frame is yaw-only: its origin is the start position, its +x
axis follows the horizontal projection from start to goal, and +z remains
parallel to world up.  This keeps gravity-related geometry unchanged while
removing the arbitrary world yaw and translation from the planning problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .model import rotation_6d_to_matrix


CANONICAL_FRAME_VERSION = "start_goal_yaw_z_up_v001"


def _yaw_components_torch(conditions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    start = conditions[..., :3]
    goal = conditions[..., 9:12]
    delta = goal - start
    horizontal = torch.linalg.vector_norm(delta[..., :2], dim=-1)
    fallback = horizontal < 1e-6
    cosine = torch.where(fallback, torch.ones_like(horizontal), delta[..., 0] / horizontal.clamp_min(1e-6))
    sine = torch.where(fallback, torch.zeros_like(horizontal), delta[..., 1] / horizontal.clamp_min(1e-6))
    return start, cosine, sine


def _rotate_vectors_world_to_canonical(value: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    x, y, z = value.unbind(dim=-1)
    while cosine.ndim < x.ndim:
        cosine = cosine.unsqueeze(-1)
        sine = sine.unsqueeze(-1)
    return torch.stack((cosine * x + sine * y, -sine * x + cosine * y, z), dim=-1)


def _rotate_vectors_canonical_to_world(value: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    x, y, z = value.unbind(dim=-1)
    while cosine.ndim < x.ndim:
        cosine = cosine.unsqueeze(-1)
        sine = sine.unsqueeze(-1)
    return torch.stack((cosine * x - sine * y, sine * x + cosine * y, z), dim=-1)


def canonicalize_pose9_batch(poses: Tensor, conditions: Tensor) -> Tensor:
    """Transform ``[B, ..., 9]`` world pose9 values into each sample frame."""

    start, cosine, sine = _yaw_components_torch(conditions)
    position = poses[..., :3] - start[:, None]
    first = _rotate_vectors_world_to_canonical(poses[..., 3:6], cosine, sine)
    second = _rotate_vectors_world_to_canonical(poses[..., 6:9], cosine, sine)
    return torch.cat((_rotate_vectors_world_to_canonical(position, cosine, sine), first, second), dim=-1)


def canonicalize_condition_batch(conditions: Tensor) -> Tensor:
    poses = conditions.reshape(len(conditions), 2, 9)
    return canonicalize_pose9_batch(poses, conditions).reshape(len(conditions), 18)


def canonicalize_obstacles_batch(obstacles: Tensor, conditions: Tensor) -> Tensor:
    """Transform padded ``[B, M, 10]`` OBB tokens into canonical frames."""

    start, cosine, sine = _yaw_components_torch(conditions)
    result = obstacles.clone()
    result[..., :3] = _rotate_vectors_world_to_canonical(
        obstacles[..., :3] - start[:, None], cosine, sine
    )
    # q_canonical = q_z(-yaw) * q_world, in wxyz convention.
    yaw = torch.atan2(sine, cosine)
    half_cosine = torch.cos(0.5 * yaw)
    half_sine = torch.sin(0.5 * yaw)
    w, x, y, z = obstacles[..., 6:10].unbind(dim=-1)
    ch = half_cosine[:, None]
    sh = half_sine[:, None]
    quaternion = torch.stack((
        ch * w + sh * z,
        ch * x + sh * y,
        ch * y - sh * x,
        ch * z - sh * w,
    ), dim=-1)
    quaternion = quaternion * torch.where(
        quaternion[..., :1] < 0, -torch.ones_like(quaternion[..., :1]),
        torch.ones_like(quaternion[..., :1]),
    )
    result[..., 6:10] = quaternion
    return result


def decanonicalize_pose9_batch(poses: Tensor, world_conditions: Tensor) -> Tensor:
    """Transform canonical pose9 values back to their world frames."""

    start, cosine, sine = _yaw_components_torch(world_conditions)
    position = _rotate_vectors_canonical_to_world(poses[..., :3], cosine, sine) + start[:, None]
    first = _rotate_vectors_canonical_to_world(poses[..., 3:6], cosine, sine)
    second = _rotate_vectors_canonical_to_world(poses[..., 6:9], cosine, sine)
    return torch.cat((position, first, second), dim=-1)


@dataclass(frozen=True)
class CanonicalFrame:
    origin: np.ndarray
    cosine: float
    sine: float

    @classmethod
    def from_pose9_condition(cls, condition: np.ndarray) -> "CanonicalFrame":
        condition = np.asarray(condition, dtype=np.float64)
        start = condition[:3]
        delta = condition[9:12] - start
        horizontal = float(np.linalg.norm(delta[:2]))
        if horizontal < 1e-6:
            cosine, sine = 1.0, 0.0
        else:
            cosine, sine = float(delta[0] / horizontal), float(delta[1] / horizontal)
        return cls(start.copy(), cosine, sine)

    def _world_to_canonical_vectors(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        x, y, z = np.moveaxis(value, -1, 0)
        return np.stack((self.cosine * x + self.sine * y, -self.sine * x + self.cosine * y, z), axis=-1)

    def _canonical_to_world_vectors(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        x, y, z = np.moveaxis(value, -1, 0)
        return np.stack((self.cosine * x - self.sine * y, self.sine * x + self.cosine * y, z), axis=-1)

    def pose9_to_canonical(self, poses: np.ndarray) -> np.ndarray:
        poses = np.asarray(poses)
        return np.concatenate((
            self._world_to_canonical_vectors(poses[..., :3] - self.origin),
            self._world_to_canonical_vectors(poses[..., 3:6]),
            self._world_to_canonical_vectors(poses[..., 6:9]),
        ), axis=-1)

    def pose9_to_world(self, poses: np.ndarray) -> np.ndarray:
        poses = np.asarray(poses)
        return np.concatenate((
            self._canonical_to_world_vectors(poses[..., :3]) + self.origin,
            self._canonical_to_world_vectors(poses[..., 3:6]),
            self._canonical_to_world_vectors(poses[..., 6:9]),
        ), axis=-1)

    def obstacles_to_canonical(self, obstacles: np.ndarray) -> np.ndarray:
        obstacles = np.asarray(obstacles)
        result = obstacles.copy()
        result[..., :3] = self._world_to_canonical_vectors(obstacles[..., :3] - self.origin)
        yaw = np.arctan2(self.sine, self.cosine)
        ch, sh = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
        w, x, y, z = np.moveaxis(obstacles[..., 6:10], -1, 0)
        quaternion = np.stack((
            ch * w + sh * z,
            ch * x + sh * y,
            ch * y - sh * x,
            ch * z - sh * w,
        ), axis=-1)
        quaternion *= np.where(quaternion[..., :1] < 0, -1.0, 1.0)
        result[..., 6:10] = quaternion
        return result


def canonical_rotation_matrices(path_metric: Tensor) -> Tensor:
    """Convenience wrapper used by collision losses."""

    return rotation_6d_to_matrix(path_metric[..., 3:9])
