"""Densified robot-sphere clearance against current observed AABB surfaces.

The 42 spheres are the existing DiT collision proxy, not simulation collision
truth. These checks sample an interpolated geometric path; they neither certify
unknown space nor validate a separately smoothed/time-parameterized trajectory.
"""
from __future__ import annotations

import math

import numpy as np

from .collision_loss import ROBOT_SPHERE_CENTERS, ROBOT_SPHERE_RADII
from .model import quaternion_wxyz_to_matrix_numpy


_CENTERS = np.asarray(ROBOT_SPHERE_CENTERS, dtype=np.float64)
_RADII = np.asarray(ROBOT_SPHERE_RADII, dtype=np.float64)
# Bounds the displacement of every proxy-sphere surface point under rotation.
ROBOT_SWEEP_RADIUS = float((np.linalg.norm(_CENTERS, axis=1) + _RADII).max())
EMPTY_OBSERVATION_DISTANCE = 100.0


def _spacing(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("spacing must be finite and positive")
    return value


def _unit_quaternions(values: np.ndarray) -> np.ndarray:
    """Normalize without overflowing for large, otherwise finite inputs."""
    scale = np.abs(values).max(axis=-1, keepdims=True)
    if not np.isfinite(values).all() or (scale == 0).any():
        raise ValueError("quaternions must be finite and nonzero")
    scaled = values / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def _path(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 7 or len(result) == 0:
        raise ValueError("path must be a nonempty Nx7 position/wxyz array")
    if not np.isfinite(result).all():
        raise ValueError("path poses must be finite")
    result = result.copy()
    result[:, 3:] = _unit_quaternions(result[:, 3:])
    return result


def dense_poses(path7: np.ndarray, spacing: float = 0.05) -> np.ndarray:
    """Include endpoints and subdivide translation plus shortest-arc rotation.

    Position is linear and quaternion wxyz follows shortest-arc SLERP. Adjacent
    samples satisfy ``translation + ROBOT_SWEEP_RADIUS * angle <= spacing``.
    This is a spatial resolution bound, not an exact continuous collision test.
    Quaternion signs may change to maintain the same continuous orientation.
    """
    spacing = _spacing(spacing)
    path = _path(path7)
    pieces = [path[:1]]
    previous = path[0].copy()
    for current in path[1:]:
        current = current.copy()
        dot = float(previous[3:] @ current[3:])
        if dot < 0:
            current[3:] *= -1
            dot = -dot
        dot = float(np.clip(dot, 0, 1))
        half_angle = math.acos(dot)
        movement = float(np.linalg.norm(current[:3] - previous[:3])) + ROBOT_SWEEP_RADIUS * (2 * half_angle)
        if not math.isfinite(movement):
            raise ValueError("path segment displacement is not finite")
        count = max(1, math.ceil(movement / spacing))
        fractions = np.arange(1, count + 1, dtype=np.float64) / count
        piece = np.empty((count, 7), dtype=np.float64)
        piece[:, :3] = previous[:3] + fractions[:, None] * (current[:3] - previous[:3])
        if half_angle < 1e-8:
            quaternions = (1 - fractions[:, None]) * previous[3:] + fractions[:, None] * current[3:]
        else:
            quaternions = (
                np.sin((1 - fractions[:, None]) * half_angle) * previous[3:]
                + np.sin(fractions[:, None] * half_angle) * current[3:]
            ) / math.sin(half_angle)
        piece[:, 3:] = _unit_quaternions(quaternions)
        # Preserve normalized input endpoints without interpolation roundoff.
        piece[-1] = current
        pieces.append(piece)
        previous = current
    return np.concatenate(pieces, axis=0)


def _boxes(value: np.ndarray) -> np.ndarray:
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.shape == (0,):
        boxes = boxes.reshape(0, 10)
    if boxes.ndim != 2 or boxes.shape[1] != 10:
        raise ValueError("boxes must have shape (N, 10)")
    if not np.isfinite(boxes).all() or (boxes[:, 3:6] <= 0).any():
        raise ValueError("boxes must be finite with strictly positive full extents")
    if len(boxes):
        orientation = _unit_quaternions(boxes[:, 6:10])
        if not np.allclose(orientation[:, 1:], 0, atol=1e-8, rtol=0):
            raise ValueError("only world-axis-aligned boxes with identity rotation are supported")
    return boxes


def observed_clearance(paths7: np.ndarray, boxesNx10: np.ndarray, spacing: float = 0.05) -> np.ndarray:
    """Return each path's minimum sampled 42-sphere/AABB signed distance.

    Accepts BxNx7 paths (or a single Nx7 path, returning a one-item array), and
    active boxes with full extents and identity wxyz rotation. Negative distance
    means overlap with the observed proxy. All original endpoints are checked.

    With no boxes the finite sentinel 100.0 means **no observed constraints**,
    never free space or a safety certificate. Callers must retain that unknown
    status separately. Distance can change between samples by up to spacing/2;
    use an appropriate margin and recheck the actual executed trajectory.
    """
    spacing = _spacing(spacing)
    paths = np.asarray(paths7, dtype=np.float64)
    if paths.ndim == 2:
        paths = paths[None, ...]
    if paths.ndim != 3 or paths.shape[2] != 7 or paths.shape[1] == 0:
        raise ValueError("paths must have shape (B, N, 7), N >= 1")
    boxes = _boxes(boxesNx10)
    result = np.empty(len(paths), dtype=np.float64)
    for index, path in enumerate(paths):
        poses = dense_poses(path, spacing)
        if not len(boxes):
            result[index] = EMPTY_OBSERVATION_DISTANCE
            continue
        rotation = quaternion_wxyz_to_matrix_numpy(poses[:, 3:])
        centers = poses[:, None, :3] + np.einsum("nij,sj->nsi", rotation, _CENTERS)
        minimum = math.inf
        # Bound working memory independently of the number of obstacle tokens.
        for box in boxes:
            delta = np.abs(centers - box[:3]) - box[3:6] / 2
            outside = np.linalg.norm(np.maximum(delta, 0), axis=-1)
            inside = np.minimum(delta.max(axis=-1), 0)
            minimum = min(minimum, float((outside + inside - _RADII).min()))
        if not math.isfinite(minimum):
            raise ValueError("observed distance cannot be represented as a finite number")
        result[index] = minimum
    return result
