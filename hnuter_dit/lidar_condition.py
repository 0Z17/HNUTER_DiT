"""Current LiDAR returns to the existing DiT's bounded box-token format.

These tokens conservatively cover the *observed surface points* inside a local
ROI. They do not reconstruct unobserved object backs, track motion, certify
free space, or make a no-return direction safe. No scene geometry is accessed.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3D vector")
    return result


def _points(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape == (0,):
        result = result.reshape(0, 3)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(result).all():
        raise ValueError("points must be finite; invalid samples cannot silently become free space")
    return result


def _partitions(points: np.ndarray, maximum: int) -> list[np.ndarray]:
    """Deterministic balanced spatial splits; every input belongs to one leaf."""
    # Lexicographic initialization makes grouping independent of return order.
    ordered = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    groups = [ordered]
    while len(groups) < maximum:
        candidate, axis, priority = None, None, None
        for index, members in enumerate(groups):
            if len(members) < 2:
                continue
            span = np.ptp(points[members], axis=0)
            dimension = int(np.argmax(span))
            if span[dimension] <= 0:
                continue
            score = (float(span[dimension]), len(members), -index)
            if priority is None or score > priority:
                candidate, axis, priority = index, dimension, score
        if candidate is None:
            break
        members = groups[candidate]
        order = np.argsort(points[members, axis], kind="stable")
        members = members[order]
        split = len(members) // 2
        groups[candidate:candidate + 1] = [members[:split], members[split:]]
    return groups


def _outward_float32_box(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep conservative bounds when centers/full extents become float32."""
    with np.errstate(over="ignore", invalid="ignore"):
        center = ((lower + upper) * 0.5).astype(np.float32)
        needed = 2 * np.maximum(center.astype(float) - lower, upper - center.astype(float))
        extent = np.nextafter(needed.astype(np.float32), np.float32(np.inf))
    if not np.isfinite(center).all() or not np.isfinite(extent).all():
        raise ValueError("point coordinates cannot be represented by finite float32 tokens")
    return center, extent


def points_to_tokens(
    points_world: Any,
    robot_position: Any,
    goal_position: Any,
    max_tokens: int = 32,
    range_m: float = 8.0,
    cell_padding: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(tokens, mask, metadata)`` from one current world-space scan.

    Tokens have shape ``(max_tokens, 10)`` and contain center xyz, **full**
    extent xyz, and quaternion wxyz. All active boxes are axis-aligned.
    The ROI is a sphere centered on the measured robot position; the goal only
    determines token ordering and never removes rear/side obstacle returns.

    When there are more points/spatial groups than slots, groups are merged
    implicitly by stopping the spatial partition tree. Every ROI point remains
    inside a padded box; nothing is randomly sampled or truncated. This may
    over-cover gaps and produce conservative false obstacles. Padding accounts
    for discretization/measurement uncertainty around observed points, but
    cannot recover the LiDAR blind zone or other unobserved regions.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, (int, np.integer)) or not 1 <= max_tokens <= 32:
        raise ValueError("max_tokens must be an integer in [1, 32]")
    if not np.isfinite(range_m) or range_m <= 0:
        raise ValueError("range_m must be finite and positive")
    if not np.isfinite(cell_padding) or cell_padding < 0:
        raise ValueError("cell_padding must be finite and nonnegative")
    points = _points(points_world)
    robot, goal = _vector(robot_position, "robot_position"), _vector(goal_position, "goal_position")
    relative = points - robot
    distance = np.linalg.norm(relative, axis=1)
    keep = distance <= float(range_m)
    roi = points[keep]
    tokens = np.zeros((max_tokens, 10), dtype=np.float32)
    tokens[:, 6] = 1.0  # Well-defined identity rotations even on padded rows.
    mask = np.zeros(max_tokens, dtype=bool)
    metadata: dict[str, Any] = {
        "source": "current_lidar_scan_only",
        "algorithm": "deterministic_longest_axis_median_partition",
        "input_points": len(points),
        "roi_points": len(roi),
        "outside_roi_points": int((~keep).sum()),
        "dropped_roi_points": 0,
        "active_tokens": 0,
        "range_m": float(range_m),
        "roi_center_world": robot.tolist(),
        "cell_padding_m": float(cell_padding),
        "extent_convention": "full_lengths",
        "quaternion_convention": "wxyz",
        "coordinate_frame": "world_z_up",
        "observed_surface_coverage_complete": True,
        "free_space_certified": False,
        "unobserved_space": "unknown; absence of tokens is not free-space evidence",
        "scan_points_sha256": hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest(),
        "token_point_counts": [],
    }
    if not len(roi):
        return tokens, mask, metadata

    groups = _partitions(roi, int(max_tokens))
    boxes = []
    segment = goal - robot
    length_squared = float(segment @ segment)
    for members in groups:
        points_in_box = roi[members]
        lower = points_in_box.min(axis=0) - cell_padding
        upper = points_in_box.max(axis=0) + cell_padding
        center, extent = _outward_float32_box(lower, upper)
        offset = center.astype(float) - robot
        fraction = float(np.clip(offset @ segment / length_squared, 0, 1)) if length_squared > 1e-12 else 0.0
        route_distance = float(np.linalg.norm(offset - fraction * segment))
        order = (route_distance, float(np.linalg.norm(offset)), *offset)
        boxes.append((order, center, extent, len(members)))
    boxes.sort(key=lambda box: box[0])
    for index, (_, center, extent, count) in enumerate(boxes):
        tokens[index, :3], tokens[index, 3:6] = center, extent
        mask[index] = True
        metadata["token_point_counts"].append(count)
    metadata["active_tokens"] = int(mask.sum())
    return tokens, mask, metadata


def frame_to_tokens(
    frame: Any,
    robot_position: Any,
    goal_position: Any,
    **options: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Transform a snapshot LiDARFrame's sensor-FLU returns into world tokens.

    Requires ``points``, ``world_T_sensor`` and ``sim_time`` attributes. An
    optional point_timestamps array must agree with this simulator's snapshot
    timestamp. Rolling scans need a separate deskew adapter, not this function.
    """
    points = _points(frame.points)
    pose = np.asarray(frame.world_T_sensor, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("world_T_sensor must be a finite 4x4 rigid transform")
    rotation = pose[:3, :3]
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7) or not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-5
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("world_T_sensor must contain an orthonormal right-handed rotation")
    timestamp = float(frame.sim_time)
    if not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError("scan simulation timestamp must be finite and nonnegative")
    point_times = getattr(frame, "point_timestamps", None)
    if point_times is not None:
        times = np.asarray(point_times, dtype=np.float64)
        if times.shape != (len(points),) or not np.isfinite(times).all() or not np.allclose(times, timestamp, atol=1e-8, rtol=0):
            raise ValueError("snapshot point timestamps must match the scan simulation timestamp")
    world_points = points @ rotation.T + pose[:3, 3]
    tokens, mask, metadata = points_to_tokens(world_points, robot_position, goal_position, **options)
    frame_index = getattr(frame, "frame_index", None)
    metadata.update(
        simulation_time=timestamp,
        frame_index=int(frame_index) if frame_index is not None else None,
        sensor_position_world=pose[:3, 3].tolist(),
        world_T_sensor=pose.tolist(),
    )
    return tokens, mask, metadata
