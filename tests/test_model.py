from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hnuter_dit.collision_loss import sphere_obb_signed_distances
from hnuter_dit.geometry import (
    CanonicalFrame, canonicalize_condition_batch, canonicalize_obstacles_batch,
    canonicalize_pose9_batch, decanonicalize_pose9_batch,
)
from hnuter_dit.model import build_model, pose7_to_pose9, pose9_to_pose7_numpy


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT / "artifacts" / "base_free_flight_159103traj_dit_v001"
    / "models" / "dit_cross" / "best.pt"
)


def test_checkpoint_loads_strictly_into_local_inference_model() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_type"], **checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_964_169
    assert checkpoint["training_step"] == 48_000


def test_pose7_pose9_roundtrip_preserves_pose() -> None:
    pose7 = np.asarray([
        [1.0, -2.0, 1.5, 1.0, 0.0, 0.0, 0.0],
        [-1.0, 2.0, 2.5, 0.9238795, 0.0, 0.3826834, 0.0],
    ])
    recovered = pose9_to_pose7_numpy(pose7_to_pose9(pose7))
    np.testing.assert_allclose(recovered[:, :3], pose7[:, :3], atol=1e-7)
    assert np.allclose(np.abs(np.sum(recovered[:, 3:] * pose7[:, 3:], axis=1)), 1, atol=1e-6)


def test_canonical_frame_roundtrip_and_axis_alignment() -> None:
    conditions = torch.tensor([[
        1.0, 2.0, 0.5, 1, 0, 0, 0, 1, 0,
        4.0, 6.0, 1.5, 0, 1, 0, -1, 0, 0,
    ]], dtype=torch.float32)
    path = conditions.reshape(1, 2, 9)
    canonical = canonicalize_pose9_batch(path, conditions)
    assert torch.allclose(canonical[0, 0, :3], torch.zeros(3), atol=1e-6)
    assert canonical[0, 1, 0] > 0
    assert abs(float(canonical[0, 1, 1])) < 1e-6
    assert abs(float(canonical[0, 1, 2]) - 1.0) < 1e-6
    recovered = decanonicalize_pose9_batch(canonical, conditions)
    assert torch.allclose(recovered, path, atol=1e-6)


def test_numpy_and_torch_canonical_transforms_match() -> None:
    condition = np.asarray([
        1.0, -1.0, 0.8, 1, 0, 0, 0, 1, 0,
        -2.0, 2.0, 1.3, 0, 1, 0, -1, 0, 0,
    ], dtype=np.float32)
    obstacle = np.asarray([[[0.5, 0.2, 1.0, 1, 2, 3, 1, 0, 0, 0]]], dtype=np.float32)
    frame = CanonicalFrame.from_pose9_condition(condition)
    numpy_value = frame.obstacles_to_canonical(obstacle[0])
    torch_value = canonicalize_obstacles_batch(
        torch.from_numpy(obstacle), torch.from_numpy(condition[None])
    )[0].numpy()
    np.testing.assert_allclose(numpy_value, torch_value, atol=1e-6)
    canonical_condition = canonicalize_condition_batch(torch.from_numpy(condition[None]))
    assert abs(float(canonical_condition[0, 10])) < 1e-6


def test_canonical_obstacle_quaternion_is_invariant_across_yaw_wrap() -> None:
    yaw = np.deg2rad(170.0)
    rotation = np.deg2rad(90.0)
    start = np.asarray([0, 0, 1], dtype=np.float32)
    goal = np.asarray([np.cos(yaw), np.sin(yaw), 1], dtype=np.float32)
    identity6 = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    condition = np.concatenate((start, identity6, goal, identity6))[None]
    obstacle_yaw = np.deg2rad(35.0)
    obstacle = np.asarray([[[
        0.3, -0.5, 1.2, 1, 2, 3,
        np.cos(obstacle_yaw / 2), 0, 0, np.sin(obstacle_yaw / 2),
    ]]], dtype=np.float32)
    cosine, sine = np.cos(rotation), np.sin(rotation)
    rotated_condition = condition.copy()
    for offset in (0, 9):
        x, y = condition[:, offset].copy(), condition[:, offset + 1].copy()
        rotated_condition[:, offset] = cosine * x - sine * y
        rotated_condition[:, offset + 1] = sine * x + cosine * y
    rotated_obstacle = obstacle.copy()
    x, y = obstacle[..., 0].copy(), obstacle[..., 1].copy()
    rotated_obstacle[..., 0] = cosine * x - sine * y
    rotated_obstacle[..., 1] = sine * x + cosine * y
    qz = np.asarray([np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)], dtype=np.float32)
    w, xq, yq, z = np.moveaxis(obstacle[..., 6:10], -1, 0)
    qw, _, _, qzv = qz
    rotated_obstacle[..., 6:10] = np.stack((
        qw * w - qzv * z, qw * xq - qzv * yq,
        qw * yq + qzv * xq, qw * z + qzv * w,
    ), axis=-1)
    first = canonicalize_obstacles_batch(
        torch.from_numpy(obstacle), torch.from_numpy(condition)
    )
    second = canonicalize_obstacles_batch(
        torch.from_numpy(rotated_obstacle), torch.from_numpy(rotated_condition)
    )
    assert torch.allclose(first, second, atol=2e-6)


def test_relative_geometry_dit_uses_pair_features() -> None:
    model = build_model(
        "dit_rel_cross", sequence_length=16, dit_dimension=32,
        dit_depth=1, dit_heads=4,
    )
    model.configure_path_normalization(np.zeros(9), np.ones(9))
    path = torch.randn(2, 16, 9)
    path[..., 3:9] = torch.tensor([1, 0, 0, 0, 1, 0])
    context = torch.cat((path[:, 0], path[:, -1]), dim=-1)
    obstacles = torch.zeros(2, 3, 10)
    obstacles[..., 3:6] = 1
    obstacles[..., 6] = 1
    mask = torch.ones(2, 3, dtype=torch.bool)
    output = model(path, torch.tensor([1, 50]), context, obstacles, mask)
    assert output.shape == path.shape
    output.square().mean().backward()
    parameter = model.blocks[0].relative_cross_attention.relative_value
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


def test_collision_sphere_distance_detects_obstacle_intersection() -> None:
    pose = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float32)
    path = pose.repeat(1, 10, 1)
    obstacle = torch.tensor([[[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]]], dtype=torch.float32)
    mask = torch.ones(1, 1, dtype=torch.bool)
    colliding = sphere_obb_signed_distances(path, obstacle, mask, waypoint_stride=1)
    far_path = path.clone()
    far_path[..., 0] = 5
    free = sphere_obb_signed_distances(far_path, obstacle, mask, waypoint_stride=1)
    assert float(colliding.min()) < 0
    assert float(free.min()) > 0
