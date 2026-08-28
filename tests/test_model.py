from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

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
