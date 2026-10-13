"""A safe polyline does not authorize a different smoothed execution curve."""

from types import SimpleNamespace

import numpy as np
import pytest

from hnuter_dit.interactive import InteractiveDiTService, rank_candidates


def test_smoothed_curve_cannot_reuse_original_polyline_clearance():
    pytest.importorskip("toppra")
    from hnuter_dit.retiming import TimedPoseReference

    original = np.array([[0, 0, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0]], dtype=float)
    source, segments = original[:-1, :3], np.diff(original[:, :3], axis=0)

    class TightCorridor:
        backend_name = "controlled_tight_corridor"

        def clearance(self, positions, quaternions):
            relative = np.asarray(positions)[:, None, :] - source[None, :, :]
            fraction = np.clip(np.sum(relative * segments, axis=-1) / np.sum(segments**2, axis=1), 0, 1)
            distances = np.linalg.norm(relative - fraction[:, :, None] * segments, axis=-1).min(axis=1)
            # The source polyline has >2 cm clearance, but even a small C2
            # corner deviation violates this deliberately narrow corridor.
            return .020001 - distances

    checker = TightCorridor()
    base = SimpleNamespace(
        _checker=lambda scene: checker,
        collector=SimpleNamespace(quaternion_roll_pitch_degrees=lambda q: [0, 0]),
    )
    service = InteractiveDiTService(base, SimpleNamespace())
    assert service.validate_execution_reference({}, original)["minimum_clearance_m"] > .02
    reference = TimedPoseReference(original)
    exact_curve = reference.audit_poses()
    with pytest.raises(ValueError, match="净空复检"):
        service.validate_execution_reference({}, exact_curve)


def test_placeholder_clearance_cannot_authorize_execution():
    class PlaceholderChecker:
        backend_name = "urdf_primitive_obb_conservative"
        supports_clearance_gradient = False

        def clearance(self, positions, quaternions):
            pytest.fail("placeholder distances must be rejected before querying")

    base = SimpleNamespace(_checker=lambda scene: PlaceholderChecker())
    service = InteractiveDiTService(base, SimpleNamespace())
    path = np.array([[0, 0, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0]], dtype=float)
    with pytest.raises(RuntimeError, match="COAL 真实净空"):
        service.validate_execution_reference({}, path)
    with pytest.raises(RuntimeError, match="COAL 真实净空"):
        rank_candidates({"trajectories": []}, {}, base)
