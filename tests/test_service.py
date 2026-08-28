from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hnuter_dit.service import DiTTrajectoryService


def _service_without_model() -> DiTTrajectoryService:
    service = DiTTrajectoryService.__new__(DiTTrajectoryService)
    from hnuter_dit.service import ReferenceCollector
    import threading
    from collections import OrderedDict
    service.collector = ReferenceCollector()
    service.runtime = SimpleNamespace(diffusion_steps=100)
    service._checker_cache = OrderedDict()
    service._checker_lock = threading.Lock()
    return service


def test_random_scene_has_editable_valid_endpoints_and_tokens() -> None:
    service = _service_without_model()
    payload = service.generate_scene_payload({
        "family": "central_block", "seed": 123,
        "obstacle_count_min": 8, "obstacle_count_max": 8,
    })
    assert payload["validation"]["valid"] is True
    assert payload["validation"]["active_tokens"] == 8
    assert len(payload["start_pose"]) == len(payload["goal_pose"]) == 7


def test_manual_endpoint_collision_is_rejected() -> None:
    service = _service_without_model()
    payload = service.generate_scene_payload({
        "family": "central_block", "seed": 123,
        "obstacle_count_min": 8, "obstacle_count_max": 8,
    })
    obstacle = next(item for item in payload["scene"]["obstacles"] if item.get("role") != "floor")
    bad_start = [*obstacle["pose"]["position"], 1, 0, 0, 0]
    result = service.validate({
        "scene": payload["scene"],
        "start_pose": bad_start,
        "goal_pose": payload["goal_pose"],
    })
    assert result["valid"] is False
    assert any("起点" in issue for issue in result["issues"])
