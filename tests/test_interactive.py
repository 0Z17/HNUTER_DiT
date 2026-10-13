from __future__ import annotations

import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hnuter_dit.interactive import (
    InteractiveDiTService,
    densify_poses,
    handler_type,
    rank_candidates,
)


START = [-1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
GOAL = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def _scene() -> dict:
    return {
        "environment_id": "fixture_seed_42",
        "generation_seed": 42,
        "obstacles": [],
        "sampling_space": {
            "position_bounds": {"min": [-3.3, -3.3, 0.65], "max": [3.3, 3.3, 3.5]},
        },
    }


def _path(y: float = 0.0) -> list[list[float]]:
    return [START.copy(), [0.0, y, 1.0, 1.0, 0.0, 0.0, 0.0], GOAL.copy()]


def _candidate(index: int, poses=None, valid: bool = True) -> dict:
    path = _path() if poses is None else poses
    segments = np.diff(np.asarray(path)[:, :3], axis=0)
    return {
        "index": index,
        "poses_wxyz": path,
        "metrics": {
            "valid": valid,
            "minimum_clearance_m": 0.2,
            "path_length_m": float(np.linalg.norm(segments, axis=1).sum()),
            "p95_turn_angle_deg": 0.0,
            "maximum_turn_angle_deg": 0.0,
            "acceleration_rms_m_per_index2": 0.0,
            "jerk_rms_m_per_index3": 0.0,
            "backtracking_fraction": 0.0,
        },
    }


def _prediction(candidates: list[dict]) -> dict:
    return {
        "start_pose": START.copy(),
        "goal_pose": GOAL.copy(),
        "trajectories": copy.deepcopy(candidates),
        "summary": {"requested_count": len(candidates), "valid_count": len(candidates)},
        "validation": {"valid": True, "issues": []},
    }


class FakeChecker:
    backend_name = "coal_full_urdf_nearest_point"

    def __init__(self, distance=None):
        self.distance = distance
        self.queries: list[np.ndarray] = []

    def clearance(self, positions, quaternions):
        xyz = np.asarray(positions, dtype=float)
        self.queries.append(xyz.copy())
        if self.distance is None:
            return np.full(xyz.shape[:-1], 0.2)
        return np.asarray(self.distance(xyz))


class FakeBaseService:
    def __init__(self, checker=None):
        self.checker = checker or FakeChecker()
        self.collector = SimpleNamespace(quaternion_roll_pitch_degrees=lambda q: (0.0, 0.0))
        self.runtime = SimpleNamespace(diffusion_steps=100, describe=lambda: {"device": "cpu"})
        self.requests = []

    def _checker(self, scene):
        return self.checker

    def catalog(self):
        return {"status": "ok", "families": ["fixture"], "model": {"device": "cpu"}}

    def generate_scene_payload(self, request):
        return {
            "scene": _scene(), "start_pose": START.copy(), "goal_pose": GOAL.copy(),
            "validation": {"valid": True, "issues": []},
        }

    def randomize_endpoints(self, request):
        return {
            "scene": copy.deepcopy(request["scene"]),
            "start_pose": START.copy(), "goal_pose": GOAL.copy(),
            "validation": {"valid": True, "issues": []},
        }

    def validate(self, request):
        return {"valid": True, "issues": [], "collision_backend": self.checker.backend_name}

    def sample_trajectories(self, request):
        self.requests.append(copy.deepcopy(request))
        result = _prediction([_candidate(i) for i in range(request["count"])])
        result["start_pose"] = copy.deepcopy(request["start_pose"])
        result["goal_pose"] = copy.deepcopy(request["goal_pose"])
        return result


class FakeExecution:
    def __init__(self):
        self.started = []
        self.previews = []
        self.stopped = False

    def describe(self):
        return {"available": True, "device": "cpu"}

    def preview(self, *args, **kwargs):
        self.previews.append((copy.deepcopy(args), copy.deepcopy(kwargs)))
        return {"status": "preview"}

    def start(self, *args, **kwargs):
        self.started.append((copy.deepcopy(args), copy.deepcopy(kwargs)))
        return {"status": "running", "run_id": "fixture_run"}

    def status(self):
        return {"state": "idle"}

    def stop(self):
        self.stopped = True
        return {"status": "stopped"}

    def close(self):
        self.stopped = True


def test_densification_checks_translation_gaps_and_preserves_endpoints():
    path = np.asarray([START, GOAL])
    dense = densify_poses(path, max_translation_step_m=0.03)
    assert len(dense) > 2
    np.testing.assert_allclose(dense[0], path[0])
    np.testing.assert_allclose(dense[-1], path[-1])
    assert np.max(np.linalg.norm(np.diff(dense[:, :3], axis=0), axis=1)) <= 0.03 + 1e-12
    np.testing.assert_allclose(np.linalg.norm(dense[:, 3:7], axis=1), 1.0)


def test_densification_uses_shortest_arc_for_antipodal_quaternions():
    path = np.asarray([START, GOAL])
    path[-1, 3:] *= -1.0
    dense = densify_poses(path)
    assert np.isfinite(dense).all()
    np.testing.assert_allclose(np.abs(dense[:, 3]), 1.0)
    np.testing.assert_allclose(dense[:, 4:], 0.0)


def test_densification_limits_rotation_even_without_translation():
    path = np.asarray([START, START])
    angle = np.deg2rad(90.0)
    path[-1, 3:] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
    dense = densify_poses(path, max_rotation_step_deg=3.0)
    adjacent_dots = np.abs(np.sum(dense[1:, 3:] * dense[:-1, 3:], axis=1))
    angular_steps = np.rad2deg(2 * np.arccos(np.clip(adjacent_dots, 0, 1)))
    assert len(dense) >= 31
    assert np.max(angular_steps) <= 3.0 + 1e-8


@pytest.mark.parametrize("path", [[], [[0.0] * 7], [[float("nan")] * 7] * 2])
def test_densification_rejects_invalid_paths(path):
    with pytest.raises(ValueError):
        densify_poses(path)


def test_rank_rejects_between_waypoint_collision():
    def obstacle_clearance(xyz):
        return np.abs(xyz[:, 0] + 0.5) - 0.04

    base = FakeBaseService(FakeChecker(obstacle_clearance))
    result = _prediction([_candidate(0)])
    rank_candidates(result, _scene(), base)
    assert result["final_candidate_id"] is None
    selected = result["selection"]["candidates"][0]
    assert selected["eligible"] is False
    assert selected["dense_sample_count"] > len(_path())
    assert selected["minimum_dense_clearance_m"] < 0.0


def test_rank_chooses_shorter_valid_path_and_breaks_ties_deterministically():
    base = FakeBaseService()
    result = _prediction([_candidate(0, _path(1.0)), _candidate(1), _candidate(2)])
    rank_candidates(result, _scene(), base)
    assert result["final_candidate_id"] == 1
    repeated = _prediction([_candidate(0, _path(1.0)), _candidate(1), _candidate(2)])
    rank_candidates(repeated, _scene(), base)
    assert repeated["selection"] == result["selection"]


def test_rank_does_not_fall_back_to_invalid_candidate():
    base = FakeBaseService(FakeChecker(lambda xyz: np.full(len(xyz), -0.1)))
    result = _prediction([_candidate(i) for i in range(8)])
    rank_candidates(result, _scene(), base)
    assert result["final_candidate_id"] is None
    assert not any(item["eligible"] for item in result["selection"]["candidates"])


def test_rank_rejects_candidate_that_misses_requested_endpoint():
    wrong_path = _path()
    wrong_path[-1][1] = 0.3
    result = _prediction([_candidate(0, wrong_path)])
    rank_candidates(result, _scene(), FakeBaseService())
    assert result["final_candidate_id"] is None
    assert result["selection"]["candidates"][0]["eligible"] is False
    assert "起终点" in result["selection"]["candidates"][0]["reason"]


def test_rank_accepts_equivalent_endpoint_quaternion_signs():
    path = _path()
    path[-1][3:] = [-value for value in path[-1][3:]]
    result = _prediction([_candidate(0, path)])
    rank_candidates(result, _scene(), FakeBaseService())
    assert result["final_candidate_id"] == 0


def _request():
    return {"scene": _scene(), "start_pose": START.copy(), "goal_pose": GOAL.copy()}


def test_interactive_sampler_forces_eight_smooth_candidates():
    base, execution = FakeBaseService(), FakeExecution()
    service = InteractiveDiTService(base, execution)
    request = {**_request(), "count": 1, "guidance_mode": "pointwise", "guidance_steps": 0}
    result = service.sample_trajectories(request)
    assert len(result["trajectories"]) == 8
    assert base.requests[-1]["count"] == 8
    assert base.requests[-1]["guidance_mode"] == "smooth"
    assert base.requests[-1]["guidance_steps"] == 8
    assert request["count"] == 1
    assert result["plan_id"]


def test_sampler_rejects_output_pair_and_paths_matching_a_wrong_request():
    class WrongPairBase(FakeBaseService):
        def sample_trajectories(self, request):
            result = super().sample_trajectories(request)
            result["start_pose"][1] = 0.3
            for candidate in result["trajectories"]:
                candidate["poses_wxyz"][0] = result["start_pose"].copy()
            return result

    execution = FakeExecution()
    service = InteractiveDiTService(WrongPairBase(), execution)
    with pytest.raises(RuntimeError, match="起终点"):
        service.sample_trajectories(_request())
    assert execution.started == []


def test_sampler_rejects_an_incomplete_eight_candidate_batch():
    class IncompleteBase(FakeBaseService):
        def sample_trajectories(self, request):
            result = super().sample_trajectories(request)
            result["trajectories"].pop()
            return result

    service = InteractiveDiTService(IncompleteBase(), FakeExecution())
    with pytest.raises(RuntimeError, match="8"):
        service.sample_trajectories(_request())


def test_editing_endpoints_invalidates_the_saved_execution_plan():
    service = InteractiveDiTService(FakeBaseService(), FakeExecution())
    request = _request()
    result = service.sample_trajectories(request)
    changed = copy.deepcopy(request)
    changed["goal_pose"][1] = 0.25
    changed["invalidate_plan_id"] = result["plan_id"]
    service.validate(changed)
    with pytest.raises(ValueError):
        service.start_execution({"plan_id": result["plan_id"]})


def test_all_invalid_batch_cannot_start_execution():
    base = FakeBaseService(FakeChecker(lambda xyz: np.full(len(xyz), -0.1)))
    execution = FakeExecution()
    service = InteractiveDiTService(base, execution)
    result = service.sample_trajectories(_request())
    assert result["final_candidate_id"] is None
    with pytest.raises(ValueError):
        service.start_execution({"plan_id": result["plan_id"]})
    assert execution.started == []


def test_wrong_plan_id_is_rejected_without_launching_mujoco():
    execution = FakeExecution()
    service = InteractiveDiTService(FakeBaseService(), execution)
    service.sample_trajectories(_request())
    with pytest.raises(ValueError):
        service.start_execution({"plan_id": "old_plan"})
    assert execution.started == []


def test_execution_uses_immutable_server_side_scene_and_poses():
    execution = FakeExecution()
    service = InteractiveDiTService(FakeBaseService(), execution)
    request = _request()
    result = service.sample_trajectories(request)
    original_scene = copy.deepcopy(request["scene"])
    original_poses = copy.deepcopy(result["trajectories"][0]["poses_wxyz"])
    request["scene"]["obstacles"].append({"id": "changed_after_planning"})
    result["scene"]["obstacles"].append({"id": "changed_response"})
    result["trajectories"][0]["poses_wxyz"][1][0] = 999.0
    service.start_execution({"plan_id": result["plan_id"]})
    args, kwargs = execution.started[0]
    assert args[0] == original_scene
    assert args[1] == original_poses
    assert len(kwargs["candidates"]) == 8
    assert kwargs["candidates"][0] == original_poses


def test_manual_selection_and_execution_keep_the_same_final_candidate():
    execution = FakeExecution()
    service = InteractiveDiTService(FakeBaseService(), execution)
    result = service.sample_trajectories(_request())
    selected = service.select_plan({"plan_id": result["plan_id"], "candidate_id": 5})
    assert selected["final_candidate_id"] == 5
    status = service.start_execution({"plan_id": result["plan_id"]})
    assert status["plan_id"] == result["plan_id"]
    args, _ = execution.started[0]
    assert args[1] == result["trajectories"][5]["poses_wxyz"]


def test_scene_generation_invalidates_saved_plan():
    service = InteractiveDiTService(FakeBaseService(), FakeExecution())
    result = service.sample_trajectories(_request())
    service.generate_scene_payload({"seed": 123})
    with pytest.raises(ValueError):
        service.start_execution({"plan_id": result["plan_id"]})


def test_delayed_validation_cannot_invalidate_a_newer_plan():
    execution = FakeExecution()
    service = InteractiveDiTService(FakeBaseService(), execution)
    old_request = _request()
    old_plan = service.sample_trajectories(old_request)
    new_request = copy.deepcopy(old_request)
    new_request["scene"]["environment_id"] = "fixture_seed_43"
    new_plan = service.sample_trajectories(new_request)
    old_request["invalidate_plan_id"] = old_plan["plan_id"]
    service.validate(old_request)
    service.start_execution({"plan_id": new_plan["plan_id"]})
    args, _ = execution.started[0]
    assert args[0]["environment_id"] == "fixture_seed_43"


def test_validation_without_invalidation_token_is_read_only():
    execution = FakeExecution()
    service = InteractiveDiTService(FakeBaseService(), execution)
    result = service.sample_trajectories(_request())
    changed = _request()
    changed["goal_pose"][1] = 0.25
    service.validate(changed)
    service.start_execution({"plan_id": result["plan_id"]})
    assert len(execution.started) == 1


@pytest.mark.parametrize("operation", ["start", "preview"])
def test_rejected_native_operation_does_not_relabel_previous_execution(operation):
    execution = FakeExecution()
    service = InteractiveDiTService(FakeBaseService(), execution)
    previous = service.sample_trajectories(_request())
    service.start_execution({"plan_id": previous["plan_id"]})
    current_request = _request()
    current_request["scene"]["environment_id"] = "fixture_new_scene"
    current = service.sample_trajectories(current_request)

    def reject_native_operation(*args, **kwargs):
        raise RuntimeError("native viewer unavailable")

    setattr(execution, operation, reject_native_operation)
    with pytest.raises(RuntimeError, match="native viewer"):
        if operation == "start":
            service.start_execution({"plan_id": current["plan_id"]})
        else:
            service.preview_execution(current_request)
    assert service.execution_status()["plan_id"] == previous["plan_id"]


def test_planning_is_blocked_while_mppi_is_running():
    execution = FakeExecution()
    execution.status = lambda: {"state": "running"}
    service = InteractiveDiTService(FakeBaseService(), execution)
    with pytest.raises(RuntimeError):
        service.sample_trajectories(_request())
    with pytest.raises(RuntimeError):
        service.generate_scene_payload({"seed": 123})


def _handler_request(service, method: str, path: str, payload=None):
    """Exercise real HTTP route handlers without needing a listening socket."""
    handler_class = handler_type(service)
    handler = handler_class.__new__(handler_class)
    encoded = json.dumps(payload).encode("utf-8") if payload is not None else b""
    handler.path = path
    handler.headers = {"Content-Length": str(len(encoded))}
    handler.rfile = io.BytesIO(encoded)
    handler.wfile = io.BytesIO()
    statuses = []
    handler.send_response = statuses.append
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None
    getattr(handler, f"do_{method}")()
    return statuses[0], json.loads(handler.wfile.getvalue())


def test_http_routes_support_plan_select_execute_and_poll():
    base, execution = FakeBaseService(), FakeExecution()
    service = InteractiveDiTService(base, execution)
    status, catalog = _handler_request(service, "GET", "/api/catalog")
    assert status == 200
    assert catalog["execution"]["batch_size"] == 8
    status, planned = _handler_request(service, "POST", "/api/sample", _request())
    assert status == 200
    assert len(planned["trajectories"]) == 8
    status, selected = _handler_request(service, "POST", "/api/plan/select", {
        "plan_id": planned["plan_id"], "candidate_id": 3,
    })
    assert status == 200
    assert selected["final_candidate_id"] == 3
    status, execution_status = _handler_request(service, "POST", "/api/execution/start", {
        "plan_id": selected["plan_id"],
    })
    assert status == 200
    assert len(execution.started) == 1
    assert execution_status["plan_id"] == selected["plan_id"]
    status, polled = _handler_request(service, "GET", "/api/execution/status")
    assert status == 200
    assert polled["plan_id"] == selected["plan_id"]
    status, _ = _handler_request(service, "POST", "/api/execution/stop", {})
    assert status == 200
    assert execution.stopped


def test_http_rejects_stale_execution_request_with_reviewable_error():
    execution = FakeExecution()
    service = InteractiveDiTService(FakeBaseService(), execution)
    status, response = _handler_request(service, "POST", "/api/execution/start", {
        "plan_id": "stale_plan",
    })
    assert status == 400
    assert "plan_id" in response["error"]
    assert execution.started == []
