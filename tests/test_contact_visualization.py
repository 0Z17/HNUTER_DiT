"""Contact replay geometry and server-owned execution-plan safety."""
from __future__ import annotations

import copy
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hnuter_dit import contact_visualization as visualization
from hnuter_dit.contact_adapter import sha256_file
from hnuter_dit.contact_metrics import _rotations, _shape_metrics
from hnuter_dit.contact_projection import ContactProjectionSpec, lift_contact_intrinsic, project_contact_pose9
from hnuter_dit.contact_smooth_projection import project_smooth_contact_pose9
from hnuter_dit.train_contact_adapter import json_hash


def _condition(kind="plane"):
    result = {
        "schema_version": "surface_task_v001", "task_type": "surface",
        "surface_geometry": {
            "type": kind, "origin": [1.2, -.4, 1.5],
            "quaternion_wxyz": [math.cos(.1), 0., math.sin(.1), 0.],
            "radius_m": 1.5, "thickness_m": .2, "supports": [],
        },
        "surface_chart": {"bounds": [[-2., 2.], [-2., 2.], [-math.pi, math.pi]],
                          "forbidden_regions": []},
        "tool_extrinsics": {
            "position_body": [.15, -.07, -.8],
            "quaternion_wxyz": [0., math.cos(.37/2), -math.sin(.37/2), 0.],
        },
        "contact_constraints": {
            "position_tolerance_m": 1e-7, "normal_tolerance_rad": 1e-6,
            "collision_margin_m": .01, "validation_step_m": .02, "validation_step_rad": .02,
            "attitude_limits_deg": {"roll": [-40., 40.], "pitch": [-70., 70.]},
        },
        "start_goal": {"start": [-1., -.3, .2], "goal": [1., .4, .25]},
        "environment": {"bounds": {"min": [-10., -10., -10.], "max": [10., 10., 10.]},
                        "obstacles": [{"id": "distant", "type": "box", "role": "obstacle",
                                       "position": [8., 8., 8.], "size": [.2, .2, .2],
                                       "quaternion_wxyz": [1., 0., 0., 0.]}]},
    }
    if kind == "cylinder":
        result["surface_geometry"]["quaternion_wxyz"] = [2**-.5, 2**-.5, 0., 0.]
        result["surface_chart"]["bounds"][0] = [1.5*2.8, 1.5*3.5]
        result["start_goal"] = {"start": [1.5*2.9, -.8, .2], "goal": [1.5*3.4, .8, .25]}
    return result


def _path(condition):
    spec = ContactProjectionSpec.from_condition(condition)
    states = np.linspace(spec.start, spec.goal, 128)
    return spec, states, lift_contact_intrinsic(states, spec).numpy()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def _write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf8") as stream:
        for record in records:
            stream.write(json.dumps(record)+"\n")


def _replace_frozen_metrics(frozen, decoder, candidate_index, changes):
    """Create an internally bound test score shard without changing predicted poses."""
    directory = frozen.smooth_directory if decoder == "on_smooth" else frozen.directory
    score_path = directory/"scores/map_000000.jsonl.gz"
    with gzip.open(score_path, "rt", encoding="utf8") as stream:
        records = [json.loads(line) for line in stream]
    for record in records:
        if record["projection"] == decoder and record["candidate_index"] == candidate_index:
            record["metrics"].update(copy.deepcopy(changes))
    _write_records(score_path, records)
    meta_path = directory/"scores/map_000000.json"
    metadata = json.loads(meta_path.read_text())
    metadata["scores_sha256"] = sha256_file(score_path)
    _write_json(meta_path, metadata)


def _curvature_failed_plan(frozen):
    """The real analytic fixture has a sharp endpoint bend; cubic fitting removes it."""
    with np.load(frozen.raw_path, allow_pickle=False) as archive:
        raw = archive["world_pose9"][0]
    spec = ContactProjectionSpec.from_condition(frozen.record["conditions"]["original"])
    analytic = project_contact_pose9(raw, spec).world_pose9.numpy()
    shape = _shape_metrics(analytic, _rotations(analytic))
    assert shape["maximum_curvature_inv_m"] > 8.
    assert shape["cumulative_backtracking_m"] <= .25
    assert shape["minimum_local_chord_arc_ratio"] >= .65
    _replace_frozen_metrics(frozen, "on", 0, shape | {
        "candidate_success": False, "failures": ["shape_quality"],
        "dense_checked": False, "dense_validation_completed": False,
        "physical_collision_checked": True, "physical_collision_free": True,
    })
    # Force selection to demonstrate that an available analytic curve is retained.
    for decoder, index in (("on", 1), ("on_smooth", 0), ("on_smooth", 1)):
        _replace_frozen_metrics(frozen, decoder, index, {
            "candidate_success": False, "failures": ["physical_collision"],
            "physical_collision_checked": True, "physical_collision_free": False,
        })
    return frozen.service.replay(frozen.request)


class _ExecutorSpy:
    def __init__(self):
        self.state = "idle"
        self.starts = []

    def status(self):
        return {"state": self.state, "executed_poses": []}

    def describe(self):
        return {"available": True}

    def start(self, scene, path, **kwargs):
        self.starts.append((copy.deepcopy(scene), np.asarray(path).copy(), kwargs))
        self.state = "running"


@pytest.fixture
def frozen(tmp_path):
    """Two genuine decoder outputs with saved, intentionally different gates."""
    benchmark, experiment = tmp_path/"benchmark", tmp_path/"experiment"
    benchmark.mkdir()
    conditions = {"original": _condition(), "obstacle_shift": _condition()}
    conditions["obstacle_shift"]["environment"]["obstacles"][0]["position"][0] += .3
    case, map_id, shard_relative = "benchmark/map_000000/condition_000", "benchmark/map_000000", "maps/map_000000/inputs.jsonl.gz"
    record = {
        "schema_version": "contact_benchmark_input_v001", "condition_id": case,
        "map_id": map_id, "source_map_index": 0, "family": "central_block",
        "kind": "plane", "difficulty": "easy", "angle_band": 0,
        "body_endpoints_pose7": [], "conditions": conditions,
        "input_sha256": {key: json_hash(value) for key, value in conditions.items()},
        "shift": {"moved_obstacle_id": "distant", "displacement_xyz": [.3, 0., 0.],
                  "shift_distance_m": .3, "shift_policy": "fixture",
                  "structural_fixture_members_moved_together": False},
    }
    shard = benchmark/shard_relative
    _write_records(shard, [record])
    index = {"condition_id": case, "map_id": map_id, "family": "central_block",
             "kind": "plane", "difficulty": "easy"}
    (benchmark/"index.jsonl").write_text(json.dumps(index)+"\n")
    np.savez_compressed(benchmark/"runtime_inputs.npz",
        conditions=np.zeros((2, 18)), condition_id=np.array([case, case]),
        scenario=np.array(["original", "obstacle_shift"]), map_id=np.array([map_id, map_id]),
        family=np.array(["central_block"]*2), kind=np.array(["plane"]*2),
        difficulty=np.array(["easy"]*2), input_shard=np.array([shard_relative]*2),
        input_row=np.zeros(2, dtype=int),
        input_sha256=np.array([record["input_sha256"][key] for key in conditions]))
    manifest = {"status": "frozen", "condition_count": 1,
                "input_files_sha256": {name: sha256_file(benchmark/name)
                    for name in ("runtime_inputs.npz", "index.jsonl", shard_relative)}}
    _write_json(benchmark/"manifest.json", manifest)
    directory = experiment/"evaluation/base"
    directory.mkdir(parents=True)
    code = Path(visualization.__file__).parent
    contract = {"model_id": "base", "base_sha256": "base-fixture", "adapter_sha256": None,
                "benchmark_manifest_sha256": sha256_file(benchmark/"manifest.json"), "ddim_steps": 25,
                "evaluation_sources": {name: sha256_file(code/name) for name in
                    ("evaluate_contact_experiment.py", "contact_metrics.py", "contact_projection.py", "contact_benchmark.py")}}
    contract["contract_sha256"] = json_hash(contract)
    _write_json(directory/"contract.json", contract)
    rows, indices = np.repeat([0, 1], 2), np.tile([0, 1], 2)
    raw, projected, intrinsic = [], [], []
    for row in rows:
        spec, _, path = _path(conditions[list(conditions)[row]])
        path[1:-1, :3] += [.02, -.01, .015]
        decoded = project_smooth_contact_pose9(path, spec)
        raw.append(path)
        projected.append(decoded.world_pose9.numpy())
        intrinsic.append(decoded.intrinsic_states.numpy())
    raw_path = directory/"raw/map_000000.npz"
    raw_path.parent.mkdir()
    np.savez_compressed(raw_path, world_pose9=np.stack(raw), input_rows=rows, candidate_indices=indices)
    raw_meta = {"raw_sha256": sha256_file(raw_path), "contract_sha256": contract["contract_sha256"]}
    _write_json(raw_path.with_suffix(".json"), raw_meta)
    scores = []
    smooth_scores = []
    for row, candidate_index in zip(rows, indices):
        scenario = list(conditions)[row]
        identity = {"condition_id": case, "scenario": scenario,
                    "input_sha256": record["input_sha256"][scenario], "candidate_index": int(candidate_index)}
        metrics = {"candidate_success": bool(candidate_index), "dense_validation_completed": True,
                   "path_length_m": .1 if candidate_index == 0 else 3.,
                   "clearance_m": [None]*128, "failures": ["fixture_gate"] if candidate_index == 0 else []}
        for decoder in ("off", "on"):
            scores.append(identity | {"projection": decoder, "metrics": copy.deepcopy(metrics)})
        smooth_scores.append(identity | {"projection": "on_smooth", "metrics": metrics})
    score_path = directory/"scores/map_000000.jsonl.gz"
    _write_records(score_path, scores)
    _write_json(score_path.with_suffix("").with_suffix(".json"),
                raw_meta | {"scores_sha256": sha256_file(score_path)})
    smooth_directory = experiment/"smooth_projection_supplement/evaluation/base"
    smooth_contract = {"model_id": "base", "parent_contract_sha256": contract["contract_sha256"],
                       "benchmark_manifest_sha256": contract["benchmark_manifest_sha256"],
                       "output_schema": {"projection": "on_smooth"}}
    smooth_contract["contract_sha256"] = json_hash(smooth_contract)
    _write_json(smooth_directory/"contract.json", smooth_contract)
    smooth_path = smooth_directory/"projected/map_000000.npz"
    smooth_path.parent.mkdir()
    np.savez_compressed(smooth_path, world_pose9=np.stack(projected), intrinsic_states=np.stack(intrinsic),
                        input_rows=rows, candidate_indices=indices)
    smooth_score_path = smooth_directory/"scores/map_000000.jsonl.gz"
    _write_records(smooth_score_path, smooth_scores)
    _write_json(smooth_directory/"scores/map_000000.json", {
        "raw_sha256": raw_meta["raw_sha256"], "contract_sha256": smooth_contract["contract_sha256"],
        "projected_sha256": sha256_file(smooth_path), "scores_sha256": sha256_file(smooth_score_path)})
    executor = _ExecutorSpy()
    service = visualization.ContactVisualizationService(experiment=experiment, benchmark=benchmark,
            base_checkpoint=tmp_path/"base.pt", device="cpu", execution=executor)
    request = {"model_id": "base", "condition_id": case, "scenario": "original"}
    return SimpleNamespace(service=service, request=request, benchmark=benchmark, experiment=experiment,
                           shard=shard, shard_relative=shard_relative, record=record, raw_path=raw_path,
                           directory=directory, smooth_directory=smooth_directory, executor=executor)


@pytest.mark.parametrize("kind", ["plane", "cylinder"])
def test_serialized_tool_points_and_normal_diagnostics_use_both_extrinsics(kind):
    spec, states, path = _path(_condition(kind))
    variant = visualization.serialize_variant(path, spec, {"clearance_m": [None]*128}, states)
    poses = np.array(variant["poses_wxyz"])
    rotation = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
    expected_tip = poses[:, :3]+rotation@spec.tool_offset
    np.testing.assert_allclose(variant["tool_points"], expected_tip, atol=2e-14)
    assert np.max(variant["diagnostics"]["contact_error_m"]) < 2e-14
    assert np.max(variant["diagnostics"]["normal_error_deg"]) < 1e-11
    assert np.linalg.norm(expected_tip-path[:, :3], axis=1).min() > .8
    json.dumps(variant, allow_nan=False)


def test_cylinder_display_follows_unwrapped_intrinsic_surface_across_atan2_seam():
    spec, states, path = _path(_condition("cylinder"))
    poses, tips = visualization.display_trajectory(path, spec, states)
    parameter = np.unique(np.r_[np.linspace(0., 127., 513), np.arange(128)])
    assert len(poses) > 513
    local = (tips-spec.origin)@spec.frame
    np.testing.assert_allclose(np.linalg.norm(local[:, :2], axis=1), spec.radius, atol=2e-14)
    angle = np.unwrap(np.arctan2(local[:, 1], local[:, 0]))
    np.testing.assert_allclose(angle, np.interp(parameter, np.arange(128), states[:, 0]/spec.radius), atol=2e-14)
    assert (np.diff(angle) > 0).all()
    chord = .5*(tips[0]+tips[-1])
    assert np.linalg.norm(((chord-spec.origin)@spec.frame)[:2]) < spec.radius-.04
    # Every frozen node is retained, including endpoint clamping.
    np.testing.assert_allclose(poses[np.searchsorted(parameter, np.arange(128)), :3], path[:, :3], atol=2e-14)


def test_raw_display_does_not_project_off_surface_predictions():
    spec, _, path = _path(_condition("cylinder"))
    path[1:-1, :3] += [.4, .2, -.3]
    poses, _ = visualization.display_trajectory(path, spec)
    parameter = np.unique(np.r_[np.linspace(0., 127., 513), np.arange(128)])
    np.testing.assert_allclose(poses[:, :3], np.column_stack([
        np.interp(parameter, np.arange(128), path[:, axis]) for axis in range(3)]))


def test_display_progress_is_node_index_even_when_geometric_speed_is_nonuniform():
    spec, states, _ = _path(_condition())
    fraction = np.linspace(0., 1., 128)**3
    states = spec.start+fraction[:, None]*(spec.goal-spec.start)
    path = lift_contact_intrinsic(states, spec).numpy()
    poses, _ = visualization.display_trajectory(path, spec, states)
    midpoint = poses[(len(poses)-1)//2]
    expected = lift_contact_intrinsic((states[63]+states[64])/2, spec).numpy()
    np.testing.assert_allclose(midpoint[:3], expected[:3], atol=2e-14)
    assert np.linalg.norm(midpoint[:3]-.5*(poses[0, :3]+poses[-1, :3])) > .5


def test_decoder_display_retains_endpoint_clamps_without_masking_contact_error():
    spec, states, path = _path(_condition())
    path[[0, -1], :3] += [.0, .0, .02]
    variant = visualization.serialize_variant(path, spec, {}, states)
    np.testing.assert_array_equal(np.array(variant["display_poses_wxyz"])[[0, -1], :3], path[[0, -1], :3])
    assert variant["diagnostics"]["contact_error_m"][0] > .019
    assert variant["diagnostics"]["contact_error_m"][-1] > .019


def test_curvature_endpoints_and_zero_translation_segments_are_null():
    spec, _, path = _path(_condition())
    path[30, :3] = path[29, :3]
    diagnostics = visualization.trajectory_diagnostics(path, spec)
    curvature = diagnostics["curvature_inv_m"]
    assert curvature[0] is curvature[-1] is None
    assert curvature[29] is curvature[30] is None
    assert curvature[31] is not None
    assert diagnostics["clearance_m"] is None
    assert diagnostics["curvature_policy"] == "interior_nodes_only_ignore_zero_translation_segments"
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("scenario", ["original", "obstacle_shift"])
def test_replay_preserves_frozen_failure_gates_and_uses_normalized_playback(frozen, scenario):
    result = frozen.service.replay(frozen.request | {"scenario": scenario})
    assert result["selected_candidate_index"] == 1  # Shorter failed candidate cannot win.
    assert result["playback_parameter"] == "normalized_node_index_not_physical_time"
    assert result["provenance"]["source"] == "frozen_saved_predictions"
    assert result["provenance"]["metrics_source"] == "frozen_saved_scores"
    assert result["condition"] == frozen.record["conditions"][scenario]
    first, second = result["candidates"]
    assert not first["variants"]["on_smooth"]["execution_eligible"]
    assert first["variants"]["on_smooth"]["metrics"]["failures"] == ["fixture_gate"]
    assert second["variants"]["on_smooth"]["execution_eligible"]
    assert second["variants"]["on"]["execution_eligible"]
    assert result["selected_decoder"] == "on_smooth"
    assert result["loading_status"] == "loaded"
    assert result["contact_force_control"] is False


@pytest.mark.parametrize("artifact", ["input", "raw"])
def test_changed_frozen_bytes_cannot_be_replayed_and_invalidate_previous_plan(frozen, artifact):
    frozen.service.replay(frozen.request)
    path = frozen.shard if artifact == "input" else frozen.raw_path
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(RuntimeError, match="哈希"):
        frozen.service.replay(frozen.request)
    assert frozen.service._plan is None


@pytest.mark.parametrize("field", ["condition_id", "map_id", "family", "kind", "difficulty"])
def test_verified_input_shard_identity_must_match_runtime_cache(frozen, field):
    record = copy.deepcopy(frozen.record)
    record[field] += "_other"
    _write_records(frozen.shard, [record])
    frozen.service.manifest["input_files_sha256"][frozen.shard_relative] = sha256_file(frozen.shard)
    with pytest.raises(RuntimeError, match="身份"):
        frozen.service.replay(frozen.request)


def test_condition_digest_cannot_be_substituted_even_with_verified_shard_bytes(frozen):
    record = copy.deepcopy(frozen.record)
    record["conditions"]["original"]["tool_extrinsics"]["position_body"][0] += .001
    _write_records(frozen.shard, [record])
    frozen.service.manifest["input_files_sha256"][frozen.shard_relative] = sha256_file(frozen.shard)
    with pytest.raises(RuntimeError, match="input_sha256"):
        frozen.service.replay(frozen.request)


@pytest.mark.parametrize("supplement", [False, True])
def test_contract_definition_tampering_is_rejected(frozen, supplement):
    path = (frozen.smooth_directory if supplement else frozen.directory)/"contract.json"
    contract = json.loads(path.read_text())
    contract["unrecorded_protocol_change"] = True
    _write_json(path, contract)
    with pytest.raises(RuntimeError, match="契约|平滑"):
        frozen.service.replay(frozen.request)


@pytest.mark.parametrize("candidate_index", [-1, 8, True, None, "1", 1.5])
def test_malformed_execution_candidate_is_rejected_before_executor_start(frozen, candidate_index):
    plan = frozen.service.replay(frozen.request)
    with pytest.raises(ValueError, match="candidate_index"):
        frozen.service.start_execution({"plan_id": plan["plan_id"], "candidate_index": candidate_index})
    assert not frozen.executor.starts


@pytest.mark.parametrize("decoder", ["off", "unknown"])
def test_execution_refuses_decoder_without_contact_reference(frozen, decoder):
    plan = frozen.service.replay(frozen.request)
    with pytest.raises(ValueError, match="on_smooth"):
        frozen.service.start_execution({"plan_id": plan["plan_id"], "decoder": decoder})
    assert not frozen.executor.starts


def test_failed_candidate_cannot_be_enabled_by_client_fields(frozen):
    plan = frozen.service.replay(frozen.request)
    with pytest.raises(ValueError, match="拦截"):
        frozen.service.start_execution({"plan_id": plan["plan_id"], "candidate_index": 0,
                                       "execution_eligible": True, "candidate_success": True})
    assert not frozen.executor.starts


def test_loaded_curvature_failure_keeps_frozen_gates_and_selects_analytic_decoder(frozen):
    plan = _curvature_failed_plan(frozen)
    assert plan["loading_status"] == "loaded"
    assert plan["saved_candidate_count"] == 2
    assert plan["selected_candidate_index"] == 0
    assert plan["selected_decoder"] == "on"
    value = plan["candidates"][0]["variants"]["on"]
    assert value["poses_wxyz"] and value["display_poses_wxyz"]
    assert value["metrics"]["failures"] == ["shape_quality"]
    assert value["metrics"]["candidate_success"] is False
    assert value["metrics"]["dense_validation_completed"] is False
    audit = value["execution_audit"]
    assert value["execution_eligible"] is True
    assert audit["mode"] == "curvature_diagnostic"
    assert audit["formal_source_success"] is False
    assert audit["curvature_warning"] is True
    assert audit["curvature_exceeded_node_count"] > 0
    assert audit["maximum_curvature_inv_m"] > 8.
    assert audit["dense_execution_recheck_required"] is True
    assert audit["blocking_failures"] == []
    assert not plan["candidates"][0]["variants"]["on_smooth"]["execution_eligible"]
    assert not plan["candidates"][0]["variants"]["off"]["execution_eligible"]
    json.dumps(plan, allow_nan=False)


@pytest.mark.parametrize("flag", ["omitted", False])
def test_curvature_failed_candidate_requires_explicit_diagnostic_flag(frozen, flag):
    plan = _curvature_failed_plan(frozen)
    request = {"plan_id": plan["plan_id"], "candidate_index": 0, "decoder": "on"}
    if flag != "omitted":
        request["allow_curvature_failure"] = flag
    with pytest.raises(ValueError, match="曲率超过"):
        frozen.service.start_execution(request)
    assert not frozen.executor.starts


@pytest.mark.parametrize("flag", [None, 1, "true", []])
def test_curvature_diagnostic_flag_requires_json_boolean(frozen, flag):
    plan = _curvature_failed_plan(frozen)
    with pytest.raises(ValueError, match="布尔"):
        frozen.service.start_execution({"plan_id": plan["plan_id"], "candidate_index": 0,
                                       "decoder": "on", "allow_curvature_failure": flag})
    assert not frozen.executor.starts


def test_curvature_diagnostic_executes_selected_analytic_curve_and_retains_failed_scores(frozen, monkeypatch):
    from hnuter_dit import contact_execution

    prepared = []
    reference = SimpleNamespace(describe=lambda: {"duration_s": 9.5, "rest_at_waypoints": True})

    def prepare_polyline(condition, states, options, *, allow_curvature_failure=False):
        prepared.append((copy.deepcopy(condition), np.asarray(states).copy(), options, allow_curvature_failure))
        return reference

    def reject_smoother(*args, **kwargs):
        raise AssertionError("selected on curve was replaced with a cubic")

    monkeypatch.setattr(contact_execution, "prepare_contact_polyline_reference", prepare_polyline)
    monkeypatch.setattr(contact_execution, "prepare_contact_reference", reject_smoother)
    plan = _curvature_failed_plan(frozen)
    on = np.asarray(plan["candidates"][0]["variants"]["on"]["poses_wxyz"])
    smooth = np.asarray(plan["candidates"][0]["variants"]["on_smooth"]["poses_wxyz"])
    assert not np.allclose(on[:, :3], smooth[:, :3], atol=1e-4)
    status = frozen.service.start_execution({"plan_id": plan["plan_id"], "candidate_index": 0,
        "decoder": "on", "allow_curvature_failure": True})
    assert status["state"] == "running"
    assert len(frozen.executor.starts) == 1
    scene, executed_reference, kwargs = frozen.executor.starts[0]
    np.testing.assert_array_equal(executed_reference, on)
    np.testing.assert_array_equal(kwargs["candidates"][0], on)
    assert kwargs["reference"] is reference
    assert prepared[0][0] == frozen.record["conditions"]["original"]
    assert prepared[0][1].shape == (128, 3)
    assert prepared[0][3] is True
    spec = ContactProjectionSpec.from_condition(prepared[0][0])
    np.testing.assert_allclose(lift_contact_intrinsic(prepared[0][1], spec).numpy()[:, :3], on[:, :3], atol=2e-14)
    assert scene["contact_condition"] == frozen.record["conditions"]["original"]
    trial = status["execution_trial"]
    assert trial["decoder"] == "on"
    assert trial["candidate_index"] == 0
    assert trial["mode"] == "curvature_diagnostic"
    assert trial["formal_source_success"] is False
    assert trial["source_metrics"]["candidate_success"] is False
    assert trial["source_metrics"]["failures"] == ["shape_quality"]
    assert trial["allow_curvature_failure"] is True
    assert trial["geometry"] == "exact_intrinsic_linear_segments_rest_at_nodes"
    assert trial["timing"]["duration_s"] == 9.5
    assert reference.execution_trial == trial


def test_curvature_override_never_bypasses_physical_collision(frozen):
    _curvature_failed_plan(frozen)
    _replace_frozen_metrics(frozen, "on", 0, {
        "failures": ["shape_quality", "physical_collision"], "physical_collision_free": False,
    })
    plan = frozen.service.replay(frozen.request)
    value = plan["candidates"][0]["variants"]["on"]
    assert value["execution_audit"]["curvature_warning"] is True
    assert value["execution_eligible"] is False
    assert "physical_collision" in value["execution_audit"]["blocking_failures"]
    with pytest.raises(ValueError, match="physical_collision"):
        frozen.service.start_execution({"plan_id": plan["plan_id"], "candidate_index": 0,
            "decoder": "on", "allow_curvature_failure": True, "execution_eligible": True})
    assert not frozen.executor.starts


@pytest.mark.parametrize("changes,blocker", [
    ({"cumulative_backtracking_m": .251}, "shape_backtracking"),
    ({"minimum_local_chord_arc_ratio": .64}, "shape_chord_arc_ratio"),
    ({"physical_collision_checked": False}, "physical_collision_unverified"),
    ({"evaluation_error": "collision backend unavailable"}, "invalid_input_or_backend_error"),
])
def test_curvature_override_does_not_hide_other_shape_or_missing_checks(frozen, changes, blocker):
    _curvature_failed_plan(frozen)
    _replace_frozen_metrics(frozen, "on", 0, changes)
    plan = frozen.service.replay(frozen.request)
    value = plan["candidates"][0]["variants"]["on"]
    assert value["execution_eligible"] is False
    assert blocker in value["execution_audit"]["blocking_failures"]
    with pytest.raises(ValueError, match=blocker):
        frozen.service.start_execution({"plan_id": plan["plan_id"], "candidate_index": 0,
                                       "decoder": "on", "allow_curvature_failure": True})
    assert not frozen.executor.starts


@pytest.mark.parametrize("options", [
    {"stop_on_contact": False}, {"maximum_tracking_error_m": 2.},
    {"num_samples": 32}, {"speed_mps": .7, "stop_on_contact": False},
    None, [],
])
def test_execution_options_cannot_override_contact_or_tracking_abort_policy(frozen, options):
    plan = frozen.service.replay(frozen.request)
    with pytest.raises(ValueError, match="只允许调整参考"):
        frozen.service.start_execution({"plan_id": plan["plan_id"], "candidate_index": 1,
                                       "options": options})
    assert not frozen.executor.starts


@pytest.mark.parametrize("speed", [0., -1., 2.01, float("nan"), True, ".7"])
def test_execution_rejects_invalid_reference_speed_before_start(frozen, speed):
    plan = frozen.service.replay(frozen.request)
    with pytest.raises(ValueError, match="speed_mps"):
        frozen.service.start_execution({"plan_id": plan["plan_id"], "candidate_index": 1,
                                       "options": {"speed_mps": speed}})
    assert not frozen.executor.starts


@pytest.mark.parametrize("decoder", ["on", "on_smooth"])
def test_reference_speed_override_reaches_selected_reference_and_executor_without_mutating_defaults(frozen, monkeypatch, decoder):
    from hnuter_dit import contact_execution

    prepared = []
    reference = SimpleNamespace(describe=lambda: {"duration_s": 7.})

    def prepare(condition, states_or_controls, options, *, allow_curvature_failure=False):
        prepared.append(options)
        return reference

    monkeypatch.setattr(contact_execution, "prepare_contact_polyline_reference", prepare)
    monkeypatch.setattr(contact_execution, "prepare_contact_reference", prepare)
    plan = frozen.service.replay(frozen.request)
    before = copy.deepcopy(frozen.service.execution_options)
    frozen.service.start_execution({"plan_id": plan["plan_id"], "candidate_index": 1,
                                   "decoder": decoder, "options": {"speed_mps": .7}})
    assert len(prepared) == 1
    config = frozen.executor.starts[0][2]["options"]
    assert prepared[0] is config
    assert config.speed_mps == .7
    assert config.max_acceleration_mps2 == before.max_acceleration_mps2
    assert config.stop_on_contact is before.stop_on_contact
    assert config.maximum_tracking_error_m == before.maximum_tracking_error_m
    assert frozen.service.execution_options == before
    assert config is not frozen.service.execution_options


def test_superseded_and_explicitly_invalidated_plans_cannot_execute(frozen):
    first = frozen.service.replay(frozen.request)
    second = frozen.service.replay(frozen.request | {"scenario": "obstacle_shift"})
    with pytest.raises(ValueError, match="失效"):
        frozen.service.start_execution({"plan_id": first["plan_id"]})
    frozen.service.invalidate_plan({"plan_id": second["plan_id"]})
    with pytest.raises(ValueError, match="失效"):
        frozen.service.start_execution({"plan_id": second["plan_id"]})
    assert not frozen.executor.starts


def test_execution_uses_server_snapshot_despite_returned_result_and_request_mutations(frozen, monkeypatch):
    from hnuter_dit import contact_execution
    prepared = []
    reference = SimpleNamespace(describe=lambda: {"duration_s": 5.})

    def prepare(condition, controls, options, *, allow_curvature_failure=False):
        prepared.append((copy.deepcopy(condition), np.asarray(controls).copy(), options))
        assert allow_curvature_failure is False
        return reference

    monkeypatch.setattr(contact_execution, "prepare_contact_reference", prepare)
    plan = frozen.service.replay(frozen.request)
    saved = np.array(plan["candidates"][1]["variants"]["on_smooth"]["poses_wxyz"])
    # Both the response object and next request are outside the execution trust boundary.
    plan["condition"]["surface_geometry"]["origin"] = [900., 900., 900.]
    plan["candidates"][1]["variants"]["on_smooth"]["poses_wxyz"][5][0] = 900.
    status = frozen.service.start_execution({"plan_id": plan["plan_id"], "candidate_index": 1,
        "condition": plan["condition"], "poses_wxyz": [[900.]*7]*128,
        "candidates": [[900.]*7]*128})
    assert status["state"] == "running"
    assert len(frozen.executor.starts) == 1
    scene, actual, kwargs = frozen.executor.starts[0]
    np.testing.assert_array_equal(actual, saved)
    np.testing.assert_array_equal(kwargs["candidates"][0], saved)
    assert kwargs["reference"] is reference
    assert scene["contact_condition"] == frozen.record["conditions"]["original"]
    assert prepared[0][0] == frozen.record["conditions"]["original"]
    assert prepared[0][1].shape == (16, 3)


def test_decoder_regeneration_mismatch_blocks_execution(frozen):
    plan = frozen.service.replay(frozen.request)
    frozen.service._plan["result"]["candidates"][1]["variants"]["on_smooth"]["poses_wxyz"][20][0] += .01
    with pytest.raises(RuntimeError, match="重建"):
        frozen.service.start_execution({"plan_id": plan["plan_id"], "candidate_index": 1})
    assert not frozen.executor.starts


@pytest.mark.parametrize("state", ["starting", "running", "stopping"])
def test_active_executor_blocks_replanning_and_invalidation(frozen, state):
    plan = frozen.service.replay(frozen.request)
    frozen.executor.state = state
    for operation, request in ((frozen.service.replay, frozen.request),
                               (frozen.service.sample, frozen.request),
                               (frozen.service.invalidate_plan, {"plan_id": plan["plan_id"]})):
        with pytest.raises(RuntimeError, match="MuJoCo"):
            operation(request)
    assert frozen.service._plan["result"]["plan_id"] == plan["plan_id"]


def test_failed_model_switch_can_reinitialize_the_previous_model(frozen, monkeypatch):
    frozen.service._runtime_model = "base"
    frozen.service._runtime = object()
    frozen.service.models["broken"] = frozen.experiment/"broken.pt"
    initialized = []

    class Runtime:
        def __init__(self, base_checkpoint, adapter_checkpoint, device):
            initialized.append(adapter_checkpoint)
            if adapter_checkpoint is not None:
                raise RuntimeError("adapter checkpoint rejected")

        def sample(self, *args, **kwargs):
            raise RuntimeError("reinitialized base reached sampling")

    monkeypatch.setattr(visualization, "ContactInferenceRuntime", Runtime)
    with pytest.raises(RuntimeError, match="checkpoint rejected"):
        frozen.service.sample(frozen.request | {"model_id": "broken", "count": 1})
    assert frozen.service._runtime is frozen.service._runtime_model is None
    with pytest.raises(RuntimeError, match="reinitialized base reached sampling"):
        frozen.service.sample(frozen.request | {"count": 1})
    assert initialized == [frozen.service.models["broken"], None]
    assert frozen.service._runtime_model == "base"
    assert isinstance(frozen.service._runtime, Runtime)


@pytest.mark.parametrize("kind", ["start", "preview"])
def test_new_execution_command_removes_previous_contact_telemetry(frozen, kind):
    from hnuter_dit.contact_execution import build_contact_scene
    from hnuter_dit.mujoco_execution import ExecutionConfig, MujocoExecutionManager

    manager = MujocoExecutionManager(viewer=False)
    # Keep the queue transition synchronous without loading external physics assets.
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._status.update(state="completed", executed_poses=[[0., 0., 1., 1., 0., 0., 0.]],
                           contact_metrics={"task_success": True, "max_contact_error_m": .0001},
                           flight_outcome={"success": True, "terminal_state": "completed"},
                           task_success=True, task_type="contact", contact_limitations="old")
    reference = SimpleNamespace(contact_curve=object(), describe=lambda: {"duration_s": 5.})
    command = {"kind": kind, "scene": build_contact_scene(frozen.record["conditions"]["original"]),
               "reference": reference, "reference_audit": {"checked": True},
               "config": ExecutionConfig(), "seed": 1}
    manager._queue(command)
    status = manager.status()
    assert status["state"] == "starting"
    assert status["executed_poses"] == []
    assert "contact_metrics" not in status
    assert "flight_outcome" not in status
    if kind == "start":
        assert status["task_success"] is None
    else:
        assert "task_success" not in status
        assert "task_type" not in status
    assert manager._commands.get_nowait()["kind"] == kind


@pytest.mark.parametrize("state,contacts,expected", [
    ("completed", 0, True), ("completed", 1, False), ("running", 0, False),
])
def test_task_result_requires_completed_flight_without_environment_contacts(frozen, monkeypatch, state, contacts, expected):
    pytest.importorskip("coal")
    condition = frozen.record["conditions"]["original"]
    spec, states, path = _path(condition)
    poses = visualization.serialize_variant(path, spec, {}, states)["poses_wxyz"]
    frozen.service._execution_condition = copy.deepcopy(condition)
    frozen.service._execution_plan_id = "executed_plan"
    monkeypatch.setattr(frozen.executor, "status", lambda: {
        "state": state, "success": True, "contact_count": contacts, "executed_poses": poses,
    })
    status = frozen.service.execution_status()
    assert status["flight_success"] is True
    assert status["contact_metrics"]["geometric_task_success"] is True
    assert status["contact_metrics"]["task_success"] is expected
    assert status["task_success"] is expected


def test_closing_completed_viewer_preserves_finished_flight_and_task_outcome(frozen, monkeypatch):
    pytest.importorskip("coal")
    condition = frozen.record["conditions"]["original"]
    spec, states, path = _path(condition)
    poses = visualization.serialize_variant(path, spec, {}, states)["poses_wxyz"]
    frozen.service._execution_condition = copy.deepcopy(condition)
    frozen.service._execution_plan_id = "finished_plan"
    snapshot = {
        "state": "completed", "success": True, "contact_count": 0, "executed_poses": poses,
        "flight_outcome": {"success": True, "terminal_state": "completed"},
    }
    monkeypatch.setattr(frozen.executor, "status", lambda: copy.deepcopy(snapshot))
    before = frozen.service.execution_status()
    assert before["flight_success"] is True
    assert before["task_success"] is True
    # stop closes the finished viewer; it does not erase the completed flight.
    snapshot.update(state="stopped", success=False)
    after = frozen.service.execution_status()
    assert after["state"] == "stopped"
    assert after["success"] is False
    assert after["plan_id"] == before["plan_id"]
    assert after["executed_poses"] == before["executed_poses"]
    assert after["flight_success"] is True
    assert after["contact_metrics"]["geometric_task_success"] is True
    assert after["contact_metrics"]["task_success"] is True
    assert after["task_success"] is True


def test_aborting_active_flight_without_finished_outcome_remains_unsuccessful(frozen, monkeypatch):
    pytest.importorskip("coal")
    condition = frozen.record["conditions"]["original"]
    spec, states, path = _path(condition)
    poses = visualization.serialize_variant(path, spec, {}, states)["poses_wxyz"]
    frozen.service._execution_condition = copy.deepcopy(condition)
    frozen.service._execution_plan_id = "aborted_plan"
    monkeypatch.setattr(frozen.executor, "status", lambda: {
        "state": "stopped", "success": False, "contact_count": 0, "executed_poses": poses,
    })
    status = frozen.service.execution_status()
    assert "flight_outcome" not in status
    assert status["flight_success"] is False
    assert status["contact_metrics"]["geometric_task_success"] is True
    assert status["contact_metrics"]["task_success"] is False
    assert status["task_success"] is False


def test_completed_old_execution_keeps_its_identity_after_loading_a_new_plan(frozen, monkeypatch):
    frozen.service._execution_plan_id = "previous_execution_plan"
    monkeypatch.setattr(frozen.executor, "status", lambda: {
        "state": "completed", "success": True, "executed_poses": [],
    })
    new_plan = frozen.service.replay(frozen.request)
    status = frozen.service.execution_status()
    assert status["plan_id"] == "previous_execution_plan"
    assert status["plan_id"] != new_plan["plan_id"]
