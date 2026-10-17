"""Geometry, analytical timing, collision cost, and executor ownership checks."""

from pathlib import Path
from contextlib import nullcontext
import threading
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from hnuter_dit import mujoco_execution as execution


def _scene():
    return {
        "environment_id": "unit_scene", "obstacles": [
            {"id": "floor", "type": "box", "role": "floor", "collision": True,
             "size_xyz": [8, 8, .16], "pose": {"position": [0, 0, -.08],
             "quaternion_wxyz": [1, 0, 0, 0]}},
            {"id": "rotated", "type": "box", "role": "obstacle", "collision": True,
             "size_xyz": [.5, 2, 3], "pose": {"position": [1, -.3, 1.5],
             "quaternion_wxyz": [.9238795325, 0, 0, .3826834324]}},
        ],
    }


@pytest.fixture
def assets(tmp_path):
    (tmp_path / execution.MODEL_FILENAME).write_text(
        '<mujoco><compiler meshdir="meshes"/><worldbody>'
        '<geom name="old_ground" type="plane" size="50 50 1"/>'
        '<body name="drone"><freejoint/></body>'
        '</worldbody></mujoco>'
    )
    (tmp_path / "meshes").mkdir()
    (tmp_path / "hnuter_control.py").write_text("")
    (tmp_path / "mppi").mkdir()
    (tmp_path / "mppi" / "__init__.py").write_text("")
    return tmp_path


def test_retiming_preserves_segments_and_bounds_all_kinematics():
    source = np.array([
        [0, 0, 1, 1, 0, 0, 0], [1, 0, 1, .9238795325, 0, 0, .3826834324],
        [1, 2, 2, 0, 0, 0, -1],
    ])
    config = execution.ExecutionConfig()
    reference = execution.LegacyStopAtWaypointReference(source, config)
    times = np.linspace(-1, reference.duration + 1, 30001)
    states, feedforward = reference.sample(times)
    assert np.linalg.norm(states[:, 3:6], axis=1).max() <= config.speed_mps * 1.000001
    assert np.linalg.norm(feedforward[:, :3], axis=1).max() <= config.max_acceleration_mps2 * 1.000001
    assert np.linalg.norm(states[:, 10:], axis=1).max() <= config.max_angular_speed_rps * 1.000001
    assert np.linalg.norm(feedforward[:, 3:], axis=1).max() <= config.max_angular_acceleration_rps2 * 1.000001
    np.testing.assert_allclose(np.linalg.norm(states[:, 6:10], axis=1), 1, atol=1e-9)
    knots, knot_acceleration = reference.sample(reference.knot_times + config.initial_hold_s)
    np.testing.assert_allclose(knots[:, :3], source[:, :3], atol=1e-12)
    np.testing.assert_allclose(knots[:, 3:6], 0, atol=1e-12)
    np.testing.assert_allclose(knots[:, 10:], 0, atol=1e-12)
    np.testing.assert_allclose(knot_acceleration, 0, atol=1e-12)
    for index in range(2):
        midpoint = reference.sample([config.initial_hold_s + .5 * (
            reference.knot_times[index] + reference.knot_times[index + 1])])[0][0]
        np.testing.assert_allclose(midpoint[:3], .5 * (source[index, :3] + source[index + 1, :3]))
    assert not states[0, 3:6].any()
    assert not states[-1, 3:6].any()


def test_scene_xml_has_identical_obbs_and_no_hidden_ground(assets):
    start = [0, 0, 1, 1, 0, 0, 0]
    goal = [2, 1, 2, 1, 0, 0, 0]
    scene = _scene()
    xml = ET.fromstring(execution.build_scene_xml(scene, assets, start, goal))
    assert xml.find("compiler").get("meshdir") == str(assets / "meshes")
    geoms = xml.findall("worldbody/geom")
    assert len(geoms) == len(scene["obstacles"]) + 2
    assert all(geom.get("name") != "old_ground" for geom in geoms)
    for geom, obstacle in zip(geoms, scene["obstacles"]):
        np.testing.assert_allclose(np.fromstring(geom.get("pos"), sep=" "), obstacle["pose"]["position"])
        np.testing.assert_allclose(np.fromstring(geom.get("size"), sep=" "), np.array(obstacle["size_xyz"]) / 2)
        np.testing.assert_allclose(np.fromstring(geom.get("quat"), sep=" "), obstacle["pose"]["quaternion_wxyz"], atol=1e-9)
        assert geom.get("contype") == "1"
        assert geom.get("conaffinity") == "6"
    assert geoms[-1].get("contype") == "0"
    clone = _scene()
    clone["obstacles"][1]["size_xyz"][0] += .1
    assert execution.scene_geometry_hash(scene) != execution.scene_geometry_hash(clone)


def test_collision_cost_penalizes_overlapping_full_robot_proxy():
    class ZeroCost:
        def trajectory_cost(self, states, controls, reference):
            return np.zeros(len(states))

    cost = execution.ScenePoseTrackingCost(ZeroCost(), _scene())
    states = np.zeros((2, 5, 13))
    states[:, :, 6] = 1
    states[0, :, :3] = [1, -.3, 1.5]
    states[1, :, :3] = [-3, -3, 3]
    penalties = cost.trajectory_cost(states, np.zeros((2, 4, 6)), states[1])
    assert penalties[0] > 100
    assert penalties[1] == 0


def test_stop_must_be_acknowledged_before_restart(assets, monkeypatch):
    entered = threading.Event()
    permit_exit = threading.Event()

    class FakeController:
        def __init__(self, path):
            self.model = object()
            self.data = object()

        def set_freejoint_pose(self, position, quaternion):
            pass

    monkeypatch.setattr(execution, "_external_runtime", lambda root: (
        SimpleNamespace(mj_resetData=lambda *args: None), FakeController, object()))
    monkeypatch.setattr(execution.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(execution, "StabilizedFlightControl", lambda *args: SimpleNamespace(
        initialize_trim=lambda *args: None))

    class ControlledManager(execution.MujocoExecutionManager):
        def _execute(self, *args):
            self._publish(state="running")
            entered.set()
            permit_exit.wait(2)

    manager = ControlledManager(assets, viewer=False, realtime=False)
    path = [[-2, -2, 2, 1, 0, 0, 0], [-1, -2, 2, 1, 0, 0, 0]]
    try:
        manager.start(_scene(), path)
        assert entered.wait(2)
        assert manager.stop()["state"] == "stopping"
        with pytest.raises(RuntimeError):
            manager.start(_scene(), path)
        with pytest.raises(RuntimeError):
            manager.preview(_scene(), path[0], path[-1])
        permit_exit.set()
        deadline = time.monotonic() + 2
        while manager.status()["state"] != "stopped" and time.monotonic() < deadline:
            time.sleep(.01)
        assert manager.status()["state"] == "stopped"
        manager.preview(_scene(), path[0], path[-1])
        deadline = time.monotonic() + 2
        while manager.status()["state"] != "preview" and time.monotonic() < deadline:
            time.sleep(.01)
        assert manager.status()["state"] == "preview"
    finally:
        permit_exit.set()
        manager.close()


def test_invalid_execution_options_rejected():
    for options in ({"speed_mps": float("nan")}, {"num_samples": 0},
                    {"horizon": 32.0}, {"stop_on_contact": "false"}, {"typo": 1}):
        with pytest.raises(ValueError):
            execution.ExecutionConfig.from_value(options)


def test_native_scene_lines_use_real_mujoco_geometry_with_capacity_limit():
    mujoco = pytest.importorskip("mujoco")
    scene = mujoco.MjvScene(mujoco.MjModel.from_xml_string("<mujoco/>"), maxgeom=20)
    viewer = SimpleNamespace(user_scn=scene, lock=nullcontext)
    path = np.column_stack((np.linspace(0, 1, 128), np.zeros((128, 2)),
                            np.ones(128), np.zeros((128, 3))))
    visualizer = execution._ViewerLines(mujoco)
    visualizer.update(viewer, [path] * 8, path, path.tolist(), None)
    assert scene.ngeom == 20
    assert all(geom.type == mujoco.mjtGeom.mjGEOM_LINE for geom in scene.geoms[:scene.ngeom])


def test_passive_viewer_close_waits_for_render_thread_teardown():
    close_requested = threading.Event()
    allow_teardown = threading.Event()
    teardown_complete = threading.Event()

    class Handle:
        cam = "camera"

        def close(self):
            close_requested.set()

    def render(model, data):
        close_requested.wait(2)
        allow_teardown.wait(2)
        teardown_complete.set()

    def launch(model, data):
        threading.Thread(target=render, args=(model, data), daemon=True).start()
        return Handle()

    module = SimpleNamespace(launch_passive=launch, _launch_internal=render)
    viewer = execution._ManagedPassiveViewer.launch(module, object(), object())
    assert viewer.cam == "camera"
    closer = threading.Thread(target=viewer.close)
    closer.start()
    try:
        assert close_requested.wait(1)
        assert closer.is_alive()
        assert not teardown_complete.is_set()
    finally:
        allow_teardown.set()
        closer.join(2)
        viewer.close()
    assert teardown_complete.is_set()
    assert not closer.is_alive()
