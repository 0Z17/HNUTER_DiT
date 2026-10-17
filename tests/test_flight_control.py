"""Physical allocation and articulated hover regression checks."""

import numpy as np
import pytest

from hnuter_dit.mujoco_execution import DEFAULT_MUJOCO_ROOT, _external_runtime
from hnuter_dit.flight_control import StabilizedFlightControl


@pytest.fixture
def flight():
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    if not (DEFAULT_MUJOCO_ROOT / "hnuter206_4_5kg.xml").exists():
        pytest.skip("Optional HNUTER MuJoCo assets are unavailable")
    mujoco, controller_type, _ = _external_runtime(DEFAULT_MUJOCO_ROOT)
    controller = controller_type(DEFAULT_MUJOCO_ROOT / "hnuter206_4_5kg.xml")
    controller.set_freejoint_pose([0, 0, 2])
    return mujoco, controller, StabilizedFlightControl(controller, mujoco)


def test_allocator_uses_loaded_rotor_geometry(flight):
    mujoco, controller, flight_control = flight
    angles = np.array([.43, -.31, .18, -.27])
    controller.data.qpos[flight_control.qpos_ids] = angles
    mujoco.mj_forward(controller.model, controller.data)
    positions, rotations, com = flight_control._geometry(angles)
    root_position = controller.data.xpos[controller._body_id]
    np.testing.assert_allclose(positions, controller.data.site_xpos[flight_control.site_ids] - root_position,
                               atol=1e-12)
    np.testing.assert_allclose(rotations, controller.data.site_xmat[flight_control.site_ids].reshape(-1, 3, 3),
                               atol=1e-12)
    np.testing.assert_allclose(com, controller.data.subtree_com[controller._body_id] - root_position,
                               atol=1e-12)


def test_trim_hovers_with_real_motor_and_servo_dynamics(flight):
    mujoco, controller, flight_control = flight
    # Arbitrary body pitch requires large articulation; starting with neutral
    # wings previously caused a visible startup transient.
    pose = np.array([0, 0, 2, .923879532511, 0, -.382683432365, 0])
    controller.set_freejoint_pose(pose[:3], pose[3:])
    flight_control.initialize_trim(pose)
    desired = {"pos": pose[:3], "vel": np.zeros(3), "acc": np.zeros(3),
               "quaternion": pose[3:], "angular_velocity": np.zeros(3),
               "angular_acceleration": np.zeros(3)}
    peak_rate = 0.0
    for _ in range(500):
        flight_control.update(desired)
        mujoco.mj_step(controller.model, controller.data)
        peak_rate = max(peak_rate, np.max(np.abs(controller.data.qvel[flight_control.dof_ids])))
    state = flight_control.get_mppi_pose_state()
    assert np.linalg.norm(state[:3] - pose[:3]) < 1e-4
    assert peak_rate < .001
    assert controller.data.time == pytest.approx(.5)
    # Rotors receive force in Newtons, including the negative pitch trim force.
    assert abs(controller.data.ctrl[flight_control.motor_ids[-1]]) < 3
    assert np.all(controller.data.ctrl[flight_control.motor_ids[:4]] > 0)
    assert np.all(controller.data.qpos[flight_control.qpos_ids] > flight_control.angle_lower)
    assert np.all(controller.data.qpos[flight_control.qpos_ids] < flight_control.angle_upper)


def test_velocity_is_measured_at_the_planned_body_origin(flight):
    mujoco, controller, flight_control = flight
    controller.data.qvel[:3] = [.3, -.2, .1]
    controller.data.qvel[3:6] = [.7, -.4, .5]
    mujoco.mj_forward(controller.model, controller.data)
    state = flight_control.get_mppi_pose_state()
    np.testing.assert_allclose(state[3:6], controller.data.qvel[:3], atol=1e-12)
    assert np.linalg.norm(controller.data.cvel[controller._body_id, 3:] - state[3:6]) > .001
