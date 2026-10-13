"""Balanced physical actuator control for the articulated HNUTER aircraft.

The sibling controller provides geometric pose feedback.  Allocation here
uses the loaded model's rotor positions, axes and centre of mass, rather than
the wing-root lever arm.  Nothing bypasses MuJoCo's motors or joint servos.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


class StabilizedFlightControl:
    """Geometric feedback, model-based allocation and bounded servo setpoints."""

    joint_names = ("rj2", "lj2", "rj1", "lj1")
    motor_names = tuple(f"motor_xy{i}" for i in range(1, 6))

    def __init__(self, controller: Any, mujoco: Any) -> None:
        try:
            from scipy.optimize import least_squares
        except ImportError as error:
            raise RuntimeError("Balanced MuJoCo control requires the simulation extra (scipy)") from error
        self._least_squares = least_squares
        self.controller = controller
        self.mujoco = mujoco
        self.model = controller.model
        self.data = controller.data
        self.dt = controller.dt
        self.body_id = controller._body_id
        obj = mujoco.mjtObj
        self.joints = np.array([mujoco.mj_name2id(self.model, obj.mjOBJ_JOINT, n)
                               for n in self.joint_names], dtype=int)
        self.servo_ids = np.array([mujoco.mj_name2id(self.model, obj.mjOBJ_ACTUATOR, f"tilt_{n}")
                                  for n in self.joint_names], dtype=int)
        self.motor_ids = np.array([mujoco.mj_name2id(self.model, obj.mjOBJ_ACTUATOR, n)
                                  for n in self.motor_names], dtype=int)
        self.site_ids = np.array([mujoco.mj_name2id(self.model, obj.mjOBJ_SITE, n)
                                 for n in self.motor_names], dtype=int)
        if np.any(np.r_[self.joints, self.servo_ids, self.motor_ids, self.site_ids] < 0):
            raise ValueError("HNUTER model is missing tilt joints or rotor actuators")
        self.qpos_ids = self.model.jnt_qposadr[self.joints].astype(int)
        self.dof_ids = self.model.jnt_dofadr[self.joints].astype(int)
        ranges = self.model.jnt_range[self.joints]
        self.angle_lower = ranges[:, 0] + 0.035
        self.angle_upper = ranges[:, 1] - 0.035
        self._joint_index = {int(j): i for i, j in enumerate(self.joints)}
        self._body_rotations = np.array([_rotation(q) for q in self.model.body_quat])
        self._site_rotations = np.array([_rotation(q) for q in self.model.site_quat])
        self._bodies = np.arange(self.body_id, self.model.nbody)
        self.mass = float(np.sum(self.model.body_mass[self._bodies]))
        self.controller.mass = self.mass
        self._decision = np.r_[self.mass * 9.81 / 2, self.mass * 9.81 / 2, 0.0, np.zeros(4)]
        self._angles = np.zeros(4)
        self._motors = np.zeros(5)
        self._last_allocation_time = -np.inf
        self._wrench_error = np.zeros(6)
        self._initialized = False
        self._velocity_buffer = np.empty(6)

    def _geometry(self, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotor poses and complete-aircraft COM in the drone body frame."""
        positions = np.zeros((self.model.nbody, 3))
        rotations = np.repeat(np.eye(3)[None], self.model.nbody, axis=0)
        for body in self._bodies[1:]:
            parent = self.model.body_parentid[body]
            before = rotations[parent] @ self._body_rotations[body]
            position = positions[parent] + rotations[parent] @ self.model.body_pos[body]
            rotation = before
            if self.model.body_jntnum[body]:
                joint = int(self.model.body_jntadr[body])
                if joint in self._joint_index:
                    angle = angles[self._joint_index[joint]]
                    axis = self.model.jnt_axis[joint]
                    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                                     [-axis[1], axis[0], 0]])
                    joint_rotation = (np.eye(3) + np.sin(angle)*skew
                                      + (1 - np.cos(angle))*(skew @ skew))
                    rotation = before @ joint_rotation
                    offset = self.model.jnt_pos[joint]
                    position += before @ offset - rotation @ offset
            positions[body] = position
            rotations[body] = rotation
        rotor_bodies = self.model.site_bodyid[self.site_ids]
        rotor_positions = positions[rotor_bodies] + np.einsum(
            "nij,nj->ni", rotations[rotor_bodies], self.model.site_pos[self.site_ids])
        rotor_rotations = rotations[rotor_bodies] @ self._site_rotations[self.site_ids]
        coms = positions[self._bodies] + np.einsum(
            "nij,nj->ni", rotations[self._bodies], self.model.body_ipos[self._bodies])
        com = np.sum(coms * self.model.body_mass[self._bodies, None], axis=0) / self.mass
        return rotor_positions, rotor_rotations, com

    def allocated_wrench(self, decision: np.ndarray) -> np.ndarray:
        """Physical rotor force and torque about the entire aircraft COM."""
        positions, rotations, com = self._geometry(decision[3:])
        thrust = np.r_[np.repeat(decision[:2] / 2, 2), decision[2]]
        gear = self.model.actuator_gear[self.motor_ids]
        force = np.einsum("nij,nj->ni", rotations, gear[:, :3]) * thrust[:, None]
        torque = np.cross(positions - com, force)
        torque += np.einsum("nij,nj->ni", rotations, gear[:, 3:]) * thrust[:, None]
        return np.r_[force.sum(axis=0), torque.sum(axis=0)]

    def _allocate(self, force: np.ndarray, torque: np.ndarray, *, initial: bool = False) -> None:
        desired = np.r_[force, torque]
        anchor = self._decision.copy()
        # Wrench units are N and Nm: prioritize accurate moments without
        # allowing an arbitrary null-space posture change on every update.
        scale = np.r_[np.ones(3), np.full(3, 8.0)]
        regularization = np.r_[np.full(2, 0.0003), 0.0005, np.full(4, 0.012)]

        def residual(decision: np.ndarray) -> np.ndarray:
            return np.r_[(self.allocated_wrench(decision) - desired) * scale,
                         (decision - anchor) * regularization]

        lower = np.r_[0.0, 0.0, -20.0, self.angle_lower]
        upper = np.r_[80.0, 80.0, 20.0, self.angle_upper]
        result = self._least_squares(residual, np.clip(anchor, lower + 1e-8, upper - 1e-8),
                                    bounds=(lower, upper), max_nfev=65 if initial else 8,
                                    ftol=1e-5, xtol=1e-5, gtol=1e-5)
        self._decision = result.x
        self._wrench_error = self.allocated_wrench(result.x) - desired
        self._last_allocation_time = float(self.data.time)

    def get_state(self) -> dict[str, np.ndarray]:
        """Read body-origin velocity, matching the position used by the planner."""
        state = self.controller.get_state()
        self.mujoco.mj_objectVelocity(self.model, self.data,
                                     self.mujoco.mjtObj.mjOBJ_XBODY, self.body_id,
                                     self._velocity_buffer, 0)
        state["velocity"] = self._velocity_buffer[3:].copy()
        state["angular_velocity"] = state["rotation_matrix"].T @ self._velocity_buffer[:3]
        return state

    def get_mppi_pose_state(self) -> np.ndarray:
        state = self.get_state()
        return np.r_[state["position"], state["velocity"], state["quaternion"],
                     state["angular_velocity"]]

    def initialize_trim(self, reference_state: np.ndarray) -> None:
        """Initialize articulation once, before the first physics step."""
        value = np.asarray(reference_state, dtype=float)
        if value.shape == (7,):
            position, quaternion = value[:3], value[3:]
        elif value.shape == (13,):
            position, quaternion = value[:3], value[6:10]
        else:
            raise ValueError("Trim reference must be pose7 or a 13-state reference")
        desired = {"pos": position, "vel": np.zeros(3), "acc": np.zeros(3),
                   "quaternion": quaternion, "angular_velocity": np.zeros(3),
                   "angular_acceleration": np.zeros(3)}
        self.controller.set_desired_state(desired)
        state = self.get_state()
        force, torque = self.controller.compute_control_wrench(state)
        self._allocate(force, torque, initial=True)
        self._angles = self._decision[3:].copy()
        self._motors = np.r_[np.repeat(self._decision[:2] / 2, 2), self._decision[2]]
        self.data.qpos[self.qpos_ids] = self._angles
        self.data.qvel[self.dof_ids] = 0
        self.mujoco.mj_forward(self.model, self.data)
        inertia = np.empty((self.model.nv, self.model.nv))
        try:
            self.mujoco.mj_fullM(self.model, self.data, inertia)
        except TypeError:
            # MuJoCo <= 3.3 used the dense-output / packed-matrix signature.
            self.mujoco.mj_fullM(self.model, inertia, self.data.qM)
        self.controller.inertia = inertia[3:6, 3:6].copy()
        # Near-critical body-rate damping based on the loaded mass matrix.
        self.controller.angular_rate_gain = 1.7 * np.sqrt(
            self.controller.attitude_gain * np.diag(self.controller.inertia))
        self._write_controls()
        self.mujoco.mj_forward(self.model, self.data)
        self._initialized = True

    def _write_controls(self) -> None:
        self.data.ctrl[self.motor_ids] = np.clip(
            self._motors, self.model.actuator_ctrlrange[self.motor_ids, 0],
            self.model.actuator_ctrlrange[self.motor_ids, 1])
        # Feed forward gravity and actual rotor loading at each tilt hinge.
        # The position actuator still supplies finite stiffness and damping;
        # articulation remains fully dynamic during all simulation steps.
        joint_load = self.data.qfrc_bias[self.dof_ids].copy()
        moments = self.data.actuator_moment
        for actuator, thrust in zip(self.motor_ids, self._motors):
            if moments.ndim == 2:
                joint_load -= moments[actuator, self.dof_ids] * thrust
            else:
                start = self.data.moment_rowadr[actuator]
                end = start + self.data.moment_rownnz[actuator]
                cols = self.data.moment_colind[start:end]
                vals = moments[start:end]
                for i, dof in enumerate(self.dof_ids):
                    joint_load[i] -= np.sum(vals[cols == dof]) * thrust
        stiffness = self.model.actuator_gainprm[self.servo_ids, 0]
        targets = self._angles + np.clip(joint_load / stiffness, -0.12, 0.12)
        self.data.ctrl[self.servo_ids] = np.clip(targets, self.angle_lower, self.angle_upper)

    def update(self, desired: dict[str, np.ndarray]) -> None:
        if not self._initialized:
            raise RuntimeError("Initialize HNUTER hover trim before advancing physics")
        self.controller.set_desired_state(desired)
        force, torque = self.controller.compute_control_wrench(self.get_state())
        if self.data.time - self._last_allocation_time >= 0.0195:
            self._allocate(force, torque)
        angle_step = (self._decision[3:] - self._angles) * (1 - np.exp(-self.dt / 0.035))
        self._angles += np.clip(angle_step, -2.5*self.dt, 2.5*self.dt)
        motor_target = np.r_[np.repeat(self._decision[:2] / 2, 2), self._decision[2]]
        motor_step = (motor_target - self._motors) * (1 - np.exp(-self.dt / 0.012))
        self._motors += np.clip(motor_step, -350*self.dt, 350*self.dt)
        self._write_controls()

    def telemetry(self) -> dict[str, Any]:
        motor_forces = self.data.ctrl[self.motor_ids]
        main_force = float(np.sum(motor_forces[:4]))
        return {
            "tilt_joint_names": list(self.joint_names),
            "tilt_positions_rad": self.data.qpos[self.qpos_ids].tolist(),
            "tilt_rates_rps": self.data.qvel[self.dof_ids].tolist(),
            "tilt_setpoints_rad": self._angles.tolist(),
            "tilt_tracking_errors_rad": (self._angles - self.data.qpos[self.qpos_ids]).tolist(),
            "motor_forces_n": motor_forces.tolist(),
            "motor_left_right_imbalance_fraction": float(
                abs(np.sum(motor_forces[:2]) - np.sum(motor_forces[2:4])) / max(main_force, 1e-9)),
            "wrench_error": self._wrench_error.tolist(),
            "allocation": "model_rotor_geometry_and_com",
        }
