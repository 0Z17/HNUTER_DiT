"""Continuous SE(3) references retimed by TOPP-RA for the MPPI controller.

TOPP-RA solves for the scalar progress ``s(t)``. Its custom constraints use
the world-frame position derivatives and *body-frame angular kinematics*,
rather than treating quaternion coordinates or Euler angles as joint angles.
The scalar TOPP-RA solution is interpolated in time with a C2 spline and
independently checked before it can be used by the executor.

Smoothing changes geometric segments slightly. Call :meth:`audit_poses` and
collision-check that exact curve before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RetimingConfig:
    speed_mps: float = 1.0
    max_acceleration_mps2: float = 1.5
    max_angular_speed_rps: float = 0.7
    max_angular_acceleration_rps2: float = 1.5
    initial_hold_s: float = 0.5
    control_dt: float = 0.05


def _config(value: Any) -> RetimingConfig:
    names = RetimingConfig.__dataclass_fields__
    if value is None:
        config = RetimingConfig()
    elif isinstance(value, dict):
        config = RetimingConfig(**{key: value[key] for key in names if key in value})
    else:
        config = RetimingConfig(**{key: getattr(value, key) for key in names})
    for key in names:
        number = getattr(config, key)
        if isinstance(number, bool) or not np.isscalar(number) or not np.isfinite(number):
            raise ValueError(f"{key} 必须为有限数值")
        if number < 0 or (key != "initial_hold_s" and number == 0):
            raise ValueError(f"{key} 必须为正值（initial_hold_s 可为零）")
    return config


def _poses(value: Any) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64).copy()
    if source.ndim != 2 or source.shape[1] != 7 or len(source) < 2 or not np.isfinite(source).all():
        raise ValueError("最终轨迹必须为有限的 [N, 7] pose7，N >= 2")
    norm = np.linalg.norm(source[:, 3:7], axis=1)
    if np.any(norm < 1e-9):
        raise ValueError("最终轨迹包含零四元数")
    source[:, 3:7] /= norm[:, None]
    for index in range(1, len(source)):
        if source[index, 3:7] @ source[index - 1, 3:7] < 0:
            source[index, 3:7] *= -1
    return source


class _SE3Curve:
    """C2 position/rotation interpolation inside a small polyline tube."""

    position_tolerance_m = 0.01
    rotation_tolerance_rad = 0.015

    def __init__(self, source: np.ndarray, config: RetimingConfig) -> None:
        from scipy.interpolate import CubicSpline
        from scipy.spatial.transform import Rotation, RotationSpline, Slerp

        rotations = Rotation.from_quat(source[:, [4, 5, 6, 3]])
        distance = np.linalg.norm(np.diff(source[:, :3], axis=0), axis=1)
        angles = (rotations[:-1].inv() * rotations[1:]).magnitude()
        keep = np.r_[True, (distance > 1e-9) | (angles > 1e-9)]
        self.poses = source[keep]
        rotations = rotations[keep]
        if len(self.poses) < 2:
            raise ValueError("最终轨迹没有位移或姿态变化")
        distance = np.linalg.norm(np.diff(self.poses[:, :3], axis=0), axis=1)
        angles = (rotations[:-1].inv() * rotations[1:]).magnitude()
        increments = np.maximum(distance / config.speed_mps, angles / config.max_angular_speed_rps)
        self.source_parameters = np.r_[0.0, np.cumsum(increments)]
        self.source_parameters /= self.source_parameters[-1]
        self._original_rotations = Slerp(self.source_parameters, rotations)
        # A dense learned trajectory carries small waypoint noise. Keeping
        # every point as an exact spline knot reproduces that noise as large
        # curvature/acceleration changes. Simplify jointly in position and
        # orientation first, preserving both endpoints and important bends.
        retained = {0, len(self.poses) - 1}
        pending = [(0, len(self.poses) - 1)]
        while pending:
            left, right = pending.pop()
            if right <= left + 1:
                continue
            inner = self.source_parameters[left + 1:right]
            fraction = (inner - self.source_parameters[left]) / (self.source_parameters[right] - self.source_parameters[left])
            linear = self.poses[left, :3] + fraction[:, None] * (self.poses[right, :3] - self.poses[left, :3])
            interpolator = Slerp(self.source_parameters[[left, right]], rotations[[left, right]])
            angle_error = (interpolator(inner).inv() * rotations[left + 1:right]).magnitude()
            score = np.maximum(
                np.linalg.norm(linear - self.poses[left + 1:right, :3], axis=1) / self.position_tolerance_m,
                angle_error / self.rotation_tolerance_rad,
            )
            if score.max() > .5:
                middle = left + 1 + int(score.argmax())
                retained.add(middle)
                pending.extend(((left, middle), (middle, right)))
        parameters = self.source_parameters[sorted(retained)]
        original_probes = (self.source_parameters[:-1, None]
                           + np.diff(self.source_parameters)[:, None] * np.linspace(0.0, 1.0, 9)[None, :]).ravel()

        for _ in range(12):
            positions = self._original_positions(parameters)
            rotation_values = self._original_rotations(parameters)
            position_spline = CubicSpline(parameters, positions, bc_type="natural")
            rotation_spline = RotationSpline(parameters, rotation_values)
            # Include multiple points per polynomial interval: corner
            # overshoot is not necessarily greatest at the midpoint.
            flat = np.unique(np.r_[original_probes, (parameters[:-1, None] + np.diff(parameters)[:, None]
                             * np.linspace(0.0, 1.0, 33)[None, :]).ravel()])
            linear_error = np.linalg.norm(position_spline(flat) - self._original_positions(flat), axis=1)
            angular_error = (self._original_rotations(flat).inv() * rotation_spline(flat)).magnitude()
            score = np.maximum(linear_error / self.position_tolerance_m, angular_error / self.rotation_tolerance_rad)
            if score.max() <= .95:
                self.parameters = parameters
                self.position = position_spline
                self.rotation = rotation_spline
                self.position_deviation_m = float(linear_error.max())
                self.rotation_deviation_rad = float(angular_error.max())
                break
            intervals = np.clip(np.searchsorted(parameters, flat, side="right") - 1, 0, len(parameters) - 2)
            extra = []
            for interval in np.unique(intervals[score > .95]):
                indices = np.flatnonzero(intervals == interval)
                extra.append(flat[indices[np.argmax(score[indices])]])
            parameters = np.unique(np.r_[parameters, extra])
        else:
            raise RuntimeError("无法在 1 cm 几何容差内平滑轨迹，请重新规划")

    def _original_positions(self, parameters: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.interp(parameters, self.source_parameters, self.poses[:, axis]) for axis in range(3)
        ])

    def geometry(self, parameters: Any) -> np.ndarray:
        parameters = np.atleast_1d(np.asarray(parameters, dtype=np.float64))
        parameters = np.clip(parameters, 0.0, 1.0)
        quaternion = self.rotation(parameters).as_quat()[:, [3, 0, 1, 2]]
        baseline = self._original_rotations(parameters).as_quat()[:, [3, 0, 1, 2]]
        quaternion *= np.where(np.sum(quaternion * baseline, axis=1) < 0, -1.0, 1.0)[:, None]
        return np.column_stack((self.position(parameters), quaternion))

    def derivatives(self, parameters: Any) -> tuple[np.ndarray, np.ndarray]:
        parameters = np.atleast_1d(np.asarray(parameters, dtype=np.float64))
        # RotationSpline(., 1/2) gives angular velocity and its derivative in
        # the rotating body frame. These are not rotvec first/second derivatives.
        first = np.column_stack((self.position(parameters, 1), self.rotation(parameters, 1)))
        second = np.column_stack((self.position(parameters, 2), self.rotation(parameters, 2)))
        return first, second


def _constraints(curve: _SE3Curve, config: RetimingConfig) -> list[Any]:
    from toppra.constraint import DiscretizationType, LinearConstraint
    from toppra.constraint.linear_constraint import canlinear_colloc_to_interpolate

    class SpeedConstraint(LinearConstraint):
        def __init__(self) -> None:
            super().__init__()
            self.dof = 1

        def compute_constraint_params(self, path: Any, gridpoints: np.ndarray) -> tuple[Any, ...]:
            first, _ = curve.derivatives(gridpoints)
            linear = np.linalg.norm(first[:, :3], axis=1)
            angular = np.linalg.norm(first[:, 3:], axis=1)
            max_progress = np.minimum(
                np.divide(config.speed_mps, linear, out=np.full_like(linear, np.inf), where=linear > 1e-12),
                np.divide(config.max_angular_speed_rps, angular, out=np.full_like(angular, np.inf), where=angular > 1e-12),
            )
            max_progress = .90 * np.minimum(max_progress, 1000.0)
            xbound = np.column_stack((np.zeros(len(gridpoints)), max_progress**2))
            return None, None, None, None, None, None, xbound

    class AccelerationConstraint(LinearConstraint):
        def __init__(self) -> None:
            super().__init__()
            self.dof = 1
            self.identical = True
            self.set_discretization_type(DiscretizationType.Interpolation)
            # All 26 normalized {-1,0,1} directions. The .80 radius leaves
            # margin for their polyhedral approximation of a norm ball.
            directions = np.array([item for item in product((-1, 0, 1), repeat=3) if any(item)], dtype=float)
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            self.F = np.zeros((2 * len(directions), 6))
            self.F[:len(directions), :3] = directions
            self.F[len(directions):, 3:] = directions
            self.g = .80 * np.r_[
                np.full(len(directions), config.max_acceleration_mps2),
                np.full(len(directions), config.max_angular_acceleration_rps2),
            ]

        def compute_constraint_params(self, path: Any, gridpoints: np.ndarray) -> tuple[Any, ...]:
            first, second = curve.derivatives(gridpoints)
            return canlinear_colloc_to_interpolate(
                first, second, np.zeros_like(first), self.F, self.g,
                None, None, gridpoints, identical=True,
            )

    return [SpeedConstraint(), AccelerationConstraint()]


class _SmoothedProgress:
    """Integrate a compact, smoothed TOPP-RA progress-speed profile.

    The kernel support extends the active interval a little to give exactly
    zero speed and acceleration at both ends. Its nonnegative convolution
    preserves forward progress; dense validation checks the continuous
    interpolation and all physical kinematic/jerk bounds afterwards.
    """

    def __init__(self, solved: Any, control_dt: float) -> None:
        from scipy.interpolate import PchipInterpolator
        from scipy.ndimage import gaussian_filter1d

        count = int(np.ceil(solved.duration / min(.003, control_dt / 16))) + 1
        original_times = np.linspace(0.0, solved.duration, count)
        dt = original_times[1]
        speed = np.maximum(np.asarray(solved(original_times, 1)).reshape(-1), 0.0)
        sigma_s = .06
        sigma_steps = sigma_s / dt
        padding = int(np.ceil(4 * sigma_steps)) + 3
        speed = np.pad(speed, (padding, padding))
        speed = gaussian_filter1d(speed, sigma_steps, mode="constant", truncate=4.0)
        speed[[0, -1]] = 0.0
        self.ss_waypoints = np.arange(len(speed)) * dt
        # Shape-preserving cubic speed interpolation keeps speed nonnegative.
        # Its C1 speed gives C2 progress, including zero endpoint acceleration
        # because the compact convolution has several flat zero samples.
        speed_spline = PchipInterpolator(self.ss_waypoints, speed)
        self.progress = speed_spline.antiderivative()
        self.progress.c /= float(self.progress(self.ss_waypoints[-1]))
        self.duration = float(self.ss_waypoints[-1])

    def __call__(self, times: Any, order: int = 0) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        return np.asarray(self.progress(times, order))[..., None]


class TimedPoseReference:
    """Rest-to-rest TOPP-RA timing with continuous motion through interior knots.

    ``sample`` returns the MPPI layout ``[p_world, v_world, q_wxyz,
    omega_body]`` and ``[acceleration_world, angular_acceleration_body]``.
    The returned curve is C2 in its active time interval. Holds before/after
    motion have exactly zero velocity and feedforward acceleration.
    """

    def __init__(self, trajectory: Any, config: Any = None, *, curve: Any = None) -> None:
        try:
            import toppra as ta
            import toppra.algorithm as algorithm
        except ImportError as error:
            raise RuntimeError("TOPP-RA 执行需要 scipy 和 toppra，请安装 simulation 依赖") from error
        self.config = _config(config)
        # A task-specific curve may preserve a geometric equality that the
        # free-flight XYZ/rotation smoother cannot preserve (surface contact).
        # Both paths use the same timing constraints and independent checks.
        self.curve = _SE3Curve(_poses(trajectory), self.config) if curve is None else curve
        self.poses = self.curve.poses.copy()
        self.gridpoints = np.unique(np.r_[
            np.linspace(0.0, 1.0, 801), self.curve.parameters,
            (self.curve.parameters[:-1] + self.curve.parameters[1:]) * .5,
        ])
        scalar_path = ta.SplineInterpolator([0.0, 1.0], [[0.0], [1.0]])
        solver = algorithm.TOPPRA(
            _constraints(self.curve, self.config), scalar_path,
            gridpoints=self.gridpoints, solver_wrapper="seidel",
            parametrizer="ParametrizeSpline",
        )
        self.toppra_solution = solver.compute_trajectory(0.0, 0.0)
        if self.toppra_solution is None:
            code = getattr(solver.problem_data, "return_code", "unknown")
            raise RuntimeError(f"TOPP-RA 无法求解连续轨迹 ({code})")
        self.scalar_trajectory = _SmoothedProgress(self.toppra_solution, self.config.control_dt)
        self._raw_duration = float(self.scalar_trajectory.duration)
        self.time_scale = 1.0
        self.motion_duration = self._raw_duration
        self.duration = self.config.initial_hold_s + self.motion_duration
        self._validate_and_scale()
        # Public arrivals are useful for inspection; their interior speeds
        # are generally nonzero, unlike the legacy per-segment stop profile.
        from scipy.optimize import brentq
        self.knot_times = np.array([
            brentq(lambda t: float(self.scalar_trajectory(t)[0]) - s, 0.0, self._raw_duration)
            if 0.0 < s < 1.0 else (0.0 if s == 0.0 else self._raw_duration)
            for s in self.curve.source_parameters
        ]) * self.time_scale

    def _sample_active(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_time = np.clip(times / self.time_scale, 0.0, self._raw_duration)
        parameters = np.asarray(self.scalar_trajectory(raw_time, 0)).reshape(-1)
        speed = np.asarray(self.scalar_trajectory(raw_time, 1)).reshape(-1) / self.time_scale
        acceleration = np.asarray(self.scalar_trajectory(raw_time, 2)).reshape(-1) / self.time_scale**2
        first, second = self.curve.derivatives(np.clip(parameters, 0.0, 1.0))
        velocity = first * speed[:, None]
        feedforward = first * acceleration[:, None] + second * speed[:, None]**2
        geometry = self.curve.geometry(parameters)
        state = np.column_stack((geometry[:, :3], velocity[:, :3], geometry[:, 3:], velocity[:, 3:]))
        return state, feedforward, parameters, speed

    def _sample_active_jerk(self, times: np.ndarray) -> np.ndarray:
        raw_time = np.clip(times / self.time_scale, 0.0, self._raw_duration)
        parameters = np.clip(np.asarray(self.scalar_trajectory(raw_time, 0)).reshape(-1), 0.0, 1.0)
        speed = np.asarray(self.scalar_trajectory(raw_time, 1)).reshape(-1) / self.time_scale
        acceleration = np.asarray(self.scalar_trajectory(raw_time, 2)).reshape(-1) / self.time_scale**2
        jerk = np.asarray(self.scalar_trajectory(raw_time, 3)).reshape(-1) / self.time_scale**3
        first, second = self.curve.derivatives(parameters)
        # Translation is piecewise cubic, so its third derivative is exact.
        # RotationSpline exposes body angular derivatives to order two; a
        # short central difference gives the next body derivative without
        # substituting rotation-vector derivatives for angular kinematics.
        low = np.maximum(parameters - 1e-6, 0.0)
        high = np.minimum(parameters + 1e-6, 1.0)
        if hasattr(self.curve, "third_derivatives"):
            third = self.curve.third_derivatives(parameters)
        else:
            angular_third = (self.curve.rotation(high, 2) - self.curve.rotation(low, 2)) / (high - low)[:, None]
            third = np.column_stack((self.curve.position(parameters, 3), angular_third))
        return (third * speed[:, None]**3
                + 3 * second * (speed * acceleration)[:, None]
                + first * jerk[:, None])

    def _validate_and_scale(self) -> None:
        # Validate every time polynomial, including immediately inside each
        # boundary, rather than only checking equally spaced wall-clock times.
        time_knots = np.asarray(self.scalar_trajectory.ss_waypoints)
        probes = np.unique(np.r_[
            np.linspace(0.0, self._raw_duration, max(8001, 16 * len(time_knots))),
            (time_knots[:-1, None] + np.diff(time_knots)[:, None] * np.linspace(0.0, 1.0, 17)).ravel(),
        ])
        state, acceleration, parameters, path_speed = self._sample_active(probes)
        if (not np.isfinite(state).all() or not np.isfinite(acceleration).all()
                or np.any(np.diff(parameters) < -1e-9) or path_speed.min() < -1e-9
                or parameters.min() < -1e-9 or parameters.max() > 1.0 + 1e-9):
            raise RuntimeError("TOPP-RA 连续时间插值无效或路径进度不单调")
        c = self.config
        linear_speed = float(np.linalg.norm(state[:, 3:6], axis=1).max())
        angular_speed = float(np.linalg.norm(state[:, 10:13], axis=1).max())
        linear_accel = float(np.linalg.norm(acceleration[:, :3], axis=1).max())
        angular_accel = float(np.linalg.norm(acceleration[:, 3:], axis=1).max())
        # C2 interpolation can increase peaks between the original TOPP-RA
        # grid points. Uniform dilation preserves the solved geometry and
        # continuity while independently respecting the requested norm limits.
        # Independent dense jerk estimates matter as well as C2 continuity:
        # a continuous acceleration can still change too rapidly for tilting
        # actuators. Time dilation scales jerk by its cube.
        jerk_times = np.linspace(0.0, self._raw_duration, max(8001, 4 * len(time_knots)))
        jerk = self._sample_active_jerk(jerk_times)
        dense_jerk = self._sample_active_jerk(probes)
        linear_jerk = np.linalg.norm(jerk[:, :3], axis=1)
        angular_jerk = np.linalg.norm(jerk[:, 3:], axis=1)
        linear_jerk_max = float(np.linalg.norm(dense_jerk[:, :3], axis=1).max())
        angular_jerk_max = float(np.linalg.norm(dense_jerk[:, 3:], axis=1).max())
        self.max_linear_jerk_mps3 = 15.0
        self.max_angular_jerk_rps3 = 8.0
        self.time_scale = float(max(
            1.0, linear_speed / (.98 * c.speed_mps),
            angular_speed / (.98 * c.max_angular_speed_rps),
            np.sqrt(linear_accel / (.98 * c.max_acceleration_mps2)),
            np.sqrt(angular_accel / (.98 * c.max_angular_acceleration_rps2)),
            np.cbrt(linear_jerk_max / (.98 * self.max_linear_jerk_mps3)),
            np.cbrt(angular_jerk_max / (.98 * self.max_angular_jerk_rps3)),
        ))
        self.motion_duration = self._raw_duration * self.time_scale
        self.duration = c.initial_hold_s + self.motion_duration
        self.validation = {
            "valid": True, "validation_point_count": len(probes),
            "max_linear_speed_mps": linear_speed / self.time_scale,
            "max_angular_speed_rps": angular_speed / self.time_scale,
            "max_linear_acceleration_mps2": linear_accel / self.time_scale**2,
            "max_angular_acceleration_rps2": angular_accel / self.time_scale**2,
            "estimated_max_linear_jerk_mps3": linear_jerk_max / self.time_scale**3,
            "estimated_rms_linear_jerk_mps3": float(np.sqrt(np.mean(linear_jerk**2))) / self.time_scale**3,
            "estimated_max_angular_jerk_rps3": angular_jerk_max / self.time_scale**3,
            "estimated_rms_angular_jerk_rps3": float(np.sqrt(np.mean(angular_jerk**2))) / self.time_scale**3,
            "progress_monotonic": True,
        }

    def sample(self, times: Any) -> tuple[np.ndarray, np.ndarray]:
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("参考时间必须为有限的一维数组")
        states, acceleration, _, _ = self._sample_active(times - self.config.initial_hold_s)
        inactive = (times <= self.config.initial_hold_s) | (times >= self.duration)
        states[inactive, 3:6] = 0.0
        states[inactive, 10:13] = 0.0
        acceleration[inactive] = 0.0
        return states, acceleration

    def audit_poses(self, max_spacing_m: float = 0.01, max_rotation_rad: float = 0.01) -> np.ndarray:
        """Dense poses from the exact smoothed geometry for collision auditing."""
        if not np.isfinite(max_spacing_m) or not np.isfinite(max_rotation_rad) or min(max_spacing_m, max_rotation_rad) <= 0:
            raise ValueError("碰撞审计采样间隔必须为正的有限数值")
        parameters = self.gridpoints.copy()
        for _ in range(12):
            first, _ = self.curve.derivatives(np.r_[parameters, (parameters[:-1] + parameters[1:]) * .5])
            count = len(parameters)
            linear = np.linalg.norm(first[:, :3], axis=1)
            angular = np.linalg.norm(first[:, 3:], axis=1)
            # Bound interval length by endpoint/midpoint derivative estimates;
            # retain the already dense TOPP-RA and geometric spline knots.
            linear = np.maximum.reduce((linear[:count - 1], linear[1:count], linear[count:]))
            angular = np.maximum.reduce((angular[:count - 1], angular[1:count], angular[count:]))
            refine = ((linear * np.diff(parameters) > .8 * max_spacing_m)
                      | (angular * np.diff(parameters) > .8 * max_rotation_rad))
            if not refine.any():
                return self.curve.geometry(parameters)
            parameters = np.unique(np.r_[parameters, (parameters[:-1][refine] + parameters[1:][refine]) * .5])
        raise RuntimeError("轨迹碰撞审计采样过密，请重新规划")

    def describe(self) -> dict[str, Any]:
        c = self.config
        return {
            "method": "toppra_continuous_se3_c2",
            "waypoint_count": len(self.poses), "gridpoint_count": len(self.gridpoints),
            "duration_s": self.duration, "motion_duration_s": self.motion_duration,
            "initial_hold_s": c.initial_hold_s, "speed_mps": c.speed_mps,
            "max_acceleration_mps2": c.max_acceleration_mps2,
            "max_angular_speed_rps": c.max_angular_speed_rps,
            "max_angular_acceleration_rps2": c.max_angular_acceleration_rps2,
            "stops_at_waypoints": False, "acceleration_continuous_in_motion": True,
            "max_linear_jerk_mps3": self.max_linear_jerk_mps3,
            "max_angular_jerk_rps3": self.max_angular_jerk_rps3,
            "progress_speed_smoothing_sigma_s": .06,
            "time_spline_safety_scale": self.time_scale,
            "position_deviation_m": self.curve.position_deviation_m,
            "orientation_deviation_rad": self.curve.rotation_deviation_rad,
            "requires_smoothed_geometry_audit": True,
            "kinematic_validation": self.validation.copy(),
        }
