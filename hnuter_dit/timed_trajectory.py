"""Continuous references for geometric DiT paths (SI, world Z up, body FLU).

This interpolates geometry, then stretches time to respect sampled derivative
limits. It is not a collision certificate: recheck the *interpolated* trajectory,
including a requested initial-velocity transition, before execution.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline


def _phase(u):
    return u ** 3 * (10. + u * (-15. + 6. * u))


def _inverse_phase(s):
    lower, upper = np.zeros_like(s), np.ones_like(s)
    for _ in range(50):
        middle = (lower + upper) / 2
        below = _phase(middle) < s
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return (lower + upper) / 2


def _vector(value, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite three-vector")
    return result


class TimedTrajectory:
    """Interpolate pose7 [xyz,wxyz] and return controller setpoint dictionaries.

    Position and orientation share the same arc-length parameter and a quintic
    time law. RotationSpline's analytic derivatives are expressed in the moving
    body frame, matching the controller's angular-velocity convention. Default
    linear/angular rates and accelerations vanish at both endpoints.

    Optional world-frame start_velocity/start_acceleration use a finite-duration
    quintic correction with matching value, first and second derivative at each
    end. This can move the path sideways; collision checking must include it.
    No positions or derivatives are clipped to conceal a limit violation.

    Limits are checked on >=16385 time samples plus samples in every geometric
    spline interval and initial transition. They are numerical bounds with a
    stretching margin, not a formal continuous-time feasibility certificate.
    """

    def __init__(self, poses, *, max_speed=.5, max_acceleration=.6,
                 max_angular_speed=.6, max_angular_acceleration=1.,
                 start_velocity=None, start_acceleration=None):
        values = np.asarray(poses, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2:
            raise ValueError("poses must have shape [N>=2,7] (xyz,wxyz)")
        if not np.isfinite(values).all():
            raise ValueError("poses must be finite")
        values = values.copy()
        norms = np.linalg.norm(values[:, 3:], axis=1)
        if np.any(norms < 1e-8):
            raise ValueError("pose quaternion must be nonzero")
        values[:, 3:] /= norms[:, None]
        for i in range(1, len(values)):
            if values[i - 1, 3:] @ values[i, 3:] < 0:
                values[i, 3:] *= -1
        limits = np.asarray([max_speed, max_acceleration,
                             max_angular_speed, max_angular_acceleration], dtype=float)
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("motion limits must be finite and positive")
        self.limits = dict(zip(("speed", "acceleration", "angular_speed",
                               "angular_acceleration"), limits.tolist()))
        velocity = _vector(np.zeros(3) if start_velocity is None else start_velocity,
                           "start_velocity")
        acceleration = _vector(np.zeros(3) if start_acceleration is None else start_acceleration,
                               "start_acceleration")
        if np.linalg.norm(velocity) > max_speed * (1 + 1e-10):
            raise ValueError("start_velocity already exceeds max_speed")
        if np.linalg.norm(acceleration) > max_acceleration * (1 + 1e-10):
            raise ValueError("start_acceleration already exceeds max_acceleration")

        # Include rotation in the metric, so pure attitude motion and repeated
        # positions have well-defined, strictly increasing spline knots.
        rotations = Rotation.from_quat(values[:, [4, 5, 6, 3]])
        translation = np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1)
        angles = (rotations[:-1].inv() * rotations[1:]).magnitude()
        increments = translation + .5 * angles
        keep = np.r_[True, increments > 1e-8]
        values = values[keep]
        if len(values) == 1:
            values = np.repeat(values, 2, axis=0)
        self.poses = values
        rotations = Rotation.from_quat(values[:, [4, 5, 6, 3]])
        translation = np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1)
        angles = (rotations[:-1].inv() * rotations[1:]).magnitude()
        increments = np.maximum(translation + .5 * angles, 1e-8)
        knots = np.r_[0., np.cumsum(increments)]
        knots /= knots[-1]
        self._position = CubicSpline(knots, values[:, :3], bc_type="natural")
        self._rotation = RotationSpline(knots, rotations)
        self._bridge_duration = 0.
        self._bridge_coefficients = np.zeros((6, 3))
        if np.any(velocity) or np.any(acceleration):
            self._bridge_duration = max(.5, 8 * np.linalg.norm(velocity) / max_acceleration,
                                        2 * np.linalg.norm(acceleration) / max_acceleration)
            tau = self._bridge_duration
            v, a = velocity * tau, acceleration * tau ** 2
            self._bridge_coefficients = np.array([
                np.zeros(3), v, .5 * a, -6 * v - 1.5 * a,
                8 * v + 1.5 * a, -3 * v - .5 * a])
        self.duration = max(.5, 1.875 * float(translation.sum()) / max_speed,
                            1.875 * float(angles.sum()) / max_angular_speed,
                            1.25 * self._bridge_duration)
        # A regular time grid alone can miss short intervals in highly uneven
        # paths. Add a grid in every original geometric spline interval.
        local_s = np.concatenate([np.linspace(a, b, 17) for a, b in zip(knots[:-1], knots[1:])])
        self._check_u = np.unique(np.r_[np.linspace(0, 1, 16385), _inverse_phase(local_s)])
        for _ in range(40):
            times = self._check_u * self.duration
            if self._bridge_duration:
                times = np.unique(np.r_[times, np.linspace(0, self._bridge_duration, 1025)])
            data = self._evaluate(times)
            maxima = np.asarray([np.linalg.norm(data[key], axis=1).max() for key in
                                 ("vel", "acc", "angular_velocity", "angular_acceleration")])
            ratio = float(max(maxima[0] / limits[0], np.sqrt(maxima[1] / limits[1]),
                              maxima[2] / limits[2], np.sqrt(maxima[3] / limits[3])))
            if ratio <= 1 + 1e-9:
                self.observed_maxima = dict(zip(self.limits, maxima.tolist()))
                self.validation_samples = len(times)
                break
            self.duration *= max(1.05, ratio * 1.03)
            if self.duration > 10000:
                raise ValueError("path cannot satisfy limits within a useful duration")
        else:
            raise ValueError("initial motion cannot be blended within motion limits")

    def _evaluate(self, times):
        t = np.clip(np.asarray(times, dtype=float), 0., self.duration)
        u = t / self.duration
        s = _phase(u)
        ds = 30 * u ** 2 * (u - 1) ** 2 / self.duration
        dds = 60 * u * (2 * u ** 2 - 3 * u + 1) / self.duration ** 2
        first = self._position(s, 1)
        position = self._position(s)
        velocity = first * ds[:, None]
        acceleration = self._position(s, 2) * ds[:, None] ** 2 + first * dds[:, None]
        if self._bridge_duration:
            active = t < self._bridge_duration
            z = t[active] / self._bridge_duration
            coefficients = self._bridge_coefficients
            for order, target in enumerate((position, velocity, acceleration)):
                derivative = np.polynomial.polynomial.polyder(coefficients, m=order, axis=0)
                correction = np.polynomial.polynomial.polyval(z, derivative).T
                target[active] += correction / self._bridge_duration ** order
        rate = self._rotation(s, 1)
        angular_velocity = rate * ds[:, None]
        angular_acceleration = self._rotation(s, 2) * ds[:, None] ** 2 + rate * dds[:, None]
        quaternion = self._rotation(s).as_quat()[:, [3, 0, 1, 2]]
        # Outside the interval this is a stationary endpoint hold, not spline
        # extrapolation. At t=0 preserve the explicitly requested initial state.
        outside = (times < 0) | (times > self.duration)
        for derivative in (velocity, acceleration, angular_velocity, angular_acceleration):
            derivative[outside] = 0.
        return dict(pos=position, vel=velocity, acc=acceleration,
                    quaternion=quaternion, angular_velocity=angular_velocity,
                    angular_acceleration=angular_acceleration)

    def sample(self, time):
        """Sample one time, or a one-dimensional time array, in seconds."""
        times = np.asarray(time, dtype=float)
        if times.ndim > 1 or not np.isfinite(times).all():
            raise ValueError("time must be finite and scalar or one-dimensional")
        scalar = times.ndim == 0
        data = self._evaluate(np.atleast_1d(times))
        return {key: value[0] if scalar else value for key, value in data.items()}
