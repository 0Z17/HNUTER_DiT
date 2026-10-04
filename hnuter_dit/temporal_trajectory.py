"""Execute a learned cubic B-spline on its original eight-second time axis.

Only NumPy/SciPy are imported here. A fixed-time prediction is never stretched,
arc-length parameterized, or shortened by deleting stationary control points.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, PPoly

FUTURE_TIMES = np.arange(65, dtype=np.float64) * .125
KNOTS = np.r_[np.zeros(4), np.arange(1, 13) * (8. / 13.), np.full(4, 8.)]


class FixedTimeTrajectory:
    """World-coordinate position spline, with measured attitude held constant.

The temporal pilot predicts translation only. Speed and acceleration maxima
are checked analytically on every cubic span. An inadmissible prediction is
rejected, never repaired by changing the obstacle encounter time.
"""

    def __init__(self, control_points, *, quaternion, max_speed=.3, max_acceleration=.4):
        controls = np.asarray(control_points, dtype=np.float64)
        q = np.asarray(quaternion, dtype=np.float64)
        if controls.shape != (16, 3) or not np.isfinite(controls).all():
            raise ValueError("temporal control_points must be finite [16,3]")
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise ValueError("temporal quaternion must be finite nonzero wxyz")
        if not np.isfinite([max_speed, max_acceleration]).all() or min(max_speed, max_acceleration) <= 0:
            raise ValueError("temporal motion limits must be positive and finite")
        self.control_points = controls.copy()
        self.quaternion = q / np.linalg.norm(q)
        self.duration = 8.
        self.spline = BSpline(KNOTS, controls, 3, extrapolate=False)
        self.velocity = self.spline.derivative(1)
        self.acceleration = self.spline.derivative(2)
        self.observed_maxima = self._extrema()
        self.limits = dict(speed=float(max_speed), acceleration=float(max_acceleration))
        for key, bound in self.limits.items():
            if self.observed_maxima[key] > bound * (1 + 1e-5) + 1e-7:
                raise ValueError(f"temporal {key} limit exceeded: {self.observed_maxima[key]:.6g} > {bound:.6g}")

    def _extrema(self):
        # ||v|| extrema satisfy v dot a = 0 (a cubic polynomial for a cubic
        # position span). ||a|| is convex on each span, so endpoints suffice.
        polynomials = [PPoly.from_spline((KNOTS, self.control_points[:, axis], 3)) for axis in range(3)]
        candidates = [0., 8.]
        for span, (left, right) in enumerate(zip(KNOTS[:-1], KNOTS[1:])):
            if right <= left:
                continue
            derivative = np.zeros(4)
            for polynomial in polynomials:
                c = polynomial.c[:, span]
                v = np.array([c[2], 2 * c[1], 3 * c[0]])
                a = np.array([2 * c[1], 6 * c[0]])
                product = np.polynomial.polynomial.polymul(v, a)
                derivative[:len(product)] += product
            roots = np.polynomial.Polynomial(derivative).roots()
            candidates.extend((left, right))
            candidates.extend(left + float(root.real) for root in roots
                              if abs(root.imag) < 1e-8 and 0 < root.real < right-left)
        times = np.asarray(candidates)
        return dict(speed=float(np.linalg.norm(self.velocity(times), axis=1).max()),
                    acceleration=float(np.linalg.norm(self.acceleration(times), axis=1).max()))

    def sample(self, time):
        times = np.asarray(time, dtype=np.float64)
        if times.ndim > 1 or not np.isfinite(times).all() or np.any(times < -1e-9) or np.any(times > self.duration+1e-9):
            raise ValueError("temporal sample time must remain inside the fixed [0,8] second prediction")
        scalar = times.ndim == 0
        t = np.clip(np.atleast_1d(times), 0., self.duration)
        result = dict(pos=self.spline(t), vel=self.velocity(t), acc=self.acceleration(t),
                      quaternion=np.repeat(self.quaternion[None], len(t), axis=0),
                      angular_velocity=np.zeros((len(t), 3)), angular_acceleration=np.zeros((len(t), 3)))
        return {key: value[0] if scalar else value for key, value in result.items()}

    @classmethod
    def from_prediction(cls, prediction, encoding, quaternion, **limits):
        """Validate the IPC samples against its spline, then restore world axes."""
        times = np.asarray(prediction["future_times"], dtype=float)
        if times.shape != (65,) or not np.allclose(times, FUTURE_TIMES, atol=1e-8, rtol=0):
            raise ValueError("temporal future_times must be the fixed 65 samples at 0.125 seconds")
        controls = np.asarray(prediction["control_points"], dtype=float)
        if controls.shape != (16, 3) or not np.isfinite(controls).all():
            raise ValueError("temporal prediction requires finite cubic control_points[16,3]")
        canonical = BSpline(KNOTS, controls, 3, extrapolate=False)
        for name, order in (("positions", 0), ("velocities", 1), ("accelerations", 2)):
            samples = np.asarray(prediction[name], dtype=float)
            if samples.shape != (65, 3) or not np.isfinite(samples).all() or not np.allclose(
                    samples, canonical(times, nu=order), atol=2e-5, rtol=2e-5):
                raise ValueError(f"temporal {name} disagree with the reported fixed-time spline")
        rotation = np.asarray(encoding["world_to_canonical"], dtype=float)
        origin = np.asarray(encoding["origin_world"], dtype=float)
        if (rotation.shape != (3, 3) or origin.shape != (3,) or
                not np.isfinite(rotation).all() or not np.isfinite(origin).all() or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or
                not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
            raise ValueError("invalid temporal world/canonical transform")
        return cls(controls @ rotation + origin, quaternion=quaternion, **limits)
