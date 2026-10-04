"""A separate position-driven sliding obstacle for physical replanning tests.

Only the obstacle's actuator target is written. The solver integrates its
position; LiDAR and contacts see the same actual body. Its known geometry and
schedule belong to the evaluator and are never passed to OnlinePlanner.
"""
from __future__ import annotations

import math
import numpy as np


class SlidingObstacle:
    ROOT = "/World/DiTEvaluationObstacle"
    BASE = ROOT + "/FixedBase"
    BODY = ROOT + "/MovingBody"
    JOINT = ROOT + "/SlideJoint"
    EXTENT = (0.7, 0.7, 2.5)

    def __init__(self, anchor, motion_axis):
        self.anchor = np.asarray(anchor, dtype=float)
        self.motion_axis = int(motion_axis)
        self.stage = self.model = None
        self.samples = 0
        self.max_error = 0.0
        self.sum_error_sq = 0.0
        self.latest = None
        self.joint_limits = (-4.1, .1)

    def position_target(self, sim_time):
        return self.target(sim_time)

    def actuator_target(self, sim_time):
        return self.position_target(sim_time)

    @staticmethod
    def anchor_for_route(origin, goal, route_axis):
        return (origin + goal) / 2.

    @staticmethod
    def target(sim_time):
        """C1 position motion: -4 m at startup, crossing, blocking, retreat."""
        if sim_time < 3.0:
            return -4.0
        if sim_time < 5.0:
            return -4.0 + 2.0 * (1.0 - math.cos(math.pi * (sim_time - 3.0) / 2.0))
        if sim_time < 13.0:
            return 0.0
        if sim_time < 15.0:
            return -2.0 * (1.0 - math.cos(math.pi * (sim_time - 13.0) / 2.0))
        return -4.0

    @classmethod
    def author(cls, stage, origin, goal, **options):
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        origin, goal = np.asarray(origin, dtype=float), np.asarray(goal, dtype=float)
        delta = goal - origin
        horizontal = np.abs(delta[:2])
        route_axis = int(np.argmax(horizontal))
        if horizontal[route_axis] < 0.5 or horizontal[1-route_axis] > 1e-3:
            raise ValueError("Dynamic fixture requires an axis-aligned XY goal at least 0.5 m from the origin")
        anchor = cls.anchor_for_route(origin, goal, route_axis)
        motion_axis = 1 - route_axis
        instance = cls(anchor, motion_axis, **options)
        if stage.GetPrimAtPath(cls.ROOT):
            raise RuntimeError("Scene already contains the evaluation obstacle path")
        UsdGeom.Xform.Define(stage, cls.ROOT)
        base = UsdGeom.Xform.Define(stage, cls.BASE)
        base.AddTranslateOp().Set(Gf.Vec3d(*anchor))
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        base_mass = UsdPhysics.MassAPI.Apply(base.GetPrim())
        base_mass.CreateMassAttr(1.0)
        base_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1, 1, 1))
        fixed = UsdPhysics.FixedJoint.Define(stage, cls.ROOT + "/WorldFixedJoint")
        fixed.CreateBody1Rel().SetTargets([base.GetPath()])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
        fixed.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
        body = UsdGeom.Xform.Define(stage, cls.BODY)
        position = anchor.copy()
        position[motion_axis] += instance.position_target(0.)
        body.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(10.0)
        # Analytic box inertia; an explicit mass avoids density-dependent changes.
        sx, sy, sz = cls.EXTENT
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(10*(sy*sy+sz*sz)/12,
                                                10*(sx*sx+sz*sz)/12,
                                                10*(sx*sx+sy*sy)/12))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        box = UsdGeom.Cube.Define(stage, cls.BODY + "/CollisionVisual")
        box.CreateSizeAttr(1.0)
        box.AddScaleOp().Set(Gf.Vec3f(*cls.EXTENT))
        box.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.22, 0.06)])
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        joint = UsdPhysics.PrismaticJoint.Define(stage, cls.JOINT)
        joint.CreateAxisAttr("X" if motion_axis == 0 else "Y")
        joint.CreateBody0Rel().SetTargets([base.GetPath()])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLowerLimitAttr(instance.joint_limits[0])
        joint.CreateUpperLimitAttr(instance.joint_limits[1])
        joint.GetPrim().CreateAttribute("newton:linear:position", Sdf.ValueTypeNames.Float).Set(instance.position_target(0.))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(5000.0)
        drive.CreateDampingAttr(300.0)
        drive.CreateMaxForceAttr(10000.0)
        drive.CreateTargetPositionAttr(instance.actuator_target(0.))
        drive.CreateTargetVelocityAttr(0.0)
        return instance

    def bind(self, stage):
        self.stage, self.model = stage, stage.model
        self.body_index = list(stage.model.body_label).index(self.BODY)
        joint_index = list(stage.model.joint_label).index(self.JOINT)
        self.dof = int(stage.model.joint_qd_start.numpy()[joint_index])
        source = stage.solver.mjc_actuator_ctrl_source.numpy()
        indices = stage.solver.mjc_actuator_to_newton_idx.numpy()
        matches = np.flatnonzero((source == 0) & (indices == self.dof))
        if len(matches) != 1:
            raise RuntimeError("Sliding obstacle joint lacks exactly one real position actuator")
        self.actuator_index = int(matches[0])
        initial = stage.state_0.body_q.numpy()[self.body_index, :3]
        expected = self.anchor.copy()
        expected[self.motion_axis] += self.position_target(0.)
        if not np.allclose(initial, expected, atol=1e-4):
            raise RuntimeError(f"Sliding obstacle initial body position {initial} differs from {expected}")
        self.apply(0.0)

    def apply(self, sim_time):
        if not self.stage.initialized or self.stage.model is not self.model:
            raise RuntimeError("Sliding obstacle physics was reset/rebuilt")
        controls = self.stage.control.joint_target_pos.numpy()
        controls[self.dof] = self.actuator_target(sim_time)
        self.stage.control.joint_target_pos.assign(controls)

    def sample(self, sim_time):
        position = self.stage.state_0.body_q.numpy()[self.body_index, :3].astype(float)
        target = self.anchor.copy()
        target[self.motion_axis] += self.position_target(sim_time)
        error = float(np.linalg.norm(position - target))
        self.samples += 1
        self.max_error = max(self.max_error, error)
        self.sum_error_sq += error*error
        self.latest = {"t": float(sim_time), "actual_position": position.tolist(),
                       "target_position": target.tolist(), "tracking_error_m": error}
        return dict(self.latest)

    def report(self):
        return {"kind": "physical prismatic position actuator", "body_path": self.BODY,
                "anchor_world": self.anchor.tolist(), "motion_axis": "XY"[self.motion_axis],
                "full_extents_m": list(self.EXTENT), "samples": self.samples,
                "max_position_tracking_error_m": self.max_error if self.samples else None,
                "position_tracking_rmse_m": math.sqrt(self.sum_error_sq/self.samples) if self.samples else None,
                "latest": self.latest, "motion_times_seconds": [3, 5, 13, 15],
                "position_offsets_m": [-4, 0, 0, -4],
                "scope": "Evaluator fixture; known schedule/geometry never supplied to planner"}


class ConstantVelocityObstacle(SlidingObstacle):
    """Industrial transfer fixture for the small constant-speed curriculum.

    A real position actuator accelerates the obstacle from rest. The commanded
    path passes lateral offset -1 m at t=.6 s with +/- .25 m/s speed. Its actual
    startup transient and contact disturbances are measured, never overwritten.
    No target clock or velocity is supplied to the model.
    """

    def __init__(self, anchor, motion_axis, *, direction=1):
        super().__init__(anchor, motion_axis)
        if isinstance(direction, bool) or direction not in (-1, 1):
            raise ValueError("constant obstacle direction must be -1 or +1")
        self.velocity = .25 * direction
        self.joint_limits = (-64., 64.)

    @staticmethod
    def anchor_for_route(origin, goal, route_axis):
        # The fixed-yaw vehicle's lateral rotor envelope needs a larger Y
        # crossing distance. At 0.9 m a crossing can hit a hovering aircraft,
        # leaving no valid wait/go solution (retained in collection audits).
        anchor = np.asarray(origin, dtype=float).copy()
        anchor[route_axis] += np.sign(goal[route_axis]-origin[route_axis]) * (1.3 if route_axis == 1 else 1.)
        return anchor

    def position_target(self, sim_time):
        if not np.isfinite(sim_time) or sim_time < 0:
            raise ValueError("obstacle simulation time must be finite and nonnegative")
        return -1. + self.velocity * (sim_time - .6)

    def actuator_target(self, sim_time):
        # The drive's damping is a physical viscous term. Position feedforward
        # balances it at the requested speed; no body pose/velocity is written.
        return self.position_target(sim_time) + self.velocity * (300. / 5000.)

    def report(self):
        value = super().report()
        value.pop("motion_times_seconds")
        value.pop("position_offsets_m")
        value.update(profile="constant_incoming" if self.velocity > 0 else "constant_outgoing",
                     target_velocity_mps=self.velocity, target_anchor_time_s=.6,
                     lateral_offset_at_anchor_m=-1.,
                     crossing_distance_rule="1.0 m for X route; 1.3 m for Y route, fixed world-yaw vehicle",
                     initial_motion="physical actuator startup from rest; measured transient retained")
        return value
