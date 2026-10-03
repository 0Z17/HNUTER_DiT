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
    def author(cls, stage, origin, goal):
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        origin, goal = np.asarray(origin, dtype=float), np.asarray(goal, dtype=float)
        delta = goal - origin
        horizontal = np.abs(delta[:2])
        route_axis = int(np.argmax(horizontal))
        if horizontal[route_axis] < 0.5 or horizontal[1-route_axis] > 1e-3:
            raise ValueError("Dynamic fixture requires an axis-aligned XY goal at least 0.5 m from the origin")
        anchor = (origin + goal) / 2.0
        motion_axis = 1 - route_axis
        instance = cls(anchor, motion_axis)
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
        position[motion_axis] -= 4.0
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
        joint.CreateLowerLimitAttr(-4.1)
        joint.CreateUpperLimitAttr(0.1)
        joint.GetPrim().CreateAttribute("newton:linear:position", Sdf.ValueTypeNames.Float).Set(-4.0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(5000.0)
        drive.CreateDampingAttr(300.0)
        drive.CreateMaxForceAttr(10000.0)
        drive.CreateTargetPositionAttr(-4.0)
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
        expected[self.motion_axis] -= 4.0
        if not np.allclose(initial, expected, atol=1e-4):
            raise RuntimeError(f"Sliding obstacle initial body position {initial} differs from {expected}")
        self.apply(0.0)

    def apply(self, sim_time):
        if not self.stage.initialized or self.stage.model is not self.model:
            raise RuntimeError("Sliding obstacle physics was reset/rebuilt")
        controls = self.stage.control.joint_target_pos.numpy()
        controls[self.dof] = self.target(sim_time)
        self.stage.control.joint_target_pos.assign(controls)

    def sample(self, sim_time):
        position = self.stage.state_0.body_q.numpy()[self.body_index, :3].astype(float)
        target = self.anchor.copy()
        target[self.motion_axis] += self.target(sim_time)
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
