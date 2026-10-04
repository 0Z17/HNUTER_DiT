"""Evaluate DiT references through the industrial scene's actual flight physics.

The installed simulation repository owns the controller and LiDAR. This module
only loads the scene, schedules sensing/planning and records physical outcomes.
Collision geometry is used solely by the evaluator, never by OnlinePlanner.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np


DT = 0.001
ROBOT_PATH = "/World/HNUTER"


def save_scan(path, frame):
    np.savez_compressed(path, points_sensor=frame.points, points_world=frame.points_world,
                        world_T_sensor=frame.world_T_sensor, ranges=frame.ranges,
                        ray_directions=frame.ray_directions, point_timestamps=frame.point_timestamps,
                        simulation_time=np.float64(frame.sim_time), frame_index=np.int64(frame.frame_index))


def json_value(value):
    """Keep diagnostic JSON valid even after a non-finite physical failure."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluation_passed(report):
    """A completed flight also requires a healthy planner and full contact truth."""
    collision = report.get("collision", {})
    planner = report.get("planner", {})
    return bool(report.get("completed") and report.get("goal_reached")
                and report.get("pose_goal_reached", True)
                and collision.get("complete") and not collision.get("collision_detected")
                and planner.get("failed") is False)


class GoalProgress:
    """Require a continuous slow hold near the goal; any collision invalidates it."""

    def __init__(self, goal_position, distance_tolerance=.35, speed_tolerance=.10, settle_seconds=.5):
        self.goal = np.asarray(goal_position, dtype=float)
        if self.goal.shape != (3,) or not np.isfinite(self.goal).all():
            raise ValueError("goal_position must be a finite 3-vector")
        self.distance_tolerance = float(distance_tolerance)
        self.speed_tolerance = float(speed_tolerance)
        self.settle_required = float(settle_seconds)
        if not all(math.isfinite(value) and value > 0 for value in
                   (self.distance_tolerance, self.speed_tolerance, self.settle_required)):
            raise ValueError("Goal tolerances and settling duration must be positive and finite")
        self.first_goal_time = None
        self.settle_start = None
        self.settled_seconds = 0.0
        self.minimum_distance = None
        self.distance = self.speed = None
        self.last_time = None
        self.collision_latched = False
        self.reached = False

    def observe(self, state, sim_time, collided=False):
        position, velocity = np.asarray(state["position"]), np.asarray(state["velocity"])
        if (position.shape != (3,) or velocity.shape != (3,) or
                not np.isfinite(position).all() or not np.isfinite(velocity).all() or
                not math.isfinite(sim_time) or sim_time < 0 or
                (self.last_time is not None and sim_time <= self.last_time)):
            raise ValueError("Goal progress requires finite state and increasing simulation time")
        self.last_time = float(sim_time)
        self.distance = float(np.linalg.norm(position - self.goal))
        self.speed = float(np.linalg.norm(velocity))
        self.minimum_distance = self.distance if self.minimum_distance is None else min(self.minimum_distance, self.distance)
        self.collision_latched |= bool(collided)
        qualifying = (not self.collision_latched and self.distance <= self.distance_tolerance
                      and self.speed <= self.speed_tolerance)
        if qualifying:
            if self.settle_start is None:
                self.settle_start = float(sim_time)
            self.settled_seconds = float(sim_time - self.settle_start)
        else:
            self.settle_start = None
            self.settled_seconds = 0.0
        self.reached = qualifying and self.settled_seconds + 1e-12 >= self.settle_required
        if self.reached and self.first_goal_time is None:
            self.first_goal_time = float(sim_time)
        return self.reached

    def report(self):
        return {"goal_reached": bool(self.reached), "first_goal_time": self.first_goal_time,
                "goal_settled_seconds": self.settled_seconds,
                "final_goal_distance_m": self.distance, "final_speed_mps": self.speed,
                "minimum_goal_distance_m": self.minimum_distance,
                "goal_distance_tolerance_m": self.distance_tolerance,
                "goal_speed_tolerance_mps": self.speed_tolerance,
                "goal_settle_required_seconds": self.settle_required}


class ContactMonitor:
    """Latch every actual robot/environment contact at the 1 ms physics rate."""

    def __init__(self, stage):
        self.stage = stage
        self.model, self.solver = stage.model, stage.solver
        self.cpu = bool(stage.cfg.solver_cfg.use_mujoco_cpu)
        self.mapping = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy())
        if self.mapping.ndim == 1:
            self.mapping = self.mapping[None, :]
        self.labels = list(self.model.shape_label)
        robot_bodies = {i for i, label in enumerate(self.model.body_label)
                        if label == ROBOT_PATH or label.startswith(ROBOT_PATH + "/")}
        bodies = self.model.shape_body.numpy()
        self.robot_shapes = np.array([
            label == ROBOT_PATH or label.startswith(ROBOT_PATH + "/") or int(bodies[i]) in robot_bodies
            for i, label in enumerate(self.labels)], dtype=bool)
        self.samples = self.contact_steps = self.penetration_steps = 0
        self.overflow_steps = self.unmapped_contacts = 0
        self.first_collision = None
        self.max_penetration = 0.0
        self.minimum_distance = None

    def sample(self, sim_time):
        if self.stage.model is not self.model or not self.stage.initialized:
            raise RuntimeError("Collision monitor physics was reset/rebuilt")
        self.samples += 1
        if self.cpu:
            data = self.solver.mj_data
            count = int(data.ncon)
            geom = np.array([[data.contact[i].geom1, data.contact[i].geom2]
                             for i in range(count)], dtype=np.int64).reshape(-1, 2)
            distance = np.array([data.contact[i].dist for i in range(count)])
            worlds = np.zeros(count, dtype=np.int64)
        else:
            data = self.solver.mjw_data
            count = int(data.nacon.numpy().reshape(-1)[0])
            capacity = int(data.naconmax)
            if count > capacity:
                self.overflow_steps += 1
            count = min(count, capacity)
            if not count:
                return False
            geom = data.contact.geom.numpy()[:count].astype(np.int64)
            distance = data.contact.dist.numpy()[:count]
            worlds = data.contact.worldid.numpy()[:count].astype(np.int64)
        if not count:
            return False
        valid = ((worlds >= 0) & (worlds < len(self.mapping)) &
                 (geom >= 0).all(axis=1) & (geom < self.mapping.shape[1]).all(axis=1))
        self.unmapped_contacts += int((~valid).sum())
        geom, distance, worlds = geom[valid], distance[valid], worlds[valid]
        shapes = self.mapping[worlds[:, None], geom].astype(np.int64)
        valid = (shapes >= 0).all(axis=1) & (shapes < len(self.labels)).all(axis=1)
        self.unmapped_contacts += int((~valid).sum())
        shapes, distance = shapes[valid], distance[valid]
        cross = self.robot_shapes[shapes[:, 0]] != self.robot_shapes[shapes[:, 1]]
        shapes, distance = shapes[cross], distance[cross]
        if not len(distance):
            return False
        if not np.isfinite(distance).all():
            raise RuntimeError("Non-finite robot contact distance")
        minimum = float(distance.min())
        self.minimum_distance = minimum if self.minimum_distance is None else min(self.minimum_distance, minimum)
        colliding = distance <= 0.0  # Positive contact margins are not collisions.
        if not colliding.any():
            return False
        self.contact_steps += 1
        self.penetration_steps += int(minimum < -1e-4)
        self.max_penetration = max(self.max_penetration, -minimum)
        if self.first_collision is None:
            self.first_collision = {
                "simulation_time": float(sim_time),
                "pairs": [{"shape_a": self.labels[int(pair[0])], "shape_b": self.labels[int(pair[1])],
                           "distance_m": float(dist)}
                          for pair, dist in zip(shapes[colliding][:8], distance[colliding][:8])],
            }
        return True

    def report(self):
        return {"backend": "native MuJoCo contacts" if self.cpu else "MuJoCo Warp GPU contacts",
                "samples": self.samples, "collision_detected": self.first_collision is not None,
                "contact_steps": self.contact_steps, "penetration_steps": self.penetration_steps,
                "first_collision": self.first_collision, "max_penetration_m": self.max_penetration,
                "minimum_contact_distance_m": self.minimum_distance,
                "overflow_steps": self.overflow_steps, "unmapped_contacts": self.unmapped_contacts,
                "complete": self.samples > 0 and self.overflow_steps == 0 and self.unmapped_contacts == 0,
                "scope": "Robot versus environment only; self contacts excluded; evaluator-only truth"}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, default=Path("/sim"))
    parser.add_argument("--scene", default="scene_hnuter.usda")
    parser.add_argument("--mode", choices=("baseline", "online", "temporal", "sdf"), default="online")
    parser.add_argument("--sdf-checkpoint", type=Path)
    parser.add_argument("--sdf-goal-rpy", type=float, nargs=3, default=(.08,-.06,.25))
    parser.add_argument("--sdf-scene-index", type=int, help="Author two static evaluation obstacles; goal is set from this held-out scenario")
    parser.add_argument("--temporal-checkpoint", type=Path,
                        help="Explicit trained temporal checkpoint; required for --mode temporal")
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--duration", type=float, default=40.0, help="Finite simulation seconds")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-speed", type=float, default=.3, help="Trajectory speed limit in m/s")
    parser.add_argument("--max-acceleration", type=float, default=.4, help="Trajectory acceleration limit in m/s^2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dynamic-obstacle", action="store_true")
    parser.add_argument("--dynamic-profile", choices=("legacy", "constant_incoming", "constant_outgoing"),
                        default="legacy", help="Evaluator obstacle motion; never supplied to the planner")
    parser.add_argument("--mujoco-cpu", action="store_true")
    parser.add_argument("--camera", choices=("follow", "fpv"), default="follow")
    parser.add_argument("--capture-views", action="store_true")
    parser.add_argument("--record-video", action="store_true", help="Record actual rendered frames and a physics-clock manifest")
    parser.add_argument("--video-fps", type=int, choices=(10, 20), default=10, help="Frames per simulation second")
    parser.add_argument("--real-time", action="store_true", help="Throttle only when simulation is faster than wall time")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < DT:
        parser.error("duration must be finite and at least 0.001 seconds")
    if not np.isfinite([args.max_speed,args.max_acceleration]).all() or min(args.max_speed,args.max_acceleration)<=0:
        parser.error("motion limits must be finite and positive")
    if args.goal is not None and not np.isfinite(args.goal).all():
        parser.error("goal must contain three finite world coordinates")
    if args.mode == 'temporal':
        if args.temporal_checkpoint is None or not args.temporal_checkpoint.is_file():
            parser.error("temporal mode requires an existing --temporal-checkpoint file")
    elif args.temporal_checkpoint is not None:
        parser.error("--temporal-checkpoint is only accepted with --mode temporal")
    if args.mode == 'sdf':
        if args.sdf_checkpoint is None or not args.sdf_checkpoint.is_file():
            parser.error('sdf mode requires an existing --sdf-checkpoint')
        if args.dynamic_obstacle:
            parser.error('sdf pilot supports static obstacles only')
        if args.sdf_scene_index is not None and args.goal is not None:
            parser.error('sdf fixture sets its own goal; omit --goal')
    elif args.sdf_checkpoint is not None or args.sdf_scene_index is not None:
        parser.error('SDF checkpoint/fixture requires --mode sdf')
    if args.dynamic_profile != 'legacy' and not args.dynamic_obstacle:
        parser.error("non-legacy --dynamic-profile requires --dynamic-obstacle")
    if not (args.sim_root / "controllers/hnuter/newton_bridge.py").is_file():
        parser.error("sim-root must point to the industrial simulation repository")
    if not (args.sim_root / args.scene).is_file():
        parser.error("scene does not exist inside sim-root")
    return args


def main(argv=None):
    args = arguments(argv)
    root = Path(__file__).resolve().parents[1]
    output = args.output or Path("results/industrial") / f"{args.mode}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    output = (root / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.sim_root.resolve()))
    report = {"passed": False, "completed": False, "mode": args.mode, "seed": args.seed,
              "scene": args.scene, "sim_root": str(args.sim_root), "physics_dt": DT,
              "requested_duration": args.duration, "dynamic_obstacle": args.dynamic_obstacle,
              "dynamic_profile": args.dynamic_profile,
              "goal_reached": False, "nonfinite_state": False}
    import hashlib
    report['implementation_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root/'hnuter_dit').glob('*.py')}
    app = planner = bridge = capture = video = contacts = obstacle = ns = timeline = last_frame = goal_progress = None
    rows, lidar_summary = [], []
    step = 0
    exit_code = 1
    wall_start = time.monotonic()
    try:
        # This custom Isaac image requires Torch before SimulationApp.
        import torch
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": args.headless, "create_new_stage": False,
                             "limit_cpu_threads": 8, "width": 1280, "height": 800,
                             "renderer": "RaytracedLighting"})
        import carb
        import omni.usd
        import omni.kit.app
        import omni.timeline
        from pxr import Sdf, UsdGeom
        from isaacsim.core.simulation_manager import SimulationManager
        from controllers.hnuter import GeometricController
        from controllers.hnuter.newton_bridge import NewtonHnuterBridge
        from perception.mid360 import Mid360Lidar
        from tools.viewport_camera import bind_camera

        carb.settings.get_settings().set("/exts/isaacsim.physics.newton/auto_switch_on_startup", False)
        manager = omni.kit.app.get_app().get_extension_manager()
        for extension in ("isaacsim.physics.newton", "isaacsim.physics.newton.tensors"):
            manager.set_extension_enabled_immediate(extension, True)
        opening = asyncio.ensure_future(omni.usd.get_context().open_stage_async(str(args.sim_root / args.scene)))
        while not opening.done():
            app.update()
        opened, error = opening.result()
        if not opened:
            raise RuntimeError(error)
        usd_stage = omni.usd.get_context().get_stage()
        usd_stage.SetEditTarget(usd_stage.GetSessionLayer())
        robot = usd_stage.GetPrimAtPath(ROBOT_PATH)
        physics = usd_stage.GetPrimAtPath("/World/PhysicsScene")
        if not robot or not physics:
            raise RuntimeError("Scene requires /World/HNUTER and /World/PhysicsScene")
        robot.GetVariantSet("Physics").SetVariantSelection("mujoco")
        authored_origin = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
            usd_stage.GetPrimAtPath(ROBOT_PATH + "/Geometry/drone")).ExtractTranslation(), dtype=float)
        if args.sdf_scene_index is not None:
            from hnuter_dit.sdf_data import scenario
            from pxr import Gf, UsdPhysics
            fixture = scenario(args.sdf_scene_index)
            for index, box in enumerate(fixture['boxes'][:2]):
                cube = UsdGeom.Cube.Define(usd_stage, f'/World/SDFStaticFixture/Box{index}')
                cube.CreateSizeAttr(1.)
                position = authored_origin + np.asarray(box[:3]) - np.asarray(fixture['origin'])
                cube.AddTranslateOp().Set(Gf.Vec3d(*position))
                cube.AddScaleOp().Set(Gf.Vec3f(*box[3:]))
                cube.CreateDisplayColorAttr([Gf.Vec3f(.9,.32,.08)])
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            args.goal = authored_origin + np.asarray(fixture['target'][:3])
            args.sdf_goal_rpy = fixture['target'][3:]
            report['static_sdf_fixture'] = fixture
        if args.dynamic_obstacle:
            authored_goal = np.asarray(args.goal) if args.goal is not None else authored_origin + [0, 3, 0]
            if args.dynamic_profile == 'legacy':
                from hnuter_dit.industrial_obstacle import SlidingObstacle
                obstacle = SlidingObstacle.author(usd_stage, authored_origin, authored_goal)
            else:
                from hnuter_dit.industrial_obstacle import ConstantVelocityObstacle
                direction = 1 if args.dynamic_profile == 'constant_incoming' else -1
                obstacle = ConstantVelocityObstacle.author(usd_stage, authored_origin, authored_goal, direction=direction)
        physics.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(1000)
        if not SimulationManager.switch_physics_engine("newton"):
            raise RuntimeError("Could not activate Newton")
        SimulationManager.set_default_physics_scene("/World/PhysicsScene")
        for scene in SimulationManager.get_physics_scenes():
            scene.set_dt(DT)
        physics.CreateAttribute("mjc:option:timestep", Sdf.ValueTypeNames.Double).Set(DT)
        import isaacsim.physics.newton as nx
        import mujoco
        import newton
        ns = nx.acquire_stage()
        ns.cfg.time_step_app = False
        ns.cfg.physics_frequency = 1000
        ns.cfg.num_substeps = 1
        ns.cfg.use_cuda_graph = not args.mujoco_cpu
        ns.cfg.solver_cfg.integrator = "rk4"
        ns.cfg.solver_cfg.iterations = 100
        ns.cfg.solver_cfg.tolerance = 1e-10
        ns.cfg.solver_cfg.use_mujoco_cpu = args.mujoco_cpu
        ns.initialize_newton(None)
        bridge = NewtonHnuterBridge(ns)
        contacts = ContactMonitor(ns)
        if obstacle:
            obstacle.bind(ns)
        lidar = Mid360Lidar(ns)
        initial = bridge.read_state()
        origin = initial["position"].copy()
        default_goal_offset = [4.4, 0, 0] if args.mode == 'sdf' else [0, 3, 0]
        goal = np.asarray(args.goal, dtype=float) if args.goal is not None else origin + default_goal_offset
        planner_options = dict(goal_position=goal, device="cuda", seed=args.seed,
                               output_dir=output / "planner_worker", max_speed=args.max_speed,
                               max_acceleration=args.max_acceleration)
        if args.mode == 'sdf':
            from hnuter_dit.sdf_online import SDFOnlinePlanner
            planner = SDFOnlinePlanner(checkpoint=args.sdf_checkpoint, goal_rpy=args.sdf_goal_rpy, **planner_options)
            report['sdf_checkpoint'] = str(args.sdf_checkpoint)
        elif args.mode == 'temporal':
            from hnuter_dit.temporal_online import TemporalOnlinePlanner
            planner = TemporalOnlinePlanner(checkpoint=args.temporal_checkpoint, **planner_options)
            report['temporal_checkpoint'] = str(args.temporal_checkpoint)
        else:
            from hnuter_dit.online import OnlinePlanner
            planner = OnlinePlanner(mode=args.mode, **planner_options)
        controller = GeometricController()
        if args.mode == 'sdf':
            # Compile sensor kernels during declared initialization, without
            # advancing physics or supplying an extra history observation.
            sensor_warmup_start = time.perf_counter()
            Mid360Lidar(ns).capture(initial, 0.)
            report['sensor_startup_warmup_seconds'] = time.perf_counter()-sensor_warmup_start
        if args.capture_views:
            from tools.capture_views import ViewCapture
            capture = ViewCapture()
        if args.record_video:
            from hnuter_dit.flight_video import FlightVideoCapture
            video = FlightVideoCapture(output / 'video', fps=args.video_fps, physics_dt=DT, camera=args.camera)
        if not args.headless and not bind_camera(usd_stage, args.camera):
            raise RuntimeError(f"Could not bind {args.camera} camera")
        if not np.isclose(SimulationManager.get_physics_dt(), DT) or not np.isclose(ns.sim_dt, DT):
            raise RuntimeError("Actual physics timestep is not 0.001 seconds")
        SimulationManager.enable_default_callbacks(enable_warm_start=False, enable_on_stop=True,
                                                   enable_stage_open=True, enable_stage_close=True)
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        app.update()
        if ns.simulation_step_count != 0:
            raise RuntimeError("Uncontrolled physics step during initialization")
        if video:
            ns.update_fabric()
            for _ in range(24):
                app.update()
                if ns.simulation_step_count != 0:
                    raise RuntimeError("Video warmup unexpectedly advanced physics")
            video.save(physics_step=0, simulation_time=0., state=initial,
                       planner_status='history_warmup', goal_distance_m=float(np.linalg.norm(origin-goal)))
        report.update(initial_position=origin.tolist(), goal_position=goal.tolist(),
                      engine=SimulationManager.get_active_physics_engine(),
                      mujoco_version=mujoco.__version__, newton_version=newton.__version__,
                      torch_version=torch.__version__, mujoco_cpu=args.mujoco_cpu,
                      lidar=lidar.describe(), actuator_mapping=bridge.mapping)
        print("INDUSTRIAL_READY " + json.dumps({"mode": args.mode, "origin": origin.tolist(),
                                                "goal": goal.tolist(), "dynamic_obstacle": args.dynamic_obstacle}), flush=True)
        wall_start = time.monotonic()
        target_steps = round(args.duration / DT)
        goal_progress = GoalProgress(goal)
        last_contact = False
        last_sensor_wall = -np.inf
        raycast_seconds = []
        while app.is_running() and step < target_steps:
            if not timeline.is_playing():
                report["stop_reason"] = "timeline_paused_or_stopped"
                break
            if SimulationManager.get_active_physics_engine() != "newton":
                raise RuntimeError("Physics backend changed during evaluation")
            state = bridge.read_state()
            sim_time = step * DT
            goal_progress.observe(state, sim_time, collided=contacts.first_collision is not None)
            # Keep the standard 10 Hz simulation cadence. SDF also refreshes
            # after 100 ms wall time if physics/rendering runs slower, so a
            # nominal simulated rate cannot silently become stale sensing.
            if step % 100 == 0 or (args.mode == 'sdf' and time.monotonic()-last_sensor_wall >= .1):
                capture_wall = time.perf_counter()
                last_sensor_wall = time.monotonic()
                frame = lidar.capture(state, sim_time)
                raycast_seconds.append(time.perf_counter()-capture_wall)
                last_frame = frame
                if not lidar_summary:
                    save_scan(output / "first_lidar.npz", frame)
                if args.mode == 'sdf':
                    planner.observe(frame, state, sim_time, capture_wall_time=capture_wall)
                else:
                    planner.observe(frame, state, sim_time)
                lidar_summary.append({"t": sim_time, "frame_index": frame.frame_index,
                                      "hits": len(frame.points), "rays": frame.rays_count,
                                      "capture_wall_seconds": capture_wall-wall_start,
                                      "capture_seconds": raycast_seconds[-1]})
            desired = planner.command(state, sim_time)
            controller.set_desired_state(desired)
            if obstacle:
                obstacle.apply(sim_time)
            bridge.apply(controller.command(state))
            goal_distance = float(np.linalg.norm(state["position"] - goal))
            if step % 10 == 0:
                row = {"t": sim_time, "position": state["position"].tolist(),
                       "quaternion": state["quaternion"].tolist(), "velocity": state["velocity"].tolist(),
                       "angular_velocity": state["angular_velocity"].tolist(),
                       "target_position": controller.target_position.tolist(),
                       "target_rotation_matrix": controller.target_rotation_matrix.tolist(),
                       "desired": json_value(desired), "goal_distance_m": goal_distance,
                       "goal_settled_seconds": goal_progress.settled_seconds,
                       "tracking_error_m": float(np.linalg.norm(state["position"] - controller.target_position)),
                       "attitude_error_rad": float(np.arccos(np.clip(
                           (np.trace(controller.target_rotation_matrix.T @ state["rotation_matrix"]) - 1) / 2, -1, 1))),
                       "robot_contact": last_contact}
                if obstacle:
                    row["obstacle"] = obstacle.sample(sim_time)
                rows.append(row)
            if ns.simulation_step_count != step:
                raise RuntimeError("Physics advanced outside the controlled loop")
            previous_time = ns.sim_time
            SimulationManager.step(steps=1)
            step += 1
            if ns.simulation_step_count != step or not np.isclose(ns.sim_time - previous_time, DT):
                raise RuntimeError("Physics did not advance exactly one 1 ms step")
            last_contact = contacts.sample(step * DT)
            if video and video.due(step):
                ns.update_fabric()
                # Flush render readback while holding the integrated physical
                # state fixed. time_step_app=False forbids hidden integration.
                for _ in range(2):
                    app.update()
                    if ns.simulation_step_count != step:
                        raise RuntimeError("Video rendering unexpectedly advanced physics")
                video_state = bridge.read_state()
                video.save(physics_step=step, simulation_time=step*DT, state=video_state,
                           planner_status=getattr(planner, 'last_command_category',
                               planner.events[-1]['status'] if planner.events else 'initializing'),
                           goal_distance_m=float(np.linalg.norm(video_state['position']-goal)),
                           collision_detected=contacts.first_collision is not None)
            elif not args.headless and step % 33 == 0:
                ns.update_fabric()
                app.update()
                if ns.simulation_step_count != step:
                    raise RuntimeError("Viewport rendering unexpectedly advanced physics")
            if step % 1000 == 0:
                print("INDUSTRIAL_PROGRESS " + json.dumps({"t": step * DT, "position": state["position"].tolist(),
                                                          "goal_distance_m": goal_distance,
                                                          "collision": contacts.first_collision is not None,
                                                          "last_plan_status": planner.events[-1]['status'] if planner.events else None,
                                                          "accepted_plans": len(planner.raw_paths)}), flush=True)
            if args.real_time and step % 10 == 0:
                delay = wall_start + step * DT - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        if ns.simulation_step_count != step:
            raise RuntimeError("Final physics step count differs from controlled steps")
        final = bridge.read_state()
        # Include the final integrated sample, including its collision result.
        if goal_progress.last_time is None or step * DT > goal_progress.last_time:
            goal_progress.observe(final, step * DT, collided=contacts.first_collision is not None)
        errors = np.array([row["tracking_error_m"] for row in rows])
        collision = contacts.report()
        report.update(completed=step == target_steps, steps=step, simulation_seconds=step * DT,
                      actual_integrated_seconds=ns.sim_time, loop_wall_seconds=time.monotonic() - wall_start,
                      final_position=final["position"].tolist(), final_quaternion=final["quaternion"].tolist(),
                      final_velocity=final["velocity"].tolist(),
                      position_tracking_rmse_m=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                      position_tracking_max_error_m=float(errors.max()) if len(errors) else None,
                      attitude_tracking_max_error_rad=max((row["attitude_error_rad"] for row in rows), default=None),
                      collision=collision, lidar_frames=len(lidar_summary))
        report.update(goal_progress.report())
        if args.mode == 'sdf':
            from scipy.spatial.transform import Rotation
            goal_rotation = Rotation.from_euler('xyz', args.sdf_goal_rpy)
            final_rotation = Rotation.from_quat(final['quaternion'][[1,2,3,0]])
            angle = float((goal_rotation.inv()*final_rotation).magnitude())
            report.update(final_goal_angle_rad=angle, goal_rpy=list(args.sdf_goal_rpy), pose_goal_reached=angle < .08)
            report['raycast_seconds'] = dict(median=float(np.median(raycast_seconds)),p95=float(np.percentile(raycast_seconds,95)),maximum=float(max(raycast_seconds)))
            report['sensing_schedule'] = 'every 0.1 simulation seconds or 0.1 wall seconds, whichever comes first; timestamped actual physics state'
            for label, records in (('all',lidar_summary),('after_history_warmup',[r for r in lidar_summary if r['t']>=.6])):
                intervals = np.diff([r['capture_wall_seconds'] for r in records])
                if len(intervals):
                    report[f'sensing_wall_intervals_{label}_seconds'] = dict(median=float(np.median(intervals)),p95=float(np.percentile(intervals,95)),maximum=float(max(intervals)))
        if obstacle:
            report["obstacle"] = obstacle.report()
        report["planner"] = planner.close()
        planner = None
        if args.mode == 'sdf':
            # Retain numerical flight evidence even if optional rendering stalls.
            (output/'flight_before_render.json').write_text(json.dumps(json_value(report), indent=2, allow_nan=False)+'\n')
        if capture:
            ns.update_fabric()
            for _ in range(24):
                app.update()
            report["camera_images"] = capture.save(output)
            if ns.simulation_step_count != step:
                raise RuntimeError("Camera capture unexpectedly advanced physics")
        report["passed"] = evaluation_passed(report)
        exit_code = 0 if report["passed"] else 2
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["nonfinite_state"] = "non-finite" in str(exc).lower()
        report["passed"] = False
        exit_code = 1
        traceback.print_exc()
    finally:
        for name, resource in (("planner", planner), ("capture", capture), ("video", video)):
            if resource is None:
                continue
            try:
                result = resource.close()
                if name in ("planner", "video"):
                    report[name] = result
            except BaseException as exc:
                report.setdefault("cleanup_errors", []).append(f"{name}: {exc}")
                report["passed"] = False
                exit_code = 1
        if contacts is not None:
            report["collision"] = contacts.report()
        if goal_progress is not None:
            report.update(goal_progress.report())
        if obstacle is not None:
            report["obstacle"] = obstacle.report()
        report.setdefault("steps", step)
        report.setdefault("simulation_seconds", step * DT)
        report["wall_seconds_including_cleanup"] = time.monotonic() - wall_start
        try:
            if bridge is not None and ns is not None and ns.initialized and ns.model is bridge.model:
                bridge.zero()
            if timeline is not None:
                timeline.stop()
        except BaseException as exc:
            report.setdefault("cleanup_errors", []).append(f"physics: {exc}")
            report["passed"] = False
            exit_code = 1
        try:
            if last_frame is not None:
                save_scan(output / "last_lidar.npz", last_frame)
            for name, value in (("report.json", report), ("trajectory.json", rows), ("lidar_frames.json", lidar_summary)):
                (output / name).write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
            print("INDUSTRIAL_RESULT " + json.dumps(json_value({key: report.get(key) for key in
                  ("passed", "completed", "steps", "goal_reached", "final_goal_distance_m", "collision", "error")})), flush=True)
        finally:
            if app is not None:
                app.close(wait_for_replicator=False, skip_cleanup=False, exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
