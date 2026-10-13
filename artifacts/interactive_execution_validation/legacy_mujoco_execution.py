"""Full HNUTER MuJoCo dynamics driven by a reproducible 6-DoF MPPI loop.

The optional native viewer and the physics data have a single owning worker.
The planner supplies world-frame box geometry and pose7 paths (wxyz).  The
selected path is time parameterized without changing its geometric segments.
The existing HNUTER rotor/tilt model and geometric actuator allocator are
loaded from ``--mujoco-root``; no kinematic teleportation is used in flight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import atexit
import copy
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import xml.etree.ElementTree as ET

import numpy as np


DEFAULT_MUJOCO_ROOT = Path("/home/z017/research/MuJoCo-for-HNUTER")
MODEL_FILENAME = "hnuter206_4_5kg.xml"
_IMPORT_LOCK = threading.Lock()
_VIEWER_LAUNCH_LOCK = threading.Lock()


def scene_geometry_hash(scene: dict[str, Any]) -> str:
    """Match the planner's geometry hash, including floor and collision flags."""
    geometry = [{
        "id": item.get("id"), "collision": item.get("collision", False),
        "role": item.get("role"), "type": item.get("type"),
        "pose": item.get("pose"), "size_xyz": item.get("size_xyz"),
    } for item in scene.get("obstacles", [])]
    return hashlib.sha256(json.dumps(
        geometry, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _poses(value: Any, label: str, minimum: int = 2) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    if (result.ndim != 2 or result.shape[1] != 7
            or len(result) < minimum or not np.isfinite(result).all()):
        raise ValueError(f"{label} 必须为有限的 [N, 7] pose7，N >= {minimum}")
    norm = np.linalg.norm(result[:, 3:7], axis=1)
    if np.any(norm < 1e-9):
        raise ValueError(f"{label} 包含零四元数")
    result[:, 3:7] /= norm[:, None]
    for index in range(1, len(result)):
        if result[index, 3:7] @ result[index - 1, 3:7] < 0:
            result[index, 3:7] *= -1
    return result


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(a, -1, 0)
    v, i, j, k = np.moveaxis(b, -1, 0)
    return np.stack((w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v), axis=-1)


def _rotation_vectors(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    conjugate = left.copy()
    conjugate[..., 1:] *= -1
    relative = _multiply(conjugate, right)
    relative *= np.where(relative[..., :1] < 0, -1.0, 1.0)
    length = np.linalg.norm(relative[..., 1:], axis=-1)
    angle = 2 * np.arctan2(length, np.maximum(relative[..., 0], 0))
    return relative[..., 1:] * np.divide(
        angle, length, out=np.full_like(length, 2.0), where=length > 1e-9
    )[..., None]


@dataclass(frozen=True)
class ExecutionConfig:
    speed_mps: float = 0.6
    max_acceleration_mps2: float = 1.5
    max_angular_speed_rps: float = 0.7
    max_angular_acceleration_rps2: float = 1.5
    control_dt: float = 0.05
    horizon: int = 32
    num_samples: int = 192
    settle_s: float = 2.5
    initial_hold_s: float = 0.5
    goal_tolerance_m: float = 0.20
    attitude_tolerance_deg: float = 15.0
    maximum_tracking_error_m: float = 0.75
    stop_on_contact: bool = True

    @classmethod
    def from_value(cls, value: "ExecutionConfig | dict[str, Any] | None") -> "ExecutionConfig":
        if isinstance(value, cls):
            result = value
        else:
            options = dict(value or {})
            unknown = set(options) - set(cls.__dataclass_fields__)
            if unknown:
                raise ValueError("未知执行选项: " + ", ".join(sorted(unknown)))
            result = cls(**options)
        result.validate()
        return result

    def validate(self) -> None:
        limits = {
            "speed_mps": (0.05, 2.0), "max_acceleration_mps2": (0.1, 4.0),
            "max_angular_speed_rps": (0.05, 2.0),
            "max_angular_acceleration_rps2": (0.1, 5.0),
            "control_dt": (0.02, 0.1), "settle_s": (0.0, 15.0),
            "initial_hold_s": (0.0, 5.0), "goal_tolerance_m": (0.02, 0.5),
            "attitude_tolerance_deg": (1.0, 40.0),
            "maximum_tracking_error_m": (0.1, 2.0),
        }
        for name, (lower, upper) in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} 必须是数值")
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} 必须在 [{lower}, {upper}]")
        for name, lower, upper in (("horizon", 8, 80), ("num_samples", 32, 1024)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} 必须是 [{lower}, {upper}] 范围的整数")
        if not isinstance(self.stop_on_contact, bool):
            raise ValueError("stop_on_contact 必须为布尔值")


class TimedPoseReference:
    """Minimum-jerk motion on exact input segments and geodesic rotations.

    Every source waypoint is reached at zero speed and acceleration.  This
    conservative timing keeps the planner's dense segment collision audit
    applicable, even when the sampled path contains a sharp corner.  Each
    segment duration bounds both translational and angular speed/acceleration
    analytically (peak quintic derivatives: 1.875 and 10/sqrt(3)).
    """

    def __init__(self, trajectory: Any, config: ExecutionConfig | None = None) -> None:
        self.config = ExecutionConfig.from_value(config)
        source = _poses(trajectory, "最终轨迹")
        translation = np.linalg.norm(np.diff(source[:, :3], axis=0), axis=1)
        angles = np.linalg.norm(_rotation_vectors(source[:-1, 3:7], source[1:, 3:7]), axis=1)
        keep = np.r_[True, (translation > 1e-9) | (angles > 1e-9)]
        self.poses = source[keep]
        if len(self.poses) < 2:
            raise ValueError("最终轨迹没有位移或姿态变化")
        self.displacements = np.diff(self.poses[:, :3], axis=0)
        self.rotation_vectors = _rotation_vectors(self.poses[:-1, 3:7], self.poses[1:, 3:7])
        lengths = np.linalg.norm(self.displacements, axis=1)
        angles = np.linalg.norm(self.rotation_vectors, axis=1)
        peak_acceleration = 10 / math.sqrt(3)
        c = self.config
        self.segment_durations = np.maximum.reduce((
            1.875 * lengths / c.speed_mps,
            np.sqrt(peak_acceleration * lengths / c.max_acceleration_mps2),
            1.875 * angles / c.max_angular_speed_rps,
            np.sqrt(peak_acceleration * angles / c.max_angular_acceleration_rps2),
            np.full(len(lengths), 2 * c.control_dt),
        ))
        self.knot_times = np.r_[0.0, np.cumsum(self.segment_durations)]
        self.motion_duration = float(self.knot_times[-1])
        self.duration = c.initial_hold_s + self.motion_duration

    def sample(self, times: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return 13-state references and six acceleration feedforwards."""
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("参考时间必须为有限的一维数组")
        motion_time = np.clip(times - self.config.initial_hold_s, 0, self.motion_duration)
        segment = np.minimum(np.searchsorted(self.knot_times, motion_time, side="right") - 1,
                             len(self.segment_durations) - 1)
        duration = self.segment_durations[segment]
        u = np.clip((motion_time - self.knot_times[segment]) / duration, 0, 1)
        phase = 10*u**3 - 15*u**4 + 6*u**5
        rate = 30*u**2 * (1-u)**2 / duration
        acceleration = (60*u - 180*u**2 + 120*u**3) / duration**2
        inactive = (times <= self.config.initial_hold_s) | (times >= self.duration)
        rate[inactive] = 0
        acceleration[inactive] = 0
        vector = self.rotation_vectors[segment]
        rotation = vector * phase[:, None]
        angle = np.linalg.norm(rotation, axis=1)
        half_scale = np.divide(np.sin(angle / 2), angle,
                               out=np.full_like(angle, 0.5), where=angle > 1e-9)
        delta = np.column_stack((np.cos(angle / 2), rotation * half_scale[:, None]))
        quaternion = _multiply(self.poses[segment, 3:7], delta)
        state = np.column_stack((
            self.poses[segment, :3] + self.displacements[segment] * phase[:, None],
            self.displacements[segment] * rate[:, None], quaternion,
            vector * rate[:, None],
        ))
        feedforward = np.column_stack((self.displacements[segment], vector)) * acceleration[:, None]
        return state, feedforward

    def describe(self) -> dict[str, Any]:
        return {
            "method": "segment_minimum_jerk_exact_polyline_slerp",
            "waypoint_count": len(self.poses), "duration_s": self.duration,
            "motion_duration_s": self.motion_duration,
            "initial_hold_s": self.config.initial_hold_s,
            "speed_mps": self.config.speed_mps,
            "max_acceleration_mps2": self.config.max_acceleration_mps2,
            "max_angular_speed_rps": self.config.max_angular_speed_rps,
            "max_angular_acceleration_rps2": self.config.max_angular_acceleration_rps2,
            "stops_at_waypoints": True,
        }


def build_scene_xml(
    scene: dict[str, Any], asset_root: Path,
    start_pose: Any, goal_pose: Any,
) -> str:
    """Replace only static world geometry in the original HNUTER model."""
    asset_root = Path(asset_root).resolve()
    root = ET.parse(asset_root / MODEL_FILENAME).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(asset_root / "meshes"))
    world = root.find("worldbody")
    if world is None:
        raise ValueError("HNUTER XML 缺少 worldbody")
    # The reference XML's large ground/sky enclosure must not become hidden
    # extra obstacles.  The input scene already contains its authoritative floor.
    for geom in list(world.findall("geom")):
        world.remove(geom)
    for index, item in enumerate(scene.get("obstacles", [])):
        if item.get("type") != "box":
            raise ValueError("MuJoCo 环境仅支持规划场景中的 box 障碍物")
        pose = _poses([[*item["pose"]["position"], *item["pose"]["quaternion_wxyz"]]],
                      f"障碍物 {index}", minimum=1)[0]
        size = np.asarray(item["size_xyz"], dtype=np.float64)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("障碍物 size_xyz 必须为正的有限三维尺寸")
        collision = bool(item.get("collision", False))
        floor = item.get("role") == "floor"
        rgba = "0.16 0.23 0.30 1" if floor else "0.34 0.49 0.59 1"
        if not item.get("visual", True):
            rgba = "0 0 0 0"
        ET.SubElement(world, "geom", {
            "name": f"scene_obstacle_{index}", "type": "box",
            "pos": " ".join(map(str, pose[:3])),
            "quat": " ".join(map(str, pose[3:7])),
            "size": " ".join(map(str, size / 2)), "rgba": rgba,
            "contype": "1" if collision else "0",
            "conaffinity": "6" if collision else "0",
        })
    endpoints = _poses([start_pose, goal_pose], "端点")
    for name, pose, rgba in zip(("desired_start", "desired_goal"), endpoints,
                               ("0.16 0.96 0.60 1", "1 0.40 0.24 1")):
        ET.SubElement(world, "geom", {
            "name": name, "type": "sphere", "size": "0.075",
            "pos": " ".join(map(str, pose[:3])), "rgba": rgba,
            "contype": "0", "conaffinity": "0",
        })
    return ET.tostring(root, encoding="unicode")


def _external_runtime(root: Path) -> tuple[Any, Any, Any]:
    """Load sibling controllers under private names to avoid import collisions."""
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("当前 Python 缺少 mujoco，请安装项目 simulation 依赖") from error
    suffix = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    with _IMPORT_LOCK:
        modules = []
        for label, path, package in (
            ("control", root / "hnuter_control.py", False),
            ("mppi", root / "mppi" / "__init__.py", True),
        ):
            name = f"_hnuter_dit_{label}_{suffix}"
            if name not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    name, path, submodule_search_locations=[str(path.parent)] if package else None
                )
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"无法加载 HNUTER 控制模块: {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(name, None)
                    raise
            modules.append(sys.modules[name])
    return mujoco, modules[0].HnuterController, modules[1]


def _quaternion_matrix(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    return np.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), axis=-1).reshape(quaternions.shape[:-1] + (3, 3))


class ScenePoseTrackingCost:
    """MPPI pose tracking plus the guidance robot-sphere/scene-OBB proxy."""

    def __init__(self, base_cost: Any, scene: dict[str, Any]) -> None:
        from .collision_loss import ROBOT_SPHERE_CENTERS, ROBOT_SPHERE_RADII

        self.base_cost = base_cost
        # Match the guidance proxy's float32 precision; actual physics and
        # pose tracking remain float64.  This halves the collision-array traffic.
        self.centers = np.asarray(ROBOT_SPHERE_CENTERS, dtype=np.float32)
        self.radii = np.asarray(ROBOT_SPHERE_RADII, dtype=np.float32)
        self.bounding_radius = float(np.max(np.linalg.norm(self.centers, axis=1) + self.radii))
        self.obstacles = []
        for item in scene.get("obstacles", []):
            if not item.get("collision", False):
                continue
            quaternion = np.asarray(item["pose"]["quaternion_wxyz"], dtype=np.float32)
            quaternion /= np.linalg.norm(quaternion)
            self.obstacles.append((
                np.asarray(item["pose"]["position"], dtype=np.float32),
                np.asarray(item["size_xyz"], dtype=np.float32) / 2,
                _quaternion_matrix(quaternion),
            ))

    def trajectory_cost(self, states: np.ndarray, controls: np.ndarray,
                        reference: np.ndarray) -> np.ndarray:
        result = self.base_cost.trajectory_cost(states, controls, reference)
        predicted = states[:, 1::2].astype(np.float32)
        rotations = _quaternion_matrix(predicted[..., 6:10])
        # World center coordinates: [samples, horizon/2, spheres, xyz].
        centers = predicted[..., None, :3] + self.centers @ np.swapaxes(rotations, -1, -2)
        minimum = np.full(predicted.shape[:2], np.inf, dtype=np.float32)
        for position, half_size, rotation in self.obstacles:
            # Exact broad phase: a box farther from every vehicle origin than
            # the sphere model's enclosing radius cannot contribute a penalty.
            body_local = (predicted[..., :3] - position) @ rotation
            body_distance = np.linalg.norm(np.maximum(np.abs(body_local) - half_size, 0), axis=-1)
            if np.all(body_distance > self.bounding_radius + 0.025):
                continue
            local = (centers - position) @ rotation
            delta = np.abs(local) - half_size
            signed = np.linalg.norm(np.maximum(delta, 0), axis=-1)
            signed += np.minimum(np.max(delta, axis=-1), 0)
            signed -= self.radii
            minimum = np.minimum(minimum, np.min(signed, axis=-1))
        violation = np.maximum(0.025 - minimum, 0)
        return result + 2.0e4 * np.sum(violation**2, axis=1)


def _native_display_available() -> bool:
    """Probe X11 safely before GLFW, whose failed initialization can exit."""
    if sys.platform in {"darwin", "win32"}:
        return True
    if not os.environ.get("DISPLAY"):
        return False
    try:
        x11 = ctypes.CDLL("libX11.so.6")
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        display = x11.XOpenDisplay(None)
        if not display:
            return False
        x11.XCloseDisplay(display)
        return True
    except OSError:
        return False


class _ViewerLines:
    def __init__(self, mujoco: Any) -> None:
        self.mujoco = mujoco
        self.identity = np.eye(3).reshape(-1)
        self.zero = np.zeros(3)

    def update(self, viewer: Any, candidates: list[np.ndarray], selected: np.ndarray | None,
               history: list[list[float]], nominal: np.ndarray | None) -> None:
        with viewer.lock():
            scene = viewer.user_scn
            scene.ngeom = 0
            # Draw selected and actual first so they survive the viewer's geometry limit.
            if selected is not None:
                self._line(scene, selected[:, :3], [1.0, 0.76, 0.16, 1], 4, 130)
            if len(history) > 1:
                self._line(scene, np.asarray(history)[:, :3], [1.0, 0.2, 0.66, 1], 4, 180)
            if nominal is not None:
                self._line(scene, nominal[:, :3], [0.15, 1.0, 0.4, 0.95], 2, 40)
            for index, candidate in enumerate(candidates):
                self._line(scene, candidate[:, :3], [0.12, 0.45 + index * 0.04, 0.95, 0.25], 1, 65)

    def _line(self, scene: Any, points: np.ndarray, color: list[float], width: float, limit: int) -> None:
        if len(points) > limit:
            points = points[np.unique(np.linspace(0, len(points) - 1, limit).astype(int))]
        for left, right in zip(points[:-1], points[1:]):
            if scene.ngeom >= scene.maxgeom:
                return
            if np.linalg.norm(right - left) < 1e-8:
                continue
            geom = scene.geoms[scene.ngeom]
            self.mujoco.mjv_initGeom(geom, self.mujoco.mjtGeom.mjGEOM_LINE,
                                   self.zero, self.zero, self.identity, np.asarray(color, dtype=np.float32))
            self.mujoco.mjv_connector(geom, self.mujoco.mjtGeom.mjGEOM_LINE,
                                     width, np.asarray(left), np.asarray(right))
            scene.ngeom += 1


class _ManagedPassiveViewer:
    """Wait for native rendering teardown before releasing a passive viewer.

    MuJoCo's public Handle.close() requests exit but does not join the daemon
    render thread.  Without that join, model replacement or interpreter exit
    can race the OpenGL context's destruction.  The thread is identified using
    the installed launcher's exact target and model/data arguments; interaction
    with the viewer still uses its public Handle API.
    """

    def __init__(self, handle: Any, render_thread: threading.Thread | None) -> None:
        self._handle = handle
        self._render_thread = render_thread
        self._close_requested = False
        self._close_lock = threading.Lock()
        # Run before MuJoCo's GLFW atexit handlers if the caller forgets close().
        atexit.register(self.close)

    @classmethod
    def launch(cls, viewer_module: Any, model: Any, data: Any) -> "_ManagedPassiveViewer":
        with _VIEWER_LAUNCH_LOCK:
            handle = viewer_module.launch_passive(model, data)
            render_thread = None
            if sys.platform != "darwin":
                for thread in threading.enumerate():
                    arguments = getattr(thread, "_args", ())
                    if (getattr(thread, "_target", None) is viewer_module._launch_internal
                            and len(arguments) >= 2
                            and arguments[0] is model and arguments[1] is data):
                        render_thread = thread
                        break
                if render_thread is None:
                    handle.close()
                    raise RuntimeError("无法定位 MuJoCo 渲染线程，不能安全管理 native viewer 生命周期")
            return cls(handle, render_thread)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def close(self) -> None:
        with self._close_lock:
            if not self._close_requested:
                self._handle.close()
                self._close_requested = True
            if self._render_thread is not None and self._render_thread is not threading.current_thread():
                self._render_thread.join(timeout=10.0)
                if self._render_thread.is_alive():
                    raise RuntimeError("MuJoCo 渲染线程尚未完成关闭")
            atexit.unregister(self.close)


class MujocoExecutionManager:
    """One background owner for a scene preview and MPPI dynamics execution."""

    def __init__(
        self, asset_root: Path = DEFAULT_MUJOCO_ROOT, *, viewer: bool = True,
        realtime: bool = True, on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.asset_root = Path(asset_root).resolve()
        self.viewer_enabled = bool(viewer)
        self.viewer = self.viewer_enabled
        self.realtime = bool(realtime)
        self.on_update = on_update
        self._lock = threading.RLock()
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._closed = False
        self._display_available: bool | None = None
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {
            "state": "idle", "message": "等待同步场景", "simulation_time_s": 0.0,
            "elapsed_s": 0.0, "progress": 0.0, "executed_poses": [],
            "history_times_s": [], "contact_count": 0, "tracking_error_m": None,
            "goal_error_m": None, "attitude_error_deg": None,
            "viewer_enabled": self.viewer_enabled, "viewer_open": False,
            "scene_geometry_hash": None,
        }

    def describe(self) -> dict[str, Any]:
        missing = [str(path) for path in (
            self.asset_root / MODEL_FILENAME, self.asset_root / "hnuter_control.py",
            self.asset_root / "mppi" / "__init__.py", self.asset_root / "meshes",
        ) if not path.exists()]
        installed = importlib.util.find_spec("mujoco") is not None
        if self._display_available is None:
            self._display_available = not self.viewer_enabled or _native_display_available()
        viewer_available = self._display_available
        return {
            "available": installed and not missing,
            "asset_root": str(self.asset_root), "model": MODEL_FILENAME,
            "physics": "full_HNUTER_rotor_tilt_MuJoCo",
            "controller": "6DoF_residual_MPPI_and_geometric_actuator_allocation",
            "viewer_enabled": self.viewer_enabled, "viewer_available": viewer_available,
            "realtime": self.realtime, "defaults": asdict(ExecutionConfig()),
            "reason": ("缺少 mujoco Python 依赖" if not installed else
                       "缺少 HNUTER 模型/控制器文件: " + ", ".join(missing) if missing else None),
        }

    def _ensure_available(self) -> None:
        description = self.describe()
        if not description["available"]:
            raise RuntimeError(description["reason"])
        if not description["viewer_available"]:
            raise RuntimeError("当前进程无法访问图形显示，使用 --headless 运行物理仿真")
        if self._closed:
            raise RuntimeError("MuJoCo 执行器已经关闭")

    def preview(self, scene: dict[str, Any], start_pose: Any, goal_pose: Any) -> dict[str, Any]:
        self._ensure_available()
        endpoints = _poses([start_pose, goal_pose], "端点")
        # Parse and validate scene geometry synchronously before accepting it.
        xml = build_scene_xml(scene, self.asset_root, *endpoints)
        with self._lock:
            if self._status["state"] in {"running", "starting", "stopping"}:
                raise RuntimeError("MuJoCo 正在执行或加载，请先停止")
            command = {"kind": "preview", "scene": copy.deepcopy(scene),
                       "endpoints": endpoints, "xml": xml}
            self._queue(command)
            return copy.deepcopy(self._status)

    def start(
        self, scene: dict[str, Any], trajectory: Any, *, candidates: Any = None,
        seed: int = 1, options: ExecutionConfig | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_available()
        config = ExecutionConfig.from_value(options)
        path = _poses(trajectory, "最终轨迹")
        candidate_paths = [_poses(item, "候选轨迹") for item in (candidates if candidates is not None else [])]
        if len(candidate_paths) > 16:
            raise ValueError("最多显示 16 条候选轨迹")
        reference = TimedPoseReference(path, config)
        xml = build_scene_xml(scene, self.asset_root, path[0], path[-1])
        with self._lock:
            if self._status["state"] in {"running", "starting", "stopping"}:
                raise RuntimeError("MuJoCo 正在执行或加载，请先停止")
            command = {"kind": "start", "scene": copy.deepcopy(scene), "xml": xml,
                       "endpoints": path[[0, -1]], "path": path, "candidates": candidate_paths,
                       "reference": reference, "config": config, "seed": int(seed)}
            self._queue(command)
            return copy.deepcopy(self._status)

    def _queue(self, command: dict[str, Any]) -> None:
        self._stop_event.clear()
        self._status.update({
            "state": "starting", "message": "加载与规划界面一致的 MuJoCo 环境",
            "scene_geometry_hash": scene_geometry_hash(command["scene"]),
            "scene_id": command["scene"].get("environment_id", "edited_scene"),
            "progress": 0.0, "simulation_time_s": 0.0, "elapsed_s": 0.0,
            "executed_poses": [], "history_times_s": [], "contact_count": 0,
            "tracking_error_m": None, "goal_error_m": None, "attitude_error_deg": None,
            "success": None, "error": None,
        })
        if command["kind"] == "start":
            self._status["retiming"] = command["reference"].describe()
            self._status["options"] = asdict(command["config"])
            self._status["seed"] = command["seed"]
        else:
            for key in ("retiming", "options", "seed"):
                self._status.pop(key, None)
        self._commands.put(command)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, name="hnuter-mujoco", daemon=True)
            self._thread.start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._status)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._status["state"] == "stopping":
                return copy.deepcopy(self._status)
            self._stop_event.set()
            if self._status["state"] in {"starting", "running", "preview"} or self._status["viewer_open"]:
                self._status.update(state="stopping", message="正在停止 MuJoCo 执行", success=False)
                self._commands.put({"kind": "stop"})
            return copy.deepcopy(self._status)

    def close(self) -> None:
        self._closed = True
        self._stop_event.set()
        self._commands.put({"kind": "close"})
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=15.0)
            if self._thread.is_alive():
                raise RuntimeError("MuJoCo 工作线程尚未完成关闭")

    def _publish(self, **values: Any) -> None:
        with self._lock:
            self._status.update(values)
            callback_state = copy.deepcopy(self._status) if self.on_update else None
        if self.on_update and callback_state is not None:
            self.on_update(callback_state)

    def _worker(self) -> None:
        viewer = None
        controller = None
        temporary = None
        active_xml = None
        try:
            mujoco, controller_type, mppi_module = _external_runtime(self.asset_root)
            visualizer = _ViewerLines(mujoco)
            while not self._closed:
                try:
                    command = self._commands.get(timeout=0.03)
                except queue.Empty:
                    if viewer is not None:
                        if not viewer.is_running():
                            viewer.close()
                            viewer = None
                            self._publish(viewer_open=False)
                            if self.status()["state"] == "preview":
                                self._publish(state="stopped", message="MuJoCo 窗口已关闭", success=False)
                        else:
                            viewer.sync()
                    continue
                if command["kind"] == "close":
                    break
                if command["kind"] == "stop":
                    if viewer is not None:
                        viewer.close()
                        viewer = None
                    self._publish(state="stopped", message="执行已停止", success=False, viewer_open=False)
                    continue
                if self._stop_event.is_set():
                    continue
                try:
                    # A plan immediately following the matching preview keeps
                    # the same model and window; changed geometry needs a new model.
                    if controller is None or active_xml != command["xml"]:
                        if viewer is not None:
                            viewer.close()
                            viewer = None
                        if temporary is not None:
                            temporary.cleanup()
                        temporary = tempfile.TemporaryDirectory(prefix="hnuter_dit_mujoco_")
                        xml_path = Path(temporary.name) / "scene.xml"
                        xml_path.write_text(command["xml"], encoding="utf-8")
                        controller = controller_type(xml_path)
                        active_xml = command["xml"]
                    else:
                        mujoco.mj_resetData(controller.model, controller.data)
                    controller.set_freejoint_pose(command["endpoints"][0, :3], command["endpoints"][0, 3:7])
                    if self._stop_event.is_set():
                        continue
                    if self.viewer_enabled and (viewer is None or not viewer.is_running()):
                        import mujoco.viewer
                        viewer = _ManagedPassiveViewer.launch(mujoco.viewer, controller.model, controller.data)
                        viewer.cam.lookat[:] = np.mean(command["endpoints"][:, :3], axis=0)
                        viewer.cam.distance = 11.0
                        viewer.cam.azimuth = 135.0
                        viewer.cam.elevation = -25.0
                    self._publish(viewer_open=viewer is not None)
                    if command["kind"] == "preview":
                        self._publish(state="preview", message="MuJoCo 场景已同步，等待轨迹执行",
                                      current_pose=command["endpoints"][0].tolist())
                        continue
                    self._execute(mujoco, controller, mppi_module, viewer, visualizer, command)
                except Exception as error:
                    self._publish(state="failed", message=str(error), error=str(error), success=False)
        except Exception as error:
            self._publish(state="failed", message=str(error), error=str(error), success=False)
        finally:
            if viewer is not None:
                viewer.close()
            if temporary is not None:
                temporary.cleanup()
            self._publish(viewer_open=False)

    def _execute(self, mujoco: Any, controller: Any, module: Any, viewer: Any,
                 visualizer: _ViewerLines, command: dict[str, Any]) -> None:
        config: ExecutionConfig = command["config"]
        reference: TimedPoseReference = command["reference"]
        substeps = max(1, round(config.control_dt / controller.dt))
        control_dt = substeps * controller.dt
        dynamics = module.FullyActuatedUAVDynamics(dt=control_dt)
        cost = ScenePoseTrackingCost(
            module.PoseTrackingCost(terminal_multiplier=4.0, workspace_radius=None), command["scene"]
        )
        mppi_config = module.MPPIConfig(
            horizon=config.horizon, num_samples=config.num_samples,
            temperature=24.0, noise_sigma=(1.0, 1.0, 0.8, 1.2, 1.2, 1.0),
            control_min=(-4.0, -4.0, -3.5, -6.0, -6.0, -5.0),
            control_max=(4.0, 4.0, 3.5, 6.0, 6.0, 5.0),
            noise_correlation=0.60, likelihood_ratio_weight=0.08,
            action_continuity_weight=2.0, control_smoothing=0.20,
            num_iterations=1, seed=command["seed"],
        )
        mppi = module.ResidualMPPIController(dynamics, cost, mppi_config,
                                             rng=np.random.default_rng(command["seed"]))
        history: list[list[float]] = []
        history_times: list[float] = []
        errors: list[float] = []
        contacts_total = 0
        wall_start = time.perf_counter()
        nominal = None
        total_duration = reference.duration + config.settle_s
        self._publish(state="running", message="MPPI 正在执行完整 HNUTER 动力学运动",
                      duration_s=total_duration, controller="6DoF_residual_MPPI")
        failed_message = None
        while controller.data.time < total_duration:
            if self._stop_event.is_set():
                return
            if viewer is not None and not viewer.is_running():
                self._publish(state="stopped", message="MuJoCo 窗口已关闭，执行停止",
                              viewer_open=False, success=False)
                return
            simulation_time = float(controller.data.time)
            state = controller.get_mppi_pose_state()
            times = simulation_time + control_dt * np.arange(config.horizon + 1)
            desired, feedforward = reference.sample(times)
            update_start = time.perf_counter()
            result = mppi.command(state, desired, feedforward[:-1])
            update_ms = (time.perf_counter() - update_start) * 1000
            nominal = result.nominal_states
            low_level_desired = {
                "pos": desired[1, :3], "vel": desired[1, 3:6],
                "acc": result.action[:3], "quaternion": desired[1, 6:10],
                "angular_velocity": desired[1, 10:13], "angular_acceleration": result.action[3:],
            }
            for _ in range(substeps):
                if self._stop_event.is_set():
                    break
                controller.set_desired_state(low_level_desired)
                controller.update_control()
                mujoco.mj_step(controller.model, controller.data)
                external_contacts = 0
                for contact in controller.data.contact[:controller.data.ncon]:
                    names = [mujoco.mj_id2name(controller.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
                             for g in (contact.geom1, contact.geom2)]
                    if any(name.startswith("scene_obstacle_") for name in names):
                        external_contacts += 1
                contacts_total += external_contacts
                if external_contacts and config.stop_on_contact:
                    failed_message = "HNUTER 与环境发生接触，物理执行已中止"
                    break
            now = float(controller.data.time)
            actual = controller.get_mppi_pose_state()
            target, _ = reference.sample([now])
            if not np.isfinite(actual).all() or not np.isfinite(controller.data.qvel).all():
                failed_message = "MuJoCo 动力学状态出现非有限值"
                break
            pose = np.r_[actual[:3], actual[6:10]].tolist()
            history.append(pose)
            history_times.append(now)
            tracking_error = float(np.linalg.norm(actual[:3] - target[0, :3]))
            attitude_error = float(np.degrees(np.linalg.norm(_rotation_vectors(target[0, 6:10], actual[6:10]))))
            goal_error = float(np.linalg.norm(actual[:3] - command["path"][-1, :3]))
            errors.append(tracking_error)
            if tracking_error > config.maximum_tracking_error_m:
                failed_message = f"轨迹跟踪误差 {tracking_error:.3f} m 超过执行上限"
            self._publish(
                simulation_time_s=now, elapsed_s=time.perf_counter() - wall_start,
                progress=min(now / total_duration, 1.0), current_pose=pose,
                reference_pose=np.r_[target[0, :3], target[0, 6:10]].tolist(),
                tracking_error_m=tracking_error, goal_error_m=goal_error,
                attitude_error_deg=attitude_error, contact_count=contacts_total,
                executed_poses=history.copy(), history_times_s=history_times.copy(),
                position_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                maximum_tracking_error_m=max(errors), mppi_update_ms=update_ms,
                mppi_effective_sample_size=float(result.effective_sample_size),
                actuator_controls=controller.data.ctrl.tolist(),
                joint_positions=controller.data.qpos[7:].tolist(),
                joint_velocities=controller.data.qvel[6:].tolist(),
                angular_velocity=actual[10:13].tolist(),
                mppi_nominal_poses=np.column_stack((nominal[:, :3], nominal[:, 6:10])).tolist(),
            )
            if viewer is not None:
                visualizer.update(viewer, command["candidates"], command["path"], history, nominal)
                viewer.sync()
            if failed_message:
                break
            if self.realtime:
                delay = now - (time.perf_counter() - wall_start)
                if delay > 0:
                    self._stop_event.wait(min(delay, control_dt))
        if self._stop_event.is_set():
            return
        elif failed_message:
            self._publish(state="failed", message=failed_message, error=failed_message, success=False)
        else:
            state = controller.get_mppi_pose_state()
            goal_error = float(np.linalg.norm(state[:3] - command["path"][-1, :3]))
            attitude_error = float(np.degrees(np.linalg.norm(
                _rotation_vectors(command["path"][-1, 3:7], state[6:10]))))
            success = goal_error <= config.goal_tolerance_m and attitude_error <= config.attitude_tolerance_deg
            message = ("MPPI 动力学执行完成，已到达期望终点" if success else
                       f"执行结束但未达到终点容差：位置 {goal_error:.3f} m，姿态 {attitude_error:.1f}°")
            self._publish(state="completed" if success else "failed", message=message,
                          success=success, progress=1.0, goal_error_m=goal_error,
                          attitude_error_deg=attitude_error, error=None if success else message)
