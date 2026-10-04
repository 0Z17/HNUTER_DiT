"""Optional diagnostic capture; never advances physics or supplies model inputs."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np


class LoopProfiler:
    """Host wall-time attribution, including any CUDA waits in each phase.

    No extra GPU synchronization is inserted. These are end-to-end phase times,
    not isolated GPU kernel timings, and worker compute overlaps the main loop.
    """
    def __init__(self):
        self.samples = defaultdict(list)
        self.steady = defaultdict(list)

    def begin(self, sim_time):
        self.sim_time = sim_time
        self.previous = time.perf_counter()

    def mark(self, label):
        now = time.perf_counter()
        elapsed = now-self.previous
        self.samples[label].append(elapsed)
        if self.sim_time >= 1.:
            self.steady[label].append(elapsed)
        self.previous = now

    def describe(self):
        def summarize(data):
            total = sum(sum(v) for v in data.values())
            return {k: dict(count=len(v), total_seconds=float(sum(v)),
                    percent_of_profiled_wall=100*sum(v)/total if total else 0,
                    median_ms=float(np.median(v)*1000), p95_ms=float(np.percentile(v,95)*1000),
                    maximum_ms=float(max(v)*1000)) for k,v in data.items() if v}
        return dict(all=summarize(self.samples), after_first_simulation_second=summarize(self.steady),
            scope='Host wall phases, including CUDA waits; no extra synchronization. Worker runs concurrently.',
            caveat='Physics launch and later readback waits can fall in different phases; not pure solver kernel time.')


class PerceptionDemoCapture:
    def __init__(self, directory, fps=10, primary_camera='follow'):
        from .flight_video import FlightVideoCapture
        self.directory=Path(directory)
        self.camera='fpv' if primary_camera=='follow' else 'follow'
        self.video=FlightVideoCapture(self.directory/self.camera, fps=fps, camera=self.camera)
        self.scans=self.directory/'scans';self.scans.mkdir(exist_ok=True)
        self.records=[]

    def save(self, *, physics_step, simulation_time, state, planner, frame, goal, collision_detected):
        self.video.save(physics_step=physics_step, simulation_time=simulation_time, state=state,
            planner_status=planner.last_command_category, goal_distance_m=float(np.linalg.norm(state['position']-goal)),
            collision_detected=collision_detected)
        scan=None
        if frame is not None:
            if frame.sim_time > simulation_time+1e-9:
                raise ValueError('Future sensor data cannot accompany an earlier video frame')
            path=self.scans/f'{len(self.records):06d}.npz'
            np.savez_compressed(path, points_world=frame.points_world.astype(np.float32),
                world_T_sensor=frame.world_T_sensor, scan_time=np.float64(frame.sim_time),
                frame_index=np.int64(frame.frame_index))
            scan=dict(filename=str(path.relative_to(self.directory)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                time=float(frame.sim_time),hits=len(frame.points))
        last=planner.scans[-1] if planner.scans else None
        self.records.append(dict(index=len(self.records), physics_step=physics_step, simulation_time=simulation_time,
            scan=scan, map_sequence=last.get('sequence') if last else None,
            map_scan_time=last.get('scan_time') if last else None,
            sensor_response_seconds=last.get('sensor_to_response_wall_seconds') if last else None,
            map_seconds=last.get('map_seconds') if last else None,
            observed_clearance=last.get('observed_clearance') if last else None,
            accepted_plans=len(planner.raw_paths), guard_interventions=planner.guard_interventions,
            mapping_context={key:last[key] for key in ('segment','world_origin','world_frame','segment_start','map_local_scan_time') if key in last} if last else None))

    def close(self):
        camera=self.video.close()
        path=self.directory/'manifest.json'
        path.write_text(json.dumps(dict(schema='hnuter_synchronized_perception_demo_v1', camera=camera,
            frames=self.records, source='Measured Newton LiDAR and actual nvblox worker snapshots; no obstacle truth.'),
            indent=2,allow_nan=False)+'\n')
        return dict(manifest=str(path),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),camera=camera)


def configure_demo_camera(stage, factory=False):
    """Session-only camera framing for the static fixture or factory aisle.

    The original forward/right camera can sit inside a test obstacle. Only a
    camera child transform and lens are overridden; robot/obstacle poses and
    all authored assets remain untouched.
    """
    from pxr import Gf, Usd, UsdGeom
    from tools.viewport_camera import FOLLOW_CAMERA
    position=(-2.,0.,1.4) if factory else (-3.,1.5,2.2)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        camera=UsdGeom.Camera(stage.GetPrimAtPath(FOLLOW_CAMERA))
        transform=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position),Gf.Vec3d(0,0,0),Gf.Vec3d(0,0,1)).GetInverse()
        camera.GetPrim().GetAttribute('xformOp:transform').Set(transform)
        camera.GetFocalLengthAttr().Set(18. if factory else 24.)
    return dict(local_position=position,look_at=[0,0,0],focal_length_mm=18. if factory else 24.,scope='session-only visualization; no physical pose writes')
