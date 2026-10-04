"""Capture actual rendered flight frames on the controlled simulation clock.

This module neither integrates physics nor writes robot poses. The caller must
update Fabric and render before capture. Every saved frame is linked to the real
physics step, state and planner status; final MP4 encoding runs outside Isaac.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class FrameLedger:
    """Validate a contiguous simulation-clock sequence and persist its evidence."""

    def __init__(self, directory, *, fps=10, physics_dt=.001, camera='follow', width=960, height=540):
        if fps not in (10, 20) or not np.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError('video needs 10 or 20 fps and a positive physics timestep')
        self.step_interval = round(1 / (fps * physics_dt))
        if not np.isclose(self.step_interval * physics_dt * fps, 1., atol=1e-12, rtol=0):
            raise ValueError('video timestamps must fall on exact physics steps')
        if camera not in ('follow', 'fpv') or width < 1 or height < 1:
            raise ValueError('invalid capture camera or resolution')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames_directory = self.directory / 'frames'
        self.frames_directory.mkdir(exist_ok=True)
        if any(self.frames_directory.iterdir()) or (self.directory / 'manifest.json').exists():
            raise ValueError('video output must be empty; preserve previous flight evidence')
        self.fps, self.physics_dt, self.camera = fps, float(physics_dt), camera
        self.width, self.height = int(width), int(height)
        self.frames = []
        self.closed = False

    def due(self, physics_step):
        return physics_step == len(self.frames) * self.step_interval

    def save(self, pixels, *, physics_step, simulation_time, state, planner_status, goal_distance_m,
             collision_detected=False):
        if self.closed:
            raise RuntimeError('video ledger is closed')
        if not self.due(physics_step) or not np.isclose(simulation_time, physics_step*self.physics_dt, atol=1e-9, rtol=0):
            raise ValueError('video frame skipped, repeated or mismatched with the physics clock')
        data = np.asarray(pixels)
        if data.shape not in ((self.height, self.width, 3), (self.height, self.width, 4)) or data.dtype != np.uint8:
            raise ValueError('video annotator did not return the requested uint8 RGB(A) image')
        position = np.asarray(state['position'], dtype=float)
        quaternion = np.asarray(state['quaternion'], dtype=float)
        velocity = np.asarray(state['velocity'], dtype=float)
        if (position.shape != (3,) or velocity.shape != (3,) or quaternion.shape != (4,)
                or not all(np.isfinite(item).all() for item in (position, quaternion, velocity))
                or not np.isfinite(goal_distance_m)):
            raise ValueError('video requires a finite measured physical state')
        from PIL import Image
        filename = f'frames/{len(self.frames):06d}.png'
        path = self.directory / filename
        Image.fromarray(data[..., :3], 'RGB').save(path)
        self.frames.append(dict(index=len(self.frames), filename=filename,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), physics_step=int(physics_step),
            simulation_time=float(simulation_time), position=position.tolist(), quaternion_wxyz=quaternion.tolist(),
            velocity=velocity.tolist(), planner_status=str(planner_status), goal_distance_m=float(goal_distance_m),
            collision_detected=bool(collision_detected)))

    def describe(self):
        return dict(schema='hnuter_actual_flight_video_v1', camera=self.camera, width=self.width,
            height=self.height, fps=self.fps, physics_dt=self.physics_dt, step_interval=self.step_interval,
            frame_count=len(self.frames), first_simulation_time=self.frames[0]['simulation_time'] if self.frames else None,
            last_simulation_time=self.frames[-1]['simulation_time'] if self.frames else None,
            complete_sequence=all(frame['physics_step'] == i*self.step_interval for i,frame in enumerate(self.frames)),
            source='Actual actuator-driven Newton/MuJoCo flight; RGB rendered during the controlled run.',
            playback='One video second equals one simulation second; not a wall-clock real-time claim.',
            pose_replay=False, directory=str(self.directory), manifest=str(self.directory / 'manifest.json'))

    def close(self):
        result = self.describe()
        (self.directory / 'manifest.json').write_text(json.dumps(dict(result, frames=self.frames), indent=2,
                                                                allow_nan=False) + '\n')
        self.closed = True
        # The report binds this exact immutable manifest; the manifest cannot
        # contain its own hash without creating a recursive digest.
        result['manifest_sha256'] = hashlib.sha256((self.directory / 'manifest.json').read_bytes()).hexdigest()
        return result


class FlightVideoCapture:
    """Replicator RGB readback only; rendering/integration stay in the runner."""

    def __init__(self, directory, *, fps=10, physics_dt=.001, camera='follow'):
        self.ledger = FrameLedger(directory, fps=fps, physics_dt=physics_dt, camera=camera)
        self.product = self.annotator = None
        from tools.viewport_camera import CAMERA_PATHS
        import omni.replicator.core as rep
        try:
            self.product = rep.create.render_product(CAMERA_PATHS[camera], (self.ledger.width, self.ledger.height))
            self.annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            self.annotator.attach(self.product)
        except BaseException:
            self.close()
            raise

    def due(self, step):
        return self.ledger.due(step)

    def save(self, **kwargs):
        self.ledger.save(self.annotator.get_data(), **kwargs)

    def close(self):
        try:
            if self.product is not None:
                try:
                    if self.annotator is not None:
                        self.annotator.detach(self.product)
                finally:
                    self.product.destroy()
        finally:
            self.product = self.annotator = None
        return self.ledger.close()
