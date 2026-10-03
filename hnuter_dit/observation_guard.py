"""Conservative observed-surface and sensor-centerline execution checks.

The angular test is explicitly a sampled centerline visibility check, not a
certificate for the full vehicle volume. Missing returns remain unknown.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .model import quaternion_wxyz_to_matrix_numpy
from .collision_loss import ROBOT_SPHERE_CENTERS, ROBOT_SPHERE_RADII


class ObservationGuard:
    def __init__(self, frame):
        self.time = float(frame.sim_time)
        self.pose = np.array(frame.world_T_sensor, dtype=float, copy=True)
        self.points = np.array(frame.points, dtype=float, copy=True)
        if (not np.isfinite(self.time) or self.time < 0 or self.pose.shape != (4,4)
                or not np.isfinite(self.pose).all() or not np.allclose(self.pose[3], [0,0,0,1])
                or not np.allclose(self.pose[:3,:3].T @ self.pose[:3,:3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(self.pose[:3,:3]), 1, atol=1e-5)):
            raise ValueError('invalid scan pose or timestamp')
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not np.isfinite(self.points).all():
            raise ValueError('non-finite LiDAR return')
        self.world = self.points @ self.pose[:3,:3].T + self.pose[:3,3]
        self.ranges = np.linalg.norm(self.points,axis=1)
        valid = self.ranges > .1
        self.ranges = self.ranges[valid]
        self.directions = self.points[valid]/self.ranges[:,None]
        if hasattr(frame,'ray_directions') != hasattr(frame,'ranges'):
            raise ValueError('incomplete full ray frame')
        self.full_ray_frame = hasattr(frame,'ray_directions') and hasattr(frame,'ranges')
        if self.full_ray_frame:
            directions=np.asarray(frame.ray_directions,dtype=float)
            ranges=np.asarray(frame.ranges,dtype=float)
            if (directions.ndim!=2 or directions.shape[1]!=3 or ranges.shape!=(len(directions),)
                    or not np.isfinite(directions).all()
                    or not np.allclose(np.linalg.norm(directions,axis=1),1.,atol=1e-5)):
                raise ValueError('invalid full ray frame')
            # Query the nearest emitted ray, including misses. A neighboring hit
            # must not fill the direction of an explicitly unreturned beam.
            self.directions=directions.copy()
            self.ranges=ranges.copy()
        self.ray_tree = cKDTree(self.directions) if len(self.directions) else None
        self.point_tree = cKDTree(self.world) if len(self.world) else None

    def check(self, poses, *, sim_time, clearance=.20, angular_tolerance_degrees=None):
        # The measured 20k-ray scan has angular holes near 2 degrees. A 3-degree
        # cap plus a 5 cm transverse bound models finite sampling resolution;
        # it is not a free-volume certificate. Points-only callers retain 1.5.
        if angular_tolerance_degrees is None:
            angular_tolerance_degrees=3. if self.full_ray_frame else 1.5
        if not np.isfinite(angular_tolerance_degrees) or not 0 < angular_tolerance_degrees <= 3.:
            raise ValueError('angular tolerance must be in (0,3] degrees')
        poses=np.asarray(poses,dtype=float)
        if poses.ndim!=2 or poses.shape[1]!=7 or not np.isfinite(poses).all():
            raise ValueError('execution poses must be finite Nx7')
        if np.any(np.linalg.norm(poses[:,3:],axis=1)<1e-8):
            raise ValueError('nonzero pose quaternion required')
        if not np.isfinite(sim_time):
            raise ValueError('finite query time required')
        if sim_time-self.time > .3 or sim_time < self.time:
            return dict(allowed=False, reason='stale_scan', minimum_clearance_m=None, unknown_fraction=1.)
        if self.point_tree is None or self.ray_tree is None:
            return dict(allowed=False, reason='no_returns_unknown', minimum_clearance_m=None, unknown_fraction=1.)
        rotations=quaternion_wxyz_to_matrix_numpy(poses[:,3:])
        centers=poses[:,:3][:,None,:]+np.einsum('nij,sj->nsi',rotations,np.asarray(ROBOT_SPHERE_CENTERS))
        distances,_=self.point_tree.query(centers.reshape(-1,3))
        robot_clearance=float((distances.reshape(len(poses),-1)-np.asarray(ROBOT_SPHERE_RADII)-.12).min())
        # Verify the mounted sensor's proposed centerline using valid hit rays.
        # A no-return ray is never filled with max range or called free.
        sensor=poses[:,:3]+np.einsum('nij,j->ni',rotations,[0.,0.,.13])
        local=(sensor-self.pose[:3,3])@self.pose[:3,:3]
        lengths=np.linalg.norm(local,axis=1)
        near=lengths < .10  # Exclude the current sensor neighborhood; never label it observed.
        unit=local/np.maximum(lengths[:,None],1e-12)
        delta,index=self.ray_tree.query(unit)
        angle=2*np.arcsin(np.clip(delta/2,0,1))
        known=((~near) & (angle<=np.deg2rad(angular_tolerance_degrees))
               & (lengths*np.sin(angle)<=.05) & np.isfinite(self.ranges[index])
               & (self.ranges[index]>.1) & (lengths+.35 < self.ranges[index]))
        reason='clear_observed_prefix'
        if robot_clearance < clearance:
            reason='observed_obstacle'
        elif not (~near).any() or not known[~near].all():
            reason='unknown_centerline'
        return dict(allowed=reason=='clear_observed_prefix', reason=reason,
                    minimum_clearance_m=robot_clearance, unknown_fraction=float((~known[~near]).mean()) if (~near).any() else 1.,
                    excluded_current_sensor_neighborhood=int(near.sum()),
                    ray_evidence='nearest_emitted_ray_including_misses' if self.full_ray_frame else 'returns_only',
                    angular_tolerance_degrees=angular_tolerance_degrees,max_transverse_beam_distance_m=.05,
                    certification='sampled sensor centerline plus 42-sphere observed surface clearance; no full-volume free-space proof')
