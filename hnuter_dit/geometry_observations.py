"""Causal measured-return point and local occupancy encodings.

This is a per-scan ray representation, not nvblox TSDF/ESDF fusion. Missing
returns remain unknown. Sampled free cells indicate ray evidence only and do
not certify the whole voxel or vehicle footprint free of obstacles.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
import numpy as np

from .temporal_observations import HISTORY_TIMES, quaternion_matrix

ROI_RADIUS = 8.
POINT_COUNT = 512
VOXEL_SIZE = .25
GRID_SIZE = 64
SCHEMAS = dict(box='canonical_observed_box_history4x32_v1',
               point='canonical_measured_point_history4x512_v1',
               voxel='canonical_ray_occupancy_history4x3x64_v1')


def input_spec(representation):
    if representation not in SCHEMAS:
        raise ValueError('input_representation must be box, point or voxel')
    shape = {'box': ([4, 32, 10], [4, 32]),
             'point': ([4, POINT_COUNT, 3], [4, POINT_COUNT]),
             'voxel': ([4, 3, GRID_SIZE, GRID_SIZE, GRID_SIZE], [4, GRID_SIZE, GRID_SIZE, GRID_SIZE])}[representation]
    return dict(input_representation=representation, schema=SCHEMAS[representation],
                tokens_shape=shape[0], mask_shape=shape[1],
                tokens_dtype='uint8' if representation == 'voxel' else 'float32')


def preprocessing_configuration(representation):
    specification = input_spec(representation)
    config = dict(**specification, roi_radius_m=ROI_RADIUS,
        coordinate_frame='one current measured body origin and goal-yaw frame',
        history_times=HISTORY_TIMES.tolist(), causal=True)
    if representation == 'box':
        config.update(algorithm='world AABB median partitions', padding_m=.12)
    elif representation == 'point':
        config.update(algorithm='lexicographic unique measured endpoints, evenly spaced rank subsampling',
                      max_points=POINT_COUNT, missing_returns='unknown, mask false',
                      spatial_pool='4x4x4 learned PointNet max pools, with xyz features')
    else:
        config.update(algorithm='measured endpoint occupancy and sampled measured-return ray traversal',
            voxel_size_m=VOXEL_SIZE, cube_side_m=16., grid_axis_order='xyz',
            channels=['occupied', 'ray_free_evidence', 'unknown'],
            traversal_step_m=VOXEL_SIZE/2, endpoint_exclusion_m=np.sqrt(3)*VOXEL_SIZE,
            occupied_overrides_free=True, unknown_outside_sphere=True,
            no_return_rays='remain unknown, no max-range free ray',
            temporal_fusion=False, signed_distance=False,
            limitation='Sampling may miss grazing cells; free evidence does not certify voxel volume.')
    encoded = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    return dict(config, configuration_sha256=hashlib.sha256(encoded).hexdigest())


def _canonical_state(state, goal):
    origin = np.asarray(state['position'], float)
    velocity = np.asarray(state['velocity'], float)
    goal = np.asarray(goal, float)
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (origin, velocity, goal)):
        raise ValueError('current state and goal must be finite xyz vectors')
    delta = goal-origin
    yaw = np.arctan2(delta[1], delta[0]) if np.linalg.norm(delta[:2]) > 1e-8 else 0.
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c,s,0],[-s,c,0],[0,0,1.]])
    body = rotation @ quaternion_matrix(state['quaternion'])
    return origin, rotation, np.r_[rotation@delta, rotation@velocity, body[:,0], body[:,1]].astype(np.float32)


def _scan_points(frame, origin, rotation):
    points = np.asarray(frame.points, float)
    if points.size == 0:
        points = points.reshape(0, 3)
    if points.ndim != 2 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError('scan points must be finite [N,3] measured returns')
    pose = np.asarray(frame.world_T_sensor, float)
    if (pose.shape != (4,4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0,0,0,1], atol=1e-7)
            or not np.allclose(pose[:3,:3].T@pose[:3,:3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3,:3]), 1., atol=1e-5)):
        raise ValueError('world_T_sensor must be a rigid right-handed transform')
    if getattr(frame, 'point_timestamps', None) is not None:
        times = np.asarray(frame.point_timestamps)
        if times.shape != (len(points),) or not np.allclose(times, frame.sim_time, atol=1e-8, rtol=0):
            raise ValueError('snapshot point timestamps must match scan timestamp')
    sensor = (pose[:3,3]-origin) @ rotation.T
    measured = (points @ pose[:3,:3].T + pose[:3,3]-origin) @ rotation.T
    return measured, sensor


def _point_encoding(measured):
    roi = measured[np.linalg.norm(measured, axis=-1) <= ROI_RADIUS]
    # np.unique sorts lexicographically, so sensor ray ordering is not an ID.
    unique = np.unique(roi, axis=0)
    selected = unique if len(unique) <= POINT_COUNT else unique[np.linspace(0, len(unique)-1, POINT_COUNT).round().astype(int)]
    output = np.zeros((POINT_COUNT,3), np.float32)
    mask = np.zeros(POINT_COUNT, bool)
    output[:len(selected)] = selected
    mask[:len(selected)] = True
    return output, mask, dict(measured_returns=len(measured), roi_returns=len(roi),
                             unique_roi_points=len(unique), retained_points=len(selected))


def _indices(points):
    # Keep a sphere inside the grid cube; the +8 boundary cannot index cell64.
    keep = (np.linalg.norm(points, axis=-1) <= ROI_RADIUS) & np.all((points >= -ROI_RADIUS)&(points < ROI_RADIUS), axis=-1)
    return np.floor((points[keep]+ROI_RADIUS)/VOXEL_SIZE).astype(np.int64)


def _voxel_encoding(measured, sensor):
    occupied = np.zeros((GRID_SIZE,)*3, bool)
    free = np.zeros_like(occupied)
    hit = _indices(measured)
    if len(hit):
        occupied[tuple(hit.T)] = True
    offsets = measured-sensor
    distances = np.linalg.norm(offsets, axis=-1)
    good = distances > 1e-8
    offsets, distances = offsets[good], distances[good]
    directions = offsets / distances[:,None]
    # Bounded chunks avoid an R x 64^3 intermediate. A ray starts at its own
    # measured sensor origin. Stop before the return with a conservative margin.
    far = 2*ROI_RADIUS+np.linalg.norm(sensor)
    samples = np.arange(0., far+VOXEL_SIZE/2, VOXEL_SIZE/2)
    for begin in range(0, len(directions), 256):
        d, length = directions[begin:begin+256], distances[begin:begin+256]
        t = samples[None,:]
        valid = t < (length[:,None]-np.sqrt(3)*VOXEL_SIZE)
        coordinates = sensor + d[:,None,:]*t[...,None]
        traversed = _indices(coordinates[valid])
        if len(traversed):
            free[tuple(traversed.T)] = True
    free[occupied] = False
    known = occupied|free
    output = np.stack((occupied, free, ~known)).astype(np.uint8)
    return output, known, dict(measured_returns=len(measured), occupied_voxels=int(occupied.sum()),
                              free_evidence_voxels=int(free.sum()), unknown_voxels=int((~known).sum()))


def encode_canonical_frames(frames, origin, rotation, representation):
    """Shared offline/live measured-ray preprocessing in a supplied rigid frame."""
    if representation not in ('point', 'voxel'):
        raise ValueError('canonical geometry encoding requires point or voxel')
    origin, rotation = np.asarray(origin, float), np.asarray(rotation, float)
    if (origin.shape != (3,) or rotation.shape != (3,3) or not np.isfinite(origin).all()
            or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError('canonical origin and rotation must define a rigid frame')
    tokens, masks, statistics = [], [], []
    for frame in frames:
        measured, sensor = _scan_points(frame, origin, rotation)
        value, mask, stats = _point_encoding(measured) if representation == 'point' else _voxel_encoding(measured, sensor)
        tokens.append(value); masks.append(mask); statistics.append(stats)
    if len(tokens) != 4:
        raise ValueError('geometry history requires four scans')
    return np.stack(tokens), np.stack(masks), statistics


def encode_geometry_history(frames, current_state, goal_position, representation, *, query_time=None):
    if representation == 'box':
        from .temporal_observations import encode_history
        return encode_history(frames, current_state, goal_position, query_time=query_time)
    frames = list(frames)
    if len(frames) != 4:
        raise ValueError('geometry history requires four scans')
    now = float(frames[-1].sim_time if query_time is None else query_time)
    ages = np.asarray([float(f.sim_time) for f in frames])-now
    if not np.isfinite(now) or not np.allclose(ages, HISTORY_TIMES, atol=1e-5, rtol=0):
        raise ValueError('history timestamps must be query_time + [-.6,-.4,-.2,0]')
    origin, rotation, condition = _canonical_state(current_state, goal_position)
    tokens, mask, stats = encode_canonical_frames(frames, origin, rotation, representation)
    return dict(history_tokens=tokens, history_mask=mask, history_times=HISTORY_TIMES.copy(),
                condition=condition, origin_world=origin, world_to_canonical=rotation,
                metadata=dict(**preprocessing_configuration(representation), frame_statistics=stats, query_time=now))


def reencode_dataset(data, representation):
    """Replace only conditioning arrays using saved raw past scans; no labels read.

    The mapping accepts full dataset dictionaries for convenience but only reads
    the raw-ray/current-frame allowlist below. It never reads teacher or gt_*.
    Dense occupancy stays uint8 in host RAM; training transfers minibatches.
    """
    input_spec(representation)
    if representation == 'box':
        return dict(data)
    ranges = np.asarray(data['history_ray_ranges'])
    poses = np.asarray(data['history_sensor_poses'])
    directions = np.asarray(data['ray_directions'])
    origins, rotations = np.asarray(data['origin_world']), np.asarray(data['world_to_canonical'])
    if ranges.ndim != 3 or ranges.shape[1] != 4 or poses.shape != (len(ranges),4,4,4) or directions.shape != (4,ranges.shape[-1],3):
        raise ValueError('raw saved history has incompatible ray or pose shapes')
    if np.isinf(ranges).any() or np.any(ranges[np.isfinite(ranges)] < .1):
        raise ValueError('measured ranges must be >=0.1m; no returns must be NaN')
    if not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions, axis=-1), 1., atol=1e-5):
        raise ValueError('ray_directions must be finite unit vectors')
    specification = input_spec(representation)
    tokens = np.empty((len(ranges),*specification['tokens_shape']), dtype=specification['tokens_dtype'])
    masks = np.empty((len(ranges),*specification['mask_shape']), dtype=bool)
    for row in range(len(ranges)):
        frames = []
        for f in range(4):
            keep = np.isfinite(ranges[row,f])
            frames.append(SimpleNamespace(points=directions[f,keep]*ranges[row,f,keep,None],
                world_T_sensor=poses[row,f], sim_time=HISTORY_TIMES[f]+.6))
        tokens[row], masks[row], _ = encode_canonical_frames(frames, origins[row], rotations[row], representation)
    return dict(data, history_tokens=tokens, history_mask=masks,
                _input_representation=representation, _preprocessing=preprocessing_configuration(representation))
