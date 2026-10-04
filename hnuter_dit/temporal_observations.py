"""Causal four-scan conditioning in one current ego/goal-yaw frame."""
from __future__ import annotations

import numpy as np

from .lidar_condition import frame_to_tokens

HISTORY_TIMES = np.array([-.6, -.4, -.2, 0.], dtype=np.float64)
FUTURE_TIMES = np.arange(65, dtype=np.float64) * .125
TEMPORAL_SCHEMA = 'hnuter_temporal_rays_v1'


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('expected finite nonzero wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def encode_history(frames, current_state, goal_position, *, query_time=None,
                   range_m=8., cell_padding=.12):
    """Encode exactly four past snapshots, without scene or future information.

    All returns first use their own measured sensor-to-world transform. World
    AABBs then share the latest measured body's origin and the current goal yaw.
    Token indices do not identify tracked objects. Missing rays remain unknown.
    """
    frames = list(frames)
    if len(frames) != 4:
        raise ValueError('temporal conditioning requires exactly four scans')
    query_time = float(frames[-1].sim_time if query_time is None else query_time)
    times = np.array([float(frame.sim_time) for frame in frames])
    if not np.isfinite(query_time) or not np.isfinite(times).all() or not np.allclose(
            times-query_time, HISTORY_TIMES, atol=1e-5, rtol=0):
        raise ValueError('history timestamps must be query_time + [-.6,-.4,-.2,0]')
    origin = np.asarray(current_state['position'], dtype=float)
    velocity = np.asarray(current_state['velocity'], dtype=float)
    goal = np.asarray(goal_position, dtype=float)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (origin, velocity, goal)):
        raise ValueError('position, velocity and goal must be finite xyz vectors')
    delta = goal-origin
    yaw = np.arctan2(delta[1], delta[0]) if np.linalg.norm(delta[:2]) > 1e-8 else 0.
    c, s = np.cos(yaw), np.sin(yaw)
    canonical = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
    rotation = canonical @ quaternion_matrix(current_state['quaternion'])
    tokens, masks, metadata = [], [], []
    for frame in frames:
        value, mask, info = frame_to_tokens(frame, origin, goal, range_m=range_m, cell_padding=cell_padding)
        value = value.copy()
        value[mask, :3] = (value[mask, :3]-origin) @ canonical.T
        # The world AABB's full lengths remain unchanged under this box rotation.
        value[mask, 6:] = [np.cos(yaw/2), 0., 0., -np.sin(yaw/2)]
        value[~mask, :6] = 0.
        value[~mask, 6:] = [1., 0., 0., 0.]
        tokens.append(value)
        masks.append(mask)
        metadata.append(info)
    return dict(history_tokens=np.stack(tokens).astype(np.float32),
                history_mask=np.stack(masks), history_times=HISTORY_TIMES.copy(),
                condition=np.r_[canonical @ delta, canonical @ velocity,
                                rotation[:, 0], rotation[:, 1]].astype(np.float32),
                origin_world=origin.copy(), world_to_canonical=canonical,
                metadata=dict(schema=TEMPORAL_SCHEMA, query_time=query_time,
                              frame_metadata=metadata, causal=True,
                              unknown_space='No-return rays and unobserved geometry are unknown; tokens cover measured returns only.'))
