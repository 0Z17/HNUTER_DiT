"""Measured single-target FPV geometry for the task-adaptation pilot.

This is synthetic line-of-sight acceptance from the current MuJoCo state,
not image recognition, dense surface coverage, or a learned perception score.
The ray intersects actual scene geometry and excludes the aircraft's group 5.
"""
from __future__ import annotations

import math
import numpy as np


FPV_POSITION_BODY = np.array([.24, 0., .025])
HFOV_DEG = 90.
VFOV_DEG = 58.7


def _thresholds(config):
    interval = np.asarray(config.get('inspection_range', [1.7, 2.5]), dtype=float)
    if (interval.shape != (2,) or not np.isfinite(interval).all() or
            not 0 < interval[0] < interval[1]):
        raise ValueError('inspection_range must be a finite increasing positive pair')
    # These gates are fixed before test execution; config may record stricter or
    # deliberately different experiments, whose thresholds remain in the report.
    angle = float(config.get('inspection_angle_max_deg', 10.))
    fraction = float(config.get('inspection_effective_fraction_min', .9))
    dwell = float(config.get('inspection_continuous_dwell_min_s', 5.))
    period = float(config.get('inspection_record_period_s', .01))
    if not (np.isfinite([angle, fraction, dwell, period]).all() and
            0 < angle <= 180 and 0 < fraction <= 1 and dwell > 0 and period > 0):
        raise ValueError('invalid inspection acceptance thresholds')
    return dict(range_m=interval.tolist(), angle_max_deg=angle,
                effective_fraction_min=fraction, continuous_dwell_min_s=dwell,
                record_period_s=period, maximum_record_gap_s=1.5*period,
                horizontal_fov_deg=HFOV_DEG, vertical_fov_deg=VFOV_DEG)


def inspection_measurement(world, config):
    """Measure the target centre with the real body-attached FPV extrinsic.

    The FPV optical axis is body +X, image right is -Y, and image up is +Z.
    Range is to the target centre; ray distance is to its first surface hit.
    Occluders are tested even when the target lies outside the field of view.
    """
    gates = _thresholds(config)
    mj, model, data = world.mj, world.model, world.data
    target_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'inspection_target'))
    if target_id < 0:
        raise ValueError('inspection requires the physical inspection_target geom')
    robot_geoms = np.asarray(world.robot_geoms, dtype=bool)
    if robot_geoms.shape != (model.ngeom,) or not np.all(model.geom_group[robot_geoms] == 5):
        raise ValueError('all aircraft geoms must use ray-excluded group 5')
    if int(model.geom_group[target_id]) == 5:
        raise ValueError('inspection target must not use excluded robot group 5')
    state = world.state()
    rotation = np.asarray(state['rotation_matrix'], dtype=float)
    position = np.asarray(state['position'], dtype=float)
    if (rotation.shape != (3, 3) or position.shape != (3,) or
            not np.isfinite(rotation).all() or not np.isfinite(position).all() or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)):
        raise ValueError('inspection requires a finite measured rigid body pose')
    camera = position + rotation @ FPV_POSITION_BODY
    optical_axis = rotation[:, 0]
    target = np.asarray(data.geom_xpos[target_id], dtype=float).copy()
    delta = target-camera
    distance = float(np.linalg.norm(delta))
    local = rotation.T @ delta
    hit = np.array([-1], dtype=np.int32)
    if distance > 1e-9 and np.isfinite(distance):
        direction = delta/distance
        angle = math.degrees(math.acos(float(np.clip(optical_axis @ direction, -1., 1.))))
        horizontal = math.degrees(math.atan2(-float(local[1]), float(local[0])))
        vertical = math.degrees(math.atan2(float(local[2]), float(local[0])))
        groups = np.ones(6, dtype=np.uint8)
        groups[5] = 0
        ray_distance = float(mj.mj_ray(model, data, camera.astype(np.float64),
                                     direction.astype(np.float64), groups, True, -1, hit))
    else:
        angle, horizontal, vertical, ray_distance = 180., 180., 180., -1.
    in_frustum = bool(local[0] > 0 and abs(horizontal) <= HFOV_DEG/2 and
                      abs(vertical) <= VFOV_DEG/2)
    line_of_sight = bool(hit[0] == target_id and ray_distance >= 0.)
    range_ok = bool(gates['range_m'][0] <= distance <= gates['range_m'][1])
    angle_ok = bool(angle <= gates['angle_max_deg'])
    first_name = (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(hit[0]))
                  if hit[0] >= 0 else None)
    return dict(camera_position=camera.tolist(), optical_axis=optical_axis.tolist(),
                camera_position_body=FPV_POSITION_BODY.tolist(), target_position=target.tolist(),
                range_m=distance, axis_angle_deg=angle,
                horizontal_angle_deg=horizontal, vertical_angle_deg=vertical,
                horizontal_fov_deg=HFOV_DEG, vertical_fov_deg=VFOV_DEG,
                in_frustum=in_frustum, line_of_sight=line_of_sight,
                visible=bool(in_frustum and line_of_sight), range_ok=range_ok, angle_ok=angle_ok,
                effective=bool(in_frustum and line_of_sight and range_ok and angle_ok),
                first_hit_geom=first_name, first_hit_geom_id=int(hit[0]),
                target_geom_id=target_id, ray_distance_m=ray_distance,
                source='MuJoCo mj_ray first hit from measured body-attached FPV',
                coverage_scope='single target centre; no image recognition or surface-area coverage')


def summarize_inspection(trace, config, operate_interval=(20/3, 40/3)):
    """Time-weighted acceptance over the entire declared operating interval.

    Records are {t, measurement}, with measurement from inspection_measurement.
    A measurement has at most one nominal recording period of validity. Missing
    trace time counts as failure rather than silently shrinking the denominator.
    No force, collision, or flight-tracking predicate is implied by this result.
    """
    gates = _thresholds(config)
    begin, end = map(float, operate_interval)
    if not np.isfinite([begin, end]).all() or end <= begin:
        raise ValueError('operate_interval must be finite and increasing')
    records = list(trace)
    times = np.asarray([item['t'] for item in records], dtype=float)
    if not np.isfinite(times).all() or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError('inspection timestamps must be finite and strictly increasing')
    totals = dict(measured=0., visible=0., angle=0., range=0., angle_range=0., effective=0.)
    longest = streak = 0.
    previous_right = None
    angles, ranges = [], []
    accepted_count = 0
    for index, item in enumerate(records):
        measurement = item['measurement']
        timestamp = float(times[index])
        right = timestamp+gates['record_period_s']
        if index+1 < len(times):
            right = min(right, float(times[index+1]))
        left, right = max(begin, timestamp), min(end, right)
        if right <= left:
            continue
        weight = right-left
        angle = float(measurement['axis_angle_deg'])
        distance = float(measurement['range_m'])
        finite = bool(np.isfinite([angle, distance]).all())
        visible = bool(measurement['in_frustum'] and measurement['line_of_sight'])
        angle_ok = bool(finite and 0 <= angle <= gates['angle_max_deg'])
        range_ok = bool(finite and gates['range_m'][0] <= distance <= gates['range_m'][1])
        effective = visible and angle_ok and range_ok
        for key, active in dict(measured=True, visible=visible, angle=angle_ok, range=range_ok,
                                angle_range=angle_ok and range_ok, effective=effective).items():
            totals[key] += weight*active
        if effective:
            if previous_right is None or left-previous_right > 1e-8:
                streak = 0.
            streak += weight
            longest = max(longest, streak)
        else:
            streak = 0.
        previous_right = right
        accepted_count += 1
        if finite:
            angles.append(angle)
            ranges.append(distance)
    duration = end-begin
    fractions = {key+'_fraction': value/duration for key, value in totals.items()}
    passed = bool(fractions['effective_fraction']+1e-10 >= gates['effective_fraction_min'] and
                  longest+1e-10 >= gates['continuous_dwell_min_s'])
    return dict(schema='hnuter_single_target_inspection_v1', operate_interval=[begin, end],
                operation_duration_s=duration, measured_records=accepted_count,
                **fractions, effective_duration_s=totals['effective'],
                longest_continuous_effective_s=longest,
                axis_angle_quantiles_deg=np.quantile(angles, [0, .5, .95, 1]).tolist() if angles else [],
                range_quantiles_m=np.quantile(ranges, [0, .5, .95, 1]).tolist() if ranges else [],
                acceptance=gates, passed=passed,
                coverage_scope='single target centre visibility; not image recognition or surface coverage',
                execution_scope='collision, trajectory tracking and saturation are checked externally',
                missing_observation_policy='unrecorded operation time counts as ineffective')
