"""Registered, deliberately small Surface/Inspection extension of the SDF pilot.

Task geometry is supplied by an external task command. Environmental features
come from measured LiDAR histories. These are separate information sources.
This is not a reproduction of the manuscript's canonical v003 pose9 model.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .sdf_pose import PoseSpline, PROGRESS, full_controls, HORIZON

SCHEMA = 'hnuter_task_adaptive_pilot_v1'
OPERATE = (HORIZON/3, 2*HORIZON/3)
TOOL = np.array([0., 0., -.75])
CAMERA = np.array([.24, 0., .025])
SHAPE = np.array([.5, .9, 1., 1., .9, .5])


def scenario(index, seed=20261004):
    """12 layout groups x two tasks x two counterfactual task conditions = 48.

    A layout, including both conditions and both tasks, belongs to one split.
    No test layout participates in fitting or checkpoint selection.
    """
    if not 0 <= index < 48:
        raise ValueError('this registered pilot has exactly 48 episodes')
    layout = index//4
    task = ('surface', 'inspection')[(index % 4)//2]
    side = (-1, 1)[index % 2]
    rng = np.random.default_rng(np.random.SeedSequence([seed, layout]))
    length = float(rng.uniform(4.25, 4.55))
    amplitude = float(rng.uniform(.82, .9))
    wall_y = float(rng.uniform(5., 7.))
    boxes = [[7., 0., 2.5, .15, 16., 5.],
             [1., wall_y, 2.5, 16., .15, 5.],
             [1., -wall_y, 2.5, 16., .15, 5.],
             [1., 0., -.05, 16., 16., .1],
             [float(rng.uniform(-1., 5.)), -4., 2., .7, .7, 4.],
             [float(rng.uniform(-1., 5.)), 4., 2., .7, .7, 4.]]
    center = [length/2, 0., 1.75]
    # This is a forbidden work-surface patch, not an opaque physical obstacle.
    # Its sign is the counterfactual task command; maps/endpoints are identical
    # between the two Surface conditions. A plane projection cannot fix it.
    keepout = [-.5, .5, -1.5 if side > 0 else -.2, .2 if side > 0 else 1.5]
    return dict(schema=SCHEMA, index=index, layout=layout, seed=seed, task=task,
        split=0 if layout < 8 else (1 if layout < 10 else 2),
        origin=[0., 0., 2.8], goal=[length, 0., 2.8], goal_euler=[0.,0.,0.],
        pad_center=center, pad_half_extents=[3., 1.5, .05],
        keepout_in_pad_xy=keepout, keepout_margin_m=.04,
        inspection_target=[length/2, side*3., 3.] if task=='inspection' else None, inspection_range=[1.7, 2.5],
        inspection_angle_max_deg=10., target_radius=.05,
        expert_direction=side, expert_amplitude_m=amplitude, boxes=boxes,
        tool_mass_kg=.02, duration_s=HORIZON, operate_interval=list(OPERATE))


def target_condition(config, origin, rotation):
    """SDF translation coordinates are world-aligned; Euler is start-relative."""
    end_rotation = Rotation.from_euler('xyz', config.get('goal_euler',[0,0,0])).as_matrix()
    return np.r_[np.asarray(config['goal'])-origin,
                 Rotation.from_matrix(rotation.T@end_rotation).as_euler('xyz')]


def descriptor(config, origin, rotation):
    """24 task-command values, no episode ID, expert route or validity labels.

    Position coordinates use the prior's world-aligned translation frame;
    orientation targets use its start-relative orientation frame. Body-fixed
    sensor/tool offsets stay body-fixed, with explicit semantics below.
    """
    result = np.zeros(24, dtype=np.float32)
    if config['task'] == 'surface':
        point = np.array(config['pad_center'],float)
        point[2] += config['pad_half_extents'][2]
        result[:3] = point-origin
        result[3:6] = rotation.T@np.array([0.,0.,1.])
        result[6:9] = rotation.T@np.array([1.,0.,0.])
        result[9:12] = [*config['pad_half_extents'][:2], config['keepout_margin_m']]
        result[12:16] = config['keepout_in_pad_xy']
    else:
        result[:3] = np.asarray(config['inspection_target'])-origin
        result[9:12] = [*config['inspection_range'], config['target_radius']]
        result[16:18] = [np.deg2rad(config['inspection_angle_max_deg']), 5./HORIZON]
    result[18:21] = CAMERA
    result[21:24] = TOOL
    return result


def _local_angles(world_euler, rotation):
    matrices = Rotation.from_euler('xyz', world_euler).as_matrix()
    return Rotation.from_matrix(rotation.T[None]@matrices).as_euler('xyz')


def decode_geometry(free, config, origin, rotation):
    """No learned weights, route search, tangential repair or collision repair.

    Surface: place all free controls on the tool-contact plane and align the
    tool normal, retaining tangential controls and world yaw. Inspection: aim
    each control's FPV axis at the target, retaining all position controls.
    Smooth spline support provides approach/release. Neither decoder guarantees
    continuous-time task success, contact force, clearance or dynamics.
    """
    result = np.array(free, dtype=float, copy=True)
    if result.shape != (6,6) or not np.isfinite(result).all():
        raise ValueError('finite six free pose controls required')
    if config['task'] == 'surface':
        top = config['pad_center'][2]+config['pad_half_extents'][2]
        result[:,2] = top-TOOL[2]-origin[2]
        world_angles = Rotation.from_matrix(rotation[None]@Rotation.from_euler('xyz',result[:,3:]).as_matrix()).as_euler('xyz')
        world_angles[:,:2] = 0.
    else:
        points = result[:,:3]+origin
        delta = np.asarray(config['inspection_target'])-points
        # The small camera +Z offset is included; +X offset is along the axis.
        yaw = np.unwrap(np.arctan2(delta[:,1], delta[:,0]))
        horizontal = np.linalg.norm(delta[:,:2],axis=1)
        pitch = -np.arctan2(delta[:,2]-CAMERA[2], horizontal)
        world_angles = np.column_stack((np.zeros(6),pitch,yaw))
    result[:,3:] = _local_angles(world_angles,rotation)
    return result


def expert(config, origin, rotation):
    target = target_condition(config,origin,rotation)
    free = PROGRESS[:,None]*target
    free[:,1] += config['expert_direction']*config['expert_amplitude_m']*SHAPE
    free = decode_geometry(free,config,origin,rotation)
    return PoseSpline(full_controls(free,target),origin=origin,start_rotation=rotation)


def surface_geometry(positions, quaternions, config):
    rotation = Rotation.from_quat(np.asarray(quaternions)[...,[1,2,3,0]]).as_matrix()
    tip = np.asarray(positions)+np.einsum('...ij,j->...i',rotation,TOOL)
    local = tip[:,:2]-np.array(config['pad_center'])[:2]
    margin = config['keepout_margin_m']
    patch = np.array(config['keepout_in_pad_xy'])
    forbidden = ((local[:,0] >= patch[0]-margin)&(local[:,0] <= patch[1]+margin)&
                 (local[:,1] >= patch[2]-margin)&(local[:,1] <= patch[3]+margin))
    footprint = np.all(np.abs(local) <= np.array(config['pad_half_extents'])[:2]-.025,axis=1)
    top = config['pad_center'][2]+config['pad_half_extents'][2]
    error = np.abs(tip[:,2]-top)
    angle = np.arccos(np.clip(rotation[:,2,2],-1.,1.))
    return dict(plane_error_max_m=float(error.max()), plane_error_mean_m=float(error.mean()),
        normal_error_max_deg=float(np.rad2deg(angle).max()),
        footprint_fraction=float(footprint.mean()), forbidden_patch_fraction=float(forbidden.mean()),
        passed=bool(error.max()<=.005 and np.rad2deg(angle).max()<=2. and footprint.all() and not forbidden.any()))


def reference_metrics(path, config):
    pose = path.sample(np.linspace(*OPERATE,667))
    if config['task']=='surface':
        return surface_geometry(pose['pos'],pose['quaternion'],config)
    rot = Rotation.from_quat(pose['quaternion'][:,[1,2,3,0]]).as_matrix()
    camera = pose['pos']+np.einsum('tij,j->ti',rot,CAMERA)
    delta = np.asarray(config['inspection_target'])-camera
    distance = np.linalg.norm(delta,axis=1)
    angle = np.rad2deg(np.arccos(np.clip(np.einsum('ti,ti->t',delta,rot[:,:,0])/distance,-1,1)))
    good = ((angle<=config['inspection_angle_max_deg']) & (distance>=config['inspection_range'][0]) &
            (distance<=config['inspection_range'][1]))
    return dict(axis_error_max_deg=float(angle.max()),range_m=[float(distance.min()),float(distance.max())],
        ideal_view_fraction=float(good.mean()), passed=bool(good.mean()>=.9),
        scope='reference geometry only; actual visibility and dwell tested by raycast during execution')
