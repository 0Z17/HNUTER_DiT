"""Geometry and task-command contracts for the small registered pilot.

These tests examine reference geometry, not measured maps, actual contact
forces, visibility dwell, or learned-model performance.
"""
import copy
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

from hnuter_dit.task_spec import (scenario, descriptor, target_condition, decode_geometry,
    expert, reference_metrics, surface_geometry, OPERATE, TOOL, CAMERA)
from hnuter_dit.sdf_pose import PoseSpline, PROGRESS, KNOTS, full_controls, HORIZON


def test_layout_disjoint_splits_and_surface_counterfactual_same_physical_map():
    all_configs=[scenario(index) for index in range(48)]
    assert [sum(c['split']==s for c in all_configs) for s in range(3)] == [32,8,8]
    physical_keys=['origin','goal','goal_euler','boxes','pad_center','pad_half_extents','tool_mass_kg']
    for layout in range(12):
        group=all_configs[4*layout:4*layout+4]
        assert len({c['split'] for c in group}) == 1
        left,right=group[:2]
        assert left['task']==right['task']=='surface'
        assert all(left[key]==right[key] for key in physical_keys)
        assert left['keepout_in_pad_xy'] != right['keepout_in_pad_xy']
        dl=descriptor(left,np.array(left['origin']),np.eye(3))
        dr=descriptor(right,np.array(right['origin']),np.eye(3))
        assert set(np.flatnonzero(dl!=dr)) == {14,15}
        np.testing.assert_array_equal(target_condition(left,np.array(left['origin']),np.eye(3)),
                                      target_condition(right,np.array(right['origin']),np.eye(3)))
    with pytest.raises(ValueError):
        scenario(48)


@pytest.mark.parametrize('index',[0,1,2,3])
def test_descriptor_coordinate_conventions_and_no_expert_or_id_leak(index):
    config=scenario(index)
    origin=np.array([.17,-.23,2.81])
    rotation=Rotation.from_euler('xyz',[.2,-.13,.37]).as_matrix()
    token=descriptor(config,origin,rotation)
    assert token.shape==(24,) and token.dtype==np.float32 and np.isfinite(token).all()
    if config['task']=='surface':
        surface=np.array(config['pad_center']);surface[2]+=config['pad_half_extents'][2]
        np.testing.assert_allclose(token[:3],surface-origin,atol=1e-7)
        np.testing.assert_allclose(rotation@token[3:6],[0,0,1],atol=1e-7)
        np.testing.assert_allclose(rotation@token[6:9],[1,0,0],atol=1e-7)
    else:
        np.testing.assert_allclose(token[:3],np.array(config['inspection_target'])-origin,atol=1e-7)
    np.testing.assert_allclose(token[18:21],CAMERA,atol=1e-8)
    np.testing.assert_array_equal(token[21:24],TOOL)
    unrelated=copy.deepcopy(config)
    unrelated.update(index=999,layout=999,split=2,expert_direction=17,expert_amplitude_m=99.)
    np.testing.assert_array_equal(descriptor(unrelated,origin,rotation),token)
    shifted=copy.deepcopy(config);delta=np.array([7.,-4.,1.])
    shifted['goal']=(np.array(config['goal'])+delta).tolist()
    shifted['pad_center']=(np.array(config['pad_center'])+delta).tolist()
    if config['inspection_target'] is not None:
        shifted['inspection_target']=(np.array(config['inspection_target'])+delta).tolist()
    np.testing.assert_allclose(descriptor(shifted,origin+delta,rotation),token,atol=1e-7)


def test_target_translation_is_world_aligned_and_orientation_is_start_relative():
    config=scenario(0);config['goal_euler']=[-.1,.17,-.4]
    origin=np.array([.2,-.3,2.83])
    rotation=Rotation.from_euler('xyz',[.21,-.11,.35]).as_matrix()
    target=target_condition(config,origin,rotation)
    np.testing.assert_allclose(target[:3],np.array(config['goal'])-origin,atol=1e-12)
    np.testing.assert_allclose(rotation@Rotation.from_euler('xyz',target[3:]).as_matrix(),
        Rotation.from_euler('xyz',config['goal_euler']).as_matrix(),atol=1e-12)


@pytest.mark.parametrize('task_index',[0,2])
def test_decoder_preserves_body_positions_it_does_not_claim_to_repair(task_index):
    config=scenario(task_index);origin=np.array(config['origin'])
    rotation=Rotation.from_euler('xyz',[.25,-.19,.31]).as_matrix()
    free=np.random.default_rng(13).normal(size=(6,6))*.2
    free[:,0]=PROGRESS*config['goal'][0]
    original=free.copy();decoded=decode_geometry(free,config,origin,rotation)
    np.testing.assert_array_equal(free,original)
    target=target_condition(config,origin,rotation)
    before=PoseSpline(full_controls(free,target),origin=origin,start_rotation=rotation)
    after=PoseSpline(full_controls(decoded,target),origin=origin,start_rotation=rotation)
    times=np.linspace(0,HORIZON,451)
    channels=slice(0,2) if config['task']=='surface' else slice(0,3)
    np.testing.assert_array_equal(decoded[:,channels],free[:,channels])
    np.testing.assert_array_equal(before.sample(times)['pos'][:,channels],after.sample(times)['pos'][:,channels])
    if config['task']=='surface':
        before_rotation=rotation[None]@Rotation.from_euler('xyz',free[:,3:]).as_matrix()
        after_rotation=rotation[None]@Rotation.from_euler('xyz',decoded[:,3:]).as_matrix()
        old_yaw=Rotation.from_matrix(before_rotation).as_euler('xyz')[:,2]
        new_angles=Rotation.from_matrix(after_rotation).as_euler('xyz')
        np.testing.assert_allclose(new_angles[:,:2],0,atol=1e-12)
        np.testing.assert_allclose(np.sin(new_angles[:,2]-old_yaw),0,atol=1e-12)
        tip=decoded[:,:3]+origin+np.einsum('nij,j->ni',after_rotation,TOOL)
        np.testing.assert_allclose(tip[:,2],config['pad_center'][2]+config['pad_half_extents'][2],atol=1e-12)


def test_operate_interval_has_exactly_free_control_support_and_is_planar():
    assert OPERATE==(20/3,40/3)
    assert KNOTS[6]==OPERATE[0] and KNOTS[9]==OPERATE[1]
    times=np.linspace(*OPERATE,1001)
    basis=BSpline(KNOTS,np.eye(12),3)(times)
    np.testing.assert_array_equal(basis[:,[0,1,2,9,10,11]],0.)
    np.testing.assert_allclose(basis[:,3:9].sum(axis=-1),1,atol=5e-16)
    config=scenario(0);origin=np.array(config['origin']);rotation=np.eye(3)
    free=np.random.default_rng(23).normal(size=(6,6))*.2
    free[:,0]=PROGRESS*config['goal'][0];free[:,1]=-.85
    decoded=decode_geometry(free,config,origin,rotation)
    path=PoseSpline(full_controls(decoded,target_condition(config,origin,rotation)),origin=origin,start_rotation=rotation)
    metrics=reference_metrics(path,config)
    assert metrics['plane_error_max_m']<1e-12 and metrics['normal_error_max_deg']<1e-10
    # Clamped endpoints remain at the original elevation, with approach/release
    # outside OPERATE. The decoder does not teleport the whole path onto a plane.
    endpoints=path.sample([0,HORIZON])['pos']
    np.testing.assert_allclose(endpoints,[config['origin'],config['goal']],atol=1e-12)


def test_registered_expert_geometry_and_counterfactual_keepout():
    for index in range(48):
        config=scenario(index);origin=np.array(config['origin'])
        path=expert(config,origin,np.eye(3));metrics=reference_metrics(path,config)
        assert metrics['passed'], (index,metrics)
        assert path.audit()['speed_bound_mps']<.41
        assert path.audit()['acceleration_bound_mps2']<.14
        if config['task']=='surface':
            opposite=scenario(index^1)
            wrong=reference_metrics(path,opposite)
            assert wrong['forbidden_patch_fraction']>0 and not wrong['passed']
            # Plane/normal projection of a straight body path cannot choose the
            # required tangential bypass around either task's forbidden patch.
            straight=PROGRESS[:,None]*target_condition(config,origin,np.eye(3))
            projected=decode_geometry(straight,config,origin,np.eye(3))
            baseline=PoseSpline(full_controls(projected,target_condition(config,origin,np.eye(3))),origin=origin)
            rejected=reference_metrics(baseline,config)
            assert rejected['forbidden_patch_fraction']>0 and not rejected['passed']


def test_control_projection_does_not_guarantee_planarity_for_arbitrary_tilted_starts():
    # Document the existing decoder's stated limitation: interpolation of local
    # Euler coordinates need not preserve a world-horizontal normal between
    # individually projected controls when start roll/pitch is nonzero.
    config=scenario(0);origin=np.array(config['origin'])
    rotation=Rotation.from_euler('xyz',[.3,-.25,.4]).as_matrix()
    free=np.zeros((6,6));free[:,0]=PROGRESS*config['goal'][0];free[:,1]=-.85
    free[:,5]=np.linspace(-1.5,1.5,6)
    decoded=decode_geometry(free,config,origin,rotation)
    path=PoseSpline(full_controls(decoded,target_condition(config,origin,rotation)),origin=origin,start_rotation=rotation)
    metrics=reference_metrics(path,config)
    assert metrics['plane_error_max_m']>1e-5
    assert metrics['normal_error_max_deg']>.1


@pytest.mark.parametrize('index',[0,2])
def test_raw_training_geometry_uses_unprojected_pose_and_has_gradients(index,monkeypatch):
    import torch
    import hnuter_dit.task_spec as specification
    from hnuter_dit.task_training import raw_task_loss
    config=scenario(index);origin=np.array(config['origin']);rotation=np.eye(3)
    path=expert(config,origin,rotation)
    target=torch.tensor(target_condition(config,origin,rotation)[None],dtype=torch.float64)
    token=torch.tensor(descriptor(config,origin,rotation)[None],dtype=torch.float64)
    rot=torch.tensor(rotation[None],dtype=torch.float64)
    times=np.linspace(*OPERATE,33)
    basis=torch.tensor(BSpline(KNOTS,np.eye(12),3)(times),dtype=torch.float64)
    free=path.controls[3:9].copy()
    clean=torch.tensor(free[None],dtype=torch.float64)
    clean_loss=raw_task_loss(clean,target,token,rot,basis,config['task'],torch)
    # Any accidental training-time projection would hide these raw violations.
    if config['task']=='surface':
        free[:,2]+=.1
    else:
        free[:,5]+=.5
    changed=torch.tensor(free[None],dtype=torch.float64,requires_grad=True)
    def forbidden_decoder(*args,**kwargs):
        raise AssertionError('training raw geometry must not call the decoder')
    monkeypatch.setattr(specification,'decode_geometry',forbidden_decoder)
    raw_loss=raw_task_loss(changed,target,token,rot,basis,config['task'],torch)
    assert raw_loss>clean_loss+.001
    raw_loss.backward()
    assert changed.grad is not None and torch.isfinite(changed.grad).all()
    assert changed.grad.abs().sum()>0
