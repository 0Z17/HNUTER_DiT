"""Contact contract tested against actual native geometry and solver steps."""
import os
from pathlib import Path

import numpy as np
import pytest

from hnuter_dit.task_physics import TaskWorld, TOOL_POLE, TIP_RADIUS, DT


@pytest.fixture
def worlds():
    pytest.importorskip('mujoco')
    from hnuter_dit.temporal_data import DEFAULT_SIM_ROOT
    sim_root=Path(os.environ.get('HNUTER_SIM_ROOT',DEFAULT_SIM_ROOT))
    if not (sim_root/'robots/hnuter/source/hnuter.xml').is_file():
        pytest.skip('external HNUTER simulation assets unavailable')
    opened = []
    def make(**changes):
        world = TaskWorld(dict(dict(task='surface',origin=[0.,0.,2.8]),**changes),sim_root=sim_root)
        opened.append(world)
        return world
    yield make
    for world in opened:
        world.close()


def hold(position=(0.,0.,2.8)):
    return dict(pos=np.array(position),quaternion=np.array([1.,0,0,0]),
                vel=np.zeros(3),acc=np.zeros(3))


def test_explicit_tool_has_mass_fixed_attachment_and_real_sphere_pole(worlds):
    angle=.2
    world=worlds(initial_quaternion=[np.cos(angle/2),0.,np.sin(angle/2),0.])
    mj,model,data=world.mj,world.model,world.data
    tool=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'task_probe')
    site=mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,'probe_contact_pole')
    assert model.body_mass.sum()==pytest.approx(4.500024+.02,abs=1e-9)
    assert model.body_mass[tool]==pytest.approx(.02)
    assert model.body_jntnum[tool]==0
    assert model.body_parentid[tool]==world.controller._body_id
    assert np.all(model.body_inertia[tool]>0)
    assert model.geom_type[world.tip_id]==mj.mjtGeom.mjGEOM_SPHERE
    assert model.geom_size[world.tip_id,0]==pytest.approx(.025)
    state=world.state()
    actual_pole=state['position']+state['rotation_matrix']@TOOL_POLE
    np.testing.assert_allclose(data.site_xpos[site],actual_pole,atol=1e-12)
    assert np.linalg.norm(actual_pole-data.geom_xpos[world.tip_id])==pytest.approx(TIP_RADIUS)
    # Matching the ball centre to the paper's -.75 m offset would fail this.
    assert not np.allclose(data.geom_xpos[world.tip_id],actual_pole)


def test_original_gains_named_cross_tilt_and_motor_allocation_survive_tool(worlds):
    world=worlds()
    control=world.controller
    np.testing.assert_array_equal(control.position_gain,np.diag([15.,15.,30.]))
    np.testing.assert_array_equal(control.velocity_gain,np.diag([8.,8.,12.]))
    np.testing.assert_array_equal(control.attitude_gain,[6.,4.,4.])
    np.testing.assert_array_equal(control.angular_rate_gain,[1.2,.8,1.])
    assert control.mass==4.5
    assert world.model.opt.timestep==pytest.approx(.001)
    assert world.model.opt.integrator==world.mj.mjtIntegrator.mjINT_RK4
    assert world.model.opt.iterations==100
    assert world.model.opt.tolerance==pytest.approx(1e-10)
    control.set_actuators(12.,8.,.5,.1,-.2,.3,-.4)
    expected=dict(tilt_rj2=-.2,tilt_lj2=.1,tilt_rj1=-.4,tilt_lj1=.3,
                  motor_xy1=4.,motor_xy2=4.,motor_xy3=6.,motor_xy4=6.,motor_xy5=6.)
    assert world.model.nu==len(expected)
    for name,value in expected.items():
        index=world.mj.mj_name2id(world.model,world.mj.mjtObj.mjOBJ_ACTUATOR,name)
        assert world.data.ctrl[index]==pytest.approx(value)


def test_same_true_top_contact_is_allowed_only_in_designated_phase(worlds):
    world=worlds(origin=[0.,0.,2.5499])
    allowed,forbidden,normal,_=world._contacts('operate')
    assert allowed and not forbidden and normal>0.
    assert {name for row in allowed for name in row['pair']}=={'probe_tip','task_pad'}
    assert all(row['distance_m']<0 for row in allowed)
    # This is actual mj_forward/mj_contactForce output, not a mocked contact.
    raw=max(row['force_contact_frame'][0] for row in world.raw_cross_contacts)
    assert normal==pytest.approx(raw)
    allowed,forbidden,_,_=world._contacts('free')
    assert forbidden and not allowed
    step=world.step_task(hold([0.,0.,2.5499]),phase='free')
    assert step['forbidden_contact'] and world.first_forbidden is not None


@pytest.mark.parametrize('config,rejected_pair',[
    ({'origin':[-.8245,0.,2.475]}, {'probe_tip','task_pad'}),
    ({'origin':[0.,0.,2.48]}, {'probe_shaft','task_pad'}),
    ({'task':'inspection','origin':[0.,0.,2.5499],
      'boxes':[[0.,0.,1.79,.2,.2,.02]]}, {'probe_tip','task_obstacle_0'}),
])
def test_side_shaft_and_unlisted_surface_contacts_fail_closed(worlds,config,rejected_pair):
    world=worlds(**config)
    allowed,forbidden,_,_=world._contacts('operate')
    assert any(set(row['pair'])==rejected_pair for row in forbidden)
    assert all(set(row['pair'])!=rejected_pair for row in allowed)


def test_inspection_cannot_enable_force_and_flight_uses_real_integrator(worlds,monkeypatch):
    world=worlds(task='inspection')
    assert world.pad_id==-1
    initial_time=float(world.data.time)
    with pytest.raises(ValueError,match='surface task'):
        world.step_task(hold(),phase='operate',force_enabled=True)
    assert world.data.time==initial_time
    def no_pose_write(*args,**kwargs):
        raise AssertionError('runtime pose write is forbidden')
    monkeypatch.setattr(world.controller,'set_freejoint_pose',no_pose_write)
    original_step=world.mj.mj_step
    calls=[]
    def measured_step(model,data):
        calls.append(float(data.time))
        original_step(model,data)
    monkeypatch.setattr(world.mj,'mj_step',measured_step)
    desired=hold([.05,0.,2.8])
    first=world.step_task(desired)
    assert abs(first['position'][0]-.05)>.04  # no teleport to the desired pose
    for _ in range(499):
        world.step_task(desired)
    assert len(calls)==500
    assert world.data.time-initial_time==pytest.approx(500*DT,abs=1e-12)
    assert world.state()['position'][0]>.005
    assert world.forbidden_steps==0
    assert np.linalg.norm(world.data.ctrl)>1.


def test_force_offset_is_bounded_rate_limited_and_only_changes_reference(worlds):
    world=worlds()
    desired=hold()
    original_position=desired['pos'].copy()
    initial_qpos=world.data.qpos.copy()
    offsets=[]
    # Synthetic force is used solely to stress the outer-loop bounds. Physical
    # force correctness is covered by the real-contact test above.
    for normal,steps in ((0.,12000),(100.,4000)):
        world.last_measurement=dict(normal_force_n=normal)
        for _ in range(steps):
            command=world._command(desired,True)
            offsets.append(world.offset)
            assert command['pos'][2]==pytest.approx(desired['pos'][2]-world.offset)
    limit=world.force_cfg['maximum_offset_m']
    assert max(offsets)==pytest.approx(limit)
    assert min(offsets)==pytest.approx(-limit)
    assert np.max(np.abs(np.diff(np.r_[0.,offsets])))<=world.force_cfg['maximum_rate_mps']*DT+1e-12
    np.testing.assert_array_equal(world.data.qpos,initial_qpos)
    np.testing.assert_array_equal(desired['pos'],original_position)
    assert world.data.time==0.
