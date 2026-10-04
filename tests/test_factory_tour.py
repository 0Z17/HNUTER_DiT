import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from hnuter_dit.factory_tour import local_state,world_command,load_mission
from hnuter_dit.sdf_online import SDFOnlinePlanner


def test_rigid_frame_roundtrip_preserves_pose_and_body_angular_derivatives():
    r=Rotation.from_euler('xyz',[.12,-.08,1.4]).as_matrix()
    frame=Rotation.from_euler('z',-1.7).as_matrix();origin=np.array([19.,7.,13.])
    state=dict(position=np.array([20.,9.,14.]),velocity=np.array([.3,-.2,.1]),rotation_matrix=r,
        quaternion=Rotation.from_matrix(r).as_quat()[[3,0,1,2]],angular_velocity=np.array([.1,.2,.3]))
    local=local_state(state,origin,frame)
    cmd=dict(pos=local['position'],vel=local['velocity'],acc=np.array([.2,.3,-.1])@frame,
        quaternion=local['quaternion'],angular_velocity=state['angular_velocity'],angular_acceleration=np.zeros(3))
    restored=world_command(cmd,origin,frame)
    assert np.allclose(restored['pos'],state['position'])
    assert np.allclose(restored['vel'],state['velocity'])
    assert np.allclose(restored['acc'],[.2,.3,-.1])
    assert np.allclose(Rotation.from_quat(restored['quaternion'][[1,2,3,0]]).as_matrix(),r)
    assert np.array_equal(restored['angular_velocity'],state['angular_velocity'])


def test_ending_leg_preserves_shared_mapper_process():
    class Runtime:
        closed=False
        def describe(self):return {}
        def close(self):self.closed=True
    runtime=Runtime()
    planner=SDFOnlinePlanner(goal_position=[4.4,0,0],checkpoint='unused',injected_runtime=runtime)
    planner.close()
    assert planner.closed and not runtime.closed


def test_mission_rejects_out_of_curriculum_goals(tmp_path):
    p=tmp_path/'mission.json'
    data=dict(schema='hnuter_factory_mission_v1',start_position=[0,0,13],waypoints=[[4.4,0,13]],goal_yaws=[.3])
    p.write_text(json.dumps(data));assert load_mission(p)==data
    data['waypoints']=[[20,0,13]];p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='curriculum'):load_mission(p)
    data['waypoints']=[[4.4,0,13]];data['goal_yaws']=[1.4];p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='yaw'):load_mission(p)


def test_elapsed_time_alone_does_not_trigger_inference_before_history_ready():
    from concurrent.futures import Future
    from types import SimpleNamespace
    from unittest.mock import patch
    class Runtime:
        def describe(self):return {}
        def scan(self,payload):return None
    planner=SDFOnlinePlanner(goal_position=[4.4,0,0],goal_rpy=[0,0,0],checkpoint='unused',injected_runtime=Runtime())
    state=dict(position=np.zeros(3),velocity=np.zeros(3),rotation_matrix=np.eye(3),quaternion=np.array([1.,0,0,0]))
    planner.scans=[dict(history_count=3,history_ready=False,map_seconds=.01,sensor_to_response_wall_seconds=.02)]
    frame=SimpleNamespace(sim_time=1.,points=np.zeros((1,3)),world_T_sensor=np.eye(4))
    future=Future();future.set_result((dict(controls=np.zeros((0,6))),dict(map_seconds=.01)))
    try:
        with patch.object(planner.pool,'submit',return_value=future) as submit:
            planner.observe(frame,state,1.)
            assert submit.call_args.args[1]['plan']==False
            assert planner.plan_attempts==0
    finally:planner.close()


def test_world_map_retention_reanchors_queries_without_relabeling_measurements():
    from unittest.mock import Mock
    from hnuter_dit.sdf_runtime import SDFRuntime
    runtime=SDFRuntime.__new__(SDFRuntime)
    runtime.map=Mock();runtime.world_origin=None;runtime.map_epoch_time=runtime.map_epoch_origin=None
    frame=Rotation.from_euler('z',1.2).as_matrix();origin=np.array([16.,7.,13.])
    runtime.reset(np.zeros(3),frame.T,[4.4,0,0,0,0,.3],world_origin=origin,world_frame=frame,global_start=0)
    assert runtime.map.reset.call_args.kwargs['clear_native']
    next_origin=origin+np.array([0,4.4,0])
    runtime.reset(np.zeros(3),frame.T,[4.4,0,0,0,0,.3],world_origin=next_origin,world_frame=frame,global_start=21.)
    args=runtime.map.reset.call_args
    assert not args.kwargs['clear_native']
    np.testing.assert_allclose(args.args[0],next_origin)
    np.testing.assert_allclose(args.args[1],frame)
    physical=np.eye(4);physical[:3,3]=next_origin+[.2,.1,.13]
    virtual=np.eye(4);virtual[:3,:3]=frame.T;virtual[:3,3]=(physical[:3,3]-next_origin)@frame
    restored,stamp=runtime.map_observation(virtual,.2)
    np.testing.assert_allclose(restored,physical,atol=1e-12)
    assert stamp==21.2
    np.testing.assert_allclose(runtime.to_world(np.array([[0,0,0],[1,0,0]])),np.array([[0,0,0],[1,0,0]])@frame.T+next_origin)
    with pytest.raises(ValueError,match='lifetime'):runtime.map_observation(virtual,100.)
    runtime.reset(np.zeros(3),frame.T,[4.4,0,0,0,0,.3],world_origin=next_origin,world_frame=frame,global_start=100.)
    assert runtime.map.reset.call_args.kwargs['clear_native']
    runtime.reset(np.zeros(3),np.eye(3),[4.4,0,0,0,0,.3])
    assert runtime.world_origin is None


def test_final_worker_failure_propagates_to_mission_report():
    from unittest.mock import Mock
    from hnuter_dit.factory_tour import FactorySDFPlanner
    planner=FactorySDFPlanner.__new__(FactorySDFPlanner)
    planner.closed=False;planner.active=Mock();planner.runtime=Mock()
    planner.active.close.return_value=dict(failed=True,failure_reason='final mapping request failed')
    planner.runtime.describe.return_value={}
    planner.legs=[];planner.failed=False;planner.failure_reason=None;planner.completed_time=20.
    planner.mission={};planner.events=[];planner.raw_paths=[];planner.scans=[];planner.guard_interventions=0
    result=planner.close()
    assert result['failed'] and result['failure_reason']=='final mapping request failed'


def test_recording_barrier_requires_new_current_state_map_and_preserves_path():
    from types import SimpleNamespace
    import time
    class Runtime:
        def describe(self):return {}
        def scan(self,payload):
            assert payload['stamp']==2.
            return dict(controls=np.zeros((0,6))),dict(map_seconds=.002)
    planner=SDFOnlinePlanner(goal_position=[4.4,0,0],goal_rpy=[0,0,0],checkpoint='unused',injected_runtime=Runtime())
    state=dict(position=np.zeros(3),velocity=np.zeros(3),rotation_matrix=np.eye(3),quaternion=np.array([1.,0,0,0]))
    frame=SimpleNamespace(sim_time=2.,points=np.zeros((1,3)),world_T_sensor=np.eye(4))
    marker=object();planner.path=marker;planner.last_map_time=1.9;planner.last_map_wall=time.perf_counter()-1
    planner.planned=True
    try:
        assert planner.recording_refresh_required()
        planner.wait_for_recorded_observation(frame,state,2.,time.perf_counter())
        assert planner.last_map_time==2. and planner.path is marker and planner.guard_interventions==0
        assert not planner.recording_refresh_required()
        planner.last_map_wall=time.perf_counter()-1
        with pytest.raises(RuntimeError,match='freshness deadline'):
            planner.wait_for_recorded_observation(frame,state,2.,time.perf_counter())
    finally:planner.close()
