from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from hnuter_dit.sdf_online import SDFOnlinePlanner,SDFProcess
from hnuter_dit.sdf_pose import full_controls,PROGRESS,PoseSpline


class DummyProcess:
    def __init__(self,**kwargs):pass
    def describe(self):return {}
    def close(self):pass


class OnlineTests(unittest.TestCase):
    def setUp(self):
        with patch('hnuter_dit.sdf_online.SDFProcess',DummyProcess):
            self.planner=SDFOnlinePlanner(goal_position=[4.4,0,2.3],checkpoint='unused')
        self.state=dict(position=np.array([0.,0,2.3]),velocity=np.zeros(3),rotation_matrix=np.eye(3),quaternion=np.array([1.,0,0,0]))
        p=self.planner;p.origin=self.state['position'];p.rotation=np.eye(3);p.target=np.array([4.4,0,0,.1,-.1,.2]);p.hold=dict(pos=p.origin,quaternion=self.state['quaternion'],vel=np.zeros(3),acc=np.zeros(3))
    def tearDown(self):self.planner.close()

    def test_stale_response_not_executed(self):
        import time
        p=self.planner;free=PROGRESS[:,None]*p.target
        p.future=Future();p.future.set_result(({'controls':full_controls(free,p.target)},dict(map_seconds=.003,feasible=True)))
        p.pending=dict(stamp=.6,plan=True,started=time.perf_counter())
        p._poll(self.state,1.0)
        self.assertIsNone(p.path);self.assertEqual(p.events[-1]['status'],'rejected')

    def test_worker_error_latched(self):
        import time
        p=self.planner;p.future=Future();p.future.set_exception(RuntimeError('mapper unavailable'))
        p.pending=dict(stamp=.6,plan=True,started=time.perf_counter())
        p._poll(self.state,.62)
        self.assertTrue(p.failed);self.assertIsNotNone(p.brake);self.assertIsNone(p.path)

    def test_stale_map_brakes(self):
        import time
        p=self.planner;p.path=PoseSpline(full_controls(PROGRESS[:,None]*p.target,p.target),p.origin)
        p.path_start=.6;p.last_map_time=.6;p.last_map_wall=time.perf_counter()
        p.command(self.state,1.1)
        self.assertIsNone(p.path);self.assertEqual(p.events[-1]['reason'],'stale_perception')

    def test_mismatched_scan_time_rejected(self):
        with self.assertRaises(ValueError):self.planner.observe(SimpleNamespace(sim_time=.4),self.state,.5)

    def test_worker_environment_uses_isolated_npp(self):
        with patch.dict('os.environ',{'PYTHONPATH':'/bad/torch','LD_PRELOAD':'bad','LD_LIBRARY_PATH':'/bad','NVBLOX_LIBRARY_PATH':'/npp'}):
            env=SDFProcess.worker_environment()
        self.assertNotIn('PYTHONPATH',env);self.assertNotIn('LD_PRELOAD',env);self.assertEqual(env['LD_LIBRARY_PATH'],'/npp')

    def test_unknown_only_candidate_not_accepted(self):
        import time
        p=self.planner;p.future=Future()
        p.future.set_result(({'controls':full_controls(PROGRESS[:,None]*p.target,p.target)},
            dict(map_seconds=.003,feasible=True,observed_clearance=dict(observed_violation=False,minimum_observed_support=False))))
        p.pending=dict(stamp=.6,plan=True,started=time.perf_counter())
        p._poll(self.state,.62)
        self.assertIsNone(p.path);self.assertEqual(p.events[-1]['status'],'rejected')

    def test_fresh_supported_candidate_accepted(self):
        import time
        p=self.planner;p.future=Future()
        p.future.set_result(({'controls':full_controls(PROGRESS[:,None]*p.target,p.target)},
            dict(map_seconds=.003,feasible=True,observed_clearance=dict(observed_violation=False,minimum_observed_support=True))))
        p.pending=dict(stamp=.6,plan=True,started=time.perf_counter())
        p._poll(self.state,.62)
        self.assertIsNotNone(p.path);self.assertEqual(p.events[-1]['status'],'accepted')


if __name__=='__main__':unittest.main()
