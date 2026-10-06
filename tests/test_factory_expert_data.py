import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from hnuter_dit.factory_expert_data import TimedPath
from hnuter_dit.factory_window_dataset import FactoryWindowDataset, INPUT_KEYS
from hnuter_dit.sdf_navigation_data import sha

class FactoryDataTests(unittest.TestCase):
    def test_spline_derivatives_and_continuous_corner(self):
        p=TimedPath([[0,0,2,1,0,0,0],[0,4,3,1,0,0,0],[4,4,2,1,0,0,0]],40)
        for t in (0,40):
            np.testing.assert_allclose(p.sample(t)['vel'],0,atol=1e-10)
            np.testing.assert_allclose(p.sample(t)['acc'],0,atol=1e-10)
        for t in (8,20,32):
            self.assertGreater(np.linalg.norm(p.sample(t)['vel']),.01)
            np.testing.assert_allclose((p.sample(t+.0001)['pos']-p.sample(t-.0001)['pos'])/.0002,p.sample(t)['vel'],atol=1e-6)
            np.testing.assert_allclose((p.sample(t+.0001)['vel']-p.sample(t-.0001)['vel'])/.0002,p.sample(t)['acc'],atol=1e-6)

    def test_bounds(self):
        for duration in (0,-1,float('nan')):
            with self.assertRaises(ValueError):TimedPath([[0,0,0,1,0,0,0],[1,0,0,1,0,0,0]],duration)
        p=TimedPath([[0,0,0,1,0,0,0],[1,0,0,1,0,0,0]],10)
        with self.assertRaises(ValueError):p.sample(11)

    def test_loader_blocks_geometry_and_cross_split_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'x.npz'
            fields=dict(fields=np.zeros((4,6,32,32,32),np.float16),ages=np.array([-.6,-.4,-.2,0]),
                current_velocity=np.zeros(3),current_angular_velocity=np.zeros(3),current_quaternion=np.array([1,0,0,0]),
                current_reference_acceleration=np.zeros(3),goal=np.ones(3),execution_delay_s=.1,
                future=np.zeros((81,19)),future_times=.1+np.arange(81)*.05,evaluator_boxes=np.ones((5,6)))
            np.savez_compressed(f,**fields)
            m=dict(schema='factory_causal_navigation_v1',split_group='same-mesh',split=0,records=[dict(file='x.npz',sha256=sha(f))])
            a=root/'a.json';a.write_text(json.dumps(m))
            data=FactoryWindowDataset([a]);self.assertEqual(len(data),1)
            self.assertEqual(set(data[0]['inputs']),set(INPUT_KEYS))
            self.assertNotIn('future',data[0]['inputs'])
            b=root/'b.json';m['split']=2;b.write_text(json.dumps(m))
            with self.assertRaises(ValueError):FactoryWindowDataset([a,b])
            f.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):data[0]

if __name__=='__main__':unittest.main()
