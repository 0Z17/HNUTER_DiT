import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from hnuter_dit.sdf_pose import PoseSpline,full_controls,PROGRESS,HORIZON
from hnuter_dit.sdf_data import scenario


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.target=np.array([4.4,0,.1,.12,-.1,.3])
        free=PROGRESS[:,None]*self.target;free[:,1]=[.5,1.2,1.5,1.5,1.2,.5]
        self.path=PoseSpline(full_controls(free,self.target))

    def test_endpoints_and_derivatives(self):
        a,b=self.path.sample(0),self.path.sample(HORIZON)
        np.testing.assert_allclose(a['pos'],0,atol=1e-12)
        np.testing.assert_allclose(b['pos'],self.target[:3],atol=1e-12)
        for row in (a,b):
            for key in ('vel','acc','angular_velocity','angular_acceleration'):
                np.testing.assert_allclose(row[key],0,atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(b['quaternion'][[1,2,3,0]]).as_euler('xyz'),self.target[3:],atol=1e-10)

    def test_body_angular_velocity_and_acceleration(self):
        t=9.;dt=1e-4;a=self.path.sample(t-dt);b=self.path.sample(t);c=self.path.sample(t+dt)
        rotation=lambda row:Rotation.from_quat(row['quaternion'][[1,2,3,0]])
        numeric=(rotation(a).inv()*rotation(c)).as_rotvec()/(2*dt)
        np.testing.assert_allclose(numeric,b['angular_velocity'],atol=1e-7)
        np.testing.assert_allclose((c['angular_velocity']-a['angular_velocity'])/(2*dt),b['angular_acceleration'],atol=1e-8)

    def test_world_transform(self):
        r=Rotation.from_euler('z',.8).as_matrix();origin=np.array([-25,-35,2.3])
        path=PoseSpline(self.path.controls,origin,r,r)
        a,b=self.path.sample(7),path.sample(7)
        np.testing.assert_allclose(b['pos'],r@a['pos']+origin)
        np.testing.assert_allclose(b['vel'],r@a['vel'])
        np.testing.assert_allclose(b['angular_velocity'],a['angular_velocity'])

    def test_invalid_time(self):
        for t in (-.1,21.,float('nan')):
            with self.assertRaises(ValueError):self.path.sample(t)

    def test_derivative_bounds(self):
        audit=self.path.audit();points=self.path.sample(np.linspace(0,HORIZON,20001))
        self.assertLessEqual(np.linalg.norm(points['vel'],axis=1).max(),audit['speed_bound_mps']+1e-9)
        self.assertLessEqual(np.linalg.norm(points['acc'],axis=1).max(),audit['acceleration_bound_mps2']+1e-9)

    def test_split_and_obstacle_variation(self):
        self.assertEqual([scenario(i)['split'] for i in (0,63,64,79,80,95)],[0,0,1,1,2,2])
        self.assertLess(scenario(0)['boxes'][0][1],0);self.assertGreater(scenario(1)['boxes'][0][1],0)
        self.assertEqual(scenario(4),scenario(4))


if __name__=='__main__':unittest.main()
