import unittest
import numpy as np
from hnuter_dit.moving_pose import MovingReference,KNOT_TIMES

class MovingPoseTests(unittest.TestCase):
 def test_moving_boundary_and_derivatives(self):
  c=np.zeros(17);c[3:6]=[.3,-.1,.02];c[9]=1;c[13:16]=[.04,0,0];c[16]=.1
  controls=np.zeros((8,6));controls[:,:3]=KNOT_TIMES[:,None]*c[3:6]
  p=MovingReference(controls,c)
  np.testing.assert_allclose(p.sample(0)['pos'],0,atol=1e-10)
  np.testing.assert_allclose(p.sample(0)['vel'],c[3:6],atol=1e-10)
  np.testing.assert_allclose(p.sample(0)['acc'],c[13:16],atol=1e-10)
  np.testing.assert_allclose(p.sample(KNOT_TIMES)['pos'],controls[:,:3],atol=1e-10)
  for t in (.1,1.,2.,4.):
   np.testing.assert_allclose((p.sample(t+1e-5)['pos']-p.sample(t-1e-5)['pos'])/2e-5,p.sample(t)['vel'],atol=1e-6)
 def test_invalid(self):
  with self.assertRaises(ValueError):MovingReference(np.zeros((6,6)),np.zeros(17))

if __name__=='__main__':unittest.main()
