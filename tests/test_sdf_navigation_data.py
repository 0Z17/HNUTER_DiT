import unittest
import numpy as np
from hnuter_dit.sdf_navigation_data import ContinuousReference, scenario, expert, validate_history


class NavigationCollectionTests(unittest.TestCase):
    def test_endpoints_and_moving_windows(self):
        p=ContinuousReference([0,0,3],[6,0,.9,0,0,.1], vertical=.8)
        np.testing.assert_allclose(p.sample(0)['pos'],[0,0,3])
        np.testing.assert_allclose(p.sample(30)['pos'],[6,0,3.9],atol=1e-10)
        for t in (0,30):
            np.testing.assert_allclose(p.sample(t)['vel'],0,atol=1e-10)
            np.testing.assert_allclose(p.sample(t)['acc'],0,atol=1e-10)
        for t in (6,12,18,24):
            self.assertGreater(np.linalg.norm(p.sample(t)['vel']),.1)
            np.testing.assert_allclose((p.sample(t+1e-4)['pos']-p.sample(t-1e-4)['pos'])/2e-4,p.sample(t)['vel'],atol=1e-7)

    def test_straight_option_is_shortest_in_open_space(self):
        c=scenario(0,20261006)
        p,m=expert(c,np.array(c['origin']))
        self.assertEqual(m['lateral_m'],0)
        self.assertEqual(m['vertical_m'],0)
        self.assertAlmostEqual(m['length_m'],np.linalg.norm(c['target'][:3]),places=6)

    def test_required_vertical_routes(self):
        for i,sign in ((3,1),(4,-1)):
            c=scenario(i,20261006)
            _,m=expert(c,np.array(c['origin']))
            self.assertGreater(sign*m['vertical_m'],0)
            self.assertGreaterEqual(m['clearance_m'],.22)

    def test_seed_and_split_are_reproducible(self):
        self.assertEqual(scenario(4,9),scenario(4,9))
        self.assertNotEqual(scenario(4,9)['target'],scenario(4,10)['target'])
        self.assertEqual(len({scenario(i,9)['kind'] for i in range(6)}),6)

    def test_causal_history_rejects_future_stale_and_duplicate(self):
        validate_history([1.4,1.6,1.8,2.],2.)
        for times, stamp in (([1.4,1.6,1.8,2.],1.9),([1.4,1.6,1.8,2.],2.2),
                             ([1.4,1.6,1.6,2.],2.),([0.,1.6,1.8,2.],2.),
                             ([1.4,1.6,float('nan'),2.],2.)):
            with self.assertRaises(ValueError): validate_history(times,stamp)

    def test_invalid_time_rejected(self):
        p=ContinuousReference([0,0,3],[6,0,0,0,0,0])
        for t in (-1,31,float('nan')):
            with self.assertRaises(ValueError):p.sample(t)


if __name__=='__main__': unittest.main()
