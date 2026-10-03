import math
import unittest

import numpy as np

from hnuter_dit.path_validation import ROBOT_SWEEP_RADIUS, dense_poses, observed_clearance


def pose(position=(0, 0, 0), yaw=0):
    return np.r_[position, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def box(center, extent):
    return np.r_[center, extent, 1., 0., 0., 0.]


class PathValidationTests(unittest.TestCase):
    def test_thin_wall_between_clear_waypoints(self):
        path = np.array([pose((-2, 0, 0)), pose((2, 0, 0))])
        obstacle = np.array([box((0, 0, 0), (.01, 3, 3))])
        endpoint_distances = observed_clearance(path[:, None, :], obstacle)
        self.assertTrue((endpoint_distances > .5).all())
        self.assertLess(observed_clearance(path, obstacle)[0], -.1)

    def test_rotating_appendage_hits_between_clear_endpoints(self):
        path = np.array([pose(yaw=-math.pi / 2), pose(yaw=math.pi / 2)])
        obstacle = np.array([box((-.868, 0, .0034), (.025, .025, .025))])
        self.assertTrue((observed_clearance(path[:, None, :], obstacle) > .1).all())
        self.assertLess(observed_clearance(path, obstacle)[0], -.1)
        self.assertGreater(len(dense_poses(path)), 50)

    def test_translation_equivariance(self):
        paths = np.array([[pose((-2, 0, 0)), pose((2, 0, 0))],
                          [pose((-2, 3, 0), .3), pose((2, 3, 0), -.4)]])
        obstacles = np.array([box((0, 0, 0), (.1, 1, 1)), box((1, 2, 0), (.3, .2, .8))])
        expected = observed_clearance(paths, obstacles)
        offset = np.array([-25, -35, 2.2])
        moved_paths, moved_boxes = paths.copy(), obstacles.copy()
        moved_paths[..., :3] += offset
        moved_boxes[:, :3] += offset
        np.testing.assert_allclose(observed_clearance(moved_paths, moved_boxes), expected, atol=1e-12)

    def test_dense_spacing_bound_and_endpoints(self):
        path = np.array([pose((0, 0, 0), -.6), pose((.7, .2, .1), 1.7)])
        dense = dense_poses(path, .037)
        translation = np.linalg.norm(np.diff(dense[:, :3], axis=0), axis=1)
        dots = np.abs((dense[:-1, 3:] * dense[1:, 3:]).sum(axis=1))
        angle = 2 * np.arccos(np.clip(dots, 0, 1))
        self.assertTrue((translation + ROBOT_SWEEP_RADIUS * angle <= .037 + 1e-12).all())
        np.testing.assert_allclose(dense[[0, -1]], path, atol=2e-16, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(dense[:, 3:], axis=1), 1)

    def test_shortest_arc_and_antipodal_equivalence(self):
        path = np.array([pose(yaw=math.radians(179)), pose(yaw=math.radians(-179))])
        self.assertEqual(len(dense_poses(path)), 2)
        identical = np.array([pose(), pose()])
        identical[1, 3:] *= -1
        np.testing.assert_array_equal(dense_poses(identical), [pose(), pose()])

    def test_empty_boxes_are_finite_unknown_sentinel(self):
        result = observed_clearance(np.array([[pose()], [pose((5, 5, 5))]]), [])
        np.testing.assert_array_equal(result, [100., 100.])
        invalid = pose()
        invalid[3:] = 0
        with self.assertRaises(ValueError):
            observed_clearance(np.array([invalid]), [])

    def test_invalid_quaternions_extents_and_rotated_boxes(self):
        for bad in (np.zeros(7), np.r_[[0, 0, 0], [np.nan, 0, 0, 0]]):
            with self.assertRaises(ValueError):
                dense_poses(np.array([bad]))
        for spacing in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                dense_poses(np.array([pose()]), spacing)
        base = box((0, 0, 0), (1, 1, 1))
        for index, value in ((3, 0), (4, -1), (0, float('inf'))):
            invalid = base.copy()
            invalid[index] = value
            with self.assertRaises(ValueError):
                observed_clearance(np.array([pose()]), [invalid])
        for quaternion in ([0, 0, 0, 0], [math.cos(.1), 0, 0, math.sin(.1)], [1, 0, np.nan, 0]):
            invalid = base.copy()
            invalid[6:] = quaternion
            with self.assertRaises(ValueError):
                observed_clearance(np.array([pose()]), [invalid])
        # Quaternion sign does not change an identity box orientation.
        negative = base.copy()
        negative[6] = -1
        np.testing.assert_equal(observed_clearance(np.array([pose()]), [negative]),
                                observed_clearance(np.array([pose()]), [base]))


if __name__ == '__main__':
    unittest.main()
