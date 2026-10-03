from types import SimpleNamespace
import unittest

import numpy as np

from hnuter_dit.lidar_condition import frame_to_tokens, points_to_tokens


def coverage(points, tokens, mask):
    boxes = tokens[mask].astype(np.float64)
    return (np.abs(np.asarray(points)[:, None] - boxes[None, :, :3]) <= boxes[None, :, 3:6] / 2 + 1e-9).all(axis=-1).any(axis=-1)


class LidarConditionTests(unittest.TestCase):
    def test_more_than_32_regions_keeps_every_point(self):
        random = np.random.default_rng(73)
        points = random.uniform(-4, 4, (3000, 3))
        tokens, mask, metadata = points_to_tokens(points, [0, 0, 0], [6, 0, 0])
        self.assertEqual(tokens.shape, (32, 10))
        self.assertEqual(int(mask.sum()), 32)
        self.assertTrue(coverage(points, tokens, mask).all())
        self.assertEqual(sum(metadata["token_point_counts"]), len(points))
        self.assertEqual(metadata["dropped_roi_points"], 0)
        self.assertFalse(metadata["free_space_certified"])
        np.testing.assert_array_equal(tokens[:, 6:10], np.tile([1, 0, 0, 0], (32, 1)))

    def test_one_slot_merges_all_observations_not_first_point(self):
        points = np.array([[-3, 1, 0], [4, -2, 0.5], [0, 1, 2]])
        tokens, mask, metadata = points_to_tokens(points, [0, 0, 0], [6, 0, 0], max_tokens=1)
        self.assertTrue(coverage(points, tokens, mask).all())
        self.assertEqual(metadata["token_point_counts"], [3])
        np.testing.assert_allclose(tokens[0, 3:6], [7.24, 3.24, 2.24], rtol=1e-6)

    def test_translation_invariance_and_roi(self):
        points = np.array([[1, 0, 0], [1.2, 0.1, 0.2], [-2, 1, 0], [9, 0, 0]])
        offset = np.array([-25.0, -35.0, 2.2])
        first, mask, metadata = points_to_tokens(points, [0, 0, 0], [6, 0, 0], max_tokens=2)
        moved, moved_mask, moved_meta = points_to_tokens(points + offset, offset, offset + [6, 0, 0], max_tokens=2)
        np.testing.assert_array_equal(mask, moved_mask)
        np.testing.assert_allclose(moved[mask, :3] - offset, first[mask, :3], atol=2e-6)
        np.testing.assert_allclose(moved[mask, 3:6], first[mask, 3:6], atol=3e-6)
        self.assertEqual(metadata["outside_roi_points"], 1)
        self.assertEqual(moved_meta["roi_points"], 3)
        self.assertTrue(coverage(points[:3] + offset, moved, mask).all())

    def test_input_order_does_not_change_partition(self):
        points = np.random.default_rng(19).uniform(-3, 3, (400, 3))
        first, mask, _ = points_to_tokens(points, [0, 0, 0], [4, 0, 0])
        reverse, reverse_mask, _ = points_to_tokens(points[::-1], [0, 0, 0], [4, 0, 0])
        np.testing.assert_array_equal(first, reverse)
        np.testing.assert_array_equal(mask, reverse_mask)

    def test_empty_means_unknown_and_valid_padding_rotations(self):
        tokens, mask, metadata = points_to_tokens([], [0, 0, 0], [3, 0, 0])
        self.assertFalse(mask.any())
        self.assertEqual(metadata["active_tokens"], 0)
        self.assertFalse(metadata["free_space_certified"])
        self.assertIn("unknown", metadata["unobserved_space"])
        np.testing.assert_array_equal(tokens[:, 6], np.ones(32))

    def test_nan_and_invalid_configuration_are_rejected(self):
        for point in ([np.nan, 0, 0], [0, np.inf, 0]):
            with self.assertRaises(ValueError):
                points_to_tokens([point], [0, 0, 0], [3, 0, 0])
        for options in ({"max_tokens": 33}, {"max_tokens": True}, {"range_m": 0}, {"cell_padding": -0.1}):
            with self.assertRaises(ValueError):
                points_to_tokens([[1, 0, 0]], [0, 0, 0], [3, 0, 0], **options)

    def test_snapshot_pose_transform_and_timestamp(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, [-25, -35, 2.33]
        points = np.array([[1, 0, 0], [2, 0, 0.1]])
        frame = SimpleNamespace(points=points, world_T_sensor=pose, sim_time=0.2,
                                point_timestamps=np.array([0.2, 0.2]), frame_index=2)
        tokens, mask, metadata = frame_to_tokens(frame, [-25, -35, 2.2], [-25, -29, 2.2])
        world = points @ rotation.T + pose[:3, 3]
        self.assertTrue(coverage(world, tokens, mask).all())
        self.assertEqual(metadata["simulation_time"], 0.2)
        self.assertEqual(metadata["frame_index"], 2)
        frame.point_timestamps[0] = 0.1
        with self.assertRaises(ValueError):
            frame_to_tokens(frame, [-25, -35, 2.2], [-25, -29, 2.2])
        frame.point_timestamps[0] = 0.2
        frame.world_T_sensor[0, 0] = 2.0
        with self.assertRaises(ValueError):
            frame_to_tokens(frame, [-25, -35, 2.2], [-25, -29, 2.2])


if __name__ == "__main__":
    unittest.main()
