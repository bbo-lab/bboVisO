import unittest
from unittest.mock import patch

import numpy as np

from bboVisO.interpolation import interpolate_imcoords


class TestInterpolateImcoords(unittest.TestCase):
    @staticmethod
    def interpolate(times, angles, schedules):
        """Use a synthetic camera whose first image coordinate is ray angle."""
        imcoords = np.stack((angles, np.zeros_like(angles)), axis=-1)

        def to_vecs(coords, calibs):
            angle = coords[..., 0]
            return np.stack((np.sin(angle), np.zeros_like(angle), np.cos(angle)), axis=-1)

        def to_imcoords(vecs, calibs):
            return np.stack((np.arctan2(vecs[..., 0], vecs[..., 2]), vecs[..., 1]), axis=-1)

        with patch('bboVisO.interpolation.landmark_vecs_from_imcoords', side_effect=to_vecs), \
                patch('bboVisO.interpolation.landmark_imcoords_from_vecs', side_effect=to_imcoords):
            return interpolate_imcoords(imcoords, times, None, schedules)

    def test_multiple_gaps_use_actual_times_and_camera_schedules(self):
        times = np.array([0, 1, 2, 4, 5, 7])
        angles = np.full((2, 6, 1), np.nan)
        angles[0, [0, 4, 5], 0] = [0, 0.5, 0.7]
        angles[1, [1, 3, 5], 0] = [0.1, 0.4, 0.7]
        out, vecs, returned_times, mask = self.interpolate(times, angles, [[0, 5, 7], [1, 4, 7]])
        np.testing.assert_allclose(out[0, :, 0, 0], times / 10, atol=1e-12)
        np.testing.assert_allclose(out[1, 1:, 0, 0], times[1:] / 10, atol=1e-12)
        self.assertTrue(np.isnan(out[1, 0, 0, 0]))
        np.testing.assert_array_equal(mask[..., 0], mask[..., 1])
        np.testing.assert_array_equal(mask[:, :, 0, 0],
                                      [[False, True, True, True, False, False],
                                       [False, False, True, False, True, False]])
        finite = np.isfinite(vecs).all(axis=-1)
        np.testing.assert_allclose(np.linalg.norm(vecs[finite], axis=-1), 1)
        self.assertIs(returned_times, times)

    def test_missing_recorded_landmark_blocks_both_neighboring_intervals(self):
        times = np.arange(5)
        angles = np.full((1, 5, 2), np.nan)
        angles[0, [0, 4], 0] = [0, 0.4]
        angles[0, [0, 2, 4], 1] = [0, 0.2, 0.4]
        out, _, _, mask = self.interpolate(times, angles, [[0, 2, 4]])
        self.assertTrue(np.isnan(out[0, 1:4, 0, 0]).all())
        self.assertFalse(mask[0, :, 0].any())
        np.testing.assert_allclose(out[0, :, 1, 0], times / 10, atol=1e-12)

    def test_identical_directions_and_existing_values(self):
        times = np.arange(5)
        angles = np.array([[[0.2], [np.nan], [0.3], [np.nan], [0.2]]])
        out, _, _, mask = self.interpolate(times, angles, [[0, 4]])
        np.testing.assert_allclose(out[0, :, 0, 0], [0.2, 0.2, 0.3, 0.2, 0.2])
        np.testing.assert_array_equal(mask[0, :, 0, 0], [False, True, False, True, False])

    def test_no_extrapolation_or_interpolation_without_two_recordings(self):
        times = np.arange(5)
        angles = np.array([[[np.nan], [0.1], [np.nan], [0.3], [np.nan]]])
        for schedule in ([[1, 3]], [[1]], [[]], None):
            with self.subTest(schedule=schedule):
                out, _, _, mask = self.interpolate(times, angles, schedule)
                self.assertTrue(np.isnan(out[0, [0, 4], 0, 0]).all())
                if schedule == [[1, 3]]:
                    self.assertAlmostEqual(out[0, 2, 0, 0], 0.2)
                    self.assertTrue(mask[0, 2].all())
                else:
                    self.assertTrue(np.isnan(out[0, 2, 0, 0]))
                    self.assertFalse(mask.any())


if __name__ == '__main__':
    unittest.main()
