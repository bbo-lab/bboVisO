import unittest

from bboVisO.interpolation import interpolate_imcoords_rebase
import os

import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import numpy as np


class TestInterpolationClass(unittest.TestCase):
    def test_interpolate_imcoords_rebase(self):
        times = np.array([1, 2])
        interp_times = np.linspace(1,1.5,6)
        vectors = np.array(
            [[[[0, 0, 1], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
             [[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [1, 0, 0]]]])

        _, vectors_interp, interp_times_out =  interpolate_imcoords_rebase(vectors, times, None, interp_times,
                                                                           omit_reprojection=True)

        np.testing.assert_allclose((np.mean(vectors, axis=1)>0)/np.sqrt(2), vectors_interp[:,-1], rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
