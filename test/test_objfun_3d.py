import unittest

import bboVisO.initialization
import bboVisO.methods.viso_3derr.optfuncs
import bboVisO.methods.viso_3derr.optfuncs_ag
from .viso_test_case import VisOTestCase
import os

import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import numpy as np


class TestObjfun3DClass(VisOTestCase):
    def test_opt_objfun_3d(self):
        error = bboVisO.methods.viso_3derr.optfuncs_ag.objfun(self.landmarks, self.positions, self.oris_flu.as_rotvec(), self.landmark_vecs)

        print(f"Max error value for ground truth: {np.max(np.abs(error))}")
        np.testing.assert_allclose(np.zeros_like(error), error, rtol=1e-10, atol=1e-10)

    def test_opt_objfun_wrapper_3d(self):
        vars_free_opt, args_opt = bboVisO.initialization.make_optim_input(self.calibs, self.opts, self.landmarks, self.positions,
                                                                          self.oris_flu.as_rotvec(),
                                                                          landmark_vecs=self.landmark_vecs)[:2]
        error = bboVisO.methods.viso_3derr.viso_3derr.objfun_3d_wrapper(vars_free_opt, args_opt)

        print(f"Max error value for ground truth: {np.max(np.abs(error))}")
        np.testing.assert_allclose(np.zeros_like(error), error, rtol=1e-10, atol=1e-10)

    def test_opt_objfun_vs_wrapper_3d(self):
        np.random.seed(0)

        for i in range(10):
            # Pertubate ground truth
            landmarks_rand, oris_flu_rand, positions_rand, landmark_vecs_rand, landmark_dists_rand = \
                self.pertubate_opt_quantities()

            # Calculate objective function directly
            error = bboVisO.methods.viso_3derr.optfuncs_ag.objfun(landmarks_rand, positions_rand, oris_flu_rand, landmark_vecs_rand)

            # Calculate objective function through wrapper
            vars_free, args = bboVisO.initialization.make_optim_input(self.calibs, self.opts, landmarks_rand, positions_rand,
                                                                      oris_flu_rand, landmark_vecs=landmark_vecs_rand)[:2]
            error_wrapper = bboVisO.methods.viso_3derr.viso_3derr.objfun_3d_wrapper(vars_free, args).reshape(error.shape)

            # Test
            np.testing.assert_allclose(error_wrapper, error, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
