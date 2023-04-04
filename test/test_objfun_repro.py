import unittest
from .viso_test_case import VisOTestCase
import os

import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import numpy as np
import bboVisO.viso as viso
import bboVisO.viso_ag as viso_ag


class TestObjfunReproClass(VisOTestCase):
    def test_opt_objfun_repro(self):
        cam_matrices, ks, xis, cam_Rs, cam_ts = viso.unpack_calibs(self.calibs)
        error = np.nansum(
            viso_ag.objfun_repro(self.landmarks, self.positions, self.oris_flu.as_rotvec(), self.landmark_imcoords,
                                 cam_matrices, ks, xis, cam_Rs,
                                 cam_ts), axis=0)
        max_error = np.max(np.abs(error))
        np.testing.assert_allclose(error, np.zeros_like(error), rtol=1e-10, atol=1e-10)

    def test_opt_objfun_wrapper_repro(self):
        vars_free_opt, args_opt = viso.make_optim_input(self.calibs, self.opts, self.landmarks, self.positions,
                                                        self.oris_flu.as_rotvec(),
                                                        landmark_imcoords=self.landmark_imcoords)
        error = viso.objfun_repro_wrapper(vars_free_opt, args_opt)
        max_error = np.max(np.abs(error))
        np.testing.assert_allclose(error, np.zeros_like(error), rtol=1e-10, atol=1e-10)

    def test_opt_objfun_vs_wrapper_repro(self):
        np.random.seed(0)

        for i in range(10):
            # Pertubate ground truth
            landmarks_rand, oris_flu_rand, positions_rand, landmark_vecs_rand, landmark_dists_rand = \
                self.pertubate_opt_quantities()

            # Calculate objective function directly
            cam_matrices, ks, xis, cam_Rs, cam_ts = viso.unpack_calibs(self.calibs)
            error = viso_ag.objfun_repro(landmarks_rand, positions_rand, oris_flu_rand, self.landmark_imcoords,
                                         cam_matrices, ks, xis, cam_Rs, cam_ts)
            error = np.nansum(error, axis=0)

            # Calculate objective function through wrapper
            vars_free, args = viso.make_optim_input(self.calibs, self.opts, landmarks_rand, positions_rand,
                                                    oris_flu_rand, landmark_imcoords=self.landmark_imcoords)
            error_wrapper = viso.objfun_repro_wrapper(vars_free, args)

            # Test
            np.testing.assert_allclose(error.ravel(), error_wrapper, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
