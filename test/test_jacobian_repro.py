import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import unittest
from .viso_test_case import VisOTestCase

import numpy as np
from bboVisO.test import check_jacobian, calc_2point_jacobian
import bboVisO.viso as viso
from jax import jacfwd as jacobian
from jax import jit

import calibcam.helper_ag
import matplotlib.pyplot as plt


class TestJacobianReproClass(VisOTestCase):
    def test_opt_jacobian(self):
        vars_free, obfun_args, optim_args = viso.make_optim_input(self.calibs, self.opts, self.landmarks,
                                                                      self.positions,
                                                                      self.oris_flu.as_rotvec(),
                                                                      landmark_imcoords=self.landmark_imcoords)

        obfun_args["vars_full"] = np.where(obfun_args["vars_full"] == 0, 1e-10, obfun_args["vars_full"])
        vars_free_opt = np.where(vars_free == 0, 1e-10, vars_free)

        print(f"Jacobian check"
              f"{check_jacobian(viso.objfun_repro_jacobian_wrapper, viso.objfun_repro_wrapper, vars_free, obfun_args)}")

    def off_test_rodrigues_as_rotmats_jacobian_zero(self):
        np.random.seed(0)

        # 0 rotvects
        rot_vecs = np.zeros((2, 3))
        # rot_vecs = np.where(rot_vecs == 0, 1e-10, rot_vecs)

        j_num = calc_2point_jacobian(jit(calibcam.helper_ag.rodrigues_as_rotmats), 1e-6, rot_vecs)
        j = jacobian(calibcam.helper_ag.rodrigues_as_rotmats)(rot_vecs)

        plt.figure(figsize=(18, 9))
        plt.subplot(1, 2, 1)
        #
        # j_dim_mat = (np.prod(calibcam.helper_ag.rodrigues_as_rotmats(rot_vecs).shape)), np.prod(rot_vecs.shape)
        # print(j_num)
        # print(j_num.shape)
        # plt.imshow(np.hstack((
        #     j_num.reshape(j_dim_mat),
        #     np.full((j_dim_mat[0], 1), np.nan),
        #     j.reshape(j_dim_mat)
        # )))
        # plt.colorbar()
        # plt.subplot(1, 2, 2)
        # plt.imshow((j_num - j).reshape(j_dim_mat))
        # plt.colorbar()
        # plt.show()

        np.testing.assert_allclose(j, j_num, rtol=1e-7, atol=1e-8)

    def off_test_rodrigues_as_rotmats_jacobian_mixed(self):
        np.random.seed(0)

        # mixed rot_vecs
        rot_vecs = np.random.permutation(np.vstack((np.zeros((10, 3)), self.oris_flu.as_rotvec())))
        rot_vecs = self.oris_flu.as_rotvec()
        rot_vecs = np.where(rot_vecs == 0, 1e-10, rot_vecs)

        j_num = calc_2point_jacobian(calibcam.helper_ag.rodrigues_as_rotmats, 1e-6, rot_vecs)
        j = jacobian(calibcam.helper_ag.rodrigues_as_rotmats)(rot_vecs)
        np.testing.assert_allclose(j, j_num, rtol=1e-7, atol=1e-8)

    def off_test_rodrigues_as_rotmats_jacobian_bulk(self):
        np.random.seed(0)

        for i in range(10):
            landmarks_rand, oris_flu_rand, positions_rand = self.pertubate_opt_quantities()
            rot_vecs = np.where(oris_flu_rand == 0, 1e-10, oris_flu_rand)

            j_num = calc_2point_jacobian(calibcam.helper_ag.rodrigues_as_rotmats, 1e-6, rot_vecs)
            j = jacobian(calibcam.helper_ag.rodrigues_as_rotmats)(rot_vecs)
            np.testing.assert_allclose(j, j_num, rtol=1e-7, atol=1e-8)

    def off_test_all_jacobians(self):
        np.random.seed(0)

        for i in range(10):
            landmarks_rand, oris_flu_rand, positions_rand = self.pertubate_opt_quantities()
            vars_free, args = viso.make_optim_input(self.calibs, self.opts, landmarks_rand, positions_rand,
                                                    oris_flu_rand, landmark_imcoords=self.landmark_imcoords)

            print(
                f"Jacobian check {check_jacobian(viso.objfun_repro_jacobian_wrapper, viso.objfun_repro_wrapper, vars_free, args)}")

            # Test
            # np.testing.assert_allclose(error.ravel(), error_wrapper, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
