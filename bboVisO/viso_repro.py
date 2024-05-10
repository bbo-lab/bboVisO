import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.sparse import block_diag

from bbo.geometry import RigidTransform
from bboVisO import viso_ag as viso_ag
from bboVisO.unravel import unravel_vars, unpack_calibs


def objfun_repro_runner(landmarks, positions, oris_flu, landmark_imcoords, calibs, opts) -> np.ndarray:
    cam_matrices, ks, xis, cam_Rs, cam_ts = unpack_calibs(calibs)
    errors2d = np.array(viso_ag.objfun_repro(landmarks, positions, oris_flu, R.from_matrix(cam_Rs).as_rotvec(), cam_ts,
                                             landmark_imcoords, cam_matrices, ks, xis,
                                             opts["lm_close_pen_weight"]))

    return errors2d


def objfun_repro_wrapper(free_vars, args, no_ravel=False) -> np.ndarray:
    (landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
     landmark_imcoords, cam_matrices, ks, xis, lm_close_pen_weight) = unravel_vars_repro(free_vars, args)

    errors2d = np.array(args['precalc']['objfunc_repro'](landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
                                                         landmark_imcoords, cam_matrices, ks, xis,
                                                         lm_close_pen_weight))
    if no_ravel:
        return errors2d
    else:
        errors2d[np.isnan(errors2d)] = 0
        return errors2d.ravel()


def objfun_repro_jacobian_wrapper(free_vars, args) -> np.ndarray:
    (landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
     landmark_imcoords, cam_matrices, ks, xis, lm_close_pen_weight) = unravel_vars_repro(free_vars, args)
    result_shape = landmark_imcoords.shape

    # TODO (minor speed) Do not calculate Jacobian if there are no related free variables
    obj_fcn_jacobian = [
        np.array(
            # Ravel over input dimensions
            j(landmarks, positions, oris_flu, cam_rotvecs, cam_ts, landmark_imcoords, cam_matrices, ks, xis,
              lm_close_pen_weight).reshape(
                result_shape + (-1,))
        )
        for j in args['precalc']['jacobians_repro']
    ]

    # print([j.shape for j in obj_fcn_jacobian])

    # for i, jac in enumerate(obj_fcn_jacobian):
    #     print(f"Jacobian {i} norm: {np.linalg.norm(jac)}")

    # Concatenate along input dimensions
    obj_fcn_jacobian = np.concatenate(
        obj_fcn_jacobian,
        axis=4)

    # print("obj_fcn_jacobian", obj_fcn_jacobian.shape)

    # Ravel variable dimension
    obj_fcn_jacobian = obj_fcn_jacobian.reshape((np.product(result_shape), args['mask_opt'].size))

    # Dump non-free variables
    obj_fcn_jacobian = obj_fcn_jacobian[:, args['mask_opt']]

    return obj_fcn_jacobian


def make_jacobian_sparsity_mask_repro(m, k, n, mask_free_input):
    d = 2  # Output dimension, imcoords
    # TODO: Make sparse matrix
    jac_sparsity_lm = np.zeros((m, d, k, n) + (n, 3), dtype=bool)
    for i_n in range(n):
        jac_sparsity_lm[:, :, :, i_n, i_n] = True
    jac_sparsity_lm = jac_sparsity_lm.reshape((m * d * k * n, -1))

    jac_sparsity_pos_rot = np.zeros((m, d, k, n) + (k, 3), dtype=bool)
    for i_k in range(k):
        jac_sparsity_pos_rot[:, :, i_k, :, i_k] = True
    jac_sparsity_pos_rot = jac_sparsity_pos_rot.reshape((m * d * k * n, -1))

    jac_sparsity_cam_pos_rot = np.ones((m, d, k, n) + (m, 3), dtype=bool)
    jac_sparsity_cam_pos_rot = jac_sparsity_cam_pos_rot.reshape((m * d * k * n, -1))

    jac_sparsity = np.hstack((jac_sparsity_lm,
                              jac_sparsity_pos_rot,
                              jac_sparsity_pos_rot,
                              jac_sparsity_cam_pos_rot,
                              jac_sparsity_cam_pos_rot,))

    jac_sparsity[:] = True  # Until this gives the same result ...
    return jac_sparsity[:, mask_free_input]


def unravel_vars_repro(free_vars, args):
    landmarks, positions, oris_flu, cam_rotvecs, cam_ts = unravel_vars(free_vars, args)

    landmark_imcoords = args["landmark_imcoords"]
    cam_matrices, ks, xis = unpack_calibs(args["calibs"])[:3]
    lm_close_pen_weight = args["lm_close_pen_weight"]
    return (landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
            landmark_imcoords, cam_matrices, ks, xis, lm_close_pen_weight)
