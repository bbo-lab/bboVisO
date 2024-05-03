import numpy as np
# from scipy.sparse import block_diag

from bboVisO import viso_ag as viso_ag
from bboVisO.unravel import unravel_vars


def objfun_angular_runner(landmarks, positions, oris_flu, landmark_vecs, opts) -> np.ndarray:
    errors2d = viso_ag.objfun_angular(landmarks, positions, oris_flu,
                                      landmark_vecs, opts["camera_positions"],
                                      opts["lm_close_pen_weight"], True)

    return np.nansum(errors2d, axis=0)


def objfun_angular_wrapper(free_vars, args, no_ravel=False) -> np.ndarray:
    landmarks, positions, oris_flu, landmark_vecs, cam_positions, lm_close_pen_weight = unravel_vars_angular(free_vars,
                                                                                                             args)

    errors_angular = np.array(args['precalc']['objfunc_angular'](landmarks, positions, oris_flu,
                                                                 landmark_vecs, cam_positions, lm_close_pen_weight))

    if no_ravel:
        return np.nansum(errors_angular, axis=0)
    else:
        errors_angular[np.isnan(errors_angular)] = 0
        return np.nansum(errors_angular, axis=0).ravel()


def objfun_angular_jacobian_wrapper(free_vars, args) -> np.ndarray:
    # landmarks, positions, oris_flu, landmark_vecs, lm_close_pen_weight = unravel_vars_angular(free_vars, args)
    #
    # positions = np.where(positions == 0, 1e-8, positions)
    # oris_flu = np.where(oris_flu == 0, 1e-8, oris_flu)
    # landmarks = np.where(landmarks == 0, 1e-8, landmarks)
    #
    # # TODO (minor speed) Do not calculate Jacobian if there are no related free variables
    # obj_fcn_jacobian = [
    #     np.array(
    #         # Ravel over input dimensions
    #         # j(landmarks, positions, oris_flu, landmark_vecs).reshape(
    #         #     landmark_imcoords.shape + (-1,))
    #     )
    #     for j in args['precalc']['jacobians_repro']
    # ]
    #
    # # for i, jac in enumerate(obj_fcn_jacobian):
    # #     print(f"Jacobian {i} norm: {np.linalg.norm(jac)}")
    #
    # # Concatenate along input dimensions
    # obj_fcn_jacobian = np.concatenate(
    #     obj_fcn_jacobian,
    #     axis=4)
    #
    # # Combine the 2 cameras (one will be nan)
    # obj_fcn_jacobian = np.nansum(obj_fcn_jacobian, axis=0)
    #
    # # Ravel variable dimension
    # obj_fcn_jacobian = obj_fcn_jacobian.reshape((-1, args['mask_opt'].size))
    #
    # # Dump non-free variables
    # obj_fcn_jacobian = obj_fcn_jacobian[:, args['mask_opt']]
    #
    # return obj_fcn_jacobian
    return []


def make_jacobian_sparsity_mask_angular(m, k, n, mask_free_input):
    d = 1  # Output dimension, angle
    # TODO: Make sparse matrix
    jac_sparsity_lm = np.zeros((d, k, n) + (n, 3), dtype=bool)
    for i_n in range(n):
        jac_sparsity_lm[:, :, i_n, i_n] = True
    jac_sparsity_lm = jac_sparsity_lm.reshape((d * k * n, -1))

    jac_sparsity_pos_rot = np.zeros((d, k, n) + (k, 3), dtype=bool)
    for i_k in range(k):
        jac_sparsity_pos_rot[:, i_k, :, i_k] = True
    jac_sparsity_pos_rot = jac_sparsity_pos_rot.reshape((d * k * n, -1))

    jac_sparsity_cam_pos_rot = np.ones((d, k, n) + (m, 3), dtype=bool)
    jac_sparsity_cam_pos_rot = jac_sparsity_cam_pos_rot.reshape((d * k * n, -1))

    jac_sparsity = np.hstack((jac_sparsity_lm,
                              jac_sparsity_pos_rot,
                              jac_sparsity_pos_rot,
                              jac_sparsity_cam_pos_rot,
                              jac_sparsity_cam_pos_rot,))

    return jac_sparsity[:, mask_free_input]


def unravel_vars_angular(free_vars, args):
    landmark_vecs = args["landmark_vecs"]
    lm_close_pen_weight = args["lm_close_pen_weight"]

    landmarks, positions, oris_flu, cam_rotations, cam_translations = unravel_vars(free_vars, args)
    return landmarks, positions, oris_flu, landmark_vecs, cam_translations, lm_close_pen_weight
