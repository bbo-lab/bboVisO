import numpy as np

from scipy.spatial.transform import Rotation as R
from bbo.geometry import RigidTransform
# from scipy.sparse import block_diag

from bboVisO import viso_ag as viso_ag
from bboVisO.unravel import unravel_vars


# def objfun_angular_runner(landmarks, positions, oris_flu, landmark_vecs, opts) -> np.ndarray:
#     errors2d = viso_ag.objfun_angular(landmarks, positions, oris_flu, opts["camera_positions"], landmark_vecs,
#                                       opts["lm_close_pen_weight"], True)
#
#     return np.nansum(errors2d, axis=0)


def objfun_angular_wrapper(free_vars, args, no_ravel=False) -> np.ndarray:
    (landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
     landmark_vecs, lm_close_pen_weight) = unravel_vars_angular(free_vars, args)

    errors_angular = np.array(args['precalc']['objfunc_angular'](landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
                                                                 landmark_vecs, lm_close_pen_weight))

    if no_ravel:
        return errors_angular
    else:
        errors_angular[np.isnan(errors_angular)] = 0
        return errors_angular.ravel()


def objfun_angular_jacobian_wrapper(free_vars, args) -> np.ndarray:
    # landmarks, positions, oris_flu, landmark_vecs, cam_positions, lm_close_pen_weight = unravel_vars_angular(free_vars,
    #                                                                                                          args)
    # result_shape = landmark_vecs.shape[:-1]+(1,)
    #
    # # TODO (minor speed) Do not calculate Jacobian if there are no related free variables
    # obj_fcn_jacobian = [
    #     np.array(
    #         # Ravel over input dimensions
    #         j(landmarks, positions, oris_flu, cam_positions, landmark_vecs, lm_close_pen_weight).reshape(
    #             result_shape + (-1,))
    #     )
    #     for j in args['precalc']['jacobians_angular']
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
    # print("obj_fcn_jacobian", obj_fcn_jacobian.shape)
    #
    # # Ravel variable dimension#
    # # TODO figure out how to deal with missing cam rotations
    # obj_fcn_jacobian = obj_fcn_jacobian.reshape((np.product(result_shape), -1))
    #
    # # Dump non-free variables
    # obj_fcn_jacobian = obj_fcn_jacobian[:, args['mask_opt']]
    #
    # return obj_fcn_jacobian
    return []


def make_jacobian_sparsity_mask_angular(m, k, n, mask_free_input):
    d = 1  # Output dimension, angle
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
                              jac_sparsity_cam_pos_rot,
                              ))

    jac_sparsity[:] = True  # Until this gives the same result ...
    return jac_sparsity[:, mask_free_input]


def unravel_vars_angular(free_vars, args):
    landmarks, positions, oris_flu, cam_rotvecs, cam_ts = unravel_vars(free_vars, args)

    landmark_vecs = args["landmark_vecs"]
    lm_close_pen_weight = args["lm_close_pen_weight"]

    return (landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
            landmark_vecs, lm_close_pen_weight)
