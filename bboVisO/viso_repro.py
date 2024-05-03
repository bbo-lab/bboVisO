import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.sparse import block_diag

from bbo.geometry import RigidTransform
from bboVisO import viso_ag as viso_ag
from bboVisO.unravel import unravel_vars, unpack_calibs


def objfun_repro_runner(landmarks, positions, oris_flu, landmark_imcoords, calibs, opts) -> np.ndarray:
    cam_matrices, ks, xis, cam_Rs, cam_ts = unpack_calibs(calibs)
    errors2d = viso_ag.objfun_repro(landmarks, positions, oris_flu,
                                    landmark_imcoords,
                                    cam_matrices, ks, xis, cam_Rs, cam_ts,
                                    opts["lm_close_pen_weight"], True)

    return np.nansum(errors2d, axis=0)


def objfun_repro_wrapper(free_vars, args, no_ravel=False) -> np.ndarray:
    landmarks, positions, oris_flu, \
        landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts, lm_close_pen_weight = unravel_vars_repro(free_vars,
                                                                                                           args)

    errors2d = args['precalc']['objfunc_repro'](landmarks, positions, oris_flu,
                                                landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts,
                                                lm_close_pen_weight)

    if no_ravel:
        return np.nansum(errors2d, axis=0)
    else:
        return np.nansum(errors2d, axis=0).ravel()


def objfun_repro_jacobian_wrapper(free_vars, args) -> np.ndarray:
    landmarks, positions, oris_flu, \
        landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts, lm_close_pen_weight = unravel_vars_repro(free_vars,
                                                                                                           args)

    landmarks = np.where(landmarks == 0, 1e-8, landmarks)
    positions = np.where(positions == 0, 1e-8, positions)
    oris_flu = np.where(oris_flu == 0, 1e-8, oris_flu)
    cam_matrices = np.where(cam_matrices == 0, 1e-8, cam_matrices)
    cam_Rs = np.where(cam_Rs == 0, 1e-8, cam_Rs)

    # TODO (minor speed) Do not calculate Jacobian if there are no related free variables
    obj_fcn_jacobian = [
        np.array(
            # Ravel over input dimensions
            j(landmarks, positions, oris_flu, landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts).reshape(
                landmark_imcoords.shape + (-1,))
        )
        for j in args['precalc']['jacobians_repro']
    ]

    # for i, jac in enumerate(obj_fcn_jacobian):
    #     print(f"Jacobian {i} norm: {np.linalg.norm(jac)}")

    # Concatenate along input dimensions
    obj_fcn_jacobian = np.concatenate(
        obj_fcn_jacobian,
        axis=4)

    # Combine the 2 cameras (one will be nan)
    obj_fcn_jacobian = np.nansum(obj_fcn_jacobian, axis=0)

    # Ravel variable dimension
    obj_fcn_jacobian = obj_fcn_jacobian.reshape((-1, args['mask_opt'].size))

    # Dump non-free variables
    obj_fcn_jacobian = obj_fcn_jacobian[:, args['mask_opt']]

    return obj_fcn_jacobian


def make_jacobian_sparsity_mask_repro(m, k, n, mask_free_input):
    d = 2  # Output dimension, imcoords
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


def unravel_vars_repro(free_vars, args):
    landmark_imcoords = args["landmark_imcoords"]
    cam_matrices, ks, xis, cam_Rs, cam_ts = unpack_calibs(args["calibs"])
    lm_close_pen_weight = args["lm_close_pen_weight"]

    landmarks, positions, oris_flu, camera_rotations, camera_translations = unravel_vars(free_vars, args)
    world2cam_trafo = RigidTransform(rotation=R.from_rotvec(camera_rotations), translation=camera_translations).inv()
    cam_Rs = world2cam_trafo.get_rotation().as_matrix()
    cam_ts = world2cam_trafo.get_translation()
    return landmarks, positions, oris_flu, landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts, lm_close_pen_weight
