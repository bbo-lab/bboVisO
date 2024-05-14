from time import time

import numpy as np
from jax import jit, jacfwd as jacobian
from scipy.spatial.transform import Rotation

from .optfuncs_ag import objfun
from ..common import objfun_jacobian_wrapper_sparse
from ...unravel import unravel_vars as unravel_vars_base, unpack_calibs


def objfun_runner(landmarks, positions, oris_flu, landmark_vecs, calibs, opts) -> np.ndarray:
    cam_Rs, cam_ts = unpack_calibs(calibs)[-2:]
    errors = np.array(objfun(landmarks, positions, oris_flu, Rotation.from_matrix(cam_Rs).as_rotvec(), cam_ts,
                             landmark_vecs, opts["lm_close_pen_weight"]))

    return errors


def objfun_wrapper(free_vars, args, no_ravel=False) -> np.ndarray:
    obfun_inputs = unravel_vars(free_vars, args)
    errors = np.array(jit(objfun)(*obfun_inputs))

    if no_ravel:
        return errors
    else:
        errors[np.isnan(errors)] = 0
        return errors.ravel()


def objfun_jacobian_wrapper_dense(free_vars, args) -> np.ndarray:
    (landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
     landmark_vecs, lm_close_pen_weight) = unravel_vars(free_vars, args)
    result_shape = landmark_vecs.shape[:-1] + (1,)

    # TODO (minor speed) Do not calculate Jacobian if there are no related free variables
    obj_fcn_jacobian = [
        np.array(
            # Ravel over input dimensions
            j(landmarks, positions, oris_flu, cam_rotvecs, cam_ts, landmark_vecs, lm_close_pen_weight).reshape(
                result_shape + (-1,))
        )
        for j in get_jacobians(args)
    ]

    # Concatenate along input dimensions
    obj_fcn_jacobian = np.concatenate(
        obj_fcn_jacobian,
        axis=4)

    # Ravel variable dimension
    obj_fcn_jacobian = obj_fcn_jacobian.reshape((np.prod(result_shape), args['mask_opt'].size))

    # Dump non-free variables
    obj_fcn_jacobian = obj_fcn_jacobian[:, args['mask_opt']]

    # There are nans here, but not in the repro wrapper. Why? Possibly arccos numerically > 1?
    obj_fcn_jacobian[np.isnan(obj_fcn_jacobian)] = 0

    return obj_fcn_jacobian


def objfun_jacobian_wrapper(free_vars, args) -> np.ndarray:
    sd = 1
    jacs = get_jacobians(args)
    jac_sparse = objfun_jacobian_wrapper_sparse(free_vars, args, jacs, unravel_vars, sd)
    return jac_sparse


def make_jacobian_sparsity_mask(m, k, n, mask_free_input):
    d = 1  # Output dimension, angle
    # TODO: Make sparse matrix
    jac_sparsity_lm = np.zeros((m, k, n, d) + (n, 3), dtype=bool)
    for i_n in range(n):
        jac_sparsity_lm[:, :, i_n, :, i_n] = True
    jac_sparsity_lm = jac_sparsity_lm.reshape((m * k * n * d, -1))

    jac_sparsity_pos_rot = np.zeros((m, k, n, d) + (k, 3), dtype=bool)
    for i_k in range(k):
        jac_sparsity_pos_rot[:, i_k, :, :, i_k] = True
    jac_sparsity_pos_rot = jac_sparsity_pos_rot.reshape((m * k * n * d, -1))

    jac_sparsity_cam_pos_rot = np.ones((m, k, n, d) + (m, 3), dtype=bool)
    jac_sparsity_cam_pos_rot = jac_sparsity_cam_pos_rot.reshape((m * k * n * d, -1))

    jac_sparsity = np.hstack((jac_sparsity_lm,
                              jac_sparsity_pos_rot,
                              jac_sparsity_pos_rot,
                              jac_sparsity_cam_pos_rot,
                              jac_sparsity_cam_pos_rot,
                              ))

    # jac_sparsity[:] = True  # Until this gives the same result ...
    return jac_sparsity[:, mask_free_input]


def unravel_vars(free_vars, args):
    landmarks, positions, oris_flu, cam_rotvecs, cam_ts = unravel_vars_base(free_vars, args)

    landmark_vecs = args["landmark_vecs"]
    lm_close_pen_weight = args["lm_close_pen_weight"]

    return (landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
            landmark_vecs, lm_close_pen_weight)


jacobians_angular = None


def get_jacobians(opts):
    global jacobians_angular
    if jacobians_angular is None:
        if "jax_backend" not in opts:
            opts["jax_backend"] = "cpu"
            print("Set JAX backend to CPU")
        jacobians_angular = [jit(jacobian(objfun, i_var), backend=opts["jax_backend"]) for i_var in range(5)]
    return jacobians_angular
