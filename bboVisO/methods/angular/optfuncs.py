import numpy as np
from jax import jit, jacfwd as jacobian
from scipy.spatial.transform import Rotation

from .optfuncs_ag import objfun
from ..common import (objfun_jacobian_wrapper_sparse as objfun_jacobian_wrapper_sparse_common,
                      objfun_jacobian_wrapper_dense as objfun_jacobian_wrapper_dense_common,
                      make_jacobian_sparsity_mask as make_jacobian_sparsity_mask_common)
from ...unravel import unravel_vars as unravel_vars_base, unpack_calibs


def objfun_runner(landmarks, positions, oris_flu, landmark_vecs, calibs, opts) -> np.ndarray:
    cam_Rs, cam_ts = unpack_calibs(calibs)[-2:]
    errors = np.array(jit(objfun)(landmarks, positions, oris_flu, Rotation.from_matrix(cam_Rs).as_rotvec(), cam_ts,
                                  landmark_vecs, opts["lm_close_pen_weight"], opts["lm_weights"]))

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
    sd = 1
    jacs = get_jacobians(args)

    jac_dense = objfun_jacobian_wrapper_dense_common(free_vars, args, jacs, unravel_vars, sd)

    return jac_dense


def objfun_jacobian_wrapper(free_vars, args) -> np.ndarray:
    sd = 1
    jacs = get_jacobians(args)

    jac_sparse = objfun_jacobian_wrapper_sparse_common(free_vars, args, jacs, unravel_vars, sd)

    return jac_sparse


def make_jacobian_sparsity_mask(m, k, n, mask_free_input):
    d = 1  # Output dimension, angle
    return make_jacobian_sparsity_mask_common(m, k, n, d, mask_free_input)


def unravel_vars(free_vars, args):
    landmarks, positions, oris_flu, cam_rotvecs, cam_ts = unravel_vars_base(free_vars, args)

    landmark_vecs = args["landmark_vecs"]
    lm_close_pen_weight = args["lm_close_pen_weight"]
    lm_weights = args["landmarks_weights"]

    return (landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
            landmark_vecs, lm_weights, lm_close_pen_weight)


jacobians_angular = None


def get_jacobians(opts):
    global jacobians_angular
    if jacobians_angular is None:
        if "jax_backend" not in opts:
            opts["jax_backend"] = "cpu"
            print("Set JAX backend to CPU")
        jacobians_angular = [jit(jacobian(objfun, i_var), backend=opts["jax_backend"]) for i_var in range(5)]
    return jacobians_angular
