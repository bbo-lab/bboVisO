import numpy as np
from jax import jit, jacfwd as jacobian

from scipy.spatial.transform import Rotation as R

from .optfuncs_ag import objfun
from ..common import (objfun_jacobian_wrapper_sparse as objfun_jacobian_wrapper_sparse_common,
                      objfun_jacobian_wrapper_dense as objfun_jacobian_wrapper_dense_common,
                      make_jacobian_sparsity_mask as make_jacobian_sparsity_mask_common)
from ...ravel import unravel_vars as unravel_vars_base, unpack_calibs
objfun = jit(objfun)

def objfun_runner(landmarks, positions, oris_flu, landmark_imcoords, calibs, opts) -> np.ndarray:
    cam_matrices, ks, xis, cam_Rs, cam_ts = unpack_calibs(calibs)
    errors = np.array(objfun(landmarks, positions, oris_flu, R.from_matrix(cam_Rs).as_rotvec(), cam_ts,
                                  landmark_imcoords,
                                  1,  # TODO: Introduce landmark weights
                                  1,  # TODO: Introduce mask weights
                                  cam_matrices, ks, xis,
                                  opts["lm_close_pen_weight"]))

    return errors


def objfun_wrapper(free_vars, args, no_ravel=False) -> np.ndarray:
    obfun_inputs = unravel_vars(free_vars, args)
    errors = np.array(objfun(*obfun_inputs))

    if no_ravel:
        return errors
    else:
        errors[np.isnan(errors)] = 0
        return errors.ravel()


def objfun_jacobian_wrapper_dense(free_vars, args) -> np.ndarray:
    sd = 2
    jacs = get_jacobians(args)

    jac_dense = objfun_jacobian_wrapper_dense_common(free_vars, args, jacs, unravel_vars, sd)

    return jac_dense


def objfun_jacobian_wrapper(free_vars, args) -> np.ndarray:
    sd = 2
    jacs = get_jacobians(args)

    jac_sparse = objfun_jacobian_wrapper_sparse_common(free_vars, args, jacs, unravel_vars, sd)

    return jac_sparse


def make_jacobian_sparsity_mask(m, k, n, mask_free_input):
    d = 2  # Output dimension, angle
    return make_jacobian_sparsity_mask_common(m, k, n, d, mask_free_input)


def unravel_vars(free_vars, args):
    landmarks, positions, oris_flu, cam_rotvecs, cam_ts = unravel_vars_base(free_vars, args["ravel_args"])

    landmark_imcoords = args["landmark_imcoords"]
    cam_matrices, ks, xis = unpack_calibs(args["calibs"])[:3]
    lm_close_pen_weight = args["lm_close_pen_weight"]
    lm_weights = args["landmarks_weights"]
    weight_mask = args["weight_mask"]
    weight = args["weight"]
    mask_weight = np.ones_like(weight_mask)
    mask_weight[weight_mask] = weight
    return (landmarks, positions, oris_flu, cam_rotvecs, cam_ts, landmark_imcoords, lm_weights, mask_weight,
            cam_matrices, ks, xis, lm_close_pen_weight)


jacobians_repro = None


def get_jacobians(opts):
    global jacobians_repro
    if jacobians_repro is None:
        if "jax_backend" not in opts:
            opts["jax_backend"] = "cpu"
            print("Set JAX backend to CPU")
        jacobians_repro = [jit(jacobian(objfun, i_var), backend=opts["jax_backend"]) for i_var in range(5)]
    return jacobians_repro
