import warnings

import numpy as np
from jax import jit, jacfwd as jacobian

from scipy.spatial.transform import Rotation as R

from .optfuncs_ag import objfun

import logging

logger = logging.getLogger(__name__)

objfun_jit = jit(objfun, static_argnames=["projection_model"]) #TODO backend?

from ..common import (objfun_jacobian_wrapper_sparse as objfun_jacobian_wrapper_sparse_common,
                      objfun_jacobian_wrapper_dense as objfun_jacobian_wrapper_dense_common,
                      make_jacobian_sparsity_mask as make_jacobian_sparsity_mask_common)
from ...ravel import unravel_vars as unravel_vars_base, unpack_calibs


def objfun_runner(landmarks, positions, oris_flu, landmark_imcoords, calibs, opts) -> np.ndarray:
    cam_matrices, ks, xis, projection_models, cam_Rs, cam_ts = unpack_calibs(calibs)
    logger.log(logging.INFO, "Unpacking Calibs")
    assert np.all(projection_models == projection_models[0]), "All cameras must have the same projection model"
    errors = np.array(objfun_jit(landmarks, positions, oris_flu, R.from_matrix(cam_Rs).as_rotvec(), cam_ts,
                                  landmark_imcoords,
                                  1,  # TODO: Introduce landmark weights
                                  {"mask_weight": 1,  # TODO: Introduce mask weights
                                   "error_cap": np.inf},  # Report true, uncapped errors
                                  cam_matrices, ks, xis, projection_models[0],
                                  opts["lm_close_pen_weight"]))
    logger.log(logging.INFO, "Computed Objfun")

    return errors


def objfun_wrapper(free_vars, args, no_ravel=False) -> np.ndarray:
    obfun_inputs = unravel_vars(free_vars, args)
    errors = np.array(objfun_jit(*obfun_inputs))

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
    cam_matrices, ks, xis, projection_models = unpack_calibs(args["calibs"])[:4]
    projection_model = projection_models[0]
    assert np.all(projection_models == projection_model), "All cameras must have the same projection model"
    lm_close_pen_weight = args["lm_close_pen_weight"]
    lm_weights = args["landmarks_weights"]
    weight_mask = args["weight_mask"]
    weight = args["weight"]
    mask_weight = np.ones_like(weight_mask)
    mask_weight[weight_mask] = weight
    # Per residual inputs, (m, k, n, 1) each. Full shape such that the sparse jacobian wrapper can slice them.
    residual_args = {
        "mask_weight": mask_weight,
        "error_cap": np.broadcast_to(np.asarray(args.get("repro_error_cap", np.inf), dtype=float), mask_weight.shape),
    }
    return (landmarks, positions, oris_flu, cam_rotvecs, cam_ts, landmark_imcoords, lm_weights, residual_args,
            cam_matrices, ks, xis, projection_model, lm_close_pen_weight)


def update_error_cap(free_vars, args, cap_min, cap_factor):
    # Set args["repro_error_cap"] per frame to max(cap_min, cap_factor * median landmark error) at free_vars.
    # The cap is computed outside the objective function and kept fixed during one least_squares run, such that the
    # objective stays consistent with its jacobian. If all landmarks of a frame are off, the cap is large and
    # effectively inactive, if only single landmarks are off, they are capped.
    if cap_factor is None or cap_factor <= 0:
        args["repro_error_cap"] = np.inf
        return args

    # Raw pixel errors: no cap, no weights, no penalty
    objfun_inputs = list(unravel_vars(free_vars, args))
    objfun_inputs[6] = np.ones_like(objfun_inputs[6])  # lm_weights
    objfun_inputs[7] = {"mask_weight": 1, "error_cap": np.inf}  # residual_args
    objfun_inputs[-1] = 0  # lm_close_pen_weight
    errors = np.array(objfun_jit(*objfun_inputs))

    norms = np.linalg.norm(errors, axis=-1)  # (m, k, n), NaN for missing labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # All-NaN frames
        median = np.nanmedian(norms, axis=(0, 2), keepdims=True)  # Per frame, over cams and landmarks
    cap = np.fmax(cap_min, cap_factor * median)  # NaN median -> cap_min
    args["repro_error_cap"] = np.broadcast_to(cap[..., np.newaxis], errors.shape[:-1] + (1,))
    return args


jacobians_repro = None


def get_jacobians(opts):
    global jacobians_repro
    if jacobians_repro is None:
        if "jax_backend" not in opts:
            opts["jax_backend"] = "cpu"
            print("Set JAX backend to CPU")
        jacobians_repro = [jit(jacobian(objfun, i_var), static_argnames=["projection_model"], backend=opts["jax_backend"]) for i_var in range(5)]
    return jacobians_repro
