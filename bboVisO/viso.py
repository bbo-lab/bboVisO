import importlib
import os
import sys
# import sys
import time

import jax
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, OptimizeResult
import concurrent.futures
from functools import partial

from tqdm import tqdm

from bboVisO.helper import realign_by_distant_features, update_calibs_extrinsics
from bboVisO.initialization import make_optim_input, fill_landmark_imcoords_vecs
from bboVisO.interpolation import interpolate_high_error, interpolate_close_marker
from bboVisO.unravel import unravel_vars

from calibcamlib.camerasystem import strip_calibs

debug = False
# single precision is insufficient for numerica gradients  # TODO: Make optional
jax.config.update("jax_enable_x64", True)


def infer_position_attitude(times, calibs, opts,
                            landmark_imcoords=None, landmark_vecs=None,
                            landmarks_init=None, positions_init=None, oris_flu_init=None,
                            landmarks_optimize=None, landmarks_far=None, landmarks_weights=None):
    # Remove unnecessary stuff from calibs
    calibs = strip_calibs(calibs)

    # Fill landmark_imcoords from landmark_vecs or vice-versa
    landmark_imcoords, landmark_vecs = fill_landmark_imcoords_vecs(landmark_imcoords, landmark_vecs, calibs)

    if positions_init is None:
        positions_init = np.full((landmark_imcoords.shape[1], 3), np.nan)
    if oris_flu_init is None:
        oris_flu_init = np.full((landmark_imcoords.shape[1], 3), np.nan)

    # Timing
    t = time.time()

    # Iterate through optimization types
    for i_iter, iter_opts in enumerate(opts["iterations"]):
        method = importlib.import_module("bboVisO.methods." + iter_opts['objective_function'] + ".optfuncs")

        # Adjust options
        if iter_opts["optimization"]["jac"] == "jax":
            iter_opts["optimization"]["jac"] = method.objfun_jacobian_wrapper
            iter_opts["sparse_jacobian"] = False  # We do not need to calculate a sparsity matrix for this
        elif iter_opts["optimization"]["jac"] == "jax_dense":
            iter_opts["optimization"]["jac"] = method.objfun_jacobian_wrapper_dense
            iter_opts["sparse_jacobian"] = True  #

        # Iterate through optimization steps
        for i_opti, opt in enumerate(iter_opts["optimize"]):
            # Interpolate high errors
            if (iter_opts["interpolate_high_error"] and iter_opts["interpolate_high_error"] != -1 and
                    not np.all(np.isnan(positions_init)) and not np.all(np.isnan(oris_flu_init))):
                print(f"Interpolating high errors")
                # This might help to get the system out of local minima in case of wrong markings.
                # The last step is always an optimization, though. The check_gui also implements this
                # interpolation such that it can be used for checking errors.
                positions_init, oris_flu_init = interpolate_high_error(times, positions_init, oris_flu_init, calibs,
                                                                       iter_opts,
                                                                       landmarks=landmarks_init,
                                                                       optimize=["positions", "oris"],
                                                                       landmark_vecs=landmark_vecs,
                                                                       landmark_imcoords=landmark_imcoords,
                                                                       landmarks_optimize=landmarks_optimize)

            # Interpolate close markers
            if iter_opts["interpolate_close_marker"] and iter_opts["interpolate_close_marker"] != -1 and i_opti > 0:
                # Positioning the camera on a landmark might reduce the reprojection error, but is obviously wrong.
                # This tries to reinitialize from an interpolated position
                positions_init, oris_flu_init = interpolate_close_marker(times, positions_init, oris_flu_init,
                                                                         opts=iter_opts,
                                                                         landmarks=landmarks_init)

            assert iter_opts["frame_batch_size"] > 1 or set(opt).issubset(["positions", "oris"]), \
                "Single frames can only be optimized for position and orientation"
            batch_size = iter_opts["frame_batch_size"]
            frame_step = iter_opts["frame_batch_step"]
            print(f'Optimizing {opt} - Batch size {batch_size} - Step {frame_step}')
            n_slices = int(np.ceil(len(positions_init)/iter_opts["frame_batch_size"]))
            frame_slice_generator = (slice(int(b_idx*batch_size), int((b_idx + 1)*batch_size), int(frame_step))
                                     for b_idx in range(n_slices))

            positions_prev = positions_init.copy()  # overall positions
            cost = np.full((n_slices, 2), np.nan)

            partial_optimize_frame_slice = partial(optimize_frame_slice,
                                                   calibs=calibs,
                                                   iter_opts=iter_opts,
                                                   landmarks_init=landmarks_init,
                                                   optimize=opt,
                                                   landmarks_optimize=landmarks_optimize,
                                                   landmarks_far=landmarks_far,
                                                   landmarks_weights=landmarks_weights,
                                                   obj_func=method.objfun_wrapper)

            if iter_opts["parallelize"]:
                with (concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor):
                    futures = {executor.submit(partial_optimize_frame_slice, (positions_init[frame_slice],
                                                                              oris_flu_init[frame_slice],
                                                                              landmark_imcoords[:, frame_slice],
                                                                              landmark_vecs[:,
                                                                              frame_slice])): frame_slice
                               for frame_slice in frame_slice_generator}

                    with tqdm(total=n_slices) as pbar:
                        for future in concurrent.futures.as_completed(futures):
                            frame_slice = futures[future]
                            slice_result, slice_cost = future.result()

                            update_slices_inplace(frame_slice, slice_result, slice_cost, landmarks_init,
                                                  positions_init, oris_flu_init, calibs, cost, opt)

                            pbar.update(1)

            else:
                with tqdm(total=n_slices) as pbar:
                    for frame_slice in frame_slice_generator:
                        slice_result, slice_cost = partial_optimize_frame_slice((positions_init[frame_slice],
                                                                                 oris_flu_init[frame_slice],
                                                                                 landmark_imcoords[:, frame_slice],
                                                                                 landmark_vecs[:, frame_slice]))

                        update_slices_inplace(frame_slice, slice_result, slice_cost, landmarks_init,
                                              positions_init, oris_flu_init, calibs, cost, opt)

                        pbar.update(1)

            # Normalize e.g. too long vectors from multiple revolutions
            oris_flu_init = R.from_rotvec(oris_flu_init).as_rotvec()

            # Calculate and display position shift
            max_pos_shift = np.nanmax(np.linalg.norm((positions_prev - positions_init), axis=-1))
            print(f"Max pos shift {max_pos_shift:.6f}")

            print(f"Cost reduced from {np.nansum(np.array(cost[:, 0])):.2f} to {np.nansum(np.array(cost[:, 1])):.2f}")

    if opts["iterations"][-1]["realign_by_distant_features"]:
        print("realign_by_distant_features")
        oris_flu_init = realign_by_distant_features(landmarks_init, landmark_imcoords, times,
                                                    positions_init, R.from_rotvec(oris_flu_init),
                                                    calibs).as_rotvec()

    print(f"Optimization took {time.time() - t} s, ")

    return landmarks_init, positions_init, oris_flu_init, calibs


def time_jacobian(jac, vars_free, objfun_args):
    if not isinstance(jac, str):
        jac(vars_free, objfun_args)
        t_jac = time.time()
        jac(vars_free, objfun_args)
        print(f"Jacobian evaluation took {time.time() - t_jac}s")


def optimize_frame_slice(sliced_inputs, calibs, iter_opts, landmarks_init, optimize,
                         landmarks_optimize, landmarks_far, landmarks_weights,
                         obj_func):
    positions_init, oris_flu_init, landmark_imcoords, landmark_vecs = sliced_inputs
    vars_free, objfun_args, optim_args = (
        make_optim_input(calibs, iter_opts,
                         landmarks_init,
                         positions_init,  # [frame_slice]
                         oris_flu_init,  # [frame_slice]
                         optimize=optimize,
                         landmarks_optimize=landmarks_optimize,
                         landmarks_far=landmarks_far,
                         landmarks_weights=landmarks_weights,
                         landmark_imcoords=landmark_imcoords,  # [:, frame_slice]
                         landmark_vecs=landmark_vecs))  # [:, frame_slice]

    error = obj_func(vars_free, objfun_args)
    cost_pre = np.sum(error ** 2) / 2

    min_result: OptimizeResult = least_squares(obj_func,
                                               vars_free,
                                               args=[objfun_args],
                                               **optim_args,
                                               **iter_opts["optimization"]
                                               )
    cost = np.array([cost_pre, np.sum(obj_func(min_result.x, objfun_args) ** 2) / 2])
    result = unravel_vars(min_result.x, objfun_args)

    return result, cost


def update_slices_inplace(frame_slice, slice_result, slice_cost,
                          landmarks_init, positions_init, oris_flu_init, calibs, cost,
                          opt):
    # Fill results into full arrays
    cost[frame_slice, :] = slice_cost
    if "landmarks" in opt:
        landmarks_init[:] = slice_result[0]
    if "positions" in opt:
        positions_init[frame_slice] = slice_result[1]
    if "oris" in opt:
        oris_flu_init[frame_slice] = slice_result[2]
    if "extrinsics_r" in opt or "extrinsics_t" in opt:
        update_calibs_extrinsics(calibs, slice_result[3], slice_result[4])
