import importlib
import os
import sys
import queue
import threading
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
from bboVisO.ravel import unravel_vars

from calibcamlib.camerasystem import strip_calibs

debug = False
# single precision is insufficient for numerica gradients  # TODO: Make optional
jax.config.update("jax_enable_x64", True)


def infer_position_attitude(times, calibs, opts,
                            landmark_imcoords=None, landmark_vecs=None,
                            landmarks_init=None, positions_init=None, oris_flu_init=None,
                            landmarks_optimize=None, landmarks_far=None, landmarks_weights=None,
                            imcoords_interp_mask=None):
    # Remove unnecessary stuff from calibs
    calibs = strip_calibs(calibs)

    # Fill landmark_imcoords from landmark_vecs or vice-versa
    landmark_imcoords, landmark_vecs = fill_landmark_imcoords_vecs(landmark_imcoords, landmark_vecs, calibs)

    if positions_init is None:
        positions_init = np.full((landmark_imcoords.shape[1], 3), np.nan)
    if oris_flu_init is None:
        oris_flu_init = np.full((landmark_imcoords.shape[1], 3), np.nan)

    if landmarks_far is None:
        landmarks_far = np.ones(landmarks_init.shape[0], dtype=bool)
    if landmarks_optimize is None:
        landmarks_optimize = np.zeros(landmarks_init.shape[0], dtype=bool)

    if landmarks_weights is None:
        landmarks_weights = np.ones(shape=(1, 1, *landmarks_init.shape[0]))
    else:
        landmarks_weights = np.sqrt(landmarks_weights)
        if landmarks_weights.ndim == 1:
            landmarks_weights = landmarks_weights[np.newaxis, np.newaxis, :] #Backward compatibility for only single weight per frame and camera

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
            print("1 ", np.sum(np.isnan(positions_init)), np.sum(np.isnan(oris_flu_init)), np.sum(np.isnan(landmarks_init)),
                  np.sum(np.isinf(positions_init)), np.sum(np.isinf(oris_flu_init)), np.sum(np.isinf(landmarks_init)))
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
            n_slices = int(np.ceil(len(positions_init) / iter_opts["frame_batch_size"]))
            frame_slice_generator = (slice(int(b_idx * batch_size), int((b_idx + 1) * batch_size), int(frame_step))
                                     for b_idx in range(n_slices))

            positions_prev = positions_init.copy()  # overall positions
            cost = np.full((n_slices, 2), np.nan)
            print("2 ", np.count_nonzero(np.isnan(positions_init)), np.count_nonzero(np.isnan(oris_flu_init)),
                  np.count_nonzero(np.isnan(landmarks_init)),
                  np.count_nonzero(np.isinf(positions_init)), np.count_nonzero(np.isinf(oris_flu_init)), np.count_nonzero(np.isinf(landmarks_init)))

            # if iter_opts["parallelize"]:
            #     print("Overwrote frame_slice_generator")
            #     frame_range = range(13400, 13500)
            #     n_slices = len(list(frame_range))
            #     frame_slice_generator = (slice(int(b_idx * batch_size), int((b_idx + 1) * batch_size), int(frame_step))
            #                              for b_idx in frame_range)
            #     iter_opts["parallelize"] = False
            # iter_opts["parallelize"] = False

            partial_optimize_frame_slice = partial(optimize_frame_slice,
                                                   calibs=calibs,
                                                   iter_opts=iter_opts,
                                                   landmarks_init=landmarks_init,
                                                   optimize=opt,
                                                   landmarks_optimize=landmarks_optimize,
                                                   landmarks_far=landmarks_far,
                                                   obj_func=method.objfun_wrapper)

            def get_sliced_args(frame_slice):
                return (positions_init[frame_slice],
                       oris_flu_init[frame_slice],
                       landmark_imcoords[:, frame_slice],
                       landmark_vecs[:, frame_slice],
                       None if landmarks_weights is None else landmarks_weights[:, [0] if landmarks_weights.shape[1] == 1 else frame_slice],
                       imcoords_interp_mask[:, frame_slice] if imcoords_interp_mask is not None else None,)

            if iter_opts["parallelize"] and n_slices > 1:
                if not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled():
                    with (concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor):
                        futures = {executor.submit(
                            partial_optimize_frame_slice, get_sliced_args(frame_slice)
                        ): frame_slice for frame_slice in frame_slice_generator}

                        for future in tqdm(concurrent.futures.as_completed(futures), total=n_slices):
                            frame_slice = futures[future]
                            slice_result, slice_cost = future.result()

                            update_slices_inplace(frame_slice, slice_result, slice_cost, landmarks_init,
                                                  positions_init, oris_flu_init, calibs, cost, opt)
                else:
                    max_workers = os.cpu_count() // 2
                    work_queue = queue.Queue(maxsize=max_workers)
                    lock = threading.Lock()
                    def worker():
                        while True:
                            frame_slice = work_queue.get()
                            try:
                                if frame_slice is None:
                                    return
                                sliced_args = get_sliced_args(frame_slice)
                                slice_result, slice_cost = partial_optimize_frame_slice(sliced_args)

                                with lock:
                                    update_slices_inplace(frame_slice, slice_result, slice_cost, landmarks_init,
                                        positions_init, oris_flu_init, calibs, cost,opt,
                                    )
                            finally:
                                work_queue.task_done()

                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        workers = [executor.submit(worker) for _ in range(max_workers)]
                        for frame_slice in tqdm(frame_slice_generator, total=n_slices):
                            work_queue.put(frame_slice)
                        for _ in range(max_workers):
                            work_queue.put(None)
                        work_queue.join()
                        for future in workers:
                            future.result()
            else:
                for frame_slice in tqdm(frame_slice_generator, total=n_slices):
                    slice_result, slice_cost = partial_optimize_frame_slice(get_sliced_args(frame_slice))

                    update_slices_inplace(frame_slice, slice_result, slice_cost, landmarks_init,
                                          positions_init, oris_flu_init, calibs, cost, opt)

            # Normalize e.g. too long vectors from multiple revolutions
            oris_flu_init = R.from_rotvec(oris_flu_init).as_rotvec()

            # Calculate and display position shift
            max_pos_shift = np.nanmax(np.linalg.norm((positions_prev - positions_init), axis=-1))
            print(f"Max pos shift {max_pos_shift:.6f}")

            old_cost, new_cost = np.nansum(np.array(cost[:, 0])), np.nansum(np.array(cost[:, 1]))
            print(f"Cost reduced from {old_cost:.2f} to {new_cost:.2f} ({new_cost / old_cost * 100:.2f}%)")

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
                         landmarks_optimize, landmarks_far, obj_func):
    positions_init, oris_flu_init, landmark_imcoords, landmark_vecs, landmarks_weights, imcoords_interp_mask = sliced_inputs
    vars_free, objfun_args, optim_args = (
        make_optim_input(calibs, iter_opts,
                         landmarks_init,
                         positions_init,  # [frame_slice]
                         oris_flu_init,  # [frame_slice]
                         optimize=optimize,
                         landmarks_optimize=landmarks_optimize,
                         landmarks_far=landmarks_far,
                         landmarks_weights=landmarks_weights,
                         weight_mask=imcoords_interp_mask,
                         weight=iter_opts["interpolated_weight"],
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
    error = obj_func(vars_free, objfun_args, no_ravel=True)
    result = unravel_vars(min_result.x, objfun_args["ravel_args"])

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
