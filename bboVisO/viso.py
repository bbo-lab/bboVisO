import importlib
import sys
import time

import jax
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, OptimizeResult
import concurrent.futures
from functools import partial

from bboVisO.helper import realign_by_distant_features, update_calibs_extrinsics
from bboVisO.initialization import make_optim_input, fill_landmark_imcoords_vecs
from bboVisO.interpolation import interpolate_high_error, interpolate_close_marker
from bboVisO.unravel import unravel_vars

debug = False
# single precision is insufficient for numerica gradients  # TODO: Make optional
jax.config.update("jax_enable_x64", True)


def infer_position_attitude(times, calibs, opts,
                            landmark_imcoords=None, landmark_vecs=None,
                            landmarks_init=None, positions_init=None, oris_flu_init=None,
                            landmarks_optimize=None, landmarks_far=None, landmarks_weights=None):
    # Fill landmark_imcoords from landmark_vecs or vice-versa
    landmark_imcoords, landmark_vecs = fill_landmark_imcoords_vecs(landmark_imcoords, landmark_vecs, calibs)

    if positions_init is None:
        positions_init = np.full((landmark_imcoords.shape[1]), np.nan)
    if oris_flu_init is None:
        oris_flu_init = np.full((landmark_imcoords.shape[1]), np.nan)

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
                    positions_init is not None and oris_flu_init is not None):
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

            if iter_opts["optimize_single_frames"]:
                assert set(opt).issubset(["positions", "oris"]), \
                    "Single frames can only be optimized for position and orientation"
                print(f"Optimizing {opt} - Single frames")
                frame_slice_generator = (slice(fr, fr + 1) for fr in range(landmark_imcoords.shape[1]))
                n_slices = landmark_imcoords.shape[1]
            else:
                print(f"Optimizing {opt} - Joined")
                frame_slice_generator = (slice(None))
                n_slices = 1

            frame_slice_prev = None
            positions_prev = positions_init  # overall positions
            cost = np.full((n_slices, 2), np.nan)

            # Timing output
            # vars_free, objfun_args, optim_args = (
            #     make_optim_input(calibs, iter_opts,
            #                      landmarks_init,
            #                      positions_init[frame_slice],
            #                      oris_flu_init[frame_slice],
            #                      optimize=opt,
            #                      landmarks_optimize=landmarks_optimize,
            #                      landmarks_far=landmarks_far,
            #                      landmarks_weights=landmarks_weights,
            #                      landmark_imcoords=landmark_imcoords[:, frame_slice],
            #                      landmark_vecs=landmark_vecs[:, frame_slice]))
            #
            # t_objfun = time.time()
            # x = method.objfun_wrapper(vars_free, objfun_args)
            # print(f"Objective function evaluation took {time.time() - t_objfun}s, value: {np.sum(x ** 2)}")
            # time_jacobian(iter_opts["optimization"]["jac"], vars_free, objfun_args)

            def optimize_frame_slice(my_slice):
                return my_slice, 1, 1

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(optimize_frame_slice, frame_slice):
                               frame_slice for frame_slice in frame_slice_generator}

                for future in concurrent.futures.as_completed(futures):
                    frame_slice, results = future.result()
                    landmarks_init = results[0]
                    positions_init[frame_slice], oris_flu_init[frame_slice] = results[1:3]
                    cam_rotvecs, cam_ts = results[3:5]

                    calibs = update_calibs_extrinsics(calibs, cam_rotvecs, cam_ts)

            for i_frame_slice, frame_slice in enumerate(frame_slice_generator):  # TODO parallelize
                vars_free, objfun_args, optim_args = (
                    make_optim_input(calibs, iter_opts,
                                     landmarks_init,
                                     positions_init[frame_slice],
                                     oris_flu_init[frame_slice],
                                     optimize=opt,
                                     landmarks_optimize=landmarks_optimize,
                                     landmarks_far=landmarks_far,
                                     landmarks_weights=landmarks_weights,
                                     landmark_imcoords=landmark_imcoords[:, frame_slice],
                                     landmark_vecs=landmark_vecs[:, frame_slice]))

                error = method.objfun_wrapper(vars_free, objfun_args)
                cost_pre = np.sum(error ** 2) / 2

                min_result: OptimizeResult = least_squares(method.objfun_wrapper,
                                                           vars_free,
                                                           args=[objfun_args],
                                                           **optim_args,
                                                           **iter_opts["optimization"]
                                                           )
                cost[i_frame_slice] = np.array([cost_pre, np.sum(method.objfun_wrapper(min_result.x, objfun_args) ** 2) / 2])

                results = unravel_vars(min_result.x, objfun_args)
                landmarks_init = results[0]
                positions_init[frame_slice], oris_flu_init[frame_slice] = results[1:3]
                cam_rotvecs, cam_ts = results[3:5]

                calibs = update_calibs_extrinsics(calibs, cam_rotvecs, cam_ts)

                frame_slice_prev = frame_slice

                print(".", end="")
                sys.stdout.flush()
            print()

            # Normalize e.g. too long vectors from multiple revolutions
            oris_flu_init = R.from_rotvec(oris_flu_init).as_rotvec()

            # Calculate and display position shift
            max_pos_shift = np.nanmax(np.linalg.norm((positions_prev - positions_init), axis=-1))
            print(f"Max pos shift {max_pos_shift}")

            print("Cost reduced from", np.round(np.sum(np.array(cost[-1][0])), 2), "to",
                  np.round(np.sum(np.array(cost[-1][1])), 2))

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
