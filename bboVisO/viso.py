import importlib
import sys
import time

import jax
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, OptimizeResult

import bboVisO
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
                            landmarks_optimize=None, landmarks_far=None):
    # Fill landmark_imcoords from landmark_vecs or vice-versa
    landmark_imcoords, landmark_vecs = fill_landmark_imcoords_vecs(landmark_imcoords, landmark_vecs, calibs)

    # Previous positions for alternating update
    positions_prev = np.full((landmark_imcoords.shape[1], 3), np.nan)

    # Timing
    t = time.time()  #

    # Iterate through optimization types
    for i_iter, iter_opts in enumerate(opts["iterations"]):
        method = importlib.import_module("bboVisO.methods." + iter_opts['objective_function'] + ".optfuncs")
        # Adjust options
        if iter_opts["optimization"]["jac"] == "jax":
            iter_opts["optimization"]["jac"] = method.objfun_jacobian_wrapper
            iter_opts["sparse_jacobian"] = False  # We do not need to calculate a sparsity matrix for this
        elif iter_opts["optimization"]["jac"] == "jax_dense":
            iter_opts["optimization"]["jac"] = method.objfun_jacobian_wrapper_dense
            iter_opts["sparse_jacobian"] = False  # We do not need to calculate a sparsity matrix for this

        # Iterate through opimization steps
        for i_opti, opt in enumerate(iter_opts["optimize"]):
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

            if iter_opts["interpolate_close_marker"] and iter_opts["interpolate_close_marker"] != -1 and i_opti > 0:
                # Positioning the camera on a landmark might reduce the reprojection error, but is obviously wrong.
                # This tries to reinitialize from an interpolated position
                positions_init, oris_flu_init = interpolate_close_marker(times, positions_init, oris_flu_init,
                                                                         opts=iter_opts,
                                                                         landmarks=landmarks_init)
            cost = []
            cost_pre = []
            if iter_opts["optimize_single_frames"]:
                print(f"Optimizing {opt} - Single frames")
                assert set(opt).issubset(["positions", "oris"]), ("Single frames can only be optimized for position "
                                                                  "and orientation")

                if positions_init is None:
                    positions_init = np.full((landmark_imcoords.shape[1], 3), np.nan)
                if oris_flu_init is None:
                    oris_flu_init = np.full((landmark_imcoords.shape[1], 3), np.nan)

                for i_fr in range(landmark_imcoords.shape[1]):  # TODO paralellize
                    # Run optimization
                    if positions_init[i_fr] is None or np.any:
                        p_init = None
                    else:
                        p_init = positions_init[(i_fr,),]
                    if oris_flu_init[i_fr] is None:
                        o_init = None
                    else:
                        o_init = oris_flu_init[(i_fr,),]

                    if iter_opts["optimize_single_init_prev"] and i_fr > 0:
                        p_init = positions_init[i_fr-1]

                    vars_free, objfun_args, optim_args = make_optim_input(calibs, iter_opts,
                                                                          landmarks_init,
                                                                          p_init,
                                                                          o_init,
                                                                          optimize=opt,
                                                                          landmarks_optimize=landmarks_optimize,
                                                                          landmarks_far=landmarks_far,
                                                                          landmark_imcoords=landmark_imcoords[:,
                                                                                            (i_fr,), ],
                                                                          landmark_vecs=landmark_vecs[:, (i_fr,), ])

                    error = method.objfun_wrapper(vars_free, objfun_args)
                    cost_pre.append(np.sum(error ** 2) / 2)
                    if i_fr == 0:
                        t_objfun = time.time()
                        method.objfun_wrapper(vars_free, objfun_args)
                        print(f"Objective function evaluation took {time.time() - t_objfun}s")

                        if not isinstance(iter_opts["optimization"]["jac"], str):
                            iter_opts["optimization"]["jac"](vars_free, objfun_args)
                            t_jac = time.time()
                            jacobian = iter_opts["optimization"]["jac"](vars_free, objfun_args)
                            print(f"Jacobian evaluation took {time.time() - t_jac}s")

                    min_result: OptimizeResult = least_squares(method.objfun_wrapper,
                                                               vars_free,
                                                               args=[objfun_args],
                                                               **optim_args,
                                                               **iter_opts["optimization"]
                                                               )
                    cost.append(np.sum(method.objfun_wrapper(min_result.x, objfun_args) ** 2) / 2)
                    positions_init[i_fr], oris_flu_init[i_fr] = unravel_vars(min_result.x, objfun_args)[1:3]
                    print(".", end="")
                    sys.stdout.flush()
                print()
            else:
                print(f"Optimizing {opt} - Joined")
                vars_free, objfun_args, optim_args = make_optim_input(calibs, iter_opts,
                                                                      landmarks_init, positions_init, oris_flu_init,
                                                                      optimize=opt,
                                                                      landmarks_optimize=landmarks_optimize,
                                                                      landmarks_far=landmarks_far,
                                                                      landmark_imcoords=landmark_imcoords,
                                                                      landmark_vecs=landmark_vecs)

                error = method.objfun_wrapper(vars_free, objfun_args)
                t_objfun = time.time()
                method.objfun_wrapper(vars_free, objfun_args)
                print(f"Objective function evaluation took {time.time() - t_objfun}s")
                cost_pre = np.sum(error ** 2) / 2

                if not isinstance(iter_opts["optimization"]["jac"], str):
                    iter_opts["optimization"]["jac"](vars_free, objfun_args)
                    t_jac = time.time()
                    jacobian = iter_opts["optimization"]["jac"](vars_free, objfun_args)
                    print(f"Jacobian evaluation took {time.time() - t_jac}s")

                    # import matplotlib
                    # import matplotlib.pyplot as plt
                    # sparsity = method.make_jacobian_sparsity_mask(*landmark_imcoords.shape[:3], objfun_args["mask_opt"])
                    # matplotlib.use('Agg')
                    # fig, axes = plt.subplots(1,3)
                    # axes[0].imshow(jacobian > 0)
                    # axes[1].imshow(sparsity)
                    # axes[2].imshow(~sparsity & (jacobian > 0))
                    # print("JACOBIAN: ", jacobian.shape)
                    # fig.savefig(f"/home/voit/tmp/jacobian_{i_iter}_{i_opti}.svg")
                    # plt.close()

                min_result: OptimizeResult = least_squares(method.objfun_wrapper,
                                                           vars_free,
                                                           args=[objfun_args],
                                                           **optim_args,
                                                           **iter_opts["optimization"]
                                                           )
                cost = np.sum(method.objfun_wrapper(min_result.x, objfun_args) ** 2) / 2

                landmarks_init, positions_init, oris_flu_init, cam_rotvecs, cam_ts = unravel_vars(min_result.x,
                                                                                                  objfun_args)
                calibs = update_calibs_extrinsics(calibs, cam_rotvecs, cam_ts)

            # Normalize e.g. too long vectors from multiple revolutions
            oris_flu_init = R.from_rotvec(oris_flu_init).as_rotvec()

            # Calculate and display position shift
            max_pos_shift = np.nanmax(np.linalg.norm((positions_prev - positions_init), axis=-1))
            positions_prev = positions_init
            print(f"Max pos shift {max_pos_shift}")

            print("Cost reduced from", np.round(np.sum(np.array(cost_pre)), 2), "to", np.round(np.sum(np.array(cost)), 2))

    if iter_opts["realign_by_distant_features"]:
        print("realign_by_distant_features")
        oris_flu_init = realign_by_distant_features(landmarks_init, landmark_imcoords, times,
                                                    positions_init, R.from_rotvec(oris_flu_init),
                                                    calibs).as_rotvec()

    print(f"Optimization took {time.time() - t} s, ")

    return landmarks_init, positions_init, oris_flu_init, calibs
