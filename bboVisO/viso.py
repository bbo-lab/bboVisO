import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, OptimizeResult

from bboVisO.helper import realign_by_distant_features, update_calibs_extrinsics
from bboVisO.initialization import make_optim_input, fill_landmark_imcoords_vecs, get_precalc
from bboVisO.interpolation import interpolate_high_error, interpolate_close_marker
from bboVisO.method_switches import objective_function
from bboVisO.unravel import unravel_vars

debug = False

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

    # Get precalculations from lazy storage
    precalc = get_precalc(opts)

    # Iterate through optimization types
    for i_iter, iter_opts in enumerate(opts["iterations"]):
        # Iterate through opimization steps
        for i_opti, opt in enumerate(iter_opts["optimize"]):
            if (iter_opts["interpolate_high_error"] and iter_opts["interpolate_high_error"] != -1
                and positions_init is not None) and oris_flu_init is not None:
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

            # Run optimization
            print(f"Optimizing {opt}")
            vars_free, objfun_args, optim_args = make_optim_input(calibs, iter_opts,
                                                                  landmarks_init, positions_init, oris_flu_init,
                                                                  optimize=opt, landmarks_optimize=landmarks_optimize,
                                                                  landmarks_far=landmarks_far,
                                                                  precalc=precalc,
                                                                  landmark_imcoords=landmark_imcoords,
                                                                  landmark_vecs=landmark_vecs)

            error = objective_function(iter_opts)(vars_free, objfun_args, no_ravel=True)
            max_errors = np.nanmax(error, axis=tuple(range(1, len(error.shape))))
            error_mask = max_errors >= iter_opts["max_error_median_factor"] * np.nanmedian(max_errors)
            if positions_init is not None:
                positions_init_iter = positions_init[error_mask]
            else:
                positions_init_iter = None
                positions_init = np.full((len(error_mask), 3), np.nan)
            if oris_flu_init is not None:
                oris_flu_init_iter = oris_flu_init[error_mask]
            else:
                oris_flu_init_iter = None
                oris_flu_init = np.full((len(error_mask), 3), np.nan)
            vars_free, objfun_args, optim_args = make_optim_input(calibs, iter_opts,
                                                                  landmarks_init,
                                                                  positions_init_iter, oris_flu_init_iter,
                                                                  optimize=opt, landmarks_optimize=landmarks_optimize,
                                                                  landmarks_far=landmarks_far,
                                                                  precalc=precalc,
                                                                  landmark_imcoords=landmark_imcoords[:, error_mask],
                                                                  landmark_vecs=landmark_vecs[:, error_mask])

            min_result: OptimizeResult = least_squares(objective_function(iter_opts),
                                                       vars_free,
                                                       args=[objfun_args],
                                                       **optim_args,
                                                       **iter_opts["optimization"]
                                                       )

            (landmarks_init, positions_init[error_mask], oris_flu_init[error_mask],
             cam_rotations, camera_positions) = unravel_vars(min_result.x, objfun_args)

            calibs = update_calibs_extrinsics(calibs, camera_positions, cam_rotations)

            # Normalize e.g. too long vectors from multiple revolutions
            oris_flu_init = R.from_rotvec(oris_flu_init).as_rotvec()
            max_pos_shift = np.nanmax(np.linalg.norm((positions_prev - positions_init), axis=-1))
            positions_prev = positions_init

            print(f"Max pos shift {max_pos_shift}")

        if iter_opts["optimize_single_frames"]:
            print(f"Optimizing single frames")
            for i_fr in range(len(positions_init)):
                # Run optimization
                vars_free, objfun_args, optim_args = make_optim_input(calibs, iter_opts,
                                                                      landmarks_init,
                                                                      positions_init[(i_fr,),], oris_flu_init[(i_fr,),],
                                                                      optimize=["positions", "oris"],
                                                                      landmarks_optimize=landmarks_optimize,
                                                                      landmarks_far=landmarks_far,
                                                                      precalc=precalc,
                                                                      landmark_imcoords=landmark_imcoords[:, (i_fr,), ],
                                                                      landmark_vecs=landmark_vecs[:, (i_fr,), ])

                min_result: OptimizeResult = least_squares(objective_function(iter_opts),
                                                           vars_free,
                                                           args=[objfun_args],
                                                           **optim_args,
                                                           **iter_opts["optimization"]
                                                           )

                landmarks_init, positions_init[i_fr], oris_flu_init[i_fr] = unravel_vars(min_result.x, objfun_args)[:3]
                # Normalize e.g. too long vectors from multiple revolutions

            oris_flu_init = R.from_rotvec(oris_flu_init).as_rotvec()
            max_pos_shift = np.nanmax(np.linalg.norm((positions_prev - positions_init), axis=-1))

            print(f"Max pos shift {max_pos_shift}")

        if iter_opts["realign_by_distant_features"]:
            print("realign_by_distant_features")
            oris_flu_init = realign_by_distant_features(landmarks_init, landmark_imcoords, times,
                                                        positions_init, R.from_rotvec(oris_flu_init),
                                                        calibs).as_rotvec()

    print(f"Optimization took {time.time() - t} s")

    return landmarks_init, positions_init, oris_flu_init, calibs


