import time

import numpy as np
import bboVisO.viso_ag as viso_ag
from jax import jit, jacfwd as jacobian  # jacfwd is recommended for 'tall' jacobians_repro, jacrev for 'wide'
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, OptimizeResult
from calibcamlib.camerasystem import Camerasystem
from bboVisO.geometry import align_positions_orientations, build_vector_bundle
from itertools import cycle
from calibcamlib.helper import intersect

debug = False
precalc = None


def infer_position_attitude(landmark_imcoords, calibs, opts_optim, opts,
                            landmarks_init=None, positions_init=None, oris_flu_opt_init=None,
                            optimize=None, landmarks_optimize=None):
    end_optim = False

    # Previous positions for alternating update
    positions_prev = np.full((landmark_imcoords.shape[1], 3), np.nan)

    # Optimize all at once or alternate
    if optimize is None:
        optimize = [["oris", "positions", "landmarks"]]
    # optimize = cycle([["oris", "positions"], "landmarks"])  # Seems to converge to the same value

    t = time.time()
    precalc = get_precalc()
    for opt in optimize:
        # If we are doing a full optimization, increase to full step count steps
        if isinstance(opt, list):
            opts_optim["max_nfev"] = opts["optimization"]["max_nfev"]

        # Run optimization
        print(f"Optimizing {opt}")
        vars_free, objfun_args, optim_args = make_optim_input(calibs, opts,
                                                              landmarks_init, positions_init, oris_flu_opt_init,
                                                              optimize=opt, landmarks_optimize=landmarks_optimize,
                                                              precalc=precalc, landmark_imcoords=landmark_imcoords)

        min_result: OptimizeResult = least_squares(objfun_repro_wrapper,
                                                   vars_free,
                                                   args=[objfun_args],
                                                   **optim_args,
                                                   **opts_optim
                                                   )

        landmarks_init, positions_init, oris_flu_opt_init = unravel_vars(min_result.x, objfun_args)[:3]
        max_pos_shift = np.max(np.abs(positions_prev - positions_init))
        positions_prev = positions_init

        print(f"Max pos shift {max_pos_shift}")

    print(f"Optimization took {time.time() - t} s")

    # Test for 3d optimization in postprocessing
    # vars_free, args = viso.make_optim_input(calibs, opts, landmarks_opt, positions_opt, oris_flu_opt, optimize=opt,
    #                                             precalc=precalc, landmark_imcoords=landmark_imcoords)
    #
    # min_result: OptimizeResult = least_squares(viso.objfun_3d_wrapper,
    #                                            vars_free,
    #                                            jac="2-point",
    #                                            bounds=np.array([[-np.inf, np.inf]] * vars_free.size).T,
    #                                            args=[args],
    #                                            **opts_optim
    #                                            )
    #
    # landmarks_opt, positions_opt, oris_flu_opt = viso.unravel_vars(min_result.x, args)[:3]
    # landmarks_aligned, positions_aligned, oris_flu_aligned = viso.align_results(landmarks,
    #                                                                             landmarks_opt, positions_opt, oris_flu_opt)
    #
    # print(f"Max position error: {np.max(np.linalg.norm(positions-positions_aligned, axis=-1))}")

    return landmarks_init, positions_init, oris_flu_opt_init


def align_results_by_landmarks(landmarks_real, landmarks, positions, oris_flu):
    positions_aligned, orientations_aligned = align_positions_orientations(landmarks_real, landmarks,
                                                                           [landmarks, positions], [oris_flu])
    landmarks = positions_aligned[0]
    positions = positions_aligned[1]
    oris_flu = orientations_aligned[0]
    return landmarks, positions, oris_flu


def align_results_by_positions(positions_real, landmarks, positions, oris_flu):
    positions_aligned, orientations_aligned = align_positions_orientations(positions_real, positions,
                                                                           [landmarks, positions], [oris_flu])
    landmarks = positions_aligned[0]
    positions = positions_aligned[1]
    oris_flu = orientations_aligned[0]
    return landmarks, positions, oris_flu


def align_results_by_perches(landmarks_real, landmarks, positions, oris_flu, fix_indices=(0, 1),
                             dir_indices=None):
    if dir_indices is not None:
        corresp_idxs = fix_indices + dir_indices
        assert len(corresp_idxs) == len(landmarks_real), \
            "fix_indices + dir_indices must correspond to landmarks_real in order"
        landmarks_ref = landmarks[corresp_idxs, ...]
    else:
        landmarks_ref = landmarks

    positions_aligned, orientations_aligned = align_positions_orientations(landmarks_real, landmarks_ref,
                                                                           [landmarks, positions],
                                                                           [oris_flu],
                                                                           fix_indices=fix_indices)
    landmarks = positions_aligned[0]
    positions = positions_aligned[1]
    oris_flu = orientations_aligned[0]
    return landmarks, positions, oris_flu


def objfun_repro_wrapper(free_vars, args, no_ravel=False):
    landmarks, positions, oris_flu, \
        landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts = unravel_vars(free_vars, args)

    errors2d = args['precalc']['objfunc_repro'](landmarks, positions, oris_flu,
                                                landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts)

    if no_ravel:
        return np.nansum(errors2d, axis=0)
    else:
        return np.nansum(errors2d, axis=0).ravel()


def objfun_3d_wrapper(free_vars, args):
    landmarks, positions, oris_flu, \
        landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts = unravel_vars(free_vars, args)

    landmark_vecs = args["landmark_vecs"]

    errors3d = args['precalc']['objfunc_3d'](landmarks, positions, oris_flu, landmark_vecs)

    return errors3d.ravel()


def objfun_repro_jacobian_wrapper(free_vars, args):
    landmarks, positions, oris_flu, \
        landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts = unravel_vars(free_vars, args)

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


def objfun_jacobian_3d_wrapper(free_vars, args):
    landmarks, positions, oris_flu, \
        landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts = unravel_vars(free_vars, args)

    obj_fcn_jacobian = [
        np.array(
            # Ravel over input dimensions
            j(landmarks, positions, oris_flu, landmark_vecs).reshape(landmark_vecs.shape + (-1,))
        )
        for j in args['precalc']['jacobians_3d']
    ]

    # Concatenate along input dimensions
    obj_fcn_jacobian = np.concatenate(
        obj_fcn_jacobian,
        axis=3)

    # Ravel variable dimension
    obj_fcn_jacobian = obj_fcn_jacobian.reshape((-1, args['mask_opt'].size))

    # Dump non-free variables
    obj_fcn_jacobian = obj_fcn_jacobian[:, args['mask_opt']]

    return obj_fcn_jacobian


def make_optim_input(calibs, opts, landmarks=None, positions=None, oris_flu=None,
                     optimize=None, landmarks_optimize=None,
                     precalc=None, landmark_imcoords=None, landmark_vecs=None):
    cs = Camerasystem.from_calibs(calibs)
    if landmark_vecs is None and landmark_imcoords is not None:
        landmark_vecs = np.nansum(cs.get_camera_lines(landmark_imcoords)[0], axis=0)
        landmark_vecs[np.all(landmark_vecs == 0, axis=-1)] = np.nan

    vars_free, vars_full, mask_free_input = \
        make_initialization(landmark_vecs, calibs, opts, landmarks, positions,
                            oris_flu,
                            optimize=optimize, landmarks_optimize=landmarks_optimize)

    k = landmark_vecs.shape[0]
    n = landmark_vecs.shape[1]

    obfun_args = {
        'vars_full': vars_full,  # All possible vars, free vars will be substituted in _free wrapper functions
        'mask_opt': mask_free_input,  # Mask of free vars within all vars
        'landmark_imcoords': landmark_imcoords,
        'landmark_vecs': landmark_vecs,
        'calibs': calibs,
        'precalc': get_precalc() if precalc is None else precalc
    }

    optim_args = {
        'bounds': np.array([[-np.inf, np.inf]] * vars_free.size).T,
        'jac_sparsity': make_jacobian_sparsity_mask(k, n, mask_free_input)
    }

    return vars_free, obfun_args, optim_args


def make_initialization(landmark_vectors, calibs, opts, landmarks=None, positions=None, oris_flu=None,
                        optimize=None, landmarks_optimize=None):
    if optimize is None:
        optimize = ["landmarks", "positions", "oris"]

    k = landmark_vectors.shape[0]
    n = landmark_vectors.shape[1]

    edges = np.array([0,
                      n * 3,
                      n * 3 + k * 3,
                      n * 3 + k * 3 + k * 3
                      ])

    vars_full = np.full(edges[-1], np.nan)
    mask_free_input = np.ones(vars_full.shape, dtype=bool)

    # Init with fixed first orientation and position
    # Init landmarks as extensions of t=0 camera vecs
    # Fix relative x position to landmark 1
    cs = Camerasystem.from_calibs(calibs)
    # cs.cameras[0]["camera"].set_center_pixel(np.array([[960, 512]]))

    # Initializations
    if landmarks is not None:
        landmarks_init = landmarks
    else:
        # Initialize to a distance of 100m into the direction of the  respective camera vectors
        landmarks_init = build_vector_bundle(landmark_vectors[0], landmark_vectors[1:]) * 100

    if positions is not None:
        positions_init = positions
    else:
        # Initialize to coordinate origin
        positions_init = np.zeros((k, 3))
        # perch_mean = np.mean(landmarks[:2], axis=0)
        # for p in positions_init:
        #     p[:] = perch_mean

    if oris_flu is not None:
        oris_flu_init = oris_flu
    else:
        # Initialize to zero
        oris_flu_init = np.zeros((k, 3))

        # Initialize other orientation to landmark vecs
        # TODO: Only optimize on 2 vectors that have the most similar angle as the reference (as proxy for "far away")
        # TODO: Move the reference iteratively
        for i in range(k):
            frame_landmark_vectors = landmarks_init - positions_init[i]
            frame_landmark_vectors /= np.linalg.norm(frame_landmark_vectors, axis=-1, keepdims=True)
            landmark_vec_mask = ~np.any(np.isnan(frame_landmark_vectors), axis=-1)
            cam_vec_mask = ~np.any(np.isnan(landmark_vectors[i]), axis=-1)
            common_mask = landmark_vec_mask & cam_vec_mask

            oris_flu_init[i] = R.align_vectors(
                frame_landmark_vectors[common_mask], landmark_vectors[i][common_mask]
            )[0].as_rotvec()

    # Fill nan landmarks as good as possible
    for lm_idx, lm in enumerate(landmarks_init):
        if np.any(np.isnan(lm)):
            lm[:] = intersect(positions_init, landmark_vectors[:, lm_idx])

    vars_full[:] = np.hstack((
        landmarks_init.ravel(),
        positions_init.ravel(),
        oris_flu_init.ravel(),
    ))

    # Fixed variables
    landmarks_free = np.ones_like(landmarks_init, dtype=bool)
    positions_free = np.ones_like(positions_init, dtype=bool)
    oris_flu_free = np.ones_like(oris_flu_init, dtype=bool)

    if "landmarks" not in optimize:
        assert landmarks is not None, "Landmarks need to be provided if they are not optimized"
        landmarks_free[:] = False
    if "positions" not in optimize:
        assert positions is not None, "Positions need to be provided if they are not optimized"
        positions_free[:] = False
    if "oris" not in optimize:
        assert oris_flu is not None, "Orientations need to be provided if they are not optimized"
        oris_flu_free[:] = False

    if landmarks_optimize is not None:
        landmarks_free[~landmarks_optimize] = False

    if "positions" in optimize and "landmarks" in optimize and "oris" in optimize \
            and (landmarks_optimize is None or np.sum(~landmarks_optimize) < 3):
        # If all variables are optimized, we need to fix origin, orientation and scale
        # Fix first two coordinates of first landmark
        landmarks_free[0, 0:2] = False

        # Fix t=0 position to 0
        positions_free[0, :] = False

        # Fix t=0 orientation to eye(3)
        oris_flu_free[0, :] = False
    elif "positions" in optimize and "landmarks" in optimize and "oris" not in optimize \
            and np.sum(~landmarks_optimize) < 3:
        # This case is missing a scale and an origin
        raise Exception("Fixed orientations need to be implemented")
    # elif "positions" not in optimize and "landmarks" in optimize and "oris" in optimize:
    #     pass
    # elif "positions" in optimize and "landmarks" not in optimize and "oris" in optimize:
    #     pass
    else:
        pass

    mask_free_input[:] = np.hstack((
        landmarks_free.ravel(),
        positions_free.ravel(),
        oris_flu_free.ravel()
    ))

    vars_free = vars_full[mask_free_input]

    if debug:
        print(f"Optimizing {np.sum(mask_free_input)} of {mask_free_input.size} variables")
    return vars_free, vars_full, mask_free_input


def get_precalc():
    global precalc

    if precalc is None:
        precalc = {
            'objfunc_repro': jit(viso_ag.objfun_repro),
            'jacobians_repro': [jit(jacobian(viso_ag.objfun_repro, i_var)) for i_var in range(3)],
            'objfunc_3d': jit(viso_ag.objfun_3d),
            'jacobians_3d': [jit(jacobian(viso_ag.objfun_3d, i_var)) for i_var in range(3)],
        }
    return precalc


def unravel_vars(free_vars, args):
    vars_full = args["vars_full"].copy()
    vars_full[args["mask_opt"]] = free_vars
    landmark_imcoords = args["landmark_imcoords"]
    landmark_vecs = args["landmark_vecs"]
    cam_matrices, ks, xis, cam_Rs, cam_ts = unpack_calibs(args["calibs"])

    if landmark_vecs is not None:
        k = landmark_vecs.shape[0]
        n = landmark_vecs.shape[1]
    elif landmark_imcoords is not None:
        k = landmark_imcoords.shape[1]
        n = landmark_imcoords.shape[2]
    else:
        raise Exception("Either landmark_vecs or landmark_imcoords must not be None")

    edges = np.array([0,
                      n * 3,
                      n * 3 + k * 3,
                      n * 3 + k * 3 + k * 3
                      ])

    landmarks = vars_full[edges[0]:edges[1]].reshape(n, 3)
    positions = vars_full[edges[1]:edges[2]].reshape(k, 3)
    oris_flu = vars_full[edges[2]:edges[3]].reshape(k, 3)
    return landmarks, positions, oris_flu, landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts


def unpack_calibs(calib):
    cam_matrices = np.array([c["A"] for c in calib])
    ks = np.array([c["k"] for c in calib])
    xis = np.array([c["xi"] for c in calib])
    cam_Rs = R.from_rotvec(np.array([c["rvec_cam"] for c in calib])).as_matrix()
    cam_ts = np.array([c["tvec_cam"] for c in calib])

    return cam_matrices, ks, xis, cam_Rs, cam_ts


def make_jacobian_sparsity_mask(k, n, mask_free_input):
    jac_sparsity = np.zeros((2 * n * k, n * 3 + k * 6), dtype=bool)
    for i_k in range(k):
        for i_n in range(n):
            # Set for landmarks
            mat_i = 2 * i_k * n + 2 * i_n
            mat_j = 3 * i_n
            jac_sparsity[mat_i:mat_i + 2, mat_j:mat_j + 3] = True

    for i_k in range(k):
        # Set for positions
        mat_i = 2 * i_k * n
        mat_j = 3 * i_k + 3 * n
        jac_sparsity[mat_i:mat_i + 2 * n, mat_j:mat_j + 3] = True

        # ... and orientations
        mat_j = 3 * i_k + 3 * n + 3 * k
        jac_sparsity[mat_i:mat_i + 2 * n, mat_j:mat_j + 3] = True

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(jac_sparsity)
    # plt.show()

    return jac_sparsity[:, mask_free_input]
