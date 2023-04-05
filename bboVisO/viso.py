import time

import numpy as np
import bboVisO.viso_ag as viso_ag
from jax import jit, jacfwd as jacobian  # jacfwd is recommended for 'tall' jacobians_repro, jacrev for 'wide'
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, OptimizeResult
from calibcamlib.camerasystem import Camerasystem
from bboVisO.geometry import umeyama

debug = False
precalc = None


def infer_position_attitude(landmark_imcoords,  calibs, opts_optim, opts):
    # Initializations, for alternating updates
    landmarks_opt = None
    positions_opt = None
    oris_flu_opt = None
    end_optim = False

    # Previous positions for alternating update
    positions_prev = np.full((landmark_imcoords.shape[1], 3), np.nan)

    # Optimize all at once or alternate
    optimize = [["oris", "positions", "landmarks"]]
    # optimize = cycle(["oris", "positions", "landmarks"])  # Seems to converge to the same value

    t = time.time()
    precalc = get_precalc()
    for opt in optimize:
        # Always end with a full optimization
        if end_optim:
            opt = ["oris", "positions", "landmarks"]

        # If we are doing a full optimization, increase to full step count steps
        if isinstance(opt, list):
            opts_optim["max_nfev"] = opts["optimization"]["max_nfev"]

        # Run optimization
        print(f"Optimizing {opt}")
        vars_free, args = make_optim_input(calibs, opts, landmarks_opt, positions_opt, oris_flu_opt, optimize=opt,
                                                precalc=precalc, landmark_imcoords=landmark_imcoords)

        min_result: OptimizeResult = least_squares(objfun_repro_wrapper,
                                                   vars_free,
                                                   bounds=np.array([[-np.inf, np.inf]] * vars_free.size).T,
                                                   args=[args],
                                                   **opts_optim
                                                   )

        landmarks_opt, positions_opt, oris_flu_opt = unravel_vars(min_result.x, args)[:3]

        if end_optim:
            break

        max_pos_shift = np.max(np.abs(positions_prev - positions_opt))
        positions_prev = positions_opt

        print(f"Max pos shift {max_pos_shift}")
        if max_pos_shift < 1e-3:
            end_optim = True

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

    return landmarks_opt, positions_opt, oris_flu_opt


def align_results_by_landmarks(landmarks_real, landmarks, positions, oris_flu):
    rot, t, s = umeyama(landmarks, landmarks_real)
    landmarks = landmarks @ rot.T * s + t
    positions = positions @ rot.T * s + t
    oris_flu = R.from_matrix(rot) * R.from_rotvec(oris_flu)

    # Combine the 2 cameras (one will be nan)
    return landmarks, positions, oris_flu


def align_results_by_positions(positions_real, landmarks, positions, oris_flu):
    rot, t, s = umeyama(positions, positions_real)
    landmarks = landmarks @ rot.T * s + t
    positions = positions @ rot.T * s + t
    oris_flu = R.from_matrix(rot) * R.from_rotvec(oris_flu)

    # Combine the 2 cameras (one will be nan)
    return landmarks, positions, oris_flu


def objfun_repro_wrapper(free_vars, args):
    landmarks, positions, oris_flu, \
        landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts = unravel_vars(free_vars, args)

    errors2d = args['precalc']['objfunc_repro'](landmarks, positions, oris_flu,
                                                landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts)

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


def make_optim_input(calibs, opts, landmarks=None, positions=None, oris_flu=None, optimize=None, precalc=None,
                     landmark_imcoords=None, landmark_vecs=None, landmark_dists=None):
    cs = Camerasystem.from_calibs(calibs)
    if landmark_vecs is None and landmark_imcoords is not None:
        landmark_vecs = np.nansum(cs.get_camera_lines(landmark_imcoords)[0], axis=0)

    vars_free, vars_full, mask_free_input = make_initialization(landmark_vecs, calibs, opts, landmarks, positions,
                                                                oris_flu,
                                                                optimize=optimize)

    args = {
        'vars_full': vars_full,  # All possible vars, free vars will be substituted in _free wrapper functions
        'mask_opt': mask_free_input,  # Mask of free vars within all vars
        'landmark_imcoords': landmark_imcoords,
        'landmark_vecs': landmark_vecs,
        'calibs': calibs,
        'precalc': get_precalc() if precalc is None else precalc
    }

    # TODO (unknown speed) return jac_sparsity for optimization
    return vars_free, args


def make_initialization(landmark_vectors, calibs, opts, landmarks=None, positions=None, oris_flu=None, optimize=None):
    if optimize is None:
        optimize = ["landmarks", "positions", "oris"]

    k = landmark_vectors.shape[0]
    n = landmark_vectors.shape[1]

    edges = np.array([0,
                      n * 3,
                      n * 3 + k * 3,
                      n * 3 + k * 3 + k * 3
                      ])

    vars_full = np.random.random(edges[-1])
    mask_free_input = np.ones(vars_full.shape, dtype=bool)

    ## Init with fixed first orientation and position
    # Init landmarks as extensions of t=0 camera vecs
    # Fix relative x position to landmark 1
    cs = Camerasystem.from_calibs(calibs)
    landmarks_init = landmark_vectors[0] * 100
    mask_free_input[edges[0]:edges[0] + 1] = False
    vars_full[edges[0]:edges[1]] = landmarks_init.ravel()

    # Fix t=0 position to 0
    mask_free_input[edges[1]:edges[1] + 3] = False
    vars_full[edges[1]:edges[2]] = 0

    # Fix t=0 orientation to eye(3)
    mask_free_input[edges[2]:edges[2] + 3] = False
    vars_full[edges[2]:edges[3]] = 0

    # Initialize other orientation to landmark vecs
    # TODO: Only optimize on the 2 vectors that have the most similar angle as the reference (as proxy for "far away")
    # TODO: Move the reference iteratively
    landmarks_init /= np.linalg.norm(landmarks_init, axis=-1)[..., np.newaxis]
    for i in range(1, k):
        vars_full[edges[2] + i * 3:edges[2] + i * 3 + 3] = R.align_vectors(
            landmarks_init, landmark_vectors[i]
        )[0].as_rotvec()

    ## Init with fixed landmarks
    # mask_free_input[0:6] = False
    # vars_full[0:3] = 0
    # vars_full[3] = 1
    # vars_full[4:6] = 0
    #
    # mask_free_input[8] = False
    # vars_full[8] = 0

    if "landmarks" not in optimize:
        mask_free_input[edges[0]:edges[1]] = False
    if "positions" not in optimize:
        mask_free_input[edges[1]:edges[2]] = False
    if "oris" not in optimize:
        mask_free_input[edges[2]:edges[3]] = False

    if landmarks is not None:
        vars_full[edges[0]:edges[1]] = landmarks.ravel()
    if positions is not None:
        vars_full[edges[1]:edges[2]] = positions.ravel()
    if oris_flu is not None:
        vars_full[edges[2]:edges[3]] = oris_flu.ravel()

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

    if landmark_imcoords is not None:
        k = landmark_imcoords.shape[1]
        n = landmark_imcoords.shape[2]

    if landmark_vecs is not None:
        k = landmark_vecs.shape[0]
        n = landmark_vecs.shape[1]

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
