import numpy as np

from calibcamlib.helper import intersect
from scipy.spatial.transform import Rotation as R

from bboVisO.geometry import build_vector_bundle
from bboVisO.helper import landmark_vecs_from_imcoords, landmark_imcoords_from_vecs, get_var_edges
from bboVisO.method_switches import make_jacobian_sparsity_mask

import warnings


def make_optim_input(calibs, opts,
                     landmarks=None,  # Initialization for landmarks
                     positions=None,  # Initialization for positions
                     oris_flu=None,  # Initialization for oris_flu
                     optimize=None,  # Subset of ("landmarks", "positions", "oris", "extrinsics_r", "extrinsics_t")
                     landmarks_optimize=None,  # Mask of landmarks to optimize
                     landmarks_far=None,  # Mask for landmarks that are far away
                     landmarks_weights=None,  # Weight for each landmark
                     weight_mask=None,  # Mask to apply "weight" to, per imcoord (might get reduced by method)
                     weight=1,  # Weight for "weight_mask", e.g. to reduce weight of interpolated points
                     landmark_imcoords=None,  # Landmark image coordinates
                     landmark_vecs=None):  # Corresponding vectors to "landmark_imcoords", in single cam coords

    if landmarks_weights is None:
        landmarks_weights = np.ones((len(landmarks),))
    if weight_mask is None:
        weight_mask = np.ones_like(landmark_imcoords, dtype=bool)

    landmark_imcoords, landmark_vecs = fill_landmark_imcoords_vecs(landmark_imcoords, landmark_vecs, calibs)

    vars_free, vars_full, mask_free_input = \
        make_initialization(landmark_vecs, calibs, opts, landmarks, positions,
                            oris_flu,
                            optimize=optimize, landmarks_optimize=landmarks_optimize, landmarks_far=landmarks_far)

    m = landmark_vecs.shape[0]
    k = landmark_vecs.shape[1]
    n = landmark_vecs.shape[2]

    t_c2w = np.array([R.from_rotvec(c["rvec_cam"]).apply(-c["tvec_cam"]) for c in calibs])

    obfun_args = {
        'vars_full': vars_full,  # All possible vars, free vars will be substituted in _free wrapper functions
        'mask_opt': mask_free_input,  # Mask of free vars within all vars
        'landmark_imcoords': landmark_imcoords,
        'landmark_vecs': landmark_vecs,
        'landmarks_weights': landmarks_weights[np.newaxis, np.newaxis, :, np.newaxis],  # Reshape for later mult.
        'weight_mask': np.any(weight_mask, axis=-1, keepdims=True),
        'weight': weight,
        'single_position': opts["single_position"],
        'calibs': calibs,
        'camera_positions': t_c2w,
        'lm_close_pen_weight': opts["lm_close_pen_weight"],
    }

    if not opts["single_position"] and opts["sparse_jacobian"]:
        jac_sparsity = make_jacobian_sparsity_mask(opts)(m, k, n, mask_free_input)
    else:
        jac_sparsity = None  # TODO: implement
    optim_args = {
        'bounds': np.array([[-np.inf, np.inf]] * vars_free.size).T,
    }

    if opts["sparse_jacobian"]:
        optim_args['jac_sparsity'] = jac_sparsity

    return vars_free, obfun_args, optim_args


def fill_landmark_imcoords_vecs(landmark_imcoords, landmark_vecs, calibs):
    assert landmark_imcoords is not None or landmark_vecs is not None, \
        "landmark_imcoords or landmark_vecs must be given"
    if landmark_imcoords is None:
        landmark_imcoords = landmark_imcoords_from_vecs(landmark_vecs, calibs)
    if landmark_vecs is None:
        landmark_vecs = landmark_vecs_from_imcoords(landmark_imcoords, calibs)
    return landmark_imcoords, landmark_vecs


def make_initialization(landmark_vecs, calibs, opts, landmarks=None, positions=None, oris_flu=None,
                        landmarks_optimize=None, landmarks_far=None,
                        optimize=("landmarks", "positions", "oris")):
    m = landmark_vecs.shape[0]
    k = landmark_vecs.shape[1]
    n = landmark_vecs.shape[2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        landmark_vectors_sphere = np.nanmean(landmark_vecs, axis=0)

    if opts["single_position"]:
        k_p = 1
    else:
        k_p = k

    edges = get_var_edges(m, k, k_p, n)

    vars_full = np.full(edges[-1], np.nan)
    mask_free_input = np.ones(vars_full.shape, dtype=bool)

    # Init with fixed first orientation and position
    # Init landmarks as extensions of t=0 camera vecs
    # Fix relative x position to landmark 1
    # cs = Camerasystem.from_calibs(calibs)
    # cs.cameras[0]["camera"].set_center_pixel(np.array([[960, 512]]))

    if landmarks is None:
        landmarks = np.full((n, 3), np.nan)
    if positions is None:
        positions = np.full((k, 3), np.nan)
    if oris_flu is None:
        oris_flu = np.full((k, 3), np.nan)

    landmarks_init = landmarks
    positions_init = positions
    oris_flu_init = oris_flu

    # Initialize nan positions
    if "positions" in optimize and np.any(nan_mask := np.any(np.isnan(positions_init), axis=-1)):
        # Calculate mean of all non-far landmarks
        landmarks_close_mean = np.nanmean(landmarks_init[~landmarks_far], axis=0, keepdims=True)
        # Lift by lift factor. For cameras with limited FOV, this will help that as many landmarks as possible are
        # in the image by moving the camera further away
        landmarks_close_mean[:, 2] += np.nanmean(
            np.linalg.norm(landmarks_init[~landmarks_far] - landmarks_close_mean, axis=-1)
        ) * opts["lift_factor"]
        # If we have a lifted mean, initialize with that ...
        if np.any(~landmarks_far) and not np.any(np.isnan(landmarks_close_mean)):
            positions_init[nan_mask] = landmarks_close_mean
        # ... else initialize to coordinate origin
        else:
            print("Initialized with far coordinates. This is possibly undesirable.")
            landmarks_mean = np.nanmean(landmarks_init, axis=0, keepdims=True)
            # Lift by lift factor. For cameras with limited FOV, this will help that as many landmarks as possible are
            # in the image by moving the camera further away
            landmarks_mean[:, 2] += np.nanmean(
                np.linalg.norm(landmarks_init - landmarks_mean, axis=-1)
            ) * opts["lift_factor"]

            positions_init[nan_mask] = landmarks_mean

    # Initialize nan orientations
    if "oris" in optimize and np.any(nan_mask := np.any(np.isnan(oris_flu_init), axis=-1)):
        # Initialize to zero
        oris_flu_init[nan_mask] = np.zeros((1, 3))

        # Initialize other orientation to landmark vecs
        # TODO: Only optimize on 2 vectors that have the most similar angle as the reference (as proxy for "far away")
        # TODO: Move the reference iteratively
        for lm_idx in np.where(nan_mask)[0]:
            frame_landmark_vectors = landmarks_init - positions_init[lm_idx]
            frame_landmark_vectors /= np.linalg.norm(frame_landmark_vectors, axis=-1, keepdims=True)
            landmark_vec_mask = ~np.any(np.isnan(frame_landmark_vectors), axis=-1)
            cam_vec_mask = ~np.any(np.isnan(landmark_vectors_sphere[lm_idx]), axis=-1)
            common_mask = landmark_vec_mask & cam_vec_mask

            oris_flu_init[lm_idx] = R.align_vectors(
                frame_landmark_vectors[common_mask], landmark_vectors_sphere[lm_idx][common_mask]
            )[0].as_rotvec()

    # Initialize camera extrinsics
    extrinsics_rotations_init = np.array([c["rvec_cam"] for c in calibs])
    extrinsics_translations_init = np.array([c["tvec_cam"] for c in calibs])

    # Initialize nan landmarks
    if "landmarks" in optimize and np.any(nan_mask := np.any(np.isnan(landmarks_init), axis=-1)):
        # Fill nans with preliminary triangulations
        for lm_idx in np.where(nan_mask)[0]:
            # TODO take camera position into account
            landmarks_init[lm_idx] = intersect(positions_init, landmark_vectors_sphere[:, lm_idx])

    if opts["single_position"]:
        positions_init = positions_init[(0,),]

    vars_full[:] = np.hstack((
        landmarks_init.ravel(),
        positions_init.ravel(),
        oris_flu_init.ravel(),
        extrinsics_rotations_init.ravel(),
        extrinsics_translations_init.ravel(),
    ))

    # Fixed variables
    landmarks_free = np.ones_like(landmarks_init, dtype=bool)
    positions_free = np.ones_like(positions_init, dtype=bool)
    oris_flu_free = np.ones_like(oris_flu_init, dtype=bool)
    extrinsics_rotations_free = np.ones_like(extrinsics_rotations_init, dtype=bool)
    extrinsics_translations_free = np.ones_like(extrinsics_translations_init, dtype=bool)

    assert not opts["single_position"] or "landmarks" not in optimize, \
        "Landmarks cannot be optimized from single position"
    if "landmarks" not in optimize:
        assert landmarks is not None, "Landmarks need to be provided if they are not optimized"
        landmarks_free[:] = False
    if "positions" not in optimize:
        assert positions is not None, "Positions need to be provided if they are not optimized"
        positions_free[:] = False
    if "oris" not in optimize:
        assert oris_flu is not None, "Orientations need to be provided if they are not optimized"
        oris_flu_free[:] = False
    if "extrinsics_r" not in optimize:
        extrinsics_rotations_free[:] = False
    extrinsics_rotations_free[0] = False  # Leave reference cam
    if "extrinsics_t" not in optimize:
        extrinsics_translations_free[:] = False
    extrinsics_translations_free[0] = False  # Leave reference cam

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
        oris_flu_free.ravel(),
        extrinsics_rotations_free.ravel(),
        extrinsics_translations_free.ravel(),
    ))

    vars_free = vars_full[mask_free_input]

    if opts["optimization"]["verbose"] > 0:  # TODO: better config option
        print(f"Optimizing {np.sum(mask_free_input)} of {mask_free_input.size} variables")

    return vars_free, vars_full, mask_free_input
