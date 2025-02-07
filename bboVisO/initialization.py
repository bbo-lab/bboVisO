import numpy as np

from calibcamlib import Camerasystem
from calibcamlib.helper import intersect
from scipy.spatial.transform import Rotation as R

from bboVisO.ravel import ravel_vars
from bboVisO.helper import landmark_vecs_from_imcoords, landmark_imcoords_from_vecs
from bboVisO.method_switches import make_jacobian_sparsity_mask

from bbo.geometry import RigidTransform

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

    # Create one landmark_imcoords from landmark_vecs or vice versa if necessary
    landmark_imcoords, landmark_vecs = fill_landmark_imcoords_vecs(landmark_imcoords, landmark_vecs, calibs)

    # Create free variables
    vars_free, ravel_args = make_free_vars(landmark_vecs, calibs, opts, landmarks, positions,
                       oris_flu,
                       optimize=optimize, landmarks_optimize=landmarks_optimize, landmarks_far=landmarks_far)

    m = landmark_vecs.shape[0]
    k = landmark_vecs.shape[1]
    n = landmark_vecs.shape[2]

    trafo_world2cam = RigidTransform(
        rotation=np.array([c["rvec_cam"] for c in calibs]),
        translation=np.array([c["tvec_cam"] for c in calibs]),
        rotation_type="rotvec"
    )

    obfun_args = {
        'ravel_args': ravel_args,
        'landmark_imcoords': landmark_imcoords,
        'landmark_vecs': landmark_vecs,
        'landmarks_weights': landmarks_weights[np.newaxis, np.newaxis, :, np.newaxis],  # Reshape for later mult.
        'weight_mask': np.any(weight_mask, axis=-1, keepdims=True),
        'weight': weight,
        'single_position': opts["single_position"],
        'calibs': calibs,
        'camera_positions': trafo_world2cam.inv().get_translation(),
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


def make_free_vars(landmark_vecs_cam, calibs, opts, landmarks=None, positions=None, oris_flu=None,
                   landmarks_optimize=None, landmarks_far=None,
                   optimize=("landmarks", "positions", "oris")):

    landmarks_init, positions_init, oris_flu_init, extrinsics_rotations_init, extrinsics_translations_init = initialize_vars(
        landmark_vecs_cam, calibs, opts, landmarks, positions, oris_flu, landmarks_far, optimize)

    vars_free, ravel_args = ravel_vars(landmark_vecs_cam, opts, landmarks_init, positions_init, oris_flu_init,
                                       extrinsics_rotations_init, extrinsics_translations_init, landmarks_optimize,
                                       optimize)

    return vars_free, ravel_args


def initialize_vars(landmark_vecs_cam, calibs, opts, landmarks, positions, oris_flu, landmarks_far, optimize):
    m = landmark_vecs_cam.shape[0]
    k = landmark_vecs_cam.shape[1]
    n = landmark_vecs_cam.shape[2]

    if landmarks is None:
        assert "landmarks" not in optimize, "Landmarks need to be provided if they are not optimized"
        landmarks_init = np.full((n, 3), np.nan)
    else:
        landmarks_init = landmarks.copy()
    if positions is None:
        assert "positions" not in optimize, "Positions need to be provided if they are not optimized"
        positions_init = np.full((k, 3), np.nan)
    else:
        positions_init = positions.copy()
    if oris_flu is None:
        assert "oris" not in optimize, "Orientations need to be provided if they are not optimized"
        oris_flu_init = np.full((k, 3), np.nan)
    else:
        oris_flu_init = oris_flu.copy()
    cs = Camerasystem.from_calibs(calibs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        landmark_vectors_sphere = np.nanmean(
            np.array([
                cs.get_camsystem_to_cam_transformation(i_cam).inv().rotate(lmvc)
                for i_cam, lmvc in enumerate(landmark_vecs_cam)]),
            axis=0)
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
    return landmarks_init, positions_init, oris_flu_init, extrinsics_rotations_init, extrinsics_translations_init

