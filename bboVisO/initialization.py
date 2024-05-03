import numpy as np
from calibcamlib.helper import intersect
from jax import jit, jacfwd as jacobian
from scipy.spatial.transform import Rotation as R

from bboVisO import viso_ag as viso_ag
from bboVisO.geometry import build_vector_bundle
from bboVisO.helper import landmark_vecs_from_imcoords, landmark_imcoords_from_vecs
from bboVisO.method_switches import make_jacobian_sparsity_mask

from bbo.geometry import RigidTransform

precalc_lazy = None


def make_optim_input(calibs, opts, landmarks=None, positions=None, oris_flu=None,
                     optimize=None, landmarks_optimize=None, landmarks_far=None,
                     precalc=None, landmark_imcoords=None, landmark_vecs=None):
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
        'single_position': opts["single_position"],
        'calibs': calibs,
        'camera_positions': t_c2w,
        'lm_close_pen_weight': opts["lm_close_pen_weight"],
        'precalc': get_precalc(opts) if precalc is None else precalc
    }

    if not opts["single_position"]:
        jac_sparsity = make_jacobian_sparsity_mask(opts)(m, k, n, mask_free_input)
    else:
        jac_sparsity = None  # TODO: implement
    optim_args = {
        'bounds': np.array([[-np.inf, np.inf]] * vars_free.size).T,
        'jac_sparsity': jac_sparsity
    }

    return vars_free, obfun_args, optim_args


def fill_landmark_imcoords_vecs(landmark_imcoords, landmark_vecs, calibs):
    assert landmark_imcoords is not None or landmark_vecs is not None, \
        "landmark_imcoords or landmark_vecs must be given"
    if landmark_imcoords is None:
        landmark_vecs = landmark_vecs_from_imcoords(landmark_imcoords, calibs)
    if landmark_vecs is None:
        landmark_imcoords = landmark_imcoords_from_vecs(landmark_vecs, calibs)
    return landmark_imcoords, landmark_vecs


def make_initialization(landmark_vecs, calibs, opts, landmarks=None, positions=None, oris_flu=None,
                        landmarks_optimize=None, landmarks_far=None,
                        optimize=("landmarks", "positions", "oris")):
    m = landmark_vecs.shape[0]
    k = landmark_vecs.shape[1]
    n = landmark_vecs.shape[2]

    landmark_vectors_sphere = np.nanmean(landmark_vecs, axis=0)

    if opts["single_position"]:
        k_p = 1
    else:
        k_p = k

    field_sizes = np.array([0,
                            n * 3,  # landmarks
                            k_p * 3,  # positions
                            k * 3,  # orientations
                            m * 3,  # cam extrinsic rotations
                            m * 3,  # cam extrinsic positions
                            ])
    edges = np.cumsum(field_sizes)

    vars_full = np.full(edges[-1], np.nan)
    mask_free_input = np.ones(vars_full.shape, dtype=bool)

    # Init with fixed first orientation and position
    # Init landmarks as extensions of t=0 camera vecs
    # Fix relative x position to landmark 1
    # cs = Camerasystem.from_calibs(calibs)
    # cs.cameras[0]["camera"].set_center_pixel(np.array([[960, 512]]))

    # Initializations
    if landmarks is not None:
        landmarks_init = landmarks
    else:
        # Initialize to a distance of 100m into the direction of the  respective camera vectors
        landmarks_init = build_vector_bundle(landmark_vectors_sphere[0], landmark_vectors_sphere[1:]) * 100

    if positions is not None:
        positions_init = positions  # TODO: Implement single position
    else:
        # Calculate mean of all non-far landmarks
        landmarks_close_mean = np.nanmean(landmarks_init[~landmarks_far], axis=0)
        # Lift by lift factor. For cameras with limited FOV, this will ensure that as many landmarks as possible are
        # in the image by moving the camera further away
        landmarks_close_mean[2] += np.nanmean(
            np.linalg.norm(landmarks_init[~landmarks_far] - landmarks_close_mean, axis=-1)
        ) * opts["lift_factor"]
        # If we have a lifted mean, initialize with that ...
        if np.any(~landmarks_far) and not np.any(np.isnan(landmarks_close_mean)):
            positions_init = np.ones((k, 3)) * landmarks_close_mean
        # ... else initialize to coordinate origin
        else:
            print("Initialized with coordinate origin. This is probably undesirable.")
            positions_init = np.zeros((k, 3))  # We initialize to the size of orientations, even for single position

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
            cam_vec_mask = ~np.any(np.isnan(landmark_vectors_sphere[i]), axis=-1)
            common_mask = landmark_vec_mask & cam_vec_mask

            oris_flu_init[i] = R.align_vectors(
                frame_landmark_vectors[common_mask], landmark_vectors_sphere[i][common_mask]
            )[0].as_rotvec()

    cam_trafo = RigidTransform(rotation=R.from_rotvec([c["rvec_cam"] for c in calibs]),
                               translation=np.array([c["tvec_cam"] for c in calibs]))
    extrinsics_rotations_init = cam_trafo.inv().get_rotation().as_rotvec()
    extrinsics_translations_init = cam_trafo.inv().get_translation()

    # Fill nan landmarks as good as possible
    for lm_idx, lm in enumerate(landmarks_init):
        if np.any(np.isnan(lm)):
            lm[:] = intersect(positions_init, landmark_vectors_sphere[:, lm_idx])

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
    extrinsics_rotations_free[:3] = False  # Leave reference cam
    if "extrinsics_t" not in optimize:
        extrinsics_translations_free[:] = False
    extrinsics_translations_free[:3] = False  # Leave reference cam

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


def get_precalc(opts):
    global precalc_lazy

    if precalc_lazy is None:
        print(opts["jax_backend"])
        precalc_lazy = {
            'objfunc_repro': jit(viso_ag.objfun_repro, backend=opts["jax_backend"]),
            'jacobians_repro': [jit(jacobian(viso_ag.objfun_repro, i_var), backend=opts["jax_backend"])
                                for i_var in range(3)],
            'objfunc_angular': jit(viso_ag.objfun_angular, backend=opts["jax_backend"]),
            'jacobians_angular': [jit(jacobian(viso_ag.objfun_angular, i_var), backend=opts["jax_backend"])
                                  for i_var in range(3)],
            # 'objfunc_3d': jit(viso_ag.objfun_3d, backend=opts["jax_backend"]),
            # 'jacobians_3d': [jit(jacobian(viso_ag.objfun_3d, i_var), backend=opts["jax_backend"])
            #                  for i_var in range(3)],
        }
    return precalc_lazy
