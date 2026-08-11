import numpy as np
from scipy.spatial.transform import Rotation as R

from bboVisO.helper import get_var_edges, load_func


def ravel_vars(landmark_vecs_cam, opts, landmarks_init, positions_init, oris_flu_init, extrinsics_rotations_init,
               extrinsics_translations_init, landmarks_optimize, optimize):
    module_name = opts['ravel']['ravel_func'].get('module', None)
    func_name = opts['ravel']['ravel_func']['func']
    func = load_func(func_name, module_name)

    vars_free, ravel_args = func(landmark_vecs_cam, opts, landmarks_init, positions_init, oris_flu_init, extrinsics_rotations_init,
               extrinsics_translations_init, landmarks_optimize, optimize, **opts['ravel']['ravel_opts'])

    ravel_args['opts'] = opts['ravel']
    return vars_free, ravel_args


def unravel_vars(free_vars, ravel_args):
    module_name = ravel_args['opts']['unravel_func'].get('module', None)
    func_name = ravel_args['opts']['unravel_func']['func']
    func = load_func(func_name, module_name)

    landmarks, positions, oris_flu, extrinsics_rotations, extrinsics_translations = (
        func(free_vars, ravel_args, **ravel_args['opts']['ravel_opts']))

    return landmarks, positions, oris_flu, extrinsics_rotations, extrinsics_translations


def default_ravel_vars(landmark_vecs_cam, opts, landmarks_init, positions_init, oris_flu_init, extrinsics_rotations_init,
               extrinsics_translations_init, landmarks_optimize, optimize):
    m = landmark_vecs_cam.shape[0]
    k = landmark_vecs_cam.shape[1]
    n = landmark_vecs_cam.shape[2]
    edges = get_var_edges(m, k, n)
    vars_full = np.full(edges[-1], np.nan)
    mask_free_input = np.ones(vars_full.shape, dtype=bool)
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
        landmarks_free[:] = False
    if "positions" not in optimize:
        positions_free[:] = False
    if "oris" not in optimize:
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
    ravel_args = {
        'shape': landmark_vecs_cam.shape,
        'vars_full': vars_full,
        'mask_free_input': mask_free_input,
    }
    return vars_free, ravel_args

def default_unravel_vars(free_vars, ravel_args):
    vars_full = ravel_args["vars_full"].copy()
    vars_full[ravel_args["mask_free_input"]] = free_vars

    m = ravel_args["shape"][0]
    k = ravel_args["shape"][1]
    n = ravel_args["shape"][2]

    field_sizes = np.array([0,
                            n * 3,  # landmarks
                            k * 3,  # positions
                            k * 3,  # orientations
                            m * 3,  # cam extrinsic rotations
                            m * 3,  # cam extrinsic positions
                            ])
    edges = np.cumsum(field_sizes)

    landmarks = vars_full[edges[0]:edges[1]].reshape(n, 3)
    positions = vars_full[edges[1]:edges[2]].reshape(k, 3)

    oris_flu = vars_full[edges[2]:edges[3]].reshape(k, 3)
    extrinsics_rotations = vars_full[edges[3]:edges[4]].reshape(m, 3)
    extrinsics_translations = vars_full[edges[4]:edges[5]].reshape(m, 3)

    return landmarks, positions, oris_flu, extrinsics_rotations, extrinsics_translations


def unpack_calibs(calib):
    cam_matrices = np.array([c["A"] for c in calib])
    ks = np.array([c["k"] for c in calib])
    xis = np.array([c["xi"] for c in calib])
    projection_models = np.array([c.get("projection_model", None) for c in calib])
    cam_Rs = R.from_rotvec(np.array([c["rvec_cam"] for c in calib])).as_matrix()
    cam_ts = np.array([c["tvec_cam"] for c in calib])

    return cam_matrices, ks, xis, projection_models, cam_Rs, cam_ts
