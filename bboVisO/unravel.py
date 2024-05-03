import numpy as np
from scipy.spatial.transform import Rotation as R


def unravel_vars(free_vars, args):
    landmark_imcoords = args["landmark_imcoords"]
    landmark_vecs = args["landmark_vecs"]

    vars_full = args["vars_full"].copy()
    vars_full[args["mask_opt"]] = free_vars
    if landmark_vecs is not None:
        m = landmark_vecs.shape[0]
        k = landmark_vecs.shape[1]
        n = landmark_vecs.shape[2]
    elif landmark_imcoords is not None:
        m = landmark_imcoords.shape[0]
        k = landmark_imcoords.shape[1]
        n = landmark_imcoords.shape[2]
    else:
        raise Exception("Either landmark_vecs or landmark_imcoords must not be None")
    if args["single_position"]:
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

    landmarks = vars_full[edges[0]:edges[1]].reshape(n, 3)
    positions = vars_full[edges[1]:edges[2]].reshape(k_p, 3)
    if args["single_position"]:
        positions = np.repeat(positions, k, axis=0)
    oris_flu = vars_full[edges[2]:edges[3]].reshape(k, 3)
    cam_rotations = vars_full[edges[3]:edges[4]].reshape(m, 3)
    cam_translations = vars_full[edges[4]:edges[5]].reshape(m, 3)

    return landmarks, positions, oris_flu, cam_rotations, cam_translations


def unpack_calibs(calib):
    cam_matrices = np.array([c["A"] for c in calib])
    ks = np.array([c["k"] for c in calib])
    xis = np.array([c["xi"] for c in calib])
    cam_Rs = R.from_rotvec(np.array([c["rvec_cam"] for c in calib])).as_matrix()
    cam_ts = np.array([c["tvec_cam"] for c in calib])

    return cam_matrices, ks, xis, cam_Rs, cam_ts
