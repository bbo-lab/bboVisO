import warnings
import os.path

import numpy as np
from scipy.spatial.transform import Rotation as R

from calibcamlib.camerasystem import Camerasystem

import viso


def get_simulated_data(calibs, opts,
                       pertubation_deg=0, n_landmarks=8, landmark_radius=15, traj_length=10, traj_start=(-5, 0, 0),
                       traj_direction=(1, 0, 0), traj_radius=1, traj_v0=1, traj_period=1, fps=1):

    camerasystem = Camerasystem.from_calibs(calibs)

    landmarks = get_landmarks(n_landmarks, radius=landmark_radius)
    positions, oris_flu = get_screw_trajectory(length=traj_length, start=traj_start, direction=traj_direction,
                                               radius=traj_radius, v0=traj_v0, period=traj_period, fps=fps)

    # Generate relative positions
    landmark_vecs, landmark_dists = get_landmark_vectors(positions, oris_flu, landmarks)

    # Pertubate
    landmark_vecs_perturbed = pertubate_landmark_vectors(landmark_vecs, pertubation_deg / 180 * np.pi)
    assert pertubation_deg != 0 or np.all(landmark_vecs == landmark_vecs_perturbed), "Pertubation error"

    # Reproject image coordinates
    landmark_imcoords_perturbed = reproject_landmarks(landmark_vecs_perturbed * landmark_dists[..., np.newaxis],
                                                      camerasystem)

    vars_free_opt, args_opt = viso.make_optim_input(calibs, opts, landmarks, positions,
                                                    oris_flu.as_rotvec(),
                                                    landmark_imcoords=landmark_imcoords_perturbed)
    error_repro = viso.objfun_repro_wrapper(vars_free_opt, args_opt)
    error_3d_repro = viso.objfun_3d_wrapper(vars_free_opt, args_opt)

    vars_free_opt, args_opt = viso.make_optim_input(calibs, opts, landmarks, positions,
                                                    oris_flu.as_rotvec(),
                                                    landmark_vecs=landmark_vecs_perturbed)
    error_3d = viso.objfun_3d_wrapper(vars_free_opt, args_opt)

    print(f"Ground truth repro cost: {0.5 * np.sum(error_repro ** 2)}")
    print(f"Ground truth 3d cost (with reprojected image coords): {0.5 * np.sum(error_3d_repro ** 2)}")
    print(f"Ground truth 3d cost (with original landmark vectors): {0.5 * np.sum(error_3d ** 2)}")

    return landmark_imcoords_perturbed, landmarks, positions, oris_flu


def get_screw_trajectory(length=10, start=(-5, 0, 0), direction=(1, 0, 0), radius=1, v0=1, period=1, fps=60):
    start = np.array(start).reshape((1, 3))
    direction = np.array(direction).reshape((1, 3))

    rot = R.align_vectors(direction, np.array([[1, 0, 0]]))[0]

    ts = np.linspace(0, length / v0, length * fps // v0)
    trig_arg = 2 * np.pi * ts / period

    positions = np.array(
        [
            v0 * ts,
            radius * np.sin(trig_arg),
            radius * np.cos(trig_arg),
        ]
    ).T

    oris_flu = np.array(
        [np.array([
            [1, 0, 0],
            [0, np.cos(a), np.sin(a)],
            [0, -np.sin(a), np.cos(a)],
        ]) for a in trig_arg]
    )

    positions = rot.apply(positions) + start

    return positions, R.from_matrix(oris_flu)


def get_landmarks(N, radius=50, offset=(0, 0, 10)):
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    landmarks = np.array(
        [
            radius * np.sin(ang),
            radius * np.cos(ang),
            np.zeros(ang.shape)
        ]
    ).T

    landmarks = landmarks + offset

    return landmarks


def get_landmark_vectors(positions, oris_flu, landmarks, normalize=True):
    if isinstance(oris_flu, np.ndarray):
        oris_flu = R.from_rotvec(oris_flu)

    landmark_vectors = landmarks.reshape((1, -1, 3)) - positions.reshape((-1, 1, 3))

    landmark_vectors = np.array([
        o.inv().apply(landmark_vectors[i]) for i, o in enumerate(oris_flu)
    ])

    landmark_dists = np.linalg.norm(landmark_vectors, axis=2)

    landmark_vectors /= landmark_dists[..., np.newaxis]

    return landmark_vectors, landmark_dists


def pertubate_landmark_vectors(landmark_vectors, sigma, seed=0):
    np.random.seed(seed)
    pertubations = sigma * np.random.randn(*landmark_vectors.shape[0:2])
    dirs = 2 * np.pi * np.random.rand(*landmark_vectors.shape[0:2])

    landmark_vectors_perturbed = landmark_vectors.copy()

    for i_t, (ps, ds) in enumerate(zip(pertubations, dirs)):
        for i_n, (p, d) in enumerate(zip(ps, ds)):
            orth_v = np.array([landmark_vectors[i_t, i_n, 1], -landmark_vectors[i_t, i_n, 0], 0])
            orth_v /= np.linalg.norm(orth_v)
            orth_v = R.from_rotvec(d * landmark_vectors[i_t, i_n]).apply(orth_v)
            landmark_vectors_perturbed[i_t, i_n] = R.from_rotvec(p * orth_v).apply(landmark_vectors[i_t, i_n])

    return landmark_vectors_perturbed


def reproject_landmarks(landmark_vectors, camerasystem: Camerasystem):
    landmark_imcoords = camerasystem.project(landmark_vectors)

    # landmark_vectors_check = camerasystem.get_camera_lines(landmark_imcoords)
    # landmark_vectors_norm = landmark_vectors / np.linalg.norm(landmark_vectors, axis=-1)[..., np.newaxis]
    # diff = np.min(np.abs(landmark_vectors_check - landmark_vectors_norm[np.newaxis, ...]), axis=0)

    # We only want to optimize on the camera with the landmark closer to its optical axis
    # Note that due to the non-linearity of the distortion, we cannot decide this on pixel
    # coordinates, as extreme angles may be reflected back into the range of sensor coordinates!
    cam_center_vecs = camerasystem.get_camera_lines(np.array([[1920, 1080], [1920, 1080]]) / 2)[0]
    for idx_t, vs in enumerate(landmark_vectors):
        for idx_lm, v in enumerate(vs):
            if np.dot(v, cam_center_vecs[0]) < 0:
                landmark_imcoords[0, idx_t, idx_lm] = np.nan
                continue
            if np.dot(v, cam_center_vecs[1]) < 0:
                landmark_imcoords[1, idx_t, idx_lm] = np.nan
                continue
            # Make sure that we always only have one camera at max to allow for nansum choice of cam_vec
            if np.dot(v, cam_center_vecs[0]) < np.dot(v, cam_center_vecs[1]):
                landmark_imcoords[0, idx_t, idx_lm] = np.nan
            else:
                landmark_imcoords[1, idx_t, idx_lm] = np.nan

    mask = np.any(np.all(np.isnan(landmark_imcoords), axis=0), axis=-1)
    landmark_vectors_check = np.nansum(camerasystem.get_camera_lines(landmark_imcoords)[0], axis=0)
    landmark_vectors_check[mask] = np.nan
    landmark_vectors_norm = landmark_vectors / np.linalg.norm(landmark_vectors, axis=-1)[..., np.newaxis]
    diff = np.abs(landmark_vectors_check - landmark_vectors_norm)
    if np.nanmax(diff) > 5e-3:
        warnings.warn(f"Projection is broken: {np.max(diff)}")

    return landmark_imcoords
