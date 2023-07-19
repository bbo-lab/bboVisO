import warnings
import os.path

import numpy as np
from scipy.spatial.transform import Rotation as R

from calibcamlib.camerasystem import Camerasystem

import bboVisO.viso as viso
from bboVisO import viso as viso


def get_simulated_data(calibs, opts,
                       pertubation_deg=0, n_landmarks=8, landmark_radius=15, landmark_offset=(0, 0, 10),
                       landmarks_add=None,
                       traj_length=10, traj_start=(-5, 0, 2), traj_direction=(1, 0, 0), traj_radius=1, traj_v0=1,
                       traj_period=1, fps=1):

    camerasystem = Camerasystem.from_calibs(calibs)

    landmarks = get_landmarks(n_landmarks, radius=landmark_radius, offset=landmark_offset)

    if landmarks_add is not None:
        landmarks = np.vstack((landmarks_add, landmarks))

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
                                                    landmark_imcoords=landmark_imcoords_perturbed)[0:2]
    error_repro = viso.objfun_repro_wrapper(vars_free_opt, args_opt)
    error_3d_repro = viso.objfun_3d_wrapper(vars_free_opt, args_opt)

    vars_free_opt, args_opt = viso.make_optim_input(calibs, opts, landmarks, positions,
                                                    oris_flu.as_rotvec(),
                                                    landmark_vecs=landmark_vecs_perturbed)[0:2]
    error_3d = viso.objfun_3d_wrapper(vars_free_opt, args_opt)

    cost = 0.5 * np.sum(error_repro ** 2)
    print(f"Ground truth repro cost: {cost}")
    print(f"Ground truth 3d cost (with reprojected image coords): {0.5 * np.sum(error_3d_repro ** 2)}")
    print(f"Ground truth 3d cost (with original landmark vectors): {0.5 * np.sum(error_3d ** 2)}")

    return landmark_imcoords_perturbed, landmarks, positions, oris_flu, cost


def get_screw_trajectory(length, start, direction, radius, v0, period, fps, pertubate_pos=True, pertubate_ori=True):
    start = np.array(start).reshape((1, 3))
    direction = np.array(direction).reshape((1, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

    oris_flu = R.from_matrix(oris_flu)

    if pertubate_pos:
        if isinstance(pertubate_pos, bool):
            pertubate_pos = 0.3
        np.random.seed(0)
        positions += (np.random.rand(*positions.shape)-0.5)*2*pertubate_pos

    if pertubate_ori:
        if isinstance(pertubate_ori, bool):
            pertubate_ori = 0.1
        np.random.seed(0)
        oris_flu = oris_flu.as_rotvec()
        oris_flu += (np.random.rand(*oris_flu.shape) - 0.5) * 2 * pertubate_ori
        oris_flu = R.from_rotvec(oris_flu)

    return positions, oris_flu


def get_landmarks(N, radius, offset, pertubate=False):
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    landmarks = np.array(
        [
            radius * np.sin(ang),
            radius * np.cos(ang),
            np.zeros(ang.shape)
        ]
    ).T

    landmarks = landmarks + offset

    if pertubate:
        if isinstance(pertubate, bool):
            pertubate_pos = 0.3
        np.random.seed(0)
        landmarks += (np.random.rand(*landmarks.shape)-0.5)*2*pertubate

    return landmarks


def get_landmark_vectors(positions, oris_flu, landmarks):
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
    diff = np.arccos(np.sqrt(np.sum(landmark_vectors_check * landmark_vectors_norm, axis=-1)))

    if np.nanmax(diff) > 0.1/180*np.pi:
        print(f"Projection: Max angular error {np.nanmax(diff)/np.pi*180}° / Median angular error {np.nanmedian(diff)/np.pi*180}°")

    return landmark_imcoords


def calc_orientation_errors(orientations_real, orientations):
    orientations_real = R.from_rotvec(orientations_real).as_matrix()
    orientations_inv = R.from_rotvec(orientations).inv().as_matrix()

    rel_ori = np.einsum("nij,njk->nik", orientations_inv, orientations_real)
    rel_ori = R.from_matrix(rel_ori)
    angles = np.linalg.norm(rel_ori.as_rotvec(),axis=-1)

    return angles


def print_errors(landmarks, positions, oris_flu, landmarks_opt, positions_opt, oris_flu_opt):
    # Realign result from position[0]-centered to landmark-centered
    landmarks_aligned, positions_aligned, oris_flu_aligned = viso.align_results_by_landmarks(landmarks, landmarks_opt,
                                                                                             positions_opt,
                                                                                             oris_flu_opt)

    print(
        f"Max position error (landmark alignment): {np.max(np.linalg.norm(positions - positions_aligned, axis=-1))} m")
    print(
        f"Max landmark error (landmark alignment): {np.max(np.linalg.norm(landmarks - landmarks_aligned, axis=-1))} m")
    print(f"Max orientation error (landmark alignment): "
          f"{np.max(calc_orientation_errors(oris_flu.as_rotvec(), oris_flu_aligned)) / np.pi * 180} deg")
    print()

    landmarks_aligned, positions_aligned, oris_flu_aligned = viso.align_results_by_positions(positions, landmarks_opt,
                                                                                             positions_opt,
                                                                                             oris_flu_opt)
    print(
        f"Max position error (position alignment): {np.max(np.linalg.norm(positions - positions_aligned, axis=-1))} m")
    print(
        f"Max landmark error (position alignment): {np.max(np.linalg.norm(landmarks - landmarks_aligned, axis=-1))} m")
    print(f"Max orientation error (position alignment): "
          f"{np.max(calc_orientation_errors(oris_flu.as_rotvec(), oris_flu_aligned)) / np.pi * 180} deg")
    print()

    landmarks_aligned, positions_aligned, oris_flu_aligned = viso.align_results_by_perches(landmarks, landmarks_opt,
                                                                                           positions_opt, oris_flu_opt)
    print(f"Max position error (perch alignment): {np.max(np.linalg.norm(positions - positions_aligned, axis=-1))} m")
    print(f"Max landmark error (perch alignment): {np.max(np.linalg.norm(landmarks - landmarks_aligned, axis=-1))} m")
    print(f"Max orientation error (perch alignment): "
          f"{np.max(calc_orientation_errors(oris_flu.as_rotvec(), oris_flu_aligned)) / np.pi * 180} deg")
    print(f"Median position error (perch alignment): {np.median(np.linalg.norm(positions - positions_aligned, axis=-1))} m")
    print(f"Median landmark error (perch alignment): {np.median(np.linalg.norm(landmarks - landmarks_aligned, axis=-1))} m")
    print(f"Median orientation error (perch alignment): "
          f"{np.median(calc_orientation_errors(oris_flu.as_rotvec(), oris_flu_aligned)) / np.pi * 180} deg")
    print()


def run_simulated_data(opts, pertubation_deg=0.4, landmark_radius=50):
    calibs = np.load(os.path.dirname(__file__) + "/../test/data/multicam_calibration.npy", allow_pickle=True)[()][
        "calibs"]
    # Additional landmarks (perches etc.)
    landmarks_add = np.zeros((0, 3))
    landmarks_add = np.vstack((landmarks_add, np.array([[16, 0, 0]])))
    landmarks_add = np.vstack((landmarks_add, np.array([[-16, 0, 0]])))
    # landmarks_add = np.vstack((landmarks_add, np.array([[0, 15, 0]])))
    # landmarks_add = np.vstack((landmarks_add, np.array([[0, -15, 0]])))
    # landmarks_add = np.vstack((landmarks_add, np.array([[0, 0, 0]])))
    # Simulate data
    landmark_imcoords, landmarks, positions, oris_flu, cost_gt = get_simulated_data(calibs,
                                                                                        opts,
                                                                                        pertubation_deg=pertubation_deg,
                                                                                        n_landmarks=8,
                                                                                        landmark_radius=landmark_radius,
                                                                                        landmarks_add=landmarks_add,
                                                                                        traj_length=30,
                                                                                        traj_start=(-15, 0, 1.5),
                                                                                        traj_direction=(1, 0, 0),
                                                                                        traj_radius=1,
                                                                                        traj_v0=1,
                                                                                        traj_period=1,
                                                                                        fps=2
                                                                                        )
    # Optimize
    # Reduce amount of steps for alternating updates
    opts_optim = opts['optimization'].copy()
    opts_optim["max_nfev"] = 10
    # opts_optim["jac"] = viso.objfun_repro_jacobian_wrapper
    opts_optim["jac"] = "2-point"
    # opts_optim["jac"] = "3-point"
    landmarks_opt, positions_opt, oris_flu_opt = viso.infer_position_attitude(landmark_imcoords, calibs,
                                                                              opts_optim, opts,
                                                                              landmarks_init=landmarks,
                                                                              positions_init=positions,
                                                                              oris_flu_opt_init=oris_flu.as_rotvec(),
                                                                              )
    # Error output
    print_errors(landmarks, positions, oris_flu, landmarks_opt, positions_opt, oris_flu_opt)
    vars_free_opt, args_opt = viso.make_optim_input(calibs, opts, landmarks_opt, positions_opt,
                                                    oris_flu_opt,
                                                    landmark_imcoords=landmark_imcoords,
                                                    )[0:2]
    error_repro = viso.objfun_repro_wrapper(vars_free_opt, args_opt)
    cost_final = 0.5 * np.sum(error_repro ** 2)
    print(f"Optimized repro cost: {cost_final}")
    print(f"Optimized cost debts: {cost_final - cost_gt}")

    landmarks_aligned, positions_aligned, oris_flu_aligned = viso.align_results_by_perches(landmarks, landmarks_opt,
                                                                                           positions_opt, oris_flu_opt)

    return landmarks_aligned, positions_aligned, oris_flu_aligned, list(range(positions.shape[0]))
