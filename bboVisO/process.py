import jax.config
import numpy as np

import bboVisO.viso as viso
import bboVisO.acm_labels as acm_labels
from calibcamlib import Camerasystem

from scipy.spatial.transform.rotation import Rotation as R

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)

# Some debug output
viso.debug = True


def run(landmark_labels, landmark_coordinates, calibs, time_bases, opts):
    # Names of all landmarks
    labels = sorted(list(landmark_coordinates["landmarks"].keys()))
    # Origin of resulting coordinate system
    origin = np.array(landmark_coordinates["origin"]["position"])

    # Frames with labels
    detected_frames = acm_labels.get_labeled_frames(landmark_labels)
    print(len(detected_frames))

    # Marked labels
    detected_labels = sorted(acm_labels.get_labels(landmark_labels))

    # All marked labels must correspond to a landmark, coordinates may be empty or nan
    assert set(detected_labels).issubset(labels), "Some detected labels are unknown"

    landmarks_all, landmarks_far, landmarks_identity, landmarks_optimize, landmarks_use = landmarks_to_array(
        landmark_coordinates, detected_labels)
    landmarks_all -= origin

    landmark_imcoords, time_base = acm_labels.to_array(landmark_labels, detected_frames, detected_labels,
                                                            time_bases=time_bases,
                                                            landmarks_identity=landmarks_identity)


    # Cut down to range actually filled with data
    data_mask = np.any(~np.isnan(landmark_imcoords[:, :, :, 0]), axis=(0, 2))
    data_idxs = np.where(data_mask)[0]
    landmark_imcoords = landmark_imcoords[:, data_idxs[0]:data_idxs[-1]+1]
    time_base = time_base[data_idxs[0]:data_idxs[-1]+1]

    # Cut down to desired time range
    time_base_mask = np.logical_and(opts["time_range"][0] <= time_base, time_base <= opts["time_range"][1])
    landmark_imcoords = landmark_imcoords[:, time_base_mask]
    time_base = time_base[time_base_mask]

    landmark_imcoords, time_base = interpolate(landmark_imcoords, time_base, time_base, calibs)

    # Cut down to points with sufficient landmarks
    sufficient_mask = np.sum(~np.isnan(landmark_imcoords[:, :, :, 0]), axis=(0, 2)) >= opts["min_labels"]

    time_base = time_base[sufficient_mask]
    landmark_imcoords = landmark_imcoords[:, sufficient_mask]

    # Optimize
    # Reduce amount of steps for alternating updates
    opts_optim = opts['optimization'].copy()
    # opts_optim["jac"] = viso.objfun_repro_jacobian_wrapper
    opts_optim["jac"] = "2-point"
    # opts_optim["jac"] = "3-point"

    landmarks_close_mean = np.nanmean(landmarks_all[~landmarks_far], axis=0)
    if np.any(~landmarks_far) and not np.any(np.isnan(landmarks_close_mean)):
        positions_opt = np.ones((len(time_base), 3)) * landmarks_close_mean
    else:
        positions_opt = None  # landmarks_real[np.newaxis, 1, ...].copy()
    oris_flu_opt = None
    landmarks_opt = landmarks_all.copy()

    # positions_opt = np.array([[28.50096, -89.76008,  -5.64328]])
    # oris_flu_opt = np.array([[1.53225326, -1.46680443, -1.89735577]])

    # Only optimize p&o for far away landmarks.
    if opts["far_first"]:
        landmarks_mask = landmarks_use & landmarks_far
        landmarks_opt_far, positions_opt, oris_flu_opt = \
            viso.infer_position_attitude(landmark_imcoords[:, :, landmarks_mask], calibs,
                                         opts_optim, opts,
                                         landmarks_init=landmarks_opt[landmarks_mask],
                                         positions_init=positions_opt,
                                         oris_flu_init=oris_flu_opt,
                                         optimize=[["positions", "oris"]],
                                         landmarks_optimize=landmarks_optimize[landmarks_mask]
                                         )
        landmarks_opt[landmarks_mask] = landmarks_opt_far

    # Only optimize p&o for close-by landmarks.
    if opts["close_first"]:
        landmarks_mask = landmarks_use & ~landmarks_far
        landmarks_opt_mask, positions_opt, oris_flu_opt = \
            viso.infer_position_attitude(landmark_imcoords[:, :, landmarks_mask], calibs,
                                         opts_optim, opts,
                                         landmarks_init=landmarks_opt[landmarks_mask],
                                         positions_init=positions_opt,
                                         oris_flu_init=oris_flu_opt,
                                         optimize=[["positions", "oris"]],
                                         landmarks_optimize=landmarks_optimize[landmarks_mask]
                                         )
        landmarks_opt[landmarks_mask] = landmarks_opt_mask

    # Optimize p&o, then p&o&l for all used landmarks
    landmarks_mask = landmarks_use
    landmarks_opt_mask, positions_opt, oris_flu_opt = \
        viso.infer_position_attitude(landmark_imcoords[:, :, landmarks_mask], calibs,
                                     opts_optim, opts,
                                     landmarks_init=landmarks_opt[landmarks_mask],
                                     positions_init=positions_opt,
                                     oris_flu_init=oris_flu_opt,
                                     optimize=[["positions", "oris"], ["positions", "oris", "landmarks"]],
                                     landmarks_optimize=landmarks_optimize[landmarks_mask]
                                     )
    landmarks_opt[landmarks_mask] = landmarks_opt_mask

    if opts["check_mode"]:
        positions_opt, oris_flu_opt = viso.interpolate_high_error(positions_opt, oris_flu_opt, calibs, opts,
                                                                  landmarks=landmarks_opt[landmarks_mask],
                                                                  optimize=["positions", "oris"],
                                                                  landmark_imcoords=landmark_imcoords[:, :,
                                                                                    landmarks_mask],
                                                                  landmarks_optimize=landmarks_optimize[landmarks_mask])

    vars_free_opt, args_opt = \
        viso.make_optim_input(calibs, opts, landmarks_opt[landmarks_mask], positions_opt,
                              oris_flu_opt, landmark_imcoords=landmark_imcoords[:, :, landmarks_mask],
                              optimize=[["positions", "oris", "landmarks"]],
                              landmarks_optimize=landmarks_optimize[landmarks_mask]
                              )[0:2]
    error_repro = viso.objfun_repro_wrapper(vars_free_opt, args_opt, no_ravel=True)
    print(f"Optimized repro cost: {0.5 * np.sum(error_repro ** 2)}")

    abserr = np.sqrt(np.sum(error_repro ** 2, axis=-1))
    error_n = min(len(time_base), 4)
    print(
        f"Highest mean error time: {time_base[np.argpartition(np.nanmean(abserr, axis=-1), -error_n)[-error_n:]]}")
    print(
        f"Highest max error time: {time_base[np.argpartition(np.nanmax(abserr, axis=-1), -error_n)[-error_n:]]}")

    landmarks, positions, oris_flu = (landmarks_opt, positions_opt, oris_flu_opt)

    # landmarks, positions, oris_flu = viso.align_results_by_perches(landmarks_real,
    #                                                                landmarks_opt, positions_opt, oris_flu_opt,
    #                                                                fix_indices=(0, 1))
    #
    # now = datetime.now()
    # create_wavefront_balls(landmarks[2:], 0.2, "/home/voit/tmp/" + now.strftime("%Y%m%d-%H%M%S") + "/" + now.strftime(
    #     "%Y%m%d-%H%M%S") + "-trees.obj")
    # create_wavefront_balls(landmarks[:2], 0.2, "/home/voit/tmp/" + now.strftime("%Y%m%d-%H%M%S") + "/" + now.strftime(
    #     "%Y%m%d-%H%M%S") + "-perches.obj")
    # create_wavefront_balls(positions, 0.1, "/home/voit/tmp/" + now.strftime("%Y%m%d-%H%M%S") + "/" + now.strftime(
    #     "%Y%m%d-%H%M%S") + "-positions.obj")
    # create_wavefront_cylinders(positions[:-1], positions[1:], 0.03,
    #                            "/home/voit/tmp/" + now.strftime("%Y%m%d-%H%M%S") + "/" + now.strftime(
    #                                "%Y%m%d-%H%M%S") + "-trajectory.obj")
    #
    # oris_flu_spR = R.from_rotvec(oris_flu)
    # dir_spR = oris_flu_spR * oris_flu_spR[7].inv() * R.from_rotvec(np.array([0, 0, 1]) * 40 / 180 * np.pi)
    # oris_dir_spRs = dir_spR.as_matrix()
    #
    # axis_R = oris_dir_spRs + positions.reshape((-1, 1, 3))
    # create_wavefront_cylinders(positions, axis_R[:, 0], 0.03,
    #                            "/home/voit/tmp/" + now.strftime("%Y%m%d-%H%M%S") + "/" + now.strftime(
    #                                "%Y%m%d-%H%M%S") + "-forward.obj")
    # create_wavefront_cylinders(positions, axis_R[:, 1], 0.03,
    #                            "/home/voit/tmp/" + now.strftime("%Y%m%d-%H%M%S") + "/" + now.strftime(
    #                                "%Y%m%d-%H%M%S") + "-left.obj")
    # create_wavefront_cylinders(positions, axis_R[:, 2], 0.03,
    #                            "/home/voit/tmp/" + now.strftime("%Y%m%d-%H%M%S") + "/" + now.strftime(
    #                                "%Y%m%d-%H%M%S") + "-up.obj")
    #

    return landmarks, positions, oris_flu, time_base, landmark_imcoords, error_repro, detected_labels


def landmarks_to_array(landmark_coordinates, detected_labels):
    # Build landmark coordinates
    landmarks_all = []  # Coordinates of landmarks, may be nan
    landmarks_optimize = []  # Should this coordinate be optimized by bboViso?
    landmarks_far = []  # Is it a far away landmark (used in first run)
    landmarks_use = []  # Should it be used at all?
    landmarks_identity = []  # Is it identical to another one?
    # Only extract landmarks taht are also marked
    for ln in detected_labels:
        #
        if "position" not in landmark_coordinates["landmarks"][ln] or \
                len(landmark_coordinates["landmarks"][ln]["position"]) == 0:
            landmark_coordinates["landmarks"][ln]["position"] = [np.nan, np.nan, np.nan]

        if "identity" in landmark_coordinates["landmarks"][ln]:
            lm_idx = detected_labels.index(landmark_coordinates["landmarks"][ln]["identity"])
            landmarks_all[lm_idx] = np.nanmean(np.array([landmarks_all[lm_idx],
                                                         landmark_coordinates["landmarks"][ln]["position"]]),
                                               axis=0)

            landmarks_all.append([np.nan, np.nan, np.nan])
            landmarks_optimize.append(True)
            landmarks_far.append(False)
            landmarks_use.append(False)
            landmarks_identity.append(landmark_coordinates["landmarks"][ln]["identity"])
            continue

        landmarks_identity.append(None)
        if not np.any(np.isnan(landmark_coordinates["landmarks"][ln]["position"])):
            landmarks_all.append(landmark_coordinates["landmarks"][ln]["position"])
            landmarks_optimize.append(landmark_coordinates["landmarks"][ln]["optimize"])
            landmarks_far.append(landmark_coordinates["landmarks"][ln]["far"])
        else:
            landmarks_all.append([np.nan, np.nan, np.nan])
            landmarks_optimize.append(True)
            landmarks_far.append(False)
        if "use" in landmark_coordinates["landmarks"][ln]:
            landmarks_use.append(landmark_coordinates["landmarks"][ln]["use"])
        else:
            landmarks_use.append(True)
    landmarks_optimize = np.array(landmarks_optimize)
    landmarks_far = np.array(landmarks_far)
    landmarks_use = np.array(landmarks_use)
    landmarks_all = np.array(landmarks_all)
    return landmarks_all, landmarks_far, landmarks_identity, landmarks_optimize, landmarks_use


def interpolate(array_2d, array_time_base, time_base, calibs):
    # TODO implement interpolation to a new time base time_base
    cs = Camerasystem.from_calibs(calibs)
    array_3d = cs.get_camera_lines(array_2d)[0]

    pretimes = array_time_base[1:] - array_time_base[:-1]

    array_3d_interp = (array_3d[:, :-2] * pretimes[:-1][np.newaxis, ..., np.newaxis, np.newaxis] +
                       array_3d[:, 2:] * pretimes[1:][np.newaxis, ..., np.newaxis, np.newaxis])
    array_3d_interp /= np.linalg.norm(array_3d_interp, axis=-1, keepdims=True)

    array_3d_ireg = array_3d[:, 1:-1]
    array_3d_ireg[np.isnan(array_3d_ireg)] = array_3d_interp[np.isnan(array_3d_ireg)]

    array_2d_interp = cs.project_dir(array_3d)
    array_2d_out = array_2d.copy()
    for i_cam in range(len(array_2d_out)):
        array_2d_out[i_cam][np.isnan(array_2d_out[i_cam])] = array_2d_interp[i_cam, i_cam][np.isnan(array_2d_out[i_cam])]


    return array_2d_out, array_time_base
