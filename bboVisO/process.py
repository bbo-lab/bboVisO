import jax.config
import numpy as np

import bboVisO.viso as viso
import bboVisO.acm_labels as acm_labels

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)

# Some debug output
viso.debug = True


def run(landmark_labels, landmark_coordinates, calibs, opts):
    # Names of all landmarks
    labels = sorted(list(landmark_coordinates["landmarks"].keys()))
    # Origin of resulting coordinate system
    origin = np.array(landmark_coordinates["origin"]["position"])

    # Frames with labels
    detected_frames = acm_labels.get_labeled_frames(landmark_labels)
    # Marked labels
    detected_labels = sorted(acm_labels.get_labels(landmark_labels))

    # All marked labels must correspond to a landmark, coordinates may be empty or nan
    assert set(detected_labels).issubset(labels), "Some detected labels are unknown"

    # Build landmark coordinates
    landmarks_all = []  # Coordinates of landmarks, may be nan
    landmarks_optimize = []  # Should this coordinate be optimized by bboViso?
    landmarks_far = []  # Is it a far away landmark (used in first run)
    landmarks_use = []  # Should it be used at all?
    landmarks_identity = []  # Is it identical to another one?
    # Only extract landmarks taht are also marked
    for ln in detected_labels:
        #
        if "position" not in landmark_coordinates["landmarks"][ln] or\
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

    landmarks_all = np.array(landmarks_all) - origin
    landmarks_optimize = np.array(landmarks_optimize)
    landmarks_far = np.array(landmarks_far)
    landmarks_use = np.array(landmarks_use)


    # cams x frames x landmarks x pix_coords
    landmark_imcoords = np.full((len(calibs), len(detected_frames), len(detected_labels), 2), np.nan)

    for i_lm, lm in enumerate(detected_labels):
        if lm in landmark_labels:
            for i_fr, fr_idx in enumerate(detected_frames):
                if landmarks_identity[i_lm] is not None:
                    lm_idx = detected_labels.index(landmarks_identity[i_lm])
                else:
                    lm_idx = i_lm
                if fr_idx in landmark_labels[lm]:
                    if not np.any(np.isnan(landmark_labels[lm][fr_idx])):
                        landmark_imcoords[:, i_fr, lm_idx] = np.nanmean(np.array([
                                                                            landmark_labels[lm][fr_idx][0],
                                                                            landmark_imcoords[0, i_fr, lm_idx]
                                                                        ]), axis=0)

    sufficient_mask = np.sum(~np.isnan(landmark_imcoords[:, :, :, 0]), axis=(0, 2)) > 3
    # sufficient_mask[:] = False
    # sufficient_mask[277:280] = True
    # sufficient_mask = np.logical_and(detected_frames > 10000, detected_frames < 20000)

    detected_frames = detected_frames[sufficient_mask]
    landmark_imcoords = landmark_imcoords[:, sufficient_mask]

    # Optimize
    # Reduce amount of steps for alternating updates
    opts_optim = opts['optimization'].copy()
    opts_optim["max_nfev"] = 10
    # opts_optim["jac"] = viso.objfun_repro_jacobian_wrapper
    opts_optim["jac"] = "2-point"
    # opts_optim["jac"] = "3-point"

    landmarks_close_mean = np.nanmean(landmarks_all[~landmarks_far], axis=0)
    if np.any(~landmarks_far) and not np.any(np.isnan(landmarks_close_mean)):
        positions_opt = np.ones((len(detected_frames), 3)) * landmarks_close_mean
    else:
        positions_opt = None  # landmarks_real[np.newaxis, 1, ...].copy()
    # positions_opt[:, 2] += 0.3
    oris_flu_opt = None
    landmarks_opt = landmarks_all.copy()

    # Only optimize p&o for far away landmarks. Very close landmarks may through convergence off
    landmarks_mask = landmarks_use & landmarks_far
    landmarks_opt_far, positions_opt, oris_flu_opt = \
        viso.infer_position_attitude(landmark_imcoords[:, :, landmarks_mask], calibs,
                                     opts_optim, opts,
                                     landmarks_init=landmarks_opt[landmarks_mask],
                                     positions_init=positions_opt,
                                     oris_flu_opt_init=oris_flu_opt,
                                     optimize=[["positions", "oris"]],
                                     landmarks_optimize=landmarks_optimize[landmarks_mask]
                                     )
    landmarks_opt[landmarks_mask] = landmarks_opt_far

    # Optimize p&o for all used landmarks
    landmarks_mask = landmarks_use
    landmarks_opt_mask, positions_opt, oris_flu_opt = \
        viso.infer_position_attitude(landmark_imcoords[:, :, landmarks_mask], calibs,
                                     opts_optim, opts,
                                     landmarks_init=landmarks_opt[landmarks_mask],
                                     positions_init=positions_opt,
                                     oris_flu_opt_init=oris_flu_opt,
                                     optimize=[["positions", "oris"]],  # , "landmarks"
                                     landmarks_optimize=landmarks_optimize[landmarks_mask]
                                     )
    landmarks_opt[landmarks_mask] = landmarks_opt_mask

    # Optimize p&o&l for all used landmarks
    landmarks_mask = landmarks_use
    landmarks_opt_mask, positions_opt, oris_flu_opt = \
        viso.infer_position_attitude(landmark_imcoords[:, :, landmarks_mask], calibs,
                                     opts_optim, opts,
                                     landmarks_init=landmarks_opt[landmarks_mask],
                                     positions_init=positions_opt,
                                     oris_flu_opt_init=oris_flu_opt,
                                     optimize=[["positions", "oris", "landmarks"]],
                                     landmarks_optimize=landmarks_optimize[landmarks_mask]
                                     )
    landmarks_opt[landmarks_mask] = landmarks_opt_mask

    vars_free_opt, args_opt = \
        viso.make_optim_input(calibs, opts, landmarks_opt[landmarks_use], positions_opt,
                              oris_flu_opt, landmark_imcoords=landmark_imcoords[:, :, landmarks_use])[0:2]
    error_repro = viso.objfun_repro_wrapper(vars_free_opt, args_opt, no_ravel=True)
    print(f"Optimized repro cost: {0.5 * np.sum(error_repro ** 2)}")

    abserr = np.sqrt(np.sum(error_repro ** 2, axis=-1))
    print(f"Highest summed error frames: {detected_frames[np.argpartition(np.sum(abserr, axis=-1), -4)[-4:]]}")
    print(f"Highest max error frames: {detected_frames[np.argpartition(np.max(abserr, axis=-1), -4)[-4:]]}")

    landmarks, positions, oris_flu = (landmarks_opt, positions_opt, oris_flu_opt)

    print(np.sqrt(np.sum((landmarks-landmarks_all)**2, axis=-1)))
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

    return landmarks, positions, oris_flu, detected_frames, landmark_imcoords, error_repro
