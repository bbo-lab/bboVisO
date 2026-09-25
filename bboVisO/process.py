import warnings
import logging

import numpy as np

import bboVisO.viso as viso
import bboVisO.methods.repro

from bboVisO.helper import (landmarks_to_array)
from bboVisO.interpolation import interpolate_imcoords

from bbo import label_lib

# Some debug output
viso.debug = True
logger = logging.getLogger(__name__)

def run(landmark_labels, landmark_coordinates, calibs, time_bases, opts):
    # Find all usable landmarks that are labeled
    landmark_names = [ln for ln in landmark_coordinates["landmarks"] if landmark_coordinates["landmarks"][ln].get("use", False)]
    landmark_labels = label_lib.update(landmark_labels, do_purge_nans=True)
    landmark_label_names = sorted(label_lib.get_labels(landmark_labels, allow_empty=False))
    tracking_landmark_names = landmark_names  # list(set(landmark_names).intersection(set(landmark_label_names)))

    # Extract landmark coordiantes as array
    landmark_array, landmarks_far, landmarks_identity, landmarks_optimize, landmarks_weights = (
        landmarks_to_array(landmark_coordinates, tracking_landmark_names))
    # Subtract origin
    origin = np.array(landmark_coordinates["origin"]["position"])
    landmark_array -= origin
    landmarks_use = np.full(len(tracking_landmark_names), True, dtype=bool)

    # Find frames with detections from tracking_landmark_names
    # Might still include all nans
    detected_frames = label_lib.get_labeled_frame_idxs(landmark_labels, label_set=tracking_landmark_names)
    detected_times = [t[detected_frames] for t in time_bases]
    # Convert labels dict to array. This will result in an n_cam x n_times x n_landmarks x 2 array.
    # If frame times between cameras do not align, the labels of respective other cameras will be nan
    landmark_imcoords, time_base = label_lib.to_numpy(landmark_labels,
                                                      extract_labels=tracking_landmark_names,
                                                      extract_frame_idxs=detected_frames,
                                                      extract_inexistent_labels=True,
                                                      time_bases=time_bases,
                                                      time_bases_complete=True,
                                                      label_identity=landmarks_identity,
                                                      strip_nans=True)

    # Disable cams if configured
    landmark_imcoords[opts["ignore_cams"]] = np.nan

    # Cut down to desired time range
    time_base_mask = (opts["time_range"][0] <= time_base) & (time_base <= opts["time_range"][1])
    landmark_imcoords = landmark_imcoords[:, time_base_mask]
    time_base = time_base[time_base_mask]

    # Interpolate landmark image coordinates and vectors over timepoints of other, non-aligned cameras
    landmark_imcoords_orig = landmark_imcoords.copy()
    # landmark_imcoords, landmark_vecs, time_base = interpolate_imcoords_rebase(landmark_imcoords_orig, time_base,
    #                                                                           calibs,
    #                                                                           1.01*np.median(np.diff(time_bases)),
    #                                                                           time_base, time_bases)
    landmark_imcoords, landmark_vecs, time_base, imcoords_interp_mask = (
        interpolate_imcoords(landmark_imcoords_orig, time_base, calibs, sub_time_bases=detected_times))

    # Cut down to points with sufficient landmarks per opts["min_labels"]. Less than 4 is ill-defined
    sufficient_mask = np.sum(~np.isnan(landmark_imcoords[:, :, :, 0]), axis=(0, 2)) >= opts["min_labels"]
    time_base = time_base[sufficient_mask]
    landmark_imcoords = landmark_imcoords[:, sufficient_mask]
    imcoords_interp_mask = imcoords_interp_mask[:, sufficient_mask]
    landmark_vecs = landmark_vecs[:, sufficient_mask]

    logger.log(logging.INFO, f"Optimizing {landmark_imcoords.shape[1]} frames with {landmark_imcoords.shape[2]} landmarks.\n"
                             f"{np.count_nonzero(~np.isnan(landmark_imcoords_orig))/2} labels before interpolation,\n"
                             f"{np.count_nonzero(~np.isnan(landmark_imcoords))/2} labels after interpolation.")

    # ====== Section for in-place testing ======
    # Change Jacobian
    # opts_optim["jac"] = viso.objfun_repro_jacobian_wrapper
    # opts_optim["jac"] = "3-point"

    # Inject manual initialization
    positions_opt = None
    oris_flu_opt = None
    landmarks_opt = landmark_array.copy()
    # ====== /Section for in-place testing ======

    remove_random_lms_idxs = False
    if opts["remove_random_lms"]>0:
        landmarks_use_idxs = np.where(landmarks_use)[0]
        remove_random_lms_idxs = landmarks_use_idxs[np.random.randint(0,landmarks_use_idxs.size,size=(opts["remove_random_lms"],))]
        landmarks_use[remove_random_lms_idxs] = False

    landmarks_mask = landmarks_use


    landmarks_opt_mask, positions_opt, oris_flu_opt, calibs = \
        viso.infer_position_attitude(time_base, calibs, opts,
                                     landmark_imcoords=landmark_imcoords[:, :, landmarks_mask],
                                     landmark_vecs=landmark_vecs[:, :, landmarks_mask],
                                     landmarks_init=landmarks_opt[landmarks_mask],
                                     positions_init=positions_opt,
                                     oris_flu_init=oris_flu_opt,
                                     landmarks_optimize=landmarks_optimize[landmarks_mask],
                                     landmarks_far=landmarks_far[landmarks_mask],
                                     landmarks_weights=landmarks_weights[landmarks_mask],
                                     imcoords_interp_mask=imcoords_interp_mask[:, :, landmarks_mask],
                                     )
    landmarks_opt[landmarks_mask] = landmarks_opt_mask

    if remove_random_lms_idxs:
        landmarks_opt[remove_random_lms_idxs] = np.nan

    error_repro = bboVisO.methods.repro.optfuncs.objfun_runner(
        landmarks_opt[landmarks_mask], positions_opt, oris_flu_opt,
        landmark_imcoords[:, :, landmarks_mask], calibs, opts["iterations"][-1])
    error_repro[error_repro == 0] = np.nan
    print(f"Optimized repro cost: {0.5 * np.sum(error_repro ** 2)}")

    abserr = np.sqrt(np.sum(error_repro ** 2, axis=-1))
    error_n = min(len(time_base), 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        print(f"Highest mean error time: "
              f"{time_base[np.argpartition(np.nanmean(abserr, axis=-1), -error_n)[-error_n:]]}")
        print(f"Highest max error time: "
              f"{time_base[np.argpartition(np.nanmax(abserr, axis=-1), -error_n)[-error_n:]]}")

    landmarks, positions, oris_flu = (landmarks_opt, positions_opt, oris_flu_opt)

    return landmarks, positions, oris_flu, time_base, landmark_imcoords, error_repro, tracking_landmark_names, calibs
