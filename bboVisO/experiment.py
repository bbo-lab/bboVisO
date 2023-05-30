import sys

import jax.config
import numpy as np
import os.path
from pathlib import Path
from datetime import datetime

import bboVisO.viso as viso
import bboVisO.viso_ag as viso_ag
from bboVisO.geometry import align_positions_orientations
from bboVisO.visualisation import plot_frames
from calibcamlib.camerasystem import Camerasystem
from svidreader import SVidReader

from bboVisO.visualisation import create_wavefront_balls, create_wavefront_cylinders
from calibcamlib.camerasystem import Camerasystem

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)

# Some debug output
viso.debug = True


def run_experimental_data(opts):
    storage = Path("/media/smb/soma-fs.ad01.caesar.de/bbo/")
    # storage = Path("/media/voit/MyBookDuo/rsync/soma-fs/bbo/")

    landmarks_real = np.array([[317880.82, 5597769.70, 537.43],  # perch_field
                               [317857.75, 5597788.75, 539.85],  # perch_park
                               [317826.89, 5597861.37, 553.15],  # tree_1
                               [317893.22, 5597841.57, 549.15],  # tree_2
                               [317787.34, 5597772.47, 569.29],  # tree_3
                               [317754.36, 5597805.84, 566.60],  # tree_4
                               [317931.19, 5597745.06, 556.38],  # tree_5
                               [317888.48, 5597717.56, 554.36],  # tree_6
                               [317861.49, 5597755.07, 546.68],  # tree_7
                               [317903.83, 5597785.36, 541.46]])  # tree_8
    landmarks_real -= landmarks_real[1]
    # perch_diff = landmarks_real[0] - landmarks_real[1]

    calib_file = storage / "projects/junker-bird/camera_calibration/20230505/4pi_calibration/multicam_calibration.npy"
    cam_label_files = {
        # 0: storage / "analysis/pose/user/Paul/20230504_CADDX000016/labels.npz",
        1: storage / "analysis/pose/user/Kay/20230504_CADDX000002/labels.npz",
    }
    video_files = {
        # 0: storage / "bulk/bird_hellenthal_FrontField/20230502_20230505_FlightWindow/20230504/fourp2/back/VIDEO/CADDX000016.MP4",
        1: storage / "bulk/bird_hellenthal_FrontField/20230502_20230505_FlightWindow/20230504/fourp2/front/VIDEO/CADDX000002.MP4",
    }
    cam_offsets = {
        # 0: -11,
        1: 0,
    }

    calibs = np.load(calib_file.as_posix(), allow_pickle=True)[()]["calibs"]

    # Extract data from labels
    cam_labels = {}
    for cam_idx in cam_label_files:
        cam_labels[cam_idx] = np.load(cam_label_files[cam_idx].as_posix(), allow_pickle=True)["arr_0"][()]
    label_frames = set()
    labels = set()
    for cam_idx in cam_labels:
        for lk in cam_labels[cam_idx]["labels"]:
            label_frames.update([int(i) - cam_offsets[cam_idx] for i in cam_labels[cam_idx]["labels"][lk].keys()])
            labels.add(lk)
    label_frames = sorted(label_frames)
    labels = sorted(labels)

    analysis_frames = np.hstack((np.arange(840, 1301, 30), np.arange(5600, 6201, 30)))

    # cams x frames x landmarks x pix_coords
    landmark_imcoords = np.full((len(calibs), len(analysis_frames), len(labels), 2), np.nan)
    for cam_idx in cam_labels:
        for i_lm, lm in enumerate(labels):
            if lm in cam_labels[cam_idx]["labels"]:
                for i_fr, fr_idx in enumerate(analysis_frames):
                    if fr_idx + cam_offsets[cam_idx] in cam_labels[cam_idx]["labels"][lm]:
                        landmark_imcoords[cam_idx, i_fr, i_lm] = cam_labels[cam_idx]["labels"][lm][fr_idx + cam_offsets[cam_idx]]

    sufficient_mask = np.sum(~np.isnan(landmark_imcoords[:, :, :, 0]), axis=(0, 2)) > 3
    # sufficient_mask[:] = False
    # sufficient_mask[np.where(analysis_frames == 6020)[0]] = True

    analysis_frames = analysis_frames[sufficient_mask]
    landmark_imcoords = landmark_imcoords[:, sufficient_mask]

    # Optimize
    # Reduce amount of steps for alternating updates
    opts_optim = opts['optimization'].copy()
    opts_optim["max_nfev"] = 10
    # opts_optim["jac"] = viso.objfun_repro_jacobian_wrapper
    opts_optim["jac"] = "2-point"
    # opts_optim["jac"] = "3-point"

    positions_opt = None
    oris_flu_opt = None

    # landmarks_opt, positions_opt, oris_flu_opt = viso.infer_position_attitude(landmark_imcoords, calibs,
    #                                                                           opts_optim, opts,
    #                                                                           optimize=[["positions", "oris", "landmarks"]]
    #                                                                           )

    # Optimize on only far away points to get a good estimate on orientation
    landmarks_opt, positions_opt, oris_flu_opt = viso.infer_position_attitude(landmark_imcoords[:, :, 2:], calibs,
                                                                              opts_optim, opts,
                                                                              landmarks_opt=landmarks_real[2:],
                                                                              optimize=[["positions", "oris"]]
                                                                              )

    # pos_fix = landmarks_real[1] + 0.5 * perch_diff + np.array([0, 0, 5])
    # landmarks_opt, positions_opt, oris_flu_opt = viso.infer_position_attitude(landmark_imcoords, calibs,
    #                                                                           opts_optim, opts,
    #                                                                           landmarks_opt=landmarks_real,
    #                                                                           positions_opt=pos_fix.reshape((-1, 3)),
    #                                                                           optimize=[["oris"]]
    #                                                                           )

    # Optimize all, initialized with good orientation
    landmarks_opt, positions_opt, oris_flu_opt = viso.infer_position_attitude(landmark_imcoords, calibs,
                                                                              opts_optim, opts,
                                                                              landmarks_opt=landmarks_real,
                                                                              positions_opt=positions_opt,
                                                                              oris_flu_opt=oris_flu_opt,
                                                                              optimize=[["positions", "oris"]]
                                                                              )

    vars_free_opt, args_opt = viso.make_optim_input(calibs, opts, landmarks_opt, positions_opt,
                                                    oris_flu_opt, landmark_imcoords=landmark_imcoords)[0:2]
    error_repro = viso.objfun_repro_wrapper(vars_free_opt, args_opt, no_ravel=True)
    print(f"Optimized repro cost: {0.5 * np.sum(error_repro ** 2)}")

    landmarks, positions, oris_flu = (landmarks_opt, positions_opt, oris_flu_opt)
    # landmarks, positions, oris_flu = viso.align_results_by_perches(landmarks_real,
    #                                                                landmarks_opt, positions_opt, oris_flu_opt,
    #                                                                fix_indices=(0, 1))

    readers = {}
    for cam_idx in video_files:
        readers[cam_idx] = SVidReader(video_files[cam_idx])

    now = datetime.now()
    create_wavefront_balls(landmarks[2:], 0.2, "/home/voit/tmp/" + now.strftime("%Y%m%d-%H%M%S") + "/" + now.strftime(
        "%Y%m%d-%H%M%S") + "-trees.obj")
    create_wavefront_balls(landmarks[:2], 0.2, "/home/voit/tmp/" + now.strftime("%Y%m%d-%H%M%S") + "/" + now.strftime(
        "%Y%m%d-%H%M%S") + "-perches.obj")
    create_wavefront_balls(positions, 0.1, "/home/voit/tmp/" + now.strftime("%Y%m%d-%H%M%S") + "/" + now.strftime(
        "%Y%m%d-%H%M%S") + "-positions.obj")
    create_wavefront_cylinders(positions[:-1], positions[1:], 0.03,
                               "/home/voit/tmp/" + now.strftime("%Y%m%d-%H%M%S") + "/" + now.strftime(
                                   "%Y%m%d-%H%M%S") + "-trajectory.obj")
    for fr_idx in analysis_frames:
        plot_frames(readers, cam_offsets, analysis_frames, calibs, landmark_imcoords, landmarks, positions, oris_flu,
                    fr_idx, save_path=f"/home/voit/tmp/"
                                      f"{now.strftime('%Y%m%d-%H%M%S')}/{now.strftime('%Y%m%d-%H%M%S')}-{fr_idx}.png")

    return landmarks, positions, oris_flu, analysis_frames
