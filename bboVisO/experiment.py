import sys

import jax.config
import numpy as np
import os.path
from pathlib import Path

import bboVisO.viso as viso
from calibcamlib.camerasystem import Camerasystem

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)

# Some debug output
viso.debug = True

def run_experimental_data(opts):
    storage = Path("/media/voit/MyBookDuo/rsync/soma-fs/bbo/")
    calib_file = storage / "projects/junker-bird/camera_calibration/20230505/4pi_calibration/multicam_calibration.npy"
    cam_label_files = [
        storage / "analysis/pose/user/Kay/20230504_CADDX000016/labels.npz",
        storage / "analysis/pose/user/Kay/20230504_CADDX000002/labels.npz",
        ]

    calibs = np.load(calib_file, allow_pickle=True)[()]

    cam_labels = [np.load(f, allow_pickle=True)["arr_0"][()][0] for f in cam_label_files]
    label_frames_cam = [set(), set()]
    for i_c, cl in enumerate(cam_labels):
        for lk in cl["labels"]:
            label_frames_cam[i_c].update([int(i) for i in cl["labels"][lk].keys()])

    landmark_imcoords = None

    # Optimize
    # Reduce amount of steps for alternating updates
    opts_optim = opts['optimization'].copy()
    opts_optim["max_nfev"] = 10
    # opts_optim["jac"] = viso.objfun_repro_jacobian_wrapper
    opts_optim["jac"] = "2-point"
    # opts_optim["jac"] = "3-point"

    landmarks_opt, positions_opt, oris_flu_opt = viso.infer_position_attitude(landmark_imcoords, calibs,
                                                                              opts_optim, opts)

    vars_free_opt, args_opt = viso.make_optim_input(calibs, opts, landmarks_opt, positions_opt,
                                                    oris_flu_opt, landmark_imcoords=landmark_imcoords)[0:2]
    error_repro = viso.objfun_repro_wrapper(vars_free_opt, args_opt)
    print(f"Optimized repro cost: {0.5 * np.sum(error_repro ** 2)}")

    # landmarks_aligned, positions_aligned, oris_flu_aligned = viso.align_results_by_perches(landmarks, landmarks_opt,
    #                                                                                            positions_opt, oris_flu_opt)

    pass



