import sys

import jax.config
import numpy as np
import os.path
from pathlib import Path

import bboVisO.opts as opts
import bboVisO.viso as viso
from calibcamlib.camerasystem import Camerasystem

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)

# Some debug output
viso.debug = True

def main():
    storage = Path("/media/voit/MyBookDuo/rsync/soma-fs/bbo/")
    calib_file = storage / "projects/junker-bird/camera_calibration/20230505/4pi_calibration/multicam_calibration.npy"
    cam_labels = [
        storage / "analysis/pose/user/Kay/20230504_CADDX000002/labels.npz",
        storage / "analysis/pose/user/Kay/20230504_CADDX000016/labels.npz",
        ]

    opts = opts.get_opts()

    calibs = np.load(calib_file, allow_pickle=True)[()]

    # Additional landmarks (perches etc.)
    landmarks_add = np.zeros((0, 3))
    landmarks_add = np.vstack((landmarks_add, np.array([[16, 0, 0]])))
    landmarks_add = np.vstack((landmarks_add, np.array([[-16, 0, 0]])))
    # landmarks_add = np.vstack((landmarks_add, np.array([[0, 15, 0]])))
    # landmarks_add = np.vstack((landmarks_add, np.array([[0, -15, 0]])))
    # landmarks_add = np.vstack((landmarks_add, np.array([[0, 0, 0]])))

    # Simulate data
    landmark_imcoords, landmarks, positions, oris_flu = sim.get_simulated_data(calibs,
                                                                               opts,
                                                                               pertubation_deg=pertubation_deg,
                                                                               n_landmarks=8,
                                                                               landmark_radius=50,
                                                                               landmarks_add=landmarks_add,
                                                                               traj_length=30,
                                                                               traj_start=(-15, 0, 1.5),
                                                                               traj_direction=(1, 0, 0),
                                                                               traj_radius=1,
                                                                               traj_v0=1,
                                                                               traj_period=1,
                                                                               fps=1
                                                                               )

    # Optimize
    # Reduce amount of steps for alternating updates
    opts_optim = opts['optimization'].copy()
    opts_optim["max_nfev"] = 10
    # opts_optim["jac"] = viso.objfun_repro_jacobian_wrapper
    opts_optim["jac"] = "2-point"
    # opts_optim["jac"] = "3-point"

    landmarks_opt, positions_opt, oris_flu_opt = viso.infer_position_attitude(landmark_imcoords, calibs,
                                                                              opts_optim, opts)

    # Error output
    sim.print_errors(landmarks, positions, oris_flu, landmarks_opt, positions_opt, oris_flu_opt)

    vars_free_opt, args_opt = viso.make_optim_input(calibs, opts, landmarks_opt, positions_opt,
                                                    oris_flu_opt, landmark_imcoords=landmark_imcoords)[0:2]
    error_repro = viso.objfun_repro_wrapper(vars_free_opt, args_opt)
    print(f"Optimized repro cost: {0.5 * np.sum(error_repro ** 2)}")

    landmarks_aligned, positions_aligned, oris_flu_aligned = viso.align_results_by_perches(landmarks, landmarks_opt,
                                                                                               positions_opt, oris_flu_opt)

    pass



