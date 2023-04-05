import sys

import jax.config
import numpy as np
from itertools import cycle
import os.path

import bboVisO.opts as opts
import bboVisO.simulation as sim
import bboVisO.viso as viso
from calibcamlib.camerasystem import Camerasystem

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)

# Some debug output
viso.debug = True

opts = opts.get_opts()

# Degree by which to pertubate camera vector angles
pertubation_deg = 0.4

calibs = np.load(os.path.dirname(__file__) + "/../test/data/multicam_calibration.npy", allow_pickle=True)[()][
    "calibs"]

# Simulate data
landmark_imcoords, landmarks, positions, oris_flu = sim.get_simulated_data(calibs,
                                                                           opts,
                                                                           pertubation_deg=pertubation_deg,
                                                                           n_landmarks=8,
                                                                           landmark_radius=15,
                                                                           traj_length=10,
                                                                           traj_start=(-5, 0, 0),
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

# Realign result from position[0]-centered to landmark-centered
landmarks_aligned, positions_aligned, oris_flu_aligned = viso.align_results(landmarks,
                                                                            landmarks_opt, positions_opt, oris_flu_opt)

print(f"Max position error: {np.max(np.linalg.norm(positions - positions_aligned, axis=-1))}")

pass
