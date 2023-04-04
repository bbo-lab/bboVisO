import time
import sys

import jax.config
import numpy as np
from scipy.optimize import least_squares, OptimizeResult
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

# Degree by which to pertubate camera vector angles
pertubation_deg = 0.4

opts = opts.get_opts()

# Simulate data
calibs = np.load(os.path.dirname(__file__) + "/../test/data/multicam_calibration.npy", allow_pickle=True)[()]["calibs"]
camerasystem = Camerasystem.from_calibs(calibs)

landmarks = sim.get_landmarks(8, radius=15)
positions, oris_flu = sim.get_screw_trajectory(length=10, start=(-5, 0, 0), direction=(1, 0, 0), radius=1, v0=1,
                                               period=1, fps=1)

# Generate relative positions
landmark_vecs, landmark_dists = sim.get_landmark_vectors(positions, oris_flu, landmarks)

# Pertubate
landmark_vecs_pertubated = sim.pertubate_landmark_vectors(landmark_vecs, pertubation_deg / 180 * np.pi)
assert pertubation_deg != 0 or np.all(landmark_vecs == landmark_vecs_pertubated), "Pertubation error"

# Reproject image coordinates
landmark_imcoords = sim.reproject_landmarks(landmark_vecs_pertubated * landmark_dists[..., np.newaxis],
                                            camerasystem)

# Optimize
# jac = viso.objfun_repro_jacobian_wrapper
jac = "2-point"
# jac = "3-point"

# Initializations, for alternating updates
landmarks_opt = None
positions_opt = None
oris_flu_opt = None
end_optim = False

# Reduce amount of steps for alternating updates
opts_optim = opts['optimization'].copy()
opts_optim["max_nfev"] = 10

# Previous positions for alternating update
positions_prev = np.full((landmark_imcoords.shape[1], 3), np.nan)

# Optimize all at once or alternate
optimize = [["oris", "positions", "landmarks"]]
# optimize = cycle(["oris", "positions", "landmarks"])  # Seems to converge to the same value

t = time.time()
precalc = viso.get_precalc()
for opt in optimize:
    # Always end with a full optimization
    if end_optim:
        opt = ["oris", "positions", "landmarks"]

    # If we are doing a full optimization, increase to full step count steps
    if isinstance(opt, list):
        opts_optim["max_nfev"] = opts["optimization"]["max_nfev"]

    # Run optimization
    print(f"Optimizing {opt}")
    vars_free, args = viso.make_optim_input(calibs, opts, landmarks_opt, positions_opt, oris_flu_opt, optimize=opt,
                                            precalc=precalc, landmark_imcoords=landmark_imcoords)

    min_result: OptimizeResult = least_squares(viso.objfun_repro_wrapper,
                                               vars_free,
                                               jac=jac,
                                               bounds=np.array([[-np.inf, np.inf]] * vars_free.size).T,
                                               args=[args],
                                               **opts_optim
                                               )

    landmarks_opt, positions_opt, oris_flu_opt = viso.unravel_vars(min_result.x, args)[:3]

    if end_optim:
        break

    max_pos_shift = np.max(np.abs(positions_prev - positions_opt))
    positions_prev = positions_opt

    print(f"Max pos shift {max_pos_shift}")
    if max_pos_shift < 1e-3:
        end_optim = True

print(f"Optimization took {time.time() - t} s")

# Realign result from position[0]-centered to landmark-centered
landmarks_aligned, positions_aligned, oris_flu_aligned = viso.align_results(landmarks,
                                                                            landmarks_opt, positions_opt, oris_flu_opt)

print(f"Max position error: {np.max(np.linalg.norm(positions-positions_aligned, axis=-1))}")

# Test for 3d optimization in postprocessing
# vars_free, args = viso.make_optim_input(calibs, opts, landmarks_opt, positions_opt, oris_flu_opt, optimize=opt,
#                                             precalc=precalc, landmark_imcoords=landmark_imcoords)
#
# min_result: OptimizeResult = least_squares(viso.objfun_3d_wrapper,
#                                            vars_free,
#                                            jac="2-point",
#                                            bounds=np.array([[-np.inf, np.inf]] * vars_free.size).T,
#                                            args=[args],
#                                            **opts_optim
#                                            )
#
# landmarks_opt, positions_opt, oris_flu_opt = viso.unravel_vars(min_result.x, args)[:3]
# landmarks_aligned, positions_aligned, oris_flu_aligned = viso.align_results(landmarks,
#                                                                             landmarks_opt, positions_opt, oris_flu_opt)
#
# print(f"Max position error: {np.max(np.linalg.norm(positions-positions_aligned, axis=-1))}")
pass
