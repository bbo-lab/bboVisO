import sys
import time
import jax.config
import numpy as np
from itertools import cycle
import os.path
import argparse

import bboVisO.opts as opts
import bboVisO.viso as viso

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)

# Some debug output

parser = argparse.ArgumentParser(description="")
parser.add_argument('--nodebug', required=False, default=False, action="store_true", help="")
parser.add_argument('--simulation', required=False, default=False, action="store_true", help="")
parser.add_argument('--pertubation_deg', type=float, required=False, nargs=1, default=0.2, help="")
parser.add_argument('--landmark_radius', type=float, required=False, nargs=1, default=50, help="")

args = parser.parse_args()

if not args.nodebug:
    viso.debug = True

opts = opts.get_opts()

if args.simulation:
    from bboVisO.simulation import run_simulated_data  # noqa: E402
    landmarks, positions, oris_flu = run_simulated_data(opts, pertubation_deg=args.pertubation_deg[0], landmark_radius=args.landmark_radius[0])
else:
    from bboVisO.experiment import run_experimental_data  # noqa: E402
    landmarks, positions, oris_flu, analysis_frames = run_experimental_data(opts)

pass