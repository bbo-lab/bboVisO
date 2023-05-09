import sys

import jax.config
import numpy as np
from itertools import cycle
import os.path

import bboVisO.opts as opts
import bboVisO.viso as viso
from calibcamlib.camerasystem import Camerasystem

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)

# Some debug output
viso.debug = True

opts = opts.get_opts()

# from bboVisO.simulation import run_simulated_data
# run_simulated_data(opts)

from bboVisO.experiment import run_experimental_data
run_experimental_data(opts)



