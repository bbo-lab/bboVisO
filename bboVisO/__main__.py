import jax.config
import numpy as np
import os.path
import argparse
from pathlib import Path

from tetherlessextractor import yaml_load
from bboVisO.acm_labels import load as acm_load
from svidreader import SVidReader

import bboVisO.opts as opts
import bboVisO.viso as viso
from bboVisO.output import save_to_text
from bboVisO.visualisation import plot_frames

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)

# Some debug output

parser = argparse.ArgumentParser(description="")
parser.add_argument('--nodebug', required=False, default=False, action="store_true", help="")
parser.add_argument('--from_yml', type=str, required=False, nargs=1, help="")
parser.add_argument('--text_out', type=str, required=False, nargs=1, help="")
parser.add_argument('--img_out', type=str, required=False, nargs=1, help="")
parser.add_argument('--simulation', required=False, default=False, action="store_true", help="")
parser.add_argument('--pertubation_deg', type=float, required=False, nargs=1, default=0.2, help="")
parser.add_argument('--landmark_radius', type=float, required=False, nargs=1, default=30, help="")

args = parser.parse_args()

if not args.nodebug:
    viso.debug = True

opts = opts.get_opts()

if args.from_yml:
    yml_path = Path(args.from_yml[0]).expanduser().resolve()
    data = yaml_load.load(yml_path, {"storage": "/media/smb/soma-fs.ad01.caesar.de/bbo/"})
    from bboVisO.process import run  # noqa: E402

    calibs = np.load(data["dataset"]["calibration_4pi"]["front"], allow_pickle=True)[()]["calibs"]
    landmarks, positions, oris, analysis_frames, landmark_imcoords, error_repro = \
        run(
            landmark_labels=acm_load(data["battery_cycle"]["landmark_labels"]["file"])["labels"],
            landmark_coordinates=data["battery_cycle"]["landmark_coordinates"],
            calibs=calibs,
            opts=opts)

    if args.text_out:
        path = Path(os.path.join(args.text_out[0], yml_path.stem)).expanduser().resolve().as_posix()
        os.makedirs(path, exist_ok=True)
        save_to_text(yml_path.stem, path,
                     analysis_frames, landmarks, positions, oris, error_repro)

    if args.img_out:
        path = Path(os.path.join(args.img_out[0], yml_path.stem)).expanduser().resolve().as_posix()
        os.makedirs(path, exist_ok=True)
        reader = SVidReader(data["battery_cycle"]["head4pi"]["front"]["video"])

        for fr_idx in analysis_frames:
            plot_frames(yml_path.stem, path, reader, analysis_frames, calibs, landmark_imcoords,
                        landmarks, positions, oris,
                        fr_idx)
elif args.simulation:
    from bboVisO.simulation import run_simulated_data  # noqa: E402

    landmarks, positions, oris, analysis_frames = run_simulated_data(opts, pertubation_deg=args.pertubation_deg[0],
                                                                     landmark_radius=args.landmark_radius[0])
else:
    raise Exception("Data or simulation needs to be provided")

pass
