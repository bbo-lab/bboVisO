import jax.config
import numpy as np
import os.path
import argparse
from pathlib import Path

from tetherlessextractor.data_object import Data
from bboVisO.acm_labels import load as acm_load
from svidreader import SVidReader

import bboVisO.opts as opts
import bboVisO.viso as viso
from bboVisO.output import save_to_text
from bboVisO.visualisation import plot_frames

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)


def main():
    global opts, path
    # Some debug output
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--nodebug', required=False, default=False, action="store_true", help="")
    parser.add_argument('--from_yml', type=str, required=False, nargs=1, help="")
    parser.add_argument('--text_out', type=str, required=False, nargs=1, help="")
    parser.add_argument('--img_out', type=str, required=False, nargs=1, help="")
    parser.add_argument('--img_out_n', type=int, required=False, nargs=1, default=0, help="")
    parser.add_argument('--check_mode', required=False, default=False, action="store_true", help="")
    parser.add_argument('--simulation', required=False, default=False, action="store_true", help="")
    parser.add_argument('--pertubation_deg', type=float, required=False, nargs=1, default=0.2, help="")
    parser.add_argument('--landmark_radius', type=float, required=False, nargs=1, default=30, help="")
    args = parser.parse_args()
    if not args.nodebug:
        viso.debug = True
    opts = opts.get_opts()

    if args.check_mode:
        opts["check_mode"] = True
    opts["img_out_n"] = args.img_out_n[0]

    if args.from_yml:
        yml_path = Path(args.from_yml[0]).expanduser().resolve()
        data_obj = Data.load(yml_path, {"storage": "/media/smb/soma-fs.ad01.caesar.de/bbo/"})
        data = data_obj.data
        from bboVisO.process import run  # noqa: E402

        calibration = data["battery_cycle"]["landmark_labels"]["calibration"]
        calibs = np.load(data["dataset"]["calibration_headcam"][calibration], allow_pickle=True)[()]["calibs"]

        offsets = data_obj.get_headcam_frame_offsets()

        landmarks, positions, oris, analysis_frames, landmark_imcoords, error_repro, landmark_labels = \
            run(
                landmark_labels=acm_load(data["battery_cycle"]["landmark_labels"]["file"])["labels"],
                landmark_coordinates=data["battery_cycle"]["landmark_coordinates"],
                calibs=calibs,
                offsets=offsets,
                opts=opts)

        if args.text_out:
            path = Path(os.path.join(args.text_out[0], yml_path.stem)).expanduser().resolve().as_posix()
            os.makedirs(path, exist_ok=True)
            save_to_text(yml_path.stem, path,
                         analysis_frames, landmarks, positions, oris, error_repro, landmark_labels)

        if args.img_out:
            path = Path(os.path.join(args.img_out[0], yml_path.stem)).expanduser().resolve().as_posix()
            readers = [SVidReader(data["battery_cycle"]["headcam"][vidname]["video"])
                       for vidname in data["battery_cycle"]["landmark_labels"]["videos"]]

            assert len(readers) == len(landmark_imcoords), "Number of readers must match first dm of landmark_imcoords"

            plot_frames(analysis_frames, calibs, landmark_imcoords, landmarks, error_repro, offsets, oris, path,
                        positions, readers, yml_path, opts)
    elif args.simulation:
        from bboVisO.simulation import run_simulated_data  # noqa: E402

        landmarks, positions, oris, analysis_frames = run_simulated_data(opts, pertubation_deg=args.pertubation_deg[0],
                                                                         landmark_radius=args.landmark_radius[0])
    else:
        raise Exception("Data or simulation needs to be provided")
    pass


if __name__ == '__main__':
    main()
