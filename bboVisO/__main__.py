import jax.config
import numpy as np
import os.path
import argparse
from pathlib import Path
import importlib

import yaml

from svidreader import SVidReader

from bbo import label_lib

import bboVisO.opts
import bboVisO.viso as viso
from bboVisO.helper import get_cam_time_bases
from bboVisO.output import save_to_text
from bboVisO.visualisation import plot_frames

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)


def main():
    print("Starting")
    #
    # global opts, path
    # Some debug output
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--nodebug', required=False, default=False, action="store_true",
                        help="Decrease debug output")
    parser.add_argument('--check_gui', required=False, default=False, action="store_true",
                        help="Opens checker window")
    parser.add_argument('--from_yml', type=str, required=False, nargs=1,
                        help="Input configuration")
    parser.add_argument('--importer', required=False, default="default", type=str,
                        help="Importer to use")
    parser.add_argument('--text_out', type=str, required=True, nargs=1,
                        help="Folder for text output")
    parser.add_argument('--img_out', type=str, required=False, nargs=1,
                        help="Folder for image output")
    parser.add_argument('--img_out_n', type=int, required=False, nargs=1, default=[0],
                        help="Number of images to generate starting from highest maxerror")
    parser.add_argument('--time_range', type=float, required=False, nargs=2, default=[-1, -1],
                        help="Number of images to generate starting from highest maxerror")
    parser.add_argument('--add_offset', type=float, required=False, nargs=1, default=[0.0],
                        help="Offset to add to second cam, for subframe resolution tests")
    parser.add_argument('--simulation', required=False, default=False, action="store_true",
                        help="Simulate input data for testing purposes")
    parser.add_argument('--pertubation_deg', type=float, required=False, nargs=1, default=[0.2],
                        help="Landmark direction pertubation in simulation")
    parser.add_argument('--landmark_radius', type=float, required=False, nargs=1, default=[30],
                        help="Landmark distance in simulation")
    args = parser.parse_args()
    if not args.nodebug:
        viso.debug = True
    opts = bboVisO.opts.get_opts()

    opts["img_out_n"] = args.img_out_n[0]

    if args.text_out:
        opts["text_out"] = args.text_out[0]
    else:
        opts["text_out"] = False

    if args.from_yml:
        yml_path = Path(args.from_yml[0]).expanduser().resolve()

        importer = importlib.import_module("bboVisO.importer."+args.importer)
        data, opts = importer.load(yml_path, opts)

        opts["add_offset"] = np.zeros((len(data["cam_offsets"])))
        opts["add_offset"][1] = args.add_offset[0]

        if args.check_gui:
            from bboVisO.gui import checker
            checker.main(data, opts)
        else:
            calibration = data["calibration"]
            label_data = data["labels"]
            landmark_coordinates = data["landmark_coordinates"]
            if "videos" in data:
                videos = data["videos"]
            else:
                videos = None

            if not args.time_range[0] == -1:
                opts["time_range"] = np.array(args.time_range)
            elif "time_range" in data:
                opts["time_range"] = data["time_range"]

            landmark_labels = label_lib.load(label_data)
            landmark_frames = label_lib.get_labeled_frame_idxs(landmark_labels)

            # These are the times of landmark_frames
            # if is supplied "cam_time_bases", it is taken from here, else calculated from fps and offsets
            if "cam_time_bases" in data:
                cam_time_bases = [ct[landmark_frames] for ct in data["cam_time_bases"]]
            else:
                cam_time_bases = get_cam_time_bases(landmark_frames, data["cam_fps"], data["cam_offsets"], opts)
            cam_time_bases = [ct + ao for ct, ao in zip(cam_time_bases, opts["add_offset"])]

            if opts["time_range"][1] == -1:
                opts["time_range"][1] = max([c[-1] for c in cam_time_bases])

            from bboVisO.process import run  # noqa: E402
            calibs = np.load(calibration, allow_pickle=True)[()]["calibs"]

            landmarks, positions, oris, res_time_base, landmark_imcoords, error_repro, landmark_labels = \
                run(
                    landmark_labels=landmark_labels,
                    landmark_coordinates=landmark_coordinates,
                    calibs=calibs,
                    time_bases=cam_time_bases,
                    opts=opts)

            if args.text_out:
                path = Path(opts["text_out"]).expanduser().resolve() / data["name"]
                os.makedirs(path, exist_ok=True)
                save_to_text(path, res_time_base, landmarks, positions, oris, error_repro, landmark_labels, landmark_coordinates)

            if args.img_out and videos is not None:
                path = Path(opts["text_out"]).expanduser().resolve() / data["name"]
                readers = [SVidReader(vid) for vid in videos]

                assert len(readers) == len(landmark_imcoords), "Number of readers must match first dm of landmark_imcoords"

                plot_frames(calibs,
                            res_time_base, oris, path.as_posix(), positions, landmark_imcoords, error_repro,
                            landmarks, landmark_frames,
                            cam_time_bases, readers,
                            yml_path, opts)
    elif args.simulation:
        from bboVisO.simulation import run_simulated_data  # noqa: E402

        landmarks, positions, oris, analysis_frames = run_simulated_data(opts, pertubation_deg=args.pertubation_deg[0],
                                                                         landmark_radius=args.landmark_radius[0])
    else:
        raise Exception("Data or simulation needs to be provided")
    pass


if __name__ == '__main__':
    main()
