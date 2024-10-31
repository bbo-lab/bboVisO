import numpy as np
import os.path
import argparse
from pathlib import Path
import importlib
import sys

import yaml

from svidreader import SVidReader

import calibcamlib
from bbo import label_lib

import bboVisO.opts
import bboVisO.viso as viso
from bboVisO.helper import get_cam_time_bases
from bboVisO.output import save_to_text
from bboVisO.visualisation import plot_frames
from calibcamlib import yaml_helper

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
import jax
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
                        help="[DEPRECATED] Folder for image output")
    parser.add_argument('--img_out_n', type=int, required=False, nargs=1, default=[0],
                        help="[DEPRECATED] Number of images to generate starting from highest maxerror")
    parser.add_argument('--time_range', type=float, required=False, nargs=2, default=[-sys.maxsize, sys.maxsize],
                        help="Number of images to generate starting from highest maxerror")
    parser.add_argument('--add_offset', type=float, required=False, nargs=1, default=[0.0],
                        help="Offset to add to second cam, for subframe resolution tests")
    parser.add_argument('--simulation', required=False, default=False, action="store_true",
                        help="Simulate input data for testing purposes")
    parser.add_argument('--perturbation_deg', type=float, required=False, nargs=1, default=[0.2],
                        help="Landmark direction perturbation in simulation")
    parser.add_argument('--landmark_radius', type=float, required=False, nargs=1, default=[30],
                        help="Landmark distance in simulation")
    args = parser.parse_args()
    if not args.nodebug:
        viso.debug = True
    opts = bboVisO.opts.get_opts()

    # jax.config.update('jax_platform_name', opts['jax_backend'])

    opts["img_out_n"] = args.img_out_n[0]

    if args.text_out:
        opts["text_out"] = args.text_out[0]
    else:
        opts["text_out"] = False

    if args.from_yml:
        from bboVisO.process import run  # noqa: E402

        # Load config from config file
        config_file = Path(args.from_yml[0]).expanduser().resolve()
        importer = importlib.import_module("bboVisO.importer." + args.importer)
        data, opts = importer.load(config_file, opts)

        # Add offset from commandline (here because this requires number of cams, TODO: clean up)
        if len(opts["add_offset"]) == 0:
            opts["add_offset"] = np.zeros((len(data["cam_offsets"])))
        opts["add_offset"][1] = args.add_offset[0]

        if args.check_gui:
            # Start check GUI
            from bboVisO.gui import checker
            checker.main(data, opts)
        else:
            # Start odometry

            # Gather inputs
            calibration = Path(data["calibration"])
            label_data = data["labels"]
            landmark_coordinates = data["landmark_coordinates"]
            if "videos" in data:
                videos = data["videos"]
            else:
                videos = None

            # Set time range to process
            for i in range(2):
                if args.time_range[i] != -sys.maxsize and args.time_range[i] != sys.maxsize:
                    opts["time_range"][i] = args.time_range[i]
                elif "time_range" in data:
                    opts["time_range"][i] = data["time_range"][i]

            # Load labels
            landmark_labels = label_lib.load(label_data)
            landmark_frames = label_lib.get_labeled_frame_idxs(landmark_labels)

            # Assign times to landmark frames ...
            if "cam_time_bases" in data:
                # ... either from directly supplied frame times
                cam_time_bases = data["cam_time_bases"]
            else:
                # ... or from FPS
                cam_time_bases = get_cam_time_bases(landmark_frames[-1]+1, data["cam_fps"], data["cam_offsets"], opts)

            # Offset time bases of cameras
            cam_time_bases = [(ct + ao).astype(opts["time_dtype"]) for ct, ao in
                              zip(cam_time_bases, opts["add_offset"])]

            # Load calibration
            multicam_calibration = calibcamlib.Camerasystem.load_dict(calibration.as_posix())
            calibs = multicam_calibration["calibs"]

            # Run odometry
            landmarks, positions, oris, res_time_base, landmark_imcoords, error_repro, landmark_labels, calibs = \
                run(
                    landmark_labels=landmark_labels,
                    landmark_coordinates=landmark_coordinates,
                    calibs=calibs,
                    time_bases=cam_time_bases,
                    opts=opts)
            multicam_calibration["calibs"] = calibs

            # Generate text files from output
            if args.text_out:
                path = Path(opts["text_out"]).expanduser().resolve() / data["name"]
                os.makedirs(path, exist_ok=True)
                save_to_text(path, res_time_base, landmarks, positions, oris, error_repro, opts, landmark_labels,
                             landmark_coordinates)
                with open((path / "multicam_calibration_viso.yml"), "w") as yml_file:
                    yaml.dump(yaml_helper.numpy_collection_to_list(multicam_calibration),
                              yml_file, default_flow_style=True)
                np.save(path / "bboVisO_opts.npy", opts, allow_pickle=True)
                print(f"Saved to {path}")

            # Generate images from output (DEPRECATED: Use check GUI instead!)
            if args.img_out and videos is not None:
                path = Path(opts["text_out"]).expanduser().resolve() / data["name"]
                readers = [SVidReader(vid) for vid in videos]

                assert len(readers) == len(
                    landmark_imcoords), "Number of readers must match first dm of landmark_imcoords"

                plot_frames(calibs,
                            res_time_base, oris, path.as_posix(), positions, landmark_imcoords, error_repro,
                            landmarks, landmark_frames,
                            cam_time_bases, readers,
                            config_file, opts)
    elif args.simulation:
        from bboVisO.simulation import run_simulated_data  # noqa: E402

        run_simulated_data(opts, pertubation_deg=args.pertubation_deg[0],
                           landmark_radius=args.landmark_radius[0])
    else:
        raise Exception("Data or simulation needs to be provided")
    pass


if __name__ == '__main__':
    main()
