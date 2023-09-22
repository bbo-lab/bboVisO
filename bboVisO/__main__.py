import jax.config
import numpy as np
import os.path
import argparse
from pathlib import Path
import sys

import yaml

from bboVisO.acm_labels import load as acm_load, get_labeled_frames
from svidreader import SVidReader

import bboVisO.opts
import bboVisO.viso as viso
from bboVisO.output import save_to_text
from bboVisO.visualisation import plot_frames

# single precision significantly increases the objective function at ground truth (1e-12 -> 1e-3)!
jax.config.update("jax_enable_x64", True)


def main():
    print("Starting")
    time_factor = int(1e6)  # us
    #
    # global opts, path
    # Some debug output
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--nodebug', required=False, default=False, action="store_true",
                        help="Decrease debug output")
    parser.add_argument('--from_yml', type=str, required=False, nargs=1,
                        help="Input configuration")
    parser.add_argument('--tetherless', required=False, default=False, action="store_true",
                        help="The YML file is in tetherless format")
    parser.add_argument('--text_out', type=str, required=True, nargs=1,
                        help="Folder for text output")
    parser.add_argument('--img_out', type=str, required=False, nargs=1,
                        help="Folder for image output")
    parser.add_argument('--img_out_n', type=int, required=False, nargs=1, default=[0],
                        help="Number of images to generate starting from highest maxerror")
    parser.add_argument('--check_mode', required=False, default=False, action="store_true",
                        help="Interpolate high error frames after last optimization")
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

    if args.check_mode:
        opts["check_mode"] = True
    opts["img_out_n"] = args.img_out_n[0]

    add_offset = args.add_offset[0]

    if args.from_yml:
        yml_path = Path(args.from_yml[0]).expanduser().resolve()

        if args.tetherless:
            import tetherless_data
            data = tetherless_data.load(yml_path)
        else:
            with open(yml_path, 'r') as file:
                data = yaml.safe_load(file)

        calibration = data["calibration"]
        acm_labels = data["acm_labels"]
        landmark_coordinates = data["landmark_coordinates"]
        if "videos" in data:
            videos = data["videos"]
        else:
            videos = None

        if "time_range" in data:
            opts["time_range"] = data["time_range"]*time_factor  # np.array([789000000, 791000000])+100/60*time_factor

        landmark_labels = acm_load(acm_labels)["labels"]
        landmark_frames = get_labeled_frames(landmark_labels).astype(int)
        print(len(landmark_frames))

        offsets = (np.array(data["cam_offsets"]) * time_factor).astype(np.uint64)
        offsets[1] += add_offset * time_factor
        fpss = data["cam_fps"]

        cam_time_bases = [
            (landmark_frames.astype(float) * float(time_factor)/float(fps)).astype(np.uint64) + offset
            for fps, offset in zip(fpss, offsets)
        ]

        # Adjust time that are just off by 1us from float rounding, to not create an additional measurement
        for i_cam in range(1, len(cam_time_bases)):
            for i_t, t in enumerate(cam_time_bases[i_cam]):
                mask = np.abs(cam_time_bases[0]-t) == 1
                s_mask = np.sum(mask)
                assert s_mask < 2, "A time seems to appear multiple times in the time base."
                if s_mask==0:
                    continue
                else:
                    # print(f"Rplacing {cam_time_bases[i_cam][i_t]} with {cam_time_bases[0][mask][0]}")
                    cam_time_bases[i_cam][i_t] = cam_time_bases[0][mask]

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
            path = Path(os.path.join(args.text_out[0], yml_path.stem)).expanduser().resolve().as_posix()
            os.makedirs(path, exist_ok=True)
            save_to_text(yml_path.stem, path, res_time_base, landmarks, positions, oris, error_repro, landmark_labels)

        if args.img_out and videos is not None:
            path = Path(os.path.join(args.img_out[0], yml_path.stem)).expanduser().resolve().as_posix()
            readers = [SVidReader(vid) for vid in videos]

            assert len(readers) == len(landmark_imcoords), "Number of readers must match first dm of landmark_imcoords"

            plot_frames(calibs,
                        res_time_base, oris, path, positions, landmark_imcoords, error_repro,
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
