import os
from scipy.spatial.transform import Rotation as R
import numpy as np


def save_to_text(name, path, time_base, landmarks, positions, oris_flu, error_repro, landmark_labels):
    cs = "\t"
    path_base = os.path.join(path, name)
    with open(path_base + "_cam_positions.csv", "w") as file:
        file.write(f't_us{cs}x0{cs}y0{cs}z0\n')
        for p, f in zip(positions, time_base):
            file.write(f'{f}{cs}{cs.join(np.char.mod("%f", p))}\n')

    with open(path_base + "_cam_orientations.csv", "w") as file:
        file.write(f't_us{cs}x0{cs}y0{cs}z0{cs}w0\n')
        for o, f in zip(oris_flu, time_base):
            file.write(f'{f}{cs}{cs.join(np.char.mod("%f", R.from_rotvec(o).as_quat()))}\n')

    abserr = np.sqrt(np.sum(error_repro ** 2, axis=-1))

    with open(path_base + "_error.csv", "w") as file:
        for o, f in zip(abserr, time_base):
            file.write(f'{f}{cs}{cs.join(np.char.mod("%f", o))}\n')

    aggr_error = np.nanmax(abserr, axis=-1)
    sort_idxs = np.argsort(aggr_error)[::-1]
    with open(path_base + "_maxerror.csv", "w") as file:
        for e, f in zip(aggr_error[sort_idxs], time_base[sort_idxs]):
            file.write(f'{f}{cs}{e}\n')

    aggr_error = np.nanmean(abserr, axis=-1)
    sort_idxs = np.argsort(aggr_error)[::-1]
    with open(path_base + "_meanerror.csv", "w") as file:
        for e, f in zip(aggr_error[sort_idxs], time_base[sort_idxs]):
            file.write(f'{f}{cs}{e}\n')

    with open(path_base + "_landmarks.csv", "w") as file:
        file.write(f'landmark{cs}x0{cs}y0{cs}z0\n')
        for i_lm, landmark in enumerate(landmarks):
            file.write(f'{landmark_labels[i_lm]}{cs}{cs.join(np.char.mod("%f", landmark))}\n')


def save_to_wavefront(name, path, landmarks, positions, oris_flu):
    pass
