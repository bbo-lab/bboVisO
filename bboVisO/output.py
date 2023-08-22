import os
from scipy.spatial.transform import Rotation as R
import numpy as np


def save_to_text(name, path, analysis_frames, landmarks, positions, oris_flu, error_repro, landmark_labels):
    column_separator = "\t"
    path_base = os.path.join(path, name)
    with open(path_base + "_cam_positions.txt", "w") as file:
        for p, f in zip(positions, analysis_frames):
            file.write(f'{f}{column_separator}{column_separator.join(np.char.mod("%f", p))}\n')

    with open(path_base + "_cam_orientations.txt", "w") as file:
        for o, f in zip(oris_flu, analysis_frames):
            file.write(f'{f}{column_separator}{column_separator.join(np.char.mod("%f", R.from_rotvec(o).as_quat()))}\n')

    abserr = np.sqrt(np.sum(error_repro ** 2, axis=-1))

    with open(path_base + "_error.txt", "w") as file:
        for o, f in zip(abserr, analysis_frames):
            file.write(f'{f}{column_separator}{column_separator.join(np.char.mod("%f", o))}\n')

    aggr_error = np.nanmax(abserr, axis=-1)
    sort_idxs = np.argsort(aggr_error)[::-1]
    with open(path_base + "_maxerror.txt", "w") as file:
        for e, f in zip(aggr_error[sort_idxs], analysis_frames[sort_idxs]):
            file.write(f'{f}{column_separator}{e}\n')

    aggr_error = np.nanmean(abserr, axis=-1)
    sort_idxs = np.argsort(aggr_error)[::-1]
    with open(path_base + "_meanerror.txt", "w") as file:
        for e, f in zip(aggr_error[sort_idxs], analysis_frames[sort_idxs]):
            file.write(f'{f}{column_separator}{e}\n')

    with open(path_base + "_landmarks.txt", "w") as file:
        for i_lm, landmark in enumerate(landmarks):
            file.write(f'{landmark_labels[i_lm]}{column_separator}{column_separator.join(np.char.mod("%f", landmark))}\n')


def save_to_wavefront(name, path, landmarks, positions, oris_flu):
    pass
