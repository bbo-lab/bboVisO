import os
from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path

from bboVisO.helper import landmarks_to_array


def save_to_text(path, time_base, landmarks, positions, oris_flu, error_repro,
                 landmark_labels=False,
                 landmark_coordinates=False):
    error_repro = error_repro.copy()
    error_repro[error_repro==0] = np.nan
    cs = "\t"
    path_base = Path(path)
    with open(path_base / "cam_positions.csv", "w") as file:
        file.write(f't_us{cs}x0{cs}y0{cs}z0\n')
        for p, f in zip(positions, time_base):
            file.write(f'{f}{cs}{cs.join(np.char.mod("%f", p))}\n')

    with open(path_base / "cam_orientations.csv", "w") as file:
        file.write(f't_us{cs}x0{cs}y0{cs}z0{cs}w0\n')
        for o, f in zip(oris_flu, time_base):
            file.write(f'{f}{cs}{cs.join(np.char.mod("%f", R.from_rotvec(o).as_quat()))}\n')

    abserr = np.sqrt(np.sum(error_repro ** 2, axis=-1))
    abserr[abserr == 0] = np.nan

    with open(path_base / "error.csv", "w") as file:
        for o, f in zip(abserr, time_base):
            file.write(f'{f}{cs}{cs.join(np.char.mod("%f", o))}\n')

    aggr_error = np.nanmax(abserr, axis=-1)
    sort_idxs = np.argsort(aggr_error)[::-1]
    with open(path_base / "maxerror.csv", "w") as file:
        for e, f in zip(aggr_error[sort_idxs], time_base[sort_idxs]):
            file.write(f'{f}{cs}{e}\n')

    aggr_error = np.nanmean(abserr, axis=-1)
    sort_idxs = np.argsort(aggr_error)[::-1]
    with open(path_base / "meanerror.csv", "w") as file:
        for e, f in zip(aggr_error[sort_idxs], time_base[sort_idxs]):
            file.write(f'{f}{cs}{e}\n')

    aggr_error = np.nanmedian(abserr, axis=-1)
    sort_idxs = np.argsort(aggr_error)[::-1]
    with open(path_base / "medianerror.csv", "w") as file:
        for e, f in zip(aggr_error[sort_idxs], time_base[sort_idxs]):
            file.write(f'{f}{cs}{e}\n')

    if landmark_labels:
        if landmark_coordinates:
            origin = np.array(landmark_coordinates["origin"]["position"])

            landmarks_all = landmarks_to_array(
                landmark_coordinates, landmark_labels)[0]
            landmarks_all -= origin
            landmarks_diff = landmarks - landmarks_all

        with open(path_base / "landmarks.csv", "w") as file:
            file.write(f'landmark{cs}x0{cs}y0{cs}z0')
            if landmark_coordinates:
                file.write(f'{cs}dx0{cs}dy0{cs}dz0\n')
            else:
                file.write('\n')
            for i_lm, landmark in enumerate(landmarks):
                file.write(f'{landmark_labels[i_lm]}{cs}{cs.join(np.char.mod("%f", landmark))}')
                if landmark_coordinates:
                    file.write(f'{cs}{cs.join(np.char.mod("%f", landmarks_diff[i_lm]))}\n')
                else:
                    file.write('\n')
