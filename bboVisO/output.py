import os
from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path
import pandas as pd

from bboVisO.helper import landmarks_to_array


def save_to_text(path, time_base, landmarks, positions, oris_flu, error_repro, opts, landmark_labels,
                 landmark_coordinates=None, combine_transformation=False, pos_formatter=None, rot_formatter=None):
    error_repro = error_repro.copy()
    error_repro[error_repro == 0] = np.nan
    path_base = Path(path)

    if not combine_transformation:
        # Positions
        csv_file = path_base / "cam_positions.csv"
        data = np.round(positions.T, decimals=8)
        df = pd.DataFrame()
        df['t_us'] = time_base
        df['x'] = data[0]
        df['y'] = data[1]
        df['z'] = data[2]

        if pos_formatter is not None:
            for col in ['x', 'y', 'z']:
                df[col] = df[col].apply(pos_formatter)
        filter_and_save(csv_file, df, opts)

        # Orientations
        csv_file = path_base / "cam_orientations.csv"
        data = R.from_rotvec(oris_flu).as_quat(canonical=True).T
        df = pd.DataFrame()
        df['t_us'] = time_base
        df['x'] = data[0]
        df['y'] = data[1]
        df['z'] = data[2]
        df['w'] = data[3]
        if rot_formatter is not None:
            for col in ['x', 'y', 'z', 'w']:
                df[col] = df[col].apply(rot_formatter)
        filter_and_save(csv_file, df, opts)
    else:
        # Positions
        csv_file = path_base / "cam_transformations.csv"
        data = np.round(positions.T, decimals=8)
        df = pd.DataFrame()
        df['t_us'] = time_base
        df['x'] = data[0]
        df['y'] = data[1]
        df['z'] = data[2]
        data = R.from_rotvec(oris_flu).as_quat(canonical=True).T
        df['t_us'] = time_base
        df['rx'] = data[0]
        df['ry'] = data[1]
        df['rz'] = data[2]
        df['rw'] = data[3]
        if pos_formatter is not None:
            for col in ['x', 'y', 'z']:
                df[col] = df[col].apply(pos_formatter)
        if rot_formatter is not None:
            for col in ['rx', 'ry', 'rz', 'rw']:
                df[col] = df[col].apply(rot_formatter)
        filter_and_save(csv_file, df, opts)

    # Errors
    absolute_error = np.sqrt(np.nansum(error_repro ** 2, axis=(0,-1)))
    # Error
    csv_file = path_base / "error.csv"
    data = absolute_error.T
    data[data == 0] = np.nan

    df_dict = {'t_us': time_base, }
    for abs_err, landmark_label in zip(data, landmark_labels):
        df_dict[landmark_label] = abs_err
    df = pd.DataFrame(df_dict)

    filter_and_save(csv_file, df, opts)

    # Scalar Error
    for name, func in {"maxerror": np.nanmax, "medianerror": np.nanmedian, "meanerror": np.nanmean, }.items():
        csv_file = path_base / (name + ".csv")
        data = func(absolute_error, axis=-1).T
        data[data == 0] = np.nan
        df = pd.DataFrame()
        df['t_us'] = time_base
        df['error'] = data
        filter_and_save(csv_file, df, opts, sort="error", ascending=False)



    # Landmarks
    if landmark_coordinates is not None:
        csv_file = path_base / "landmarks.csv"

        origin = np.array(landmark_coordinates["origin"]["position"])
        landmarks_all = landmarks_to_array(
            landmark_coordinates, landmark_labels)[0]
        landmarks_all -= origin
        landmarks_diff = landmarks - landmarks_all

        df = pd.DataFrame()
        df['landmark'] = landmark_labels
        df['x'] = landmarks[:, 0]
        df['y'] = landmarks[:, 1]
        df['z'] = landmarks[:, 2]
        df['dx'] = landmarks_diff[:, 0]
        df['dy'] = landmarks_diff[:, 1]
        df['dz'] = landmarks_diff[:, 2]
        df.set_index("landmark", inplace=True)

        if pos_formatter is not None:
            for col in ['x', 'y', 'z']:
                df[col] = df[col].apply(pos_formatter)

        if rot_formatter is not None:
            for col in ['dx', 'dy', 'dz']:
                df[col] = df[col].apply(rot_formatter)

        df.to_csv(csv_file, sep='\t', index=True, header=list(df.columns))


def filter_and_save(csv_file, df, opts, sort="t_us", ascending=True):
    if csv_file.is_file():
        old_pd_table = pd.read_csv(csv_file, sep="\t", comment="#")
        # Remove previous output
        mask = (old_pd_table["t_us"] < opts["time_range"][0]) | (old_pd_table["t_us"] > opts["time_range"][-1])
        old_pd_table = old_pd_table[mask]
        if np.any(mask):
            df = pd.concat((df, old_pd_table))
    df.set_index("t_us", inplace=True)
    df.sort_values(by=sort, inplace=True, ascending=ascending)
    df.to_csv(csv_file, sep='\t', index=True, header=list(df.columns), float_format="%.3f")
