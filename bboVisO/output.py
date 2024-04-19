import os
from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path
import pandas as pd

from bboVisO.helper import landmarks_to_array


def save_to_text(path, time_base, landmarks, positions, oris_flu, error_repro, opts, landmark_labels,
                 landmark_coordinates):
    error_repro = error_repro.copy()
    error_repro[error_repro==0] = np.nan
    cs = "\t"
    path_base = Path(path)

    # Positions
    csv_file = path_base / "cam_positions.csv"
    data = np.round(positions.T, decimals=3)
    df = pd.DataFrame()
    df['t_us'] = time_base
    df['x0'] = data[0]
    df['y0'] = data[1]
    df['z0'] = data[2]
    if csv_file.is_file():
        old_pd_table = pd.read_csv(csv_file, sep="\t", comment="#")
        # Remove previous output
        mask = (old_pd_table["t_us"] < opts["time_range"][0]) | (old_pd_table["t_us"] > opts["time_range"][-1])
        old_pd_table = old_pd_table[mask]
        df = pd.concat((df, old_pd_table))
    df.set_index("t_us", inplace=True)
    df.sort_values(by="t_us", inplace=True)
    df.to_csv(csv_file, sep='\t', index=True, header=list(df.columns))

    # Orientations
    csv_file = path_base / "cam_orientations.csv"
    data = np.round(R.from_rotvec(oris_flu).as_quat(canonical=True).T, decimals=3)
    df = pd.DataFrame()
    df['t_us'] = time_base
    df['x0'] = data[0]
    df['y0'] = data[1]
    df['z0'] = data[2]
    df['w0'] = data[3]
    if csv_file.is_file():
        old_pd_table = pd.read_csv(csv_file, sep="\t", comment="#")
        # Remove previous output
        mask = (old_pd_table["t_us"] < opts["time_range"][0]) | (old_pd_table["t_us"] > opts["time_range"][-1])
        old_pd_table = old_pd_table[mask]
        if np.any(mask):
            df = pd.concat((df, old_pd_table))
    df.set_index("t_us", inplace=True)
    df.sort_values(by="t_us", inplace=True)
    df.to_csv(csv_file, sep='\t', index=True, header=list(df.columns))

    # Errors
    absolute_error = np.sqrt(np.sum(error_repro ** 2, axis=-1))
    # Error
    csv_file = path_base / "error.csv"
    data = absolute_error.T
    data[data == 0] = np.nan

    df_dict = {'t_us': time_base,}
    for abs_err, landmark_label in zip(data, landmark_labels):
        df_dict[landmark_label] = abs_err
    df = pd.DataFrame(df_dict)

    if csv_file.is_file():
        old_pd_table = pd.read_csv(csv_file, sep="\t", comment="#")
        # Remove previous output
        mask = (old_pd_table["t_us"] < opts["time_range"][0]) | (old_pd_table["t_us"] > opts["time_range"][-1])
        old_pd_table = old_pd_table[mask]
        if np.any(mask):
            df = pd.concat((df, old_pd_table))
    df.set_index("t_us", inplace=True)
    df.sort_values(by="t_us", inplace=True)
    df.to_csv(csv_file, sep='\t', index=True, header=list(df.columns))

    # Scalar Error
    for name, func in {"maxerror": np.nanmax, "medianerror": np.nanmedian, "meanerror": np.nanmean, }.items():
        csv_file = path_base / (name+".csv")
        data = func(absolute_error, axis=-1).T
        data[data == 0] = np.nan
        df = pd.DataFrame()
        df['t_us'] = time_base
        df['error'] = data
        if csv_file.is_file():
            old_pd_table = pd.read_csv(csv_file, sep="\t", comment="#")
            # Remove previous output
            mask = (old_pd_table["t_us"] < opts["time_range"][0]) | (old_pd_table["t_us"] > opts["time_range"][-1])
            old_pd_table = old_pd_table[mask]
            if np.any(mask):
                df = pd.concat((df, old_pd_table))
        df.set_index("t_us", inplace=True)
        df.sort_values(by="error", inplace=True, ascending=False)
        df.to_csv(csv_file, sep='\t', index=True, header=list(df.columns))

    # Landmarks
    csv_file = path_base / "landmarks.csv"

    origin = np.array(landmark_coordinates["origin"]["position"])
    landmarks_all = landmarks_to_array(
        landmark_coordinates, landmark_labels)[0]
    landmarks_all -= origin
    landmarks_diff = landmarks - landmarks_all

    df = pd.DataFrame()
    df['landmark'] = landmark_labels
    df['x0'] = landmarks[:, 0]
    df['y0'] = landmarks[:, 1]
    df['z0'] = landmarks[:, 2]
    df['dx0'] = landmarks_diff[:, 0]
    df['dy0'] = landmarks_diff[:, 1]
    df['dz0'] = landmarks_diff[:, 2]
    df.set_index("landmark", inplace=True)
    df.to_csv(csv_file, sep='\t', index=True, header=list(df.columns))
