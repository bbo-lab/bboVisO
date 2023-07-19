import numpy as np
from pathlib import Path


def load(file_path):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    labels = np.load(file_path, allow_pickle=True)["arr_0"][()]

    return labels


def get_labeled_frames(labels):
    frames = set()
    for ln in labels:
        frames.update(set(labels[ln].keys()))

    return np.array(sorted(frames))


def get_labels(labels):
    return list(labels.keys())
