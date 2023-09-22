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


def get_labels_from_frame(labels, frame_idx):
    frame_labels = {}

    for lb in labels:
        if frame_idx in labels[lb] and not np.all(np.isnan(labels[lb][frame_idx])):
            frame_labels[lb] = labels[lb][frame_idx]

    return frame_labels


def to_array(landmark_labels, detected_frames, detected_labels,
             time_bases, landmarks_identity=None):
    cams_n = len(next(iter(next(iter(landmark_labels.values())).values())))

    time_base = np.unique(np.concatenate(time_bases, axis=0))

    # cams x frames x landmarks x pix_coords
    landmark_imcoords = np.full((cams_n, len(time_base), len(detected_labels), 2), np.nan)
    for i_lm, lm in enumerate(detected_labels):
        if lm in landmark_labels:
            for i_fr, fr_idx in enumerate(detected_frames):
                # Map identities
                if landmarks_identity is not None and landmarks_identity[i_lm] is not None:
                    lm_idx = detected_labels.index(landmarks_identity[i_lm])
                else:
                    lm_idx = i_lm

                for i_cam in range(cams_n):
                    cam_time = time_bases[i_cam][i_fr]
                    time_base_idx = np.where(time_base == cam_time)[0][0]
                    if fr_idx in landmark_labels[lm]:
                        cam_coords = landmark_labels[lm][fr_idx][i_cam]
                        if not np.any(np.isnan(cam_coords)):
                            # if (fr_idx == 7529 and i_cam == 1) or (fr_idx == 7572 and i_cam == 0):
                            #     print(f"Writing cam {i_cam}, frame {fr_idx} to {time_base_idx}")
                            landmark_imcoords[i_cam, time_base_idx, lm_idx] = (
                                np.nanmean(np.array(
                                    [cam_coords, landmark_imcoords[i_cam, time_base_idx, lm_idx]]
                                ), axis=0))
    return landmark_imcoords, time_base
