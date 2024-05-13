from jax import numpy as np

from calibcam.helper_ag import rodrigues_as_rotmats
from calibcam import camfunctions_ag as camfuncs_ag


def world_to_camerasystem(landmarks, positions, oris_flu):
    landmark_coords = landmarks.reshape(1, -1, 3) - positions.reshape(-1, 1, 3)
    landmark_coords = np.einsum('nji,nmj->nmi', rodrigues_as_rotmats(oris_flu), landmark_coords)
    return landmark_coords


def camerasystem_to_cameras(landmark_coords, cam_rotvecs, cam_ts):
    landmark_coords = camfuncs_ag.map_world_board_to_cams(
        landmark_coords,
        rodrigues_as_rotmats(cam_rotvecs).reshape((cam_rotvecs.shape[0], 1, 1, 3, 3)),
        cam_ts.reshape((cam_rotvecs.shape[0], 1, 1, 3))
    )
    return landmark_coords


def calc_inverse_lm_pos_distance_sum(landmarks, positions):
    return np.sum(
        1 / np.linalg.norm(landmarks[np.newaxis] - positions[:, np.newaxis], axis=-1),
        axis=-1)
