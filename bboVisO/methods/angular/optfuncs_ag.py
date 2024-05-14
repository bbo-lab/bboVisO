from jax import numpy as np

from bboVisO.methods.common_ag import world_to_camerasystem, camerasystem_to_cameras

def objfun(landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
           landmark_vecs, lm_close_pen_weight):
    # Transform landmarks into coordinate system of camera system
    landmark_coords = world_to_camerasystem(landmarks, positions, oris_flu)

    # Transform landmarks into coordinate system of individual cameras
    landmark_coords = camerasystem_to_cameras(landmark_coords, cam_rotvecs, cam_ts)

    # Normalize vectors
    landmark_coords /= np.linalg.norm(landmark_coords, axis=-1, keepdims=True)

    errors_ang = np.arccos(np.sum(landmark_coords * landmark_vecs, axis=-1))

    # We lightly penalize positions too close to landmarks
    penalty = np.zeros(len(positions))  # lm_close_pen_weight * calc_inverse_lm_pos_distance_sum(landmarks, positions)
    errors_ang += penalty[:, np.newaxis]

    return errors_ang
