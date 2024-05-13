from jax import numpy as np

from calibcam import camfunctions_ag as camfuncs_ag

from bboVisO.methods.common_ag import world_to_camerasystem, camerasystem_to_cameras, calc_inverse_lm_pos_distance_sum


def objfun(landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
           landmark_imcoords, cam_matrices, ks, xis, lm_close_pen_weight):
    # Transform landmarks into coordinate system of camera system
    landmark_coords = world_to_camerasystem(landmarks, positions, oris_flu)

    # Transform landmarks into coordinate system of individual cameras
    landmark_coords = camerasystem_to_cameras(landmark_coords, cam_rotvecs, cam_ts)

    landmark_coords = camfuncs_ag.board_to_unit_sphere(landmark_coords)

    landmark_coords = camfuncs_ag.shift_camera(landmark_coords, xis)

    landmark_coords = camfuncs_ag.to_ideal_plane(landmark_coords)
    landmark_coords = camfuncs_ag.distort(landmark_coords, ks)
    landmark_coords = camfuncs_ag.ideal_to_sensor(landmark_coords, cam_matrices)

    errors2d = np.abs(landmark_coords - landmark_imcoords)

    # We lightly penalize positions too close to landmarks
    penalty = lm_close_pen_weight * calc_inverse_lm_pos_distance_sum(landmarks, positions)
    errors2d += penalty[np.newaxis, :, np.newaxis, np.newaxis]

    return errors2d


