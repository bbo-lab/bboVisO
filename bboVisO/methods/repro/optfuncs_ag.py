
from jax import numpy as np

from calibcam import camfunctions_ag as camfuncs_ag

from bboVisO.methods.common_ag import world_to_camerasystem, camerasystem_to_cameras, calc_inverse_lm_pos_distance_sum
import logging

logger = logging.getLogger(__name__)


def cap_errors(errors2d, error_cap):
    # Smoothly cap the 2D error norm of each landmark at error_cap (px), ~identity for norms << error_cap.
    # error_cap=inf disables capping. Smooth everywhere (no division by the norm), so safe for jit and autodiff.
    r2 = np.sum(errors2d ** 2, axis=-1, keepdims=True)
    return errors2d / np.sqrt(1 + r2 / error_cap ** 2)


def objfun(landmarks, positions, oris_flu, cam_rotvecs, cam_ts, landmark_imcoords,
           lm_weights, residual_args,
           cam_matrices, ks, xis,
           projection_model, lm_close_pen_weight):
    # Transform landmarks into coordinate system of camera system
    landmark_coords = world_to_camerasystem(landmarks, positions, oris_flu)

    # Transform landmarks into coordinate system of individual cameras
    landmark_coords = camerasystem_to_cameras(landmark_coords, cam_rotvecs, cam_ts)

    landmark_coords = camfuncs_ag.board_to_unit_sphere(landmark_coords)

    if projection_model is None:
        logger.log(logging.WARNING, "Deprecated: projection_model is None, defaulting to perspective")
        projection_model = "perspective"

    if projection_model == "fisheye_equidistant":
        landmark_coords = camfuncs_ag.to_ideal_plane_fisheye_equidistant(landmark_coords)
    elif projection_model == "perspective" or projection_model is None:
        landmark_coords = camfuncs_ag.shift_camera(landmark_coords, xis)
        landmark_coords = camfuncs_ag.to_ideal_plane(landmark_coords)
    else:
        raise ValueError(f"Unknown projection model: {projection_model}")


    landmark_coords = camfuncs_ag.distort(landmark_coords, ks)
    landmark_coords = camfuncs_ag.ideal_to_sensor(landmark_coords, cam_matrices)

    errors2d = landmark_coords - landmark_imcoords

    # Limit influence of single wrong landmarks, e.g. projected to enormous values outside the field of view
    errors2d = cap_errors(errors2d, residual_args["error_cap"])

    # Apply mask weights
    errors2d *= residual_args["mask_weight"]

    # Apply landmark weights
    errors2d *= lm_weights

    # We lightly penalize positions too close to landmarks
    penalty = lm_close_pen_weight * calc_inverse_lm_pos_distance_sum(landmarks, positions)
    errors2d += penalty[np.newaxis, :, np.newaxis, np.newaxis]

    return errors2d


