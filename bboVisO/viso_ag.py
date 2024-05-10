import jax.numpy as np
from calibcam.helper_ag import rodrigues_as_rotmats
import calibcam.camfunctions_ag as camfuncs_ag


# TODO initialize attitude with lowest variance vecs
def objfun_repro(landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
                 landmark_imcoords, cam_matrices, ks, xis, lm_close_pen_weight):
    # Transform landmarks into coordinate system of camera system
    landmark_coords = landmarks.reshape(1, -1, 3) - positions.reshape(-1, 1, 3)
    landmark_coords = np.einsum('nji,nmj->nmi', rodrigues_as_rotmats(oris_flu), landmark_coords)

    # Transform landmarks into coordinate system of individual cameras
    landmark_coords = camfuncs_ag.map_world_board_to_cams(
        landmark_coords,
        rodrigues_as_rotmats(cam_rotvecs).reshape((cam_rotvecs.shape[0], 1, 1, 3, 3)),
        cam_ts.reshape((cam_rotvecs.shape[0], 1, 1, 3))
    )

    landmark_coords = camfuncs_ag.board_to_unit_sphere(landmark_coords)

    landmark_coords = camfuncs_ag.shift_camera(landmark_coords, xis)

    landmark_coords = camfuncs_ag.to_ideal_plane(landmark_coords)
    landmark_coords = camfuncs_ag.distort(landmark_coords, ks)
    landmark_coords = camfuncs_ag.ideal_to_sensor(landmark_coords, cam_matrices)
    # print("ideal2sensor", landmark_coords.shape, landmark_coords[3])
    # print("landmark_imcoords", landmark_imcoords.shape, landmark_imcoords[3])
    #
    # print("Repro landmark_coords: ",landmark_coords)
    # print("Repro landmark_imcoords: ", landmark_imcoords)

    errors2d = np.abs(landmark_coords - landmark_imcoords)

    # print("Repro errors2d: ", errors2d)

    # We lightly penalize positions too close to landmarks
    penalty = lm_close_pen_weight * calc_lm_close_penalty(landmarks, positions)
    # print("penalty: ", penalty)
    errors2d += penalty[np.newaxis, :, np.newaxis, np.newaxis]

    # raise Exception()

    return errors2d


def objfun_angular(landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
                   landmark_vecs, lm_close_pen_weight):
    # Transform landmarks into coordinate system of camera system
    landmark_coords = landmarks.reshape(1, -1, 3) - positions.reshape(-1, 1, 3)
    landmark_coords = np.einsum('nji,nmj->nmi', rodrigues_as_rotmats(oris_flu), landmark_coords)

    # Transform landmarks into coordinate system of individual cameras
    landmark_coords = camfuncs_ag.map_world_board_to_cams(
        landmark_coords,
        rodrigues_as_rotmats(cam_rotvecs).reshape((cam_rotvecs.shape[0], 1, 1, 3, 3)),
        cam_ts.reshape((cam_rotvecs.shape[0], 1, 1, 3))
    )

    # Normalize vectors
    landmark_coords /= np.linalg.norm(landmark_coords, axis=-1, keepdims=True)

    errors_ang = np.arccos(np.sum(landmark_coords * landmark_vecs, axis=-1))

    # We lightly penalize positions too close to landmarks
    penalty = np.zeros(len(positions))  # lm_close_pen_weight * calc_lm_close_penalty(landmarks, positions)
    errors_ang += penalty[:, np.newaxis]

    return errors_ang


def objfun_3d(landmarks, positions, oris_flu, landmark_vecs):
    dists = np.linalg.norm(landmarks.reshape(1, -1, 3) - positions.reshape(-1, 1, 3), axis=-1)

    errors3d = landmarks.reshape(1, -1, 3) - positions.reshape(-1, 1, 3) \
               - dists[..., np.newaxis] * np.einsum('nij,nmj->nmi', rodrigues_as_rotmats(oris_flu), landmark_vecs)

    # errors3d = errors3d + 1 / np.tanh(np.sum((landmarks[:2]-landmarks[3][np.newaxis, :])**2)) - 1

    return errors3d


def calc_lm_close_penalty(landmarks, positions):
    return np.sum(
        1 / np.linalg.norm(landmarks[np.newaxis] - positions[:, np.newaxis], axis=-1),
        axis=-1)
