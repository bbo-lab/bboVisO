import jax.numpy as np
from calibcam.helper_ag import rodrigues_as_rotmats
import calibcam.camfunctions_ag as camfuncs_ag


# TODO initialize attitude with lowest variance vecs
def objfun_repro(landmarks, positions, oris_flu, landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts):
    # Generate landmark positions relative to camera system
    landmark_vecs = landmarks.reshape(1, -1, 3) - positions.reshape(-1, 1, 3)
    landmark_vecs = np.einsum('nji,nmj->nmi', rodrigues_as_rotmats(oris_flu), landmark_vecs)

    # Generate landmark positions relative to camera system
    landmark_coords = camfuncs_ag.map_world_board_to_cams(
        landmark_vecs,
        cam_Rs.reshape((cam_Rs.shape[0], 1, 1, 3, 3)),
        cam_ts.reshape((cam_Rs.shape[0], 1, 1, 3))
    )

    landmark_coords = camfuncs_ag.board_to_unit_sphere(landmark_coords)

    landmark_coords = camfuncs_ag.shift_camera(landmark_coords, xis)

    landmark_coords = camfuncs_ag.to_ideal_plane(landmark_coords)
    landmark_coords = camfuncs_ag.distort(landmark_coords, ks)
    landmark_coords = camfuncs_ag.ideal_to_sensor(landmark_coords, cam_matrices)

    errors2d = landmark_coords - landmark_imcoords

    return errors2d


def objfun_3d(landmarks, positions, oris_flu, landmark_vecs):
    dists = np.linalg.norm(landmarks.reshape(1, -1, 3) - positions.reshape(-1, 1, 3), axis=-1)

    errors3d = landmarks.reshape(1, -1, 3) - positions.reshape(-1, 1, 3) \
               - dists[..., np.newaxis] * np.einsum('nij,nmj->nmi', rodrigues_as_rotmats(oris_flu), landmark_vecs)

    # errors3d = errors3d + 1 / np.tanh(np.sum((landmarks[:2]-landmarks[3][np.newaxis, :])**2)) - 1

    return errors3d
