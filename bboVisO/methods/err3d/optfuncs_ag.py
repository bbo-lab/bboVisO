from jax import numpy as np

from calibcam.helper_ag import rodrigues_as_rotmats


def objfun(landmarks, positions, oris_flu, landmark_vecs):
    dists = np.linalg.norm(landmarks.reshape(1, -1, 3) - positions.reshape(-1, 1, 3), axis=-1)

    errors3d = landmarks.reshape(1, -1, 3) - positions.reshape(-1, 1, 3) \
               - dists[..., np.newaxis] * np.einsum('nij,nmj->nmi', rodrigues_as_rotmats(oris_flu), landmark_vecs)

    # errors3d = errors3d + 1 / np.tanh(np.sum((landmarks[:2]-landmarks[3][np.newaxis, :])**2)) - 1

    return errors3d
