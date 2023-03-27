import numpy as np
from scipy.spatial.transform import Rotation as R


def get_screw_trajectory(length=10, start=(-5, 0, 0), direction=(1, 0, 0), radius=1, v0=1, period=1, fps=60):
    start = np.array(start).reshape((1, 3))
    direction = np.array(direction).reshape((1, 3))

    rot = R.align_vectors(direction, np.array([[1, 0, 0]]))[0]

    ts = np.linspace(0, length / v0, length * fps // v0)
    trig_arg = 2 * np.pi * ts / period

    positions = np.array(
        [
            v0 * ts,
            radius * np.sin(trig_arg),
            radius * np.cos(trig_arg),
        ]
    ).T

    oris_flu = np.array(
        [np.array([
            [1, 0, 0],
            [0, np.cos(a), np.sin(a)],
            [0, -np.sin(a), np.cos(a)],
        ]) for a in trig_arg]
    )

    positions = rot.apply(positions) + start

    return positions, R.from_matrix(oris_flu)


def get_landmarks(N, radius=50, offset=(0, 0, 10)):
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    landmarks = np.array(
        [
            radius * np.sin(ang),
            radius * np.cos(ang),
            np.zeros(ang.shape)
        ]
    ).T

    landmarks = landmarks + offset

    return landmarks


def get_landmark_vectors(positions, oris_flu, landmarks):
    landmark_vectors = landmarks.reshape((1, -1, 3)) - positions.reshape((-1, 1, 3))
    landmark_dists = np.linalg.norm(landmark_vectors, axis=2)
    landmark_vectors /= landmark_dists[..., np.newaxis]

    landmark_vectors = np.array([
        o.inv().apply(landmark_vectors[i]) for i, o in enumerate(oris_flu)
    ])

    return landmark_vectors, landmark_dists


def pertubate_landmark_vectors(landmark_vectors, sigma, seed=0):
    np.random.seed(seed)
    pertubations = sigma * np.random.randn(*landmark_vectors.shape[0:2])
    dirs = 2 * np.pi * np.random.rand(*landmark_vectors.shape[0:2])

    for i_t, (ps, ds) in enumerate(zip(pertubations, dirs)):
        for i_n, (p, d) in enumerate(zip(ps, ds)):
            orth_v = np.array([landmark_vectors[i_t, i_n, 1], -landmark_vectors[i_t, i_n, 0], 0])
            orth_v /= np.linalg.norm(orth_v)
            orth_v = R.from_rotvec(d * landmark_vectors[i_t, i_n]).apply(orth_v)
            landmark_vectors[i_t, i_n] = R.from_rotvec(p * orth_v).apply(landmark_vectors[i_t, i_n])

    return landmark_vectors
