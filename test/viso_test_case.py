import unittest
import os
import sys
import numpy as np

from calibcamlib.camerasystem import Camerasystem

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from bboVisO.test import check_jacobian
import bboVisO.simulation as sim
import bboVisO.viso as viso
import bboVisO.opts as opts


class VisOTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.opts = opts.get_opts()

        self.calibs = np.load(os.path.dirname(__file__) + "/data/multicam_calibration.npy",
                              allow_pickle=True)[()]["calibs"]
        self.camerasystem = Camerasystem.from_calibs(self.calibs)

        self.landmarks = sim.get_landmarks(8, radius=15, offset=(0, 0, 10))
        self.positions, self.oris_flu = sim.get_screw_trajectory(length=10, start=(-5, 0, 0), direction=(1, 0, 0),
                                                                 radius=1, v0=1, period=1, fps=1)

        # landmark_vecs, landmark_dists = sim.get_landmark_vectors(positions, oris_flu, landmarks, normalize=True)
        # landmark_vecs_pertubated = sim.pertubate_landmark_vectors(landmark_vecs, 0 * 1 / 180 * np.pi)
        self.landmark_vecs, self.landmark_dists = sim.get_landmark_vectors(self.positions, self.oris_flu,
                                                                           self.landmarks)

        self.landmark_imcoords = sim.reproject_landmarks(self.landmark_vecs * self.landmark_dists[..., np.newaxis],
                                                         self.camerasystem)

    def pertubate_opt_quantities(self):
        landmarks_rand = self.landmarks + (np.random.rand(*self.landmarks.shape) - 0.5) * np.max(
            self.landmarks.shape)
        positions_rand = self.positions + (np.random.rand(*self.positions.shape) - 0.5) * np.max(
            self.positions.shape)
        oris_flu_rand = self.oris_flu.as_rotvec()
        oris_flu_rand = oris_flu_rand + (np.random.rand(*oris_flu_rand.shape) - 0.5) * np.max(oris_flu_rand.shape)

        landmark_vecs_rand, landmark_dists_rand = sim.get_landmark_vectors(positions_rand, oris_flu_rand,
                                                                           landmarks_rand)

        return landmarks_rand, oris_flu_rand, positions_rand, landmark_vecs_rand, landmark_dists_rand
