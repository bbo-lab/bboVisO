import sys
from time import time

import numpy as np
from pathlib import Path

from bbo.geometry import RigidTransform

from tetherlessextractor.data_object import Data
from calibcam.camcalibrator import save_multicalibration
from calibcamlib import Camerasystem

from bboVisO.methods import angular, repro
import bboVisO.opts, bboVisO.interpolation, bboVisO.helper


# Generate ground truth
np.random.seed(0)

multicalibration = Camerasystem.load_dict(Path(sys.path[0]).parent / "data/omnidirectional.yml")
camera_angle = np.pi

calibs = multicalibration["calibs"]
cs = Camerasystem.from_calibs(calibs)

m = len(calibs)
k = 5
n = 3

positions = np.random.rand(k, 3)*10
oris = np.random.rand(k, 3)*np.pi
landmarks = np.random.rand(n, 3)*20

# Transform landmarks to per-frame camsystem
camsystem2world = RigidTransform(rotation=oris, translation=positions)
landmarks_camsystem = camsystem2world.inv().apply_broadcast(landmarks)
landmarks_camsystem.shape


# Image coordinates from landmarks outside the FOV are nonsense, remove
# Find center of FOV
sensor_centers = np.array([np.flip(vh["sensorsize"]) for vh in multicalibration['vid_headers']])/2
cam_center_vecs_cam = np.array([c["camera"].sensor_to_space(sensor_centers[i_c]) for i_c, c in enumerate(cs.cameras)])
# Set vectors larger than opeing angle to nan
landmark_vecs_cams = np.empty((m, k, n, 3))
for i_cam in range(len(cs.cameras)):
    camsystem_to_cam = cs.get_camsystem_to_cam_transformation(i_cam)
    landmarks_cam = camsystem_to_cam.apply(landmarks_camsystem)
    landmark_vecs_cams[i_cam] = landmarks_cam / np.linalg.norm(landmarks_cam, axis=-1, keepdims=True)
    mask = np.arccos(np.sum(landmark_vecs_cams[i_cam] * cam_center_vecs_cam[i_cam], axis=-1)) > np.pi/2
    landmark_vecs_cams[i_cam, mask] = np.nan

# Find image coordinates
landmark_imcoords = bboVisO.helper.landmark_imcoords_from_vecs(landmark_vecs_cams, calibs)

# Check consistency of derivation
assert (np.nanmax(np.abs(landmark_vecs_cams - bboVisO.helper.landmark_vecs_from_imcoords(landmark_imcoords, calibs))))<1e-8, "Error calculating landmark_vecs or imcoords"
assert (np.nanmax(np.abs(landmark_imcoords - cs.project(landmarks_camsystem))))<1e-8, "Error calculating landmark_vecs or imcoords"

# Check ground truth on objective functions
assert np.nanmax(np.abs(angular.objfun_runner(landmarks, positions, oris, landmark_vecs_cams, calibs, bboVisO.opts.angular_jax_po_pol))) < 1e-2, "Error in angular objective function"
assert np.nanmax(np.abs(repro.objfun_runner(landmarks, positions, oris, landmark_imcoords, calibs, bboVisO.opts.repro_jax_po_pol_pol))) < 1e-3, "Error in repro objective function"

# Time (Note that this must not be the first time to run the objective functions because of JIT)
t = time()
angular.objfun_runner(landmarks, positions, oris, landmark_vecs_cams, calibs, bboVisO.opts.angular_jax_po_pol)
print("Angular objective function took", time() - t, "s")
t = time()
repro.objfun_runner(landmarks, positions, oris, landmark_imcoords, calibs, bboVisO.opts.repro_jax_po_pol_pol)
print("Repro objective function took", time() - t, "s")