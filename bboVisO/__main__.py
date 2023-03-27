import numpy as np
from scipy.optimize import least_squares, OptimizeResult

import bboVisO.simulation as sim
import bboVisO.viso_ag as viso_ag
import bboVisO.viso as viso

opts = {
    'optimization': {
        'method': 'trf',
        'ftol': 1e-4,
        'xtol': 1e-8,
        'gtol': 1e-8,
        'x_scale': 'jac',
        'loss': 'linear',
        'tr_solver': 'lsmr',
        'max_nfev': 150,
        'verbose': 2,
    }
}

landmarks = sim.get_landmarks(5)
positions, ori_flu = sim.get_screw_trajectory()

landmark_vecs, landmark_dists = sim.get_landmark_vectors(positions, ori_flu, landmarks)
landmark_vecs_pertubated = sim.pertubate_landmark_vectors(landmark_vecs, 0 * 1 / 180 * np.pi)



vars_free, args = viso.make_optim_input(landmark_vecs, opts, landmarks, landmark_dists, positions, ori_flu.as_rotvec())

# Test wrapper function
error = viso_ag.objfun(landmarks, landmark_dists, positions, ori_flu.as_rotvec(), landmark_vecs_pertubated)
error2 = viso.objfun_wrapper(vars_free, args)
assert np.sum(np.abs(error.ravel()-error2.ravel())) < 1e-10, "objfun wrapper does not work"

vars_free, args = viso.make_optim_input(landmark_vecs, opts)
jac = viso.objfun_jacobian_wrapper


min_result: OptimizeResult = least_squares(viso.objfun_wrapper,
                                           vars_free,
                                           jac=jac,
                                           bounds=np.array([[-np.inf, np.inf]] * vars_free.size).T,
                                           args=[args],
                                           **opts['optimization']
                                           )

landmarks1, landmark_dists1, positions1, oris_flu1, landmark_vecs1 = viso.unravel_vars(min_result.x, args)
pass
