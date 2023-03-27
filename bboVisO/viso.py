import numpy as np
import bboVisO.viso_ag as viso_ag
from jax import jit, jacfwd as jacobian  # jacfwd is recommended for 'tall' Jacobians, jacrev for 'wide'

def infer_position_attitude(landmark_vectors):
    pass


def align_results(landmarks_real, landmarks, positions, oris_flu):
    return landmarks, positions, oris_flu


def objfun_jacobian_wrapper(free_vars, args):
    landmarks, dists, positions, oris_flu, landmark_vecs = unravel_vars(free_vars, args)
    k = landmark_vecs.shape[0]
    n = landmark_vecs.shape[1]

    obj_fcn_jacobian = [
        np.array(
            # Ravel over input dimensions
            j(landmarks, dists, positions, oris_flu, landmark_vecs).reshape(landmark_vecs.shape + (-1,))
        )
        for j in args['precalc']['jacobians']
    ]

    # Concatenate along input dimensions
    obj_fcn_jacobian = np.concatenate(
        obj_fcn_jacobian,
        axis=3)

    obj_fcn_jacobian = obj_fcn_jacobian.reshape((-1, args['mask_opt'].size))

    # Dump non-free variables
    obj_fcn_jacobian = obj_fcn_jacobian[:, args['mask_opt']]

    return obj_fcn_jacobian


def make_optim_input(landmark_vecs, opts, landmarks=None, dists=None, positions=None, oris_flu=None):
    vars_free, vars_full, mask_free_input = make_initialization(landmark_vecs, opts, landmarks, dists, positions, oris_flu)
    args = {
        'vars_full': vars_full,  # All possible vars, free vars will be substituted in _free wrapper functions
        'mask_opt': mask_free_input,  # Mask of free vars within all vars
        'landmark_vecs': landmark_vecs,
        'precalc': get_precalc(),
    }
    return vars_free, args


def make_initialization(landmark_vecs, opts, landmarks=None, dists=None, positions=None, oris_flu=None):
    k = landmark_vecs.shape[0]
    n = landmark_vecs.shape[1]
    np.random.seed(0)

    vars_full = np.random.random((k * n + 6 * k + 3 * n))
    mask_free_input = np.ones(vars_full.shape, dtype=bool)

    mask_free_input[0:6] = False
    vars_full[0:3] = 0
    vars_full[3] = 1
    vars_full[4:6] = 0

    mask_free_input[8] = False
    vars_full[8] = 0

    edges = np.array([0,
                      n * 3,
                      n * 3 + k * n,
                      n * 3 + k * n + k * 3,
                      n * 3 + k * n + k * 6
                      ])
    if landmarks is not None:
        vars_full[edges[0]:edges[1]] = landmarks.ravel()
    if dists is not None:
        vars_full[edges[1]:edges[2]] = dists.ravel()
    if positions is not None:
        vars_full[edges[2]:edges[3]] = positions.ravel()
    if oris_flu is not None:
        vars_full[edges[3]:edges[4]] = oris_flu.ravel()

    vars_free = vars_full[mask_free_input]

    return vars_free, vars_full, mask_free_input


def get_precalc():
    return {
        'objfunc': jit(viso_ag.objfun),
        'jacobians': [jit(jacobian(viso_ag.objfun, i_var)) for i_var in range(4)]
    }


def objfun_wrapper(free_vars, args):
    landmarks, dists, positions, oris_flu, landmark_vecs = unravel_vars(free_vars, args)

    errors3d = viso_ag.objfun(landmarks, dists, positions, oris_flu, landmark_vecs)

    return errors3d.ravel()


def unravel_vars(free_vars, args):
    vars_full = args["vars_full"].copy()
    vars_full[args["mask_opt"]] = free_vars
    landmark_vecs = args["landmark_vecs"]
    k = landmark_vecs.shape[0]
    n = landmark_vecs.shape[1]

    start = 0
    end = n * 3
    landmarks = vars_full[start:end].reshape(n, 3)
    start = end
    end += k * n
    dists = vars_full[start:end].reshape(k, n)
    start = end
    end += k * 3
    positions = vars_full[start:end].reshape(k, 3)
    start = end
    end += k * 3
    oris_flu = vars_full[start:end].reshape(k, 3)
    return landmarks, dists, positions, oris_flu, landmark_vecs
