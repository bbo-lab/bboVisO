import numpy as np
from jax import jit, jacfwd as jacobian
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix, hstack as sparse_hstack

from .optfuncs_ag import objfun
from ...unravel import unravel_vars as unravel_vars_base, unpack_calibs
from ...initialization import get_var_edges


def objfun_runner(landmarks, positions, oris_flu, landmark_vecs, calibs, opts) -> np.ndarray:
    cam_Rs, cam_ts = unpack_calibs(calibs)[-2:]
    errors = np.array(objfun(landmarks, positions, oris_flu, Rotation.from_matrix(cam_Rs).as_rotvec(), cam_ts,
                             landmark_vecs, opts["lm_close_pen_weight"]))

    return errors


def objfun_wrapper(free_vars, args, no_ravel=False) -> np.ndarray:
    obfun_inputs = unravel_vars(free_vars, args)
    errors = np.array(jit(objfun)(*obfun_inputs))

    if no_ravel:
        return errors
    else:
        errors[np.isnan(errors)] = 0
        return errors.ravel()


def objfun_jacobian_wrapper(free_vars, args) -> np.ndarray:
    (landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
     landmark_vecs, lm_close_pen_weight) = unravel_vars(free_vars, args)
    result_shape = landmark_vecs.shape[:-1] + (1,)

    # TODO (minor speed) Do not calculate Jacobian if there are no related free variables
    obj_fcn_jacobian = [
        np.array(
            # Ravel over input dimensions
            j(landmarks, positions, oris_flu, cam_rotvecs, cam_ts, landmark_vecs, lm_close_pen_weight).reshape(
                result_shape + (-1,))
        )
        for j in get_jacobians(args)
    ]

    # Concatenate along input dimensions
    obj_fcn_jacobian = np.concatenate(
        obj_fcn_jacobian,
        axis=4)

    # Ravel variable dimension
    obj_fcn_jacobian = obj_fcn_jacobian.reshape((np.prod(result_shape), args['mask_opt'].size))

    # Dump non-free variables
    obj_fcn_jacobian = obj_fcn_jacobian[:, args['mask_opt']]

    # There are nans here, but not in the repro wrapper. Why? Possibly arccos numerically > 1?
    obj_fcn_jacobian[np.isnan(obj_fcn_jacobian)] = 0

    return obj_fcn_jacobian


def objfun_jacobian_wrapper_sparse(free_vars, args) -> np.ndarray:
    unraveled_vars = unravel_vars(free_vars, args)
    landmarks, positions, oris_flu, cam_rotvecs, cam_ts = unraveled_vars[:5]
    marker_var = unraveled_vars[5]
    extra_args = unraveled_vars[6:]

    jacs = get_jacobians(args)

    sm, sk, sn = marker_var.shape[:3]
    sd = 1

    rm = np.arange(sm)
    rk = np.arange(sk)
    rn = np.arange(sn)
    rd = np.arange(sd)

    result_size = (sm * sk * sn * sd)

    def make_squared_indices(m_range, k_range, n_range, d_range, var_range, var_length):
        idx_ravel = np.ravel_multi_index(
            (m_range, k_range, n_range, d_range, var_range),
            (sm, sk, sn, sd, var_length)
        )
        return np.unravel_index(idx_ravel, (sm * sk * sn * sd, var_length))

    edges = get_var_edges(sm, sk, sk, sn)
    jacs_sparse = []

    # Landmarks
    rows = []
    cols = []
    data = []
    element_length = 3
    jac_idx = 0
    mask = args['mask_opt'][edges[jac_idx]:edges[jac_idx + 1]]
    n_cols = np.sum(mask)
    next_col = 0
    if not np.any(mask):
        jacs_sparse.append(csr_matrix((data, (rows, cols)), shape=(result_size, 0)))
    else:
        for i_n in rn:
            start_idx = i_n * element_length
            element_mask = mask[start_idx:(start_idx + element_length)]
            j = jacs[jac_idx](landmarks[(i_n,),], positions, oris_flu, cam_rotvecs, cam_ts,
                              marker_var[:, :, (i_n,)], *extra_args)
            j = j[..., element_mask]
            for i_m in rm:
                for i_k in rk:
                    ridxs, cidxs = make_squared_indices(i_m, i_k, i_n, rd,  # Result dimensions
                                                        np.arange(next_col, next_col + np.sum(element_mask)),
                                                        n_cols)
                    data.append(j[i_m, i_k, i_n, :, :].ravel())
                    rows.append(ridxs)
                    cols.append(cidxs)
            next_col += np.sum(element_mask)
        data = np.hstack(data)
        rows = np.hstack(rows)
        cols = np.hstack(cols)
        data[np.isnan(data)] = 0  # Possibly arccos numerically > 1?
        jacs_sparse.append(csr_matrix((data, (rows, cols)), shape=(result_size, n_cols)))

    # Positions, orientaions
    for jac_idx in range(1, 3):
        rows = []
        cols = []
        data = []
        element_length = 3
        mask = args['mask_opt'][edges[jac_idx]:edges[jac_idx + 1]]
        n_cols = np.sum(mask)
        next_col = 0
        if not np.any(mask):
            jacs_sparse.append(csr_matrix((data, (rows, cols)), shape=(result_size, 0)))
        else:
            for i_k in rk:
                start_idx = i_k * element_length
                element_mask = mask[start_idx:(start_idx + element_length)]
                j = jacs[jac_idx](landmarks, positions[(i_k,),], oris_flu[(i_k,),], cam_rotvecs, cam_ts,
                                  marker_var[:, (i_k,)], *extra_args)
                j = j[..., element_mask]
                for i_m in rm:
                    for i_n in rn:
                        ridxs, cidxs = make_squared_indices(i_m, i_k, i_n, rd,  # Result dimensions
                                                            np.arange(next_col, next_col + np.sum(element_mask)),
                                                            n_cols)
                        data.append(j[i_m, i_k, i_n, :, :].ravel())
                        rows.append(ridxs)
                        cols.append(cidxs)
                next_col += np.sum(element_mask)
            data = np.hstack(data)
            rows = np.hstack(rows)
            cols = np.hstack(cols)
            data[np.isnan(data)] = 0  # Possibly arccos numerically > 1?
            jacs_sparse.append(csr_matrix((data, (rows, cols)), shape=(result_size, n_cols)))

    # cam_rotvecs, cam_ts and potential others, expected to be dense but small matricies
    for jac_idx in range(3, len(jacs)):
        mask = args['mask_opt'][edges[jac_idx]:edges[jac_idx + 1]]
        n_cols = np.sum(mask)
        j = np.array(jacs[jac_idx](landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
                          marker_var, *extra_args).reshape(result_size, -1))
        j[np.isnan(j)] = 0  # Possibly arccos numerically > 1?
        jacs_sparse.append(csr_matrix(j[:, mask]))

    return sparse_hstack(jacs_sparse)


def make_jacobian_sparsity_mask(m, k, n, mask_free_input):
    d = 1  # Output dimension, angle
    # TODO: Make sparse matrix
    jac_sparsity_lm = np.zeros((m, k, n, d) + (n, 3), dtype=bool)
    for i_n in range(n):
        jac_sparsity_lm[:, :, i_n, :, i_n] = True
    jac_sparsity_lm = jac_sparsity_lm.reshape((m * k * n * d, -1))

    jac_sparsity_pos_rot = np.zeros((m, k, n, d) + (k, 3), dtype=bool)
    for i_k in range(k):
        jac_sparsity_pos_rot[:, i_k, :, :, i_k] = True
    jac_sparsity_pos_rot = jac_sparsity_pos_rot.reshape((m * k * n * d, -1))

    jac_sparsity_cam_pos_rot = np.ones((m, k, n, d) + (m, 3), dtype=bool)
    jac_sparsity_cam_pos_rot = jac_sparsity_cam_pos_rot.reshape((m * k * n * d, -1))

    jac_sparsity = np.hstack((jac_sparsity_lm,
                              jac_sparsity_pos_rot,
                              jac_sparsity_pos_rot,
                              jac_sparsity_cam_pos_rot,
                              jac_sparsity_cam_pos_rot,
                              ))

    # jac_sparsity[:] = True  # Until this gives the same result ...
    return jac_sparsity[:, mask_free_input]


def unravel_vars(free_vars, args):
    landmarks, positions, oris_flu, cam_rotvecs, cam_ts = unravel_vars_base(free_vars, args)

    landmark_vecs = args["landmark_vecs"]
    lm_close_pen_weight = args["lm_close_pen_weight"]

    return (landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
            landmark_vecs, lm_close_pen_weight)


jacobians_angular = None


def get_jacobians(opts):
    global jacobians_angular
    if jacobians_angular is None:
        if "jax_backend" not in opts:
            opts["jax_backend"] = "cpu"
            print("Set JAX backend to CPU")
        jacobians_angular = [jit(jacobian(objfun, i_var), backend=opts["jax_backend"]) for i_var in range(5)]
    return jacobians_angular
