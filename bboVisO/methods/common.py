from time import time

import numpy as np
from scipy.sparse import csr_matrix, hstack as sparse_hstack

from ..helper import get_var_edges


def objfun_jacobian_wrapper_sparse(free_vars, args, jacs, unravel_func, sd) -> np.ndarray:
    unraveled_vars = unravel_func(free_vars, args)
    landmarks, positions, oris_flu, cam_rotvecs, cam_ts, marker_var, lm_weights, mask_weights = unraveled_vars[:8]
    extra_args = unraveled_vars[8:]

    sm, sk, sn = marker_var.shape[:3]

    rm = np.arange(sm)
    rk = np.arange(sk)
    rn = np.arange(sn)
    rd = np.arange(sd)

    rm_b = rm[:, None, None, None]
    rk_b = rk[None, :, None, None]
    rn_b = rn[None, None, :, None]
    rd_b = rd[None, None, None, :]

    result_size = (sm * sk * sn * sd)

    edges = get_var_edges(sm, sk, sk, sn)
    jacs_sparse = []

    # Landmarks
    jac_idx = 0
    mask = args['mask_opt'][edges[jac_idx]:edges[jac_idx + 1]]
    if not np.any(mask):
        jacs_sparse.append(csr_matrix(([], ([], [])), shape=(result_size, 0)))
    else:
        element_length = 3
        n_cols = np.sum(mask)
        next_col = 0
        rows = np.empty((sm * sk * sd * n_cols,))
        cols = np.empty((sm * sk * sd * n_cols,))
        data = np.empty((sm * sk * sd * n_cols,))
        data_ptr = 0

        row_indices_base = (rm_b * (sk * sn * sd) +
                            rk_b * (sn * sd) +
                            rd_b).reshape(-1)

        for i_n in rn:
            start_idx = i_n * element_length
            element_mask = mask[start_idx:(start_idx + element_length)]
            j = np.array(jacs[jac_idx](landmarks[(i_n,),], positions, oris_flu, cam_rotvecs, cam_ts,
                                       marker_var[:, :, (i_n,)], lm_weights[:, :, (i_n,)], mask_weights[:, :, (i_n,)],
                                       *extra_args))
            j = j[..., element_mask]

            en_cols = np.sum(element_mask)
            row_indices = row_indices_base + i_n * sd
            column_indices = np.tile(np.arange(en_cols) + next_col, row_indices.size)
            row_indices = np.repeat(row_indices, np.sum(element_mask))

            data_size = column_indices.size
            rows[data_ptr:data_ptr + data_size] = row_indices
            cols[data_ptr:data_ptr + data_size] = column_indices
            data[data_ptr:data_ptr + data_size] = j.ravel()
            data_ptr += data_size

            next_col += np.sum(element_mask)
        data[np.isnan(data)] = 0  # Possibly arccos numerically > 1?
        jacs_sparse.append(csr_matrix((data, (rows, cols)), shape=(result_size, n_cols)))

    # Positions, orientations
    for jac_idx in range(1, 3):
        mask = args['mask_opt'][edges[jac_idx]:edges[jac_idx + 1]]

        if not np.any(mask):
            jacs_sparse.append(csr_matrix(([], ([], [])), shape=(result_size, 0)))
        else:
            element_length = 3
            n_cols = np.sum(mask)
            next_col = 0
            rows = np.empty((sm * sn * sd * n_cols,))
            cols = np.empty((sm * sn * sd * n_cols,))
            data = np.empty((sm * sn * sd * n_cols,))
            data_ptr = 0

            row_indices_base = (rm_b * (sk * sn * sd) +
                                rn_b * sd +
                                rd_b).reshape(-1)

            for i_k in rk:
                start_idx = i_k * element_length
                element_mask = mask[start_idx:(start_idx + element_length)]

                j = np.array(jacs[jac_idx](landmarks, positions[(i_k,),], oris_flu[(i_k,),], cam_rotvecs, cam_ts,
                                           marker_var[:, (i_k,)], lm_weights, mask_weights[:, (i_k,)], *extra_args))
                j = j[..., element_mask]

                en_cols = np.sum(element_mask)
                row_indices = row_indices_base + i_k * (sn * sd)
                column_indices = np.tile(np.arange(en_cols) + next_col, row_indices.size)
                row_indices = np.repeat(row_indices, np.sum(element_mask))

                data_size = column_indices.size
                rows[data_ptr:data_ptr + data_size] = row_indices
                cols[data_ptr:data_ptr + data_size] = column_indices
                data[data_ptr:data_ptr + data_size] = j.ravel()
                data_ptr += data_size

                next_col += np.sum(element_mask)

            data[np.isnan(data)] = 0  # Possibly arccos numerically > 1?
            jacs_sparse.append(csr_matrix((data, (rows, cols)), shape=(result_size, n_cols)))

    # cam_rotvecs, cam_ts and potential others, expected to be dense but small matrices
    for jac_idx in range(3, len(jacs)):
        mask = args['mask_opt'][edges[jac_idx]:edges[jac_idx + 1]]
        j = np.array(jacs[jac_idx](landmarks, positions, oris_flu, cam_rotvecs, cam_ts,
                                   marker_var, lm_weights, mask_weights, *extra_args).reshape(result_size, -1))
        j[np.isnan(j)] = 0  # Possibly arccos numerically > 1?
        jacs_sparse.append(csr_matrix(j[:, mask]))

    jac_sparse = sparse_hstack(jacs_sparse)
    jac_sparse.data[np.isinf(jac_sparse.data)] = 1e9
    return jac_sparse


def objfun_jacobian_wrapper_dense(free_vars, args, jacs, unravel_func, sd):
    obfun_inputs = unravel_func(free_vars, args)
    result_shape = obfun_inputs[5].shape[:-1] + (sd,)

    # TODO (minor speed) Do not calculate Jacobian if there are no related free variables
    obj_fcn_jacobian = [
        np.array(
            # Ravel over input dimensions
            j(*obfun_inputs).reshape(
                result_shape + (-1,))
        )
        for j in jacs
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


def make_jacobian_sparsity_mask(m, k, n, d, mask_free_input):
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
