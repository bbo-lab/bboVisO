import numpy as np
from scipy.sparse import csr_matrix, hstack as sparse_hstack

from ..helper import get_var_edges


def objfun_jacobian_wrapper_sparse(free_vars, args, jacs, unravel_func, sd) -> np.ndarray:
    unraveled_vars = unravel_func(free_vars, args)
    landmarks, positions, oris_flu, cam_rotvecs, cam_ts = unraveled_vars[:5]
    marker_var = unraveled_vars[5]
    extra_args = unraveled_vars[6:]

    sm, sk, sn = marker_var.shape[:3]

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
                    for i_d in rd:
                        ridxs, cidxs = make_squared_indices(i_m, i_k, i_n, i_d,  # Result dimensions
                                                            np.arange(next_col, next_col + np.sum(element_mask)),
                                                            n_cols)
                        data.append(j[i_m, i_k, i_n, i_d, :].ravel())
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
                        for i_d in rd:
                            ridxs, cidxs = make_squared_indices(i_m, i_k, i_n, i_d,  # Result dimensions
                                                                np.arange(next_col, next_col + np.sum(element_mask)),
                                                                n_cols)
                            data.append(j[i_m, i_k, i_n, i_d, :].ravel())
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