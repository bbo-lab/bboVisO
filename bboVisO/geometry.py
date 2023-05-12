import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import fmin
import warnings


def align_positions_orientations(reference_points, points, positions, orientations, fix_indices=None):
    # If `fix_indices` is None, umeyama algorithm is used to find the best alignment of `points` to `reference_points`,
    # and `positions` and `orientations` are transformed accordingly.
    # If 2 fix_indices are given, these 2 points are aligned exactly to the respective references, while the rest is
    # used to determine the rotation around the remaining axis, minimizing least-square errors.
    assert fix_indices is None or len(fix_indices) == 2, "Needs exactly 2 fix indices to fix two points" \
                                                         " and orient the rest"

    if fix_indices is None:
        rot, t, s = umeyama(points, reference_points)
    else:
        # Align 2 points at fix_indices
        ref_line = np.diff(reference_points[fix_indices, ...], axis=0)
        ref_line_len = np.linalg.norm(ref_line)
        ref_line /= ref_line_len
        p_line = np.diff(points[fix_indices, ...], axis=0)
        p_line_len = np.linalg.norm(p_line)
        p_line /= p_line_len

        s = ref_line_len / p_line_len
        with warnings.catch_warnings():
            # We know that the problem is underdefined, but we don't care because we solve
            # the free rotation afterwards.
            warnings.simplefilter("ignore")
            rot = R.align_vectors(ref_line, p_line)[0].as_matrix()
        t = reference_points[fix_indices[0], ...] - points[fix_indices[0], ...] @ rot.T * s

        # Align the other points by a rotation around the fix_indices axis
        points = (points @ rot.T * s + t) - reference_points[fix_indices[0]]
        reference_points = reference_points - reference_points[fix_indices[0]]

        axis = np.diff(reference_points[fix_indices, ...], axis=0)
        axis /= np.linalg.norm(axis, axis=-1)[..., np.newaxis]
        # TODO: There's probably a more elegant way that optimization?
        xopt = fmin(
            func=lambda x: np.nansum((R.from_rotvec(x * axis).apply(points) - reference_points) ** 2),
            x0=0,
            disp=0)

        rot2 = R.from_rotvec(xopt[0] * axis).as_matrix()[0]
        rot = rot2 @ rot
        t = rot2 @ (t - reference_points[fix_indices[0]].T) + reference_points[fix_indices[0]].T

    positions_aligned = []
    for p in positions:
        positions_aligned.append(p @ rot.T * s + t)

    orientations_aligned = []
    for o in orientations:
        orientations_aligned.append((R.from_matrix(rot) * R.from_rotvec(o)).as_rotvec())

    pass

    return positions_aligned, orientations_aligned


def umeyama(src, dst, estimate_scale=True):
    assert len(src) == len(dst)

    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = np.dot(dst_demean.T, src_demean) / len(src_demean)
    U, d, Vt = np.linalg.svd(A)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1

    R = np.dot(np.dot(U, S), Vt)

    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.sum(d * S[dim - 1, dim - 1])
        T = dst_mean - scale * np.dot(src_mean, R.T)
        return R, T, scale
    else:
        T = dst_mean - np.dot(src_mean, R.T)
        return R, T


def test_umeyama():
    # Generate random source and destination points
    src = np.random.rand(10, 3)
    R_true = np.array([[0.707, -0.707, 0], [0.707, 0.707, 0], [0, 0, 1]])
    t_true = np.array([1, 2, 3])
    s_true = 0.5
    dst = s_true * np.dot(R_true, src.T).T + t_true

    # Test with estimated scaling factor
    R_est, t_est, s_est = umeyama(src, dst, estimate_scale=True)

    print(R_est)
    print(t_est)
    print(s_est)

    assert np.allclose(R_est, R_true, atol=1e-3)
    assert np.allclose(t_est, t_true, atol=1e-3)
    assert np.isclose(s_est, s_true, atol=1e-3)

    # Test without scaling factor estimation
    R_est, t_est = umeyama(src, dst, estimate_scale=False)
    assert np.allclose(R_est, R_true, atol=1e-3)
    assert np.allclose(t_est, t_true, atol=1e-3)


def build_vector_bundle(base_bundle: np.ndarray, add_bundles: np.ndarray):
    return_bundle = base_bundle.copy()

    for j in range(10):
        for i_fr, add_bundle in enumerate(add_bundles):
            return_mask = ~np.any(np.isnan(return_bundle), axis=-1)
            add_mask = ~np.any(np.isnan(add_bundle), axis=-1)
            common_mask = return_mask & add_mask
            replace_mask = add_mask & ~return_mask
            if np.sum(common_mask) > 1 and np.any(replace_mask):
                align_R = R.align_vectors(return_bundle[common_mask], return_bundle[common_mask])[0]
                return_bundle[replace_mask] = align_R.apply(add_bundle[replace_mask])

            if not np.any(np.isnan(return_bundle)):
                return return_bundle

    warnings.warn("Could not fill return_bundle")
    return return_bundle


if __name__ == '__main__':
    test_umeyama()
