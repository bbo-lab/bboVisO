import numpy as np


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
        S[dim-1, dim-1] = -1

    R = np.dot(np.dot(U, S), Vt)

    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.sum(d * S[dim-1, dim-1])
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


if __name__ == '__main__':
    test_umeyama()
