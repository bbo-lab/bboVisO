import numpy as np
import matplotlib.pyplot as plt


def check_jacobian(jac, fun, x, args, eps=1e-6):
    j = jac(x, args)
    # Compute the Jacobian numerically using finite differences
    jac_num = calc_2point_jacobian(fun, x, eps, args)

    k = args["landmark_imcoords"].shape[1]
    n = args["landmark_imcoords"].shape[2]

    edges = np.array([0,
                      n * 3,
                      n * 3 + k * 3,
                      n * 3 + k * 3 + k * 3
                      ])

    free_edges = np.array([0,
                           np.sum(args["mask_opt"][edges[0]:edges[1]]),
                           np.sum(args["mask_opt"][edges[0]:edges[2]]),
                           np.sum(args["mask_opt"][edges[0]:edges[3]]),
                           ])

    for i in range(3):
        plt.figure(figsize=(18, 9))
        plt.subplot(1, 2, 1)
        plt.imshow(np.hstack((
            j[:, free_edges[0 + i]:free_edges[1 + i]],
            np.full((j.shape[0], 1), np.nan),
            jac_num[:, free_edges[0 + i]:free_edges[1 + i]],
        )))
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(j[:, free_edges[0 + i]:free_edges[1 + i]] - jac_num[:, free_edges[0 + i]:free_edges[1 + i]])
        plt.colorbar()
        plt.draw()

    plt.show()

    return np.linalg.norm(jac_num - j) / np.linalg.norm(jac_num)


def calc_2point_jacobian(fun, x, eps=np.sqrt(np.finfo(float).eps), *args, **kwargs):
    j_shape = fun(x, *args, **kwargs).shape + x.shape
    jac_num = np.zeros(j_shape)

    x_shape = x.shape
    x_size = x.size
    e = np.zeros_like(x)
    for i in range(x_size):
        e.fill(0)
        index_e = np.unravel_index(i, x_shape)
        e[index_e] = eps

        indices_j = (Ellipsis,) + index_e
        jac_num[indices_j] = ((fun(x + e, *args, **kwargs) - fun(x - e, *args, **kwargs)) / (2 * eps))

    return jac_num

# def calc_2point_jacobian(fun, eps, x, *args, **kwargs):
#     jac_num = np.zeros((len(fun(x, *args, **kwargs)), len(x)))
#
#     for i in range(len(x)):
#         e = np.zeros_like(x)
#         e[i] = eps
#         jac_num[:, i] = (fun(x + e, *args, **kwargs) - fun(x - e, *args, **kwargs)) / (2 * eps)
#
#     return jac_num
