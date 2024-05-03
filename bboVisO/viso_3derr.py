## Not updated to newer code and currently unsupported

# import numpy as np
#
# from bboVisO.unravel import unravel_vars
#
#
# def objfun_3d_wrapper(free_vars, args):
#     landmarks, positions, oris_flu, \
#         landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts, lm_close_pen_weight = unravel_vars(free_vars, args)
#
#     landmark_vecs = args["landmark_vecs"]
#
#     errors3d = args['precalc']['objfunc_3d'](landmarks, positions, oris_flu, landmark_vecs)
#
#     return errors3d.ravel()
#
#
# def objfun_jacobian_3d_wrapper(free_vars, args):
#     landmarks, positions, oris_flu, \
#         landmark_imcoords, cam_matrices, ks, xis, cam_Rs, cam_ts = unravel_vars(free_vars, args)
#
#     obj_fcn_jacobian = [
#         np.array(
#             # Ravel over input dimensions
#             j(landmarks, positions, oris_flu, landmark_vecs).reshape(landmark_vecs.shape + (-1,))
#         )
#         for j in args['precalc']['jacobians_3d']
#     ]
#
#     # Concatenate along input dimensions
#     obj_fcn_jacobian = np.concatenate(
#         obj_fcn_jacobian,
#         axis=3)
#
#     # Ravel variable dimension
#     obj_fcn_jacobian = obj_fcn_jacobian.reshape((-1, args['mask_opt'].size))
#
#     # Dump non-free variables
#     obj_fcn_jacobian = obj_fcn_jacobian[:, args['mask_opt']]
#
#     return obj_fcn_jacobian
