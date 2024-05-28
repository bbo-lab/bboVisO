from bboVisO.methods import angular, repro
from bboVisO import viso, opts, test
import numpy as np
from scipy.optimize import approx_fprime
from calibcamlib import Camerasystem

import jax

jax.config.update("jax_enable_x64", True)

np.random.seed(0)

viso_opts = opts.get_opts()

calibs = Camerasystem.load_dict("/media/smb/soma-fs.ad01.caesar.de/bbo/projects/junker-bird/experiments/20230612_20230615_FlightWindow/calibrations/eyecams/20230615_0/multicam_calibration_viso_viso.yml")["calibs"]
landmarks = np.random.rand(5, 3)*20
positions = np.random.rand(7, 3)*10
oris = np.random.rand(7, 3)*np.pi
imcoords = np.random.rand(2, 7, 5, 2)

opts.angular_jax_po_pol["optimize"][1]

vars_free, objfun_args, optim_args = viso.make_optim_input(calibs, opts.angular_jax_po_pol, landmarks, positions, oris,
                                                           landmark_imcoords=imcoords,
                                                           optimize=['positions', 'oris', 'landmarks', 'extrinsics_r', 'extrinsics_t'])

objfun_angular_wrapper_lambda = lambda x: angular.objfun_wrapper(x, objfun_args)
objfun_repro_wrapper_lambda = lambda x: repro.objfun_wrapper(x, objfun_args)

eps = 1e-6

repro_jax_dense_jac = repro.optfuncs.objfun_jacobian_wrapper_dense(vars_free, objfun_args)
repro_jax_sparse_jac = repro.optfuncs.objfun_jacobian_wrapper(vars_free, objfun_args).todense()
repro_approx_fprime_jac = approx_fprime(vars_free, repro.objfun_wrapper, eps, objfun_args)
repro_calc_2point_jacobian_jac = test.calc_2point_jacobian(repro.objfun_wrapper, vars_free, eps, objfun_args)
repro_sparsity_mask = repro.make_jacobian_sparsity_mask(*imcoords.shape[:3], objfun_args["mask_opt"])

angular_jax_dense_jac = angular.optfuncs.objfun_jacobian_wrapper_dense(vars_free, objfun_args)
angular_jax_sparse_jac = angular.optfuncs.objfun_jacobian_wrapper(vars_free, objfun_args).todense()
angular_approx_fprime_jac = approx_fprime(vars_free, angular.objfun_wrapper, eps, objfun_args)
angular_calc_2point_jacobian_jac = test.calc_2point_jacobian(angular.objfun_wrapper, vars_free, eps, objfun_args)
angular_sparsity_mask = angular.make_jacobian_sparsity_mask(*imcoords.shape[:3], objfun_args["mask_opt"])


assert not np.any((np.abs(angular_jax_dense_jac)+np.abs(angular_approx_fprime_jac)+np.abs(angular_calc_2point_jacobian_jac))[~angular_sparsity_mask]), "Angular: Jacobian entries outside sparsity matrix"
assert not np.any((np.abs(repro_jax_dense_jac)+np.abs(repro_approx_fprime_jac)+np.abs(repro_calc_2point_jacobian_jac))[~repro_sparsity_mask]), "Repro: Jacobian entries outside sparsity matrix"
assert np.sum(np.abs(angular_jax_dense_jac-angular_jax_sparse_jac))==0, "Angular: Sparse jacobian does not match dense one"
assert np.sum(np.abs(repro_jax_dense_jac-repro_jax_sparse_jac))==0, "Repro: Sparse jacobian does not match dense one"
print("Angular: Max difference dense to fprime: ", np.max(np.abs(angular_jax_dense_jac-angular_approx_fprime_jac)))
print("Angular: Max difference dense to 2-point: ", np.max(np.abs(angular_jax_dense_jac-angular_calc_2point_jacobian_jac)))
print("")
print("Warning: for repro method, numerical issues in jacobian see to be extreme")
print("Repro: Max difference dense to fprime: ", np.max(np.abs(repro_jax_dense_jac-repro_approx_fprime_jac)))
print("Repro: Max difference dense to 2-point: ", np.max(np.abs(repro_jax_dense_jac-repro_calc_2point_jacobian_jac)))