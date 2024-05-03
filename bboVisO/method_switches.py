from bboVisO.viso_angular import objfun_angular_wrapper, objfun_angular_runner, objfun_angular_jacobian_wrapper, \
    make_jacobian_sparsity_mask_angular
from bboVisO.viso_repro import objfun_repro_wrapper, objfun_repro_runner, objfun_repro_jacobian_wrapper, \
    make_jacobian_sparsity_mask_repro


def objective_function(opts):
    match opts['objective_function']:
        case 'repro':
            return objfun_repro_wrapper
        case 'angular':
            return objfun_angular_wrapper


def objective_function_runner(opts):
    match opts['objective_function']:
        case 'repro':
            return objfun_repro_runner
        case 'angular':
            return objfun_angular_runner


def jacobian_function(opts):
    match opts['objective_function']:
        case 'repro':
            return objfun_repro_jacobian_wrapper
        case 'angular':
            return objfun_angular_jacobian_wrapper


def make_jacobian_sparsity_mask(opts):
    match opts['objective_function']:
        case 'repro':
            return make_jacobian_sparsity_mask_repro
        case 'angular':
            return make_jacobian_sparsity_mask_angular
