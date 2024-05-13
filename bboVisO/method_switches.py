import importlib

from bboVisO.methods.angular import optfuncs as viso_angular
from bboVisO.methods.repro import optfuncs as viso_repro

def objective_function(opts):
    module = importlib.import_module("bboVisO.methods."+opts['objective_function']+".optfuncs")
    return module.objfun_wrapper


def objective_function_runner(opts):
    module = importlib.import_module("bboVisO.methods."+opts['objective_function']+".optfuncs")
    return module.objfun_runner


def jacobian_function(opts):
    module = importlib.import_module("bboVisO.methods."+opts['objective_function']+".optfuncs")
    return module.objfun_jacobian_wrapper


def make_jacobian_sparsity_mask(opts):
    module = importlib.import_module("bboVisO.methods."+opts['objective_function']+".optfuncs")
    return module.make_jacobian_sparsity_mask
