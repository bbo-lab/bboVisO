def get_opts():
    return {
        'optimization': {
            'method': 'trf',
            'ftol': 1e-13,
            'xtol': 1e-13,
            'gtol': 1e-13,
            'x_scale': 'jac',
            'loss': 'linear',
            'tr_solver': 'lsmr',
            'max_nfev': 2000,
            'verbose': 2,
        }
    }