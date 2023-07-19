def get_opts():
    return {
        'optimization': {
            'method': 'trf',
            'ftol': 1e-6,
            'xtol': 1e-6,
            'gtol': 1e-8,
            'x_scale': 'jac',
            'loss': 'linear',
            'tr_solver': 'lsmr',  # 'lsmr' is forced since jac_sparsity is also passed to least_squares function
            'max_nfev': 500,
            'verbose': 2,
        }
    }
