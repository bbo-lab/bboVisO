import sys

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
        },
        'far_first': False,  # Optimize on only far landmarks first
        'close_first': False,  # Optimize on only close landmarks first
        'interpolate_high_error': True,  # After a first optimization run, interpolate position and orientation of
                                         #  high error frames as initialization for next opimization
        'check_mode': False,  # as interpolate_high_error, but after last opimization. This often makes finding label
                              #  errors much easier
        'time_range': [0, 3600*24*365],
        'min_labels': 4,  # Require min_labels detected per frame
        'single_position': False,  # object stays in one position, only estimate single position + rotations
    }
