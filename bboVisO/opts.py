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
            'jac': '2-point',
        },
        'far_first': False,  # Optimize on only far landmarks first
        'close_first': False,  # Optimize on only close landmarks first
        'interpolate_high_error': 15,  # After a first optimization run, interpolate position and orientation of
                                       #  high error frames as initialization for next optimization. Error is distance
                                       #  between marked and reprojected point. Deactivate with -1
        'time_range': [0, -1],
        'min_labels': 4,  # Require min_labels detected per frame
        'single_position': False,  # object stays in one position, only estimate single position + rotations
        'time_dtype': float,  # Data type of time output. Mostly for resetting within custom importers, e.g. with µs
        'add_offset': [],  # Add an offset to time, for subframe alignment
        'realign_by_distant_features': True,  # Realign camera with only further half of landmarks after optimization
    }
