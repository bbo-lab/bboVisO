import sys


def get_opts():
    return {
        'iterations': [
            angular_jax_po_pol,
            repro_jax_po_pol_pol,
        ],
        'time_range': [-sys.maxsize, sys.maxsize],
        # Require min_labels detected per frame
        'min_labels': 4,
        # object stays in one position, only estimate single position + rotations
        'single_position': False,
        # Data type of time output. Mostly for resetting within custom importers, e.g. with µs
        'time_dtype': float,
        # Add an offset to time, for subframe alignment
        'add_offset': [],
        'jax_backend': 'cpu',
        # Options specific to the checker
        'checker': {
            # Interpolation threshold for check mode ()
            'interpolate_high_error': 100,
        }
    }


angular_jax_po_pol = {
    'optimization': {
        'method': 'trf',
        'ftol': 1e-3,
        'xtol': 1e-6,
        'gtol': 1e-6,
        'x_scale': 'jac',
        'loss': 'linear',
        'tr_solver': 'lsmr',
        'max_nfev': 500,
        'verbose': 2,
        'jac': 'jax',
    },
    'optimize': [["positions", "oris"],
                 ["positions", "oris", "landmarks"]],
    'objective_function': 'angular',
    # 'far_first': Optimize on only far landmarks first
    'far_first': False,
    # 'close_first': Optimize on only close landmarks first
    'close_first': False,
    # 'interpolate_high_error': After a first optimization run, interpolate position and orientation of
    #  high error frames as initialization for next optimization. Error is distance
    #  between marked and reprojected point. Deactivate with -1
    'interpolate_high_error': -1,
    # 'interpolate_close_marker': After first optimization run, interpolate any position that is closer
    # than this value to a labeled feature.
    # Deactivate with -1
    'interpolate_close_marker': -1,
    # 'realign_by_distant_features': Realign camera with only further half of landmarks after optimization
    'realign_by_distant_features': False,
    # 'lm_close_pen_weight': penalty factor for positions close to landmarks (avoids positions falling
    # exactly on landmarks)
    'lm_close_pen_weight': 0 * 1e-1,
    # In the last step, pass over the data, optimizing pos and ori for each frame
    # individually
    'optimize_single_frames': False,
    # object stays in one position, only estimate single position + rotations
    'single_position': False,
    # initial positions are lifted up by this factor
    'lift_factor': 0.0,
    # Only process frames above 'max_error_median_factor'*median(error)
    'max_error_median_factor': 0,
    # Supply jacobian sparsity matrix to least_squares
    'sparse_jacobian': False,
}

repro_jax_po_pol_pol = {
    'optimization': {
        'method': 'trf',
        'ftol': 1e-6,
        'xtol': 1e-6,
        'gtol': 1e-8,
        'x_scale': 'jac',
        'loss': 'linear',
        # 'tr_solver': 'lsmr' is forced since jac_sparsity is also passed to least_squares function
        'tr_solver': 'lsmr',
        'max_nfev': 500,
        'verbose': 2,
        'jac': 'jax',
    },
    'optimize': [["positions", "oris"],
                 ["positions", "oris", "landmarks"],
                 ["positions", "oris", "landmarks"]],
    'objective_function': 'repro',
    # 'far_first': Optimize on only far landmarks first
    'far_first': False,
    # 'close_first': Optimize on only close landmarks first
    'close_first': False,
    # 'interpolate_high_error': After a first optimization run, interpolate position and orientation of
    #  high error frames as initialization for next optimization. Error is distance
    #  between marked and reprojected point. Deactivate with -1
    'interpolate_high_error': 100,  # 15,
    # 'interpolate_close_marker': After first optimization run, interpolate any position that is closer
    # than this value to a labeled feature.
    # Deactivate with -1
    'interpolate_close_marker': -1,  # 1e-3,
    # 'realign_by_distant_features': Realign camera with only further half of landmarks after optimization
    'realign_by_distant_features': False,
    # 'lm_close_pen_weight': penalty factor for positions close to landmarks (avoids positions falling
    # exactly on landmarks)
    'lm_close_pen_weight': 0 * 1e-2,
    # In the last step, pass over the data, optimizing pos and ori for each frame
    # individually
    'optimize_single_frames': False,
    # object stays in one position, only estimate single position + rotations
    'single_position': False,
    # initial positions are lifted up by this factor
    'lift_factor': 0,
    # Only process frames above 'max_error_median_factor'*median(error)
    'max_error_median_factor': 0,
    # Supply jacobian sparsity matrix to least_squares
    'sparse_jacobian': False,
}

angular_jax_po_single = {
    'optimization': {
        'method': 'trf',
        'ftol': 1e-3,
        'xtol': 1e-6,
        'gtol': 1e-6,
        'x_scale': 'jac',
        'loss': 'linear',
        'tr_solver': 'lsmr',
        'max_nfev': 500,
        'verbose': 0,
        'jac': 'jax',
    },
    'optimize': [["positions", "oris"],],
    'objective_function': 'angular',
    # 'far_first': Optimize on only far landmarks first
    'far_first': False,
    # 'close_first': Optimize on only close landmarks first
    'close_first': False,
    # 'interpolate_high_error': After a first optimization run, interpolate position and orientation of
    #  high error frames as initialization for next optimization. Error is distance
    #  between marked and reprojected point. Deactivate with -1
    'interpolate_high_error': -1,
    # 'interpolate_close_marker': After first optimization run, interpolate any position that is closer
    # than this value to a labeled feature.
    # Deactivate with -1
    'interpolate_close_marker': -1,
    # 'realign_by_distant_features': Realign camera with only further half of landmarks after optimization
    'realign_by_distant_features': False,
    # 'lm_close_pen_weight': penalty factor for positions close to landmarks (avoids positions falling
    # exactly on landmarks)
    'lm_close_pen_weight': 0 * 1e-1,
    # In the last step, pass over the data, optimizing pos and ori for each frame
    # individually
    'optimize_single_frames': True,
    # object stays in one position, only estimate single position + rotations
    'single_position': False,
    # initial positions are lifted up by this factor
    'lift_factor': 0.0,
    # Only process frames above 'max_error_median_factor'*median(error)
    'max_error_median_factor': 0,
    # Supply jacobian sparsity matrix to least_squares
    'sparse_jacobian': False,
}

repro_jax_po_single = {
    'optimization': {
        'method': 'trf',
        'ftol': 1e-6,
        'xtol': 1e-6,
        'gtol': 1e-8,
        'x_scale': 'jac',
        'loss': 'linear',
        # 'tr_solver': 'lsmr' is forced since jac_sparsity is also passed to least_squares function
        'tr_solver': 'lsmr',
        'max_nfev': 500,
        'verbose': 0,
        'jac': 'jax',
    },
    'optimize': [["positions", "oris"],],
    'objective_function': 'repro',
    # 'far_first': Optimize on only far landmarks first
    'far_first': False,
    # 'close_first': Optimize on only close landmarks first
    'close_first': False,
    # 'interpolate_high_error': After a first optimization run, interpolate position and orientation of
    #  high error frames as initialization for next optimization. Error is distance
    #  between marked and reprojected point. Deactivate with -1
    'interpolate_high_error': 100,  # 15,
    # 'interpolate_close_marker': After first optimization run, interpolate any position that is closer
    # than this value to a labeled feature.
    # Deactivate with -1
    'interpolate_close_marker': -1,  # 1e-3,
    # 'realign_by_distant_features': Realign camera with only further half of landmarks after optimization
    'realign_by_distant_features': False,
    # 'lm_close_pen_weight': penalty factor for positions close to landmarks (avoids positions falling
    # exactly on landmarks)
    'lm_close_pen_weight': 0 * 1e-2,
    # In the last step, pass over the data, optimizing pos and ori for each frame
    # individually
    'optimize_single_frames': True,
    # object stays in one position, only estimate single position + rotations
    'single_position': False,
    # initial positions are lifted up by this factor
    'lift_factor': 0,
    # Only process frames above 'max_error_median_factor'*median(error)
    'max_error_median_factor': 0,
    # Supply jacobian sparsity matrix to least_squares
    'sparse_jacobian': False,
}
