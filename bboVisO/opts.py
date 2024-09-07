import sys


def get_opts():
    return {
        'iterations': [
            angular_jax_po_pol_bALL_s10(),
            repro_jax_po_pol_bALL_s10(),
            angular_jax_po_b1_s1(),
            repro_jax_po_b1_s1(),
            repro_jax_po_b1_s1_nointerp(),
            repro_jax_po_pol_bALL_s10_nointerp(),
            repro_jax_po_b1_s1_nointerp(),
            # repro_jax_po_pol_bALL_cam(),
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


def default_iter_opts():
    return {
        'optimization': {
            'method': 'trf',
            'ftol': 1e-7,
            'xtol': 1e-7,
            'gtol': 1e-7,
            'x_scale': 'jac',
            'loss': 'linear',
            'tr_solver': 'lsmr',
            'max_nfev': 500,
            'verbose': 2,
            'jac': 'jax',
        },
        'optimize': [["positions", "oris"],
                     ["positions", "oris", "landmarks"]],
        'objective_function': 'repro',
        # 'far_first': Optimize on only far landmarks first
        'far_first': False,
        # 'close_first': Optimize on only close landmarks first
        'close_first': False,
        # 'interpolate_high_error': After a first optimization run, interpolate position and orientation of
        #  high error frames as initialization for next optimization. Error is distance
        #  between marked and reprojected point. Deactivate with -1
        'interpolate_high_error': 100,
        # 'interpolate_close_marker': After first optimization run, interpolate any position that is closer
        # than this value to a labeled feature.
        # Deactivate with -1
        'interpolate_close_marker': -1,
        # 'realign_by_distant_features': Realign camera with only further half of landmarks after optimization
        'realign_by_distant_features': False,
        # 'lm_close_pen_weight': penalty factor for positions close to landmarks (avoids positions falling
        # exactly on landmarks)
        'lm_close_pen_weight': 1e-1,
        # Size of frame batches
        'frame_batch_size': int(1e6),
        # Steps of frames
        'frame_batch_step': int(1),
        # object stays in one position, only estimate single position + rotations
        # TODO: this should only show up here OR in global options
        'single_position': False,
        # initial positions are lifted by this factor
        'lift_factor': 0.0,
        # Only process frames above 'max_error_median_factor'*median(error)
        'max_error_median_factor': 0,
        # Supply jacobian sparsity matrix to least_squares
        'sparse_jacobian': False,
        # Parallelize over batches
        'parallelize': False,
        # Weight to apply to residuals for interpolated image coordinates
        'interpolated_weight': 0.2
    }


def default_repro_iter_opts():
    iter_opts = default_iter_opts()
    iter_opts['optimization']['ftol'] = 1e-7
    iter_opts['optimization']['gtol'] = 1e-7
    iter_opts['optimization']['xtol'] = 1e-7
    iter_opts['optimization']['verbose'] = 2
    iter_opts['objective_function'] = 'repro'
    return iter_opts


def default_angular_iter_opts():
    iter_opts = default_iter_opts()
    iter_opts['optimization']['ftol'] = 1e-3
    iter_opts['optimization']['gtol'] = 1e-6
    iter_opts['optimization']['xtol'] = 1e-6
    iter_opts['optimization']['verbose'] = 2
    iter_opts['objective_function'] = 'angular'
    return iter_opts


def repro_jax_po_pol_bALL_s10():
    iter_opts = default_repro_iter_opts()
    iter_opts['optimize'] = [
        ["positions", "oris"],
        ["positions", "oris", "landmarks"], ]
    iter_opts['frame_batch_step'] = int(10)
    return iter_opts

def repro_jax_po_pol_bALL_cam():
    iter_opts = default_repro_iter_opts()
    iter_opts['optimize'] = [
        ["positions", "oris"],
        ["positions", "oris", "landmarks", "extrinsics_r", "extrinsics_t"],
    ]
    iter_opts['frame_batch_step'] = int(10)
    return iter_opts

def repro_jax_po_pol_bALL_s10_nointerp():
    iter_opts = default_repro_iter_opts()
    iter_opts['interpolate_high_error'] = -1
    return iter_opts


def angular_jax_po_pol_bALL_s10():
    iter_opts = default_angular_iter_opts()
    iter_opts['optimize'] = [
        ["positions", "oris"],
        ["positions", "oris", "landmarks"], ]
    iter_opts['frame_batch_step'] = int(10)
    return iter_opts


def repro_jax_po_b1_s1():
    iter_opts = default_repro_iter_opts()
    iter_opts['optimization']['verbose'] = 0
    iter_opts['optimize'] = [
        ["positions", "oris"], ]
    iter_opts['frame_batch_step'] = int(1)
    iter_opts['frame_batch_size'] = int(1)
    iter_opts['parallelize'] = False
    return iter_opts


def repro_jax_po_b1_s1_nointerp():
    iter_opts = repro_jax_po_b1_s1()
    iter_opts['optimization']['max_nfev'] = 2000
    iter_opts['interpolate_high_error'] = -1
    return iter_opts


def angular_jax_po_b1_s1():
    iter_opts = default_angular_iter_opts()
    iter_opts['optimization']['verbose'] = 0
    iter_opts['optimization']['max_nfev'] = 2000
    iter_opts['optimize'] = [
        ["positions", "oris"], ]
    iter_opts['frame_batch_step'] = int(1)
    iter_opts['frame_batch_size'] = int(1)
    iter_opts['parallelize'] = False
    return iter_opts


def angular_jax_po_b1_s1_nointerp():
    iter_opts = angular_jax_po_b1_s1()
    iter_opts['optimization']['max_nfev'] = 2000
    iter_opts['interpolate_high_error'] = -1
    return iter_opts