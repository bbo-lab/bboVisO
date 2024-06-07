import warnings

import numpy as np
from calibcamlib.camerasystem import Camerasystem
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

from bboVisO.helper import generate_immediate_neighbor_mask, landmark_vecs_from_imcoords, landmark_imcoords_from_vecs
from bboVisO.methods.repro.optfuncs import objfun_runner as objfun_runner_repro


def interpolate_high_error(times, positions, oris_flu, calibs, opts, landmarks, optimize,
                           landmark_imcoords=None,
                           landmark_vecs=None, landmarks_optimize=None):
    if opts['interpolate_high_error'] == 0 or opts['interpolate_high_error'] == -1:
        return positions, oris_flu

    objfun_error = objfun_runner_repro(landmarks, positions, oris_flu,
                                       landmark_imcoords, calibs, opts)
    abserrors = np.sqrt(np.nansum(objfun_error ** 2, axis=-1))
    abserrors[abserrors == 0] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        frame_errors_max = np.nanmax(abserrors, axis=(0, -1))
        frame_errors_median = np.nanmedian(abserrors, axis=(0, -1))
    low_error_mask = frame_errors_max < opts['interpolate_high_error']
    low_error_mask[(0, -1),] = True  # We never want extrapolation
    #
    # if np.sum(~low_error_mask)/len(low_error_mask)>0.7:
    #     print("Too many high max errors, reverting to median error.")
    #     frame_errors = np.nanmedian(abserrors, axis=-1)
    #     low_error_mask = frame_errors < opts['interpolate_high_error']

    if (np.sum(~low_error_mask) / low_error_mask.shape[0]) > 0.95:
        print("Too many high median errors, not interpolating.")
        return positions, oris_flu

    positions_interp, oris_flu_interp = interpolate_mask(times, positions, R.from_rotvec(oris_flu), low_error_mask)
    oris_flu_interp = oris_flu_interp.as_rotvec()

    objfun_error_interp = objfun_runner_repro(
        landmarks, positions_interp, oris_flu_interp, landmark_imcoords, calibs, opts
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        abserrors_interp = np.sqrt(np.nansum(objfun_error_interp ** 2, axis=-1))
        abserrors_interp[abserrors_interp == 0] = np.nan
        frame_errors_max_interp = np.nanmax(abserrors_interp, axis=(0, -1))
        frame_errors_median_interp = np.nanmedian(abserrors_interp, axis=(0, -1))
    low_error_mask_interp = frame_errors_max_interp < opts['interpolate_high_error']

    # We never want to make the max or median error worse
    worse_mask = (
            (frame_errors_median_interp > frame_errors_median) &
            (frame_errors_max_interp > frame_errors_max))
    positions_interp[worse_mask] = positions[worse_mask]
    oris_flu_interp[worse_mask] = oris_flu[worse_mask]

    print(f"High error")
    print(f"Interpolated frames ({np.sum(~low_error_mask)}/{low_error_mask.shape[0]})")
    print(f"Replaced frames     ({np.sum((~low_error_mask) & (~worse_mask))}/{low_error_mask.shape[0]})")
    print(f"Left hi_err frames  ({np.sum(~low_error_mask_interp)}/{low_error_mask.shape[0]})")

    return positions_interp, oris_flu_interp


def interpolate_close_marker(times, positions, oris_flu, landmarks, opts):
    if opts['interpolate_close_marker'] == 0 or opts['interpolate_close_marker'] == -1:
        return positions, oris_flu

    close_marker_mask = np.min(np.linalg.norm(positions[:, np.newaxis] - landmarks[np.newaxis], axis=-1), axis=1) < \
                        opts['interpolate_close_marker']

    if not np.any(close_marker_mask):
        return positions, oris_flu
    elif opts['lm_close_pen_weight'] > 0:
        print("Found positions in landmarks. This should be prevented by the penalty ...")

    print(f"In landmark - Interpolated times ({np.sum(~close_marker_mask)}/{close_marker_mask.shape[0]}): ",
          ", ".join([str(n) for n in times[~close_marker_mask]]))

    positions_interp, oris_flu_interp = interpolate_mask(times, positions, R.from_rotvec(oris_flu), close_marker_mask)

    return positions_interp, oris_flu_interp.as_rotvec()


def interpolate_mask(time, pos, ori, mask):
    if np.sum(mask) / len(mask) < 0.25 or np.sum(mask) == 1:
        return pos, ori

    pos_interp_f = [interp1d(time[mask], p, fill_value='extrapolate') for p in pos[mask].T]
    pos_interp = np.array([i_f(time) for i_f in pos_interp_f]).T

    mask[0] = True  # Slerp cannot extrapolate
    mask[-1] = True
    ori_spline = Slerp(time[mask], ori[mask])
    # ori_q = ori.as_quat()
    # ori_q = ori_q * np.sign(ori_q[:, (0,)])
    # ori_interp_f = [interp1d(time[mask], o, fill_value='extrapolate') for o in ori_q[mask].T]
    # ori_interp = np.array([i_f(time) for i_f in ori_interp_f]).T

    return pos_interp, ori_spline(time)


def interpolate_imcoords(imcoords, time_base, calibs, sub_time_bases=None):
    # Interpolates time points in array_2d that are not part of sub_time_bases
    vecs_cams = landmark_vecs_from_imcoords(imcoords, calibs)

    pretimes = time_base[1:] - time_base[:-1]

    sub_time_base_members = []
    for i_c, vecs_cam in enumerate(vecs_cams):
        # create mask of frames that belong to one of the sub_time_bases that make up the array_time_base
        if sub_time_bases is None:
            sub_time_base_members.append(np.ones_like(time_base, dtype=bool))
        else:
            sub_time_base_members.append(np.isin(time_base, sub_time_bases[i_c]))

        vecs_cam_interp = (vecs_cam[:-2] * pretimes[1:][..., np.newaxis, np.newaxis] +
                           vecs_cam[2:] * pretimes[:-1][..., np.newaxis, np.newaxis])
        vecs_cam_interp /= np.linalg.norm(vecs_cam_interp, axis=-1, keepdims=True)

        # This changes cam_vecs IN PLACE!, replacing only the values previously nan
        vecs_cam_ireg = vecs_cam[1:-1]
        mask = np.isnan(vecs_cam_ireg) & (~sub_time_base_members[-1][1:-1, np.newaxis, np.newaxis])
        vecs_cam_ireg[mask] = vecs_cam_interp[mask]

    imcoords_interp = landmark_imcoords_from_vecs(vecs_cams, calibs)
    interpolation_mask = np.zeros_like(imcoords, dtype=bool)
    interpolation_mask[np.array(sub_time_base_members)] = True

    return imcoords_interp, vecs_cams, time_base, interpolation_mask


def interpolate_imcoords_rebase(array_2d, array_time_base, calibs, dt, time_base=None):
    cs = Camerasystem.from_calibs(calibs)
    array_3d = cs.get_camera_lines(array_2d)[0]

    if time_base is None:
        time_base = array_time_base

    array_3d_interp = np.full((array_2d.shape[0], len(time_base), array_2d.shape[2], 3), np.nan)
    for i_c, array_3d_cam in enumerate(array_3d):
        array_3d_cam = array_3d_cam.transpose((1, 0, 2))
        for i_m, array_3d_cam_marker in enumerate(array_3d_cam):
            nan_mask = np.any(np.isnan(array_3d_cam_marker), axis=-1)
            if np.sum(~nan_mask) < 2:
                continue
            no_nan_array_time_base = array_time_base[~nan_mask]
            array_3d_cam_marker_if = [interp1d(array_time_base[~nan_mask], a, fill_value="extrapolate") for a in
                                      array_3d_cam_marker[~nan_mask].T]
            array_3d_cam_marker_i = np.array([i_f(time_base) for i_f in array_3d_cam_marker_if]).T
            array_3d_cam_marker_i[~generate_immediate_neighbor_mask(time_base, no_nan_array_time_base, dt)] = np.nan
            array_3d_interp[i_c, :, i_m] = array_3d_cam_marker_i

    array_2d_interp = cs.project_dir(array_3d_interp)
    array_2d_out = np.diagonal(array_2d_interp, axis1=0, axis2=1).transpose((3, 0, 1, 2))

    return array_2d_out, array_3d_interp, time_base
