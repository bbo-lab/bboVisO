import warnings

import numpy as np
from calibcamlib.camerasystem import Camerasystem
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

from bboVisO.helper import generate_immediate_neighbor_mask, landmark_vecs_from_imcoords, landmark_imcoords_from_vecs
from bboVisO.methods.repro.optfuncs import objfun_runner as objfun_runner_repro

from bbo.geometry import RigidTransform

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

    trafo = RigidTransform(ori[mask], pos[mask])
    trafo_interp = trafo.interpolate(time[mask], fill_boundary="constant")(time)

    return trafo_interp.get_translation(), trafo_interp.get_rotation()


def interpolate_imcoords(imcoords, time_base, calibs, sub_time_bases=None):
    """Fill unrecorded times between consecutive valid camera measurements.

    Interpolate each landmark only when both adjacent recorded frames have a
    valid direction. Missing detections at recorded times are not bridged, and
    times outside the camera's measurements are not extrapolated. With no
    sub_time_bases, every time is treated as recorded. The returned boolean
    mask marks successfully interpolated image coordinates.
    """
    vecs_cams = landmark_vecs_from_imcoords(imcoords, calibs)
    times = np.asarray(time_base)
    interpolation_mask = np.zeros_like(imcoords, dtype=bool)

    for i_c, vecs_cam in enumerate(vecs_cams):
        if sub_time_bases is None:
            continue
        recorded = np.isin(times, sub_time_bases[i_c])
        recorded_idxs = np.flatnonzero(recorded)
        if recorded_idxs.size < 2:
            continue

        # Bracket every additional time by consecutive recorded frames, even
        # when multiple rows of the shared time base fall between them.
        target_idxs = np.flatnonzero(~recorded)
        post = np.searchsorted(recorded_idxs, target_idxs)
        inside = (post > 0) & (post < recorded_idxs.size)
        target_idxs = target_idxs[inside]
        post = post[inside]
        if target_idxs.size == 0:
            continue
        pre_idxs = recorded_idxs[post - 1]
        post_idxs = recorded_idxs[post]

        pre_vecs = vecs_cam[pre_idxs]
        post_vecs = vecs_cam[post_idxs]
        fill = (np.isnan(vecs_cam[target_idxs]).any(axis=-1)
                & np.isfinite(pre_vecs).all(axis=-1)
                & np.isfinite(post_vecs).all(axis=-1))
        rows, landmarks = np.nonzero(fill)
        if rows.size == 0:
            continue

        pre_vecs = pre_vecs[rows, landmarks]
        post_vecs = post_vecs[rows, landmarks]
        t = ((times[target_idxs[rows]] - times[pre_idxs[rows]])
             / (times[post_idxs[rows]] - times[pre_idxs[rows]]))[:, np.newaxis]
        th = np.arccos(np.clip(np.sum(pre_vecs * post_vecs, axis=-1), -1, 1))
        small_angle = th < 1e-3
        th_safe = np.where(small_angle, 1e-3, th)[:, np.newaxis]
        interp_vecs = (pre_vecs * np.sin((1 - t) * th_safe)
                       + post_vecs * np.sin(t * th_safe)) / np.sin(th_safe)

        # Normalized LERP avoids instability for nearly identical directions.
        lerp = pre_vecs[small_angle] * (1 - t[small_angle]) + post_vecs[small_angle] * t[small_angle]
        lerp /= np.linalg.norm(lerp, axis=-1, keepdims=True)
        interp_vecs[small_angle] = lerp

        valid = np.isfinite(interp_vecs).all(axis=-1)
        filled_idxs = target_idxs[rows[valid]]
        filled_landmarks = landmarks[valid]
        vecs_cam[filled_idxs, filled_landmarks] = interp_vecs[valid]
        interpolation_mask[i_c, filled_idxs, filled_landmarks] = True

    imcoords_interp = landmark_imcoords_from_vecs(vecs_cams, calibs)
    interpolation_mask &= np.isfinite(imcoords_interp)
    return imcoords_interp, vecs_cams, time_base, interpolation_mask


def interpolate_imcoords_rebase(imcoords_or_vecs_cams, im_array_time_base, calibs, interp_times, interp_dt=None,
                                omit_reprojection=False):
    # This interpolates within each cam, ensuring that all interpolated vectors lie within the valid FOV of each cam
    if calibs is not None:
        vecs_cams = landmark_vecs_from_imcoords(imcoords_or_vecs_cams, calibs)
    else:
        vecs_cams = imcoords_or_vecs_cams
    vecs_cams_interp = np.full(vecs_cams.shape[0:1] + (len(interp_times),) + vecs_cams.shape[2:3] + (3,), np.nan)

    for i_c, vecs_cam in enumerate(vecs_cams):
        nnanmask = np.any(~np.isnan(vecs_cam[..., 0]), axis=1) # Over x coordinate of all labels, frame dim remains
        vecs_cam = vecs_cam[nnanmask]
        time_base_cam = im_array_time_base[nnanmask]

        base_dt = np.diff(time_base_cam)

        if interp_dt is None:
            interp_dt = np.nanmin(base_dt)

        post_idxs = np.digitize(interp_times, time_base_cam)
        pre_idxs = post_idxs - 1
        interp_times_mask = (pre_idxs >= 0) & (post_idxs < len(time_base_cam))  # Rest ist outside of bins

        pre_idxs = pre_idxs[interp_times_mask]
        post_idxs = post_idxs[interp_times_mask]

        pretimes = interp_times[interp_times_mask] - time_base_cam[pre_idxs]
        posttimes = base_dt[pre_idxs] - pretimes

        t = (pretimes / (pretimes + posttimes))[..., np.newaxis]
        th = np.arccos(np.clip(np.sum((vecs_cam[pre_idxs] * vecs_cam[post_idxs]), axis=-1), -1, 1))
        th_d = th.copy()
        th_d[th==0] = np.nan

        vecs_cam_pre = vecs_cam[pre_idxs]
        vecs_cam_interp = (
                vecs_cam_pre * (np.sin((1 - t) * th_d) / np.sin(th_d))[..., np.newaxis] +
                vecs_cam[post_idxs] * (np.sin(t * th_d) / np.sin(th_d))[..., np.newaxis]
        )  # Slerp

        # Slerp divides by 0 if angle betweeh pre and post is 0, fix here
        vecs_cam_interp[th==0] = vecs_cam_pre[th==0]

        vecs_cams_interp[i_c, interp_times_mask] = vecs_cam_interp

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,1)
        # for i_d in range(1):
        #     line = ax.plot(interp_times, vecs_cams_interp[i_c,:,:, i_d])
        #     ax.plot(time_base_cam, vecs_cam[:, :, i_d], '+', color=line[0].get_color())
        # plt.xlim((2e7, 2.02e7))
        # plt.show()

        # Remove interpolations with too large distances to adjacent frames
        interp_times_idxs = np.where(interp_times_mask)[0]
        interp_remove_mask = (pretimes > interp_dt) | (posttimes > interp_dt)
        vecs_cams_interp[i_c, interp_times_idxs[interp_remove_mask]] = np.nan

    if not omit_reprojection and calibs is not None:
        imcoords_interp = landmark_imcoords_from_vecs(vecs_cams_interp, calibs)
    else:
        imcoords_interp = np.full(vecs_cams_interp.shape[0:3] + (2,), np.nan)

    return imcoords_interp, vecs_cams_interp, interp_times
