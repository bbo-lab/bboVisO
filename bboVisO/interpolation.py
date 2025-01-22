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
    # Interpolates time points in array_2d that are not part of sub_time_bases
    vecs_cams = landmark_vecs_from_imcoords(imcoords, calibs)

    pretimes = np.diff(time_base)

    sub_time_base_members = []
    for i_c, vecs_cam in enumerate(vecs_cams):
        # create mask of frames that belong to one of the sub_time_bases that make up the array_time_base
        if sub_time_bases is None:
            sub_time_base_members.append(np.ones_like(time_base, dtype=bool))
        else:
            sub_time_base_members.append(np.isin(time_base, sub_time_bases[i_c]))

        t = (pretimes[:-1] / (pretimes[:-1] + pretimes[1:]))[..., np.newaxis]
        th = np.arccos(np.clip(np.sum((vecs_cam[:-2] * vecs_cam[2:]), axis=-1), -1, 1))
        vecs_cam_interp = (
                vecs_cam[:-2] * (np.sin((1 - t) * th) / np.sin(th))[..., np.newaxis] +
                vecs_cam[2:] * (np.sin(t * th) / np.sin(th))[..., np.newaxis]
        )  # Slerp

        # This changes cam_vecs IN PLACE!, replacing only the values previously nan
        vecs_cam_ireg = vecs_cam[1:-1]
        mask = np.isnan(vecs_cam_ireg) & (~sub_time_base_members[-1][1:-1, np.newaxis, np.newaxis])
        vecs_cam_ireg[mask] = vecs_cam_interp[mask]

    imcoords_interp = landmark_imcoords_from_vecs(vecs_cams, calibs)
    interpolation_mask = np.zeros_like(imcoords, dtype=bool)
    interpolation_mask[np.array(sub_time_base_members)] = True

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
