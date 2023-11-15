import numpy as np
from calibcamlib import Camerasystem
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation as R

def get_cam_time_bases(frame_idxs, cam_fps, cam_offsets, opts):
    offsets = (np.array(cam_offsets)).astype(np.uint64)

    cam_time_bases = [
        (frame_idxs.astype(float) / float(fps)).astype(np.uint64) + offset
        for fps, offset in zip(cam_fps, offsets)
    ]
    # Adjust time that are just off by 1us from float rounding, to not create an additional measurement
    for i_cam in range(1, len(cam_time_bases)):
        for i_t, t in enumerate(cam_time_bases[i_cam]):
            mask = np.abs(cam_time_bases[0] - t) == 1
            s_mask = np.sum(mask)
            assert s_mask < 2, "A time seems to appear multiple times in the time base."
            if s_mask == 0:
                continue
            else:
                # print(f"Rplacing {cam_time_bases[i_cam][i_t]} with {cam_time_bases[0][mask][0]}")
                cam_time_bases[i_cam][i_t] = cam_time_bases[0][mask]

    cam_time_bases = [ctb.astype(opts["time_dtype"]) for ctb in cam_time_bases]
    return cam_time_bases


def landmarks_to_array(landmark_coordinates, detected_labels):
    # Build landmark coordinates
    landmarks_all = []  # Coordinates of landmarks, may be nan
    landmarks_optimize = []  # Should this coordinate be optimized by bboViso?
    landmarks_far = []  # Is it a far away landmark (used in first run)
    landmarks_use = []  # Should it be used at all?
    landmarks_identity = []  # Is it identical to another one?
    # Only extract landmarks taht are also marked
    for ln in detected_labels:
        #
        if "position" not in landmark_coordinates["landmarks"][ln] or \
                len(landmark_coordinates["landmarks"][ln]["position"]) == 0:
            landmark_coordinates["landmarks"][ln]["position"] = [np.nan, np.nan, np.nan]

        if "identity" in landmark_coordinates["landmarks"][ln]:
            lm_idx = detected_labels.index(landmark_coordinates["landmarks"][ln]["identity"])
            landmarks_all[lm_idx] = np.nanmean(np.array([landmarks_all[lm_idx],
                                                         landmark_coordinates["landmarks"][ln]["position"]]),
                                               axis=0)

            landmarks_all.append([np.nan, np.nan, np.nan])
            landmarks_optimize.append(True)
            landmarks_far.append(False)
            landmarks_use.append(False)
            landmarks_identity.append(landmark_coordinates["landmarks"][ln]["identity"])
            continue

        landmarks_identity.append(None)
        if not np.any(np.isnan(landmark_coordinates["landmarks"][ln]["position"])):
            landmarks_all.append(landmark_coordinates["landmarks"][ln]["position"])
            landmarks_optimize.append(landmark_coordinates["landmarks"][ln]["optimize"])
            landmarks_far.append(landmark_coordinates["landmarks"][ln]["far"])
        else:
            landmarks_all.append([np.nan, np.nan, np.nan])
            landmarks_optimize.append(True)
            landmarks_far.append(False)
        if "use" in landmark_coordinates["landmarks"][ln]:
            landmarks_use.append(landmark_coordinates["landmarks"][ln]["use"])
        else:
            landmarks_use.append(True)
    landmarks_optimize = np.array(landmarks_optimize)
    landmarks_far = np.array(landmarks_far)
    landmarks_use = np.array(landmarks_use)
    landmarks_all = np.array(landmarks_all)
    return landmarks_all, landmarks_far, landmarks_identity, landmarks_optimize, landmarks_use


def interpolate_mask(time, pos, ori, mask):
    pos_interp_f = [interp1d(time[mask], p, fill_value='extrapolate') for p in pos[mask].T]
    pos_interp = np.array([i_f(time) for i_f in pos_interp_f]).T

    mask[0] = True  # Serp cannot extrapolate
    mask[-1] = True
    ori_spline = Slerp(time[mask], ori[mask])
    # ori_q = ori.as_quat()
    # ori_q = ori_q * np.sign(ori_q[:, (0,)])
    # ori_interp_f = [interp1d(time[mask], o, fill_value='extrapolate') for o in ori_q[mask].T]
    # ori_interp = np.array([i_f(time) for i_f in ori_interp_f]).T

    return pos_interp, ori_spline(time)


def interpolate_imcoords(array_2d, array_time_base, calibs):
    print("DEPRECATED:  interpolate_imcoords_rebase(array_2d, array_time_base, calibs, time_base=array_time_base) "
          "is just as fast and more potent.")
    cs = Camerasystem.from_calibs(calibs)

    # This has shape cam x times x markers x coords
    array_3d = cs.get_camera_lines(array_2d)[0]

    pretimes = array_time_base[1:] - array_time_base[:-1]

    array_3d_interp = (array_3d[:, :-2] * pretimes[1:][np.newaxis, ..., np.newaxis, np.newaxis] +
                       array_3d[:, 2:] * pretimes[:-1][np.newaxis, ..., np.newaxis, np.newaxis])
    array_3d_interp /= np.linalg.norm(array_3d_interp, axis=-1, keepdims=True)

    # This changes array_3d in place, replacing only the values previously nan
    array_3d_ireg = array_3d[:, 1:-1]
    array_3d_ireg[np.isnan(array_3d_ireg)] = array_3d_interp[np.isnan(array_3d_ireg)]

    # This has shape cam x cam x times x markers x coords
    array_2d_interp = cs.project_dir(array_3d)

    # Replace only entries that were previously nan.
    array_2d_out = array_2d.copy()
    for i_cam in range(len(array_2d_out)):
        array_2d_out[i_cam][np.isnan(array_2d_out[i_cam])] = array_2d_interp[i_cam, i_cam][np.isnan(array_2d_out[i_cam])]


    return array_2d_out, array_time_base


def interpolate_imcoords_rebase(array_2d, array_time_base, calibs, dt, time_base=None):
    cs = Camerasystem.from_calibs(calibs)
    array_3d = cs.get_camera_lines(array_2d)[0]

    if time_base is None:
        time_base = array_time_base

    array_3d_interp = np.full((array_2d.shape[0], len(time_base), array_2d.shape[2], 3), np.nan)
    for i_c, array_3d_cam in enumerate(array_3d):
        array_3d_cam = array_3d_cam.transpose((1, 0, 2))
        for i_m, array_3d_cam_marker in enumerate(array_3d_cam):
            if i_m == 123:
                pass
            nan_mask = np.any(np.isnan(array_3d_cam_marker), axis=-1)
            if np.sum(~nan_mask)<2:
                continue
            no_nan_array_time_base = array_time_base[~nan_mask]
            array_3d_cam_marker_if = [interp1d(array_time_base[~nan_mask], a, fill_value="extrapolate") for a in array_3d_cam_marker[~nan_mask].T]
            array_3d_cam_marker_i = np.array([i_f(time_base) for i_f in array_3d_cam_marker_if]).T
            array_3d_cam_marker_i[~generate_immediate_neighbor_mask(time_base, no_nan_array_time_base, dt)] = np.nan
            array_3d_interp[i_c, :, i_m] = array_3d_cam_marker_i


    array_2d_interp = cs.project_dir(array_3d_interp)
    array_2d_out = np.diagonal(array_2d_interp, axis1=0, axis2=1).transpose((3, 0, 1, 2))

    return array_2d_out, time_base

def generate_immediate_neighbor_mask(ts, neighbor_ts, max_dist):
    ts = ts.reshape((-1,))
    neighbor_ts = neighbor_ts.reshape((-1,1))
    return np.any((ts >= neighbor_ts - max_dist) & (ts <= neighbor_ts), axis=0) & np.any((ts <= neighbor_ts + max_dist) & (ts >= neighbor_ts), axis=0)