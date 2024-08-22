import numpy as np
from calibcamlib import Camerasystem
from scipy.spatial.transform import Rotation as R

from bbo.geometry import RigidTransform


def get_cam_time_bases(max_frame, cam_fps, cam_offsets, opts):
    offsets = (np.array(cam_offsets)).astype(np.uint64)

    cam_time_bases = []
    for fps, offset in zip(cam_fps, offsets):
        fps = float(fps)
        offset_fr = offset * fps
        # Round frame offset if very close to integer
        if offset_fr - np.round(offset_fr) < 0.01:
            offset_fr = np.round(offset_fr)

        cam_time_bases.append(((np.arange(max_frame).astype(float) + offset_fr) / fps).astype(np.uint64))

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
    landmarks_weights = []  # Should it be used at all?
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
            landmarks_weights.append(landmark_coordinates["landmarks"][ln].get("weight", 1))
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

        landmarks_use.append(landmark_coordinates["landmarks"][ln].get("use", True))
        landmarks_weights.append(landmark_coordinates["landmarks"][ln].get("weight", 1))

    landmarks_optimize = np.array(landmarks_optimize)
    landmarks_far = np.array(landmarks_far)
    landmarks_use = np.array(landmarks_use)
    landmarks_weights = np.array(landmarks_weights)
    landmarks_all = np.array(landmarks_all)
    return landmarks_all, landmarks_far, landmarks_identity, landmarks_optimize, landmarks_use, landmarks_weights


def generate_immediate_neighbor_mask(ts, neighbor_ts, max_dist):
    ts = ts.reshape((-1,))
    neighbor_ts = neighbor_ts.reshape((-1, 1))
    return np.any((ts >= neighbor_ts - max_dist) & (ts <= neighbor_ts), axis=0) & np.any(
        (ts <= neighbor_ts + max_dist) & (ts >= neighbor_ts), axis=0)


def realign_by_distant_features(landmark_coords3d, imcoords, times, positions, orientations, calibs):
    orientations_filtered = orientations.as_quat()
    landmark_vecs = landmark_vecs_from_imcoords(imcoords, calibs)
    no_common = 0
    for i_t, t in enumerate(times):
        frame_landmark_vectors = landmark_coords3d - positions[i_t]
        landmark_dists = np.linalg.norm(frame_landmark_vectors, axis=-1, keepdims=True)
        frame_landmark_vectors /= landmark_dists
        landmark_vec_mask = ~np.any(np.isnan(frame_landmark_vectors), axis=-1)
        cam_vec_mask = ~np.any(np.isnan(landmark_vecs[i_t]), axis=-1)
        common_mask = landmark_vec_mask & cam_vec_mask

        if np.sum(common_mask) < 2:
            no_common += 1
            continue

        landmark_dists[~common_mask] = np.nan
        furthest_mask = landmark_dists[:, 0] >= np.sort(landmark_dists[common_mask, 0])[
            np.min((int(np.sum(common_mask) / 2), np.sum(common_mask) - 2))]

        align = R.align_vectors(
            frame_landmark_vectors[common_mask], landmark_vecs[i_t][common_mask]
        )[0]
        orientations_filtered[i_t] = align.as_quat()
    print("Not optimized due to lack of common vectors: ", no_common, "/", len(times))
    return R.from_quat(orientations_filtered)


def landmark_vecs_from_imcoords(landmark_imcoords, calibs):
    cs = Camerasystem.from_calibs(calibs)

    landmark_vecs_cams = np.empty(landmark_imcoords.shape[:-1] + (3,))
    for i_c, c in enumerate(cs.cameras):
        landmark_vecs_cams[i_c] = c["camera"].sensor_to_space(landmark_imcoords[i_c])

    return landmark_vecs_cams


def landmark_imcoords_from_vecs(landmark_vecs, calibs):
    cs = Camerasystem.from_calibs(calibs)

    landmark_imcoords = np.empty(landmark_vecs.shape[:-1] + (2,))
    for i_c, c in enumerate(cs.cameras):
        landmark_imcoords[i_c] = c["camera"].space_to_sensor(landmark_vecs[i_c])

    return landmark_imcoords


def update_calibs_extrinsics(calibs, cam_rotvecs, cam_ts):
    for i_c, c in enumerate(calibs):
        c["rvec_cam"] = cam_rotvecs[i_c]
        c["tvec_cam"] = cam_ts[i_c]

    return calibs


def get_var_edges(m, k, k_p, n):
    field_sizes = np.array([0,
                            n * 3,  # landmarks
                            k_p * 3,  # positions
                            k * 3,  # orientations
                            m * 3,  # cam extrinsic rotations
                            m * 3,  # cam extrinsic positions
                            ])
    edges = np.cumsum(field_sizes)
    return edges
