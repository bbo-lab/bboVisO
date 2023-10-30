import numpy as np


def get_cam_time_bases(frame_idxs, cam_fps, cam_offsets, opts):
    offsets = (np.array(cam_offsets)).astype(np.uint64) + opts["add_offset"]

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
