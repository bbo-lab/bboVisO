from tetherlessextractor.data_object import Data


def load(yml_path):
    data_tl_obj = Data.load(yml_path)
    data_tl = data_tl_obj.data

    calibration_type = data_tl["battery_cycle"]["landmark_labels"]["calibration"]
    data = {
        "calibration": data_tl["dataset"]["calibration_headcam"][calibration_type],
        "cam_offsets": data_tl_obj.get_headcam_time_offsets().astype(float)/data_tl_obj.time_factor,
        "cam_fps": data_tl_obj.get_headcam_fps(),
        "acm_labels": data_tl["battery_cycle"]["landmark_labels"]["file"],
        "landmark_coordinates": data_tl["dataset"]["landmark_coordinates"],
        "time_range": data_tl_obj.get_time_range().astype(float)/data_tl_obj.time_factor,
        "videos": [data_tl["battery_cycle"]["headcam"][vidname]["video"]
                   for vidname in data_tl["battery_cycle"]["landmark_labels"]["videos"]],
    }

    return data
