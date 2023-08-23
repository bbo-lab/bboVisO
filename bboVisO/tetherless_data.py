from tetherlessextractor.data_object import Data

def load(yml_path):
    data_tl_obj = Data.load(yml_path)
    data_tl = data_tl_obj.data

    calibration_type = data_tl["battery_cycle"]["landmark_labels"]["calibration"]
    data = {
        "calibration": data_tl["dataset"]["calibration_headcam"][calibration_type],
        "cam_offsets": data_tl_obj.get_headcam_frame_offsets(),
        "acm_labels": data_tl["battery_cycle"]["landmark_labels"]["file"],
        "landmark_coordinates": data_tl["battery_cycle"]["landmark_coordinates"],
        "videos": [data_tl["battery_cycle"]["headcam"][vidname]["video"]
                   for vidname in data_tl["battery_cycle"]["landmark_labels"]["videos"]]
    }

    return data