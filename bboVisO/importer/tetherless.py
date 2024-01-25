import sys

from tetherlessextractor.data_object import Data
from bbo import yaml

def load(yml_path, opts):
    data_tl_obj = Data.load(yml_path)
    data_tl = data_tl_obj.data

    calibration_type = data_tl["battery_cycle"]["landmark_labels"]["calibration"]
    data = {
        "name": data_tl["battery_cycle"]["name"],
        "calibration": data_tl["dataset"]["calibration_headcam"][calibration_type],
        "cam_offsets": data_tl_obj.get_headcam_time_offsets(timebase="battery_cycle").astype(float),
        "cam_fps": data_tl_obj.get_headcam_fps().astype(float) / data_tl_obj.time_factor,
        "cam_time_bases": data_tl_obj.get_headcam_times(timebase="battery_cycle"),
        "labels": data_tl["battery_cycle"]["landmark_labels"]["file"],
        "landmark_coordinates": yaml.load(data_tl["dataset"]["landmark_coordinates"])["landmark_coordinates"],
        "time_range": data_tl_obj.get_time_range().astype(float),
        "videos": data_tl_obj.get_headcam_video_files(),
    }
    opts['time_dtype'] = int

    if opts['text_out'] == "canonical":
        opts['text_out'] = (yml_path.parents[1] / "data" / "bboVisO").expanduser().resolve().as_posix()

    return data, opts
