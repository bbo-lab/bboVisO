import sys
import matplotlib

matplotlib.use('Qt5Agg')

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QLineEdit, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from svidreader import get_reader

from bboVisO.acm_labels import get_labeled_frames, load as acm_load
from bboVisO.helper import get_cam_time_bases


class bboVisOchecker(QMainWindow):
    def __init__(self, data, opts):
        super().__init__()

        self.components = {
            'main': {},
            'controls': {},
        }

        # data = {
        #     "calibration": data_tl["dataset"]["calibration_headcam"][calibration_type],
        #     "cam_offsets": data_tl_obj.get_headcam_time_offsets(timebase="battery_cycle") / data_tl_obj.time_factor,
        #     "cam_fps": data_tl_obj.get_headcam_fps(),
        #     "acm_labels": data_tl["battery_cycle"]["landmark_labels"]["file"],
        #     "landmark_coordinates": data_tl["dataset"]["landmark_coordinates"],
        #     "time_range": data_tl_obj.get_time_range() / data_tl_obj.time_factor,
        #     "videos": data_tl_obj.get_headcam_video_files(),
        # }

        self.data = data
        self.opts = opts

        path = Path(opts["text_out"]).expanduser().resolve() / data["name"]

        cam_positions_pd = pd.read_csv(path / f"{data['name']}_cam_positions.csv",
                                       sep="\t")
        self.cam_positions = np.array([
            cam_positions_pd["x0"],
            cam_positions_pd["y0"],
            cam_positions_pd["z0"],
        ]).T
        cam_orientations_pd = pd.read_csv(path / f"{data['name']}_cam_orientations.csv", sep="\t")
        cam_orientations_q = np.array([
            cam_orientations_pd["x0"],
            cam_orientations_pd["y0"],
            cam_orientations_pd["z0"],
            cam_orientations_pd["w0"],
        ]).T
        self.cam_orientations = R.from_quat(cam_orientations_q)
        self.times = np.unique(np.concatenate((np.array(cam_positions_pd["t_us"]),
                                               np.array(cam_orientations_pd["t_us"]))))

        self.landmark_labels = acm_load(data["acm_labels"])["labels"]
        self.landmark_frames = get_labeled_frames(self.landmark_labels).astype(int)
        self.cam_time_bases = get_cam_time_bases(self.landmark_frames, data["cam_fps"], data["cam_offsets"], self.opts)
        self.time_idx = 0
        self.cam_idx = 0

        self.init_ui()

        self.open_readers()
        # self.load_frame()

    def open_readers(self):
        print("Opening readers ... ", end="")
        readers = [get_reader(v.as_posix()) for v in self.data["videos"]]
        print("done")

    def init_ui(self):
        self.setWindowTitle('bboVisO checker')

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Matplotlib canvas
        self.components["main"]["figure"] = Figure()
        self.components["main"]["canvas"] = FigureCanvas(self.components["main"]["figure"])
        layout.addWidget(self.components["main"]["canvas"])

        ax = self.components["main"]["figure"].add_subplot(111)
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        ax.plot(t, s)

        # Slider
        self.components["controls"]["slider_t"] = QSlider()
        self.components["controls"]["slider_t"].setOrientation(1)  # Vertical orientation
        self.components["controls"]["slider_t"].setMinimum(0)  # Multiply by 10 to get actual values
        self.components["controls"]["slider_t"].setMaximum(len(self.times) - 1)  # Multiply by 10 to get actual values
        self.components["controls"]["slider_t"].setValue(0)  # Set default value to 3.1
        layout.addWidget(self.components["controls"]["slider_t"])

        # Text input
        self.components["controls"]["edit_t"] = QLineEdit()
        self.components["controls"]["edit_t"].setValidator(
            QtGui.QDoubleValidator())  # Allowing only floating point numbers
        self.components["controls"]["edit_t"].setText(str(self.times[self.time_idx]))
        layout.addWidget(self.components["controls"]["edit_t"])

        # Dropdown
        self.components["controls"]["dropdown_cams"] = QComboBox()
        self.components["controls"]["dropdown_cams"].addItems(["cam1", "cam2"])
        layout.addWidget(self.components["controls"]["dropdown_cams"])

        # Connecting slider and text input
        self.components["controls"]["slider_t"].sliderReleased.connect(self.update_values)
        self.components["controls"]["edit_t"].textChanged.connect(self.update_slider)

    def update_values(self):
        self.time_idx = self.components["controls"]["slider_t"].value()  # Divide by 10 to get actual values
        self.components["controls"]["edit_t"].setText(str(self.times[self.time_idx]))
        self.load_frame()

    def update_slider(self):
        value = float(self.components["controls"]["edit_t"].text())
        self.time_idx = np.argmin(np.abs(self.times - value))
        self.components["controls"]["slider_t"].setValue(self.time_idx)  # Multiply by 10 to set the slider value
        self.load_frame()

    def load_frame(self):
        time = self.times[self.time_idx]
        cam_mask = np.array([np.isin(time, ctb) for ctb in self.cam_time_bases])
        print(cam_mask)


def main(data, opts):
    app = QApplication(sys.argv)
    main_window = bboVisOchecker(data, opts)
    main_window.show()
    sys.exit(app.exec_())
