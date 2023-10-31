import sys
import matplotlib

from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget,
                             QSlider, QLabel, QLineEdit, QComboBox, QPushButton)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from svidreader import get_reader
from calibcamlib import Camerasystem

from bboVisO import acm_labels
from bboVisO import helper

matplotlib.use('Qt5Agg')


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

        calibration = data["calibration"]
        calibs = np.load(calibration, allow_pickle=True)[()]["calibs"]
        self.camera_system = Camerasystem.from_calibs(calibs)

        path = Path(opts["text_out"]).expanduser().resolve() / data["name"]

        cam_positions_pd = pd.read_csv(path / f"{data['name']}_cam_positions.csv",
                                       sep="\t")
        self.viso_positions = np.array([
            cam_positions_pd["x0"],
            cam_positions_pd["y0"],
            cam_positions_pd["z0"],
        ]).T
        viso_orientations_pd = pd.read_csv(path / f"{data['name']}_cam_orientations.csv", sep="\t")
        viso_orientations_q = np.array([
            viso_orientations_pd["x0"],
            viso_orientations_pd["y0"],
            viso_orientations_pd["z0"],
            viso_orientations_pd["w0"],
        ]).T
        self.viso_orientations = R.from_quat(viso_orientations_q)
        self.viso_times = np.array(cam_positions_pd["t_us"])

        label_data = acm_labels.load(data["acm_labels"])["labels"]
        self.landmark_frames = acm_labels.get_labeled_frames(label_data).astype(int)
        self.landmark_labels = sorted(acm_labels.get_labels(label_data))

        self.cam_time_bases = helper.get_cam_time_bases(self.landmark_frames, data["cam_fps"], data["cam_offsets"],
                                                        self.opts)

        landmark_data = data["landmark_coordinates"]
        origin = np.array(landmark_data["origin"]["position"])
        self.landmark_coords3d, landmarks_far, landmarks_identity, landmarks_optimize, landmarks_use = helper.landmarks_to_array(
            landmark_data, self.landmark_labels)
        self.landmark_coords3d -= origin

        self.landmark_imcoords, self.times = acm_labels.to_array(label_data, self.landmark_frames, self.landmark_labels,
                                                                 time_bases=self.cam_time_bases,
                                                                 landmarks_identity=landmarks_identity)

        self.time_idx = 0
        self.cam_idx = 0
        self.cam_frame_idx = 0

        self.readers = []
        self.open_readers()

        self.init_ui()

        self.set_time_idx(0)
        self.set_cam_idx(0)

    def set_time_idx(self, idx, update_frame=True):
        self.time_idx = idx
        time = self.times[self.time_idx]

        # Set cam for time point
        self.set_cam_idx(self.cam_idx, update_frame=False)

        # Set camera frame for time point
        self.cam_frame_idx = np.where(self.cam_time_bases[self.cam_idx] == self.times[self.time_idx])[0][0]
        self.components["controls"]["edit_fr"].blockSignals(True)
        self.components["controls"]["edit_fr"].setText(str(self.landmark_frames[self.cam_frame_idx]))
        self.components["controls"]["edit_fr"].blockSignals(False)

        # Set slider
        self.components["controls"]["slider_t"].blockSignals(True)
        self.components["controls"]["slider_t"].setValue(idx)
        self.components["controls"]["slider_t"].blockSignals(False)

        # Set slider
        self.components["controls"]["edit_t"].blockSignals(True)
        self.components["controls"]["edit_t"].setText(str(time))
        self.components["controls"]["edit_t"].blockSignals(False)

        if update_frame:
            self.update_frame()

    def set_cam_frame_idx(self, idx, update_frame=True):
        time = self.cam_time_bases[self.cam_idx][np.min((len(self.cam_time_bases[self.cam_idx])-1, idx))]
        time_idx = np.argmin(np.abs(self.times - time))
        self.set_time_idx(time_idx, update_frame)

    def set_time(self, time, update_frame=True):
        time_idx = np.argmin(np.abs(self.times - time))
        self.set_time_idx(time_idx, update_frame)

    def set_cam_idx(self, idx, update_frame=True):
        self.cam_idx = idx

        time = self.times[self.time_idx]
        # Set cam for time point
        cam_mask = np.array([np.isin(time, ctb) for ctb in self.cam_time_bases])
        avail_cam_idxs = np.where(cam_mask)[0]
        if self.cam_idx not in avail_cam_idxs:
            self.cam_idx = avail_cam_idxs[0]
        self.components["controls"]["dropdown_cams"].blockSignals(True)
        self.components["controls"]["dropdown_cams"].clear()
        self.components["controls"]["dropdown_cams"].addItems([str(i) for i in avail_cam_idxs])
        self.components["controls"]["dropdown_cams"].setCurrentIndex(np.where(avail_cam_idxs == self.cam_idx)[0][0])
        self.components["controls"]["dropdown_cams"].blockSignals(False)

        if update_frame:
            self.update_frame()

    def open_readers(self):
        print("Opening readers ... ", end="")
        self.readers = [get_reader(v.as_posix(), backend="iio") for v in self.data["videos"]]
        print("done")

    def init_ui(self):
        self.setWindowTitle('bboVisO checker')

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        vlayout = QVBoxLayout(central_widget)

        # Matplotlib canvas
        self.components["main"]["figure"] = Figure()
        self.components["main"]["canvas"] = FigureCanvas(self.components["main"]["figure"])
        vlayout.addWidget(self.components["main"]["canvas"])

        ax = self.components["main"]["figure"].add_subplot(111)
        self.components["main"]["axis"] = ax
        self.components["main"]["figure"].subplots_adjust(left=0, right=1, top=1, bottom=0)

        time_layout = QHBoxLayout()
        # Slider
        self.components["controls"]["slider_t"] = QSlider()
        self.components["controls"]["slider_t"].setOrientation(1)  # Vertical orientation
        self.components["controls"]["slider_t"].setMinimum(0)
        self.components["controls"]["slider_t"].setMaximum(len(self.times) - 1)
        time_layout.addWidget(self.components["controls"]["slider_t"])

        self.components["controls"]["button_t_prev"] = QPushButton("<")
        self.components["controls"]["button_t_prev"].setFixedWidth(20)
        time_layout.addWidget(self.components["controls"]["button_t_prev"])
        self.components["controls"]["button_t_next"] = QPushButton(">")
        self.components["controls"]["button_t_next"].setFixedWidth(20)
        time_layout.addWidget(self.components["controls"]["button_t_next"])

        # Text input
        self.components["controls"]["edit_t"] = QLineEdit()
        self.components["controls"]["edit_t"].setValidator(
            QtGui.QDoubleValidator())  # Allowing only floating point numbers
        self.components["controls"]["edit_t"].setFixedWidth(80)
        time_layout.addWidget(self.components["controls"]["edit_t"])
        vlayout.addLayout(time_layout)

        cam_layout = QHBoxLayout()
        # Dropdown
        self.components["controls"]["dropdown_cams"] = QComboBox()
        cam_layout.addWidget(self.components["controls"]["dropdown_cams"])

        self.components["controls"]["button_fr_prev"] = QPushButton("<")
        self.components["controls"]["button_fr_prev"].setFixedWidth(20)
        cam_layout.addWidget(self.components["controls"]["button_fr_prev"])
        self.components["controls"]["button_fr_next"] = QPushButton(">")
        self.components["controls"]["button_fr_next"].setFixedWidth(20)
        cam_layout.addWidget(self.components["controls"]["button_fr_next"])

        # Text input
        self.components["controls"]["edit_fr"] = QLineEdit()
        self.components["controls"]["edit_fr"].setValidator(
            QtGui.QDoubleValidator())  # Allowing only floating point numbers
        self.components["controls"]["edit_fr"].setFixedWidth(80)
        cam_layout.addWidget(self.components["controls"]["edit_fr"])
        vlayout.addLayout(cam_layout)

        # Connecting slider and text input
        self.components["controls"]["slider_t"].sliderReleased.connect(self.on_slider_t_released)
        self.components["controls"]["edit_t"].returnPressed.connect(self.on_edit_t_change)
        self.components["controls"]["edit_t"].editingFinished.connect(self.on_edit_t_change)
        self.components["controls"]["dropdown_cams"].currentIndexChanged.connect(self.on_dropdown_change)
        self.components["controls"]["button_fr_prev"].clicked.connect(self.on_button_fr_prev_change)
        self.components["controls"]["button_fr_next"].clicked.connect(self.on_button_fr_next_change)
        self.components["controls"]["button_t_prev"].clicked.connect(self.on_button_t_prev_change)
        self.components["controls"]["button_t_next"].clicked.connect(self.on_button_t_next_change)

    def on_button_fr_prev_change(self):
        self.set_cam_frame_idx(np.max((0, self.cam_frame_idx - 1)))

    def on_button_fr_next_change(self):
        self.set_cam_frame_idx(np.min((len(self.landmark_frames), self.cam_frame_idx + 1)))

    def on_button_t_prev_change(self):
        self.set_time_idx(np.max((0, self.time_idx - 1)))

    def on_button_t_next_change(self):
        self.set_time_idx(np.min((len(self.times) - 1, self.time_idx + 1)))

    def on_slider_t_released(self):
        self.set_time_idx(self.components["controls"]["slider_t"].value())  # Divide by 10 to get actual values

    def on_edit_t_change(self):
        value = float(self.components["controls"]["edit_t"].text())
        self.set_time_idx(np.argmin(np.abs(self.times - value)))

    def on_dropdown_change(self, index):
        selected_option = self.components["controls"]["dropdown_cams"].itemText(index)
        self.set_cam_idx(int(selected_option))

    def update_frame(self):
        img = self.readers[self.cam_idx].get_data(self.landmark_frames[self.cam_frame_idx])
        imcoords = self.landmark_imcoords[self.cam_idx, self.time_idx]

        print("===== ", self.cam_idx, " == ", self.landmark_frames[self.cam_frame_idx], " =====")
        for i, ic in enumerate(imcoords):
            if not np.any(np.isnan(ic)):
                print(f"{self.landmark_labels[i]}\t{ic[0]}\t{ic[1]}")

        ax = self.components["main"]["axis"]
        ax.clear()
        ax.imshow(img)
        ax.plot(*imcoords.T, 'b+')

        viso_idx = np.where(self.viso_times == self.times[self.time_idx])[0]
        if len(viso_idx > 0):
            viso_idx = viso_idx[0]

            pos = self.viso_positions[viso_idx]
            ori_flu = self.viso_orientations[viso_idx]

            dirs = ori_flu.inv().apply(self.landmark_coords3d - pos)

            imcoords_proj = self.camera_system.project(dirs)[self.cam_idx]
            imcoords_proj[np.isnan(imcoords)] = np.nan

            ax.plot(*imcoords_proj.T, 'bx')
            for ic, ic_p in zip(imcoords, imcoords_proj):
                ax.plot(*np.vstack((ic, ic_p)).T, "r--", zorder=0)

        ax.set_xlim((0, img.shape[1]))
        ax.set_ylim((0, img.shape[0]))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axis('off')
        ax.invert_yaxis()
        self.components["main"]["canvas"].draw()


def main(data, opts):
    app = QApplication(sys.argv)
    main_window = bboVisOchecker(data, opts)
    main_window.show()
    sys.exit(app.exec_())
