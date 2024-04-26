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

from bboVisO.opts import get_opts
from bboVisO.viso import objfun_repro_runner
from calibcamlib import Camerasystem

from bboVisO import helper

from bbo import label_lib

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
        #     "label_lib": data_tl["battery_cycle"]["landmark_labels"]["file"],
        #     "landmark_coordinates": data_tl["dataset"]["landmark_coordinates"],
        #     "time_range": data_tl_obj.get_time_range() / data_tl_obj.time_factor,
        #     "videos": data_tl_obj.get_headcam_video_files(),
        # }

        self.data = data
        self.opts = opts

        calibration = data["calibration"]
        self.calibs = np.load(calibration, allow_pickle=True)[()]["calibs"]
        self.camera_system = Camerasystem.from_calibs(self.calibs)

        path = Path(opts["text_out"]).expanduser().resolve() / data["name"]

        viso_maxerrors_pd = pd.read_csv(path / f"maxerror.csv", sep="\t")
        viso_maxerrors_pd.sort_values('t_us', ascending=True, inplace=True)
        self.viso_maxerrors = np.array(viso_maxerrors_pd['error']).T

        viso_errors_pd = pd.read_csv(path / f"error.csv", sep="\t")
        viso_errors_pd.sort_values('t_us', ascending=True, inplace=True)
        self.viso_errors = np.array(viso_errors_pd[viso_errors_pd.columns[1:]]).T

        viso_positions_pd = pd.read_csv(path / f"cam_positions.csv", sep="\t")
        self.viso_positions = np.array([
            viso_positions_pd["x0"],
            viso_positions_pd["y0"],
            viso_positions_pd["z0"],
        ]).T

        viso_orientations_pd = pd.read_csv(path / f"cam_orientations.csv", sep="\t")
        viso_orientations_q = np.array([
            viso_orientations_pd["x0"],
            viso_orientations_pd["y0"],
            viso_orientations_pd["z0"],
            viso_orientations_pd["w0"],
        ]).T
        self.viso_orientations = R.from_quat(viso_orientations_q)

        viso_landmarks_pd = pd.read_csv(path / f"landmarks.csv", sep="\t")
        self.landmark_labels = list(viso_landmarks_pd["landmark"])
        self.landmark_coords3d = np.array([
            viso_landmarks_pd["x0"],
            viso_landmarks_pd["y0"],
            viso_landmarks_pd["z0"],
        ]).T

        self.viso_times = np.array(viso_positions_pd["t_us"])

        self.check_mode = False

        low_error_mask = self.viso_maxerrors < self.opts["checker"]["interpolate_high_error"]
        self.viso_positions_interp, self.viso_orientations_interp = helper.interpolate_mask(
            self.viso_times,
            self.viso_positions,
            self.viso_orientations,
            low_error_mask
        )

        landmark_data = data["landmark_coordinates"]
        landmarks_identity = helper.landmarks_to_array(landmark_data, self.landmark_labels)[2]

        label_data = label_lib.load(data["labels"])
        self.landmark_frames = label_lib.get_labeled_frame_idxs(label_data, label_set=self.landmark_labels)
        # self.landmark_labels = sorted(label_lib.get_labels(label_data))
        # detected_labels = [d for d in detected_labels if d in self.landmark_labels]

        if "cam_time_bases" in data:
            self.cam_time_bases = [ct[self.landmark_frames] for ct in data["cam_time_bases"]]
        else:
            self.cam_time_bases = helper.get_cam_time_bases(self.landmark_frames, data["cam_fps"], data["cam_offsets"],
                                                            self.opts)

        self.landmark_imcoords, self.times = label_lib.to_numpy(label_data, self.landmark_frames, self.landmark_labels,
                                                                time_bases=self.cam_time_bases,
                                                                label_identity=landmarks_identity)

        self.landmark_imcoords = helper.interpolate_imcoords(self.landmark_imcoords, self.times, self.calibs,
                                                                           self.cam_time_bases)[0]
        # self.landmark_imcoords[1] = np.nan

        self.time_idx = 0
        self.cam_idx = 0
        self.cam_frame_idx = 0

        self.readers = []
        self.frame_dimensions = []
        self.open_readers()

        self.init_ui()

        self.set_time_idx(0)
        self.set_cam_idx(0)

    def set_time_idx(self, idx, update_frame=True, switch_cam=False):
        self.time_idx = idx
        time = self.times[self.time_idx]

        cam_mask = np.array([np.isin(time, ctb) for ctb in self.cam_time_bases])
        avail_cam_idxs = np.where(cam_mask)[0]
        if switch_cam and self.cam_idx not in avail_cam_idxs:
            self.set_cam_idx(avail_cam_idxs[0], update_time=False)

        # Set camera frame for time point
        cam_idxs = np.where(self.cam_time_bases[self.cam_idx] == self.times[self.time_idx])[0]
        if len(cam_idxs) > 0:
            self.cam_frame_idx = cam_idxs[0]
            landmark_frame = self.landmark_frames[self.cam_frame_idx]
        else:
            self.cam_frame_idx = -1
            landmark_frame = -1
        self.components["controls"]["edit_fr"].blockSignals(True)
        self.components["controls"]["edit_fr"].setText(str(landmark_frame))
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
        time = self.cam_time_bases[self.cam_idx][np.min((len(self.cam_time_bases[self.cam_idx]) - 1, idx))]
        time_idx = np.argmin(np.abs(self.times - time))
        self.set_time_idx(time_idx, update_frame)

    def set_time(self, time, update_frame=True):
        time_idx = np.argmin(np.abs(self.times - time))
        self.set_time_idx(time_idx, update_frame)

    def set_cam_idx(self, idx, update_time=True):
        self.cam_idx = idx
        #
        # time = self.times[self.time_idx]
        # # Set cam for time point
        # cam_mask = np.array([np.isin(time, ctb) for ctb in self.cam_time_bases])
        # avail_cam_idxs = np.where(cam_mask)[0]
        # if idx in avail_cam_idxs:
        #     self.cam_idx = idx
        # else:
        #     self.cam_idx = avail_cam_idxs[0]
        #
        # self.components["controls"]["dropdown_cams"].blockSignals(True)
        # self.components["controls"]["dropdown_cams"].clear()
        # self.components["controls"]["dropdown_cams"].addItems([str(i) for i in avail_cam_idxs])
        # self.components["controls"]["dropdown_cams"].setCurrentIndex(np.where(avail_cam_idxs == self.cam_idx)[0][0])
        # self.components["controls"]["dropdown_cams"].blockSignals(False)
        #
        if update_time:
            self.set_time_idx(self.time_idx)

    def open_readers(self):
        print("Opening readers ... ", end="")
        self.readers = [get_reader(v.as_posix(), backend="iio") for v in self.data["videos"]]
        self.frame_dimensions = [r.get_data(0).shape for r in self.readers]
        print("done")

    def init_ui(self):
        self.setWindowTitle('bboVisO checker')

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        vlayout = QVBoxLayout(central_widget)
        # Matplotlib canvas
        self.components["main"]["figure"] = Figure()
        self.components["main"]["canvas"] = FigureCanvas(self.components["main"]["figure"])

        ax = self.components["main"]["figure"].add_subplot(111)
        self.components["main"]["axis"] = ax
        self.components["main"]["figure"].subplots_adjust(left=0, right=1, top=1, bottom=0)
        vlayout.addWidget(self.components["main"]["canvas"])

        time_layout = QHBoxLayout()
        # Slider
        self.components["controls"]["slider_t"] = QSlider()
        self.components["controls"]["slider_t"].setOrientation(1)  # Vertical orientation
        self.components["controls"]["slider_t"].setMinimum(0)
        self.components["controls"]["slider_t"].setMaximum(len(self.times) - 1)
        time_layout.addWidget(self.components["controls"]["slider_t"])

        self.components["controls"]["button_t_prev"] = QPushButton("<")
        self.components["controls"]["button_t_prev"].setFixedWidth(20)
        self.components["controls"]["button_t_prev"].setFixedHeight(30)
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
        self.components["controls"]["dropdown_cams"].addItems([str(i) for i, _ in enumerate(self.cam_time_bases)])
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

        check_layout = QHBoxLayout()
        self.components["controls"]["button_check"] = QPushButton('Check mode', self)
        self.components["controls"]["button_check"].setFixedWidth(80)
        self.components["controls"]["button_check"].setCheckable(True)
        check_layout.addWidget(self.components["controls"]["button_check"])

        self.components["controls"]["button_snap"] = QPushButton('Snap SVG', self)
        self.components["controls"]["button_snap"].setFixedWidth(80)
        check_layout.addWidget(self.components["controls"]["button_snap"])

        vlayout.addLayout(check_layout)

        # Connecting slider and text input
        self.components["controls"]["slider_t"].sliderReleased.connect(self.on_slider_t_released)
        self.components["controls"]["edit_t"].returnPressed.connect(self.on_edit_t_change)
        self.components["controls"]["edit_t"].editingFinished.connect(self.on_edit_t_change)
        self.components["controls"]["edit_fr"].returnPressed.connect(self.on_edit_fr_change)
        self.components["controls"]["edit_fr"].editingFinished.connect(self.on_edit_fr_change)
        self.components["controls"]["dropdown_cams"].currentIndexChanged.connect(self.on_dropdown_change)
        self.components["controls"]["button_fr_prev"].clicked.connect(self.on_button_fr_prev_change)
        self.components["controls"]["button_fr_next"].clicked.connect(self.on_button_fr_next_change)
        self.components["controls"]["button_t_prev"].clicked.connect(self.on_button_t_prev_change)
        self.components["controls"]["button_t_next"].clicked.connect(self.on_button_t_next_change)
        self.components["controls"]["button_check"].clicked.connect(self.on_button_check_toggle)
        self.components["controls"]["button_snap"].clicked.connect(self.on_button_snap)

    def on_button_check_toggle(self):
        self.check_mode = self.components["controls"]["button_check"].isChecked()
        self.update_frame()

    def on_button_fr_prev_change(self):
        self.set_cam_frame_idx(np.max((0, self.cam_frame_idx - 1)))

    def on_button_fr_next_change(self):
        self.set_cam_frame_idx(np.min((len(self.landmark_frames), self.cam_frame_idx + 1)))

    def on_button_t_prev_change(self):
        self.set_time_idx(np.max((0, self.time_idx - 1)), switch_cam=True)

    def on_button_t_next_change(self):
        self.set_time_idx(np.min((len(self.times) - 1, self.time_idx + 1)), switch_cam=True)

    def on_button_snap(self):
        filename = Path('~') / f"bboViso_t{self.times[self.time_idx]}_c{self.cam_idx}.svg"
        self.components["main"]["figure"].savefig(filename.expanduser().resolve().as_posix())
        print(f"Canvas saved to {filename}")

    def on_slider_t_released(self):
        self.set_time_idx(self.components["controls"]["slider_t"].value())  # Divide by 10 to get actual values

    def on_edit_t_change(self):
        value = float(self.components["controls"]["edit_t"].text())
        self.set_time_idx(np.argmin(np.abs(self.times - value)))

    def on_edit_fr_change(self):
        idx = int(self.components["controls"]["edit_fr"].text())

        cam_time_idx = np.where(self.landmark_frames == idx)[0]
        if len(cam_time_idx) == 0:
            print("Frame not in viso")
            self.set_cam_frame_idx(self.cam_frame_idx)
            return

        time = self.cam_time_bases[self.cam_idx][cam_time_idx]

        self.set_time(time)

    def on_dropdown_change(self, index):
        selected_option = self.components["controls"]["dropdown_cams"].itemText(index)
        self.set_cam_idx(int(selected_option))

    def update_frame(self):
        cam_mask = np.array([np.isin(self.times[self.time_idx], ctb) for ctb in self.cam_time_bases])
        avail_cam_idxs = np.where(cam_mask)[0]
        if self.cam_idx in avail_cam_idxs:
            img = self.readers[self.cam_idx].get_data(self.landmark_frames[self.cam_frame_idx])
        else:
            img = np.zeros(self.frame_dimensions[self.cam_idx])

        imcoords = self.landmark_imcoords[self.cam_idx, self.time_idx]

        ax = self.components["main"]["axis"]
        ax.clear()
        ax.set_xlim((0, img.shape[1]))
        ax.set_ylim((0, img.shape[0]))
        ax.imshow(img)
        ax.plot(*imcoords.T, 'b+')

        viso_idx = np.where(self.viso_times == self.times[self.time_idx])[0]

        print("===== ", self.cam_idx, " == ", self.landmark_frames[self.cam_frame_idx], " =====")
        print("===== ", self.times[self.time_idx], " == ", viso_idx, " =====")

        for i, ic in enumerate(imcoords):
            if not np.any(np.isnan(ic)):
                print(f"{self.landmark_labels[i]}\t{ic[0]}\t{ic[1]}")

        if len(viso_idx > 0):
            viso_idx = viso_idx[0]

            if not self.check_mode:
                pos = self.viso_positions[viso_idx]
                ori_flu = self.viso_orientations[viso_idx]
                print("Original mode")
            else:
                pos = self.viso_positions_interp[viso_idx]
                ori_flu = self.viso_orientations_interp[viso_idx]
                print("Check mode")

            # pos = np.array([16.805, -79.253, -3.318])
            # ori_flu = R.from_quat([-0.467, 0.808, -0.290, 0.214])

            opts = get_opts()
            repro_error = objfun_repro_runner(
                self.landmark_coords3d, pos[np.newaxis], ori_flu.as_rotvec()[np.newaxis],
                self.landmark_imcoords[:, (self.time_idx,),],
                self.calibs, opts["iterations"][-1]
            )
            abserrors = np.sqrt(np.nansum(repro_error ** 2, axis=-1))
            abserrors[abserrors == 0] = np.nan

            print("Max error (odo):", self.viso_maxerrors[viso_idx])
            print("Max error:", np.nanmax(abserrors))
            print("Median error:", np.nanmedian(abserrors))
            print("Mean error:", np.nanmean(abserrors))
            print("Ssq error:", np.nansum(repro_error ** 2))
            print("Pos:", pos)
            print("Ori:", ori_flu.as_quat())
            print("Dir:", ori_flu.apply(np.array([0, 0, 1])))

            rel_points = ori_flu.inv().apply(self.landmark_coords3d - pos)

            self.camera_system = Camerasystem.from_calibs(self.calibs)
            imcoords_proj = self.camera_system.project(rel_points)[self.cam_idx]
            print("Cameralib: ", imcoords_proj)
            print("Labels: ", imcoords)

            imcoords_proj[np.isnan(imcoords)] = np.nan

            ax.plot(*imcoords_proj.T, 'bx')
            for ic, ic_p in zip(imcoords, imcoords_proj):
                ax.plot(*np.vstack((ic, ic_p)).T, "r--", zorder=0)

            for i, ic in enumerate(imcoords_proj):
                if not np.any(np.isnan(ic)):
                    print(f"{self.landmark_labels[i]}\t{ic[0]}\t{ic[1]}")
        else:
            print(self.times[self.time_idx], "not in viso")

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
