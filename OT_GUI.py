# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:50:13 2022

@author: marti
"""
import sys
import cv2 # Certain versions of this won't work

from PyQt6.QtWidgets import (
    QMainWindow, QApplication,
    QLabel, QCheckBox, QComboBox, QListWidget, QLineEdit, QSpinBox,
    QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QFileDialog, 
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QRunnable, QObject, QPoint, QRect, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QBrush, QColor, QAction, QDoubleValidator, QPen

from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from random import randint

import BaslerCameras
import ThorlabsCameras
from CameraControlsNew import CameraThread, VideoWriterThread
from ControlParameters import default_c_p, get_data_dicitonary_new
from TemperatureControllerTED4015 import TemperatureThread
from TemperatureControllerWidget import TempereatureControllerWindow
from ReadPicUart import PicReader
from LivePlots import PlotWindow
from SaveDataWidget import SaveDataWindow
import numpy as np
from time import sleep
from functools import partial

from PIStage import PIStageThread
from PIStageWidget import PIStageWidget

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QThread):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    Used to update the screen continoulsy with the images of the camera
    '''
    changePixmap = pyqtSignal(QImage)

    def __init__(self, c_p, data, test_mode=True, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.c_p = c_p
        self.data_channels = data
        self.args = args
        self.kwargs = kwargs
        self.test_mode = test_mode
        self.signals = WorkerSignals()

    def testDataUpdate(self, max_length=10_000):
        # Fill data dicitonary with fake data to test the interface.
        # Used only for testing
        """
        # Old version
        if len(self.data_channels['Time'].data) < max_length:
            self.dt = 1000/max_length
            self.data_channels['Time'].data = np.linspace(0, 1000, num=max_length)
            self.data_channels['Y-force'].data = np.sin(self.data_channels['Time'].data / 10)
            self.data_channels['X-force'].data = np.cos(self.data_channels['Time'].data / 10)
            self.data_channels['Z-force'].data = np.cos(self.data_channels['Time'].data / 10)**2
            self.data_channels['X-position'].data = np.random.rand(max_length) * 2 - 1
            self.data_channels['Y-position'].data = np.random.rand(max_length) * 2 - 1
            self.data_channels['Z-position'].data = np.random.rand(max_length) * 2 - 1
            self.data_channels['Motor_position'].data = np.sin(self.data_channels['Time'].data / 10) + np.random.rand(max_length)
        else:
            # Shift the data
            self.data_channels['Time'].data[:-1] = self.data_channels['Time'].data[1:]
            self.data_channels['Y-force'].data[:-1] = self.data_channels['Y-force'].data[1:]
            self.data_channels['X-force'].data[:-1] = self.data_channels['X-force'].data[1:]
            self.data_channels['Z-force'].data[:-1] = self.data_channels['Z-force'].data[1:]
            
            self.data_channels['X-position'].data[:-1] = self.data_channels['X-position'].data[1:]
            self.data_channels['Y-position'].data[:-1] = self.data_channels['Y-position'].data[1:]
            self.data_channels['Z-position'].data[:-1] = self.data_channels['Z-position'].data[1:]
            self.data_channels['Motor_position'].data[:-1] = self.data_channels['Motor_position'].data[1:]
            
            # Update last element
            self.data_channels['Time'].data[-1] = self.data_channels['Time'].data[-1] + self.dt

            self.data_channels['Y-force'].data[-1] = np.sin(self.data_channels['Time'].data[-1] / 10)
            self.data_channels['X-force'].data[-1] = np.cos(self.data_channels['Time'].data[-1] / 10)
            self.data_channels['Z-force'].data[-1] = np.cos(self.data_channels['Time'].data[-1] / 10)**2

            self.data_channels['X-position'].data[-1] = np.random.rand() * 2 - 1
            self.data_channels['Y-position'].data[-1] = np.random.rand() * 2 - 1
            self.data_channels['Z-position'].data[-1] = np.random.rand() * 2 - 1
            self.data_channels['Motor_position'].data[-1] = np.sin(self.data_channels['Time'].data[-1] / 10) + np.random.rand()
        """
        self.dt = 1000/max_length
        
        if len(self.data_channels['Time'].data) < max_length:
            self.data_channels['Time'].put_data(np.linspace(0, 1000, num=max_length))
            self.data_channels['Y-force'].put_data(np.sin(self.data_channels['Time'].data / 10))
            self.data_channels['X-force'].put_data(np.cos(self.data_channels['Time'].data / 10))
            self.data_channels['Z-force'].put_data(np.cos(self.data_channels['Time'].data / 10)**2)
            self.data_channels['X-position'].put_data(np.random.rand(max_length) * 2 - 1)
            self.data_channels['Y-position'].put_data(np.random.rand(max_length) * 2 - 1)
            self.data_channels['Z-position'].put_data(np.random.rand(max_length) * 2 - 1)
            self.data_channels['Motor_position'].put_data(np.sin(self.data_channels['Time'].data / 10))
        else:
            # Shift the data
            # Update last element
            self.data_channels['Time'].put_data(self.data_channels['Time'].get_data(1) + self.dt)

            self.data_channels['Y-force'].put_data(np.sin(self.data_channels['Time'].get_data(1) / 10))
            self.data_channels['X-force'].put_data(np.cos(self.data_channels['Time'].get_data(1) / 10))
            self.data_channels['Z-force'].put_data(np.cos(self.data_channels['Time'].get_data(1) / 10)**2)

            self.data_channels['X-position'].put_data(np.random.rand() * 2 - 1)
            self.data_channels['Y-position'].put_data(np.random.rand() * 2 - 1)
            self.data_channels['Z-position'].put_data(np.random.rand() * 2 - 1)
            self.data_channels['Motor_position'].put_data((self.data_channels['Time'].get_data(1) / 10) + np.random.rand())
    def draw_particle_positions(self, centers):
        # TODO add function also for crosshair to help with alignment.
        for x, y in zip(centers):
            #self.brush.
            pass

    def draw_central_circle(self):
        self.blue_pen.setColor(QColor('blue'))
        # self.qp.setPen(QPen(QColor('blue')))
        cx = int((self.c_p['camera_width']/2 - self.c_p['AOI'][0])/self.c_p['image_scale'])
        cy = int((self.c_p['camera_height']/2 - self.c_p['AOI'][2])/self.c_p['image_scale'])
        # TODO find out why there seem to be a small offset in the position of this
        # and fix so that the scale changes when zooming in.
        rx=50
        ry=50
        self.qp.drawEllipse(cx-int(rx/2)-1, cy-int(ry/2)-1, rx, ry)

    def run(self):

        self.blue_pen = QPen()
        self.blue_pen.setColor(QColor('blue'))
        self.blue_pen.setWidth(2)
        self.red_pen = QPen()
        self.red_pen.setColor(QColor('red'))
        self.red_pen.setWidth(2)

        while True:
            if self.test_mode:
                self.testDataUpdate()

            self.image = self.c_p['image']
            # self.image[int(self.image.shape[0]/2), int(self.image.shape[1]/2),:] = [250,250,250]
            W, H = self.c_p['frame_size']
            self.c_p['image_scale'] = max(self.image.shape[1]/W, self.image.shape[0]/H)
            # It is quite sensitive to the format here, won't accept any missmatch
            if len(np.shape(self.image)) < 3:
                convertToQtFormat = QImage(self.image, self.image.shape[1],
                                       self.image.shape[0],
                                       QImage.Format.Format_Grayscale8)
            else:
                # TODO have the image always generated as a rgb image.
                convertToQtFormat = QImage(self.image, self.image.shape[1],
                                       self.image.shape[0],
                                       QImage.Format.Format_RGB888)
                # Different format needed for color image, needs fixing.
            
            picture = convertToQtFormat.scaled(
                W,H,
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            # Give other things time to work, roughly 50 fps default.
            sleep(0.02) # Sets the FPS
            
            # Paint extra items on the screen
            self.qp = QPainter(picture)
            # Draw zoom in rectangle
            if self.c_p['mouse_params'][0]:
                # TODO use mouse params [0] to index different tools.
                # self.qp.setBrush(QColor(255, 255, 0, 20))#self.br)
                self.qp.setPen(self.red_pen)                
                x1,y1,x2,y2 = self.c_p['mouse_params'][1:5]
                self.qp.drawRect(x1,y1,x2-x1,y2-y1)
            self.qp.setPen(self.blue_pen)
            self.draw_central_circle()
            self.qp.end()
            self.changePixmap.emit(picture)


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Optical tweezers: Main window")
        self.c_p = default_c_p()
        self.data_channels = get_data_dicitonary_new()
        # Start camera threads
        self.CameraThread = None
        try:
            # camera = BaslerCameras.BaslerCamera()
            # TODO fix error of program not quitting when trying to connect to basler
            # camera if there is no camera connected.
            # camera = ThorlabsCameras.ThorlabsCamera()
            camera = None
            if camera is not None:
                self.CameraThread = CameraThread(self.c_p, camera)
                self.CameraThread.start()
        except Exception as E:
            print(f"Camera error!\n{E}")
        self.TemperatureThread = None
        """
        try:
            self.TemperatureThread = TemperatureThread(1,'Temperature Thread',
                                                       self.c_p, self.data_channels)
            #(self, threadID, name, c_p, temperature_controller=None, max_diff=0.05)
            self.TemperatureThread.start()
        except Exception as E:
            print(E)
        
        try:
            self.PiezoThread = PIStageThread(3, "PI piezo thread", self.c_p)
            self.PiezoThread.start()
        except Exception as E:
            print(E)
        """
        try:
            self.PICReaderT = PicReader(self.c_p, self.data_channels)
            self.PICReaderT.start()
        except Exception as E:
            print(E)

        self.VideoWriterThread = VideoWriterThread(2, 'video thread', self.c_p)
        self.VideoWriterThread.start()

        self.plot_windows = None

        # Set up camera window
        H = int(1080/4)
        W = int(1920/4)
        sleep(0.5)
        self.c_p['frame_size'] = int(self.c_p['camera_width']/2), int(self.c_p['camera_height']/2)
        self.label = QLabel("Hello")
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setCentralWidget(self.label)
        self.label.setMinimumSize(W,H)
        self.painter = QPainter(self.label.pixmap())
        th = Worker(c_p=self.c_p, data=self.data_channels)
        th.changePixmap.connect(self.setImage)
        th.start()

        # Create toolbar
        self.create_toolbar()
        # Create menu
        self.create_filemenu()
        self.show()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def start_threads(self):
        pass
        
    def create_toolbar(self):
        toolbar = QToolBar("Main tools")
        self.addToolBar(toolbar)
        self.add_default_actions(toolbar) # TODO create toolbar in a neater way

        self.exposure_time_LineEdit = QLineEdit()
        self.exposure_time_LineEdit.setValidator(QDoubleValidator(0.99,99.99,2))
        self.exposure_time_LineEdit.setText(str(self.c_p['exposure_time']))
        toolbar.addWidget(self.exposure_time_LineEdit)
        toolbar.addAction(self.set_exp_tim)

    def create_filemenu(self):
        self.menu = self.menuBar()

        file_menu = self.menu.addMenu("File")
        self.menu.addAction(self.open_plot_window)
        file_menu.addSeparator()

        # Create submenu for setting recording(video) format
        format_submenu = file_menu.addMenu("Recording format")
        video_formats = ['avi','mp4','npy']
        for f in video_formats :

            format_command= partial(self.set_video_format, f)
            format_action = QAction(f, self)
            format_action.setStatusTip(f"Set recording format to {f}")
            format_action.triggered.connect(format_command)
            format_submenu.addAction(format_action)

        # Submenu for setting the image format
        image_format_submenu = file_menu.addMenu("Image format")
        image_formats = ['png','jpg','npy']
        for f in image_formats:

            format_command= partial(self.set_image_format, f)
            format_action = QAction(f, self)
            format_action.setStatusTip(f"Set recording format to {f}")
            format_action.triggered.connect(format_command)
            image_format_submenu.addAction(format_action)

        # Add command to set the savepath of the experiments.
        set_save_action = QAction("Set save path", self)
        set_save_action.setStatusTip("Set save path")
        set_save_action.triggered.connect(self.set_save_path)
        file_menu.addAction(set_save_action)

    def add_default_actions(self, toolbar):
        # TODO add action config function so one can easily choose which functions to include
        self.zoom_action = QAction("Zoom out", self)
        self.zoom_action.setToolTip("Resets the field of view of the camera.")
        self.zoom_action.triggered.connect(self.ZoomOut)
        self.zoom_action.setCheckable(False)

        self.record_action = QAction("Record video", self)
        self.record_action.setToolTip("Turn ON recording.")
        self.record_action.setShortcut('Ctrl+R')
        self.record_action.triggered.connect(self.ToggleRecording)
        self.record_action.setCheckable(True)

        self.snapshot_action = QAction("Snapshot", self)
        self.snapshot_action.setToolTip("Take snapshot of camera view.")
        self.snapshot_action.setShortcut('Shift+S')
        self.snapshot_action.triggered.connect(self.snapshot)
        self.snapshot_action.setCheckable(False)

        self.open_plot_window = QAction("Open plotter", self)
        self.open_plot_window.setToolTip("Open live plotting window.")
        self.open_plot_window.triggered.connect(self.show_new_window)
        self.open_plot_window.setCheckable(False)

        """
        self.open_temperature = QAction("Temperature control", self)
        self.open_temperature.setToolTip("Open temperature control window.")
        self.open_temperature.triggered.connect(self.OpenTemperatureWindow)
        self.open_temperature.setCheckable(False)

        self.open_PI_stage = QAction("PI stage", self)
        self.open_PI_stage.setToolTip("Open Pi stage controls window.")
        self.open_PI_stage.triggered.connect(self.OpenPIStage)
        self.open_PI_stage.setCheckable(False)
        
        self.open_data_window = QAction("Data window", self)
        self.open_data_window.setToolTip("Open data control window")
        self.open_data_window.triggered.connect(self.DataWindow)
        self.open_data_window.setCheckable(False)
        """
        self.set_exp_tim = QAction("Set exposure time", self)
        self.set_exp_tim.setToolTip("Sets exposure time to the value in the textboox")
        self.set_exp_tim.triggered.connect(self.set_exposure_time)
        self.open_plot_window.setCheckable(False)

        toolbar.addAction(self.zoom_action)
        toolbar.addAction(self.record_action)
        toolbar.addAction(self.snapshot_action)
        # toolbar.addAction(self.open_temperature)
        # toolbar.addAction(self.open_data_window)

    def set_video_format(self, video_format):
        self.c_p['video_format'] = video_format

    def set_image_format(self, image_format):
        self.c_p['image_format'] = image_format
        
    def set_video_name(self, string):
        self.c_p['video_name'] = string

    def set_exposure_time(self):
        # Updates the exposure time of the camera to what is inside the textbox
        self.c_p['exposure_time'] = float(self.exposure_time_LineEdit.text())
        self.c_p['new_settings_camera'] = [True, 'exposure_time']

    def set_save_path(self):
        fname = QFileDialog.getExistingDirectory(self,"Save path")
        if len(fname) > 3:
            self.c_p['recording_path'] = fname

    def ZoomOut(self):
        self.c_p['AOI'] = [0, self.c_p['camera_width'], 0,
                   self.c_p['camera_height']]
        self.c_p['new_settings_camera'] = [True, 'AOI']

    def ToggleRecording(self):
        # Turns on/off recording
        # Need to add somehting to indicate the number of frames left to save when recording.
        self.c_p['recording'] = not self.c_p['recording']
        if self.c_p['recording']:
            self.record_action.setToolTip("Turn OFF recording.")
        else:
            self.record_action.setToolTip("Turn ON recording.")

    def snapshot(self):
        # Captures a snapshot of what the camera is viewing and saves that
        # in the fileformat specified by the image_format parameter.
        idx = str(self.c_p['image_idx'])
        filename = self.c_p['recording_path'] + '/image_' + idx +'.'+\
            self.c_p['image_format']
        if self.c_p['image_format'] == 'npy':
            np.save(filename[:-4], self.c_p['image'])
        else:
            cv2.imwrite(filename, cv2.cvtColor(self.c_p['image'],
                                           cv2.COLOR_RGB2BGR))

        self.c_p['image_idx'] += 1

    def resizeEvent(self, event):
        super().resizeEvent(event)
        H = event.size().height()
        W = event.size().width()
        self.c_p['frame_size'] = W, H

    def mouseMoveEvent(self, e):
        self.c_p['mouse_params'][3] = e.pos().x()-self.label.pos().x()
        self.c_p['mouse_params'][4] = e.pos().y()-self.label.pos().y()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton or \
            e.button() == Qt.MouseButton.RightButton:
            self.c_p['mouse_params'][0] = 0
            return
        if e.button() == Qt.MouseButton.LeftButton and not self.c_p['mouse_params'][0]:
            # handle the left-button press in here            if
            self.c_p['mouse_params'][0] = 1
            self.c_p['mouse_params'][1] = e.pos().x()-self.label.pos().x()
            self.c_p['mouse_params'][2] = e.pos().y()-self.label.pos().y()

    def mouseReleaseEvent(self, e):

        if e.button() == Qt.MouseButton.MiddleButton or \
            e.button() == Qt.MouseButton.RightButton or \
            not self.c_p['mouse_params'][0]:
            return
        self.c_p['mouse_params'][0] = 0
        self.c_p['mouse_params'][3] = e.pos().x()-self.label.pos().x()
        self.c_p['mouse_params'][4] = e.pos().y()-self.label.pos().y()
        x0, y0, x1, y1 = self.c_p['mouse_params'][1:5]
        dx = x1 - x0
        dy = y1 - y0
        if dx**2 < 100 or dy**2 < 100:
            print(dx,dy)
            return
        self.image_scale = self.c_p['image_scale']
        left = int(x0 * self.image_scale)
        right = int(x1 *self.image_scale)
        if right < left:
            tmp = right
            right = left
            left = tmp
        up = int(y0 * self.image_scale)
        down = int(y1 * self.image_scale)
        if up < down:
            tmp = up
            up = down
            down = tmp

        self.c_p['AOI'] = [self.c_p['AOI'][0] + left,self.c_p['AOI'][0] + right,
                           self.c_p['AOI'][2] + down,self.c_p['AOI'][2] + up]
        print("Udpating settings")
        self.c_p['new_settings_camera'] = [True, 'AOI']

    def mouseDoubleClickEvent(self, e):
        # Double click to move center?
        x = e.pos().x()-self.label.pos().x()
        y = e.pos().y()-self.label.pos().y()
        print(x*self.c_p['image_scale'] ,y*self.c_p['image_scale'] )

    def show_new_window(self, checked):
        if self.plot_windows is None:
            self.plot_windows = []
        self.plot_windows.append(PlotWindow(self.c_p, data=self.data_channels, #data=get_data_dicitonary(), # Major change here
                                          x_keys=['Time','Time'], y_keys=['X-force','Y-position']))

        self.plot_windows[-1].show()

    def OpenTemperatureWindow(self):
        self.temp_control_window = TempereatureControllerWindow(self.c_p)
        self.temp_control_window.show()
        
    def OpenPIStage(self):
        self.PI_window = PIStageWidget(self.c_p)
        self.PI_window.show()

    def DataWindow(self):
        self.data_window= SaveDataWindow(self.c_p, self.data_channels)
        self.data_window.show()

    def closeEvent(self, event):
        if self.plot_windows is not None:
            for w in self.plot_windows:
                w.close()


    def __del__(self):
        self.c_p['program_running'] = False
        # TODO organize this better
        if self.CameraThread is not None:
            self.CameraThread.join()
        if self.TemperatureThread is not None:
            self.TemperatureThread.join()
        if self.self.PICReaderT is not None:
            self.PICReaderT.join()

        self.VideoWriterThread.join()

app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
w.c_p['program_running'] = False
# w.CameraThread.join()
