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
from PyQt6.QtGui import QPixmap, QImage, QPainter, QBrush, QColor, QAction, QDoubleValidator, QPen, QIntValidator

from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from random import randint

import BaslerCameras
import ThorlabsCameras
from CameraControlsNew import CameraThread, VideoWriterThread
from ControlParameters import default_c_p, get_data_dicitonary_new
from TemperatureControllerTED4015 import TemperatureThread
from TemperatureControllerWidget import TempereatureControllerWindow
from ReadPicUart import PicReader, PicWriter
from LivePlots import PlotWindow
from SaveDataWidget import SaveDataWindow
import numpy as np
from time import sleep
from functools import partial
from PIStage import PIStageThread
from PIStageWidget import PIStageWidget
import MotorControlWidget

#class WorkerSignals(QObject):
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
 #   finished = pyqtSignal()
  #  error = pyqtSignal(tuple)
   # result = pyqtSignal(object)

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
        # self.signals = WorkerSignals()

    def testDataUpdate(self, max_length=10_000):
        # Fill data dicitonary with fake data to test the interface.
        # Used only for testing
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
    
    def preprocess_image(self):

        # Check if offset and gain should be applied.
        if self.c_p['image_offset'] != 0:
            self.image += int(self.c_p['image_offset'])
            
        if self.c_p['image_gain'] != 1:
            # TODO unacceptably slow
            self.image = (self.image*self.c_p['image_gain'])

        self.image = np.uint8(self.image)
        s = np.shape(self.image)
        if len(s) < 3 or s[2] == 1:
            new_image = np.uint8(np.zeros([s[0], s[1], 3]))
            # TODO check if we can do this more efficiently
            new_image [:,:,0] = self.image
            new_image [:,:,1] = self.image
            new_image [:,:,2] = self.image
            self.image = new_image

    def draw_central_circle(self):
        self.blue_pen.setColor(QColor('blue'))
        # self.qp.setPen(QPen(QColor('blue')))
        cx = int((self.c_p['camera_width']/2 - self.c_p['AOI'][0])/self.c_p['image_scale'])
        cy = int((self.c_p['camera_height']/2 - self.c_p['AOI'][2])/self.c_p['image_scale'])
        rx=50
        ry=50
        self.qp.drawEllipse(cx-int(rx/2)-1, cy-int(ry/2)-1, rx, ry)

    def run(self):

        # Initialize pens to draw on the images
        self.blue_pen = QPen()
        self.blue_pen.setColor(QColor('blue'))
        self.blue_pen.setWidth(2)
        self.red_pen = QPen()
        self.red_pen.setColor(QColor('red'))
        self.red_pen.setWidth(2)

        while True:
            if self.test_mode:
                self.testDataUpdate()

            self.image = np.array(self.c_p['image'])
            self.preprocess_image()
            W, H = self.c_p['frame_size']
            self.c_p['image_scale'] = max(self.image.shape[1]/W, self.image.shape[0]/H)
            # It is quite sensitive to the format here, won't accept any missmatch
            """
            if len(np.shape(self.image)) < 3:
                convertToQtFormat = QImage(self.image, self.image.shape[1],
                                       self.image.shape[0],
                                       QImage.Format.Format_Grayscale8)
            else:
                # TODO have the image always generated as a rgb image.
                """
            convertToQtFormat = QImage(self.image, self.image.shape[1],
                                   self.image.shape[0],
                                   QImage.Format.Format_RGB888)
            
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
            camera = None
            camera = BaslerCameras.BaslerCamera()
            # TODO fix error of program not quitting when trying to connect to basler
            # camera if there is no camera connected.
            # camera = ThorlabsCameras.ThorlabsCamera()
            
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
            sleep(0.1)
            self.PICWriterT = PicWriter(self.c_p, self.PICReaderT.serial_channel)
            self.PICWriterT.start()
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
        #self.create_camera_toolbar()
        create_camera_toolbar_external(self)
        # Create menu
        self.create_filemenu()
        self.show()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def start_threads(self):
        pass
        
    def create_camera_toolbar(self):
        self.camera_toolbar = QToolBar("Camera tools")
        self.addToolBar(self.camera_toolbar)
        # self.add_camera_actions(self.camera_toolbar)
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

        self.set_exp_tim = QAction("Set exposure time", self)
        self.set_exp_tim.setToolTip("Sets exposure time to the value in the textboox")
        self.set_exp_tim.triggered.connect(self.set_exposure_time)
        self.set_exp_tim.setCheckable(False)
        
        self.camera_toolbar.addAction(self.zoom_action)
        self.camera_toolbar.addAction(self.record_action)
        self.camera_toolbar.addAction(self.snapshot_action)
        
        self.exposure_time_LineEdit = QLineEdit()
        self.exposure_time_LineEdit.setValidator(QDoubleValidator(0.99,99.99,2))
        self.exposure_time_LineEdit.setText(str(self.c_p['exposure_time']))
        self.camera_toolbar.addWidget(self.exposure_time_LineEdit)
        self.camera_toolbar.addAction(self.set_exp_tim)

        # TODO add offset and label to this        
        self.gain_LineEdit = QLineEdit()
        self.gain_LineEdit.setToolTip("Set software gain on displayed image.")
        self.gain_LineEdit.setValidator(QDoubleValidator(0.1,3,3))
        self.gain_LineEdit.setText(str(self.c_p['image_gain']))
        self.gain_LineEdit.setText(str(self.c_p['motor_x_target_speed']))
        self.gain_LineEdit.textChanged.connect(self.set_gain)
        self.camera_toolbar.addWidget(self.gain_LineEdit)

    def set_gain(self, gain):
        try:
            g = min(float(gain), 255)
            self.c_p['image_gain'] = g
            print(f"Gain is now {gain}")
        except ValueError:
            # Harmless, someone deleted all the numbers
            pass

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

    def add_camera_actions(self, toolbar):
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
        
        self.test_led = QAction("Toggle led1", self)
        self.test_led.setToolTip("Turn on/off led1 on the controller.")
        self.test_led.triggered.connect(self.toggle_led)
        self.test_led.setCheckable(True)

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
        self.set_exp_tim = QAction("Set exposure time", self)
        self.set_exp_tim.setToolTip("Sets exposure time to the value in the textboox")
        self.set_exp_tim.triggered.connect(self.set_exposure_time)
        self.open_plot_window.setCheckable(False)

        toolbar.addAction(self.zoom_action)
        toolbar.addAction(self.record_action)
        toolbar.addAction(self.snapshot_action)
        toolbar.addAction(self.test_led)
        # toolbar.addAction(self.open_temperature)
        # toolbar.addAction(self.open_data_window)
        
    def set_video_format(self, video_format):
        self.c_p['video_format'] = video_format

    def toggle_led(self):
        # TestFunction
        self.MCW = MotorControlWidget.MotorControllerWindow(self.c_p)
        self.MCW.show()
        """
        if self.c_p['motor_x_target_speed'] < 0:
            self.c_p['motor_x_target_speed'] = 2
        else:
            self.c_p['motor_x_target_speed'] = -2
        """
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
        # TODO close also other widgets here
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
        if self.PICReaderT is not None:
            self.PICReaderT.join()
        if self.PICWriterT is not None:
            self.PICWriterT.join()

        self.VideoWriterThread.join()

def create_camera_toolbar_external(main_window):
    # TODO do not have this as an external function, urk
    main_window.camera_toolbar = QToolBar("Camera tools")
    main_window.addToolBar(main_window.camera_toolbar)

    # Add test button for led
    main_window.test_led = QAction("Toggle led1", main_window)
    main_window.test_led.setToolTip("Turn on/off led1 on the controller.")
    main_window.test_led.triggered.connect(main_window.toggle_led)
    main_window.test_led.setCheckable(True)
    main_window.camera_toolbar.addAction(main_window.test_led)

    # main_window.add_camera_actions(main_window.camera_toolbar)
    main_window.zoom_action = QAction("Zoom out", main_window)
    main_window.zoom_action.setToolTip("Resets the field of view of the camera.")
    main_window.zoom_action.triggered.connect(main_window.ZoomOut)
    main_window.zoom_action.setCheckable(False)

    main_window.record_action = QAction("Record video", main_window)
    main_window.record_action.setToolTip("Turn ON recording.")
    main_window.record_action.setShortcut('Ctrl+R')
    main_window.record_action.triggered.connect(main_window.ToggleRecording)
    main_window.record_action.setCheckable(True)

    main_window.snapshot_action = QAction("Snapshot", main_window)
    main_window.snapshot_action.setToolTip("Take snapshot of camera view.")
    main_window.snapshot_action.setShortcut('Shift+S')
    main_window.snapshot_action.triggered.connect(main_window.snapshot)
    main_window.snapshot_action.setCheckable(False)

    main_window.open_plot_window = QAction("Open plotter", main_window)
    main_window.open_plot_window.setToolTip("Open live plotting window.")
    main_window.open_plot_window.triggered.connect(main_window.show_new_window)
    main_window.open_plot_window.setCheckable(False)

    main_window.set_exp_tim = QAction("Set exposure time", main_window)
    main_window.set_exp_tim.setToolTip("Sets exposure time to the value in the textboox")
    main_window.set_exp_tim.triggered.connect(main_window.set_exposure_time)
    main_window.open_plot_window.setCheckable(False)

    main_window.camera_toolbar.addAction(main_window.zoom_action)
    main_window.camera_toolbar.addAction(main_window.record_action)
    main_window.camera_toolbar.addAction(main_window.snapshot_action)

    main_window.exposure_time_LineEdit = QLineEdit()
    main_window.exposure_time_LineEdit.setValidator(QDoubleValidator(0.99,99.99,2))
    main_window.exposure_time_LineEdit.setText(str(main_window.c_p['exposure_time']))
    main_window.camera_toolbar.addWidget(main_window.exposure_time_LineEdit)
    main_window.camera_toolbar.addAction(main_window.set_exp_tim)

    # TODO add offset and label to this        
    main_window.gain_LineEdit = QLineEdit()
    main_window.gain_LineEdit.setToolTip("Set software gain on displayed image.")
    main_window.gain_LineEdit.setValidator(QDoubleValidator(0.1,3,3))
    main_window.gain_LineEdit.setText(str(main_window.c_p['image_gain']))
    main_window.gain_LineEdit.textChanged.connect(main_window.set_gain)
    main_window.camera_toolbar.addWidget(main_window.gain_LineEdit)

app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
w.c_p['program_running'] = False
