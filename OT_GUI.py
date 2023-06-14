# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:50:13 2022

@author: marti
"""
import sys
import cv2 # Certain versions of this won't work
import pickle

from PyQt6.QtWidgets import (
    QMainWindow, QApplication,
    QLabel, QCheckBox, QComboBox, QListWidget, QLineEdit, QSpinBox,
    QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QFileDialog, QInputDialog
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QRunnable, QObject, QPoint, QRect, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QBrush, QColor, QAction, QDoubleValidator, QPen, QIntValidator

from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from random import randint
import numpy as np
from time import sleep
from functools import partial


import BaslerCameras
import ThorlabsCameras
from CameraControlsNew import CameraThread, VideoWriterThread, CameraClicks
from ControlParameters import default_c_p, get_data_dicitonary_new
from TemperatureControllerTED4015 import TemperatureThread
from TemperatureControllerWidget import TempereatureControllerWindow
from ReadPicUart import PicReader, PicWriter
from LivePlots import PlotWindow
from SaveDataWidget import SaveDataWindow
from PIStage import PIStageThread
from PIStageWidget import PIStageWidget
import MotorControlWidget
from LaserPiezosControlWidget import LaserPiezoWidget, MinitweezersLaserMove
from DeepLearningThread import MouseAreaSelect, DeepLearningAnalyserLDS, DeepLearningControlWidget
from PlanktonViewWidget import PlanktonViewer
from DataChannelsInfoWindow import CurrentValueWindow
from ReadArduinoPortenta import PortentaComms
from StepperObjective import ObjectiveStepperController
import AutoController
import LaserController


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

    def __init__(self, c_p, data, test_mode=False, *args, **kwargs):
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
            self.data_channels['X-position'].put_data(self.c_p['stepper_current_position'][0])
            self.data_channels['Y-position'].put_data(self.c_p['stepper_current_position'][1])
            self.data_channels['Z-position'].put_data(np.random.rand(max_length) * 2 - 1)
            self.data_channels['Motor_position'].put_data(np.sin(self.data_channels['Time'].data / 10))
        else:
            # Shift the data
            # Update last element
            self.data_channels['Time'].put_data(self.data_channels['Time'].get_data(1) + self.dt)

            self.data_channels['Y-force'].put_data(np.sin(self.data_channels['Time'].get_data(1) / 10))
            self.data_channels['X-force'].put_data(np.cos(self.data_channels['Time'].get_data(1) / 10))
            self.data_channels['Z-force'].put_data(np.cos(self.data_channels['Time'].get_data(1) / 10)**2)

            self.data_channels['X-position'].put_data(self.c_p['stepper_current_position'][0])
            self.data_channels['Y-position'].put_data(self.c_p['stepper_current_position'][1])
            self.data_channels['Z-position'].put_data(np.random.rand() * 2 - 1)
            self.data_channels['Motor_position'].put_data((self.data_channels['Time'].get_data(1) / 10) + np.random.rand())

    def draw_particle_positions(self, centers, pen=None):
        # TODO add function also for crosshair to help with alignment.
        rx = int(250/self.c_p['image_scale'])
        ry = rx
        if pen is None:
            self.qp.setPen(self.red_pen)
        else:
            self.qp.setPen(pen)
        for pos in centers:
            x = int(pos[0]/ self.c_p['image_scale']) # Which is which?
            y = int(pos[1]/ self.c_p['image_scale'])

            self.qp.drawEllipse(x-int(rx/2)-1, y-int(ry/2)-1, rx, ry)
    
    def preprocess_image(self):

        # Check if offset and gain should be applied.
        if self.c_p['image_offset'] != 0:
            self.image += int(self.c_p['image_offset'])
            
        if self.c_p['image_gain'] != 1:
            # TODO unacceptably slow
            self.image = (self.image*self.c_p['image_gain'])

        self.image = np.uint8(self.image)

    def draw_central_circle(self):
        self.blue_pen.setColor(QColor('blue'))
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
                # TODO have this add data channels if they are not already created.
                self.testDataUpdate()

            if self.c_p['image'] is not None:
                self.image = np.array(self.c_p['image'])
            else:
                print("Frame missed!")
            W, H = self.c_p['frame_size']
            
            self.c_p['image_scale'] = max(self.image.shape[1]/W, self.image.shape[0]/H)
            
            self.preprocess_image()
            

            # It is quite sensitive to the format here, won't accept any missmatch
            
            if len(np.shape(self.image)) < 3:
                QT_Image = QImage(self.image, self.image.shape[1],
                                       self.image.shape[0],
                                       QImage.Format.Format_Grayscale8)
                QT_Image = QT_Image.convertToFormat(QImage.Format.Format_RGB888)
            else:                
                QT_Image = QImage(self.image, self.image.shape[1],
                                       self.image.shape[0],
                                       QImage.Format.Format_RGB888)
                
            picture = QT_Image.scaled(
                W,H,
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            # Give other things time to work, roughly 40-50 fps default.
            sleep(0.04) # Sets the FPS
            
            # Paint extra items on the screen
            self.qp = QPainter(picture)
            # Draw zoom in rectangle
            self.c_p['click_tools'][self.c_p['mouse_params'][5]].draw(self.qp)
            self.qp.setPen(self.blue_pen)
            self.draw_central_circle()
            if self.c_p['tracking_on']:
                self.draw_particle_positions(self.c_p['predicted_particle_positions'])
            if self.c_p['locate_pippette'] and self.c_p['pipette_location'][0] is not None:
                self.draw_particle_positions([self.c_p['pipette_location']])
            self.qp.end()
            self.changePixmap.emit(picture)


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Optical tweezers: Main window")
        self.c_p = default_c_p()
        self.data_channels = get_data_dicitonary_new()
        self.video_idx = 0
        self.data_idx = 0 # Index of data save
        self.saving = False
        # Start camera threads
        self.CameraThread = None
        try:
            camera = None
            camera = BaslerCameras.BaslerCamera()
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
        self.PortentaReaderT = None
        try:
            
            self.PortentaReaderT = PortentaComms(self.c_p, self.data_channels) #PicReader(self.c_p, self.data_channels)
            self.PortentaReaderT.start()
            sleep(0.1)
            """
            self.PICReaderT = PicReader(self.c_p, self.data_channels)
            self.PICReaderT.start()
            self.PICWriterT = PicWriter(self.c_p, self.PICReaderT.serial_channel)
            self.PICWriterT.start()
            """
            
        except Exception as E:
            print(E)

        try:
            self.AotuControllerThread = AutoController.autoControllerThread(self.c_p, self.data_channels)
            self.AotuControllerThread.start()
            print("Auto controller started")
        except Exception as E:
            print(E)

        self.ArduinoUnoSerial = None
        try:
            import serial
            port = self.c_p['objective_stepper_port']
            self.ArduinoUnoSerial = serial.Serial(port, 9600)
            print("Connected to Arduino Uno.")
        except Exception as E:
            print(E)
            print("Could not connect to Arduino Uno for objective stepper control!")

        self.VideoWriterThread = VideoWriterThread(2, 'video thread', self.c_p)
        self.VideoWriterThread.start()

        self.DeepThread = DeepLearningAnalyserLDS(self.c_p, self.data_channels)
        self.DeepThread.start()

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
        self.addToolBarBreak() 
        self.create_mouse_toolbar()

        # Create menus and drop down options
        self.menu = self.menuBar()
        self.create_filemenu()
        self.drop_down_window_menu()
        self.action_menu()
        self.show()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def start_threads(self):
        pass
    
    def create_mouse_toolbar(self):
        # Here is where all the tools in the mouse toolbar are added
        self.c_p['click_tools'].append(CameraClicks(self.c_p))
        self.c_p['click_tools'].append(MotorControlWidget.MinitweezersMouseMove(self.c_p, self.data_channels))
        self.c_p['click_tools'].append(MotorControlWidget.MotorClickMove(self.c_p,)) # Thorlabs motors
        self.c_p['click_tools'].append(MinitweezersLaserMove(self.c_p))
        #self.c_p['click_tools'].append(MouseAreaSelect(self.c_p))
        self.c_p['click_tools'].append(AutoController.SelectLaserPosition(self.c_p))

        self.c_p['mouse_params'][5] = 0

        self.mouse_toolbar = QToolBar("Mouse tools")
        self.addToolBar(self.mouse_toolbar)
        self.mouse_actions = []
        
        for idx, tool in enumerate(self.c_p['click_tools']):
            self.mouse_actions.append(QAction(tool.getToolName(), self))
            self.mouse_actions[-1].setToolTip(tool.getToolTip())
            command = partial(self.set_mouse_tool, idx)
            self.mouse_actions[-1].triggered.connect(command)
            self.mouse_actions[-1].setCheckable(True)
            self.mouse_toolbar.addAction(self.mouse_actions[-1])
        self.mouse_actions[self.c_p['mouse_params'][5]].setChecked(True)
        
    def set_mouse_tool(self, tool_no=0):
        if tool_no > len(self.c_p['click_tools']):
            return
        self.c_p['mouse_params'][5] = tool_no
        for tool in self.mouse_actions:
            tool.setChecked(False)
        self.mouse_actions[tool_no].setChecked(True)
        print("Tool set to ", tool_no)

    def set_gain(self, gain):
        try:
            g = min(float(gain), 255)
            self.c_p['image_gain'] = g
            print(f"Gain is now {gain}")
        except ValueError:
            # Harmless, someone deleted all the numbers in the line-edit
            pass

    def create_filemenu(self):

        file_menu = self.menu.addMenu("File")
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

        set_filename_action = QAction("Set filename", self)
        set_filename_action.setStatusTip("Set filename for saved, data, video and image files")
        set_filename_action.triggered.connect(self.set_default_filename)
        file_menu.addAction(set_filename_action)

        # Add command to save the data
        save_data_action = QAction("Save data", self)
        save_data_action.setStatusTip("Save data to an npy file")
        save_data_action.triggered.connect(self.save_data_to_dict) # Dump data before
        file_menu.addAction(save_data_action)

    def dump_data(self):
        text, ok = QInputDialog.getText(self, 'Filename dialog', 'Set name for data to be saved:')
        if not ok:
            print("No valid name entered")
            return
        path = self.c_p['recording_path'] + '/' + text
        print(f"Saving data to {path}")
        np.save(path,  self.data_channels, allow_pickle=True)
    
    def record_data(self):
        if (self.saving):
            self.stop_saving()
            self.toggle_data_record_action.setText("Start recording data")
        else:
            self.start_saving()
            self.toggle_data_record_action.setText("Stop recording data")

    def start_saving(self):
        # TODO make continous saving possible to avoid unneccesary saving
        self.start_idx = self.data_channels['PSD_A_P_X'].index
        self.saving = True
        print("Saving started")

    def stop_saving(self):
        self.saving = False
        print("Saving stopped")
        self.stop_idx = self.data_channels['PSD_A_P_X'].index
        sleep(0.1) # Waiting for all channels to reach this point
        data = {}
        for channel in self.data_channels:
            if self.data_channels[channel].saving_toggled:
                data[channel] = self.data_channels[channel].data[self.start_idx:self.stop_idx]
        filename = self.c_p['recording_path'] + '/' + self.c_p['filename']
        with open(filename, 'wb') as f:
                pickle.dump(data, f)

    def save_data_to_dict(self):

        text, ok = QInputDialog.getText(self, 'Filename dialog', 'Set name for data to be saved:')
        if not ok:
            print("No valid name entered")
            return
        filename = self.c_p['recording_path'] + '/' + text
        self.c_p['save_idx'] = self.data_channels['PSD_A_P_X'].index
        sleep(0.1) # Make sure all channels have reached this point
        data = {}
        for channel in self.data_channels:
            if self.data_channels[channel].saving_toggled:
                data[channel] = self.data_channels[channel].get_data_spaced(1e9)
        print(f"Saving data to {filename}")
        #np.save(path, data, allow_pickle=True)
        with open(filename, 'wb') as f:
                pickle.dump(data, f)

    def action_menu(self):
        action_menu = self.menu.addMenu("Actions")

        self.save_position_action = QAction("Save position", self)
        self.save_position_action.setStatusTip("Save current position")
        self.save_position_action.triggered.connect(self.save_position)
        action_menu.addAction(self.save_position_action)

        self.zero_force_action = QAction("Zero force", self)
        self.zero_force_action.setStatusTip("Zero force for current value, resets it if it'z already zeroed")
        self.zero_force_action.triggered.connect(self.zero_force_PSDs)
        action_menu.addAction(self.zero_force_action)

        self.reset_force_psds_action = QAction("Reset force PSDs", self)
        self.reset_force_psds_action.setStatusTip("Reset force PSDs their default values")
        self.reset_force_psds_action.triggered.connect(self.reset_force_PSDs)
        action_menu.addAction(self.reset_force_psds_action)

        self.saved_positions_submenu = action_menu.addMenu("Saved positions")

        for idx in range(len(self.c_p['saved_positions'])):
            self.add_position(idx)

    def zero_force_PSDs(self):
        self.c_p['PSD_means'][0] = 32768 + np.uint16(np.mean(self.data_channels['PSD_A_F_X'].get_data_spaced(1000)))
        self.c_p['PSD_means'][1] = 32768 + np.uint16(np.mean(self.data_channels['PSD_A_F_Y'].get_data_spaced(1000)))

        self.c_p['PSD_means'][2] = 32768 + np.uint16(np.mean(self.data_channels['PSD_B_F_X'].get_data_spaced(1000)))
        self.c_p['PSD_means'][3] = 32768 + np.uint16(np.mean(self.data_channels['PSD_B_F_Y'].get_data_spaced(1000)))
        print(self.c_p['PSD_means'])
        self.c_p['portenta_command_1'] = 1

    def reset_force_PSDs(self):
        self.c_p['portenta_command_1'] = 2

    def add_position(self,idx):
        # Adds position to submenu
        position_command = partial(self.goto_position, idx)
        position_action = QAction(self.c_p['saved_positions'][idx][0], self)
        position_action.setStatusTip(f"Move to saved position {self.c_p['saved_positions'][idx][0]}")
        position_action.triggered.connect(position_command)
        self.saved_positions_submenu.addAction(position_action) # Check how to remove this
        # TODO add a remove position option as well as rename and check position values.

    def set_default_filename(self):
        text, ok = QInputDialog.getText(self, 'Filename dialog', 'Enter name of your files:')
        if ok:
            self.video_idx = 0
            self.c_p['image_idx'] = 0
            self.c_p['filename'] = text
            self.c_p['video_name'] = text + '_video' + str(self.video_idx)
            print(f"Filename is now {text}")

    def save_position(self):
        if not self.c_p['minitweezers_connected']:
            x = self.c_p['stepper_current_position'][0]
            y = self.c_p['stepper_current_position'][1]
            z = 0
        else:
            x = self.data_channels['Motor_x_pos'].get_data(1)[0]
            y = self.data_channels['Motor_y_pos'].get_data(1)[0]
            z = self.data_channels['Motor_z_pos'].get_data(1)[0]

        text, ok = QInputDialog.getText(self, 'Save position dialog', 'Enter name of position:')
        if ok:
            self.c_p['saved_positions'].append([text, x, y, z])
            print(f"Saved position {x}, {y} as position: {text}")
            self.add_position(len(self.c_p['saved_positions'])-1)
        else:
            print("No position saved")

    def goto_position(self,idx):
        if idx>len(self.c_p['saved_positions']):
            return
        if self.c_p['move_to_location']:
            # Stop first then start moving, TODO add this as separate button/command
            self.c_p['move_to_location'] = False
            self.c_p['motor_x_target_speed'] = 0
            self.c_p['motor_y_target_speed'] = 0
            self.c_p['motor_z_target_speed'] = 0
            return
        # TODO make it use same code for both the options for the motors
        if self.c_p['minitweezers_connected']:
            self.c_p['minitweezers_target_pos'][0] = int(self.c_p['saved_positions'][idx][1]) # Added +32768
            self.c_p['minitweezers_target_pos'][1] = int(self.c_p['saved_positions'][idx][2])
            self.c_p['minitweezers_target_pos'][2] = int(self.c_p['saved_positions'][idx][3])
            self.c_p['move_to_location'] = True
        else:
            self.c_p['stepper_target_position'][0:2] = self.c_p['saved_positions'][idx][1:3]

    def drop_down_window_menu(self):
        # Create windows drop down menu
        window_menu = self.menu.addMenu("Windows")
        window_menu.addSeparator()

        self.open_plot_window = QAction("Live plotter", self)
        self.open_plot_window.setToolTip("Open live plotting window.")
        self.open_plot_window.triggered.connect(self.show_new_window)
        self.open_plot_window.setCheckable(False)
        window_menu.addAction(self.open_plot_window)

        self.open_positions_window = QAction("Position PSDs", self)
        self.open_positions_window.setToolTip("Open window for position PSDs.\n This is a specially configured version of the live plotter")
        self.open_positions_window.triggered.connect(self.open_Position_PSD_window)
        self.open_positions_window.setCheckable(False)
        window_menu.addAction(self.open_positions_window)

        self.open_force_window = QAction("Force PSDs", self)
        self.open_force_window.setToolTip("Open window for force PSDs.\n This is a specially configured version of the live plotter")
        self.open_force_window.triggered.connect(self.open_Force_PSD_window)
        self.open_force_window.setCheckable(False)
        window_menu.addAction(self.open_force_window)
        
        self.open_motor_window = QAction("Minitweezers motor window", self)
        self.open_motor_window.setToolTip("Open window for manual motor control.")
        self.open_motor_window.triggered.connect(self.open_motor_control_window)
        self.open_motor_window.setCheckable(False)
        window_menu.addAction(self.open_motor_window)


        self.open_stepper_window = QAction("Objective motor window", self)
        self.open_stepper_window.setToolTip("Open window for manual motor control, objective stepper motor.")
        self.open_stepper_window.triggered.connect(self.open_stepper_objective)
        self.open_stepper_window.setCheckable(False)
        window_menu.addAction(self.open_stepper_window)

        """
        self.open_thorlabsM_window = QAction("Thorlabs motor window", self)
        self.open_thorlabsM_window.setToolTip("Open window for manual motor control, thorlabs motors.")
        self.open_thorlabsM_window.triggered.connect(self.open_thorlabs_motor_control_window)
        self.open_thorlabsM_window.setCheckable(False)
        window_menu.addAction(self.open_thorlabsM_window)

        self.open_plankton_window = QAction("Plankton viewer", self)
        self.open_plankton_window.setToolTip("Open plankton viewer window.")
        self.open_plankton_window.triggered.connect(self.openPlanktonViwer)
        self.open_plankton_window.setCheckable(False)
        window_menu.addAction(self.open_plankton_window)
        """
        self.open_deep_window = QAction("DL window", self)
        self.open_deep_window.setToolTip("Open window for deep learning control.")
        self.open_deep_window.triggered.connect(self.OpenDeepLearningWindow)
        self.open_deep_window.setCheckable(False)
        window_menu.addAction(self.open_deep_window)
        
        self.open_laser_piezo_window_action = QAction("Laser piezos window", self)
        self.open_laser_piezo_window_action.setToolTip("Open window controlling piezos of lasers.")
        self.open_laser_piezo_window_action.triggered.connect(self.OpenLaserPiezoWidget)
        self.open_laser_piezo_window_action.setCheckable(False)
        window_menu.addAction(self.open_laser_piezo_window_action)
        
        self.open_channel_viewer = QAction("Data channels", self)
        self.open_channel_viewer.setToolTip("Opens a separate window in which the current values \n of the data channels is displayed.")
        self.open_channel_viewer.triggered.connect(self.open_channels_winoow)
        self.open_channel_viewer.setCheckable(False)
        window_menu.addAction(self.open_channel_viewer)

        self.auto_controller_action = QAction("Auto controller", self)
        self.auto_controller_action.setToolTip("Opens a window for interfacing the auto controller.")
        self.auto_controller_action.triggered.connect(self.openAutoControllerWidnow)
        self.auto_controller_action.setCheckable(False)
        window_menu.addAction(self.auto_controller_action)

        self.open_laser_window_action = QAction("Laser controller", self)
        self.open_laser_window_action.setToolTip("Opens a window for interfacing the laser controller.")
        self.open_laser_window_action.triggered.connect(self.open_laser_window)
        self.open_laser_window_action.setCheckable(False)
        window_menu.addAction(self.open_laser_window_action)



    def openPlanktonViwer(self):
        self.planktonView = PlanktonViewer(self.c_p)

    def open_laser_window(self):
        # TODO make it impossible to open more than one of these windows
        self.laser_window = LaserController.LaserControllerWidget(self.c_p)
        self.laser_window.show()

    def open_channels_winoow(self):
        self.channelView = CurrentValueWindow(self.c_p, self.data_channels)
        self.channelView.show()

    def set_video_format(self, video_format):
        self.c_p['video_format'] = video_format

    def open_motor_control_window(self):
        self.MCW = MotorControlWidget.MotorControllerWindow(self.c_p)
        self.MCW.show()

    def open_stepper_objective(self):
        self.obective_controller = ObjectiveStepperController(self.c_p, self.ArduinoUnoSerial)
        self.obective_controller.show()

    def open_thorlabs_motor_control_window(self):
        self.MCW_T = MotorControlWidget.ThorlabsMotorWindow(self.c_p)
        self.MCW_T.show()

    def set_image_format(self, image_format):
        self.c_p['image_format'] = image_format
        
    def set_video_name(self, string):
        self.c_p['video_name'] = string

    def set_exposure_time(self):
        # Updates the exposure time of the camera to what is inside the textbox
        self.c_p['exposure_time'] = float(self.exposure_time_LineEdit.text())
        self.c_p['new_settings_camera'] = [True, 'exposure_time']

    def set_save_path(self):
        fname = QFileDialog.getExistingDirectory(self, "Save path")
        if len(fname) > 3:
            # If len is less than 3 then the action was cancelled and we should not update
            # the path.
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
            self.c_p['video_name'] = self.c_p['filename'] + '_video' + str(self.video_idx)
            self.video_idx += 1
            self.record_action.setToolTip("Turn OFF recording.")
        else:
            self.record_action.setToolTip("Turn ON recording.")

    def snapshot(self):
        # Captures a snapshot of what the camera is viewing and saves that
        # in the fileformat specified by the image_format parameter.
        idx = str(self.c_p['image_idx'])
        filename = self.c_p['recording_path'] + '/'+self.c_p['filename']+'image_' + idx +'.'+\
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
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mouseMove()


    def mousePressEvent(self, e):
        
        self.c_p['mouse_params'][1] = e.pos().x()-self.label.pos().x()
        self.c_p['mouse_params'][2] = e.pos().y()-self.label.pos().y()

        if e.button() == Qt.MouseButton.LeftButton:
            self.c_p['mouse_params'][0] = 1
        if e.button() == Qt.MouseButton.RightButton:
            self.c_p['mouse_params'][0] = 2
        if e.button() == Qt.MouseButton.MiddleButton:
            self.c_p['mouse_params'][0] = 3
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mousePress()

    def mouseReleaseEvent(self, e):


        self.c_p['mouse_params'][3] = e.pos().x()-self.label.pos().x()
        self.c_p['mouse_params'][4] = e.pos().y()-self.label.pos().y()
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mouseRelease()
        self.c_p['mouse_params'][0] = 0


    def mouseDoubleClickEvent(self, e):
        # Double click to move center?
        x = e.pos().x()-self.label.pos().x()
        y = e.pos().y()-self.label.pos().y()
        print(x*self.c_p['image_scale'] ,y*self.c_p['image_scale'] )
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mouseDoubleClick()


    def show_new_window(self, checked):
        if self.plot_windows is None:
            self.plot_windows = []
        self.plot_windows.append(PlotWindow(self.c_p, data=self.data_channels,
                                          x_keys=['T_time'], y_keys=['PSD_A_P_X']))

        self.plot_windows[-1].show()

    def open_Position_PSD_window(self):
        if self.plot_windows is None:
            self.plot_windows = []
        self.plot_windows.append(PlotWindow(self.c_p, data=self.data_channels,
                                          x_keys=['PSD_A_P_X','PSD_B_P_X'], y_keys=['PSD_A_P_Y','PSD_B_P_Y'],
                                          aspect_locked=True, grid_on=True))
        self.plot_windows[-1].show()

    def open_Force_PSD_window(self):
        if self.plot_windows is None:
            self.plot_windows = []
        self.plot_windows.append(PlotWindow(self.c_p, data=self.data_channels,
                                          x_keys=['PSD_A_F_X','PSD_B_F_X'], y_keys=['PSD_A_F_Y','PSD_B_F_Y'],
                                          aspect_locked=True, grid_on=True))
        self.plot_windows[-1].show()

    def OpenTemperatureWindow(self):
        self.temp_control_window = TempereatureControllerWindow(self.c_p)
        self.temp_control_window.show()
        
    def OpenPIStage(self):
        self.PI_window = PIStageWidget(self.c_p)
        self.PI_window.show()

    def openAutoControllerWidnow(self):
        self.auto_controller_window = AutoController.AutoControlWidget(self.c_p, self.data_channels)
        self.auto_controller_window.show()

    def DataWindow(self):
        self.data_window= SaveDataWindow(self.c_p, self.data_channels)
        self.data_window.show()

    def OpenDeepLearningWindow(self):
        self.dep_learning_window = DeepLearningControlWidget(self.c_p)
        self.dep_learning_window.show()

    def OpenLaserPiezoWidget(self):
        self.laser_piezo_window = LaserPiezoWidget(self.c_p, self.data_channels)
        self.laser_piezo_window.show()

    def closeEvent(self, event):
        # TODO close also other widgets here
        if self.plot_windows is not None:
            for w in self.plot_windows:
                w.close()
        self.__del__


    def __del__(self):
        self.c_p['program_running'] = False
        # TODO organize this better
        if self.CameraThread is not None:
            self.CameraThread.join()
        if self.TemperatureThread is not None:
            self.TemperatureThread.join()
        if self.PortentaReaderT is not None:
            self.PortentaReaderT.join()
        if self.PICReaderT is not None:
            self.PICReaderT.join()
        if self.PICWriterT is not None:
            self.PICWriterT.join()

        if self.ArduinoUnoSerial is not None and self.ArduinoUnoSerial.is_open:
            self.ArduinoUnoSerial.close()

        self.DeepThread.join()

        self.VideoWriterThread.join()

def create_camera_toolbar_external(main_window):
    # TODO do not have this as an external function, urk
    main_window.camera_toolbar = QToolBar("Camera tools")
    main_window.addToolBar(main_window.camera_toolbar)

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

    main_window.set_exp_tim = QAction("Set exposure time", main_window)
    main_window.set_exp_tim.setToolTip("Sets exposure time to the value in the textboox")
    main_window.set_exp_tim.triggered.connect(main_window.set_exposure_time)

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

    main_window.toggle_data_record_action = QAction("Start saving data", main_window)
    main_window.toggle_data_record_action.setToolTip("Turn ON recodording of data.\n Data will be saved to fileneame set in files windows.")
    main_window.toggle_data_record_action.setCheckable(True)
    main_window.toggle_data_record_action.triggered.connect(main_window.record_data)
    main_window.camera_toolbar.addAction(main_window.toggle_data_record_action)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()
    w.c_p['program_running'] = False
