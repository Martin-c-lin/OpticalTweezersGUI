from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel
)

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QIntValidator
from PyQt6.QtCore import Qt

import numpy as np
from functools import partial
from threading import Thread
from ThorlabsMotor import MotorThreadV2, PiezoThread
from CustomMouseTools import MouseInterface
from time import sleep

# Maybe have this as a QThread?
class LaserPiezoWidget(QWidget):

    def __init__(self, c_p, data_channels):
        super().__init__()
        self.c_p = c_p
        self.data_channels = data_channels
        self.setWindowTitle("Piezo wiggler Controller")
        layout = QVBoxLayout()
        self.label = QLabel("Piezo A x")
        layout.addWidget(self.label)

        # Create the slider for the piezo A x-axis
        self.piezo_Ax_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.piezo_Ax_slider.setGeometry(50,50, 200, 50)
        self.piezo_Ax_slider.setMinimum(0)
        self.piezo_Ax_slider.setMaximum(65535)
        self.piezo_Ax_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.piezo_Ax_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.piezo_Ax_slider.setTickInterval(1)
        self.piezo_Ax_slider.valueChanged[int].connect(self.set_piezo_Ax_value) 

        layout.addWidget(self.piezo_Ax_slider)
        
        self.label_AY = QLabel("Piezo A y")
        layout.addWidget(self.label_AY)
        # Create the slider for the piezo A y-axis
        self.piezo_Ay_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.piezo_Ay_slider.setGeometry(50,50, 200, 50)
        self.piezo_Ay_slider.setMinimum(0)
        self.piezo_Ay_slider.setMaximum(65535)
        self.piezo_Ay_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.piezo_Ay_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.piezo_Ay_slider.setTickInterval(1)
        self.piezo_Ay_slider.valueChanged[int].connect(self.set_piezo_Ay_value) 

        layout.addWidget(self.piezo_Ay_slider)

        self.label_BX = QLabel("Piezo B x")
        layout.addWidget(self.label_BX)
        # Create the slider for the piezo B x-axis
        self.piezo_Bx_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.piezo_Bx_slider.setGeometry(50,50, 200, 50)
        self.piezo_Bx_slider.setMinimum(0)
        self.piezo_Bx_slider.setMaximum(65535)
        self.piezo_Bx_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.piezo_Bx_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.piezo_Bx_slider.setTickInterval(1)
        self.piezo_Bx_slider.valueChanged[int].connect(self.set_piezo_Bx_value) 

        layout.addWidget(self.piezo_Bx_slider)
        
        self.label_BY = QLabel("Piezo B y")
        layout.addWidget(self.label_BY)
        # Create the slider for the piezo B y-axis
        self.piezo_By_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.piezo_By_slider.setGeometry(50,50, 200, 50)
        self.piezo_By_slider.setMinimum(0)
        self.piezo_By_slider.setMaximum(65535)
        self.piezo_By_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.piezo_By_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.piezo_By_slider.setTickInterval(1)
        self.piezo_By_slider.valueChanged[int].connect(self.set_piezo_By_value) 

        layout.addWidget(self.piezo_By_slider)

        self.autoalign_A = QCheckBox("Autoalign A")
        self.autoalign_A.setChecked(self.c_p['portenta_command_2']==1)
        self.autoalign_A.stateChanged.connect(self.toggle_autoalign_A)
        layout.addWidget(self.autoalign_A)

        self.autoalign_B = QCheckBox("Autoalign B")
        self.autoalign_B.setChecked(self.c_p['portenta_command_2']==2)
        self.autoalign_B.stateChanged.connect(self.toggle_autoalign_B)
        layout.addWidget(self.autoalign_B)

        self.center_piezos_button = QPushButton("Center Piezos")
        self.center_piezos_button.clicked.connect(self.center_piezos)
        self.center_piezos_button.setCheckable(False)
        layout.addWidget(self.center_piezos_button)

        self.set_slider_values()
        self.setLayout(layout)
        # TODO make this widget update itself when the piezos are moved by a separate function such as the
        #  pulling protocol. Also make it possible to autoalign both A and B at the same time.

    def set_slider_values(self):
        self.piezo_Ax_slider.setValue(self.c_p['piezo_A'][0])
        self.piezo_Ay_slider.setValue(self.c_p['piezo_A'][1])
        self.piezo_By_slider.setValue(self.c_p['piezo_B'][1])
        self.piezo_Bx_slider.setValue(self.c_p['piezo_B'][0])

    def center_piezos(self):
        self.c_p['piezo_A'] = [32768, 32768]
        self.c_p['piezo_B'] = [32768, 32768]
        self.set_slider_values()

    def set_piezo_Ax_value(self):
        self.c_p['piezo_A'][0] = int(self.piezo_Ax_slider.value())

    def set_piezo_Ay_value(self):
        self.c_p['piezo_A'][1] = int(self.piezo_Ay_slider.value())
    
    def set_piezo_Bx_value(self):
        self.c_p['piezo_B'][0] = int(self.piezo_Bx_slider.value())

    def set_piezo_By_value(self):
        self.c_p['piezo_B'][1] = int(self.piezo_By_slider.value())

    def toggle_autoalign_A(self):
        if self.c_p['portenta_command_2'] == 0 or self.c_p['portenta_command_2'] == 2:
            self.c_p['portenta_command_2'] = 1
            print('toggling autoalign of trap A')
            self.autoalign_B.setChecked(False)   
        else:
            self.c_p['piezo_A'] = np.int32([self.data_channels['dac_ax'].get_data_spaced(1)[0],
                                            self.data_channels['dac_ay'].get_data_spaced(1)[0]])
            self.c_p['piezo_B'] = np.int32([self.data_channels['dac_bx'].get_data_spaced(1)[0],
                                            self.data_channels['dac_by'].get_data_spaced(1)[0]])
            self.c_p['portenta_command_2'] = 0

            self.set_slider_values()

            print('Disabling autoalign of trap A')

    def toggle_autoalign_B(self):
        if self.c_p['portenta_command_2'] == 0 or self.c_p['portenta_command_2'] == 1:
            self.c_p['portenta_command_2'] = 2
            self.autoalign_A.setChecked(False)
            print('toggling autoalign of trap B')
        else:
            self.c_p['piezo_A'] = np.int32([self.data_channels['dac_ax'].get_data_spaced(1)[0],
                                            self.data_channels['dac_ay'].get_data_spaced(1)[0]])
            self.c_p['piezo_B'] = np.int32([self.data_channels['dac_bx'].get_data_spaced(1)[0],
                                            self.data_channels['dac_by'].get_data_spaced(1)[0]])
            self.c_p['portenta_command_2'] = 0

            self.set_slider_values()

            print('Disabling autoalign of trap B')


    
class MinitweezersLaserMove(MouseInterface):
    
    def __init__(self, c_p ):
        self.c_p = c_p
        self.speed_factor = 300
        self.x_prev_A = 0
        self.y_prev_A = 0
        self.x_prev_B = 0
        self.y_prev_B = 0

    def mousePress(self):

        # left click
        if self.c_p['mouse_params'][0] == 1:
            self.x_prev_A = self.c_p['mouse_params'][1]
            self.y_prev_A = self.c_p['mouse_params'][2]
        # Right click -drag
        if self.c_p['mouse_params'][0] == 2:
            self.x_prev_B = self.c_p['mouse_params'][1]
            self.y_prev_B = self.c_p['mouse_params'][2]
        # Scroll wheel
        elif self.c_p['mouse_params'][0] == 3:
            pass
        
    def mouseRelease(self):
        if self.c_p['mouse_params'][0] == 2:
            pass
        
    def mouseDoubleClick(self):
        pass
    
    def draw(self, qp):
        pass
    def check_limit(self, number):
        if number < 0:
            return 0
        if number>65535:
            return 65535
        return number

    def mouseMove(self):
        if self.c_p['mouse_params'][0] == 1: # A
            # TODO maybe not round here
            dx = int((self.c_p['mouse_params'][3] - self.x_prev_A)*self.speed_factor)
            dy = int((self.c_p['mouse_params'][4] - self.y_prev_A)*self.speed_factor)
            self.c_p['piezo_A'][0] = self.check_limit(dx+self.c_p['piezo_A'][0])
            self.c_p['piezo_A'][1] = self.check_limit(dy+self.c_p['piezo_A'][1])

            #print(f"Y Position of Piezo A is {self.c_p['piezo_A']}")
            self.x_prev_A = self.c_p['mouse_params'][3]
            self.y_prev_A = self.c_p['mouse_params'][4] 

        if self.c_p['mouse_params'][0] == 2: # B
            dx = int((self.c_p['mouse_params'][3] - self.x_prev_B)*self.speed_factor)
            dy = int((self.c_p['mouse_params'][4] - self.y_prev_B)*self.speed_factor)
            self.c_p['piezo_B'][0] = self.check_limit(-dx+self.c_p['piezo_B'][0])
            self.c_p['piezo_B'][1] = self.check_limit(dy+self.c_p['piezo_B'][1])

            #print(f"Y Position of piezo B is {self.c_p['piezo_B']}")
            self.x_prev_B = self.c_p['mouse_params'][3]
            self.y_prev_B = self.c_p['mouse_params'][4] 

    def getToolName(self):
        return "Piezo manual move tool"

    def getToolTip(self):
        return "Move the laser by dragging on the screen.\n Left click for laser A, right click for laser B."
        