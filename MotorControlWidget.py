# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:57:06 2023

@author: marti
"""


from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel
)

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QIntValidator

import numpy as np
from functools import partial
from threading import Thread

# Maybe have this as a QThread?
class MotorControllerWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        self.motor_speed = 1
        self.y_movement = 0
        
        self.label = QLabel("Motor controller")
        layout.addWidget(self.label)

        self.SpeedLineEdit = QLineEdit()
        self.SpeedLineEdit.setValidator(QIntValidator(0,100))
        self.SpeedLineEdit.setText(str(self.motor_speed))
        self.SpeedLineEdit.textChanged.connect(self.set_motor_speed)
        layout.addWidget(self.SpeedLineEdit) 
        
        self.up_button = QPushButton('UP')
        self.up_button.pressed.connect(self.move_up)
        self.up_button.released.connect(self.stop_y)
        self.up_button.setCheckable(False)
        layout.addWidget(self.up_button)

        self.down_button = QPushButton('DOWN')
        self.down_button.pressed.connect(self.move_down)
        self.down_button.released.connect(self.stop_y)
        self.down_button.setCheckable(False)
        layout.addWidget(self.down_button)

        self.setLayout(layout)
        
    def set_motor_speed(self, speed):
        # Maybe have this toggleable
        self.motor_speed = int(speed)

    def move_up(self):
        self.c_p['motor_x_target_speed'] = self.motor_speed
    def stop_y(self):
        self.c_p['motor_x_target_speed'] = 0
    def move_down(self):
        self.c_p['motor_x_target_speed'] = - self.motor_speed

        
class MotorControlThread(Thread):
    # Maybe can make do without this
    def __init__(self, c_p, motor_id=0):
        Thread.__init__(self)
        self.c_p = c_p
        self.motor_id = motor_id
        
    def run(self):
        
        while self.c_p['program_running']:
            pass