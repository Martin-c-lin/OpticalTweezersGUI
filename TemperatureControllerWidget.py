# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:47:32 2022

@author: Martin
"""

import sys
from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel
)

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QDoubleValidator, QIntValidator
sys.path.append("..")
from TemperatureControllerTED4015 import TED4015
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from random import randint

import numpy as np
from functools import partial

class TempereatureControllerWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        self.label = QLabel("Temperature controller")
        layout.addWidget(self.label)
        self.QLE = QLineEdit()

        self.QLE.setValidator(QDoubleValidator(0.99,39.999,3))
        self.QLE.setText(str(self.c_p['setpoint_temperature']))
        self.QLE.textChanged.connect(self.set_setpoint_T)
        self.output_button = QPushButton('Output on')
        self.output_button.clicked.connect(self.ToggleOutput)
        self.output_button.setCheckable(True)
        self.output_button.setChecked(self.c_p['temperature_output_on'])
        layout.addWidget(self.QLE)
        layout.addWidget(self.output_button)

        self.setLayout(layout)
        
    def set_setpoint_T(self, temperature):
        # Maybe have this toggleable
        self.c_p['setpoint_temperature'] = float(temperature)

    def ToggleOutput(self, _):
        print("output yeah")
        self.c_p['temperature_output_on'] = not self.c_p['temperature_output_on']
