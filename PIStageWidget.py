# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:14:09 2022

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

import numpy as np
from functools import partial

class PIStageWidget(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        self.label = QLabel("Stage controller")
        layout.addWidget(self.label)
        self.QLE = QLineEdit()

        self.QLE.setValidator(QDoubleValidator(0.99, 29.999, 4))
        self.QLE.setText(str(self.c_p['piezo_targets'][0]))
        self.QLE.textChanged.connect(self.set_setpoint_T)
        self.position_button = QPushButton('Update position')
        self.position_button.clicked.connect(self.UpdatePosition)
        self.position_button.setCheckable(True)
        self.position_button.setChecked(self.c_p['temperature_output_on'])
        layout.addWidget(self.QLE)
        layout.addWidget(self.position_button)

        self.setLayout(layout)
        
    def set_setpoint_T(self, temperature):
        # Maybe have this toggleable
        if len(temperature) == 0:
            return
        try:
            t = float(temperature)
            if 0 < t <= 30:
                self.c_p['piezo_targets'][0] = t
        except ValueError as VE:
            pass

    def UpdatePosition(self, _):

        self.label.setText(f"Current pos {self.c_p['piezo_pos']}.")