from PyQt6.QtWidgets import QVBoxLayout, QLabel, QSpinBox, QWidget, QApplication, QPushButton

from PyQt6.QtCore import Qt

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QIntValidator
from PyQt6.QtCore import QTimer, QTime
from threading import Thread
import numpy as np
import serial
from time import sleep, time
from functools import partial




class PullingProtocolWidget(QWidget):
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        self.setWindowTitle("Pulling Protocol")
        self.initUI()
        self.updateParameters()  # Initial update

    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel("Protocol controller")
        layout.addWidget(self.label)

        self.lowerLimitSpinBox = QSpinBox()
        self.lowerLimitSpinBox.setRange(0, 65535)
        self.lowerLimitSpinBox.valueChanged.connect(self.updateParameters)
        self.lowerLimitSpinBox.setValue(self.c_p['protocol_data'][3]*256 + self.c_p['protocol_data'][4])
        layout.addWidget(QLabel("Lower Limit:"))
        layout.addWidget(self.lowerLimitSpinBox)

        self.upperLimitSpinBox = QSpinBox()
        self.upperLimitSpinBox.setRange(0, 65535)
        self.upperLimitSpinBox.valueChanged.connect(self.updateParameters)
        self.upperLimitSpinBox.setValue(self.c_p['protocol_data'][1]*256 + self.c_p['protocol_data'][2])
        layout.addWidget(QLabel("Upper Limit:"))
        layout.addWidget(self.upperLimitSpinBox)

        self.stepSizeSpinBox = QSpinBox()
        self.stepSizeSpinBox.setRange(0, 65535)
        self.stepSizeSpinBox.valueChanged.connect(self.updateParameters)
        self.stepSizeSpinBox.setValue(self.c_p['protocol_data'][5]*256 + self.c_p['protocol_data'][6])
        layout.addWidget(QLabel("Step Size:"))
        layout.addWidget(self.stepSizeSpinBox)

        # Add toggle protocol button
        self.toggleProtocolButton = QPushButton("Toggle constant speed protocol")
        self.toggleProtocolButton.clicked.connect(self.toggleProtocol)
        self.toggleProtocolButton.setCheckable(True)
        self.toggleProtocolButton.setChecked(self.c_p['protocol_data'][0])
        layout.addWidget(self.toggleProtocolButton)

        self.setLayout(layout)

    def getParametersAs8BitArrays(self):
        lower_limit = self.lowerLimitSpinBox.value()
        upper_limit = self.upperLimitSpinBox.value()

        if upper_limit < lower_limit:
            print("Upper limit must be larger than lower limit!")
            lower_limit = upper_limit

        step_size = self.stepSizeSpinBox.value()

        # Function to split a 16-bit number into two 8-bit numbers
        split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF]

        return split_16_bit(lower_limit), split_16_bit(upper_limit), split_16_bit(step_size)

    def updateParameters(self):
        lower_limit, upper_limit, step_size = self.getParametersAs8BitArrays()
        print(f"Lower Limit: {lower_limit}, Upper Limit: {upper_limit}, Step Size: {step_size}")
        self.c_p['protocol_data'][1:3] = upper_limit
        self.c_p['protocol_data'][3:5] = lower_limit
        self.c_p['protocol_data'][5:7] = step_size

    def toggleProtocol(self):
        self.c_p['protocol_data'][0] = 1 - self.c_p['protocol_data'][0]
       #  print(f"Protocol toggled to {self.c_p['protocol_data'][0]}")
