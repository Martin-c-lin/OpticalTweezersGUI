from PyQt6.QtWidgets import (
 QVBoxLayout, QWidget, QLabel, QPushButton, QSlider, QFormLayout,QLineEdit, QHBoxLayout
)
from PyQt6.QtCore import Qt

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QIntValidator
from PyQt6.QtCore import QTimer
from threading import Thread
import numpy as np
import serial
from time import sleep, time
from functools import partial

class LaserControllerThread(Thread):
    def __init__(self, c_p):
        Thread.__init__(self)
        self.setDaemon(True)
        self.c_p = c_p


class LaserControllerWidget(QWidget):

    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        #layout = QVBoxLayout()
        
        self.current_A_edit_val = self.c_p['laser_A_current']
        self.current_B_edit_val = self.c_p['laser_B_current']
        self.laser_A_ser = None
        self.laser_B_ser = None
        self.connect_laser('A')
        self.connect_laser('B')
        self.init_laser_A()
        self.laser_A_on = False
        self.laser_B_on = False
        #self.label = QLabel("Laser controller")
        #layout.addWidget(self.label)
        #self.addLayout(layout)
        self.initUI()

    def init_laser_A(self):
        if self.laser_A_ser is None:
            return
        self.set_laser_A_current()
        # Turn on temperature controller
        message = "TCR\r"
        self.laser_A_ser.write(message.encode('utf-8'))

        # TODO query if the laser is on or off

    def initUI(self):
        layout = QVBoxLayout()

        # Laser A
        self.laserA_label = QLabel("Laser A", self)
        layout.addWidget(self.laserA_label)
        self.laserA_layout = QFormLayout()
        self.connectA_btn = QPushButton("Connect", self)
        connect_laserA = partial(self.connect_laser, 'A')
        self.connectA_btn.clicked.connect(connect_laserA)
        self.laserA_layout.addRow("Connect", self.connectA_btn)
        self.onOffA_btn = QPushButton("Turn ON", self)
        self.onOffA_btn.clicked.connect(self.onOff_laserA)
        self.laserA_layout.addRow("Output", self.onOffA_btn)

        # Craete a layout for the current edit box and the set current button
        self.currentA_layout = QHBoxLayout()
        # Create lineedit for current
        self.laserA_CurrentEdit = QLineEdit()
        self.laserA_CurrentEdit.setValidator(QIntValidator(0,400))
        self.laserA_CurrentEdit.setText(str(self.c_p['laser_A_current']))
        self.laserA_CurrentEdit.textChanged.connect(self.tempCurrentA)
        self.currentA_layout.addWidget(self.laserA_CurrentEdit)

        self.setCurrentButton = QPushButton("Set current", self)
        self.setCurrentButton.clicked.connect(self.set_laser_A_current)
        self.currentA_layout.addWidget(self.setCurrentButton)
        self.laserA_layout.addRow("Set current", self.currentA_layout)
        layout.addLayout(self.laserA_layout)

        # Laser B
        self.laserB_label = QLabel("Laser B", self)
        layout.addWidget(self.laserB_label)
        self.laserB_layout = QFormLayout()
        self.connectB_btn = QPushButton("Connect", self)
        connect_laserB = partial(self.connect_laser, 'B')

        self.connectB_btn.clicked.connect(connect_laserB)
        self.laserB_layout.addRow("Connect", self.connectB_btn)
        self.onOffB_btn = QPushButton("Turn ON", self)
        self.onOffB_btn.clicked.connect(self.onOff_laserB)
        self.laserB_layout.addRow("Power", self.onOffB_btn)

        # Craete a layout for the current edit box and the set current button
        self.currentB_layout = QHBoxLayout()
        # Create lineedit for current
        self.laserB_CurrentEdit = QLineEdit()
        self.laserB_CurrentEdit.setValidator(QIntValidator(0,400))
        self.laserB_CurrentEdit.setText(str(self.c_p['laser_B_current']))
        self.laserB_CurrentEdit.textChanged.connect(self.tempCurrentB)
        self.currentB_layout.addWidget(self.laserB_CurrentEdit)

        self.setCurrentButton = QPushButton("Set current", self)
        self.setCurrentButton.clicked.connect(self.set_laser_B_current)
        self.currentB_layout.addWidget(self.setCurrentButton)
        self.laserB_layout.addRow("Set current", self.currentB_layout)


        #self.currentB_slider = QSlider(Qt.Orientation.Horizontal, self)
        #current_laserB = partial(self.set_laser_currrent, 'B')
        #self.currentB_slider.valueChanged.connect(current_laserB)
        #self.laserB_layout.addRow("Current", self.currentB_slider)
        layout.addLayout(self.laserB_layout)

        self.setLayout(layout)

    def connect_laser(self, laser):
        if laser == 'A' and self.laser_A_ser is None:
            port = self.c_p['laser_A_port']
            print("Trying to connect laser A")
            try:
                self.laser_A_ser = serial.Serial(port=port,  # Port to connect to
                    baudrate=9600,  # Baud rate
                    bytesize=serial.EIGHTBITS,  # Data bits
                    parity=serial.PARITY_NONE,  # Parity
                    stopbits=serial.STOPBITS_ONE,  # Stop bits
                    timeout=2,  # Read timeout in seconds
                    xonxoff=False,  # Software flow control
                    rtscts=False,  # Hardware flow control (RTS/CTS)
                    write_timeout=2,  # Write timeout, this is not defined in the config but I assumed the same as the read timeout

                )
                print("Connected laser A")
            except Exception as E:
                print(E)
                print("its okay")

        if laser == 'B' and self.laser_B_ser is None:
            port = self.c_p['laser_B_port']
            try:
                self.laser_B_ser = serial.Serial(port=port,  # Port to connect to
                    baudrate=9600,  # Baud rate
                    bytesize=serial.EIGHTBITS,  # Data bits
                    parity=serial.PARITY_NONE,  # Parity
                    stopbits=serial.STOPBITS_ONE,  # Stop bits
                    timeout=2,  # Read timeout in seconds
                    xonxoff=False,  # Software flow control
                    rtscts=False,  # Hardware flow control (RTS/CTS)
                    write_timeout=2,  # Write timeout, this is not defined in the config but I assumed the same as the read timeout
                )
            except Exception as E:
                print(E)
    def tempCurrentA(self, current):
        self.current_A_edit_val = int(current)
    
    def tempCurrentB(self, current):
        self.current_B_edit_val = int(current)

    def set_laser_A_current(self):
        current = int(self.current_A_edit_val) # TODO get the value directly from the edit box 
        if int(current)>400 or int(current)<0:
            return
        self.c_p['laser_A_current'] = current
        if self.laser_A_ser is not None:
            message = "LCT" + str(self.current_A_edit_val) + "\r"
            self.laser_A_ser.write(message.encode('utf-8'))

    def set_laser_B_current(self):
        current = int(self.current_B_edit_val)
        if int(current)>400 or int(current)<0:
            return
        self.c_p['laser_B_current'] = current
        if self.laser_B_ser is not None:
            message = "LCT" + str(self.current_B_edit_val) + "\r"
            self.laser_B_ser.write(message.encode('utf-8'))

    def set_laser_currrent(self, laser, current):
        if int(current)>400 or int(current)<0:
            return
        message = "LCT"+str(current)+"\r"
        if laser == 'A' and self.laser_A_ser is not None:
            message = "LCT" + str(self.current_A_edit_val) + "\r"
            self.laser_A_ser.write(message.encode('utf-8'))
        elif laser == 'B' and self.laser_B_ser is not None:
            self.laser_B_ser.write(message.encode('utf-8'))

    def onOff_laserA(self):
        if self.laser_A_on:
            self.laser_A_on = False
            self.onOffA_btn.setText("Turn ON")
            self.laser_A_ser.write(b"LS\r")
            self.c_p['laser_A_on'] = False
        else:
            self.laser_A_on = True
            self.onOffA_btn.setText("Turn OFF")
            self.laser_A_ser.write(b"LR\r")
            self.c_p['laser_A_on'] = True


    def onOff_laserB(self):
        if self.laser_B_on:
            self.laser_B_on = False
            self.onOffB_btn.setText("Turn ON")
            self.laser_B_ser.write(b"LS\r")
            self.c_p['laser_B_on'] = False
        else:
            self.laser_B_on = True
            self.onOffB_btn.setText("Turn OFF")
            self.laser_B_ser.write(b"LR\r")
            self.c_p['laser_B_on'] = True
    """
    def start_laser(self, laser):
        if laser == 'A':
            self.laser_A_ser.write(b"LR\r")
        elif laser == 'B':
            self.laser_B_ser.write(b"LR\r")
    
    def stop_laser(self, laser):
        if laser == 'A':
            self.laser_A_ser.write(b"LS\r")
        elif laser == 'B':
            self.laser_B_ser.write(b"LS\r")
    """

    def disconnect_laser(self, laser):
        if laser == 'A' and self.laser_A_ser is not None:
            self.laser_A_ser.close()    
            self.laser_A_ser = None
        elif laser == 'B' and self.laser_B_ser is not None:
            self.laser_B_ser.close()
            self.laser_B_ser = None

    def closeEvent(self, event):
        self.disconnect_laser('A')
        self.disconnect_laser('B')
        event.accept()