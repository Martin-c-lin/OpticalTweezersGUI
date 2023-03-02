from PyQt6.QtWidgets import (
 QCheckBox, QVBoxLayout, QWidget, QLabel, QTableWidget, QTableWidgetItem
)

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QTimer
from threading import Thread
import numpy as np
import serial
from time import sleep

class PortentaComms(Thread):

    def __init__(self, c_p, data_channels, com_port):
        Thread.__init__(self)
        self.setDaemon(True)
        self.c_p = c_p
        try:
            self.serial_channel = serial.Serial(com_port, baudrate=5000000, timeout=.001, write_timeout =0.001)
        except Exception as ex:
            print("No comm port!")
            self.serial_channel = None
            print(ex)
        print('Serial channel opened')
        self.data_channels = data_channels
        self.outdata = np.uint8(np.zeros(32))
        self.indata = np.uint8(np.zeros(64))

    def send_data(self):
        """
        Sends data to the portenta.
        """
        if self.serial_channel is None:
            return
        self.serial_channel.reset_output_buffer()

        # Start bytes for the portenta
        self.outdata[0] = 123
        self.outdata[1] = 123


        # TODO make sure whether the divisor is 256 or 256
        # Send the target position and speed
        self.outdata[2] = self.c_p['minitweezers_target_pos'][0]/256
        self.outdata[3] = self.c_p['minitweezers_target_pos'][0]%256
        self.outdata[4] = self.c_p['minitweezers_target_pos'][1]/256
        self.outdata[5] = self.c_p['minitweezers_target_pos'][1]%256
        self.outdata[6] = self.c_p['minitweezers_target_pos'][2]/256
        self.outdata[7] = self.c_p['minitweezers_target_pos'][2]%256

        self.outdata[8] = (self.c_p['motor_x_target_speed']+130)/256
        self.outdata[9] = (self.c_p['motor_x_target_speed']+130)%256
        self.outdata[10] = (self.c_p['minitweezers_target_speed'][1]+120)/256
        self.outdata[11] = (self.c_p['minitweezers_target_speed'][1]+120)%256
        self.outdata[12] = self.c_p['minitweezers_target_speed'][2]/256
        self.outdata[13] = self.c_p['minitweezers_target_speed'][2]%256

        # Send the piezo voltages
        self.outdata[14] = int(self.c_p['piezo_A'][0]/256)
        self.outdata[15] = int(self.c_p['piezo_A'][0]%256)
        self.outdata[16] = int(self.c_p['piezo_A'][1]/256)
        self.outdata[17] = int(self.c_p['piezo_A'][1]%256)

        self.outdata[18] = int(self.c_p['piezo_B'][0]/256)
        self.outdata[19] = int(self.c_p['piezo_B'][0]%256)
        self.outdata[20] = int(self.c_p['piezo_B'][1]/256)
        self.outdata[21] = int(self.c_p['piezo_B'][1]%256)

        # Rest of the data is reserved for future use

        # End bytes for the portenta
        self.outdata[30] = 125
        self.outdata[31] = 125
        try:
            self.serial_channel.write(self.outdata)
            #print(self.outdata[15])
        except serial.serialutil.SerialTimeoutException as e:
            pass

    def get_data(self):
        if self.serial_channel is None:
            return
        raw_data = self.serial_channel.read(64)
        self.serial_channel.reset_input_buffer()
        # Check if the correct amount of bytes are received
        if len(raw_data) != 64:
            #print(f'Wrong amount of bytes received {len(raw_data)}')
            return
        self.indata = raw_data

        # Check if the start bytes are correct
        if self.indata[0] != 123 or self.indata[1] != 123:
            print('Wrong start bytes')
            return
        for idx, channel in enumerate(self.c_p['pic_channels']):
            number = self.indata[2*idx+2] * 256 + self.indata[2*idx+3] # Is it 256 or 256?
            # TODO don't put one number at a time
            self.data_channels[channel].put_data([number])

    def connect_port(self):
        # TODO cannot restart program without restarting the portenta.
        pass

    def run(self):

        while self.c_p['program_running']:
            self.send_data()
            self.get_data()
            #sleep(1e-6)
        if self.serial_channel is not None:
            self.serial_channel.close()
            