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

    def __init__(self, c_p, data_channels):
        Thread.__init__(self)
        self.setDaemon(True)
        self.c_p = c_p
        try:
            self.serial_channel = serial.Serial(self.c_p['COM_port'], baudrate=5000000, timeout=.001, write_timeout =0.001)
            self.c_p['minitweezers_connected'] = True
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
            try:
                self.serial_channel = serial.Serial(self.c_p['COM_port'], baudrate=5000000, timeout=.001, write_timeout =0.001)
                self.c_p['minitweezers_connected'] = True
                print("Reconnected")
            except Exception as ex:
                #print("No comm port!")
                self.serial_channel = None
                #print(ex)
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

        self.outdata[8] = (self.c_p['motor_x_target_speed']+32768)/256
        self.outdata[9] = (self.c_p['motor_x_target_speed']+32768)%256
        self.outdata[10] = (self.c_p['motor_y_target_speed']+32768)/256
        self.outdata[11] = (self.c_p['motor_y_target_speed']+32768)%256
        self.outdata[12] = (self.c_p['motor_z_target_speed']+32768)/256
        self.outdata[13] = (self.c_p['motor_z_target_speed']+32768)%256

        # Send the piezo voltages
        self.outdata[14] = int(self.c_p['piezo_A'][0]/256)
        self.outdata[15] = int(self.c_p['piezo_A'][0]%256)
        self.outdata[16] = int(self.c_p['piezo_A'][1]/256)
        self.outdata[17] = int(self.c_p['piezo_A'][1]%256)

        self.outdata[18] = int(self.c_p['piezo_B'][0]/256)
        self.outdata[19] = int(self.c_p['piezo_B'][0]%256)
        self.outdata[20] = int(self.c_p['piezo_B'][1]/256)
        self.outdata[21] = int(self.c_p['piezo_B'][1]%256)

        self.outdata[22] = int(self.c_p['motor_travel_speed']/256)
        self.outdata[23] = int(self.c_p['motor_travel_speed']%256)

        # Rest of the data is reserved for future use

        # End bytes for the portenta
        self.outdata[30] = 125
        self.outdata[31] = 125
        try:
            self.serial_channel.write(self.outdata)
        except serial.serialutil.SerialTimeoutException as e:
            pass
        except serial.serialutil.SerialException as e:
            print(f"Serial exception: {e}")
            self.serial_channel = None
            self.c_p['minitweezers_connected'] = False

    def get_data(self):
        if self.serial_channel is None:
            #print("No serial channel!")
            return
        try:
            raw_data = self.serial_channel.read(64)
        except serial.serialutil.SerialException as e:
            print(f"Serial exception: {e}")
            self.serial_channel = None
            self.c_p['minitweezers_connected'] = False
            return
        self.serial_channel.reset_input_buffer()
        # Check if the correct amount of bytes are received
        if len(raw_data) != 64:
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

    def combine_bytes(self, high_byte, low_byte):
        return (high_byte << 8) | low_byte

    def get_data_new(self):
        if self.serial_channel is None:
            return

        try:
            bytes_to_read = self.serial_channel.in_waiting
            raw_data = self.serial_channel.read(bytes_to_read)
        except serial.serialutil.SerialException as e:
            print(f"Serial exception: {e}")
            self.serial_channel = None
            self.c_p['minitweezers_connected'] = False
            return

        # Process the received data in chunks of 64 bytes
        for i in range(0, len(raw_data), 64):
            chunk = raw_data[i:i+64]

            # Check if the correct amount of bytes are received
            if len(chunk) != 64:
                continue

            # Check if the start bytes are correct
            if chunk[0] != 123 or chunk[1] != 123:
                print('Wrong start bytes')
                continue

            for idx, channel in enumerate(self.c_p['pic_channels']):
                number = self.combine_bytes(chunk[2 * idx + 2], chunk[2 * idx + 3])
                self.data_channels[channel].put_data([number])

    def connect_port(self):
        # TODO cannot restart program without restarting the portenta.
        pass

    def move_to_location(self):
        # This should be done in a different thread maybe.
        dist_x = self.c_p['minitweezers_target_pos'][0] - self.data_channels['Motor_x_pos'].get_data(1)[0]
        dist_y = self.c_p['minitweezers_target_pos'][1] - self.data_channels['Motor_y_pos'].get_data(1)[0]
        dist_z = self.c_p['minitweezers_target_pos'][2] - self.data_channels['Motor_z_pos'].get_data(1)[0]

        if dist_x**2>100:
            self.c_p['motor_x_target_speed'] = self.c_p['motor_travel_speed'] if dist_x > 0 else -self.c_p['motor_travel_speed']
        else:
            self.c_p['motor_x_target_speed'] = 0

        if dist_y**2>100:
            self.c_p['motor_y_target_speed'] = self.c_p['motor_travel_speed'] if dist_y > 0 else -self.c_p['motor_travel_speed']
        else:
            self.c_p['motor_y_target_speed'] = 0

        if dist_z**2>100:
            self.c_p['motor_z_target_speed'] = self.c_p['motor_travel_speed'] if dist_z > 0 else -self.c_p['motor_travel_speed']
        else:
            self.c_p['motor_z_target_speed'] = 0

        if dist_x**2+dist_y**2+dist_z**2<300:
            self.c_p['motor_x_target_speed'] = 0
            self.c_p['motor_y_target_speed'] = 0
            self.c_p['motor_z_target_speed'] = 0
            self.c_p['move_to_location'] = False

        
    def run(self):

        while self.c_p['program_running']:
            if self.c_p['move_to_location']:
                self.move_to_location()
            self.send_data()
            self.get_data_new()
            sleep(1e-7)
        if self.serial_channel is not None:
            self.serial_channel.close()
            