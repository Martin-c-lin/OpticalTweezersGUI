from PyQt6.QtWidgets import (
 QCheckBox, QVBoxLayout, QWidget, QLabel, QTableWidget, QTableWidgetItem
)

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QTimer
from threading import Thread
import numpy as np
import serial
from time import sleep, time
import timeit


class portentaReaderThread(Thread):
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
    def run(self):

        while self.c_p['program_running']:
            self.read_data_to_channels_ultra()
        if self.serial_channel is not None:
            self.serial_channel.close()
            print('Serial connection to minitweezers closed')

    def read_data(self):
        """
        Reads the data from the serial port and returns it as a numpy array.

        """
        chunk_length = 32 # Number of 16 bit numbers sent each time.
        if self.serial_channel is None:
            return None
        try:
            bytes_to_read = self.serial_channel.in_waiting
            if bytes_to_read < chunk_length:
                return None
            raw_data = self.serial_channel.read(bytes_to_read)
        except serial.serialutil.SerialException as e:
            print(f"Serial exception: {e}")
            self.serial_channel = None
            self.c_p['minitweezers_connected'] = False
            return None
        if raw_data[0] != 123 or raw_data[1] != 123:
                print('Wrong start bytes')
                return None
        return np.frombuffer(raw_data, dtype=np.uint16) # Immediately does the correct conversion

    def read_data_to_channels_ultra(self, chunk_length=256):
        """
        Reads data from the arduino assuming that it has been ordered into channels and puts it into
        the correct data channels.
        First there are two control bytes. If these are not correct the data is discarded.
        """

        chunk_length = 256 # Number of 16 bit numbers sent each time.
        numbers = self.read_data()
        zero_offset = 1 + len(self.c_p['single_sample_channels'])
        if numbers is None:
            sleep(0.001)
            return
        L = len(numbers)
        nbr_chunks = int(L/chunk_length)
        nbr_channels = len(self.c_p['multi_sample_channels'])

        # Are a few empty bytes in the end which need to be accounted for
        unused_indices = (chunk_length-zero_offset)
        unused_indices %= nbr_channels

        # Single sample channels
        for idx, channel in enumerate(self.c_p['single_sample_channels']):
            if channel in self.c_p['offset_channels']:
                self.data_channels[channel].put_data(numbers[idx+1:L:chunk_length].astype(int)-32768)
            else:
                self.data_channels[channel].put_data(numbers[idx+1:L:chunk_length].astype(int))
        # Multi sample channels
        for idx, channel in enumerate(self.c_p['multi_sample_channels']):
            if channel == 'T_time':
                pass
            else:
                starts = zero_offset + np.arange(nbr_chunks) * chunk_length + idx
                stops = chunk_length * np.arange(1, nbr_chunks+1) - unused_indices

                indices = np.array([np.arange(start, stop, nbr_channels) for start, stop in zip(starts, stops)]).flatten()
                if channel in self.c_p['offset_channels']:
                    self.data_channels[channel].put_data(numbers[indices].astype(int)-32768)
                else:
                    self.data_channels[channel].put_data(numbers[indices].astype(int))
        # Add the time channel... Maybe have this as a separte function?
        data_length = len(indices)
        low = self.data_channels['Time_micros_low'].get_data(data_length).astype(np.uint32)
        high = self.data_channels['Time_micros_high'].get_data(data_length).astype(np.uint32)
        self.data_channels['T_time'].put_data((high << 16) | low)


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
        self.outdata = np.uint8(np.zeros(48))
        self.indata = np.uint8(np.zeros(64))
        self.start_time = time()
        nbr_multisamples = 1
        self.channel_array = self.c_p['single_sample_channels'].copy()
        for _ in range(nbr_multisamples):
            self.channel_array.extend(self.c_p['multi_sample_channels'])

    def send_data_fast(self):
        """
        Sends data to the portenta.
        """
        if self.serial_channel is None:
            try:
                self.serial_channel = serial.Serial(self.c_p['COM_port'], baudrate=5000000, timeout=.001, write_timeout=0.001)
                self.c_p['minitweezers_connected'] = True
                print("Reconnected")
            except Exception as ex:
                self.serial_channel = None
            return

        self.serial_channel.reset_output_buffer()

        # Start bytes for the portenta
        self.outdata[0:2] = [123, 123]

        # Send the target position and speed
        for i in range(3):
            self.outdata[2 + i * 2] = self.c_p['minitweezers_target_pos'][i] >> 8
            self.outdata[3 + i * 2] = self.c_p['minitweezers_target_pos'][i] & 0xFF
            self.outdata[8 + i * 2] = (self.c_p[f'motor_{["x", "y", "z"][i]}_target_speed'] + 32768) >> 8
            self.outdata[9 + i * 2] = (self.c_p[f'motor_{["x", "y", "z"][i]}_target_speed'] + 32768) & 0xFF

        # Send the piezo voltages
        for i in range(2):
            self.outdata[14 + i * 2] = self.c_p['piezo_A'][i] >> 8
            self.outdata[15 + i * 2] = self.c_p['piezo_A'][i] & 0xFF
            self.outdata[18 + i * 2] = self.c_p['piezo_B'][i] >> 8
            self.outdata[19 + i * 2] = self.c_p['piezo_B'][i] & 0xFF
        #print(self.outdata[14:21]) Correct
        self.outdata[22] = self.c_p['motor_travel_speed'][0] >> 8
        self.outdata[23] = self.c_p['motor_travel_speed'][0] & 0xFF
        self.outdata[24] = self.c_p['portenta_command_1']
        self.c_p['portenta_command_1'] = 0
        self.outdata[25] = self.c_p['portenta_command_2']
        # End bytes for the portenta
        self.outdata[26] = self.c_p['PSD_means'][0] >> 8
        self.outdata[27] = self.c_p['PSD_means'][0] & 0xFF
        self.outdata[28] = self.c_p['PSD_means'][1] >> 8
        self.outdata[29] = self.c_p['PSD_means'][1] & 0xFF

        self.outdata[30] = self.c_p['PSD_means'][2] >> 8
        self.outdata[31] = self.c_p['PSD_means'][2] & 0xFF
        self.outdata[32] = self.c_p['PSD_means'][3] >> 8
        self.outdata[33] = self.c_p['PSD_means'][3] & 0xFF

        self.outdata[34] = self.c_p['blue_led']

        self.outdata[35:] = self.c_p['protocol_data']

        try:
            self.serial_channel.write(self.outdata)
        except serial.serialutil.SerialTimeoutException as e:
            pass
        except serial.serialutil.SerialException as e:
            print(f"Serial exception: {e}")
            self.serial_channel = None
            self.c_p['minitweezers_connected'] = False

    def calc_quote_fast(self, quote, channel1, channel2, chunk_length):
        D1 = self.data_channels[channel1].get_data(chunk_length)
        D2 = np.copy(self.data_channels[channel2].get_data(chunk_length).astype(float))
        D2[D2==0] = np.inf
        self.data_channels[quote].put_data(D1/D2)

    def calculate_quotes_fast(self, chunk_length):
        # TODO suspect that this is too slow and maybe will make things go bonkers slow.
        self.calc_quote_fast('F_A_X','PSD_A_F_X','PSD_A_F_sum', chunk_length)
        self.calc_quote_fast('F_A_Y','PSD_A_F_Y','PSD_A_F_sum', chunk_length)
        self.calc_quote_fast('F_B_X','PSD_B_F_X','PSD_B_F_sum', chunk_length)
        self.calc_quote_fast('F_B_Y','PSD_B_F_Y','PSD_B_F_sum', chunk_length)
        self.calc_quote_fast('Photodiode/PSD SUM A','Photodiode_A','PSD_A_F_sum', chunk_length)
        self.calc_quote_fast('Photodiode/PSD SUM B','Photodiode_B','PSD_B_F_sum', chunk_length)

    def read_data(self):
        """
        Reads the data from the serial port and returns it as a numpy array.

        """
        chunk_length = 32 # Number of 16 bit numbers sent each time.
        if self.serial_channel is None:
            return None
        try:
            bytes_to_read = self.serial_channel.in_waiting
            if bytes_to_read < chunk_length:
                return None
            raw_data = self.serial_channel.read(bytes_to_read)
        except serial.serialutil.SerialException as e:
            print(f"Serial exception: {e}")
            self.serial_channel = None
            self.c_p['minitweezers_connected'] = False
            return None
        if raw_data[0] != 123 or raw_data[1] != 123:
                print('Wrong start bytes')
                return None
        return np.frombuffer(raw_data, dtype=np.uint16) # Immediately does the correct conversion

    def read_data_to_channels_experimental(self, chunk_length=256):
        numbers = self.read_data()
        if numbers is None:
            sleep(0.001)
            return
        numbers = numbers.astype(int)

        L = len(numbers)
        nbr_chunks = int(L/chunk_length)
        nbr_channels = len(self.c_p['multi_sample_channels'])

        zero_offset = 1 + len(self.c_p['single_sample_channels'])
        unused_indices = (chunk_length-zero_offset) % nbr_channels

        # Single sample channels
        for idx, channel in enumerate(self.c_p['single_sample_channels']):
            data = numbers[idx+1:L:chunk_length]
            if channel in self.c_p['offset_channels']:
                data -= 32768
            self.data_channels[channel].put_data(data)
        data_length = len(data)

        # Compute starts and stops once
        base_starts = zero_offset + np.arange(nbr_chunks) * chunk_length
        base_stops = chunk_length * np.arange(1, nbr_chunks+1) - unused_indices
        indices = np.concatenate([np.arange(start, stop, nbr_channels) for start, stop in zip(base_starts, base_stops)])

        # Multi sample channels
        for idx, channel in enumerate(self.c_p['multi_sample_channels']):
            if channel == 'T_time':
                continue
            data = numbers[indices+idx]
            if channel in self.c_p['offset_channels']:
                data -= 32768
            self.data_channels[channel].put_data(data)

        # Add the time channel... Maybe have this as a separate function?
        data_length = len(indices)
        low = self.data_channels['Time_micros_low'].get_data(data_length).astype(np.uint32)
        high = self.data_channels['Time_micros_high'].get_data(data_length).astype(np.uint32)
        T_time = (high << 16) | low
        self.data_channels['T_time'].put_data(T_time)#((high << 16) | low)
        # Use the first of the time samples of the PSD is roughyl equivalent to the motor time
        #TODO: This does not appear to be working correctly!
        # Error appears to have been that the last index of the data is not retrieved correctly. I.e
        #self.data_channels['Motor time'].put_data(self.data_channels['T_time'].get_data_spaced(nbr_chunks,14)) # 14 data points per chunk
        self.data_channels['Motor time'].put_data(T_time[::14]) # 14 data points per chunk
        self.calculate_quotes_fast(data_length) # This works but is a bit slow...

    def move_to_location(self):
        # TODO This should be done in a different thread maybe.
        dist_x = self.c_p['minitweezers_target_pos'][0] - self.data_channels['Motor_x_pos'].get_data(1)[0]
        dist_y = self.c_p['minitweezers_target_pos'][1] - self.data_channels['Motor_y_pos'].get_data(1)[0]
        dist_z = self.c_p['minitweezers_target_pos'][2] - self.data_channels['Motor_z_pos'].get_data(1)[0]

        # Adjust speed depending on how far we are going
        if dist_x**2 >10_000:
            self.c_p['motor_travel_speed'][0] = 25000
        else:
            self.c_p['motor_travel_speed'][0] = 1500

        if dist_y**2 >10_000:
            self.c_p['motor_travel_speed'][1] = 25000
        else:
            self.c_p['motor_travel_speed'][1] = 1500

        # Changed the signs of this function
        if dist_x**2>100:
            self.c_p['motor_x_target_speed'] = -self.c_p['motor_travel_speed'][0] if dist_x > 0 else self.c_p['motor_travel_speed'][0]
        else:
            self.c_p['motor_x_target_speed'] = 0

        if dist_y**2>100:
            self.c_p['motor_y_target_speed'] = self.c_p['motor_travel_speed'][1] if dist_y > 0 else -self.c_p['motor_travel_speed'][1]
        else:
            self.c_p['motor_y_target_speed'] = 0
        """
        # Z movement is dangerous, wait with that.
        if dist_z**2>100:
            self.c_p['motor_z_target_speed'] = -self.c_p['motor_travel_speed'] if dist_z > 0 else self.c_p['motor_travel_speed']
        else:
            self.c_p['motor_z_target_speed'] = 0
        """
        if dist_x**2+dist_y**2<200:
            self.c_p['motor_x_target_speed'] = 0
            self.c_p['motor_y_target_speed'] = 0
            self.c_p['motor_z_target_speed'] = 0
            self.c_p['move_to_location'] = False


    def run(self):

        while self.c_p['program_running']:
            if self.c_p['move_to_location']:
                self.move_to_location()
            self.send_data_fast()
            self.read_data_to_channels_experimental()
            sleep(1e-4)

        if self.serial_channel is not None:
            self.serial_channel.close()
            print('Serial connection to minitweezers closed')
