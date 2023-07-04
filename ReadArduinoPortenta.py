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

        self.outdata[22] = int(self.c_p['motor_travel_speed'][0]/256) # TODO Check that these really are needed
        self.outdata[23] = int(self.c_p['motor_travel_speed'][0]%256)

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
            # if channel not in ['T_time','Time_micros_high','Time_micros_low','Photodiode_A','Photodiode_B']

            self.data_channels[channel].put_data([number])

    def combine_bytes(self, high_byte, low_byte):
        return (high_byte << 8) | low_byte

    def calc_quote(self, quote, channel1, channel2):
        D1 = self.data_channels[channel1].data[self.data_channels[channel1].index-1]   #get_data_spaced(1)[0] # Is data.[index] significantly faster?
        D2 = self.data_channels[channel2].data[self.data_channels[channel2].index-1]  #get_data_spaced(1)[0]
        if D2 != 0:
            self.data_channels[quote].put_data([D1/D2])
        else:
            self.data_channels[quote].put_data([0])

    def calculate_quotes(self):
        self.calc_quote('F_A_X','PSD_A_F_X','PSD_A_F_sum')
        self.calc_quote('F_A_Y','PSD_A_F_Y','PSD_A_F_sum')
        self.calc_quote('F_B_X','PSD_B_F_X','PSD_B_F_sum')
        self.calc_quote('F_B_Y','PSD_B_F_Y','PSD_B_F_sum')
        self.calc_quote('Photodiode/PSD SUM A','Photodiode_A','PSD_A_F_sum')
        self.calc_quote('Photodiode/PSD SUM B','Photodiode_B','PSD_B_F_sum')


    def calc_quote_fast(self, quote, channel1, channel2, chunk_length):
        D1 = self.data_channels[channel1].get_data(chunk_length)
        D2 = np.copy(self.data_channels[channel2].get_data(chunk_length).astype(float))
        D2[D2==0] = np.inf
        self.data_channels[quote].put_data(D1/D2)
        
    def calculate_quotes_fast(self, chunk_length):
        self.calc_quote_fast('F_A_X','PSD_A_F_X','PSD_A_F_sum',chunk_length)
        self.calc_quote_fast('F_A_Y','PSD_A_F_Y','PSD_A_F_sum',chunk_length)
        self.calc_quote_fast('F_B_X','PSD_B_F_X','PSD_B_F_sum',chunk_length)
        self.calc_quote_fast('F_B_Y','PSD_B_F_Y','PSD_B_F_sum',chunk_length)
        self.calc_quote_fast('Photodiode/PSD SUM A','Photodiode_A','PSD_A_F_sum',chunk_length)
        self.calc_quote_fast('Photodiode/PSD SUM B','Photodiode_B','PSD_B_F_sum',chunk_length)

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
        start = timeit.default_timer()
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
                if channel in self.c_p['offset_channels']:
                    self.data_channels[channel].put_data([number-32768])
                elif channel == 'Time_micros_high':# ['T_time','Time_micros_high','Time_micros_low']:
                    T = number * 2**16
                elif channel == 'Time_micros_low':
                    T += number
                    self.data_channels['T_time'].put_data([T])
                elif channel == 'T_time':
                    pass
                else:
                    self.data_channels[channel].put_data([number])
            # Check if message was received
            if self.data_channels['message'] != 0:
                self.c_p['portenta_command_1'] = 0
            #Addding computers own time
            self.data_channels['Time'].put_data([time() - self.start_time])
            #self.calculate_quotes()
        print(f"Time for calculation {(timeit.default_timer()-start)*1e6} microseconds, {len(raw_data)/64}.")

    def combine_bytes(self, high_byte, low_byte):
        return (high_byte << 8) | low_byte
    
    def get_data_fast(self):
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
        if len(raw_data) < 64:
            return
        if raw_data[0] != 123 or raw_data[1] != 123:
                print('Wrong start bytes')
                return
        # start = timeit.default_timer()
        
        # Process the received data using indexing
        nbr_channels = len(self.c_p['pic_channels'])
        nbr_chunks = int(len(raw_data)/64)
        ch_indices = np.array([2 * idx + 2 for idx in range(nbr_channels)])
        data_indices = np.array([ch_indices+64*i for i in range(nbr_chunks)]).flatten()
        raw_data = np.frombuffer(raw_data, dtype=np.uint8)
        high_bytes = raw_data[data_indices].astype(np.uint16)
        low_bytes = raw_data[data_indices+1].astype(np.uint16) # np.uint16
        numbers = (high_bytes << 8) | low_bytes
        pic_indices = np.linspace(0,nbr_channels * (nbr_chunks-1) ,nbr_chunks, dtype=int)

        for idx, channel in enumerate(self.c_p['pic_channels']):
            ch_data = numbers[pic_indices + idx]
            if channel in self.c_p['offset_channels']:
                self.data_channels[channel].put_data(ch_data.astype(np.int)-32768)
            elif channel == 'Time_micros_high':
                T = ch_data.astype(np.int32) * 2**16
            elif channel == 'Time_micros_low':
                T += ch_data
                self.data_channels['T_time'].put_data(T)
            elif channel == 'T_time':
                pass
            else:
                self.data_channels[channel].put_data(ch_data)
            #self.data_channels[channel].put_data(ch_data)
        if self.data_channels['message'] != 0:
                self.c_p['portenta_command_1'] = 0 # TODO check that this matches the previous code
            #Addding computers own time
        # self.data_channels['Time'].put_data([time() - self.start_time])# TODO fix this
        self.calculate_quotes_fast(nbr_chunks) # TODO fix this
        # print(f"Time for calculation {(timeit.default_timer()-start)*1e6} microseconds, {nbr_chunks}.")


    def connect_port(self):
        # TODO cannot restart program without restarting the portenta.
        pass

    def move_to_location(self):
        # TODO This should be done in a different thread maybe.
        dist_x = self.c_p['minitweezers_target_pos'][0] - self.data_channels['Motor_x_pos'].get_data(1)[0]
        dist_y = self.c_p['minitweezers_target_pos'][1] - self.data_channels['Motor_y_pos'].get_data(1)[0]
        dist_z = self.c_p['minitweezers_target_pos'][2] - self.data_channels['Motor_z_pos'].get_data(1)[0]

        # Adjust speed depending on how far we are going
        if dist_x**2 >10_000:
            self.c_p['motor_travel_speed'][0] = 5000
        else:
            self.c_p['motor_travel_speed'][0] = 500

        if dist_y**2 >10_000:
            self.c_p['motor_travel_speed'][1] = 5000
        else:
            self.c_p['motor_travel_speed'][1] = 500

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
        if dist_x**2+dist_y**2<200: #+dist_z**2<300: Removed z-dependence
            self.c_p['motor_x_target_speed'] = 0
            self.c_p['motor_y_target_speed'] = 0
            self.c_p['motor_z_target_speed'] = 0
            self.c_p['move_to_location'] = False

        
    def run(self):

        while self.c_p['program_running']:
            if self.c_p['move_to_location']:
                self.move_to_location()
            # Idea only send data when there is actually something to send, don't send while
            # recording an experiment to get sampling more consistent.
            self.send_data_fast() # Have also an older well tested but probably slower version.
            # self.get_data_new() # Works reliably but struggles when the sample rate exceeds like 4khz
            self.get_data_fast()
            sleep(2e-3) # Increased here from 1e-5
        if self.serial_channel is not None:
            self.serial_channel.close()
            print('Serial connection to minitweezers closed')
            