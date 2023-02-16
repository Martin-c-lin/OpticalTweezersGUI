# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:16:22 2022

@author: marti
"""
import serial
import numpy as np
from threading import Thread
from time import sleep, time
import struct

def convert_array(array, start, spacing=9):
    """
    Converts a 16 bit array to numpy array. Works with fast_write_uart.
    Also reads delimitter bits
    Inputs:
      array - array to be converted to 16 bit numbers
      start - value put between the data values to indicate start/stop
      spacing - spacing between the start values in number of bytes
    """
    # TODO put also this in a thread
    if len(array)<1:
        return []
    for idx, a in enumerate(array):
        try:
            if a == start and array[idx+spacing] == start and array[idx+spacing*2] == start:
                # TODO add more checks for that the number is the same
                break
        except:
            return []
    res = []
    counter = idx+1
    while counter < len(array):
        for i in range(round((spacing-1)/2)):
            try:
                tmp = array[counter+i*2] * 256
                tmp += array[counter+i*2+1]
                res.append(tmp)
                # TODO fix too large numbers
                #if len(res) < 4:
                #    res.append(tmp)
                #elif np.abs(tmp-res[-1]) < 400:
                #    res.append(tmp)
            except IndexError as IE:
                pass            
        counter += spacing
    return np.array(res)

def fill_data_array(array, nbr_channels):
    # TODO rewrite so that no values are  missed, may have the number of readings of the different channels
    # Drifting otherwise
    nbr_elements = int(len(array)/nbr_channels) # Round down to avoid entries outside of array
    ch_data = np.zeros([nbr_channels, nbr_elements]) # TODO don't reinitialize this each time
    for i in range(nbr_channels):
        for j in range(nbr_elements):
            try:
                ch_data[i,j] = array[j*nbr_channels+i]
                # TODO UART IS REALLY UNREALIABLE FOR THIS!
                if j>0 and (ch_data[i,j-1] - ch_data[i,j]) > 20_000:
                    ch_data[i,j-1] = ch_data[i, j]
                elif j>0 and (ch_data[i,j] - ch_data[i,j-1]) > 20_000:
                    ch_data[i,j] = ch_data[i, j-1]
            except IndexError as IE:
                print(f"index_error idx: {j*nbr_channels+i}, array len {len(array)}")
                pass
    #TODO If there are not enough numbers it may start putting zeros...
    return ch_data
    

class PicReader(Thread):
    def __init__(self, c_p, data_channels, com_ch='COM3'):
        """
        

        Parameters
        ----------
        c_p : TYPE
            DESCRIPTION.
        data_channels : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        # TODO add time channel.
        # Maybe not have this as a separate thread?

        """

        Thread.__init__(self)
        self.c_p = c_p
        self.data_channels = data_channels
        self.serial_channel = None
        try:
            # try to open port. Timout helps make the reading more consistent
            self.serial_channel = serial.Serial(com_ch, baudrate=115200, timeout=50)
        except Exception as ex:
            print("No comm port")
            print(ex)
        print("Pic reader started")

    def run(self):
        spacing = len(self.c_p['pic_channels']) * 2 + 1
        while self.c_p['program_running']:
            
            if self.serial_channel is not None:
                # TODO change to read_until to avoid discarding in-between numbers
                raw = self.serial_channel.read(200)# 150 is smooth ish
                
                d = convert_array(raw, 62, spacing) # TODO make sure the separator byte is correct!
                
                arr = fill_data_array(d, len(self.c_p['pic_channels']))
                for idx, channel in enumerate(self.c_p['pic_channels']):
                    self.data_channels[channel].put_data(arr[idx, :])
                # TODO add time thing for PSD which looks a little better

                sleep(0.01)
            else:

                sleep(0.05)

class PicWriter(Thread):
    def __init__(self,c_p, serial_channel=None):
        Thread.__init__(self)
        self.serial_channel = serial_channel
        self.c_p = c_p
        self.send_data = np.uint8(np.zeros(8))
        self.send_data[0] = 255
        self.s = []
        
    def put_data(self):
        self.send_data[1] = self.c_p['motor_x_target_speed'] + 126
        self.send_data[2] = self.c_p['motor_y_target_speed'] + 126
        self.send_data[3] = self.c_p['motor_z_target_speed'] + 126
        self.send_data[4] = self.c_p['piezo_A'][0]
        self.send_data[5] = self.c_p['piezo_A'][1]
        self.send_data[6] = self.c_p['piezo_B'][0]
        self.send_data[7] = self.c_p['piezo_B'][1]
        self.send_data = np.uint8(self.send_data)
        self.s = struct.pack('!{0}B'.format(len(self.send_data)), *self.send_data)

    def run(self):
        while self.c_p['program_running']:
            if self.serial_channel is not None:
                self.put_data()
                # print(self.send_data[1])
                self.serial_channel.write(self.send_data)
            sleep(0.01)

























            