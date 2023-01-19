# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:48:24 2022

@author: marti
"""

import numpy as np
from PIL import Image
from queue import Queue


def default_c_p():
    """
    Initiates the control parameters to default values.

    Returns
    -------
    c_p : TYPE
        DESCRIPTION. Dictionary containing the important parameters and values
        that need to be shared between different threads.

    """
    c_p = {
           'program_running': True,
           'mouse_params': [0, 0, 0, 0, 0],
           # Camera c_p
           'image': np.ones([500, 500]),#, 1]),
           'image_idx': 0, # Index of snapshot image
           'color': "mono",  # Options are mono and color # Maybe add bit depth too
           'new_settings_camera': [False, None],
           'camera_width': 1920,
           'camera_height': 1080,
           'recording': False,
           'exposure_time': 5000,
           'fps': 50,  # Frames per second of camera
           'video_name': 'Video',
           'video_format': 'avi',
           'image_format': 'png',
           'image_gain': 1,
           'image_offset': 0,
           'AOI':[0,1000,0,1000], # Area of interest of camera
           'recording_path': '../Example data/',
           'bitrate': '30000000', #'300000000',
           'frame_queue': Queue(maxsize=2_000_000),  # Frame buffer essentially
           'image_scale': 1,
           # Temperature c_p
           'temperature_output_on':False,
           
           # Piezo c_p
           'piezo_targets': [10,10,10],
           'piezo_pos': [10,10,10],
           
           # PIC reader c_p
           # TODO add all the necessary channels here. PSDs and motors.
           'pic_channels':['Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos', 
                           'Motor_x_speed', 'Motor_y_speed', 'Motor_z_speed',
                           'PSD_pA_X_diff', 'PSD_pA_Y_diff', 'PSD_pA_sum'],
           
           # Motor output
           'motor_x_target_speed': 0,
           'motor_y_target_speed': 0,
           'motor_z_target_speed': 0,
           }
    return c_p


from dataclasses import dataclass
"""
# Old implementation of datachannel class
@dataclass
class DataChannel:
    # TODO change data to a queue 
    name: str
    unit: str
    data: np.array
    saving_toggled: bool = False
    max_len: int = 1000_000
    index: int = 1
    
    def put_data(self, d):
        # Check that it is correct length
        if len(d)==1:
            print(d)
        if len(d) < 2:
            # TODO fix error when d = 1
            return
        L = len(self.data) + len(d)
        if L < self.max_len:
            tmp = np.zeros(L)#[self.data[:], d])
            tmp[:len(self.data)] = self.data
            tmp[-len(d):] = d
            self.data = tmp
        else:
            tmp = np.zeros(self.max_len)
            diff = self.max_len - len(self.data)
            tmp[:-len(d)] = self.data[len(d)-diff:] # Index error here before
            tmp[-len(d):] = d
            self.data = tmp
 """  
@dataclass
class DataChannel:
    # New faster implementation of data channel class
    name: str
    unit: str
    data: np.array
    saving_toggled: bool = False
    max_len: int = 1000_000
    index: int = 1
    full: bool = False # True if all elements have been filled
    max_retrivable: int = 1
    
    def put_data(self, d):
        # Check that it is correct length
        try:
            if len(d) > self.max_len:
            # TODO throw an error
                return
        except TypeError:
            # Someone very naughty sent something other than an array
            d = [d]
        if len(self.data) < self.max_len:
            tmp = np.zeros(self.max_len)
            tmp[:len(self.data)] = self.data
            self.index = len(self.data)
            self.data = tmp

        if self.index+len(d) < self.max_len:
            self.data[self.index:self.index+len(d)] = d
            self.index += len(d)
            if not self.full:
                self.max_retrivable = self.index 
        else:
            end_points = self.max_len - self.index
            self.data[-end_points:] = d[:end_points]
            self.index = len(d) - end_points
            self.data[:self.index] = d[end_points:]
            self.full = True
            self.max_retrivable = self.max_len

    def get_data(self, nbr_points):
        nbr_points = min(nbr_points, self.max_retrivable)
        diff = self.index-nbr_points
        if diff > 0:
            # Simple case
            ret = self.data[diff:self.index]
        else:
            ret = np.concatenate([self.data[diff:], self.data[:self.index]]).ravel()
        if not len(ret) == nbr_points:
            return None
        return ret

    def get_data_spaced(self, nbr_points, spacing=1):
        """
        Function that returns nbr_points data with spacing as specified by spacing.
        If there are not enough data it will return a lesser number of datapoints
        keeping the specified spacing.

        Parameters
        ----------
        nbr_points : TYPE int
            DESCRIPTION. Maximum number of points to retrieve
        spacing : TYPE, optional int
            DESCRIPTION. The default is 1. Number of points between each desired
            data points

        Returns
        -------
        ret : TYPE np array
            DESCRIPTION. Datapoints of channel

        """
        
        nbr_points = min(nbr_points, self.max_retrivable)
        diff = self.index-nbr_points
        final = self.index
        start = final - (final % spacing) - (nbr_points * spacing)
        if diff > 0:
            # Simple case
            ret = self.data[start:final:spacing]
        else:
            last = (nbr_points*spacing+start)
            ret = np.concatenate([self.data[start:-1:spacing],
                                  self.data[final%spacing:last:spacing]]).ravel()

        return ret

def get_data_dicitonary():
    # TODO replace the entries with data channels
    data = {
            'Time':[0],
            'X-force':[0],
            'Y-force':[0],
            'Z-force':[0],
            'Motor_position':[0],
            'X-position':[0],
            'Y-position':[0],
            'Z-position':[0],
            'Temperature':[0],
            'T_time':[0], # Todo have just a single time
            }
    return data

def get_data_dicitonary_new():
    """
    ['PSD_pA_x1','bits'],
    ['PSD_pA_x2','bits'],
    ['PSD_pA_y1','bits'],
    ['PSD_pA_y2','bits'],
    """
    # TODO replace the entries with data channels
    data = [['Time','(s)'],
    ['X-force','(pN)'],
    ['Y-force','(pN)'],
    ['Z-force','(pN)'],
    ['Motor_position','ticks'],
    ['X-position','(microns)'],
    ['Y-position','(microns)'],
    ['Z-position','(microns)'],
    ['Temperature', 'Celsius'],
    ['Motor_x_pos', 'ticks'],
    ['Motor_y_pos','ticks'],
    ['Motor_z_pos', 'ticks'],
    ['Motor_x_speed','ticks/s'],
    ['Motor_y_speed','ticks/s'],
    ['Motor_z_speed','ticks/s'],
    ['PSD_pA_sum','bits'],
    ['PSD_pA_X_diff','bits'],
    ['PSD_pA_Y_diff','bits'],
    ['T_time','Seconds']]
    
    data_dict = {}
    for channel in data:
        data_dict[channel[0]] = DataChannel(channel[0],channel[1],[0])
    return data_dict


def get_unit_dictionary(self):
    units = {
        'Time':'(s)',
        'X-force':'(pN)',
        'Y-force':'(pN)',
        'Z-force':'(pN)',
        'Motor_position':'ticks',
        'X-position':'(microns)',
        'Y-position':'(microns)',
        'Z-position':'(microns)',
        'Temperature': 'Celsius',
        'T_time':'Seconds',
    }
    return units

def load_example_image(c_p):
    """
    Loads an example image so new functions of the software
    can be tested also without the camera connected.

    Parameters
    ----------
    c_p : TYPE
        DESCRIPTION. Control parameters to add the fake image
        in

    Returns
    -------
    None.

    """
    img = Image.open("./Example data/BG_image.jpg")
    c_p['image'] = np.asarray(img)
