# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:48:24 2022

@author: marti
"""

import numpy as np
from PIL import Image # Errors with this, dont know why
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
           'mouse_params': [0, 0, 0, 0, 0, 0],
           'click_tools': [],

           # Camera c_p
           'image': np.ones([500, 500]),#, 1]),
           'image_idx': 0, # Index of snapshot image, add also video idx maybe.
           'color': "mono",  # Options are mono and color # Maybe add bit depth too
           'new_settings_camera': [False, None],
           'camera_width': 1920,
           'camera_height': 1080,
           'recording': False,
           'exposure_time': 5000,
           'fps': 50,  # Frames per second of camera
           'filename': '',
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
           'microns_per_pix': 30/5000 * 1e-3, # 5000 pixels per 30 micron roughly, changed to have more movements

           # Temperature c_p
           'temperature_output_on':False,
           
           # Piezo c_p (xyz separete stage)
           'piezo_targets': [10,10,10],
           'piezo_pos': [10,10,10],
           
           # PIC reader c_p
           # TODO add all the necessary channels here. PSDs and motors.
           'pic_channels':[
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_A_F_X', 'PSD_A_F_Y', 'PSD_A_F_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',
                            'PSD_B_F_X', 'PSD_B_F_Y', 'PSD_B_F_sum',
                            'Photodiode_A','Photodiode_B',
                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos', 
                            #'Motor_x_speed', 'Motor_y_speed', 'Motor_z_speed',
                            'T_time','Time_micros_high','Time_micros_low',
                           ],
                           
           # Temporary solution to use both PIC and Portenta
           'old_pic_channels':[
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_A_F_X', 'PSD_A_F_Y', 'PSD_A_F_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',
                            'PSD_B_F_X', 'PSD_B_F_Y', 'PSD_B_F_sum',
                            'T_time',
                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos', 
                            'Motor_x_speed', 'Motor_y_speed', 'Motor_z_speed',
                           ],
            
            'used_pic_channels':[
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',],
                            
           # Piezo outputs
           'piezo_A': np.uint16([20_000, 20_000]),
           'piezo_B': np.uint16([20_000, 20_000]),

           # Deep learning tracking
           'network': None,
           'tracking_on': False,
           'prescale_factor': 1, # Factor with which the image is to be prescaled before doing the tracking/traing
           'alpha': 1,
           'cutoff': 0.9995,
           'train_new_model': False,
           'model':None,
           'device': None, # Pytorch device on which the model runs
           'training_image': np.zeros([64,64]),
           'epochs': 30,
           'epochs_trained': 0,
           'predicted_particle_positions': np.array([]),

            # Autocontroller parameters
            'centering_on': False,
            'trap_particle': False,
            'search_and_trap': False,
            'laser_position': [1520,1830], # Default 

            # Minitweezers controller parameters
            'COM_port': 'COM9',
            'minitweezers_connected': False,

           # Minitweezers motors
           'motor_x_target_speed': 0,
           'motor_y_target_speed': 0,
           'motor_z_target_speed': 0,
           'minitweezers_target_pos': [32678,32678,32678],
           'minitweezers_target_speed': [0,0,0],
           'motor_travel_speed': 5_000,
           'move_to_location': False, # Should the motors move to a location rather than listen to the speed?

           # Thorlabs motors
           'disconnect_motor':[False,False,False],
           'thorlabs_motor_threads': [],
           'serial_nums_motors':["27502419","27502438",""], # Serial numbers of x,y, and z motors
           'stepper_serial_no': '70167314',
           'thorlabs_threads': [None,None,None],
           'stepper_starting_position': [0, 0, 0],
           'stepper_controller': None,
           'polling_rate': 250,

            # Common motor parameters
           'disconnect_motor':[False,False,False],
           'stage_stepper_connected': [False, False, False],
           'stepper_current_position': [0, 0, 0],
           'stepper_target_position': [2.3, 2.3, 7],
           'stepper_move_to_target': [False, False, False],
           'stepper_next_move': [0, 0, 0],
           'stepper_max_speed': [0.01, 0.01, 0.01],
           'stepper_acc': [0.005, 0.005, 0.005],
           'new_stepper_velocity_params': [False, False, False],
           'connect_steppers': [False,False,False], # Should steppers be connected?
           'steppers_connected': [False, False, False], # Are the steppers connected?
           'saved_positions':[],

           # Thorlabs piezo k-cube
           'z_starting_position': 0,
           'z_current_position': 0,
           'z_piezo_connected': False,
           'connect_z_piezo': True,
           'z_movement':0,
        }
    return c_p


from dataclasses import dataclass

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


def get_data_dicitonary_new():
    """
    ['PSD_pA_x1','bits'],
    ['PSD_pA_x2','bits'],
    ['PSD_pA_y1','bits'],
    ['PSD_pA_y2','bits'],
    """
    data = [['Time','(s)'],
    ['particle_trapped','(bool)'],
    ['X-force','(pN)'],
    ['Y-force','(pN)'],
    ['Z-force','(pN)'],
    ['Motor_position','ticks'],
    ['X-position','(microns)'], # Remove thos that are not used
    ['Y-position','(microns)'],
    ['Z-position','(microns)'],
    ['Temperature', 'Celsius'],
    ['Motor_x_pos', 'ticks'],
    ['Motor_y_pos','ticks'],
    ['Motor_z_pos', 'ticks'],
    ['Motor_x_speed','ticks/s'],
    ['Motor_y_speed','ticks/s'],
    ['Motor_z_speed','ticks/s'],
    ['PSD_A_P_X','bits'],
    ['PSD_A_P_Y','bits'],
    ['PSD_A_P_sum','bits'],
    ['PSD_A_F_X', 'bits'],
    ['PSD_A_F_Y','bits'],
    ['PSD_A_F_sum','bits'],
    ['PSD_B_P_X', 'bits'],
    ['PSD_B_P_Y','bits'],
    ['PSD_B_P_sum','bits'],
    ['PSD_B_F_X', 'bits'],
    ['PSD_B_F_Y','bits'],
    ['PSD_B_F_sum','bits'],
    ['Photodiode_A','bits'],
    ['Photodiode_B','bits'],
    ['T_time','Seconds'],
    ['Time_micros_high','microseconds'],
    ['Time_micros_low','microseconds'],
    ['Time_micros','microseconds'], # Make this the time from the portenta.
    ]
    
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
