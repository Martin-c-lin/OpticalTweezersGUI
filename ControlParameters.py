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
            # General c_p of the UI
           'program_running': True,
           'mouse_params': [0, 0, 0, 0, 0, 0],
           'click_tools': [],
           'central_circle_on': True,

           # Camera c_p
           'image': np.ones([500, 500]),#, 1]),
           'image_idx': 0, # Index of snapshot image, add also video idx maybe.
           'color': "mono",  # Options are mono and color # Maybe add bit depth too
           'new_settings_camera': [False, None],
           'camera_width': 1920,
           'camera_height': 1080,
           'recording': False,
           'exposure_time': 5000,
           'fps': 50,  # Frames per second of camera, measured
           'target_frame_rate': 50, # Target frame rate of the camera, if you want it limited.
           'filename': '',
           'video_name': 'Video',
           'video_format': 'avi',
           'image_format': 'png',
           'image_gain': 1,
           'image_offset': 0,
           'AOI':[0,1000,0,1000], # Area of interest of camera
           'recording_path': '../Example data/',
           'bitrate': '30000000', # Bitrate of video to be saved
           'frame_queue': Queue(maxsize=2_000_000),  # Frame buffer essentially
           'image_scale': 1,
           'microns_per_pix': 1/18.28, # Note this parameter is system dependent!

           # Temperature c_p
           'temperature_output_on':False,
           
           # Piezo c_p (xyz separete stage)
           'piezo_targets': [10,10,10],
           'piezo_pos': [10,10,10],
           
           # PIC reader c_p
           'pic_channels':[# Channels to read from the controller
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_A_F_X', 'PSD_A_F_Y', 'PSD_A_F_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',
                            'PSD_B_F_X', 'PSD_B_F_Y', 'PSD_B_F_sum',
                            'Photodiode_A','Photodiode_B',
                            'T_time','Time_micros_low','Time_micros_high', # Moved this up
                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos', 
                            #
                            #'T_time','Time_micros_high','Time_micros_low',
                            'message',
                            'dac_ax','dac_ay','dac_bx','dac_by',
                            #'Motor_x_speed', 'Motor_y_speed', 'Motor_z_speed',
                           ],
           'offset_channels':[ #TODO change so that the offset is default setting.
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_A_F_X', 'PSD_A_F_Y', 'PSD_A_F_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',
                            'PSD_B_F_X', 'PSD_B_F_Y', 'PSD_B_F_sum',
                            'Photodiode_A','Photodiode_B',

                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos'],

            'single_sample_channels':[
                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos', 
                            'message', # TODO fix message, actually saved force here
                            'dac_ax','dac_ay','dac_bx','dac_by',
                            'PSD_Force_A_saved',
            ],

            'multi_sample_channels':[
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_A_F_X', 'PSD_A_F_Y', 'PSD_A_F_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',
                            'PSD_B_F_X', 'PSD_B_F_Y', 'PSD_B_F_sum',
                            'Photodiode_A','Photodiode_B',
                            'T_time','Time_micros_low','Time_micros_high', # Moved this up
                            ],
            'save_idx': 0, # Index of the saved data     
           # Piezo outputs
           'averaging_interval': 1000, # How many samples to average over in the data channels window
           'piezo_A': np.uint16([32768, 32768]),
           'piezo_B': np.uint16([32768, 32768]),
           'portenta_command_1': 0, # Command to send to the portenta, zero force etc.
           'portenta_command_2': 0, # Command to send to the portenta, dac controls
           'PSD_means':  np.uint16([0,0,0,0]), # Means of the PSD channels

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
           'particle_prediction_made': False,


            # Autocontroller parameters
            'centering_on': False,
            'trap_particle': False,
            'search_and_trap': False,
            'laser_position': [2660, 1502.3255814], #[1520,1830], # Default 
            'locate_pippette': False, # TODO fix speling error
            'pipette_location': [0,0], # Location of the pipette in the image
            'pipette_location_chamber': [0,0,0], # Location of the pipette in the chamber
            'pipette_located': False,
            'center_pipette': False,
            'move_avoiding_particles': False,
            'AD_tube_position': [0,0,0], # Position of the AD tube in the chamber, motor coordinates

            # Minitweezers controller parameters
            'COM_port': 'COM6',#'COM9',
            'minitweezers_connected': False,
            'blue_led': 0, # Wheter the blue led is on or off, 0 for on and 1 for off
            'objective_stepper_port': 'COM4',
            'PSD_bits_per_micron_sum': 0.0703, # Conversion factor between the PSD x(or y)/sum channel and microns i.e x/sum / psd_bits_per_micron_sum = microns 

            # Minitweezers protocols parameters
            'protocol_running': False,
            'protocol_type': 'Constant speed', # Options are constant force, constant velocity, constant distance
            'protocol_data': np.uint8(np.zeros(13)),

            # Laser parameters
            'laser_A_port':'COM8',
            'laser_B_port':'COM9',
            'laser_A_current': 370, # Current in mA
            'laser_B_current': 330, # Current in mA
            'laser_A_on': False,
            'laser_B_on': False,

           # Minitweezers motors
           'motor_x_target_speed': 0,
           'motor_y_target_speed': 0,
           'motor_z_target_speed': 0,
           'minitweezers_target_pos': [32678,32678,32678],
           'minitweezers_target_speed': [0,0,0],
           'motor_travel_speed': [2_000,2_000], # 5000 was somewhat high Speed of move to location.
           'move_to_location': False, # Should the motors move to a location rather than listen to the speed?
           'ticks_per_micron': 5.48,#24.45, # How many ticks per micron
           'ticks_per_pixel': 0.3, #1.337, # How many pixels per micron
            # TODO add a fix to when the controller is disconnected.
           

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
# TODO have the max_len be a tunable parameter for configuration.

@dataclass
class DataChannel:
    name: str
    unit: str
    data: np.array
    saving_toggled: bool = True
    max_len: int = 10_000_000 # 10_000_000 default
    index: int = 0
    full: bool = False
    max_retrivable: int = 1

    def __post_init__(self):
        # Preallocate memory for the maximum length
        self.data = np.zeros(self.max_len)

    def put_data(self, d):
        try:
            if len(d) > self.max_len:
                return
        except TypeError:
            d = [d]
        
        if self.index + len(d) >= self.max_len:
            end_points = self.max_len - self.index
            self.data[self.index:] = d[:end_points]
            self.data[:len(d) - end_points] = d[end_points:]
            self.full = True
            self.index = (self.index + len(d)) % self.max_len
            self.max_retrivable = self.max_len
        else:
            self.data[self.index:self.index + len(d)] = d
            self.index += len(d)
            self.max_retrivable = self.index

    def get_data(self, nbr_points):
        nbr_points = min(nbr_points, self.max_retrivable)
        if nbr_points <= self.index:
            return self.data[self.index-nbr_points:self.index]
        else:
            return np.concatenate([self.data[self.index-nbr_points:], self.data[:self.index]])
    
    def get_data_spaced(self, nbr_points, spacing=1):
        nbr_points = min(nbr_points, self.max_retrivable)
        final = self.index
        start = final - (final % spacing) - (nbr_points * spacing)
        if start >= 0:
            return self.data[start:final:spacing]
        else:
            last = (nbr_points * spacing + start) % self.max_len
            return np.concatenate([self.data[start::spacing], self.data[:last:spacing]])
    """
    def get_data_spaced(self, nbr_points, spacing=1):
        # Updated this to include the last index correctly.
        nbr_points = min(nbr_points, int(self.max_retrivable/spacing))
        final = self.index
        start = final - ((nbr_points-1) * spacing)
        if start >= 0:
            return self.data[start:final+1:spacing]
        else:
            last = (nbr_points * spacing + start) % self.max_len
            return np.concatenate([self.data[start::spacing], self.data[:last:spacing]])
    
    def get_data_spaced(self, nbr_points, spacing=1):
        nbr_points = min(nbr_points, self.max_retrivable)
        final = self.index + 1  # +1 to make sure the final index is included
        start = final - (nbr_points - 1) * spacing - 1  # Calculate start based on nbr_points and spacing

        # Make sure the start index is within bounds
        if start >= 0:
            return self.data[start:final:spacing]
        else:
            # Handle the case where start is negative
            last = (nbr_points * spacing + start) % self.max_len
            return np.concatenate([self.data[start::spacing], self.data[:last:spacing]])
    """

"""
@dataclass
class DataChannel:
# Works fine and equally fast as the other one.
    name: str
    unit: str
    data: np.array
    saving_toggled: bool = True
    max_len: int = 10_000_000 # 10_000_000 default
    index: int = 1
    full: bool = False
    max_retrivable: int = 1

    def put_data(self, d):
        try:
            if len(d) > self.max_len:
                return
        except TypeError:
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
        
        if self.index == self.max_len:  # Added this line
            self.index = 0

    def get_data(self, nbr_points):
        nbr_points = min(nbr_points, self.max_retrivable)
        diff = self.index-nbr_points
        if diff >= 0:
            ret = self.data[self.index-nbr_points:self.index]
        else:
            ret = np.concatenate([self.data[diff:], self.data[:self.index]]).ravel()
        if not len(ret) == nbr_points:
            print("Error lengths of data is", len(ret), nbr_points, len(self.data))
            return None
        return ret

    def get_data_spaced(self, nbr_points, spacing=1):
        nbr_points = min(nbr_points, self.max_retrivable)
        diff = self.index-nbr_points
        final = self.index
        start = final - (final % spacing) - (nbr_points * spacing)
        if diff > 0:
            ret = self.data[start:final:spacing]
        else:
            last = (nbr_points * spacing + start) % self.max_len  # Updated calculation for last
            ret = np.concatenate([self.data[start::spacing], self.data[:last:spacing]]).ravel()

        return ret
"""

def get_data_dicitonary_new():
    data = [
    ['Time','Seconds'], # Time measured by the computer.
    ['particle_trapped','(bool)'],
    ['Temperature', 'Celsius'],
    ['Motor_x_pos', 'ticks'],
    ['Motor_y_pos','ticks'],
    ['Motor_z_pos', 'ticks'],
    ['Motor_x_speed','ticks/s'],
    ['Motor_y_speed','ticks/s'],
    ['Motor_z_speed','ticks/s'],
    ['Motor time','microseconds'],
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
    ['T_time','microseconds'], # Time measured on the controller
    ['Time_micros_high','microseconds'],
    ['Time_micros_low','microseconds'],
    ['F_A_X','pN'],
    ['F_A_Y','pN'],
    ['F_B_X','pN'],
    ['F_B_Y','pN'],
    ['F_total_X','pN'],
    ['F_total_Y','pN'],
    ['F_total_Z','pN'],
    ['PSD_Force_A_saved','pN'],
    ['Photodiode/PSD SUM A','a.u.'],
    ['Photodiode/PSD SUM B','a.u.'],
    ['message','string'],
    ['dac_ax','bits'],
    ['dac_ay','bits'],
    ['dac_bx','bits'],
    ['dac_by','bits'],
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
