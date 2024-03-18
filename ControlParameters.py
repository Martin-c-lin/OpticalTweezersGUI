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
           'video_format': 'npy', # Changed default to npy to reduce risk of losing data
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
                            'message', # Note how this is handled affects what number you get out.

                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos'],
            # These are the channels which are sampled once per sample cycle. Default 
            'single_sample_channels':[
                            # TODO make this a part of the data_Channels class. e.g sampling rate parameter
                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos',
                            'message', # This is used for debugging, i.e sending data from the controller to the computer
                            # for testing purposes.
                            'dac_ax','dac_ay','dac_bx','dac_by',
                            'PSD_Force_A_saved',
            ],
            # These channels are sampled multiple times per sample cycle.
            'multi_sample_channels':[
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_A_F_X', 'PSD_A_F_Y', 'PSD_A_F_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',
                            'PSD_B_F_X', 'PSD_B_F_Y', 'PSD_B_F_sum',
                            'Photodiode_A','Photodiode_B',
                            'T_time','Time_micros_low','Time_micros_high', # Moved this up
                            ],
            # These channels have values calculated from the "mulit sample channels" (i.e force converted from PSD reading)
            'derived_PSD_channels': ['F_A_X','F_A_Y','F_B_X','F_B_Y','F_A_Z','F_B_Z',
                                     'F_total_X','F_total_Y','F_total_Z',
                                     'Position_A_X', 'Position_A_Y','Position_B_X','Position_B_Y',
                                     ],

            'save_idx': 0, # Index of the saved data
           # Piezo outputs
           'averaging_interval': 1_000, # How many samples to average over in the data channels window
           'piezo_A': np.uint16([32768, 32768]),
           'piezo_B': np.uint16([32768, 32768]),
           'portenta_command_1': 0, # Command to send to the portenta, zero force etc.
           'portenta_command_2': 0, # Command to send to the portenta, dac controls
           'PSD_means':  np.uint16([0,0,0,0]), # Means of the PSD channels
           'PSD_force_means':  np.array([0,0,0,0]), # Means of the PSD channels
           'PSD_position_means': np.array([0,0,0,0]), # Means of the PSD channels

           # Deep learning tracking
           'network': None,
           'tracking_on': False,
           'z-tracking': False,
           'crop_width': 64,
           'prescale_factor': 1.5, # Factor with which the image is to be prescaled before doing the tracking/traing
           'alpha': 1,
           'cutoff': 0.9995,
           'train_new_model': False,
           'model':None,
           'z-model':None,
           'device': None, # Pytorch device on which the model runs
           'training_image': np.zeros([64,64]),
           'epochs': 30,
           'epochs_trained': 0,
           'predicted_particle_positions': np.array([]),
           'z-predictions': np.array([]),
           'particle_prediction_made': False,
           'default_unet_path': "NeuralNetworks\TorchBigmodelJune_1",
           'default_z_model_path': "NeuralNetworks\Z_model_large_range.pth",

            # Autocontroller parameters
            'centering_on': False,
            'trap_particle': False,
            'particle_in_pipette': False,
            'search_and_trap': False,# TODO clean up the laser position parameters
            'focus_z_trap_pipette': False, # Focus the particle in the trap with the one in the pipette
            'laser_position_A': [2660, 1502.3255814], #[1520,1830], # Default
            'laser_position_B': [2660, 1502.3255814], #[1520,1830], # Default
            'laser_position_A_predicted': [2660, 1502.3255814],
            'laser_position_B_predicted': [2660, 1502.3255814],
            'laser_position': [2660, 1502.3255814], # Updated as the average of position A and B
            # Laser a approximate x position is lpx = laser_a_transfer_matrix[0]*psd_a_x + laser_a_transfer_matrix[1]*psd_a_y
            'laser_a_transfer_matrix': np.array([-0.00082831, 0.00013445, -0.00020288, -0.00105998]),
            'laser_b_transfer_matrix': np.array([ 8.03026858e-04, 5.70465993e-05,  2.76983717e-05, -8.77370335e-04]), # 
            'z-threshold': 8, # Threshold for the z-tracking
            'locate_pipette': False, # TODO fix speling error
            'pipette_location': [0,0], # Location of the pipette in the image
            'pipette_location_chamber': [0,0,0], # Location of the pipette in the chamber
            'pipette_located': False,
            'center_pipette': False,
            'move2area_above_pipette': False,
            'move_avoiding_particles': False,
            'find_laser_position': False, # Updates the laser position to the current closest particle
            'AD_tube_position': [0,0,0], # Position of the AD tube in the chamber, motor coordinates
            'Trapped_particle_position': [0,0,0], # Position of the trapped particle in the image
            'pipette_particle_location': [1200,1200,0], # Location of the pipette particle in the image
            'attach_DNA_automatically': False,

            # Minitweezers controller parameters
            'COM_port': 'COM6',#'COM6',
            'minitweezers_connected': False,
            'blue_led': 0, # Wheter the blue led is on or off, 0 for on and 1 for off
            'objective_stepper_port': 'COM4', # COM4
            #'PSD_bits_per_micron_sum': [0.0703,0.0703], # Conversion factor between the PSD x(or y)/sum channel and microns i.e x/sum / psd_bits_per_micron_sum = microns
            'PSD_to_pos': [14.252,12.62], # The calibration factor for the position PSDs,
            'PSD_to_force': [0.02505,0.02565,0.02755,0.0287], # The calibration factor for the force PSDs, AX,AY, BX,BY
            'Photodiode_sum_to_force': [1,1], # The calibration factor for the photodiode/PSD sum channel to force

            # Minitweezers protocols parameters
            'protocol_running': False,
            'protocol_type': 'Constant speed', # Options are constant force, constant velocity, constant distance
            'protocol_data': np.uint8(np.zeros(13)),

            # Protocol for electrostatic interactions:
            "electrostatic_protocol_toggled": False,
            'electrostatic_protocol_running': False,
            'electrostatic_protocol_finished': False,
            'electrostatic_protocol_start': 20_000, # First postiion
            'electrostatic_protocol_end': 30_000, # Last postion
            'electrostatic_protocol_steps': 10, # stops of the protocol
            'electrostatic_protocol_duration': 20, # Duration of the protocol in seconds per step
            

            # Laser parameters
            'laser_A_port':'COM11',
            'laser_B_port':'COM12',
            'laser_A_current': 370, # Current in mA
            'laser_B_current': 330, # Current in mA
            'laser_A_on': False,
            'laser_B_on': False,
            'reflection_A': 0.0693,
            'reflection_B': 0.0816,#0.1579,
            'sum2power_A': 0.00692,
            'sum2power_B': 0.00682,
            'reflection_fac': 1.0057, #1.0111, # Factor relatets to the compensation when calculating the true sum readings.

            # Pump parameters
            'pump_adress': 'COM3',
            'target_pressures': np.array([0.0, 0.0 , 0.0, 0.0]),
            'current_pressures': np.array([0.0, 0.0 , 0.0, 0.0]),


           # Minitweezers motors
           'motor_x_target_speed': 0,
           'motor_y_target_speed': 0,
           'motor_z_target_speed': 0,
           'minitweezers_target_pos': [32678,32678,32678],
           'minitweezers_target_speed': [0,0,0],
           'motor_travel_speed': [2_000,2_000], # 5000 was somewhat high Speed of move to location.
           'move_to_location': False, # Should the motors move to a location rather than listen to the speed?
           'ticks_per_micron': 6.24,#24.45, # How many ticks per micron
           'microns_per_tick': 1/6.24, #0.0408, # How many microns per tick
           'ticks_per_pixel': 6.24/18.28, #1.337, # How many pixels per micron
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
# TODO implement a custom return funciton for the "derived" channels, i.e the ones which are calculated from other channels.
# TODO have sampling rate as a parameter for the data channels.
@dataclass
class DataChannel:
    name: str
    unit: str
    data: np.array
    saving_toggled: bool = True
    max_len: int = 10_000_000 # 10_000_000 default
    index: int = 0
    full: bool = False
    max_retrivable: int = 1 # number of datapoints which have been saved.

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
    ['PSD_A_F_sum_compensated','bits'],
    ['PSD_B_P_X', 'bits'],
    ['PSD_B_P_Y','bits'],
    ['PSD_B_P_sum','bits'],
    ['PSD_B_F_X', 'bits'],
    ['PSD_B_F_Y','bits'],
    ['PSD_B_F_sum','bits'],
    ['PSD_B_F_sum_compensated','bits'],
    ['Photodiode_A','bits'],
    ['Photodiode_B','bits'],
    ['Laser_A_power','mW'],
    ['Laser_B_power','mW'],
    ['T_time','microseconds'], # Time measured on the controller
    ['Time_micros_high','microseconds'],
    ['Time_micros_low','microseconds'],
    ['F_A_X','pN'],
    ['F_A_Y','pN'],
    ['F_B_X','pN'],
    ['F_B_Y','pN'],
    ['F_A_Z','pN'],
    ['F_B_Z','pN'],
    ['F_total_X','pN'],
    ['F_total_Y','pN'],
    ['F_total_Z','pN'],
    ['Position_A_X','microns'],
    ['Position_A_Y','microns'],
    ['Position_B_X','microns'],
    ['Position_B_Y','microns'],
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
