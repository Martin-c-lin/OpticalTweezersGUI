# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:57:06 2023

@author: marti
"""


from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel
)

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QIntValidator

import numpy as np
from functools import partial
from threading import Thread
from ThorlabsMotor import MotorThreadV2, PiezoThread
from CustomMouseTools import MouseInterface
from time import time

# Maybe have this as a QThread?
class MotorControllerWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        self.motor_speed = 4 # TODO replace this with a motor speed in c_p
        self.y_movement = 0
        
        self.label = QLabel("Motor controller")
        layout.addWidget(self.label)

        self.SpeedLineEdit = QLineEdit()
        self.SpeedLineEdit.setValidator(QIntValidator(0,100))
        self.SpeedLineEdit.setText(str(self.motor_speed))
        self.SpeedLineEdit.textChanged.connect(self.set_motor_speed)
        layout.addWidget(self.SpeedLineEdit) 
        
        # TODO fix naming of functions/buttons confusing even if okay in GUI
        self.up_button = QPushButton('LEFT')
        self.up_button.pressed.connect(self.move_up)
        self.up_button.released.connect(self.stop_y)
        self.up_button.setCheckable(False)
        layout.addWidget(self.up_button)

        self.down_button = QPushButton('RIGHT')
        self.down_button.pressed.connect(self.move_down)
        self.down_button.released.connect(self.stop_y)
        self.down_button.setCheckable(False)
        layout.addWidget(self.down_button)


        self.right_button = QPushButton('DOWN')
        self.right_button.pressed.connect(self.move_right)
        self.right_button.released.connect(self.stop_x)
        self.right_button.setCheckable(False)
        layout.addWidget(self.right_button)

        self.left_button = QPushButton('UP')
        self.left_button.pressed.connect(self.move_left)
        self.left_button.released.connect(self.stop_x)
        self.left_button.setCheckable(False)
        layout.addWidget(self.left_button)

        self.objective_forward_button = QPushButton('Objective forward')
        self.objective_forward_button.pressed.connect(self.objective_forward)
        self.objective_forward_button.released.connect(self.objective_stop)
        self.objective_forward_button.setCheckable(False)
        layout.addWidget(self.objective_forward_button)

        self.objective_backward_button = QPushButton('Objective backward')
        self.objective_backward_button.pressed.connect(self.objective_backward)
        self.objective_backward_button.released.connect(self.objective_stop)
        self.objective_backward_button.setCheckable(False)
        layout.addWidget(self.objective_backward_button)

        self.setLayout(layout)
        
    def set_motor_speed(self, speed):
        # Maybe have this toggleable
        self.motor_speed = int(speed)
    # TODO check if x and y have been mixed up somewhere
    def move_up(self):
        self.c_p['motor_x_target_speed'] = self.motor_speed
    def stop_y(self):
        self.c_p['motor_x_target_speed'] = 0
    def move_down(self):
        self.c_p['motor_x_target_speed'] = - self.motor_speed


    def move_right(self):
        self.c_p['motor_y_target_speed'] = self.motor_speed
    def stop_x(self):
        self.c_p['motor_y_target_speed'] = 0
    def move_left(self):
        self.c_p['motor_y_target_speed'] = - self.motor_speed

    def objective_forward(self):
        self.c_p['motor_z_target_speed'] = self.motor_speed
    def objective_stop(self):
        self.c_p['motor_z_target_speed'] = 0
    def objective_backward(self):
        self.c_p['motor_z_target_speed'] = - self.motor_speed

    
class ThorlabsMotorWindow(QWidget):
    # Allow for connecting and disconnecting motors from this function
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        self.motor_speed = 1
        self.y_movement = 0
        self.motor_x = None
        self.motor_y = None
        self.piezo_z = None
        
        self.label = QLabel("Motor controller")
        layout.addWidget(self.label)

        
        self.SpeedLineEdit = QLineEdit()
        self.SpeedLineEdit.setValidator(QIntValidator(0,200))
        self.SpeedLineEdit.setText(str(self.motor_speed))
        self.SpeedLineEdit.textChanged.connect(self.set_motor_speed)
        layout.addWidget(self.SpeedLineEdit) 
        

        self.start_x = QPushButton('start x thread')
        self.start_x.pressed.connect(self.start_motor_x)
        self.start_x.setCheckable(False)
        layout.addWidget(self.start_x)

        self.start_y = QPushButton('start y thread')
        self.start_y.pressed.connect(self.start_motor_y)
        #self.start_y.released.connect(self.stop_y)
        self.start_y.setCheckable(False)
        layout.addWidget(self.start_y)
        
        self.start_z = QPushButton('start z thread')
        self.start_z.pressed.connect(self.start_piezo_z)
        self.start_z.setCheckable(False)
        layout.addWidget(self.start_z)

        
        self.print_position_button = QPushButton('Print position')
        self.print_position_button.pressed.connect(self.motor_y_pos)
        # self.print_position_button.released.connect(self.stop_y)
        self.print_position_button.setCheckable(False)
        layout.addWidget(self.print_position_button)

        self.move_up_button = QPushButton('Move up')
        self.move_up_button.pressed.connect(self.move_up)
        self.move_up_button.setCheckable(False)
        layout.addWidget(self.move_up_button)

        self.move_down_button = QPushButton('Move down')
        self.move_down_button.pressed.connect(self.move_down)
        self.move_down_button.setCheckable(False)
        layout.addWidget(self.move_down_button)

        self.setLayout(layout)
        
    def start_motor_y(self):
        if self.motor_y is None:
            self.motor_y = MotorThreadV2(channel=1, axis=1, c_p=self.c_p)
            self.motor_y.start()
            
    def start_motor_x(self):
        if self.motor_x is None:
            self.motor_x = MotorThreadV2(channel=0, axis=0, c_p=self.c_p)
            self.motor_x.start()

    def set_motor_speed(self, speed):
        try:
            speed = float(speed) * 1e-3
        except:
            return
        if speed > 1:
            print("Speed too high")
            return
        self.c_p['stepper_max_speed'][0] = speed
        self.c_p['stepper_max_speed'][1] = speed
        self.c_p['stepper_acc'][0] = speed * 2 
        self.c_p['stepper_acc'][1] = speed * 2
        self.c_p['new_stepper_velocity_params'][0] = True
        self.c_p['new_stepper_velocity_params'][1] = True        

    def start_piezo_z(self):
        if self.piezo_z is None:
            serial_no = "97100532"
            channel = 1
            self.piezo_z = PiezoThread(serial_no, channel, self.c_p)
            self.piezo_z.start()
            

    def motor_y_pos(self):
        print(self.c_p['stepper_current_position'][1])
        
    def move_up(self):
        if self.motor_y is not None:
            self.c_p['stepper_target_position'][1] += 0.1

    def move_down(self):
        if self.motor_y is not None:
            self.c_p['stepper_target_position'][1] -= 0.1

    def close(self):
        self.motor_y.join()
        super().close()


class MotorClickMove(MouseInterface):
    
    def __init__(self, c_p):
        self.c_p = c_p
        self.x_0 = 0
        self.y_0 = 0
        self.x_0_motor = 0
        self.y_0_motor = 0

    def mousePress(self):

        # left click
        if self.c_p['mouse_params'][0] == 1:
            center_x = int((self.c_p['camera_width']/2 - self.c_p['AOI'][0])/self.c_p['image_scale'])
            center_y = int((self.c_p['camera_height']/2 - self.c_p['AOI'][2])/self.c_p['image_scale'])
            dx_pix = (self.c_p['mouse_params'][1] - center_x) * self.c_p['image_scale']
            dy_pix = (self.c_p['mouse_params'][2] - center_y) * self.c_p['image_scale'] 
            
            self.c_p['stepper_target_position'][0] += dx_pix * self.c_p['microns_per_pix']
            self.c_p['stepper_target_position'][1] += dy_pix * self.c_p['microns_per_pix']
            
        # Right click -drag
        if self.c_p['mouse_params'][0] == 2:
            self.x_0 = self.c_p['mouse_params'][1]
            self.y_0 = self.c_p['mouse_params'][2]
            self.x_0_motor = self.c_p['stepper_current_position'][0]
            self.y_0_motor = self.c_p['stepper_current_position'][1]

        # Scroll wheel
        elif self.c_p['mouse_params'][0] == 3:
            self.y_0 = self.c_p['mouse_params'][2]
        
    def mouseRelease(self):
        if self.c_p['mouse_params'][0] == 2:
            self.c_p['stepper_target_position'][0] = self.c_p['stepper_current_position'][0]
            self.c_p['stepper_target_position'][1] = self.c_p['stepper_current_position'][1]
        
    def mouseDoubleClick(self):
        pass
    
    def draw(self, qp):
        pass
    
    def mouseMove(self):
        if self.c_p['mouse_params'][0] == 2:
            dx = (self.c_p['mouse_params'][3] - self.x_0) * self.c_p['image_scale'] 
            dy = (self.c_p['mouse_params'][4] - self.y_0) * self.c_p['image_scale'] 
            self.c_p['stepper_target_position'][0] = self.x_0_motor - dx * self.c_p['microns_per_pix']
            self.c_p['stepper_target_position'][1] = self.y_0_motor - dy * self.c_p['microns_per_pix']
            print(dx* self.c_p['microns_per_pix'], dy* self.c_p['microns_per_pix'])

        elif self.c_p['mouse_params'][0] == 3:
            dy = (self.y_0 - self.c_p['mouse_params'][4])
            self.y_0 = self.c_p['mouse_params'][4]
            if np.abs(dy) > 2:
                self.c_p['z_movement'] += dy * 4
            self.c_p['z_movement'] += dy
            print(dy, self.c_p['z_current_position'])

    def getToolName(self):
        return "motor tool"

    def getToolTip(self):
        return "Move the motors by clicking or dragging on the screen"
        
class MinitweezersMouseMove(MouseInterface):
    
    def __init__(self, c_p, data_channels):
        self.c_p = c_p
        self.data_channels = data_channels # Needed to know the position and speed
        self.x_0 = 0
        self.y_0 = 0
        self.z_0 = 0
        self.x_prev = 0
        self.y_prev = 0
        self.z_prev = 0
        self.prev_t = time()
        self.speed_factor = 2 # TODO make speed more accurate

    def mousePress(self):

        # left click
        if self.c_p['mouse_params'][0] == 1:
            # TODO change to laser position
            center_x = int((self.c_p['camera_width']/2 - self.c_p['AOI'][0])/self.c_p['image_scale'])
            center_y = int((self.c_p['camera_height']/2 - self.c_p['AOI'][2])/self.c_p['image_scale'])
            dx_pix = (self.c_p['mouse_params'][1] - center_x) * self.c_p['image_scale']
            dy_pix = (self.c_p['mouse_params'][2] - center_y) * self.c_p['image_scale'] 
  
        # Right click -drag
        if self.c_p['mouse_params'][0] == 2:
            self.x_prev= self.c_p['mouse_params'][1]
            self.y_prev= self.c_p['mouse_params'][2]

        if self.c_p['mouse_params'][0] == 3:
            self.z_0 = self.c_p['mouse_params'][2]

        # Scroll wheel
        elif self.c_p['mouse_params'][0] == 3:
            self.y_prev = self.c_p['mouse_params'][2]
        
    def mouseRelease(self):
        self.c_p['motor_x_target_speed'] = 0
        self.c_p['motor_y_target_speed'] = 0
        self.c_p['motor_z_target_speed'] = 0
        
    def mouseDoubleClick(self):
        pass
    
    def draw(self, qp):
        pass
    
    def check_speed(self, speed):
        if 0 < speed < 2:
            return 2
        if 0 > speed > -2:
            return -2
        if speed > 100:
            return 100
        if speed < -100:
            return  -100
        return speed

    def mouseMove(self):
        """
        t = time()
        dt = t - self.prev_t
        if dt==0:
            return
        self.prev_t = t
        """
        if self.c_p['mouse_params'][0] == 2:

            dx = (self.c_p['mouse_params'][3] - self.x_prev)#/dt
            dy = (self.c_p['mouse_params'][4] - self.y_prev)#/dt
            print(dy)

            x_speed = self.check_speed(dx * self.speed_factor)
            y_speed = self.check_speed(dy * self.speed_factor)
            self.c_p['motor_x_target_speed'] = int(x_speed)
            self.c_p['motor_y_target_speed'] = int(-y_speed)
            #print(f"Y moovement {y_speed}")
            self.x_prev = self.c_p['mouse_params'][3]
            self.y_prev = self.c_p['mouse_params'][4]
            
        elif self.c_p['mouse_params'][0] == 3:
            dz = (self.c_p['mouse_params'][4] - self.z_prev)#/dt
            z_speed = self.check_speed(dz * self.speed_factor)
            self.c_p['motor_z_target_speed'] = int(z_speed)
            self.z_prev = self.c_p['mouse_params'][4]

    def getToolName(self):
        return "motor tool"

    def getToolTip(self):
        return "Move the motors by clicking or dragging on the screen"
        
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        