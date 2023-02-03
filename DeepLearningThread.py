# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:48:59 2023

@author: marti
"""

import deeptrack as dt
import numpy as np
from threading import Thread
from time import sleep
from CustomMouseTools import MouseInterface
from PyQt6.QtGui import  QColor,QPen
from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel
)

import cv2
import matplotlib.pyplot as plt

# TODO the main network should be able to have multiple DL threads each with
# its own network alternatively we should have the thread capable of having 
# multiple networks.
class DeepLearningAnalyserLDS(Thread):
    """
    Thread which analyses the real-time image for detecting particles
    """
    def __init__(self, c_p, particle_type=0, model=None):
        """
        

        Parameters
        ----------
        c_p : TYPE
            DESCRIPTION.
        model : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        Thread.__init__(self)
        self.c_p = c_p
        self.model = model
        self.training_target_size = (64, 64)
        self.particle_type = particle_type # Type of particle to be tracked/analyzed
        self.setDaemon(True)


    def train_new_model(self, training_data):
        """
        Trains a Lode-star model on the data supplied in training data.

        Parameters
        ----------
        training_data : TYPE numpy array 
            DESCRIPTION. A NxN array of numbers or NxNx3 (if color image) on
            which a network is to be trained.

        Returns
        -------
        None.

        """
        
        
        # Check that the data is square
        assert np.shape(training_data)[0] == np.shape(training_data)[1], "Training data not square"

        self.model = dt.models.LodeSTAR(input_shape=(None, None, 1))
        # Rescale training data to fit the standard size which is 64
        self.pred_image_scale = 1
        original_width = np.shape(training_data)[0]
        if original_width > 64:
            self.c_p['prescale_factor'] = 64 / original_width
            training_data = cv2.rescale(training_data, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
        training_data = dt.Value(training_data)
        self.model.fit(training_data, epochs=self.c_p['epochs'], batch_size=8) # Default
        """
        while (self.c_p['epochs_trained'] < self.c_p['epochs']) and self.c_p['program_running']:
            self.model.fit(training_data, epochs=1, batch_size=8)
            self.c_p['epochs_trained'] += 1
        self.c_p['epochs_trained'] = 0        
        """
    def make_prediction(self):
        """
        Predicts particle positions in the center square of the current image
        being displayed.        

        Returns
        -------
        positions : TYPE
            DESCRIPTION.

        """
        
        assert self.model is not None, "No model to make the prediction"
        
        # Prepare the image for prediction
        data = np.array(self.c_p['image'])
        s = np.shape(data)
        # Cut out square in center to analyze. Check with benjamin if this is necessary
        # TODO check how this handles color
        if s[0] < s[1]:
            diff = int((s[1] - s[0])/2)
            data = data[:,diff:diff+s[0]]
        elif s[0] > s[1]:
            diff = int((s[0] - s[1])/2)
            data = data[diff:diff+s[0],:]
        # rescale data
        new_size = int(np.shape(data)[0] * self.c_p['prescale_factor'])
        data = cv2.resize(data, dsize=(new_size, new_size), interpolation=cv2.INTER_CUBIC)
        data = np.reshape(data, [1, new_size, new_size, 1])
        try:
            positions = self.model.predict_and_detect(data)# TODO have alpha, cut_off etc adaptable.
        except Exception as e:
            print("Deeptrack error \n", e)
            # Get the error "h = 0 is ambiguous, use local_maxima() instead?"
            return np.array([[300,300]])
        # print(np.array(positions[0])/self.c_p['prescale_factor'])
        return np.array(positions[0]) / self.c_p['prescale_factor'] * self.c_p['image_scale']

    def run(self):
        
        while self.c_p['program_running']:
            # By default check a central square of the frame. Maybe even have a ROI for this thread
            if self.model is not None and self.c_p['tracking_on']:
                self.c_p['predicted_particle_positions'] = self.make_prediction()
            if self.c_p['train_new_model']:
                print("training new model")
                self.train_new_model(self.c_p['training_image'])
                self.c_p['train_new_model'] = False
            sleep(0.1)
         
    

class DeepLearningControlWidget(QWidget):
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        
        self.label = QLabel("Deep larning controller")
        layout.addWidget(self.label)

        """
        self.SpeedLineEdit = QLineEdit()
        self.SpeedLineEdit.setValidator(QIntValidator(0,200))
        self.SpeedLineEdit.setText(str(self.motor_speed))
        self.SpeedLineEdit.textChanged.connect(self.set_motor_speed)
        layout.addWidget(self.SpeedLineEdit) 
        """

        self.toggle_tracking_button = QPushButton('Tracking on')
        self.toggle_tracking_button.pressed.connect(self.toggle_tracking)
        self.toggle_tracking_button.setCheckable(True)
        layout.addWidget(self.toggle_tracking_button)

        self.training_image_button = QPushButton('Display training image')
        self.training_image_button.pressed.connect(self.show_training_image)
        self.training_image_button.setCheckable(False)
        layout.addWidget(self.training_image_button)

        self.train_network_button = QPushButton('Train network')
        self.train_network_button.pressed.connect(self.train_network)
        self.train_network_button.setCheckable(False)
        layout.addWidget(self.train_network_button)

        self.setLayout(layout)

    def save_network(self, name):
        pass
    
    def load_network(self, network_path):
        pass

    def toggle_tracking(self):
        self.c_p['tracking_on'] = not self.c_p['tracking_on']

    def set_tracking_prescale_factor(self, scale):
        self.c_p['prescale_factor'] = scale

    def set_alpha(self):
        pass
 
    def set_cut_off(self):
        pass
 
    def train_network(self):
        # TODO make sure one cannot do this while a network is being trained
        self.c_p['train_new_model'] = True

    def show_training_image(self):
        plt.imshow(self.c_p['training_image'])
        plt.show()

class MouseAreaSelect(MouseInterface):
    
    def __init__(self, c_p):
        self.c_p = c_p
        self.x_0 = 0
        self.y_0 = 0
        self.x_0_motor = 0
        self.y_0_motor = 0
        self.image = np.zeros([64,64,3])
        
        self.blue_pen = QPen()
        self.blue_pen.setColor(QColor('blue'))
        self.blue_pen.setWidth(2)


    def draw(self, qp):
        if self.c_p['mouse_params'][0] == 1:
            qp.setPen(self.blue_pen)                
            x1,y1,x2,y2 = self.c_p['mouse_params'][1:5]
            qp.drawRect(x1,y1,x2-x1,y2-y1)
            return

    def mousePress(self):

        # left click
        if self.c_p['mouse_params'][0] == 1:
            pass
        # Right click -drag
        if self.c_p['mouse_params'][0] == 2:
            pass
        
    def mouseRelease(self):
        if self.c_p['mouse_params'][0] != 1:
            return
        x0, y0, x1, y1 = self.c_p['mouse_params'][1:5]
        dx = x1 - x0
        dy = y1 - y0
        if dx**2 < 100 or dy**2 < 100:
            print(dx,dy)
            return
        left = int(x0 * self.c_p['image_scale'])
        right = int(x1 *self.c_p['image_scale'])
        if right < left:
            tmp = right
            right = left
            left = tmp
        up = int(y0 * self.c_p['image_scale'])
        down = int(y1 * self.c_p['image_scale'])
        if up < down:
            tmp = up
            up = down
            down = tmp
        im = self.c_p['image']
        if len(np.shape(im)) > 2:
            image = im[down:up,left:right,:]
        else:
            image = im[down:up, left:right]
        plt.imshow(image)
        plt.show()

        if up-down > right-left:
            width = right-left
        else:
            width = up-down
        crop = im[down:down+width, left:left+width]
        self.c_p['prescale_factor'] = 64 / width
        print(self.c_p['prescale_factor'])
        res = cv2.resize(crop, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
        plt.imshow(res)
        plt.show()
        self.c_p['training_image'] = np.reshape(res,[64,64,1])

    def mouseDoubleClick(self):
        pass
    
    def mouseMove(self):
        if self.c_p['mouse_params'][0] == 2:
            pass
        
