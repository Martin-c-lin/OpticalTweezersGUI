# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:48:59 2023

@author: marti
"""
# TODO make sure that image orientations are correct and that we are using GPU.
import torch
import cv2
import sys
# Todo remove deeptrack
# import deeptrack as dt
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from threading import Thread
from time import sleep
from PIL import Image
from PyQt6.QtGui import  QColor,QPen
from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
)


sys.path.append('C:/Users/Martin/OneDrive/PhD/AutOT/') # TODO move this to same folder as this file

import find_particle_threshold as fpt
from unet_model import UNet
from CustomMouseTools import MouseInterface

# TODO the main network should be able to have multiple DL threads each with
# its own network alternatively we should have the thread capable of having 
# multiple networks.

def torch_unet_prediction(model, image, device, fac=1.4, threshold=260):

    new_size = [int(np.shape(image)[1]/fac),int(np.shape(image)[0]/fac)]
    rescaled_image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)
    s = np.shape(rescaled_image)
    rescaled_image = rescaled_image[:s[0]-s[0]%32, :s[1]-s[1]%32]
    if np.shape(rescaled_image)[0] < 100 or np.shape(rescaled_image)[1] <100:
        return np.array([])
    # TODO do more of this in pytorch which is faster since it works on GPU
    rescaled_image = np.float32(np.reshape(rescaled_image,[1,1,np.shape(rescaled_image)[0],np.shape(rescaled_image)[1]]))
    try:
        torch.cuda.empty_cache() # TODO only do this if device is GPU
        predicted_image = model(torch.tensor(rescaled_image).to(device))
    except Exception as E:
        print("GPU out of memory, using CPU instead")
        print(E)
        model.to("cpu")
        predicted_image = model(torch.tensor(rescaled_image).to("cpu"))
    resulting_image = predicted_image.detach().cpu().numpy()
    x,y,_ = fpt.find_particle_centers(np.array(resulting_image[0,0,:,:]), threshold)
    ret = []
    for x_,y_ in zip(x,y):
        ret.append([x_*fac,y_*fac])
    return np.array(ret)

def load_torch_unet(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU {torch.cuda.is_available()}\nDevice name: {torch.cuda.get_device_name(0)}")
    model = UNet(
        input_shape=(1, 1, 256, 256),
        number_of_output_channels=1,  # 2 for binary segmentation and 3 for multiclass segmentation
        conv_layer_dimensions=(8, 16, 32, 64, 128, 256),  # smaller UNet (faster training)
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model, device

class DeepLearningAnalyserLDS(Thread):
    """
    Thread which analyses the real-time image for detecting particles
    """
    def __init__(self, c_p, data_channels, particle_type=0, model=None):
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
        # TODO replace model with network
        Thread.__init__(self)
        self.c_p = c_p
        self.data_channels = data_channels
        self.c_p['model'] = model
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

        self.c_p['model'] = dt.models.LodeSTAR(input_shape=(None, None, 1))
        # Rescale training data to fit the standard size which is 64
        self.pred_image_scale = 1
        original_width = np.shape(training_data)[0]
        if original_width > 64:
            self.c_p['prescale_factor'] = 64 / original_width
            # TODO use PIL rescale and not cv2, may make a difference!
            training_data = cv2.resize(training_data, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
        training_data = dt.Value(training_data)
        
        self.c_p['model'].fit(training_data, epochs=self.c_p['epochs'], batch_size=8) # Default

    def setModel(self, model):
        self.c_p['model'] = model

    def make_unet_prediction(self):
        fac = 4 # TODO make this a parameter
        s = np.shape(self.c_p['image'])
        crop = self.c_p['image'][0:s[0]-s[0]%32,0:s[1]-s[1]%32] # TODO check indices
        new_size = (int(np.shape(crop)[1]/fac),int(np.shape(crop)[0]/fac))
        rescaled_image = cv2.resize(crop, dsize=new_size, interpolation=cv2.INTER_CUBIC)
        rescaled_image = np.reshape(rescaled_image,[1,np.shape(rescaled_image)[0],np.shape(rescaled_image)[1],1])
        tmp = np.float64(rescaled_image)/ np.max(rescaled_image)
        tmp *= 2
        tmp -= (np.min(tmp)/2)
        predicted_image = self.c_p['model'].predict(tmp)
        x,y,_ = fpt.find_particle_centers(predicted_image[0,:,:,0],self.c_p['cutoff'])
        return np.array(x)*fac, np.array(y)*fac

    def weak_gpu_torch_unet_prediction(self):
        # TODO cut to area of interest here
        width = self.c_p['image'].shape[0]
        height = self.c_p['image'].shape[1]
        max_w = 1200
        if width > max_w:
            x0 = int(width/2-max_w/2)
            x1 = int(width/2+max_w/2)
        else:
            x0=0
            x1 = width
        if height > max_w:
            y0 = int(height/2-max_w/2)
            y1 = int(height/2+max_w/2)
        else:
            y0=0
            y1 = height
        start_pos_x = self.data_channels['Motor_x_pos'].get_data(1)[0]
        start_pos_y = self.data_channels['Motor_y_pos'].get_data(1)[0]
        prediction = torch_unet_prediction(self.c_p['model'], self.c_p['image'][x0:x1,y0:y1], self.c_p['device'] ) 
        if len(prediction) == 0:
            return prediction
        dx = (start_pos_x - self.data_channels['Motor_x_pos'].get_data(1)[0])/self.c_p['ticks_per_pixel']
        dy = (start_pos_y - self.data_channels['Motor_y_pos'].get_data(1)[0])/self.c_p['ticks_per_pixel']
        prediction[:,1] += x0 #- dy # SIgne etc wrong maybe?
        prediction[:,0] += y0 #+ dx
        #print(prediction)
        self.c_p['particle_prediction_made'] = True
        return prediction

    def make_prediction(self):
        """
        Predicts particle positions in the center square of the current image
        being displayed.        

        Returns
        -------
        positions : TYPE
            DESCRIPTION.

        """
        
        assert self.c_p['model'] is not None, "No model to make the prediction"
        
        if self.c_p['network'] == "DeepTrack Unet":
            return self.make_unet_prediction()
        if self.c_p['network'] == "Pytorch Unet":
            return self.weak_gpu_torch_unet_prediction()  # When running on weak laptop GPU
            # return torch_unet_prediction(self.c_p['model'], self.c_p['image'], self.c_p['device'] ) 

        # Prepare the image for prediction
        data = np.array(self.c_p['image'])

        height = int(self.c_p['prescale_factor']*np.shape(data)[0])
        width = int(self.c_p['prescale_factor']*np.shape(data)[1])
        data = np.array(Image.fromarray(data).resize((width,height)))
        data = np.reshape(data,[1,height, width,1])
        try:
            alpha = self.c_p['alpha']
            cutoff= self.c_p['cutoff']
            beta = 1-alpha
            positions = self.c_p['model'].predict_and_detect(data, alpha=alpha, cutoff=cutoff, beta=beta)# TODO have alpha, cut_off etc adaptable.

        except Exception as e:
            print("Deeptrack error \n", e)
            # Get the error "h = 0 is ambiguous, use local_maxima() instead?"
            return np.array([[300,300]])
        return np.array(positions[0]) / self.c_p['prescale_factor']# / self.c_p['image_scale'] Using pixels of camera as default unit
    
    def locate_pipette(self):
        # TODO check if cupy is installed. If not, use numpy
        start_pos_x = self.data_channels['Motor_x_pos'].get_data(1)[0]
        start_pos_y = self.data_channels['Motor_y_pos'].get_data(1)[0]
        self.c_p['pipette_location'][1], self.c_p['pipette_location'][0], _ = fpt.find_pipette_top_GPU(self.c_p['image']) #fpt.find_pipette_top_GPU(self.c_p['image'])
        if self.c_p['pipette_location'][0] is None:
            return

        dx = start_pos_x - self.data_channels['Motor_x_pos'].get_data(1)[0]
        dy = start_pos_y - self.data_channels['Motor_y_pos'].get_data(1)[0]
        self.c_p['pipette_location'][1] -= dy/self.c_p['ticks_per_pixel']
        self.c_p['pipette_location'][0] += dx/self.c_p['ticks_per_pixel']

        print("Pipette at", self.c_p['pipette_location'][1], self.c_p['pipette_location'][0])
        
        self.c_p['pipette_located'] = True # TODO add location in motor steps as well.

    def run(self):
        
        while self.c_p['program_running']:
            # By default check a central square of the frame. Maybe even have a ROI for this thread
            if self.c_p['model'] is not None and self.c_p['tracking_on']:
                self.c_p['predicted_particle_positions'] = self.make_prediction()
            else:
                sleep(0.1)
            if self.c_p['locate_pippette']:
                self.locate_pipette()
            if self.c_p['train_new_model']:
                print("training new model")
                self.train_new_model(self.c_p['training_image'])
                self.c_p['train_new_model'] = False
            

class DeepLearningControlWidget(QWidget):
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        
        self.label = QLabel("Deep learning controller")
        layout.addWidget(self.label)
        self.setWindowTitle("Deep learning controller")

        self.toggle_tracking_button = QPushButton('Tracking on')
        self.toggle_tracking_button.pressed.connect(self.toggle_tracking)
        self.toggle_tracking_button.setCheckable(True)
        self.toggle_tracking_button.setChecked(self.c_p['tracking_on'])
        layout.addWidget(self.toggle_tracking_button)

        self.training_image_button = QPushButton('Display training image')
        self.training_image_button.pressed.connect(self.show_training_image)
        self.training_image_button.setCheckable(False)
        layout.addWidget(self.training_image_button)
        """
        self.train_network_button = QPushButton('Train network')
        self.train_network_button.pressed.connect(self.train_network)
        self.train_network_button.setCheckable(False)
        layout.addWidget(self.train_network_button)
        
        self.load_network_button = QPushButton('Load network')
        self.load_network_button.pressed.connect(self.load_network)
        self.load_network_button.setCheckable(False)
        layout.addWidget(self.load_network_button)

        self.load_unet_button = QPushButton('Load deeptrack U-Net')
        self.load_unet_button.pressed.connect(self.load_deeptrack_unet)
        self.load_unet_button.setCheckable(False)
        layout.addWidget(self.load_unet_button)
        """
        self.load_pytorch_unet_button = QPushButton('Load pytorch U-Net')
        self.load_pytorch_unet_button.pressed.connect(self.load_pytorch_unet)
        self.load_pytorch_unet_button.setCheckable(False)
        layout.addWidget(self.load_pytorch_unet_button)

        self.save_network_button = QPushButton('Save network')
        self.save_network_button.pressed.connect(self.save_network)
        self.save_network_button.setCheckable(False)
        layout.addWidget(self.save_network_button)

        self.locate_pipette_button = QPushButton('Locate pipette')
        self.locate_pipette_button.pressed.connect(self.locate_pipette)
        self.locate_pipette_button.setCheckable(True)
        self.locate_pipette_button.setChecked(self.c_p['locate_pippette'])
        self.locate_pipette_button.setToolTip("Locate the pipette tip in the image")
        layout.addWidget(self.locate_pipette_button)

        self.setLayout(layout)

    # TODO add thereshold slider for the network

    def save_network(self):
        # Not finished
        filename = QFileDialog.getSaveFileName(self, 'Save network',
            self.c_p['recording_path'],"Network (*.h5)")
        print(f"Filename for saving {filename} .")
    
    def load_network(self):
        filename = QFileDialog.get(self, 'Load network', self.c_p['recording_path'])
        print(f"You want to open network {filename}")
        backend = tf.keras.models.load_model(filename) 
        self.c_p['model'] = dt.models.LodeSTAR(backend.model) 
        self.c_p['prescale_factor'] = 0.106667 # TODO fix so this is changeable

    def load_deeptrack_unet(self):
        network_name = QFileDialog.getExistingDirectory(self, 'Load network', self.c_p['recording_path'])
        custom_objects = {"unet_crossentropy": dt.losses.weighted_crossentropy((10, 1))}
        with tf.keras.utils.custom_object_scope(custom_objects):
            #try:
            self.c_p['model'] = tf.keras.models.load_model(network_name)
            self.c_p['network'] = "DeepTrack Unet"
            
    def load_pytorch_unet(self):

        network_name = QFileDialog.getOpenFileName(self, 'Load network')
        print(f"Opening network {network_name[0]}")
        self.c_p['model'], self.c_p['device']  = load_torch_unet(network_name[0])
        self.c_p['network'] = "Pytorch Unet"

    def locate_pipette(self):
        self.c_p['locate_pippette'] = not self.c_p['locate_pippette']


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
        res = cv2.resize(crop, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
        plt.imshow(res)
        plt.show()
        self.c_p['training_image'] = np.reshape(res,[32,32,1])

    def mouseDoubleClick(self):
        pass
    
    def mouseMove(self):
        if self.c_p['mouse_params'][0] == 2:
            pass
    
    def getToolName(self):
        return "Area select tool"

    def getToolTip(self):
        return "Use the mouse to select an area to train network on by dragging."
        
