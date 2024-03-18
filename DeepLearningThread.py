# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:48:59 2023

@author: marti
"""
# TODO make sure that image orientations are correct and that we are using GPU.
import torch
import torch.nn as nn

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
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

sys.path.append('C:/Users/Martin/OneDrive/PhD/AutOT/') # TODO move this to same folder as this file

import find_particle_threshold as fpt
from unet_model import UNet
from CustomMouseTools import MouseInterface

# TODO the main network should be able to have multiple DL threads each with
# its own network alternatively we should have the thread capable of having 
# multiple networks.
class ParticleCNN(nn.Module):
    def __init__(self):
        super(ParticleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # Assuming grayscale images
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Size now 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Size now 32x32
        )
        # Calculate the size after convolutions and pooling
        # For 128x128 input, after two pooling layers, the size is 32x32
        # And if you have 64 output channels from the last conv layer, then:
        self.size_after_convs = 64 * 32 * 32
        self.fc_layers = nn.Sequential(
            nn.Linear(self.size_after_convs, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Predicting a single value
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1) # Flatten the output for the fully connected layer
        x = self.fc_layers(x)
        return x


def torch_unet_prediction(model, image, device, fac=1.4, threshold=260):

    new_size = [int(np.shape(image)[1]/fac),int(np.shape(image)[0]/fac)]
    rescaled_image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)
    s = np.shape(rescaled_image)
    rescaled_image = rescaled_image[:s[0]-s[0]%32, :s[1]-s[1]%32]
    if np.shape(rescaled_image)[0] < 100 or np.shape(rescaled_image)[1] <100:
        return np.array([])
    # TODO do more of this in pytorch which is faster since it works on GPU
    rescaled_image = np.float32(np.reshape(rescaled_image,[1,1,np.shape(rescaled_image)[0],np.shape(rescaled_image)[1]]))
    rescaled_image /= np.std(rescaled_image) # TODO check if this helped in any way with the stability.
    torch.cuda.empty_cache() # TODO only do this if device is GPU
    with torch.no_grad():
        predicted_image = model(torch.tensor(rescaled_image).to(device))
        resulting_image = predicted_image.detach().cpu().numpy()

    """
    try:
        torch.cuda.empty_cache() # TODO only do this if device is GPU
        with torch.no_grad():
            predicted_image = model(torch.tensor(rescaled_image).to(device))
    except Exception as E:
        print("GPU out of memory, using CPU instead")
        print(E)
        model.to("cpu")
        with torch.no_grad():
            predicted_image = model(torch.tensor(rescaled_image).to("cpu"))
    """
    x,y,_ = fpt.find_particle_centers_fast(np.array(resulting_image[0,0,:,:]), threshold)
    ret = []
    for x_,y_ in zip(x,y):
        ret.append([x_*fac, y_*fac])
    return np.array(ret)

def load_torch_unet(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU {torch.cuda.is_available()}\nDevice name: {torch.cuda.get_device_name(0)}")
    try:
        model = UNet(
            input_shape=(1, 1, 256, 256),
            number_of_output_channels=1,  # 2 for binary segmentation and 3 for multiclass segmentation
            conv_layer_dimensions=(8, 16, 32, 64, 128, 256),  # smaller UNet (faster execution)
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model, device
    except Exception as e:
        print(e)
        print("Could not load small model")
    try:
        model = UNet(
            input_shape=(1, 1, 256, 256),
            number_of_output_channels=1,  # 2 for binary segmentation and 3 for multiclass segmentation
            conv_layer_dimensions=(64, 128, 256, 512, 1024),  # standard UNet
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model, device    
    except Exception as e:
        print(e)
        print("Could not load big model")
        return None, None


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
        # TODO load some standard models automatically.
        Thread.__init__(self)
        self.c_p = c_p
        self.data_channels = data_channels
        self.c_p['model'] = model
        self.training_target_size = (64, 64)
        self.particle_type = particle_type # Type of particle to be tracked/analyzed
                # Load the default networks for tracking in xy and z.
        if self.c_p['default_unet_path'] is not None:
            self.c_p['model'], self.c_p['device']  = load_torch_unet(self.c_p['default_unet_path'])
            self.c_p['network'] = "Pytorch Unet"
            print(f"Loaded default network from {self.c_p['default_unet_path']}")
        if self.c_p['default_z_model_path'] is not None:
            self.c_p['z-model'] = torch.load(self.c_p['default_z_model_path'])
            print(f"Loaded default z-model from {self.c_p['default_z_model_path']}")
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
        tmp = np.float64(rescaled_image) / np.max(rescaled_image) # TODO test if change to 255 instead of max of image is more reliable
        tmp *= 2
        tmp -= (np.min(tmp)/2)
        predicted_image = self.c_p['model'].predict(tmp)
        x,y,_ = fpt.find_particle_centers_fast(predicted_image[0,:,:,0],self.c_p['cutoff']) # TODO Changed to fast here, check if it works
        return np.array(x)*fac, np.array(y)*fac

    def weak_gpu_torch_unet_prediction(self):
        # TODO make it so that this also incorporates the threhsolding on the GPU.
        # TODO cut to area of interest here
        width = self.c_p['image'].shape[0]
        height = self.c_p['image'].shape[1]
        max_w = 2400
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
        #start_pos_x = self.data_channels['Motor_x_pos'].get_data(1)[0]
        #start_pos_y = self.data_channels['Motor_y_pos'].get_data(1)[0]
        prediction = torch_unet_prediction(self.c_p['model'], self.c_p['image'][x0:x1,y0:y1], self.c_p['device'], fac = self.c_p['prescale_factor'], threshold=self.c_p['cutoff']) 
        if len(prediction) == 0:
            return prediction
        #dx = (start_pos_x - self.data_channels['Motor_x_pos'].get_data(1)[0])/self.c_p['ticks_per_pixel']
        #dy = (start_pos_y - self.data_channels['Motor_y_pos'].get_data(1)[0])/self.c_p['ticks_per_pixel']
        prediction[:,1] += x0 #- dy # SIgne etc wrong maybe?
        prediction[:,0] += y0 #+ dx
        self.c_p['particle_prediction_made'] = True
        return prediction

    def make_prediction(self, data=None):
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
            # return torch_unet_prediction(self.c_p['model'], self.c_p['image'], self.c_p['device'], fac=self.c_p['prescale_factor']) 
        # TODO this is not really used anymore, can probably remove it
        # Prepare the image for prediction
        if data is None:
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
        if self.c_p['tracking_on'] and len(self.c_p['predicted_particle_positions'])>0:
            # Remove the particle in the pipette from the image prediction
            self.c_p['pipette_location'][1], self.c_p['pipette_location'][0], _ = fpt.find_pipette_top_GPU(self.c_p['image'],subtract_particles=True,
                                                                                                           positions=self.c_p['predicted_particle_positions'])
        else:
            self.c_p['pipette_location'][1], self.c_p['pipette_location'][0], _ = fpt.find_pipette_top_GPU(self.c_p['image'])
        if self.c_p['pipette_location'][0] is None:
            return

        dx = start_pos_x - self.data_channels['Motor_x_pos'].get_data(1)[0]
        dy = start_pos_y - self.data_channels['Motor_y_pos'].get_data(1)[0]
        self.c_p['pipette_location'][1] -= dy / self.c_p['ticks_per_pixel']
        self.c_p['pipette_location'][0] += dx / self.c_p['ticks_per_pixel']

        self.c_p['pipette_located'] = True # TODO add location in motor steps as well.

    def predict_z_positions(self):
        """
        Function which makes a prediction of the z-positions of the particles located with
        the deep learning model. The z-positions are then stored in the control parameters.

        # TODO make it so that the reshaping is more flexible.
        # TODO add a scaling factor to get something that is similar to true z-positions. Could be model dependent.
        # TODO make it so that all the predictions are made at the same time by the network to save time.
        """

        if not self.c_p['tracking_on'] or self.c_p['z-model'] is None:
            return

        # Pre-compute constants
        image_shape = np.shape(self.c_p['image'])
        image_width, image_height = image_shape[1], image_shape[0]
        device = self.c_p['device']
        width = self.c_p['crop_width']

        # List to collect crops
        crops = []

        # Loop through predicted positions to collect crops
        for pos in self.c_p['predicted_particle_positions']:
            x, y = int(pos[0]), int(pos[1])

            # Check if the crop is within the image
            if 0 <= x - width < x + width <= image_width and 0 <= y - width < y + width <= image_height:
                try:
                    crop = self.c_p['image'][y - width:y + width, x - width:x + width].astype(np.float32)
                    crop /= 2 * np.std(crop)
                    crop -= np.mean(crop)
                    crop = np.reshape(crop, (128, 128, 1))
                    crops.append(crop)
                except ValueError as e:
                    print(e) # Most likely the shape of the image changed during prediction, no worries.
                    pass

        # Convert list of crops to a tensor and prepare for the model
        if crops:  # Check if there are any crops to process
            crops_tensor = torch.tensor(crops, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():  # No gradients needed for inference
                predictions = self.c_p['z-model'](crops_tensor)
            z_vals = predictions.squeeze().tolist()  # Convert predictions to a list
        else:
            z_vals = []
    
        # Convert to list of z-values
        #if not isinstance(z_vals, (list, tuple, np.ndarray)):
            # If it's not, convert it to an array
            # This will work for single numbers, converting them into an array with one element
        #    input_variable = np.array([z_vals])
            
        if isinstance(z_vals, float):
            z_vals = np.array([z_vals])
        self.c_p['z-predictions'] = z_vals
        

    def run(self):
        
        while self.c_p['program_running']:
            # By default check a central square of the frame. Maybe even have a ROI for this thread
            if self.c_p['model'] is not None and self.c_p['tracking_on']:
                data = self.c_p['image']
                self.c_p['predicted_particle_positions'] = self.make_prediction(np.copy(data)) # TODO test if this copy is needed
                if self.c_p['z-tracking']:
                    self.predict_z_positions()
            else:
                sleep(0.1)
            if self.c_p['locate_pipette']:
                self.locate_pipette()
            else:
                self.c_p['pipette_located'] = False # TODO check if we should really reset this all the time
            if self.c_p['train_new_model']:
                print("training new model")
                self.train_new_model(self.c_p['training_image'])
                self.c_p['train_new_model'] = False
            

class PlotParticleProfileWidget(QMainWindow):
    """
    Helps plot the two particle profiles in real time to compare the z-positions.
    Will need to update this to make sure that it works properly.
    
    """

    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p   # Control parameters
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.setWindowTitle('Particle profiles')

        self.image_idx = 0
        self.filename = "TrappedAndPipetteimage_"

        trappped_center = (200,204) # Placeholder
        pipette_center = (204,323)

        self.centers = [trappped_center, pipette_center]
        self.particle_width = 60

        self.x = list(range(100))  # 100 time points
        self.y = [np.random.normal() for _ in range(100)]  # 100 data points
        self.y2 = [np.random.normal() for _ in range(100)]

        self.graphWidget.setBackground('w')
        self.pen1 = pg.mkPen(color=(255, 0, 0))
        self.data_line1 = self.graphWidget.plot(self.x, self.y, pen=self.pen1)

        self.pen2 = pg.mkPen(color=(0, 255, 0))
        self.data_line2 = self.graphWidget.plot(self.x, self.y2, pen=self.pen2)
        


        self.timer = QTimer()
        self.timer.setInterval(200)  # Update interval in milliseconds
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()



    def update_plot_data(self):
        if len(self.c_p['predicted_particle_positions']) < 2:
            return

        self.x = np.linspace(-self.particle_width, self.particle_width, 2*self.particle_width)
        
        for idx, pos in enumerate(self.c_p['predicted_particle_positions']):

            if idx>=2:
                break
            if idx==0:
                center_1 = [int(pos[0]), int(pos[1])]
            else:
                center_2 = [int(pos[0]), int(pos[1])]
        # TODO check if the image has the right shape
        self.y = self.c_p['image'][center_1[1],
                            center_1[0]-self.particle_width:
                            center_1[0]+self.particle_width]
    
        # TODO set the legend to indicate where the particle is located, also have the software detect which particle is
        # in the trap and which is not.
        self.data_line1.setData(self.x, self.y)  # Update the data.
        self.y2 = self.c_p['image'][center_2[1],
                            center_2[0]-self.particle_width:
                            center_2[0]+self.particle_width]
        self.data_line2.setData(self.x, self.y2)

        # print(F" {np.mean((self.y-self.y2)**2)}")

class DeepLearningControlWidget(QWidget):
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        
        # self.label = QLabel("Deep learning controller")
        # layout.addWidget(self.label)
        self.setWindowTitle("Deep learning controller")

        self.toggle_tracking_button = QPushButton('Tracking on')
        self.toggle_tracking_button.pressed.connect(self.toggle_tracking)
        self.toggle_tracking_button.setCheckable(True)
        self.toggle_tracking_button.setChecked(self.c_p['tracking_on'])
        layout.addWidget(self.toggle_tracking_button)
        """
        self.training_image_button = QPushButton('Display training image')
        self.training_image_button.pressed.connect(self.show_training_image)
        self.training_image_button.setCheckable(False)
        layout.addWidget(self.training_image_button)

        self.save_network_button = QPushButton('Save network')
        self.save_network_button.pressed.connect(self.save_network)
        self.save_network_button.setCheckable(False)
        layout.addWidget(self.save_network_button)
        """

        self.load_pytorch_unet_button = QPushButton('Load pytorch U-Net')
        self.load_pytorch_unet_button.pressed.connect(self.load_pytorch_unet)
        self.load_pytorch_unet_button.setCheckable(False)
        layout.addWidget(self.load_pytorch_unet_button)

        self.locate_pipette_button = QPushButton('Locate pipette')
        self.locate_pipette_button.pressed.connect(self.locate_pipette)
        self.locate_pipette_button.setCheckable(True)
        self.locate_pipette_button.setChecked(self.c_p['locate_pipette'])
        self.locate_pipette_button.setToolTip("Locate the pipette tip in the image")
        layout.addWidget(self.locate_pipette_button)

        self.slider_label = QLabel("Set the cut-off for tracking")
        layout.addWidget(self.slider_label)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        #self.threshold_slider.setOrientation(1)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(400)
        self.threshold_slider.setValue(int(self.c_p['cutoff']))
        self.threshold_slider.valueChanged.connect(self.set_threshold)
        self.threshold_slider.setToolTip("Set the threshold for the particle detection")
        layout.addWidget(self.threshold_slider)

        self.openParticleProfileButton = QPushButton('Open particle profile')
        self.openParticleProfileButton.pressed.connect(self.openPlotWindow)
        self.openParticleProfileButton.setCheckable(False)
        layout.addWidget(self.openParticleProfileButton)

        self.load_z_model_button = QPushButton('Load z-model')
        self.load_z_model_button.pressed.connect(self.load_z_model)
        self.load_z_model_button.setCheckable(False)
        layout.addWidget(self.load_z_model_button)

        self.toggle_z_tracking_button = QPushButton('Z-tracking on')
        self.toggle_z_tracking_button.pressed.connect(self.toggle_z_tracking)
        self.toggle_z_tracking_button.setCheckable(True)
        self.toggle_z_tracking_button.setChecked(self.c_p['z-tracking'])
        layout.addWidget(self.toggle_z_tracking_button)

        self.scale_label = QLabel("Set the prescale factor for tracking")
        layout.addWidget(self.scale_label)

        self.prescale_factor_spinbox = QDoubleSpinBox()
        self.prescale_factor_spinbox.setRange(0.4, 3)
        self.prescale_factor_spinbox.setSingleStep(0.1)
        self.prescale_factor_spinbox.setValue(self.c_p['prescale_factor'])
        self.prescale_factor_spinbox.valueChanged.connect(self.set_tracking_prescale_factor)
        self.prescale_factor_spinbox.setToolTip("Set the particle scale factor for the tracking")
        layout.addWidget(self.prescale_factor_spinbox)

        self.setLayout(layout)

    def set_threshold(self, threshold):
        self.c_p['cutoff'] = threshold

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

    def load_z_model(self):
        network_name = QFileDialog.getOpenFileName(self, 'Load network')
        print(f"Opening network {network_name[0]}")
        try:
            self.c_p['z-model'] = torch.load(network_name[0])
        except Exception as e:
            print(e)
            print("Could not load model")

    def locate_pipette(self):
        self.c_p['locate_pipette'] = not self.c_p['locate_pipette']

    def toggle_tracking(self):
        self.c_p['tracking_on'] = not self.c_p['tracking_on']

    def toggle_z_tracking(self):
        self.c_p['z-tracking'] = not self.c_p['z-tracking']

    def set_tracking_prescale_factor(self, scale):
        self.c_p['prescale_factor'] = scale

    def openPlotWindow(self):
        self.plotWindow = PlotParticleProfileWidget(self.c_p)
        self.plotWindow.show()
 
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
        
