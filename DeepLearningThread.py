# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:48:59 2023

@author: marti
"""

import deeptrack as dt
import numpy as np
from threading import Thread
import cv2

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
        if np.shape(training_data)[0] > 64:
            self.rescale_factor = 64 / np.shape(training_data)[0]
            training_data = cv2.rescale(training_data, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
        self.model.fit(training_data, epochs=30, batch_size=8) # Default

    def load_model(self, model_path):
        pass
    
    def save_model(self):
        pass

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
        if s[0] < s[1]:
            d = data
        elif s[0]>s[1]:
            d = data
        # rescale data
        
        positions = self.model.predict_and_detect(data, predict_kwargs)# TODO have alpha, cut_off etc adaptable.
        return positions

    def run(self):
        
        while self.c_p['program_running']:
            # By default check a central square of the frame. Maybe even have a ROI for this thread
            if self.model is not None and self.c_p['live_tracking']:
                self.c_p['predicted_particle_positions'] = self.make_prediction()
         
    
        