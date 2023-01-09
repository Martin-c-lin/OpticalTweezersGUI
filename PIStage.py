# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:44:24 2022

@author: Martin
"""

from pipython import GCSDevice
from threading import Thread


class PIStageThread(Thread):
    
    
    def __init__(self , threadID, name, c_p, serialnum='111029773'):
        Thread.__init__(self)
        self.name = name
        self.c_p = c_p
        # Connect the device
        self.pi_device = GCSDevice()
        self.pi_device.ConnectUSB(serialnum)
        # Turn on the servos
        self.pi_device.SVO([1,2,3],[True, True, True])
        self.set_position()
        
    def set_position(self):
        pos = self.pi_device.qPOS()
        self.c_p['piezo_pos'] = [pos['1'], pos['2'], pos['3']]
        
    def run(self):
        
        while self.c_p['program_running']:
            self.set_position()
            #for idx, target_pos in enumerate(self.c_p['piezo_targets']):
            self.pi_device.MOV([1, 2, 3], self.c_p['piezo_targets'])
                
