import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
)
from PyQt6 import  QtGui

from CustomMouseTools import MouseInterface
from threading import Thread
from time import sleep

class AutoControlWidget(QWidget):
    def __init__(self, c_p, data_channels):
        super().__init__()
        self.c_p = c_p
        self.data_channels = data_channels
        layout = QVBoxLayout()
        self.setWindowTitle("Auto Controller")


        # TODO make it so that only one of these options can be selected at a time
        self.toggle_centering_button = QPushButton('Centering on')
        self.toggle_centering_button.pressed.connect(self.center_particle)
        self.toggle_centering_button.setCheckable(True)
        self.toggle_centering_button.setChecked(self.c_p['centering_on'])
        layout.addWidget(self.toggle_centering_button)

        self.toggle_trap_button = QPushButton('Trap particle')
        self.toggle_trap_button.pressed.connect(self.trap_particle)
        self.toggle_trap_button.setCheckable(True)
        self.toggle_trap_button.setChecked(self.c_p['trap_particle'])
        layout.addWidget(self.toggle_trap_button)

        self.search_and_trap_button = QPushButton('Search and trap')
        self.search_and_trap_button.pressed.connect(self.search_and_trap)
        self.search_and_trap_button.setCheckable(True)
        self.search_and_trap_button.setChecked(self.c_p['search_and_trap'])
        layout.addWidget(self.search_and_trap_button)

        self.center_pipette_button = QPushButton('Center pipette')
        self.center_pipette_button.pressed.connect(self.center_pipette)
        self.center_pipette_button.setCheckable(True)
        self.center_pipette_button.setChecked(self.c_p['center_pipette'])
        layout.addWidget(self.center_pipette_button)

        self.set_pippette_location_chamber_button = QPushButton('Set pipette location')
        self.set_pippette_location_chamber_button.pressed.connect(self.set_pippette_location_chamber)
        self.set_pippette_location_chamber_button.setToolTip("Sets the approximate location of the pipette to the current motor position.\n Default is that the pipette is at 0,0 of the motors. Use before trying to locate the pipette.")
        layout.addWidget(self.set_pippette_location_chamber_button)

        self.setLayout(layout)


    def center_particle(self):
        """
        Function that seeks to keep a particle in the center 
        of the image. 
        """
        self.c_p['centering_on'] = not self.c_p['centering_on']
    
    def trap_particle(self):
        """
        Function that seeks to keep a particle in the center 
        of the image. 
        """
        self.c_p['trap_particle'] = not self.c_p['trap_particle']
    
    def search_and_trap(self):
        """
        Function that seeks to keep a particle in the center 
        of the image. 
        """
        self.c_p['search_and_trap'] = not self.c_p['search_and_trap']

    def center_pipette(self):
        """
        Moves the stage so that the pipette is in the center of the image
        """
        self.c_p['center_pipette'] = not self.c_p['center_pipette']

        # If the button was just pressed then we should find the exact pipette location.
        if self.c_p['center_pipette']:
            self.c_p['pipette_located'] = False
        else:
            self.c_p['locate_pippette'] = False
            self.c_p['move_to_location'] = False

    def set_pippette_location_chamber(self):
        """
        Sets the approximate location of the pipette to the current motor position.
        Default is that the pipette is at 0,0 of the motors.
        Use before trying to locate the pipette.
        """
        x = self.data_channels['Motor_x_pos'].get_data(1)[0]
        y = self.data_channels['Motor_y_pos'].get_data(1)[0]
        z = self.data_channels['Motor_z_pos'].get_data(1)[0]
        self.c_p['pipette_location_chamber'] = [x, y, z]
        print("Approximate pipette location set to: ", self.c_p['pipette_location_chamber'])

class SelectLaserPosition(MouseInterface):
    def __init__(self, c_p):
        self.c_p = c_p
        self.pen = QtGui.QPen(QtGui.QColor(0, 255, 0))

    def draw(self, qp):
        qp.setPen(self.pen)
        # TODO check where image scale parameter should be applied
        r = 10
        x = int((self.c_p['laser_position'][0] - self.c_p['AOI'][0]) / self.c_p['image_scale'] - r/2)
        y = int((self.c_p['laser_position'][1] - self.c_p['AOI'][2]) / self.c_p['image_scale'] - r/2)
        qp.drawEllipse(x,y, r, r)

    def mousePress(self):
        if self.c_p['mouse_params'][0] == 1:
            self.c_p['laser_position'] = np.array(self.c_p['mouse_params'][1:3])*self.c_p['image_scale']
            self.c_p['laser_position'][0] += self.c_p['AOI'][0]
            self.c_p['laser_position'][1] += self.c_p['AOI'][2]
            print("Laser position set to: ", self.c_p['laser_position'])
    def mouseRelease(self):
        pass
    def mouseDoubleClick(self):
        pass
    def mouseMove(self):
        pass
    def getToolName(self):
        return "Laser position"
    def getToolTip(self):
        return "Click on the screen where the laser is located"

class autoControllerThread(Thread):
    def __init__(self, c_p, data_channels):
        super().__init__()
        self.c_p = c_p
        self.setDaemon(True)
        self.particles_in_view = False
        self.data_channels = data_channels
        self.search_direction = 1 
        self.y_lim_pos = 1 # search limits
        self.x_lim_pos = 0.1


    def find_closest_particle(self, center):
        if len(self.c_p['predicted_particle_positions']) == 0:
            return None

        # Find particle closest to the center
        distances = [(x-center[1])**2+(y-center[0])**2 for x,y in self.c_p['predicted_particle_positions']] # Error with axis before
        return np.argmin(distances)

    def center_particle(self, center, move_limit=10):

        #if len(self.c_p['predicted_particle_positions']) == 0:
        #    return

        # Find particle closest to the center
        # distances = [(x-center[1])**2+(y-center[0])**2 for x,y in self.c_p['predicted_particle_positions']] # Error with axis before
        #center_particle = np.argmin(distances)

        center_particle = self.find_closest_particle(center)
        if center_particle is None:
            return

        # Calculate the distance to move the stage
        dx = center[1] - self.c_p['predicted_particle_positions'][center_particle][0]
        dy = center[0] - self.c_p['predicted_particle_positions'][center_particle][1]
        if np.abs(dx)<move_limit and np.abs(dy)<move_limit:
            print(f"Limits {dx} {dy}")
            return
        # Tell motors to move
        # NOTE the minus sign here is because the camera is mounted upside down
        self.c_p['stepper_target_position'][0] = self.c_p['stepper_current_position'][0] - dx*self.c_p['microns_per_pix']
        self.c_p['stepper_target_position'][1] = self.c_p['stepper_current_position'][1] - dy*self.c_p['microns_per_pix']

    def trap_particle_minitweezers(self, center, move_limit=10):
        # Center is the laser position on which to center the particle.
        if not self.c_p['particle_prediction_made']:
            sleep(0.1)
            return
        self.c_p['particle_prediction_made'] = False

        center_particle = self.find_closest_particle(center)
        if center_particle is None:
            sleep(0.1)
            return
        dx = (center[1] - self.c_p['predicted_particle_positions'][center_particle][0])* self.c_p['ticks_per_pixel']
        dy = (center[0] - self.c_p['predicted_particle_positions'][center_particle][1])* self.c_p['ticks_per_pixel']
        if np.abs(dx)<move_limit and np.abs(dy)<move_limit:
            print(f"Limits {dx} {dy}")
            return
        
        print(dx,dy)
        target_x_pos = int(self.data_channels['Motor_x_pos'].get_data(1)[0] - dx)
        target_y_pos = int(self.data_channels['Motor_y_pos'].get_data(1)[0] + dy) # Offest since we don't want to collide with the pipette.
        # TODO add particles located.
        self.c_p['minitweezers_target_pos'] = [target_x_pos, target_y_pos, self.c_p['minitweezers_target_pos'][2]]
        self.c_p['move_to_location'] = True


    def center_pippette(self):
        dx = self.c_p['pipette_location_chamber'][0] - self.data_channels['Motor_x_pos'].get_data(1)[0]
        dy = self.c_p['pipette_location_chamber'][1] - self.data_channels['Motor_y_pos'].get_data(1)[0]
        
        # Start by locating the pipette roughly
        if dx**2+dy**2 > 1000_000: # If more than ca 500 pixels away then we are too far off
            print("Too far from pipette location")
            # Move to locaiton
            self.c_p['center_pipette'] = False
            return
        # Check it's location in the frame
        if not self.c_p['pipette_located']:
            self.c_p['locate_pippette'] = True
            return
        # Check that we actually have a pipette to move to.
        if self.c_p['pipette_location'][0] is None:
            return
        # Move the stage
        dx_i = (self.c_p['laser_position'][0] - (self.c_p['pipette_location'][0] + self.c_p['AOI'][0])) * self.c_p['ticks_per_pixel']
        dy_i = (self.c_p['laser_position'][1] - (self.c_p['pipette_location'][1] - self.c_p['AOI'][2])) * self.c_p['ticks_per_pixel']


        target_x_pos = int(self.data_channels['Motor_x_pos'].get_data(1)[0] - dx_i)
        target_y_pos = int(self.data_channels['Motor_y_pos'].get_data(1)[0] + dy_i) # Offest since we don't want to collide with the pipette.

        self.c_p['minitweezers_target_pos'] = [target_x_pos, target_y_pos, self.c_p['minitweezers_target_pos'][2]]
        print(f"Should move to {self.c_p['minitweezers_target_pos']} to center the pipette. {dx_i} {dy_i}")
        self.c_p['move_to_location'] = True
        self.c_p['pipette_located'] = False    

    def check_trapped(self, threshold=10_000):
        if len(self.c_p['predicted_particle_positions'])<1:
            self.particles_in_view = False
            return False
        self.particles_in_view = True

        LX = self.c_p['laser_position'][0]
        LY = self.c_p['laser_position'][1]
        distances = [(x-LX)**2+(y-LY)**2 for x,y in self.c_p['predicted_particle_positions']]
        return min(distances) < threshold

    def look_for_particles(self):
        if len(self.c_p['predicted_particle_positions']) > 0:
            self.particles_in_view = True
            return
        # TODO make this move in more advanced way, for instance grid search
        if self.search_direction == 1:
            self.c_p['stepper_target_position'][1] = self.c_p['stepper_current_position'][1] + 0.04
            if self.c_p['stepper_current_position'][1] > self.c_p['stepper_starting_position'][1] + self.y_lim_pos:
                self.search_direction = 2
        elif self.search_direction == 2:
            self.c_p['stepper_target_position'][0] = self.c_p['stepper_current_position'][0] + 0.04
            if self.c_p['stepper_current_position'][0] > self.c_p['stepper_starting_position'][0] + self.x_lim_pos:
                self.search_direction = 3
        elif self.search_direction == 3:
            self.c_p['stepper_target_position'][1] = self.c_p['stepper_current_position'][1] - 0.04
            if self.c_p['stepper_current_position'][1] < self.c_p['stepper_starting_position'][1] - self.y_lim_pos: # TODO errir if start is 0.
                self.search_direction = 1
        print(f"Looking for particles{self.search_direction}")


    def search_and_trap(self):
        if self.particles_in_view:
            self.center_particle(self.c_p['laser_position'], move_limit=30)

            if self.check_trapped():
                self.c_p['trap_particle'] = False
                self.c_p['centering_on'] = False
                return
        else:
            self.look_for_particles()


    def run(self):

        while self.c_p['program_running']:
            trapped = self.check_trapped()
            if self.c_p['centering_on']:
                center = [np.shape(self.c_p['image'])[0]/2, np.shape(self.c_p['image'])[1]/2]
                #self.center_particle(center)
                self.trap_particle_minitweezers(center)
            elif self.c_p['trap_particle']:
                self.center_particle(self.c_p['laser_position'], move_limit=30)
            elif self.c_p['search_and_trap']:
                if not trapped:
                    self.search_and_trap()
                else:
                    print("Particle alrady trapped")
                    self.c_p['search_and_trap'] = False
            elif self.c_p['center_pipette']:
                self.center_pippette()

            self.data_channels['particle_trapped'].put_data(trapped)
            sleep(0.1)