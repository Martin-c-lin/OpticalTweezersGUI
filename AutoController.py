import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
)
from PyQt6 import  QtGui

from CustomMouseTools import MouseInterface
from threading import Thread
from time import sleep, time
from queue import PriorityQueue


# A* algorthm can be found at  https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
# Can also just use this version which is a blend of chatgpt and my own code.

directions = [(0, 1), (1, 0), (1, 1), (-1, 1),(1, -1),(-1, -1), (0, -1), (-1, 0)]  # Up, Right, Down, Left

def heuristic(point, end):
    return abs(point[0] - end[0]) + abs(point[1] - end[1])

def a_star(grid, start, end):
    pq = PriorityQueue()
    pq.put((0, start))  # The queue stores tuples (priority, point)
    
    came_from = {start: None}  # Dictionary to store the path
    cost_so_far = {start: 0}  # Dictionary to store current cost
    
    while not pq.empty():
        _, current = pq.get()
        
        if current == end:
            break
        
        for direction in directions:
            next_cell = (current[0] + direction[0], current[1] + direction[1])
            
            if (0 <= next_cell[0] < len(grid) and 0 <= next_cell[1] < len(grid[0]) and grid[next_cell[0]][next_cell[1]]):
                new_cost = cost_so_far[current] + 1
                
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + heuristic(next_cell, end)
                    pq.put((priority, next_cell))
                    came_from[next_cell] = current
    
    # Build the path from end to start
    path = []
    current = end
    while current is not None:
        path.append(current)
        try:
            current = came_from[current]
        except KeyError as ke:
            return None
    path.reverse()
    
    return path

def generate_move_map(size, width, height, positions, radii, start_pos=None):
    y_size = int(size * height/width)
    area = np.ones((size,y_size))
    # Convert from pixel to grid size
    #norm_positions = positions/width * size
    radii = radii / width * size 
    for pos in positions:
        x = pos[0]/width * size
        y = pos[1]/height * y_size
        x_min = max(int(x - radii), 0)
        x_max = min(int(x + radii + 1.5), size)
        y_min = max(int(y - radii), 0)
        y_max = min(int(y + radii + 1.5), y_size)
        
        #print("setting area to 1",x_min,x_max,y_min,y_min)
        area[x_min:x_max, y_min:y_max] = 0
    return area# , [start_pos[0]/width * size, start_pos[1]/height * y_size]

def simplify_path(path):
    """ Simplifies a given path by removing unnecessary points """
    simplified_path = [path[0]]

    for i in range(1, len(path)-1):
        dx1 = path[i][0] - path[i-1][0]
        dy1 = path[i][1] - path[i-1][1]
        dx2 = path[i+1][0] - path[i][0]
        dy2 = path[i+1][1] - path[i][1]

        # If the direction changed, add this point
        if dx1 != dx2 or dy1 != dy2:
            simplified_path.append(path[i])

    simplified_path.append(path[-1])  # Always add the last point

    return simplified_path



class AutoControlWidget(QWidget):
    def __init__(self, c_p, data_channels):
        # TODO ensure that the buttons are synced with what is actually happening.
        # Maybe have the widget poll the different parameters and update the buttons accordingly.
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
        self.center_pipette_button.pressed.connect(self.toggle_center_pipette)
        self.center_pipette_button.setCheckable(True)
        self.center_pipette_button.setChecked(self.c_p['center_pipette'])
        layout.addWidget(self.center_pipette_button)

        self.attach_DNA_button = QPushButton('Attach DNA')
        self.attach_DNA_button.pressed.connect(self.attach_DNA)
        self.attach_DNA_button.setCheckable(True)
        self.attach_DNA_button.setChecked(self.c_p['attach_DNA_automatically'])
        layout.addWidget(self.attach_DNA_button)

        self.calculate_laser_position_button = QPushButton('Calculate laser position')
        self.calculate_laser_position_button.pressed.connect(self.update_laser_position)
        self.calculate_laser_position_button.setToolTip("Calculates the true laser position based on the predicted particle positions. \n This is done by finding the particle closest to the laser position and setting the laser position to that particle.")
        self.calculate_laser_position_button.setCheckable(False)
        layout.addWidget(self.calculate_laser_position_button)

        self.set_pippette_location_chamber_button = QPushButton('Set pipette location')
        self.set_pippette_location_chamber_button.pressed.connect(self.set_pippette_location_chamber)
        self.set_pippette_location_chamber_button.setToolTip("Sets the approximate location of the pipette to the current motor position.\n Default is that the pipette is at 0,0 of the motors. Use before trying to locate the pipette.")
        layout.addWidget(self.set_pippette_location_chamber_button)

        self.set_tube_AD_tube_position_button = QPushButton('Set tube AD tube position')
        self.set_tube_AD_tube_position_button.pressed.connect(self.set_tube_AD_tube_position)
        self.set_tube_AD_tube_position_button.setToolTip("Sets an approximate position of the AD tube- \n Specifically a position in which it is easy to see beads flowing by.")
        layout.addWidget(self.set_tube_AD_tube_position_button)

        self.move_while_avoiding_beads_button = QPushButton('Move while avoiding beads')
        self.move_while_avoiding_beads_button.pressed.connect(self.move_while_avoiding)
        self.move_while_avoiding_beads_button.setToolTip("Moves the pipette to the specified location while avoiding beads. \n This is done by generating a map of where the beads are and then using A* to find a path to the destination.")
        self.move_while_avoiding_beads_button.setCheckable(True)
        self.move_while_avoiding_beads_button.setChecked(self.c_p['move_avoiding_particles'])
        layout.addWidget(self.move_while_avoiding_beads_button)

        self.EP_toggled_button = QPushButton('Toggle EP')
        self.EP_toggled_button.pressed.connect(self.toggle_EP)
        self.EP_toggled_button.setCheckable(True)
        self.EP_toggled_button.setChecked(self.c_p['electrostatic_protocol_toggled'])
        layout.addWidget(self.EP_toggled_button)


        self.EP_start_spinbox = QSpinBox()
        self.EP_start_spinbox.setRange(0, 65535)
        self.EP_start_spinbox.valueChanged.connect(self.update_EP_start)
        self.EP_start_spinbox.setValue(self.c_p['electrostatic_protocol_start'])
        layout.addWidget(QLabel("EP start:"))
        layout.addWidget(self.EP_start_spinbox)

        self.EP_end_spinbox = QSpinBox()
        self.EP_end_spinbox.setRange(0, 65535)
        self.EP_end_spinbox.valueChanged.connect(self.update_EP_end)
        self.EP_end_spinbox.setValue(self.c_p['electrostatic_protocol_end'])
        layout.addWidget(QLabel("EP end:"))
        layout.addWidget(self.EP_end_spinbox)

        self.EP_step_spinbox = QSpinBox()
        self.EP_step_spinbox.setRange(0, 65535)
        self.EP_step_spinbox.valueChanged.connect(self.update_EP_step)
        self.EP_step_spinbox.setValue(self.c_p['electrostatic_protocol_steps'])
        layout.addWidget(QLabel("EP step:"))
        layout.addWidget(self.EP_step_spinbox)

        self.EP_duration_spinbox = QSpinBox()
        self.EP_duration_spinbox.setRange(0, 65535)
        self.EP_duration_spinbox.valueChanged.connect(self.update_EP_duration)
        self.EP_duration_spinbox.setValue(self.c_p['electrostatic_protocol_duration'])
        layout.addWidget(QLabel("EP duration:"))
        layout.addWidget(self.EP_duration_spinbox)

        self.setLayout(layout)

    def toggle_EP(self):
        if not self.EP_toggled_button.isChecked() and not self.c_p['electrostatic_protocol_toggled']:
            self.c_p['electrostatic_protocol_toggled'] = not self.c_p['electrostatic_protocol_toggled']
        #print(self.c_p['electrostatic_protocol_toggled'])
        if self.EP_toggled_button.isChecked() and self.c_p['electrostatic_protocol_toggled']:
            self.c_p['electrostatic_protocol_toggled'] = not self.c_p['electrostatic_protocol_toggled']
            self.c_p['electrostatic_protocol_finished'] = False
            self.c_p['electrostatic_protocol_running'] = False

        if self.c_p['electrostatic_protocol_toggled']:
            self.c_p['electrostatic_protocol_finished'] = False
        print("Toggled EP to ", self.c_p['electrostatic_protocol_toggled'], self.EP_toggled_button.isChecked())
    
    def update_EP_start(self):
        val = int(self.EP_start_spinbox.value())
        # if val > self.c_p['electrostatic_protocol_end']:
        #     self.EP_start_spinbox.setValue(self.c_p['electrostatic_protocol_start'])
        #    return
        self.c_p['electrostatic_protocol_start'] = val

    def update_EP_end(self):
        val = int(self.EP_end_spinbox.value())
        # if val < self.c_p['electrostatic_protocol_start']:
        #     self.EP_end_spinbox.setValue(self.c_p['electrostatic_protocol_end'])
        #     return
        self.c_p['electrostatic_protocol_end'] = val

    def update_EP_step(self):
        val = int(self.EP_step_spinbox.value())
        self.c_p['electrostatic_protocol_steps'] = val

    def update_EP_duration(self):
        val = int(self.EP_duration_spinbox.value())
        self.c_p['electrostatic_protocol_duration'] = val

    def update_laser_position(self):
        self.c_p['find_laser_position'] = True

    def center_particle(self):
        """
        Function that seeks to keep a particle in the center 
        of the image. 
        """
        self.c_p['centering_on'] = not self.c_p['centering_on']

    def attach_DNA(self):
        self.c_p['attach_DNA_automatically'] = not self.attach_DNA_button.isChecked()

    
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

    def toggle_center_pipette(self):
        """
        Moves the stage so that the pipette is in the center of the image
        """
        self.c_p['center_pipette'] = not self.center_pipette_button.isChecked()

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

    def set_tube_AD_tube_position(self):
        x = self.data_channels['Motor_x_pos'].get_data(1)[0]
        y = self.data_channels['Motor_y_pos'].get_data(1)[0]
        z = self.data_channels['Motor_z_pos'].get_data(1)[0]
        self.c_p['AD_tube_position'] = [x, y, z]

    def move_while_avoiding(self):

        self.c_p['move_avoiding_particles'] = not self.c_p['move_avoiding_particles']



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
        return "Click on the screen where the laser is located\n Used to tell the auto controll functions where to expect particles to be trappable."

class autoControllerThread(Thread):
    def __init__(self, c_p, data_channels, main_window=None  ):
        super().__init__()
        self.c_p = c_p
        self.setDaemon(True)
        self.particles_in_view = False
        self.data_channels = data_channels
        self.search_direction = 1 
        self.y_lim_pos = 1 # search limits
        self.x_lim_pos = 0.1
        self.main_window = main_window

        # DNA attachment parameters
        self.DNA_move_direction = 0
        self.bits_per_pixel = 500/self.c_p['microns_per_pix'] # Number of bits we need to change the piezo dac to move 1 micron, approximate
        self.DNA_length_pix = 160  # approximate particle-particle-distance at which we should see a force ramp in the DNA
        self.closest_distance = 70
        self.sleep_counter = 0
        self.force_limit = 30
  
        # Parameters for the electrostatic protocol experiments

        # Explanation of parameters:
        # pipette_located - Have the pipette been located in the latest image?
        # center_pipette - Should the pipette be centered close to the laser position?
        # 


    def find_closest_particle(self, center):
        if len(self.c_p['predicted_particle_positions']) == 0:
            return None

        # Find particle closest to the center
        distances = [(x-center[1])**2+(y-center[0])**2 for x,y in self.c_p['predicted_particle_positions']] # Error with axis before
        return np.argmin(distances)

    def move_while_avoiding_particles(self):
        """
        Function that moves the stage while avoiding double trapping particles. 
        """

        if not self.data_channels['particle_trapped'].get_data(1)[0]:
            # No particle trapped so no point in avoiding particles.
            print("No particle trapped so no point in avoiding particles.")
            return
        x0 = self.data_channels['Motor_x_pos'].get_data(1)[0]
        y0 = self.data_channels['Motor_y_pos'].get_data(1)[0]
        dx = self.c_p['pipette_location_chamber'][0] - x0
        dy = self.c_p['pipette_location_chamber'][1] - y0

        move_lim = 100 # Max movemement per "step"

        if dx**2+dy**2 < 200:
            self.c_p['move_to_location'] = False
            return True

        if len( self.c_p['predicted_particle_positions']) == 1:
            # Only one particle in view so no point in avoiding particles.
            # move towards pipette
            if dx<-move_lim:
                dx = -move_lim
            elif dx>move_lim:
                dx = move_lim
            if dy<-move_lim:
                dy = -move_lim
            elif dy>move_lim:
                dy = move_lim
            self.c_p['minitweezers_target_pos'][0] = int(x0 + dx)
            self.c_p['minitweezers_target_pos'][1] = int(y0 + dy)
            print("Updating target position to ", self.c_p['minitweezers_target_pos'])
            self.c_p['move_to_location'] = True
            return

        size = 100
        radii = 150
        image_shape = np.shape(self.c_p['image']) # TODO may give error if changing size of AOI
        #print("Generating movemment map ", image_shape,self.c_p['predicted_particle_positions'])

        # Calculate a distance matrix for positions of particles that are not the trapped one
        LX = self.c_p['laser_position'][0]
        LY = self.c_p['laser_position'][1]
        distances = [[(x-LX),(y-LY)] for x,y in self.c_p['predicted_particle_positions']]
        
        # Pop the trapped particle to remove it from the area check.
        distances_squared = [(x-LX)**2+(y-LY)**2 for x,y in self.c_p['predicted_particle_positions']]
        positions = np.copy(self.c_p['predicted_particle_positions'])
        positions = np.delete(positions, np.argmin(distances_squared), axis=0)
        area = generate_move_map(size, image_shape[1], image_shape[0], positions, radii, None)

        s = np.shape(area)
        # Assume that we start with particle in middle
        start = int(s[0]/2),int(s[1]/2)
        
        # Find the correct target positions
        
        end_pos = [s[0]-1, s[1]-1]
        if dx<0:
            end_pos = [0, end_pos[1]]
        if dy<0:
            end_pos = [end_pos[0], 0]
        if np.abs(dx)<move_lim/4:
            end_pos = [start[0], end_pos[1]]
        if np.abs(dy)<move_lim/4:
            end_pos = [end_pos[0], start[1]]

        # Find the path
        start = tuple(start)
        end_pos = tuple(end_pos)
        path = a_star(area, start, end_pos)


        if path is None:
            print("No path found"," searched  path from ", start, " to ", end_pos)
            #print(area)
            return 
        path = simplify_path(path)
        print("Finding path from ", start, " to ", end_pos, "pAth is ", path)

        # Convert target positiont to motor ticks
        factor = image_shape[0]/size * self.c_p['ticks_per_pixel']
        self.c_p['minitweezers_target_pos'][0] = int(x0 + (path[1][0]-start[0])*factor)
        self.c_p['minitweezers_target_pos'][1] = int(y0 + (path[1][1]-start[1])*factor)
        print("Smart move to ", self.c_p['minitweezers_target_pos'])
        self.c_p['move_to_location'] = True
        sleep(0.2)

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
        
        #print(dx,dy)
        target_x_pos = int(self.data_channels['Motor_x_pos'].get_data(1)[0] - dx)
        target_y_pos = int(self.data_channels['Motor_y_pos'].get_data(1)[0] + dy) # Offest since we don't want to collide with the pipette.
        # TODO add particles located.
        self.c_p['minitweezers_target_pos'] = [target_x_pos, target_y_pos, self.c_p['minitweezers_target_pos'][2]]
        self.c_p['move_to_location'] = True

    def find_movement_direction(self, margin=10):
        if not self.c_p['move_to_location']:
            return 0
        dx = self.data_channels['Motor_x_pos'].get_data(1)[0] - self.c_p['minitweezers_target_pos'][0]
        dy = self.data_channels['Motor_y_pos'].get_data(1)[0] - self.c_p['minitweezers_target_pos'][1]

        if dx**2<margin**2 and dy**2<margin**2:
            return 0
        
        if dx>=margin and dy>=margin:
            return 1
        elif dx>=margin and -margin<=dy<=margin:
            return 2
        elif dx>=margin and dy<=-margin:
            return 3
        elif -margin<=dx<=margin and dy<=-margin:
            return 4
        elif dx<=-margin and dy<=-margin:
            return 5
        elif dx<=-margin and -margin<=dy<=margin:
            return 6
        elif dx<=-margin and dy>=margin:
            return 7
        elif -margin<=dx<=margin and dy>=margin:
            return 8

        
    def check_if_path_clear(self, margin=10):
        dir = self.find_movement_direction(margin=margin)
        if dir == 0 or len(self.c_p['predicted_particle_positions'])<2: # At most one particle (the trapped one) in view
            return True
        
        # Calculate a distance matrix for positions of particles that are not the trapped one
        LX = self.c_p['laser_position'][0]
        LY = self.c_p['laser_position'][1]
        distances = [[(x-LX),(y-LY)] for x,y in self.c_p['predicted_particle_positions']]
        
        # Pop the trapped particle to remove it from the check.
        distances_squared = [(x-LX)**2+(y-LY)**2 for x,y in self.c_p['predicted_particle_positions']]
        distances.pop(np.argmin(distances_squared))

    def center_pipette(self, offset_pixels = 20):
        """
        In the long run this will need to be done also with the path search to avoid bumping the particles into the pipette.

        
        """
        
        dx = self.c_p['pipette_location_chamber'][0] - self.data_channels['Motor_x_pos'].get_data(1)[0]
        dy = self.c_p['pipette_location_chamber'][1] - self.data_channels['Motor_y_pos'].get_data(1)[0]
        
        # Start by locating the pipette roughly
        if dx**2+dy**2 > 1000_000: # If more than ca 500 pixels away then we are too far off
            print("Too far from pipette location")
            # Move to locaiton
            self.c_p['center_pipette'] = False
            return
        
        if not self.c_p['locate_pippette']:
            self.c_p['locate_pippette'] = True
            print("Locating pipette in image")
            sleep(0.2)
        # Check it's location in the frame and that we actually have a pipette to move to. 
        if not self.c_p['pipette_located'] or self.c_p['pipette_location'][0]:
            return
        
        # Move the stage
        # Does not take into account the zoom in correctly.
        dx_i = (self.c_p['laser_position'][0] - (self.c_p['pipette_location'][0] + self.c_p['AOI'][0])) * self.c_p['ticks_per_pixel']
        dy_i = (self.c_p['laser_position'][1] - (self.c_p['pipette_location'][1] + self.c_p['AOI'][2])-offset_pixels) * self.c_p['ticks_per_pixel']
        
        if dx_i**2+dy_i**2 < 100:
            self.c_p['center_pipette'] = False
            print("Pipette centered")
            return
        # TODO Move along y first to avoid colliding with the pipette
        target_x_pos = int(self.data_channels['Motor_x_pos'].get_data(1)[0] - dx_i)
        target_y_pos = int(self.data_channels['Motor_y_pos'].get_data(1)[0] + dy_i) # Offest since we don't want to collide with the pipette.

        self.c_p['minitweezers_target_pos'] = [target_x_pos, target_y_pos, self.c_p['minitweezers_target_pos'][2]]
        print(f"Should move to {self.c_p['minitweezers_target_pos']} to center the pipette. {dx_i} {dy_i}")
        self.c_p['move_to_location'] = True

    def check_trapped(self, threshold=10_000):
        # TODO threshold is a bit big I think.
        # TODO does not work when zoomed in
        if len(self.c_p['predicted_particle_positions'])<1:
            self.particles_in_view = False
            return False
        self.particles_in_view = True

        LX = self.c_p['laser_position'][0] - self.c_p['AOI'][0]
        LY = self.c_p['laser_position'][1] - self.c_p['AOI'][2]
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


    def custom_RBC_protocol(self):
        pass

    def initiate_electrostatic_protocol(self):
        self.EP_positions = np.linspace(self.c_p['electrostatic_protocol_start'],
                                        self.c_p['electrostatic_protocol_end'],
                                        self.c_p['electrostatic_protocol_steps'],
                                        dtype=int)
        self.c_p['electrostatic_protocol_running'] = True
        self.c_p['electrostatic_protocol_steps'] = 0
        self.c_p['electrostatic_protocol_finished'] = False
        self.measurement_start_time = time()
 
    def custom_electrostatic_protocol(self):

        if not self.c_p['electrostatic_protocol_running']:
            print("Initiating protocol")
            self.initiate_electrostatic_protocol()

        # Check if measurement needs to be updated
        if time() - self.measurement_start_time < self.c_p['electrostatic_protocol_duration']:
            return
        
        # Update the piezo position
        self.c_p['piezo_B'][1] = self.EP_positions[self.c_p['electrostatic_protocol_steps']]

        print(self.EP_positions[self.c_p['electrostatic_protocol_steps']])

        # Toggle autoalign A for a short time
        self.c_p['portenta_command_2'] = 1
        sleep(0.5)
        # Need to reset the piezo position to the autoaligned one
        self.c_p['piezo_A'] = np.int32([np.mean(self.data_channels['dac_ax'].get_data_spaced(10)),
                                np.mean(self.data_channels['dac_ay'].get_data_spaced(10))])
        self.c_p['portenta_command_2'] = 0

        self.c_p['electrostatic_protocol_steps'] += 1
        if self.c_p['electrostatic_protocol_steps'] >= len(self.EP_positions):
            self.c_p['electrostatic_protocol_running'] = False
            self.c_p['electrostatic_protocol_finished'] = True
            # TODO make it turn of automatically
            return

        # Update measurement start-time
        self.measurement_start_time = time()
        

    def find_true_laser_position(self):
        if not self.check_trapped() or not self.c_p['tracking_on']:
            print("No particles detected close enough to current laser position")
            return
        LX = self.c_p['laser_position'][0] - self.c_p['AOI'][0]
        LY = self.c_p['laser_position'][1] - self.c_p['AOI'][2]

        min_pos = self.find_closest_particle([LX, LY])
        #min_x = 1000
        #min_dist = 2e10
        #min_y = 1000
        #for x,y in self.c_p['predicted_particle_positions']:
        #    if (x-LX)**2+(y-LY)**2<min_dist:
        #        min_dist = (x-LX)**2+(y-LY)**2
        #        min_x = x
        #        min_y = y
        self.c_p['laser_position'] = [min_pos[0]+self.c_p['AOI'][0], min_pos[1]+self.c_p['AOI'][2]]
        print("Laser position set to: ", self.c_p['laser_position'])

    def find_closest_particle(self,reference_position):
        try:
            LX = reference_position[0]
            LY = reference_position[1]
            min_x = 1000
            min_dist = 2e10
            min_y = 1000
            for x,y in self.c_p['predicted_particle_positions']:
                if (x-LX)**2+(y-LY)**2<min_dist:
                    min_dist = (x-LX)**2+(y-LY)**2
                    min_x = x
                    min_y = y
            return [min_x, min_y]
        except Exception as e:
            return None
        
    def find_particle_in_pippette(self, offset=0):
        if not self.c_p['tracking_on'] and self.c_p['locate_pippette'] and self.c_p['pipette_located']:
            return # Need to have the trackings turned on to do this 
        if len(self.c_p['predicted_particle_positions']) == 0:
            return
        if self.c_p['pipette_location'] is None or self.c_p['pipette_location'][1] is None:
            return
        min_pos = self.find_closest_particle([self.c_p['pipette_location'][0],self.c_p['pipette_location'][1]-offset])
        if min_pos is None:
            return
        dx = (min_pos[0]-self.c_p['pipette_location'][0])
        dy = (min_pos[1]-self.c_p['pipette_location'][1])
        if dx**2> 1000:
            print("Particle too far away from pipette", dx, dy, min_pos)
            return
        if dy**2> 10_000:
            print("Particle too far away from pipette", dx, dy, min_pos)
            return
        return min_pos
    
    def attach_DNA(self):
        # Check that the two necessary particles are in view, someitmes the bead in the pipette is misstaken for part of the pipette, thereof the offset.
        pipette_particle = self.find_particle_in_pippette(offset=0)
        if pipette_particle is None:
            print("No particle in pipette")
            return
        if not self.check_trapped():
            print("No particle trapped")
            return
        if not self.c_p['portenta_command_2'] == 1:
            print("Not autoaligned")
            return 
        self.find_true_laser_position()

        dx = ((self.c_p['laser_position'][0]-self.c_p['AOI'][0]) - pipette_particle[0])
        dy = -((self.c_p['laser_position'][1]-self.c_p['AOI'][2]) - pipette_particle[1])

        # Adjust DX
        print("DX",self.c_p['laser_position'][0],self.c_p['AOI'][0],pipette_particle[0])
        print("DY",self.c_p['laser_position'][1],self.c_p['AOI'][2],pipette_particle[1],dy)

        if np.abs(dy)< 0.1 and np.abs(dx)< 1:
            return

        if dy<self.closest_distance-20:
            self.c_p['piezo_B'][1] = max(0,self.c_p['piezo_B'][1] - 200)
            if self.c_p['piezo_B'][1] == 0:
                # Error which we cannot fix from here

                return False
            return
        # TODO do a proper check if we can reasonably move to the correct location
        if dx>3:
            self.c_p['piezo_B'][0] = min(2**16-1,self.c_p['piezo_B'][0]+min(dx*self.bits_per_pixel, 400))
            return
        if dx<-3:
            self.c_p['piezo_B'][0] = max(0,self.c_p['piezo_B'][0]+max(dx*self.bits_per_pixel, -400))
            return
        
        
        if dy>self.DNA_length_pix and self.data_channels['F_total_Y'].get_data(1)[0] > self.force_limit:
            print("DNA found and attached")
            return True
        # Need to handle situation in which we start out with the bead in the trap being lower than that of the pipette.
        if self.DNA_move_direction == 0:
            # Move towards the pipette bead
            if dy<self.closest_distance:

                # Take a break of ca 10 seconds to let the DNA attach before separating the beads
                if self.sleep_counter < 100:
                    self.sleep_counter += 1
                    return
                self.sleep_counter = 0
                self.DNA_move_direction = 1
                return
            self.c_p['piezo_B'][1] = min(2**16-1,self.c_p['piezo_B'][1]+ 200)

        elif self.DNA_move_direction == 1:
            # Move away from the pipette bead            
            self.c_p['piezo_B'][1] = max(0,self.c_p['piezo_B'][1] - 200)
            if dy>self.DNA_length_pix+10:
                # Moved to far let's turn down
                self.DNA_move_direction = 0
                return
        # 

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
                self.center_pipette()
            elif self.c_p['move_avoiding_particles']:
                self.move_while_avoiding_particles()
            elif self.c_p['electrostatic_protocol_toggled'] and not self.c_p['electrostatic_protocol_finished']:
                self.custom_electrostatic_protocol()
            elif self.c_p['find_laser_position']:
                self.find_true_laser_position()
                print(self.find_particle_in_pippette())
                self.c_p['find_laser_position'] = False
            elif self.c_p['attach_DNA_automatically']:
                self.attach_DNA()

            self.data_channels['particle_trapped'].put_data(trapped)
            sleep(0.1)