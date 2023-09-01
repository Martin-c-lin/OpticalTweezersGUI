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
        return "Click on the screen where the laser is located"

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
        # TODO threshold is a bit big I think.
        # TODO does not work when zoomed in
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


    def custom_RBC_protocol(self):
        pass


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
            elif self.c_p['move_avoiding_particles']:
                self.move_while_avoiding_particles()

            self.data_channels['particle_trapped'].put_data(trapped)
            sleep(0.1)