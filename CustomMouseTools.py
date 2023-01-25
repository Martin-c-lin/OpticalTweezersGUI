# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 10:28:23 2023

@author: marti
"""
import abc

class MouseInterface(metaclass=abc.ABCMeta):
    # TODO add id for each tool
    @classmethod
    def __subclasshook__(cls, subclass):
        # TODO Add a icon or similar bar
        return (hasattr(subclass, 'mouseMove') and
                callable(subclass.mouseMove) and
                hasattr(subclass, 'mousePress') and
                callable(subclass.mousePress) and
                hasattr(subclass, 'mouseRelease') and
                callable(subclass.mouseRelease) and
                hasattr(subclass, 'draw') and
                callable(subclass.draw) and
                hasattr(subclass, 'mouseDoubleClick') and
                callable(subclass.mouseDoubleClick) or
                NotImplemented)
    

class MarkPositionTool(MouseInterface):
    
    def __init__(self, c_p):
        self.c_p = c_p
        self.marker_positions = [] # Marker positions in image coordinates
        self.distance_limit = 30
        # TODO send in central position
    def resize_coordinate(self):
        pass
    def mousePressLeft(self):
        # Adds a marker in specified position
        # TODO handle resizing of the screen
        x_pos = self.c_p['mouse_params'][1] * self.c_p['image_scale']
        y_pos = self.c_p['mouse_params'][2] * self.c_p['image_scale']
        self.marker_positions.append(x_pos, y_pos)
        
    def mousePressRight(self):
        # Remove a marker
        if len(self.marker_positions) < 1:
            return
        min_distance = 80000
        min_x = -1
        min_y = -1
        # Find closest item
        click_x = self.c_p['mouse_params'][1] * self.c_p['image_scale']
        click_y = self.c_p['mouse_params'][2] * self.c_p['image_scale']
        for x,y in zip(self.marker_positions):
            distance = (x - click_x)**2 + (y - click_y) ** 2
            if distance < min_distance:
                min_distance = distance
                min_x = click_x
                min_y = click_y
        # Pop item
    
    def draw(self, qp):
        if len(self.marker_positions) < 1:
            return
        for x,y in zip(self.marker_positions):
            rx=50
            ry=50
            self.qp.drawEllipse(x, y, rx, ry)

        
"""
    def mouseMoveEvent(self, e):
        self.c_p['mouse_params'][3] = e.pos().x()-self.label.pos().x()
        self.c_p['mouse_params'][4] = e.pos().y()-self.label.pos().y()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton or \
            e.button() == Qt.MouseButton.RightButton:
            self.c_p['mouse_params'][0] = 0
            return
        if e.button() == Qt.MouseButton.LeftButton and not self.c_p['mouse_params'][0]:
            # handle the left-button press in here            if
            self.c_p['mouse_params'][0] = 1
            self.c_p['mouse_params'][1] = e.pos().x()-self.label.pos().x()
            self.c_p['mouse_params'][2] = e.pos().y()-self.label.pos().y()

    def mouseReleaseEvent(self, e):

        if e.button() == Qt.MouseButton.MiddleButton or \
            e.button() == Qt.MouseButton.RightButton or \
            not self.c_p['mouse_params'][0]:
            return
        self.c_p['mouse_params'][0] = 0
        self.c_p['mouse_params'][3] = e.pos().x()-self.label.pos().x()
        self.c_p['mouse_params'][4] = e.pos().y()-self.label.pos().y()
        x0, y0, x1, y1 = self.c_p['mouse_params'][1:5]
        dx = x1 - x0
        dy = y1 - y0
        if dx**2 < 100 or dy**2 < 100:
            print(dx,dy)
            return
        self.image_scale = self.c_p['image_scale']
        left = int(x0 * self.image_scale)
        right = int(x1 *self.image_scale)
        if right < left:
            tmp = right
            right = left
            left = tmp
        up = int(y0 * self.image_scale)
        down = int(y1 * self.image_scale)
        if up < down:
            tmp = up
            up = down
            down = tmp

        self.c_p['AOI'] = [self.c_p['AOI'][0] + left,self.c_p['AOI'][0] + right,
                           self.c_p['AOI'][2] + down,self.c_p['AOI'][2] + up]
        print("Udpating settings")
        self.c_p['new_settings_camera'] = [True, 'AOI']

    def mouseDoubleClickEvent(self, e):
        # Double click to move center?
        x = e.pos().x()-self.label.pos().x()
        y = e.pos().y()-self.label.pos().y()
        print(x*self.c_p['image_scale'] ,y*self.c_p['image_scale'] )
"""