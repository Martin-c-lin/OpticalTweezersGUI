# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:37:17 2022

@author: marti
"""
import sys
from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel
)

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QDoubleValidator, QIntValidator
sys.path.append("C:/Users/marti/OneDrive/PhD/OT software")
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from random import randint

import numpy as np
from functools import partial



Colors = {
    'red': (255,0,0),
    'blue': (0,0,255),
    'green': (0,255,0),
    'gray': (150,150,150),
    'white': (255,255,255),
    'black': (0,0,0),
    'cyan': (0,255,255),
    'Transparent': (0,0,0,0),
    }

# Dictionary whith all the symbols that can be used for plotting.
# Changed with the Setsymbol function of the plot.
Symbols = {
    'o' : "Default symbol, round circle symbol",
    't' : "Triangle pointing downwards symbol",
    't1' : "Triangle pointing upwards symbol",
    't2' : "Triangle pointing right side symbol",
    't3' : "Triangle pointing left side symbol",
    's' : "Square symbol",
    'p' : "Pentagon symbol",
    'h' : "Hexagon symbol",
    'star' : "Star symbol",
    '+' : "Plus symbol",
    'd' : "Prism symbol",
    'x' : "Cross symbol",
    None : "No marker",
}

class PlotLengthWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, plot_data, idx):
        super().__init__()
        self.plot_data = plot_data
        self.idx = idx
        layout = QVBoxLayout()
        self.label = QLabel(f"Plot length of plot {idx}")
        layout.addWidget(self.label)
        self.QLE = QLineEdit()

        self.QLE.setValidator(QIntValidator(1, 100_000, self))
        self.QLE.setText(str(self.plot_data['L'][idx]))
        self.QLE.textChanged.connect(self.set_plot_length)
        #self.addWidget(self.QLE)
        layout.addWidget(self.QLE)
        self.setLayout(layout)
        
    def set_plot_length(self, plot_length):
        try:
            self.plot_data['L'][self.idx] = int(plot_length)
        except:
            print("Could not set plot length")     

class PlotSubsamplehWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, plot_data, idx):
        super().__init__()
        self.plot_data = plot_data
        self.idx = idx
        layout = QVBoxLayout()
        self.label = QLabel(f"Subsampling of plot {idx}")
        layout.addWidget(self.label)
        self.QLE = QLineEdit()

        self.QLE.setValidator(QIntValidator(1, 10_000, self))
        self.QLE.setText(str(self.plot_data['sub_sample'][idx]))
        self.QLE.textChanged.connect(self.set_subsamples)
        #self.addWidget(self.QLE)
        layout.addWidget(self.QLE)
        self.setLayout(layout)
        
    def set_subsamples(self, subsamples):
        try:
            self.plot_data['sub_sample'][self.idx] = int(subsamples)
            print(self.plot_data['sub_sample'])
        except Exception as E:
            print("Could not set plotting subsampler")
            print(E)

class PlotAxisWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, graphWidget):
        super().__init__()
        self.graphWidget = graphWidget
        layout = QVBoxLayout()

        # Get default values of the axis
        axX = self.graphWidget.getAxis('bottom')
        lower_limit_X = axX.range[0]
        upper_limit_X = axX.range[1]

        axY = self.graphWidget.getAxis('left')
        lower_limit_Y = axY.range[0]
        upper_limit_Y = axY.range[1]

        # Add x-axis lower limit text and textbox
        self.label_X_low = QLabel("Set the x-axis lower limit")
        layout.addWidget(self.label_X_low)
        self.QLE_X_low = QLineEdit()
        self.QLE_X_low.setValidator(QDoubleValidator(-1000_000, 1000_000, 10))        
        self.QLE_X_low.setText(str(lower_limit_X))
        self.QLE_X_low.textChanged.connect(self.set_x_low)
        layout.addWidget(self.QLE_X_low)

        # Add x-axis lower limit text and textbox
        self.label_X_high = QLabel("Set the x-axis upper limit")
        layout.addWidget(self.label_X_high)
        self.QLE_X_high = QLineEdit()
        self.QLE_X_high.setValidator(QDoubleValidator(-1000_000, 1000_000, 10))
        self.QLE_X_high.setText(str(upper_limit_X))
        self.QLE_X_high.textChanged.connect(self.set_x_high)
        layout.addWidget(self.QLE_X_high)

        # Add x-axis lower limit text and textbox
        self.label_Y_low = QLabel("Set the y-axis lower limit")
        layout.addWidget(self.label_Y_low)
        self.QLE_Y_low = QLineEdit()
        self.QLE_Y_low.setValidator(QDoubleValidator(-1000_000, 1000_000, 10))        
        self.QLE_Y_low.setText(str(lower_limit_Y))
        self.QLE_Y_low.textChanged.connect(self.set_y_low)
        layout.addWidget(self.QLE_Y_low)

        # Add x-axis lower limit text and textbox
        self.label_Y_high = QLabel("Set the y-axis upper limit")
        layout.addWidget(self.label_Y_high)
        self.QLE_Y_high = QLineEdit()
        self.QLE_Y_high.setValidator(QDoubleValidator(-1000_000, 1000_000, 10))        
        self.QLE_Y_high.setText(str(upper_limit_Y))
        self.QLE_Y_high.textChanged.connect(self.set_y_high)
        layout.addWidget(self.QLE_Y_high)


        self.setLayout(layout)
        
    def set_x_low(self, limit):
        axX = self.graphWidget.getAxis('bottom')
        upper_limit = axX.range[1]
        lower_limit = float(limit)
        try:
            lower_limit = float(limit)
        except ValueError:
            # Harmless, probably only due to not having any number printed
            return

        if lower_limit >= upper_limit:
            return

        if (lower_limit-axX.range[0])/(np.abs(lower_limit+axX.range[0])+1e-4)<0.01:
            # Don't want it reacting to too small changes in the values
            return

        self.graphWidget.setXRange(lower_limit, upper_limit, padding=0)

    def set_x_high(self, limit):
        axX = self.graphWidget.getAxis('bottom')
        lower_limit = axX.range[0]
        try:
            upper_limit = float(limit)
        except ValueError:
            # Harmless, probably only due to not having any number printed
            return

        if lower_limit >= upper_limit:
            return

        if (upper_limit-axX.range[1])/(np.abs(upper_limit+axX.range[1])+1e-4)<0.01:
            # Don't want it reacting to too small changes in the values
            return

        self.graphWidget.setXRange(lower_limit, upper_limit, padding=0)

    def set_y_low(self, limit):
        axY = self.graphWidget.getAxis('left')
        upper_limit = axY.range[1]
        try:
            lower_limit = float(limit)
        except ValueError:
            # Harmless, probably only due to not having any number printed
            return
        if lower_limit >= upper_limit:
            return

        if (lower_limit-axY.range[0])/(np.abs(lower_limit+axY.range[0])+1e-4)<0.01:
            # Don't want it reacting to too small changes in the values
            return

        self.graphWidget.setYRange(lower_limit, upper_limit, padding=0)

    def set_y_high(self, limit):
        axY = self.graphWidget.getAxis('left')
        lower_limit = axY.range[0]
        try:
            upper_limit = float(limit)
        except ValueError:
            # Harmless, probably only due to not having any number printed
            return

        if lower_limit >= upper_limit:
            return

        if (upper_limit-axY.range[1])/(np.abs(upper_limit+axY.range[1])+1e-4)<0.01:
            # Don't want it reacting to too small changes in the values
            return

        self.graphWidget.setYRange(lower_limit, upper_limit, padding=0)


class PlotWindow(QMainWindow):
    
    # NOTE there was a bug in pyqtgraph 0.12.3 which made the window
    # crash when setting back auto zoom on the axis
    def __init__(self, c_p, data, x_keys, y_keys):
        super().__init__()

        self.c_p = c_p
        self.data = data
        self.graphWidget = pg.PlotWidget()
        self.sub_widgets = []
        self.x = list(range(100))  # 100 time points
        self.y = [randint(0, 100) for _ in range(100)]  # 100 data points. todo remove if not really needed
        self.y2 = [randint(0, 100) for _ in range(100)]  # 100 data points
        self.default_plot_length = 500
        # Set up plot data
        self.plot_data = {
            'x':x_keys,
            'y':y_keys,
            }# TODO add settings

        self.plot_data['L'] = np.ones(len(x_keys), int) * self.default_plot_length
        self.plot_data['sub_sample'] = np.ones(len(x_keys), int)

        self.plot_running = True
        self.graphWidget.setBackground('k')
        self.setWindowTitle("Data plotter")

        pen = pg.mkPen(color=Colors['red'])
        self.pen2 = pg.mkPen(color=Colors['green'])
        self.data_lines = []
        self.data_lines.append(self.graphWidget.plot(self.x, self.y, pen=pen) )
        self.data_lines.append(self.graphWidget.plot(self.x, self.y2, pen=self.pen2, symbolPen ='w'))

        self.timer = QTimer()
        self.timer.setInterval(50) # 20 fps
        self.timer.timeout.connect(self.update_plot_data)
        
        self.toolbar = QToolBar("Main tools")
        self.toolbar_extra = QToolBar("Main tools")

        self.stop_plot = QAction("Stop plotter", self)
        self.stop_plot.setToolTip("Momentarily freezes the plotting window")
        self.stop_plot.triggered.connect(self.toggle_live_plotting)
        self.stop_plot.setCheckable(True)

        self.add_plot_action = QAction("Add plot", self)
        self.add_plot_action.setToolTip("Adds another plot to the window")
        self.add_plot_action.triggered.connect(self.add_plot)
        self.add_plot_action.setCheckable(False)

        self.plot_axis_action = QAction("Adjust plot axis", self)
        self.plot_axis_action.setToolTip("Manually change the axis limits.")
        self.plot_axis_action.triggered.connect(self.open_plot_axis_window)
        self.plot_axis_action.setCheckable(False)
        

        self.toolbar.addAction(self.stop_plot)
        self.toolbar.addAction(self.add_plot_action)
        self.toolbar.addAction(self.plot_axis_action)
        self.setCentralWidget(self.graphWidget)
        self.addToolBar(self.toolbar)
        self.addToolBar(self.toolbar_extra)
        self.create_plot_menus()
        
        # TODO have all the subwindows close automatically when main application close
        # TODO fix label sizes, font and so that they automatically have correct axis
        self.set_axis_labels()        
        self.timer.start()

    def set_axis_labels(self):
        """
        Sets the axis labels of the plots the the values specified by the
        units of the first[0th] plot.

        Returns
        -------
        None.

        """
        
        y_label = f"{self.plot_data['y'][0]} {self.data[self.plot_data['y'][0]].unit}"
        self.graphWidget.setLabel('left', y_label, color='red', fontsize=200)
        x_label = f"{self.plot_data['x'][0]} {self.data[self.plot_data['x'][0]].unit}"
        self.graphWidget.setLabel('bottom', x_label, color='red', fontsize=200)


    def add_plot(self):
        # Adds another data line to the plot
        # almost works, problem with it adding extra lines for changing the number of plot points
        self.plot_data['x'].append('Time')
        self.plot_data['y'].append('Y-force')
        tmp = np.ones(len(self.plot_data['x']), int) * self.default_plot_length
        tmp[:-1] = self.plot_data['L'][:]
        self.plot_data['L'] = tmp # Add sub_sample here
        
        tmp = np.ones(len(self.plot_data['x']), int)
        tmp[:-1] = self.plot_data['sub_sample'][:]
        self.plot_data['sub_sample'] = tmp # Add sub_sample here
        #self.plot_data['sub_sample'].append[1]

        pen = pg.mkPen(color=Colors['red']) # Default color
        self.data_lines.append(self.graphWidget.plot(self.x, self.y, pen=pen) )
        self.menu.clear()
        self.create_plot_menus()
        self.set_axis_labels()

    def delete_plot(self, plot_idx=0):

        self.plot_data['x'].pop(plot_idx)
        self.plot_data['y'].pop(plot_idx)
        self.data_lines[plot_idx].setVisible(False)
        self.data_lines.pop(plot_idx)

        self.plot_data['L'] = np.delete(self.plot_data['L'], int(plot_idx))
        self.plot_data['sub_sample'] = np.delete(self.plot_data['sub_sample'], int(plot_idx))
        if len(self.plot_data['x']) > 0:
            self.set_axis_labels()
        self.menu.clear()
        self.create_plot_menus()

    def create_plot_menus(self):
        # Add menu
        self.menu = self.menuBar()
                
        BG_color_submenu = self.menu.addMenu("BG color")
        for col in Colors:
            
            color_command = partial(self.set_bg_color, Colors[col])
            set_col = QAction(col, self)
            set_col.setStatusTip(f"Set BG color to {col}")
            set_col.triggered.connect(color_command)
            BG_color_submenu.addAction(set_col)

        for idx, line in enumerate(self.data_lines):
            Plot_1_menu = self.menu.addMenu(f"Plot {idx}")
            Plot_1_menu.addSeparator()
            
            # Create submenu for setting colors
            color_submenu = Plot_1_menu.addMenu("Color")
            for col in Colors:
                
                color_command = partial(self.set_plot_color, Colors[col], line)
                set_col = QAction(col, self)
                set_col.setStatusTip(f"Set plot color to {col}")
                set_col.triggered.connect(color_command)
                color_submenu.addAction(set_col)

            y_data_submenu = Plot_1_menu.addMenu("Y-data")
            for data_idx, y in enumerate(self.data):
                data_command = partial(self.set_y_data, idx, y)
                set_data = QAction(y, self)
                set_data.setStatusTip("Set y-data")
                set_data.triggered.connect(data_command)
                y_data_submenu.addAction(set_data)

            x_data_submenu = Plot_1_menu.addMenu("X-data")
            for data_idx, x in enumerate(self.data):
                data_command = partial(self.set_x_data, idx, x)
                set_data = QAction(x, self)
                set_data.setStatusTip("Set x-data")
                set_data.triggered.connect(data_command)
                x_data_submenu.addAction(set_data)

            plot_symbol_submenu = Plot_1_menu.addMenu("Plot symbol")
            for symbol in Symbols:
                symbol_command = partial(self.set_plot_symbol, idx, symbol)
                set_symbol = QAction(Symbols[symbol], self)
                set_symbol.setStatusTip("Select symbol")
                set_symbol.triggered.connect(symbol_command)
                plot_symbol_submenu.addAction(set_symbol)
            # Maybe add sliders for marker sizes

            # Add option to remove this plot specifically
            remove_plot_action = QAction("Remove plot", self)
            remove_command = partial(self.delete_plot, idx)
            remove_plot_action.setToolTip("Removes a plot from the window")
            remove_plot_action.triggered.connect(remove_command)
            remove_plot_action.setCheckable(False)
            Plot_1_menu.addAction(remove_plot_action)

            change_length_action = QAction("Adjust plot length", self)
            length_command = partial(self.create_plot_length_window, idx)
            change_length_action.setStatusTip("Change the number of points being plotted")
            change_length_action.triggered.connect(length_command )
            change_length_action.setCheckable(False)
            Plot_1_menu.addAction(change_length_action)


            change_subsample_action = QAction("Adjust subsampling length", self)
            subsample_command = partial(self.create_plot_subsampler_window, idx)
            change_subsample_action.setStatusTip("Change subsampling frequency")
            change_subsample_action.triggered.connect(subsample_command )
            change_subsample_action.setCheckable(False)
            Plot_1_menu.addAction(change_subsample_action)
 
    def set_plot_symbol(self, plot_idx, symbol):
        try:
            self.data_lines[plot_idx].setSymbol(symbol)
        except Exception as E:
            print(E)

    def create_plot_length_window(self, idx):
        self.PlotLengthWindow = PlotLengthWindow(self.plot_data, idx)
        self.PlotLengthWindow.show()
        self.sub_widgets.append(self.PlotLengthWindow)

    def create_plot_subsampler_window(self, idx):
        self.PlotSubsamplehWindow = PlotSubsamplehWindow(self.plot_data, idx)
        self.PlotSubsamplehWindow.show()
        self.sub_widgets.append(self.PlotSubsamplehWindow)

    def open_plot_axis_window(self):
        self.PlotAxisWindow = PlotAxisWindow(self.graphWidget)
        self.PlotAxisWindow.show()
        self.sub_widgets.append(self.PlotAxisWindow)

    def update_plot_data(self):
        # If the program has closed then we should also close this window
        if self.plot_running:
            for idx,_ in enumerate(self.plot_data['x']):
                x_key = self.plot_data['x'][idx]
                y_key = self.plot_data['y'][idx]
                L = int(self.plot_data['L'][idx])
                S = int(self.plot_data['sub_sample'][idx])

                try:
                    x_data = self.data[x_key].get_data(L)
                    y_data = self.data[y_key].get_data(L)
                    self.data_lines[idx].setData(x_data[0:-1:S],y_data[0:-1:S])

                except Exception as e:
                    print(x_key, y_key)
                    print('Plotting error: Here is the error', e)

    def toggle_live_plotting(self):
        self.plot_running = not self.plot_running
        if self.plot_running:
            self.stop_plot.setIconText("Stop plotter")
        else:
            self.stop_plot.setIconText("Start plotter")

    def set_bg_color(self, color):
        self.graphWidget.setBackground(color)

    def set_plot_color(self, color, line):
        pen = pg.mkPen(color=color)
        line.setPen(pen)
        line.setSymbolPen(pen)
        # setSymbolBrush
        
    def set_x_data(self, idx, x_data):
        self.plot_data['x'][idx] = x_data
        
    def set_y_data(self, idx, y_data):
        self.plot_data['y'][idx] = y_data

    def __del__(self):
        for widget in self.sub_widgets:
            widget.close()
        