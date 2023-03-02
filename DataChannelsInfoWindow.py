from PyQt6.QtWidgets import (
 QCheckBox, QVBoxLayout, QWidget, QLabel, QTableWidget, QTableWidgetItem
)

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QTimer

import numpy as np

class CurrentValueWindow(QWidget):
    """
    Window for displaying the current values of the data channels.
    """
    def __init__(self, c_p, data_channels):
        # TODO add checkbox for saving data(continously) and a button to save data snapshot
        super().__init__()
        self.c_p = c_p
        self.data_channels = data_channels
        self.setWindowTitle("Data channels current values")
        self.resize(400, 250)

        self.vBox = QVBoxLayout()
        self.CreateTable()
        self.vBox.addWidget(self.table)        

        self.timer = QTimer()
        self.timer.setInterval(100) # sets the fps of the timer
        self.timer.timeout.connect(self.set_data)

        self.timer.start()
        self.setLayout(self.vBox)

    def CreateTable(self):

        self.table = QTableWidget(len(self.data_channels)+1, 4)
        self.table.setHorizontalHeaderLabels(["Channel", "Value", "Unit", "Save"])
        
        for idx,channel in enumerate(self.data_channels):

            self.table.setItem(idx,0, QTableWidgetItem(f"{self.data_channels[channel].name}"))
            self.table.setItem(idx,1, QTableWidgetItem(f"{self.data_channels[channel].get_data(1)}"))
            self.table.setItem(idx,2, QTableWidgetItem(f"{self.data_channels[channel].unit}"))
            # self.table.setItem(idx,3, QTableWidgetItem(f"{self.data_channels[channel].save}"))
    
    def set_data(self):
        
        for idx,channel in enumerate(self.data_channels):
            self.table.setItem(idx,1, QTableWidgetItem(f"{self.data_channels[channel].get_data(1)}"))


    