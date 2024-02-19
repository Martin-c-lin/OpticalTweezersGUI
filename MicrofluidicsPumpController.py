import sys
from email.header import UTF8

sys.path.append("C:/Users/Martin/Downloads/ESI_V3_08_02/ESI_V3_08_02/ESI_V3_08_02/SDK_V3_08_02/DLL/DLL64")#add the path of the library here
sys.path.append("C:/Users/Martin/Downloads/ESI_V3_08_02/ESI_V3_08_02/ESI_V3_08_02/SDK_V3_08_02/DLL/Python/Python_64")#add the path of the LoadElveflow.py

from array import array
from ctypes import *

import abc

from ctypes import *
from Elveflow64 import *

from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,QHBoxLayout,
    QPushButton, QVBoxLayout, QWidget, QLabel
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal

class MicrofluidicsControllerInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'setPressure') and
                callable(subclass.setPressure) and
                hasattr(subclass, 'setFlowRate') and
                callable(subclass.setFlowRate) and
                hasattr(subclass, 'setValve') and
                callable(subclass.setValve) and
                hasattr(subclass, 'setPump') and
                callable(subclass.setPump) and
                hasattr(subclass, 'getPressure') and
                callable(subclass.getPressure) and
                hasattr(subclass, 'getNumberChannels') and
                callable(subclass.getNumberChannels) and
                hasattr(subclass, 'getValve') and
                callable(subclass.getValve) and
                hasattr(subclass, 'getPump') and
                callable(subclass.getPump) or
                NotImplemented)

class ElvesysMicrofluidicsController(MicrofluidicsControllerInterface):
    def __init__(self):
        super().__init__()
        self.nbr_channels = 3
        self.Calib = (c_double*1000)()


    def connect(self, adress):
        self.Instr_ID=c_int32()
        print("Instrument name and regulator types are hardcoded in the Python script")
        #see User Guide to determine regulator types and NIMAX to determine the instrument name 
        error = OB1_Initialization(adress.encode('ascii'),0,0,0,0,byref(self.Instr_ID)) # Seems to be either 3 or 6 when looking in NI-MAX
        #all functions will return error codes to help you to debug your code, for further information refer to User Guide
        print('error:%d' % error)
        print("OB1 ID: %d" % self.Instr_ID.value)

    def disconnect(self):
        error = OB1_Close(self.Instr_ID.value)
        return self.check_error(error)

    def setPressure(self, channel, pressure):
        set_channel = int(channel)#convert to int
        set_channel = c_int32(set_channel)#convert to c_int32

        #Pressure
        set_target=float(pressure) 
        set_target=c_double(set_target)#convert to c_double

        error = OB1_Set_Press(self.Instr_ID.value, set_channel, set_target,  byref(self.Calib), 1000)
        return self.check_error(error)

    def check_error(self, error):
        return error == 0
    
    def getNumberChannels(self):
        return self.nbr_channels

    def get_pressure(self, channel):
        """
        Get the pressure of a channel. Automatically updates the control parameters.
        """
        set_channel = int(channel)#convert to int
        set_channel = c_int32(set_channel)#convert to c_int32
        get_pressure = c_double()
        error = OB1_Get_Press(self.Instr_ID.value, set_channel, 1, byref(self.Calib),byref(get_pressure), 1000)#Acquire_data=1 -> read all the analog values
        if self.check_error(error):
            
            return get_pressure.value
        return None


class MicrofluidicsMonitorThread(QThread):
    # Define signals to communicate with the main thread
    finished = pyqtSignal()
    progress = pyqtSignal(list)

    def __init__(self, controller, c_p):
        super().__init__()
        self.controller = controller
        self.c_p = c_p

    def set_pressures(self):

        for channel in range(self.controller.getNumberChannels()):
            self.controller.setPressure(channel, self.c_p['target_pressures'][channel])

    def get_pressures(self):
        for channel in range(self.controller.getNumberChannels()):
            self.c_p['current_pressures'][channel] = self.controller.get_pressure(channel)

    def run(self):
        # Place your background task here
        while self.c_p['program_running']:
            self.set_pressures()
            self.get_pressures()
            self.progress.emit(self.c_p['current_pressures'])
            QThread.msleep(500) # Sleep for specified number of milliseconds
        self.finished.emit()



class MicrofluidicsControllerWidget(QWidget):
    """
    A widget for controlling the microfluidics system. Will automatically
    create buttons to control each of the channels in the system.
    """

    def __init__(self, c_p, controller=None):
        super().__init__()
        self.c_p = c_p
        self.controller = controller

        self.initUI()
        self.pumpMonitorThread = MicrofluidicsMonitorThread(self.controller, self.c_p)
        self.pumpMonitorThread.progress.connect(self.updatePressures)
        self.pumpMonitorThread.start()
        print("Pump monitor started")

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setWindowTitle("Microfluidics Controller")
        self.create_channel_controls()

        # Create button for calibrating the pump

        # Also create button for connecting the pump and potentially also for disconnecting it

        # Eventually we will also need to add the valves here.

        self.setLayout(self.layout)

        self.show()
    
    def create_channel_controls(self):
        """
        Crates the UI elments needed to control the channels of the pump
        """
        self.pressure_spinboxes = []
        self.pressure_monitor_labels = []

        for channel in range(self.controller.getNumberChannels()):
            # Create a label for the channel
            label = QLabel("Channel " + str(channel+1))
            self.layout.addWidget(label)

            # Create a spinbox for setting the pressure
            self.pressure_spinboxes.append(QDoubleSpinBox())
            self.pressure_spinboxes[-1].setRange(0, 200) # TODO fix so that this actually corresponds to the pressure range of the pump
            self.pressure_spinboxes[-1].setSingleStep(0.1)
            self.pressure_spinboxes[-1].setSuffix(" mbar")
            self.pressure_spinboxes[-1].valueChanged.connect(lambda value, channel=channel: self.setPressure(channel, value))
            self.layout.addWidget(self.pressure_spinboxes[-1])

            # Create a label for monitoring the pressure
            self.pressure_monitor_labels.append(QLabel(f"Pressure {self.c_p['current_pressures'][channel]} mbar"))
            self.layout.addWidget(self.pressure_monitor_labels[-1])

    def updatePressures(self, values):
        for channel in range(self.controller.getNumberChannels()):
            self.pressure_monitor_labels[channel].setText(f"Pressure {self.c_p['current_pressures'][channel]} mbar")

    def setPressure(self, channel, pressure):
        self.c_p['target_pressures'][channel] = float(pressure)

    def closeEvent(self, event):
        #self.pumpMonitorThread.
        event.accept()