'''
File containing a simplified interface for thorlabs motors. Supports
the max381 stage as well as kcube motors. The motors are handled as threads
'''

# Import packages
# from ctypes import *
import clr, sys
from System import Decimal, Int32
from time import sleep
from threading import Thread
import numpy as np

# Import DLLs
"""
Note when usin this code on other computer than the one in the biophysics lab these paths may need changing.
"""

clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.DeviceManagerCLI.dll')
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.GenericMotorCLI.dll')
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.KCube.DCServoCLI.dll')
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.KCube.InertialMotorCLI.dll ')
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.GenericPiezoCLI.dll')
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.Benchtop.PiezoCLI.dll')
clr.AddReference('C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.Benchtop.StepperMotorCLI.dll')

# TODO replace import * wiht something better
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import MotorDirection
from Thorlabs.MotionControl.KCube.InertialMotorCLI import *
from Thorlabs.MotionControl.KCube.InertialMotorCLI import *
from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import *
from Thorlabs.MotionControl.Benchtop.PiezoCLI import *
from Thorlabs.MotionControl.Benchtop.StepperMotorCLI import *


timeoutVal = 30000

class PiezoMotor():
    '''
    Piezo motor class.
    '''

    def __init__(self, serialNumber, channel, pollingRate=20, timeout=10000):
        self.serial_number = serialNumber
        self.polling_rate = pollingRate
        self.connect_piezo_motor()
        self.channel = channel
        self.timeout = timeout

    def connect_piezo_motor(self):
        self.motor = InitiatePiezoMotor(self.serial_number, self.polling_rate)
        self.is_connected = True if not self.motor is None else False

    def move_to_position(self, position):
        '''
        Function for moving the motor to a specified position

        Parameters
        ----------
        position : Float, given in units of mm.
            Target position to move motor to.

        Returns
        -------
        bool
            True if move was successfull otherwise false.

        '''
        try:
            self.motor.MoveTo(self.channel, position, self.timeout)
            return True
        except Exception:
            print('Could not move piezo to target position')
            return False

    def set_timeout(self, timeout):
        '''
        Function for setting the timeout of the piezo motor

        Parameters
        ----------
        timeout : int
            Timeout of motot in ms.

        Returns
        -------
        bool
            True if the timeout was okay and set.

        '''
        if timeout >= 1:
            self.timeout = timeout
            return True
        else:
            print("Timeout NOK")
            return False

    def get_timeout(self):
        '''
        Returns
        -------
        int
            Timeout of piezo motor.

        '''
        return self.timeout

    def move_relative(self, distance):
        '''
        Function for moving the piezo a fixed distance relative to it's
        current position.

        Parameters
        ----------
        distance : Int
            Distance in ticks to move.

        Returns
        -------
        boolean
            True if move was successfull otherwise false.

        '''
        print(f"Moving relative {distance}")
        target_position = self.get_position() + distance
        return self.move_to_position(target_position)

    def get_position(self):
        '''
        Returns
        -------
        int
            Current position of piezo(in ticks).

        '''
        try:
            return self.motor.GetPosition(self.channel)
        except Exception:
            # TODO find the right errorcode for this
            print('Could not find piezo position')
            return 0

    def disconnect_piezo(self):
        if self.is_connected:
            self.motor.StopPolling()
            self.motor.Disconnect()
            self.is_connected = False

    def __del__(self):
        self.disconnect_piezo()


def InitiatePiezoMotor(serialNumber, pollingRate=250):
    '''
    Function for initalizing a piezo motor.

    Parameters
    ----------
    serialNumber : String
        Serialnumber of controller which is being contacted.
    pollingRate : float, optional
        The default is 250. Polling rate of controller.

    Returns
    -------
    motor :
        A PiezoMotor controller object. None if initalization failed.

    '''
    DeviceManagerCLI.BuildDeviceList()
    DeviceManagerCLI.GetDeviceListSize()
    motor = KCubeInertialMotor.CreateKCubeInertialMotor(serialNumber)
    #for attempts in range(3):
    try:
        motor.Connect(serialNumber)
    except Exception as ex:
        print(f"Could not connect to piezo \n {ex}")
        return None
        # print("Connection attempt", attempts, "failed")
            # if attempts < 2:
            #     print("Will wait 2 seconds and try again")
            #     sleep(2)
            # else:
            #     print("Cannot connect to device.\n Please ensure that the \
            #           device is connected to your computer and not in use in\
            #               any other program!")
                # return None
    motor.WaitForSettingsInitialized(5000)
    # configure the stage
    motorSettings = motor.GetInertialMotorConfiguration(serialNumber)
    motorSettings.DeviceSettingsName = 'PIA'
    # update the RealToDeviceUnit converter
    motorSettings.UpdateCurrentConfiguration()
    # push the settings down to the device
    currentDeviceSettings = ThorlabsInertialMotorSettings.GetSettings(motorSettings)

    motor.SetSettings(currentDeviceSettings, True, False)
    # Start polling and enable the device
    motor.StartPolling(pollingRate)
    motor.EnableDevice()

    return motor


class StageMotor():
    '''
    Class for the motors used by the stage. Class currently not in use
    '''
    def __init__(self, serialNumber, pollingRate=200, mmToPixel=16140,timeoutVal=30000):
        self.serialNumber = serialNumber
        self.pollingRate = pollingRate
        self.timeoutVal = timeoutVal
        self.connect_motor()
        self.mmToPixel = mmToPixel

    def SetJogSpeed(self,jogSpeed,jogAcc=0.1):
        try:
            self.motor.SetJogVelocityParams(Decimal(jogSpeed),Decimal(jogAcc))
        except Exception as ex:
            print(f"Failed to set jogSpeed \n {ex}")

    def connect_motor(self):
        self.motor = InitiateMotor(self.serialNumber, self.pollingRate)
        self.is_connected = False if self.motor is None else True
        if self.is_connected:
            self.startingPosition = self.motor.GetPosition()
        else:
            self.startingPosition = 0

    def disconnect_motor(self):
        if self.is_connected:
            motor.StopPolling()
            motor.Disconnect()
            self.is_connected = False

    def MoveMotor(self, distance):
        '''
        Helper function for moving a motor.

        Parameters
        ----------
        motor : thorlabs motor
            Motor to be moved.
        distance : float
            Distance to move the motor.

        Returns
        -------
        bool
            True if the move was a success, otherwise false.

        '''
        if not self.is_connected:
            return False

        if distance > 0.1 or distance < -0.1:
            print("Trying to move too far")
            return False
        # For unknown reason python thinks one first must convert to float but
        # only when running from console...
        self.motor.SetJogStepSize(Decimal(float(distance)))
        try:
            motor.MoveJog(1, timeoutVal)# Jog in forward direction
        except Exception as ex:
            print(f"Move failed \n {ex}")
            return False
        return True

    def MoveMotorPixels(self, distance):
        '''
        Moves motor a specified number of pixels.

        Parameters
        ----------
        motor : TYPE - thorlabs motor
             Motor to be moved
        distance : TYPE number
             Distance to move the motor
        mmToPixel : TYPE number for converting from mm(motor units) to pixels, optional
             The default is 16140, valid for our 100x objective and setup.

        Returns
        -------
        bool
            True if move was successfull, false otherwise.
        '''
        self.motor.SetJogStepSize(Decimal(float(distance/self.mmToPixel)))
        try:
            self.motor.MoveJog(1, timeoutVal)  # Jog in forward direction
        except Exception as ex:
            print(f"Motoro move failed \n {ex}")
            return False
        return True


    def MoveMotorToPixel(self, targetPixel, currentPixel, maxPixel=1280):
        '''

        Parameters
        ----------
        motor : TYPE
            DESCRIPTION.
        targetPixel : TYPE
            DESCRIPTION.
        currentPixel : TYPE
            DESCRIPTION.
        maxPixel : TYPE, optional
            DESCRIPTION. The default is 1280.
        mmToPixel : TYPE, optional
            DESCRIPTION. The default is 16140.

        Returns
        -------
        bool
            DESCRIPTION.

        '''
        if(targetPixel < 0 or targetPixel > maxPixel): # Fix correct boundries
            print("Target pixel outside of bounds")
            return False
        if not self.is_connected:
            return False
        # There should be a minus here, this is due to the setup
        dx = -(targetPixel-currentPixel)/self.mmToPixel
        self.motor.SetJogStepSize(Decimal(float(dx)))
        try:
            self.motor.MoveJog(1,timeoutVal)# Jog in forward direction
        except Exception as ex:
            print(f"Motor move failed \n {ex}")
            return False
        return True

    def __del__(self):
        self.motor.StopPolling()
        self.motor.Disconnect()


def InitiateMotor(serialNumber, pollingRate=250, DeviceSettingsName='Z812'):
    '''
    Function for initalizing contact with a thorlabs k-cube controller object.

    Parameters
    ----------
    serialNumber : String
        Serial number of device to be connected. Written on the back of the
    pollingRate : int, optional
        Polling rate of device in ms. The default is 250.
    DeviceSettingsName : string, optional
        Indicates which type of motor is connectd to the controller.
        The default is 'Z812'.

    Returns
    -------
    motor : k-cube controller
        k-cube controller which can be used to control a thorlabs motor.

    '''
    DeviceManagerCLI.BuildDeviceList()
    DeviceManagerCLI.GetDeviceListSize()

    motor = KCubeDCServo.CreateKCubeDCServo(serialNumber)
#    for attempts in range(3):
    try:
        motor.Connect(serialNumber)
    except Exception as ex:
        print(f"Failed to connect motor\n {ex}")
        return None

        # except:
        #     print("Connection attempt", attempts, "failed")
        #     if attempts < 2:
        #         print("Will wait 2 seconds and try again")
        #         sleep(2)
        #     else:
        #         print("Cannot connect to device.\n Please ensure that the" +\
        #               " device is connected to your computer and not in"+\
        #                   " use by any other program!")
        #         return None
    motor.WaitForSettingsInitialized(5000)
    # configure the stage
    motorSettings = motor.LoadMotorConfiguration(serialNumber)
    motorSettings.DeviceSettingsName = DeviceSettingsName
    # update the RealToDeviceUnit converter
    motorSettings.UpdateCurrentConfiguration()
    # push the settings down to the device
    MotorDeviceSettings = motor.MotorDeviceSettings
    motor.SetSettings(MotorDeviceSettings, True, False)
    # Start polling the device
    motor.StartPolling(pollingRate)

    motor.EnableDevice()
    # Jogging parameters set to minimum
    motor.SetJogVelocityParams(Decimal(0.01), Decimal(0.01))
    return motor


def DisconnectMotor(motor):
    '''
    Function for safely disconnecting a motor so that other programs may use
    it.
    Parameters
    ----------
    motor : Thorlabs motor.
        Motor to be disconnected.

    Returns
    -------
    None.

    '''
    motor.StopPolling()
    motor.Disconnect()


def MoveMotor(motor, distance):
    '''
    Helper function for moving a motor.

    Parameters
    ----------
    motor : thorlabs motor
        Motor to be moved.
    distance : float
        Distance to move the motor.

    Returns
    -------
    bool
        True if the move was a success, otherwise false.

    '''
    if distance > 0.1 or distance < -0.1:
        print("Trying to move too far")
        return False
    # For unknown reason python thinks one first must convert to float but
    # only when running from console...
    motor.SetJogStepSize(Decimal(float(distance)))
    try:
        motor.MoveJog(1, timeoutVal)# Jog in forward direction
    except Exception as ex:
        print(f"Failed to move motor \n {ex}")
        return False
    return True


def MoveMotorPixels(motor, distance, mmToPixel=16140):
    '''
    Moves motor a specified number of pixels.

    Parameters
    ----------
    motor : TYPE - thorlabs motor
         Motor to be moved
    distance : TYPE number
         Distance to move the motor
    mmToPixel : TYPE number for converting from mm(motor units) to pixels, optional
         The default is 16140, valid for our 100x objective and setup.

    Returns
    -------
    bool
        True if move was successfull, false otherwise.
    '''
    motor.SetJogStepSize(Decimal(float(distance/mmToPixel)))
    try:
        motor.MoveJog(MotorDirection(1), timeoutVal)  # Jog in forward direction
    except Exception as ex:
        print(f"Failed to move motor \n {ex}")
        return False
    return True


def MoveMotorToPixel(motor, targetPixel,
                     currentPixel, maxPixel=1280, mmToPixel=16140):
    '''


    Parameters
    ----------
    motor : TYPE
        DESCRIPTION.
    targetPixel : TYPE
        DESCRIPTION.
    currentPixel : TYPE
        DESCRIPTION.
    maxPixel : TYPE, optional
        DESCRIPTION. The default is 1280.
    mmToPixel : TYPE, optional
        DESCRIPTION. The default is 16140.

    Returns
    -------
    bool
        DESCRIPTION.

    '''
    if(targetPixel<0 or targetPixel>maxPixel): # Fix correct boundries
        print("Target pixel outside of bounds")
        return False
    # There should be a minus here, this is due to the setup
    dx = -(targetPixel-currentPixel)/mmToPixel
    motor.SetJogStepSize(Decimal(float(dx)))
    try:
        motor.MoveJog(MotorDirection(1),timeoutVal)# Jog in forward direction
    except Exception as ex:
        print(f"Failed to move motor \n {ex}")
        return False
    return True

def MoveTrapToPosition(motorX, motorY, targetX, targetY, trapX, trapY):
    '''

    Parameters
    ----------
    motorX : TYPE
        DESCRIPTION.
    motorY : TYPE
        DESCRIPTION.
    targetX : TYPE
        DESCRIPTION.
    targetY : TYPE
        DESCRIPTION.
    trapX : TYPE
        DESCRIPTION.
    trapY : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    x = MoveMotorToPixel(motorX, targetX, trapX) # move X
    y = MoveMotorToPixel(motorY, targetY, trapY) # move Y
    return x and y


def setJogSpeed(motor, jog_speed, jog_acc=0.01):
    """
    Sets the jog-speed in mm/s of the motor as well as the jog acceleration
    """
    return motor.SetJogVelocityParams(Decimal(jog_speed), Decimal(jog_acc))


class MotorThread(Thread):
    '''
    Thread in which a motor is controlled. The motor object is available globally.
    '''
    # TODO: Try replacing some of the c_p with events.
    # TODO: Make this run in the same way as the newer versions which allow for
    # better click_move
    def __init__(self, threadID, name, axis, c_p):

      Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.axis = axis # 0 = x-axis, 1 = y axis
      self.c_p = c_p
      # Initiate contact with motor
      if self.axis == 0 or self.axis == 1:
          self.motor = InitiateMotor(c_p['serial_nums_motors'][self.axis],
            pollingRate=c_p['polling_rate'])
      else:
          raise Exception("Invalid choice of axis, no motor available.")

      # Read motor starting position
      if self.motor is not None:
          c_p['motor_starting_pos'][self.axis] = self.get_current_position()
          print('Motor is at ', c_p['motor_starting_pos'][self.axis])
          c_p['motors_connected'][self.axis] = True
      else:
          c_p['motors_connected'][self.axis] = False
      self.setDaemon(True)

    def get_current_position(self):
      tmp = str(self.motor.Position)
      tmp = tmp.replace(",", ".")
      return float(tmp)

    def run(self):
        print('Running motor thread')
        # global c_p
        c_p = self.c_p
        while c_p['motor_running'] and c_p['program_running']:
            # If motor connected and it should be connected, check for next move
            if c_p['motors_connected'][self.axis] and \
                c_p['connect_motor'][self.axis] and c_p['motors_connected'][self.axis]:
                # Acquire lock to ensure that it is safe to move the motor
                if np.abs(c_p['motor_movements'][self.axis])>0:

                    # The movement limit must be positive
                    c_p['xy_movement_limit'] = np.abs(c_p['xy_movement_limit'])

                    # Check how much the motor is allowed to move
                    if np.abs(c_p['motor_movements'][self.axis]) <= c_p['xy_movement_limit']:
                        MoveMotorPixels(self.motor,
                            c_p['motor_movements'][self.axis],
                            mmToPixel=c_p['mmToPixel'])
                    else:
                        if c_p['motor_movements'][self.axis] > 0:
                            MoveMotorPixels(self.motor,
                                c_p['xy_movement_limit'],
                                mmToPixel=c_p['mmToPixel'])
                        else:
                            MoveMotorPixels(self.motor,
                                -c_p['xy_movement_limit'],
                                mmToPixel=c_p['mmToPixel'])

                    c_p['motor_movements'][self.axis] = 0
                c_p['motor_current_pos'][self.axis] = self.get_current_position()
            # Motor is connected but should be disconnected
            elif c_p['motors_connected'][self.axis] and not c_p['connect_motor'][self.axis]:
                DisconnectMotor(self.motor)
                c_p['motors_connected'][self.axis] = False
                self.motor = None
            # Motor is not connected but should be
            elif not c_p['motors_connected'][self.axis] and c_p['connect_motor'][self.axis]:
                self.motor = InitiateMotor(c_p['serial_nums_motors'][self.axis],
                  pollingRate=c_p['polling_rate'])
                # Check if motor was successfully connected.
                if self.motor is not None:
                    c_p['motors_connected'][self.axis] = True
                    c_p['motor_current_pos'][self.axis] = self.get_current_position()
                    c_p['motor_starting_pos'][self.axis] = c_p['motor_current_pos'][self.axis]
                else:
                    motor_ = 'x' if self.axis == 0 else 'y'
                    print('Failed to connect motor '+motor_)
            sleep(0.1) # To give other threads some time to work
        if c_p['motors_connected'][self.axis]:
            DisconnectMotor(self.motor)

def compensate_focus(c_p):
    '''
    Function for compensating the change in focus caused by x-y movement.
    Returns the positon in ticks which z  should take to compensate for the focus
    '''
    new_z_pos = (c_p['z_starting_position']
        +c_p['z_x_diff']*(c_p['motor_starting_pos'][0] - c_p['motor_current_pos'][0])
        +c_p['z_y_diff']*(c_p['motor_starting_pos'][1] - c_p['motor_current_pos'][1]) )
    new_z_pos += c_p['temperature_z_diff']*(c_p['current_temperature']-c_p['starting_temperature'])
    return int(new_z_pos)

class PiezoThread(Thread):
    '''
    Thread for controling movement of the objective in z-direction.
    Will also help with automagically adjusting the focus to the sample.
    '''

    def __init__(self, serial_no, channel, c_p,
                 polling_rate=250):
        Thread.__init__(self)
        self.piezo = PiezoMotor(serial_no, channel=channel,
                                pollingRate=polling_rate)
        if self.piezo.is_connected:
            c_p['z_starting_position'] = self.piezo.get_position()
            c_p['z_current_position'] = self.piezo.get_position()
            c_p['z_piezo_connected'] = self.piezo.is_connected
            print(f"Piezo is at {c_p['z_starting_position'] }")
        self.setDaemon(True)
        self.c_p = c_p


    def run(self):
        c_p = self.c_p
        lifting_distance = 0
        while c_p['program_running']:
            c_p['z_piezo_connected'] = self.piezo.is_connected
            
            # Check if piezo connected and should be connected
            if self.piezo.is_connected and c_p['connect_z_piezo']:
                # Check if the objective should be moved
                c_p['z_current_position'] = self.piezo.get_position()
                if c_p['z_movement'] != 0:
                    c_p['z_movement'] = int(c_p['z_movement'])
                    self.piezo.move_relative(c_p['z_movement'])
                    c_p['z_movement'] = 0
            # Piezomotor not connected but should be
            elif not self.piezo.is_connected and c_p['connect_z_piezo']:
                self.piezo.connect_piezo_motor()
                sleep(0.4)
                if self.piezo.is_connected:
                    # If the motor was just connected then reset positions
                    c_p['z_current_position'] = self.piezo.get_position()
                    c_p['z_starting_position'] = c_p['z_current_position']
            # Piezo motor connected but should not be
            elif self.piezo.is_connected and not c_p['connect_ z_piezo']:
                self.piezo.disconnect_piezo()

            sleep(0.05)
        del(self.piezo)


def ConnectBenchtopPiezoController(serialNo):
    DeviceManagerCLI.BuildDeviceList()
    DeviceManagerCLI.GetDeviceListSize()
    device = BenchtopPiezo.CreateBenchtopPiezo(serialNo)
    device.Connect(serialNo)
    return device


def ConnectPiezoStageChannel(device, channel, polling_rate=100):
    # DeviceManagerCLI.BuildDeviceList()
    # DeviceManagerCLI.GetDeviceListSize()
    # device = BenchtopPiezo.CreateBenchtopPiezo(serialNo)
    # device.Connect(serialNo)
    channel = device.GetChannel(channel)

    piezoConfiguration = channel.GetPiezoConfiguration(channel.DeviceID)
    currentDeviceSettings = channel.PiezoDeviceSettings
    channel.SetSettings(currentDeviceSettings, True, False)

    channel.WaitForSettingsInitialized(5000)

    channel.StartPolling(polling_rate)
    # Needs a delay so that the current enabled state can be obtained

    deviceInfo = channel.GetDeviceInfo()
    # Enable the channel otherwise any move is ignored
    channel.EnableDevice()
    channel.SetPositionControlMode(2) # Set to closed loop mode.

    return channel


def get_default_piezo_c_p():
    '''
    Generates c_p needed for the piezo with default values

    Returns
    -------
    piezo_c_p : Dictionary
        Dictionary containing the control parameters needed for the piezso
        specifically.

    '''
    # TODO use pos and position consitently
    piezo_c_p = {
        'piezo_serial_no': '71165844',
        'piezo_starting_position': [0, 0, 0],
        'piezo_target_position': [10, 10, 18],
        'piezo_current_position': [0, 0, 0],
        'stage_piezo_connected': [False, False, False],
        'running': True,
        'piezo_move_to_target': [False, False, False],
        'connect_piezos': False,
        'piezo_controller': None, # TODO finish implementation so that Piezos could be disconnected on the fly.
    }
    return piezo_c_p


class XYZ_piezo_stage_motor(Thread):
    '''
    Class to help a main program of Automagic Trapping interface with a
    thorlabs max381 stage piezo motors.
    '''

    # TODO make it possible to connect/disconnect these motors on the fly.
    def __init__(self, threadID, name, channel, axis, c_p, target_key,
                 controller_device=None, serialNo='71165844',
                 sleep_time=0.15, step=0.1):

        Thread.__init__(self)
        self.c_p = c_p
        self.name = name
        self.threadID = threadID
        self.setDaemon(True)
        self.channel = channel
        self.axis = axis
        self.sleep_time = sleep_time
        self.target_key = target_key
        self.step = step
        self.piezo_channel = None

    def connect_piezo_channel(self):
        try:
            print('Connecting channel', self.channel)
            self.piezo_channel = ConnectPiezoStageChannel(self.c_p['piezo_controller'], self.channel)
            print('Piezo channel connected')
        except Exception as ex:
            print(f"Failed to connect piezo channel \n {ex}")

    def update_current_position(self):
        if self.piezo_channel.IsConnected:
            tmp = str(self.piezo_channel.GetPosition())
            self.c_p['piezo_current_position'][self.axis] = float(tmp.replace(',', '.'))

    def update_position(self):
        # Update c_p position
        if not self.piezo_channel.IsConnected:
            return
        self.update_current_position()
        if 0 <= self.c_p['piezo_target_position'][self.axis] <= 20:
            d = self.c_p['piezo_target_position'][self.axis] - self.c_p['piezo_current_position'][self.axis]
            if np.abs(d) > self.step:
                if d < 0:
                    next_pos = self.c_p['piezo_current_position'][self.axis] - self.step#+ max(, d)
                else:
                    next_pos = self.c_p['piezo_current_position'][self.axis] + self.step#min(, d)
                self.piezo_channel.SetPosition(Decimal(next_pos))
            # TODO fix so that channel 2(z) behaves
            if self.axis == 2:
                self.piezo_channel.SetPosition(Decimal(self.c_p['piezo_target_position'][self.axis]))

        self.update_current_position()

    def run(self):
        '''
        Main loop of program. Used to automatically move the piezo in response
        to changes in c_p made by main program.
        '''
        sleep(self.sleep_time)
        # TODO test change made to prevent excess moves
        while self.c_p['program_running']:
            if self.piezo_channel is not None and self.piezo_channel.IsConnected:

                # Check if we should move the piezo to a specific location.
                if self.c_p['piezo_move_to_target'][self.axis]:
                    index = self.c_p['QDs_placed'] if self.axis < 2 else 0
                    d =  self.c_p['piezo_current_position'][self.axis] - self.c_p[self.target_key][index]

                    # Don't make really small moves
                    if d < -0.05:
                        self.c_p['piezo_target_position'][self.axis] = self.c_p['piezo_current_position'][self.axis] - max(-self.step, d)
                    elif d > 0.05:
                        self.c_p['piezo_target_position'][self.axis] = self.c_p['piezo_current_position'][self.axis] - min(self.step, d)
                    # if np.abs(d) < 0.05:
                    #     self.c_p['piezo_move_to_target'][self.axis] = False

                self.update_position()
            elif self.c_p['connect_piezos']:
                self.connect_piezo_channel()
                if self.piezo_channel is not None and self.piezo_channel.IsConnected:
                    self.update_current_position()
                    self.c_p['piezo_starting_position'][self.axis] = self.c_p['piezo_current_position'][self.axis]#self.piezo_channel.GetPosition()
                    self.c_p['stage_piezo_connected'][self.axis] = True
            sleep(self.sleep_time)

    # Had trouble with piezos not reconnecting after restartng program.
    # May have had something to do with the __del__ function.
    def __del__(self):
        try:
            self.piezo_channel.StopPolling()
            self.piezo_channel.Disconnect()
        except Exception as ex:
            print(f"Device already disconnected \n {ex}")

def ConnectBenchtopStepperController(serialNo):
    '''
    Connects a benchtop stepper controller.
    '''

    DeviceManagerCLI.BuildDeviceList()
    DeviceManagerCLI.GetDeviceListSize()
    device = BenchtopStepperMotor.CreateBenchtopStepperMotor(serialNo)
    device.Connect(serialNo)
    return device


def ConnectBenchtopStepperChannel(device, channel, polling_rate=20):
    '''
    Connects a stepper controller channel.

    Parameters
    ----------
    device : TYPE
        The benchtop stepper controller
    channel : TYPE
        Channel to connect.
    polling_rate : TYPE, optional
        How often to poll the device in ms intervals. The default is 20.

    Returns
    -------
    channel : TYPE
        Benchtop stepper channel.

    '''

    channel = device.GetChannel(channel)
    channel.WaitForSettingsInitialized(5000)
    channel.StartPolling(polling_rate)

    # Needs a delay so that the current enabled state can be obtained
    motorConfiguration = channel.LoadMotorConfiguration(channel.DeviceID)
    currentDeviceSettings = channel.MotorDeviceSettings
    channel.SetSettings(currentDeviceSettings, True, False)
    deviceInfo = channel.GetDeviceInfo()

    # Enable the channel otherwise any move is ignored
    channel.EnableDevice()

    return channel


def get_default_stepper_c_p():
    '''
    Returns a dictionary containg default values for all control parameters
    needed for the benchtop stepper controller and it's motors.

    Returns
    -------
    stepper_c_p : Dictionary
        Dictionary containing the control parameters needed for the stepper
        motors.

    '''
    stepper_c_p = {
        'stepper_serial_no': '70167314',
        'stepper_starting_position': [0, 0, 0],
        #'stage_stepper_connected': [False, False, False],
        'stepper_current_position': [0, 0, 0],
        'stepper_target_position': [2.3, 2.3, 7],
        'stepper_move_to_target': [False, False, False],
        'stepper_next_move': [0, 0, 0],
        'stepper_max_speed': [0.01, 0.01, 0.01],
        'stepper_acc': [0.005, 0.005, 0.005],
        'new_stepper_velocity_params': [False, False, False],
        'tilt': [0, 0], # How much the stage is tilting in x and y direction
        'connect_steppers': False, # Should steppers be connected?
        'steppers_connected': [False, False, False], # Are the steppers connected?
        'stepper_controller': None,
    }
    return stepper_c_p


class XYZ_stepper_stage_motor(Thread):
    """
    Controls nanomax stage stepper motors
    """
    def __init__(self, threadID, name, channel, axis, c_p,
                 controller_device=None, serialNo='70167314', sleep_time=0.01,
                 step=0.0006):

        Thread.__init__(self)
        self.c_p = c_p
        self.name = name
        self.threadID = threadID
        self.setDaemon(True)
        self.channel = channel
        self.axis = axis
        self.sleep_time = sleep_time
        self.step = step
        self.is_moving = False
        self.move_direction = MotorDirection(1)
        self.stepper_channel = None

    def connect_channel(self):
        try:
            self.stepper_channel = ConnectBenchtopStepperChannel(
                self.c_p['stepper_controller'], self.channel)
        except Exception as ex:
            print(f"Failed to connect stepper \n {ex}")
            # TODO check if some of these error handlings give too many printouts

    def update_current_position(self):
        decimal_pos = self.stepper_channel.Position
        self.c_p['stepper_current_position'][self.axis] = float(str(decimal_pos).replace(',','.'))
        return self.c_p['stepper_current_position'][self.axis]

    def move_absolute(self):
        target_pos = Decimal(float(self.c_p['stepper_target_position'][self.axis]))
        self.stepper_channel.MoveTo(target_pos, Int32(100000))

    def move_distance(self, distance):
        # TODO Check if replacing 1 with MotorDirection(1) fixed bug
        self.stepper_channel.MoveRelative(MotorDirection(1), Decimal(distance), Int32(100000))
        self.update_current_position()

    def move_fast_to(self, position):
        """
        Function for moving quickly to a specified position
        """
        # Check that intended posiiton is ok
        assert 0 < position < 8

        # Save old settings
        v = self.stepper_channel.GetVelocityParams().MaxVelocity
        a = self.stepper_channel.GetVelocityParams().Acceleration

        # Change motor to move quickly
        self.stepper_channel.SetVelocityParams( # TODO test if my own function works better
                            Decimal(float(2)),
                            Decimal(float(2)))
        # Move to target position
        sleep(self.sleep_time)
        self.stepper_channel.MoveTo(Decimal(position), Int32(100000))
        # Reset the velocity settings
        # TODO use a different function such as jog to move, could be more reliable
        self.stepper_channel.SetVelocityParams(v, a)
        self.c_p['stepper_target_position'][self.axis]  = self.update_current_position()
        self.c_p['stepper_starting_position'][self.axis] = self.c_p['stepper_target_position'][self.axis]

    def move_to_position(self, position):
        distance = position - self.c_p['stepper_current_position'][self.axis]
        self.move_distance(float(distance))

    def get_jog_distance(self):
        jog_distance = self.c_p['stepper_target_position'][self.axis] - self.c_p['stepper_current_position'][self.axis]
        return jog_distance

    def jog_move(self, jog_distance):
        # Does not work well at the moment

        if np.abs(jog_distance) > 1e-4:
            self.stepper_channel.SetJogStepSize(Decimal(float(jog_distance)))
            self.stepper_channel.MoveJog(MotorDirection(1), Int32(10000))

    # TODO z-motor does not like that one changes speed while moving
    def set_velocity_params(self):
        tmp = self.stepper_channel.GetVelocityParams()
        stepper_speed =  float(str(tmp.MaxVelocity).replace(',', '.'))
        trials = 0
        while stepper_speed != self.c_p['stepper_max_speed'][self.axis] and trials < 20:
            try:
                self.stepper_channel.SetVelocityParams(
                    Decimal(float(self.c_p['stepper_max_speed'][self.axis])),
                    Decimal(float(self.c_p['stepper_acc'][self.axis])))
            except Exception as ex:
                print(f"Could  not set velocity params, {ex}")

            tmp = self.stepper_channel.GetVelocityParams()
            stepper_speed =  float(str(tmp.MaxVelocity).replace(',', '.'))
            trials += 1
            sleep(self.sleep_time)
        if trials >= 20:
            print('Failed to set velocity params for motor no ', self.axis)

    def set_jog_velocity_params(self):
        try:
            self.stepper_channel.SetJogVelocityParams(
                Decimal(float(self.c_p['stepper_max_speed'][self.axis])),
                Decimal(float(self.c_p['stepper_acc'][self.axis])))
        except Exception as ex:
            print(f"Could  not set velocity params, {ex}")

    def run(self):

        while self.c_p['program_running']:
            if self.stepper_channel is not None and self.stepper_channel.IsConnected:
                if self.c_p['new_stepper_velocity_params'][self.axis]:
                    self.c_p['new_stepper_velocity_params'][self.axis] = False
                    self.stepper_channel.StopImmediate()
                    self.is_moving = False
                    self.set_velocity_params()

                self.update_current_position()
                jog_distance = self.get_jog_distance()
                move_dir = 1 if np.sign(jog_distance) > 0 else 2

                if move_dir != self.move_direction:
                    self.stepper_channel.StopImmediate()
                    self.is_moving = False
                    self.move_direction = move_dir

                if not self.is_moving and np.abs(jog_distance) >= self.step:

                    # if np.abs(jog_distance < 2e-3):
                    #     # Do a jog move and then update current and target position
                    #     self.move_distance(jog_distance)
                    #     self.c_p['stepper_target_position'][self.axis] = self.update_current_position()
                    # else:
                    self.stepper_channel.MoveContinuous(MotorDirection(self.move_direction))
                    self.is_moving = True

                elif np.abs(jog_distance) <= self.step:
                    self.stepper_channel.StopImmediate()
                    self.is_moving = False
                    #self.c_p['stepper_target_position'][self.axis] = self.update_current_position()

                # Check if "fast" move is required
                if self.c_p['stepper_move_to_target'][self.axis]:
                    self.stepper_channel.StopImmediate()
                    self.is_moving = False
                    self.move_fast_to(self.c_p['stepper_next_move'][self.axis])
                    self.c_p['stepper_move_to_target'][self.axis] = False

            elif self.c_p['connect_steppers']:
                self.connect_channel()
                if self.stepper_channel is not None and self.stepper_channel.IsConnected:
                    self.c_p['stepper_starting_position'][self.axis] = self.update_current_position()
                    self.c_p['stepper_target_position'][self.axis] =  self.c_p['stepper_starting_position'][self.axis]
                    self.c_p['steppers_connected'][self.axis] = True
                    self.set_jog_velocity_params()
                    self.set_velocity_params()
                    self.move_absolute()
            sleep(self.sleep_time)
            if self.stepper_channel is not None:
                self.c_p['steppers_connected'][self.axis] = self.stepper_channel.IsConnected

        # Program is terminating. Stop the motor
        try:
            self.stepper_channel.StopImmediate()
        except AttributeError as AE:
            print(f"Motor stopping failed, {AE}")

        self.__del__()

    def __del__(self):
        try:
            self.stepper_channel.StopImmediate()
            self.stepper_channel.StopPolling()
            self.stepper_channel.Disconnect()
        except Exception as ex:
            print(f"Error {ex}")


class MotorThreadV2(Thread):
        """
        New class for controlling the old k-cube motors using the same interface
        as the XYZ_stepper stage
        """
        def __init__(self, channel, axis, c_p, sleep_time=0.005, step=0.0002):

            Thread.__init__(self)
            self.c_p = c_p
            self.setDaemon(True)
            self.channel = channel
            self.axis = axis
            self.sleep_time = sleep_time
            self.step = step
            self.is_moving = False
            self.move_direction = MotorDirection(1)
            # self.stepper_channel = None # Replaced with motor
            try:
                self.stepper_channel = InitiateMotor(c_p['serial_nums_motors'][self.axis],
                    pollingRate=c_p['polling_rate'])
                self.update_current_position()
                self.c_p['stepper_target_position'][self.axis] = self.c_p['stepper_current_position'][self.axis]
            except Exception as ex:
                print(f"Could  not connect stepper, {ex}")
                self.stepper_channel = None

        def connect_channel(self):
            try:
                self.stepper_channel = ConnectBenchtopStepperChannel(
                    self.c_p['stepper_controller'], self.channel)
            except Exception as ex:
                print(f"Could  not set velocity params, {ex}")

        def update_current_position(self):
            decimal_pos = self.stepper_channel.Position
            self.c_p['stepper_current_position'][self.axis] = float(str(decimal_pos).replace(',','.'))
            return self.c_p['stepper_current_position'][self.axis]

        def move_absolute(self):
            target_pos = Decimal(float(self.c_p['stepper_target_position'][self.axis]))
            self.stepper_channel.MoveTo(target_pos, Int32(100000))

        def move_distance(self, distance):
            self.stepper_channel.MoveRelative(MotorDirection(1), Decimal(distance), Int32(100000))
            self.update_current_position()

        def move_to_position(self, position):
            distance = position - self.c_p['stepper_current_position'][self.axis]
            self.move_distance(float(distance))

        def get_jog_distance(self):
            jog_distance = self.c_p['stepper_target_position'][self.axis] - self.c_p['stepper_current_position'][self.axis]
            return jog_distance

        def jog_move(self, jog_distance):
            # Does not work well at the moment

            if np.abs(jog_distance) > 1e-4:
                self.stepper_channel.SetJogStepSize(Decimal(float(jog_distance)))
                self.stepper_channel.MoveJog(MotorDirection(1), Int32(10000))

        # TODO z-motor does not like that one changes speed while moving
        def set_velocity_params(self):
            tmp = self.stepper_channel.GetVelocityParams()
            stepper_speed =  float(str(tmp.MaxVelocity).replace(',', '.'))
            trials = 0
            while stepper_speed != self.c_p['stepper_max_speed'][self.axis] and trials < 20:
                try:
                    self.stepper_channel.SetVelocityParams(
                        Decimal(float(self.c_p['stepper_max_speed'][self.axis])),
                        Decimal(float(self.c_p['stepper_acc'][self.axis])))
                except Exception as ex:
                    print(f"Could  not set velocity params, {ex}")
                tmp = self.stepper_channel.GetVelocityParams()
                stepper_speed =  float(str(tmp.MaxVelocity).replace(',', '.'))
                trials += 1
                sleep(self.sleep_time)
            if trials >= 20:
                print('Falsed to set velocity params for motor no ', self.axis)

        def set_jog_velocity_params(self):
            try:
                self.stepper_channel.SetJogVelocityParams(
                    Decimal(float(self.c_p['stepper_max_speed'][self.axis])),
                    Decimal(float(self.c_p['stepper_acc'][self.axis])))
            except Exception as ex:
                print(f"Could  not set velocity params, {ex}")

        def run(self):

            while self.c_p['program_running']:
                # Disconnect how?
                if self.stepper_channel is not None and self.stepper_channel.IsConnected:
                    if self.c_p['new_stepper_velocity_params'][self.axis]:
                        self.c_p['new_stepper_velocity_params'][self.axis] = False
                        self.stepper_channel.StopImmediate()
                        self.is_moving = False
                        self.set_velocity_params()

                    self.update_current_position()
                    jog_distance = self.get_jog_distance()
                    move_dir = 1 if np.sign(jog_distance) > 0 else 2

                    if move_dir != self.move_direction:
                        self.stepper_channel.StopImmediate()
                        self.is_moving = False
                        self.move_direction = move_dir
                        print("Changing move_dir")

                    if not self.is_moving and np.abs(jog_distance) >= self.step:
                        if np.abs(jog_distance < 1e-3):
                            # Do a jog move and then update current and target position
                            self.move_distance(jog_distance)
                            self.c_p['stepper_target_position'][self.axis] = self.update_current_position()
                        else:
                            self.stepper_channel.MoveContinuous(MotorDirection(self.move_direction))
                            self.is_moving = True

                    elif np.abs(jog_distance) <= self.step:
                        self.stepper_channel.StopImmediate()
                        self.is_moving = False
                        self.c_p['stepper_target_position'][self.axis] = self.update_current_position()


                elif self.c_p['connect_steppers']:
                    self.stepper_channel = InitiateMotor(self.c_p['serial_nums_motors'][self.axis],
                        pollingRate=self.c_p['polling_rate'])

                    if self.stepper_channel is not None and self.stepper_channel.IsConnected:
                        self.c_p['stepper_starting_position'][self.axis] = self.update_current_position()
                        self.c_p['stepper_target_position'][self.axis] =  self.c_p['stepper_starting_position'][self.axis]
                        self.c_p['steppers_connected'][self.axis] = True
                        self.set_jog_velocity_params()
                        self.set_velocity_params()
                        self.move_absolute()
                sleep(self.sleep_time)
                if self.stepper_channel is not None:
                    self.c_p['steppers_connected'][self.axis] = self.stepper_channel.IsConnected

            # Program is terminating. Stop the motor
            try:
                self.stepper_channel.StopImmediate()
            except Exception as ex:
                print(f"Could  not stop properly, {ex}")

            self.__del__()

        def __del__(self):
            try:
                self.stepper_channel.StopImmediate()
                self.stepper_channel.StopPolling()
                self.stepper_channel.Disconnect()
            except Exception as ex:
                print(f"Could disconnect properly, {ex}")
