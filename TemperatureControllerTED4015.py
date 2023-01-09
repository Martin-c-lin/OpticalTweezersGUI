# Control of the temperature controller

import pyvisa
import numpy as np
from time import sleep, time
from threading import Thread


def get_resource_list():
    '''
    Returns
    -------
    resuoure manager list
        List of pyvisa resources.

    '''
    rm = pyvisa.ResourceManager()
    return rm.list_resources()


class TED4015():
    """
    Class for controlling the TED4015 instrument.
    Initiates the instrument to upon creation.
    Essentially a wrapper class for the visa resource
    """

    def __init__(self, index=0):
        '''
        Parameters
        ----------
        index : int, optional
            Index of the temperature controller int the pyvisa list.
            The default is 0.

        Returns
        -------
        None.

        '''
        try:
            self.rm = pyvisa.ResourceManager()
            self.resource_list = self.rm.list_resources()
            self.TED4015 = self.rm.open_resource(self.resource_list[index])
            self.TED4015.read_termination = '\n'
            self.TED4015.write_termination = '\n'
        except Exception as ex:
            self.TED4015 = None
    def get_setpoint_temperature(self):
        """
        Returns the setpoint temperature of the instrument
        """
        return float(self.TED4015.query('SOUR:TEMP?'))

    def set_setpoint_temperature(self, temperature):
        """
        Sets the target temperature(setpoint temperature) of the instrument
        """
        # TODO are there any limits to te temperature?
        if 0 < temperature < 40:
            temperature_command = 'SOUR:TEMP '+str(temperature)+'C'
            return self.TED4015.write(temperature_command)
        else:
            print('Temperature not in OK range!')
    def query_output(self):
        """
        Checks if the output is on or off.
        0 => off
        1 => on
        """
        return self.TED4015.query('OUTP?')

    def turn_on_output(self):
        """
        Activates the output of the TED4015
        """
        return self.TED4015.write('OUTP ON')

    def turn_off_output(self):
        """
        Deactivates the output of the TED4015
        """
        return self.TED4015.write('OUTP OFF')

    def measure_temperature(self):
        """
        Measueres the temperature of the thermocouple.
        Note that this is not the same as the setpoint temperature.
        """
        return float(self.TED4015.query('MEAS:SCAL:TEMP?'))

    def measure_power(self):
        """
        Returns the instananeous power consumption of the intrument
        """
        return float(self.TED4015.query('MEAS:SCAL:POW?'))

    def close_device(self):
        """
        Closes the communication to the device
        """
        return self.TED4015.close()

    def query_min_current(self):
        """
        Check minimum current(in Amps) setting on the temperature controller
        """
        return float(self.TED4015.query('SOURce:CURRent? MINimum'))

    def query_max_current(self):
        """
        Check maximum current(in Amps) setting on the temperature controller
        """
        return float(self.TED4015.query('SOURce:CURRent? MAXimum'))

    def set_max_current(self, current):
        """
        Sets the maximum output current of the device
        """
        self.TED4015.write('OUTP OFF')
        sleep(1)  # Wait for the output to be turned off
        command = 'SOUR:CURR:LIM '+str(current)
        self.TED4015.write(command)
        self.TED4015.write('OUTP ON')
        return self.query_max_current()

    def set_gain(self, gain):
        """
        Function for setting the gain of the PID controller (P in PID).
        Default should be around 50-60 in our case as a reference
        """
        command = 'SOURce:TEMPerature:LCONstants:GAIN ' + str(gain)
        return self.TED4015.write(command)

    def query_gain(self):
        """
        Function for reading the gain of the PID controller (P in PID).
        """
        return float(self.TED4015.query('SOURce:TEMPerature:LCONstants:GAIN?'))


class TemperatureThread(Thread):
        '''
        Class for running the temperature controller in the background
        '''
        def __init__(self, threadID, name, c_p, data, temperature_controller=None, max_diff=0.05):
            '''
            Parameters
            ----------
            threadID : int
                Thread id number.
            name : String
                Name of thread.
            temperature_controller : temperaturecontroller, optional
                Controller of objective temperature. The default is None.
            max_diff : Float, optional
                 Maximum value by which temperature is allowed to deviate from
                 target temperature for temperature to be considered as stable.
                 The default is 0.01.

            Returns
            -------
            None.

            '''
            Thread.__init__(self)
            self.threadID = threadID
            self.name = name
            self.data = data # Dictionary in which data is saved
            self.temperature_history = []
            self.temp_hist_length = 100
            self.max_diff = max_diff
            self.start_time = time()

            if temperature_controller is not None:
                self.temperature_controller = temperature_controller
                c_p['starting_temperature'] =\
                    self.temperature_controller.measure_temperature()
                c_p['current_temperature'] =\
                    c_p['starting_temperature']
                c_p['setpoint_temperature'] = c_p['starting_temperature']
                c_p['temperature_controller_connected'] = True
            else:
                try:
                    self.temperature_controller = TED4015()
                    c_p['starting_temperature'] =\
                        self.temperature_controller.measure_temperature()
                    c_p['current_temperature'] =\
                        c_p['starting_temperature']
                    c_p['setpoint_temperature'] = c_p['starting_temperature']
                    c_p['temperature_controller_connected'] = True
                except Exception as ex:
                    # Handling the case of not having a temperature controller
                    print(f"\nWARNING, COULD NOT ESTABLISH CONTACT WITH \
                          TEMEPERATURE CONTROLLER!\n {ex}")
                    self.temperature_controller = None
            self.c_p = c_p
            self.setDaemon(True)

        def run(self):
            c_p = self.c_p
            self.data['T_time'][0] = time() - self.start_time
            self.data['Temperature'][0] = c_p['starting_temperature']

            if self.temperature_controller is not None:
                # Turn on output and continuosly set and query the temperature.
                if c_p['temperature_output_on']:
                    self.temperature_controller.turn_on_output()
                while c_p['program_running']:
                    if 0 < c_p['setpoint_temperature'] < 40:
                        self.temperature_controller.set_setpoint_temperature(c_p['setpoint_temperature'])
                    else:
                        print('Setpoint temperature NOK')
                    c_p['current_temperature'] =\
                        self.temperature_controller.measure_temperature()
                    self.temperature_history.append(
                        c_p['current_temperature'])
                    self.data['Temperature'].append(c_p['current_temperature'])
                    self.data['T_time'].append(time()-self.start_time)

                    # Update and check history
                    if len(self.temperature_history)>self.temp_hist_length:
                        self.temperature_history.pop()
                    history = [T-c_p['setpoint_temperature'] for T in self.temperature_history]
                    if max(np.abs(history))<self.max_diff:
                        c_p['temperature_stable'] = True

                    else:
                        c_p['temperature_stable'] = False

                    # Check output and if it shoould be on
                    O = int(self.temperature_controller.query_output())
                    if O == 0 and c_p['temperature_output_on']:
                        print("Turning on")
                        self.temperature_controller.turn_on_output()
                    elif O == 1 and not c_p['temperature_output_on']:
                        self.temperature_controller.turn_off_output()
                    sleep(0.05) # We do not need to update the temperature very often
                self.temperature_controller.turn_off_output()
