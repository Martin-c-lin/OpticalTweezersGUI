
from PyQt6.QtWidgets import (
    QPushButton, QVBoxLayout, QWidget, QLabel
)

class ObjectiveStepperController(QWidget):
    """
    Simple widget to control the objective stepper motor. Can move the stepper motor
    either towards or away from the sample. Can move slowly or fast (big or small steps).
    """
    def __init__(self, c_p, ArduinoUnoSerial):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        self.last_write = "Q"
        self.ArduinoUnoSerial = ArduinoUnoSerial

        self.label = QLabel("Objective controller")
        layout.addWidget(self.label)
        
        # Add move towards button
        self.slow_towards_button = QPushButton('Slow towards sample')
        self.slow_towards_button.pressed.connect(self.slow_towards_sample)
        self.slow_towards_button.setCheckable(False)
        layout.addWidget(self.slow_towards_button)

        # Add move away button
        self.slow_away_button = QPushButton('Slow away from sample')
        self.slow_away_button.pressed.connect(self.slow_away_from_sample)   
        self.slow_away_button.setCheckable(False)
        layout.addWidget(self.slow_away_button)

        # Add move fast towards button
        self.fast_towards_button = QPushButton('Fast towards sample')
        self.fast_towards_button.pressed.connect(self.fast_towards_sample)
        self.fast_towards_button.setCheckable(False)
        layout.addWidget(self.fast_towards_button)

        # Add move fast away button
        self.fast_away_sample_button = QPushButton('Fast away from sample')
        self.fast_away_sample_button.pressed.connect(self.fast_away_from_sample)
        self.fast_away_sample_button.setCheckable(False)
        layout.addWidget(self.fast_away_sample_button)

        self.setLayout(layout)
    
    def slow_towards_sample(self):
        self.last_write = 'Q'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)
        #print('Moving slowly towards sample.')

    def slow_away_from_sample(self):
        self.last_write = 'W'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)
        #print('Moving slowly away from sample.')

    def fast_towards_sample(self):
        self.last_write = 'E'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)
        #print('Moving fast towards sample.')

    def fast_away_from_sample(self):
        self.last_write = 'R'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)
        #print('Moving fast away from sample.')