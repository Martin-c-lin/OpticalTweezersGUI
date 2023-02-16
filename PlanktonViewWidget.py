import numpy as np
from threading import Thread
from time import sleep
from CustomMouseTools import MouseInterface
from PyQt6.QtGui import  QPixmap,QImage
from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
)


class PlanktonViewer(QWidget):
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        self.label = QLabel("Deep larning controller")

        self.image = np.load("C:/Users/marti/OneDrive/PhD/OT software/Example data/image_4.npy")

        QT_Image = QImage(self.image, self.image.shape[1],
                                       self.image.shape[0],
                                       QImage.Format.Format_Grayscale8)
        self.label.setPixmap(QPixmap.fromImage(QT_Image))
        layout.addWidget(self.label)

        self.setLayout(layout)

