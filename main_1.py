import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QSlider, QFormLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.load_image_btn = QPushButton("Load Image", self)
        self.load_image_btn.clicked.connect(self.load_image)

        self.original_image_label = QLabel(self)
        self.masked_image_label = QLabel(self)

        self.red_slider = QSlider(Qt.Horizontal, self)
        self.green_slider = QSlider(Qt.Horizontal, self)
        self.blue_slider = QSlider(Qt.Horizontal, self)
        self.red_slider.setMinimum(0)
        self.blue_slider.setMinimum(0)
        self.green_slider.setMinimum(0)
        self.red_slider.setMaximum(255)
        self.blue_slider.setMaximum(255)
        self.green_slider.setMaximum(255)
        self.red_slider.valueChanged.connect(self.update_thresholds)
        self.green_slider.valueChanged.connect(self.update_thresholds)
        self.blue_slider.valueChanged.connect(self.update_thresholds)

        sliders_layout = QFormLayout()
        sliders_layout.addRow("Red Threshold", self.red_slider)
        sliders_layout.addRow("Green Threshold", self.green_slider)
        sliders_layout.addRow("Blue Threshold", self.blue_slider)

        layout = QVBoxLayout()
        layout.addWidget(self.load_image_btn)
        layout.addLayout(sliders_layout)

        images_layout = QHBoxLayout()
        images_layout.addWidget(self.original_image_label)
        images_layout.addWidget(self.masked_image_label)
        layout.addLayout(images_layout)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.original_image = cv2.imread(file_name)
            self.display_image(self.original_image, self.original_image_label)
            self.update_thresholds()

    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def update_thresholds(self):
        red_threshold = self.red_slider.value()
        green_threshold = self.green_slider.value()
        blue_threshold = self.blue_slider.value()

        self.lower_red = np.array([0, 0, 255 - red_threshold])
        self.upper_red = np.array([50, 50, 255])
        self.lower_green = np.array([0, 255 - green_threshold, 0])
        self.upper_green = np.array([50, 255, 50])
        self.lower_blue = np.array([255 - blue_threshold, 0, 0])
        self.upper_blue = np.array([255, 50, 50])

        if hasattr(self, 'original_image'):
            self.apply_mask_and_display()

    def apply_mask_and_display(self):

        # Apply masks for each color and combine them.
        red_mask = cv2.inRange(self.original_image, self.lower_red, self.upper_red)
        green_mask = cv2.inRange(self.original_image, self.lower_green, self.upper_green)
        blue_mask = cv2.inRange(self.original_image, self.lower_blue, self.upper_blue)
        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

        masked_image = cv2.bitwise_and(self.original_image, self.original_image, mask=combined_mask)
        self.display_image(masked_image, self.masked_image_label)

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
