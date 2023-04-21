import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QPushButton
import cv2
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create load image widget
        self.load_image_btn = QPushButton("Load Image", self)
        self.load_image_btn.clicked.connect(self.load_image)

        # Create label for displaying image
        self.original_image_label = QLabel(self)
        self.masked_image_label = QLabel(self)

        # Create slider widgets for tminColor and tdiffColor
        self.tminColor_slider = QSlider(Qt.Horizontal)
        self.tdiffColor_slider = QSlider(Qt.Horizontal)

        # Set default values for tminColor and tdiffColor
        self.tminColor = 50
        self.tdiffColor = 100

        # Connect slider widgets to update functions
        self.tminColor_slider.valueChanged.connect(self.update_tminColor)
        self.tdiffColor_slider.valueChanged.connect(self.update_tdiffColor)

        # Create labels for slider widgets
        tminColor_label = QLabel('tminColor: ')
        tdiffColor_label = QLabel('tdiffColor: ')

        # Create layout for slider widgets and labels
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(tminColor_label)
        slider_layout.addWidget(self.tminColor_slider)
        slider_layout.addWidget(tdiffColor_label)
        slider_layout.addWidget(self.tdiffColor_slider)

        # Create layout for widgets
        layout = QVBoxLayout()
        layout.addWidget(self.load_image_btn)
        layout.addLayout(slider_layout)

        images_layout = QHBoxLayout()
        images_layout.addWidget(self.original_image_label)
        images_layout.addWidget(self.masked_image_label)
        layout.addLayout(images_layout)

        # Create central widget and set layout
        central_widget = QWidget()
        central_widget.setLayout(layout)

        # Set central widget
        self.setCentralWidget(central_widget)

        # Set window properties
        self.setWindowTitle('Image Segmentation')
        self.resize(800, 600)


    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.original_image = cv2.imread(file_name)
            self.display_image(self.original_image, self.original_image_label)
            # Generate initial segmentation mask
            self.generate_mask()


    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
    
    def generate_mask(self):
        # Compute minimum and maximum intensities for each pixel
        min_intensities = np.min(self.original_image, axis=2)
        max_intensities = np.max(self.original_image, axis=2)

        # Create mask based on threshold values
        mask = ((min_intensities < self.tminColor) | (max_intensities - min_intensities > self.tdiffColor)).astype(np.uint8) * 255

        # Apply mask to original original_image
        result = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)

        # Update image label with new image
        self.display_image(result, self.masked_image_label)

    def update_tminColor(self, value):
        # Update tminColor value
        self.tminColor = value

        # Regenerate segmentation mask
        self.generate_mask()

    def update_tdiffColor(self, value):
        # Update tdiffColor value
        self.tdiffColor = value

        # Regenerate segmentation mask
        self.generate_mask()

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()