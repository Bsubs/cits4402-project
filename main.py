import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QDoubleSpinBox, QWidget, QFileDialog, QPushButton, QGridLayout, QScrollArea
import cv2
import numpy as np
from skimage.measure import label, regionprops


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create load image widget
        self.load_image_btn = QPushButton("Load Image", self)
        self.load_image_btn.clicked.connect(self.load_image)

        # Create label for displaying image
        self.original_image_label = QLabel(self)
        self.masked_image_label = QLabel(self)
        self.cca_label = QLabel(self)
        self.eigen_label = QLabel(self)

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

        # Create slider widgets for tminArea and tmaxArea
        self.tminArea_slider = QSlider(Qt.Horizontal)
        self.tminArea_slider.setMaximum(200)
        self.tmaxArea_slider = QSlider(Qt.Horizontal)
        self.tmaxArea_slider.setMaximum(200)
        # Set default values for tminArea and tmaxArea
        self.tminArea = 40
        self.tmaxArea = 150
        # Connect slider widgets to update functions
        self.tminArea_slider.valueChanged.connect(self.update_tminArea)
        self.tmaxArea_slider.valueChanged.connect(self.update_tmaxArea)
        # Create labels for slider widgets
        tminArea_label = QLabel('tminArea: ')
        tmaxArea_label = QLabel('tmaxArea: ')

        # Create slider widgets for taxisratio
        self.taxisRatio_slider = QDoubleSpinBox()
        self.taxisRatio_slider.setRange(0, 1)
        self.taxisRatio_slider.setSingleStep(0.1)
        self.cca_btn = QPushButton("Connected Components Analysis", self)
        # taxisratio=1 would be a perfect circle, 0.8 allows for some noise
        self.taxisRatio = 0.8
        # Connect widgets to update functions
        self.taxisRatio_slider.valueChanged.connect(self.update_taxisRatio)
        self.cca_btn.clicked.connect(self.filter_clusters)
        # Create labels for slider widgets
        taxisRatio_label = QLabel('taxisRatio: ')

        # Create layout for widgets
        layout = QVBoxLayout()

        # Create grid layout
        grid_layout = QGridLayout()

        # Add widgets to grid layout
        grid_layout.addWidget(self.load_image_btn, 0, 0, 1, 4)
        grid_layout.addWidget(tminColor_label, 1, 0)
        grid_layout.addWidget(self.tminColor_slider, 1, 1, 1, 3)
        grid_layout.addWidget(tdiffColor_label, 2, 0)
        grid_layout.addWidget(self.tdiffColor_slider, 2, 1, 1, 3)
        grid_layout.addWidget(self.original_image_label, 3, 1)
        grid_layout.addWidget(self.masked_image_label, 3, 2)
        grid_layout.addWidget(tminArea_label, 4, 0)
        grid_layout.addWidget(self.tminArea_slider, 4, 1, 1, 3)
        grid_layout.addWidget(tmaxArea_label, 5, 0)
        grid_layout.addWidget(self.tmaxArea_slider, 5, 1, 1, 3)
        grid_layout.addWidget(taxisRatio_label, 6, 0)
        grid_layout.addWidget(self.taxisRatio_slider, 6, 1, 1, 3)
        grid_layout.addWidget(self.cca_btn, 7, 0, 1, 4)
        grid_layout.addWidget(self.cca_label, 8,1)
        grid_layout.setColumnStretch(2, 1)  # add stretch to the empty cell

        # Add grid layout to main layout
        layout.addLayout(grid_layout)

        # Create central widget and set layout
        self.scroll = QScrollArea()
        central_widget = QWidget()
        central_widget.setLayout(layout)

        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(central_widget)

        # Set central widget
        self.setCentralWidget(self.scroll)

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
        self.masked_image = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)

        # Update image label with new image
        self.display_image(self.masked_image, self.masked_image_label)

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

    def filter_clusters(self):
        # Convert image to grayscale
        gray_image = cv2.cvtColor(self.masked_image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to obtain binary image
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

       # Perform connected component analysis on binary image using skimage.measure.label()
        labeled_image = label(binary_image, connectivity=2)
        props = regionprops(labeled_image)

        # Filter out small and large clusters based on area thresholds
        cluster_mask = np.zeros_like(binary_image)
        for prop in props:
            if self.tminArea <= prop.area <= self.tmaxArea:
                # Compute covariance matrix of pixel coordinates in region
                coords = prop.coords
                cov = np.cov(coords.T)

                # Compute eigenvalues and eigenvectors of covariance matrix
                evals, evecs = np.linalg.eig(cov)

                # Sort eigenvalues in descending order
                idx = np.argsort(evals)[::-1]
                evals = evals[idx]
                evecs = evecs[:, idx]

                # Compute minor and major axis lengths and their ratio
                sigma_min, sigma_max = np.sqrt(evals)
                ratio = sigma_min / sigma_max

                # Filter out clusters with minor-to-major axis ratio below threshold
                if 3 > ratio > self.taxisRatio:
                    print(ratio)
                    cluster_mask[labeled_image == prop.label] = 255

        # Apply mask to original image
        filtered_image = cv2.bitwise_and(self.masked_image, self.masked_image, mask=cluster_mask)
        self.display_image(filtered_image, self.cca_label)
    
    def update_tminArea(self, value):
        self.tminArea = value
    
    def update_tmaxArea(self, value):
        self.tmaxArea = value

    def update_taxisRatio(self, value):
        self.taxisRatio = value

        
def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()