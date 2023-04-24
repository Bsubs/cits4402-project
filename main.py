import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QDoubleSpinBox, QWidget, QFileDialog, QPushButton, QGridLayout, QScrollArea
import cv2
import numpy as np
from skimage.measure import label, regionprops
from scipy.spatial import KDTree, distance_matrix
from scipy.optimize import least_squares


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
        self.hexagon_label= QLabel(self)

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
        self.taxisRatio_slider.setRange(0, 10)
        self.taxisRatio_slider.setSingleStep(0.1)
        self.cca_btn = QPushButton("Connected Components Analysis", self)
        # taxisratio=1 would be a perfect circle, 0.8 allows for some noise
        self.taxisRatio = 2.5
        # Connect widgets to update functions
        self.taxisRatio_slider.valueChanged.connect(self.update_taxisRatio)
        self.cca_btn.clicked.connect(self.filter_clusters)
        # Create labels for slider widgets
        taxisRatio_label = QLabel('taxisRatio: ')

        # Create slider widgets for tellipse
        self.tellipse_slider = QDoubleSpinBox()
        self.tellipse_slider.setRange(0, 10)
        self.tellipse_slider.setSingleStep(0.1)
        self.cca_btn = QPushButton("Connected Components Analysis", self)
        self.tellipse = 7
        # Connect widgets to update functions
        self.tellipse_slider.valueChanged.connect(self.update_tellipse)
        self.cca_btn.clicked.connect(self.filter_clusters)
        # Create labels for slider widgets
        tellipse_label = QLabel('tellipse: ')

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
        grid_layout.addWidget(tellipse_label, 7, 0)
        grid_layout.addWidget(self.tellipse_slider, 7, 1, 1, 3)
        grid_layout.addWidget(self.cca_btn, 8, 0, 1, 4)
        grid_layout.addWidget(self.cca_label, 9, 1)
        grid_layout.addWidget(self.hexagon_label, 9, 2)
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

    def filter_clusters(self):
        # Convert image to grayscale
        gray_image = cv2.cvtColor(self.masked_image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to obtain binary image
        _, self.binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

       # Perform connected component analysis on binary image using skimage.measure.label()
        self.labeled_image = label(self.binary_image, connectivity=2)
        self.props = regionprops(self.labeled_image)

        # Filter out small and large clusters based on area thresholds
        cluster_mask = np.zeros_like(self.binary_image)
        for prop in self.props:
            if self.tminArea <= prop.area <= self.tmaxArea:
                # Compute minor and major axis lengths and their ratio
                sigma_min, sigma_max = np.sqrt(prop.inertia_tensor_eigvals)
                ratio = sigma_min / sigma_max

                # Filter out clusters with minor-to-major axis ratio below threshold
                if ratio < self.taxisRatio:
                    cluster_mask[self.labeled_image == prop.label] = 255

        # Apply mask to original image
        self.eigen_image = cv2.bitwise_and(self.masked_image, self.masked_image, mask=cluster_mask)
        self.display_image(self.eigen_image, self.cca_label)
        self.find_target_clusters()

    def find_target_clusters(self):
        # Find the five nearest clusters for each cluster
        centroids = np.array([prop.centroid for prop in self.props])
        kdtree = KDTree(centroids)
        nearest_indices = kdtree.query(centroids, k=6)[1][:, 1:]
        
        # computes residual error of the ellipse fit 
        def ellipse_residual_error(params, selected_centroids):
            # Helper function to compute residual error of ellipse fitting
            x0, y0, a, b, theta = params
            t = np.linspace(0, 2 * np.pi, 100)
            ellipse_points = np.column_stack((a * np.cos(t), b * np.sin(t)))
            # rotates ellipse points by angle theta
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # translates them to center coordinates
            ellipse_points_rotated = np.dot(ellipse_points, rotation_matrix) + np.array([x0, y0])
            # distance matrix between rotated ellipse points and selected centroids
            dists = distance_matrix(selected_centroids, ellipse_points_rotated)
            # compute squared distances from each centroid to the closest ellipse point
            min_dists = np.min(dists, axis=1)
            return min_dists ** 2

        # Fit an ellipse on the six centroids
        target_clusters = []
        for i, nearest in enumerate(nearest_indices):
            selected_centroids = np.vstack((centroids[i], centroids[nearest]))
            x0, y0 = np.mean(selected_centroids, axis=0)
            a, b = (np.max(selected_centroids, axis=0) - np.min(selected_centroids, axis=0)) / 2
            theta = 0
            initial_params = [x0, y0, a, b, theta]

            # Compute residual error
            result = least_squares(ellipse_residual_error, initial_params, args=(selected_centroids,))
            residual_error = np.sum(result.fun)

            # Check if the largest residual error is less than the threshold
            if residual_error < self.tellipse:
                target_clusters.append(i)

        # Apply a mask to the original image to display only the target clusters
        target_mask = np.zeros_like(self.binary_image)
        target_mask[np.isin(self.labeled_image, [self.props[i].label for i in target_clusters])] = 255
        target_image = cv2.bitwise_and(self.eigen_image, self.eigen_image, mask=target_mask)
        self.display_image(target_image, self.hexagon_label)
    
    # Functions for sliders
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

    def update_tminArea(self, value):
        self.tminArea = value
    
    def update_tmaxArea(self, value):
        self.tmaxArea = value

    def update_taxisRatio(self, value):
        self.taxisRatio = value

    def update_tellipse(self, value):
        self.tellipse = value

        
def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()