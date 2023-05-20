from pickle import TRUE
from PyQt5 import QtWidgets
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QSlider,
    QVBoxLayout,
    QDoubleSpinBox,
    QWidget,
    QFileDialog,
    QPushButton,
    QGridLayout,
    QScrollArea,
)
import cv2
import math
import numpy as np
from skimage.measure import label, regionprops
from scipy.spatial import KDTree, distance_matrix
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io


class TriangulateImage (QtWidgets.QWidget):
      
    def __init__(self, widget_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.widgets = widget_dict
        self.load_image_btn = QPushButton("Print", self)
        self.load_image_btn.clicked.connect(self.pnpproblem)
        self.original_image_label = QLabel(self)

        # Create grid layout
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)

        # Add widgets to grid layout
        self.grid_layout.addWidget(self.load_image_btn, 0, 0, 1, 4)
        self.grid_layout.setColumnStretch(2, 1)  # add stretch to the empty cell

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.bmp);;All Files (*)",
            options=options,
        )
        if file_name:
            self.original_image = cv2.imread(file_name)

            self.display_image(self.original_image, self.original_image_label)
            # Generate initial segmentation mask
            self.generate_mask()

    def display_image(self, img, label):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def process_data(self, data):
        coordinates = []
        for targets in data:
            for target in targets:
                if 'center' in target:
                    coordinates.append(target['center'])
        return coordinates
    
    def pnpproblem(self):
        stuff1 = self.widgets['Camera 11 RGB Left'].ret_sorted_clusters()
        sorted_cluster_2D = np.array(self.process_data(stuff1), dtype=np.float32)

        stuff2 = self.widgets['Camera 11 RGB Left'].ret_3D_coords()
        sorted_cluster_3D = np.array(self.process_data(stuff2), dtype=np.float32)

        # Define the camera intrinsic matrix 
        # Left Camera intrinsic parameters
        f_x = 697.4555053710938
        f_y = 697.4555053710938
        c_x = 621.2157592773438
        c_y = 353.8743896484375
        camera_matrix = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]], dtype=np.float32)

        # Estimate the camera pose using the Perspective-n-Point (PnP) algorithm
        _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(sorted_cluster_3D, sorted_cluster_2D, camera_matrix, None)

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Define the arrow length for visualization
        arrow_length = 10

        # Calculate the endpoint of the arrow based on camera position and direction
        arrow_end = rotation_matrix.T @ np.array([0, 0, arrow_length])

        # Plot the 3D points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sorted_cluster_3D[:, 0], sorted_cluster_3D[:, 1], sorted_cluster_3D[:, 2], c='blue')

        # Plot the camera position as a red arrow
        origin = translation_vector.flatten()
        ax.quiver(*origin, *arrow_end, color='red')

        # Set axes labels and display the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()