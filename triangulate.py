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
        self.load_image_btn.clicked.connect(self.plot_triangulate)
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

    def plot_triangulate(self):
    
        stuff1 = self.widgets['Camera 11 RGB Left'].ret_sorted_clusters()
        sorted_cluster_left = np.array(self.process_data(stuff1), dtype=np.float32)
        image_left = self.widgets['Camera 11 RGB Left'].ret_masked_img()
        
        stuff2 = self.widgets['Camera 11 RGB Right'].ret_sorted_clusters()
        sorted_cluster_right = np.array(self.process_data(stuff2), dtype=np.float32)
        image_right = self.widgets['Camera 11 RGB Right'].ret_masked_img()

        # Left Camera intrinsic parameters
        fx_l = 640
        fy_l = 640
        cx_l = 641.63525390625
        cy_l = 355.5729675292969

        # Right Camera intrinsic parameters
        fx_r = 640
        fy_r = 640
        cx_r = 641.63525390625
        cy_r = 355.5729675292969

        # Intrinsic matrix for left and right camera
        K_l = np.array([[fx_l, 0, cx_l],
                        [0, fy_l, cy_l],
                        [0, 0, 1]])

        K_r = np.array([[fx_r, 0, cx_r],
                        [0, fy_r, cy_r],
                        [0, 0, 1]])

        # Extrinsic parameters for left camera (reference camera)
        R_l = np.eye(3)
        t_l = np.zeros((3, 1))

        # Extrinsic matrix for left camera
        E_l = np.hstack((R_l, t_l))

        # Stereo parameters translation 
        oofs_vals = np.array([119.9543228149414, 0.005216991528868675, 0.03254878148436546])

        # Rotation for right camera from Euler angles
        oofsr_vals = np.array([0.0027044836897403, -0.018978217616677284, 0.0011518205283209682])
        roll, pitch, yaw = oofsr_vals

        # Compute rotation matrix
        R_r = np.array([
            [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
            [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)],
            [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
        ])

        # Translation for right camera is the stereo parameters translation
        t_r = oofs_vals.reshape(3, 1)

        # Extrinsic matrix for right camera
        E_r = np.hstack((R_r, t_r))

        # Projection matrices
        projection_left = np.dot(K_l, E_l)
        projection_right = np.dot(K_r, E_r)


        points_3d = cv2.triangulatePoints(projection_left, projection_right, sorted_cluster_left.T, sorted_cluster_right.T)
        print('test')
        #points_3d = points_3d / points_3d[3]  # Homogeneous coordinates to 3D Cartesian
        print('test')
        points_3d = points_3d[:3].T
        print('test')
        print(points_3d)
        
        # rvec = np.zeros((3, 1))
        # tvec = np.zeros((3, 1))

        # #success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, sorted_cluster_left, camera_matrix_left, np.zeros((5,)))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def process_data(self, data):
        coordinates = []
        for targets in data:
            for target in targets:
                if 'center' in target:
                    coordinates.append(target['center'])
        return coordinates