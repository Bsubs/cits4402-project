from pickle import TRUE
from PyQt5 import QtWidgets
import sys
import os
import json
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QLabel,
    QFileDialog,
    QPushButton,
    QGridLayout,
)
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io


class TriangulateImage (QtWidgets.QWidget):
      
    def __init__(self, widget_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load information from widgets
        self.widgets = widget_dict

        # Create Array to store 3D points & camera poses
        self.overall_3D_points = []
        self.overall_camera_pose = []

        # Camera Calibration buttons
        self.calibrate_11_L_btn = QPushButton("Calibrate 11 Left", self)
        self.calibrate_11_L_btn.clicked.connect(self.calibrate_11_L)
        self.original_image_label = QLabel(self)

        self.calibrate_11_R_btn = QPushButton("Calibrate 11 Right", self)
        self.calibrate_11_R_btn.clicked.connect(self.calibrate_11_R)

        self.calibrate_72_btn = QPushButton("Calibrate 72", self)
        self.calibrate_72_btn.clicked.connect(self.calibrate_72)

        # Create grid layout
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)

        # Add widgets to grid layout
        self.grid_layout.addWidget(self.calibrate_11_L_btn, 0, 0, 1, 4)
        self.grid_layout.addWidget(self.calibrate_11_R_btn, 1, 0, 1, 4)
        self.grid_layout.addWidget(self.calibrate_72_btn, 2, 0, 1, 4)
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
    
    def calibrate_11_L(self):
        stuff1 = self.widgets['Camera 11 RGB Left'].ret_sorted_clusters()
        sorted_cluster_2D = np.array(self.process_data(stuff1), dtype=np.float32)

        self.cam_11_L_3D = self.widgets['Camera 11 RGB Left'].ret_3D_coords()
        sorted_cluster_3D = np.array(self.process_data(self.cam_11_L_3D), dtype=np.float32)

        self.overall_3D_points.append(sorted_cluster_3D)

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

        arrow_params = [origin, arrow_end]
        self.overall_camera_pose.append(arrow_params)

        # Set axes labels and display the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def get_matching_coordinates(self, coords_2D1, coords_2D2, coords_3D):
        # Flatten the data structures into lists of labels
        labels_1 = [item['label'] for sublist in coords_2D1 for item in sublist if 'label' in item]
        labels_2 = [item['label'] for sublist in coords_2D2 for item in sublist if 'label' in item]

        # Find matching labels
        matching_labels = set(labels_1).intersection(labels_2)

        coordinates_2D_1 = []
        coordinates_2D_2 = []
        coordinates_2D_remainder = []
        coordinates_3D = []

        for label in matching_labels:
            for targets in coords_2D1:
                if targets[0]['label'] == label:
                    for target in targets:
                        if 'center' in target:
                            coordinates_2D_1.append(target['center'])
        
            for targets in coords_2D2:
                if targets[0]['label'] == label:
                    for target in targets:
                        if 'center' in target:
                            coordinates_2D_2.append(target['center'])

            for targets in coords_3D:
                if targets[0]['label'] == label:
                    for target in targets[1:]:
                        if 'center' in target:
                            coordinates_3D.append(target['center'])

        for targets in coords_2D2:
            isRemainder = True
            for label in matching_labels:
                if targets[0]['label'] == label:
                    isRemainder = False
            
            if isRemainder:
                for target in targets:
                    coordinates_2D_remainder.append(target['center'])

        return matching_labels, np.array(coordinates_2D_1, dtype=np.float32), np.array(coordinates_2D_2, dtype=np.float32), np.array(coordinates_2D_remainder, dtype=np.float32), np.array(coordinates_3D, dtype=np.float32)

    def calibrate_11_R(self):

        # Camera 11_R Intrinsic Parameters
        camera_11_R_json = self.widgets['Camera 11 RGB Right'].jsonName
        json_path_R = os.path.join("data", "camera parameters", camera_11_R_json)

        with open(json_path_R, 'r') as f:
            data = json.load(f)

        # Camera 11_R intrinsic parameters
        fx2 = data["f"]["val"]
        fy2 = data["f"]["val"]
        cx2 = data["ocx"]["val"]
        cy2 = data["ocy"]["val"]
        k1 = data["ok1"]["val"]
        k2 = data["ok2"]["val"]
        k3 = data["ok3"]["val"]
        p1 = data["op1"]["val"]
        p2 = data["op2"]["val"]

        camera2_intrinsic = np.array([[fx2, 0, cx2],
                                    [0, fy2, fy2],
                                    [0, 0, 1]])

        # Camera 2 Distortion Parameters
        camera2_distortion = np.array([k1, k2, p1, p2, k3])
        camera_matrix = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]], dtype=np.float32)
        # matched_labels, camera1_2D, camera2_2D, camera_2D_remainder, camera_1_3D = self.get_matching_coordinates(self.widgets['Camera 11 RGB Left'].ret_sorted_clusters(), self.widgets['Camera 11 RGB Right'].ret_sorted_clusters(), self.cam_11_L_3D)

        stuff1 = self.widgets['Camera 11 RGB Left'].ret_sorted_clusters()
        sorted_cluster_2D = np.array(self.process_data(stuff1), dtype=np.float32)
        self.cam_11_L_3D = self.widgets['Camera 11 RGB Left'].ret_3D_coords()
        sorted_cluster_3D = np.array(self.process_data(self.cam_11_L_3D), dtype=np.float32)
        _, rotation_vec, translation_vec, _ = cv2.solvePnPRansac(sorted_cluster_3D,
                                                            sorted_cluster_2D,
                                                            camera_matrix,
                                                            None)


        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        # Define the arrow length for visualization
        arrow_length = 10
        # Calculate the endpoint of the arrow based on camera position and direction
        arrow_end = rotation_matrix.T @ np.array([0, 0, arrow_length])
        
        origin = translation_vec.flatten()
        arrow_params = [origin, arrow_end]
        self.overall_camera_pose.append(arrow_params)

        # Plot the 3D points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for points in self.overall_3D_points:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue')

        for arrows in self.overall_camera_pose:
            ax.quiver(*arrows[0], *arrows[1], color='blue')

        # Set axes labels and display the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    
    def calibrate_72(self):
        # # Camera 11_L Intrinsic Parameters
        # camera_11_L_json = self.widgets['Camera 11 RGB Left'].jsonName
        # json_path_L = os.path.join("data", "camera parameters", camera_11_L_json)

        # with open(json_path_L, 'r') as f:
        #     data = json.load(f)

        # # Camera 11_L intrinsic parameters
        # fx1 = data["f"]["val"]
        # fy1 = data["f"]["val"]
        # cx1 = data["ocx"]["val"]
        # cy1 = data["ocy"]["val"]
        # k1 = data["ok1"]["val"]
        # k2 = data["ok2"]["val"]
        # k3 = data["ok3"]["val"]
        # p1 = data["op1"]["val"]
        # p2 = data["op2"]["val"]

        # # Camera 11_L Intrinsic Parameters
        # camera1_intrinsic = np.array([[fx1, 0, cx1],
        #                             [0, fy1, cy1],
        #                             [0, 0, 1]])

        # # Camera 11_L Distortion Parameters
        # camera1_distortion = np.array([k1, k2, p1, p2, k3])

        # Camera 72 Intrinsic Parameters
        camera_72_json = self.widgets['Camera 72 RGB'].jsonName
        json_path_R = os.path.join("data", "camera parameters", camera_72_json)

        with open(json_path_R, 'r') as f:
            data = json.load(f)

        # Camera 72 intrinsic parameters
        fx2 = data["f"]["val"]
        fy2 = data["f"]["val"]
        cx2 = data["ocx"]["val"]
        cy2 = data["ocy"]["val"]
        k1 = data["ok1"]["val"]
        k2 = data["ok2"]["val"]
        k3 = data["ok3"]["val"]
        p1 = data["op1"]["val"]
        p2 = data["op2"]["val"]

        camera2_intrinsic = np.array([[fx2, 0, cx2],
                                    [0, fy2, cy2],
                                    [0, 0, 1]])

        # Camera 2 Distortion Parameters
        camera2_distortion = np.array([k1, k2, p1, p2, k3])
        camera_matrix = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]], dtype=np.float32)

        matched_labels, camera1_2D, camera2_2D, camera2_2D_remainder,camera_1_3D = self.get_matching_coordinates(self.widgets['Camera 11 RGB Right'].ret_sorted_clusters(), self.widgets['Camera 72 RGB'].ret_sorted_clusters(), self.cam_11_L_3D)

        # 2D Coordinates of Common Labeled Targets
        common_targets_2d = np.array(camera2_2D, dtype=np.float32)

        # 3D Coordinates of Common Labeled Targets
        common_targets_3d = np.array(camera_1_3D, dtype=np.float32)
        print(common_targets_2d)
        print(common_targets_3d)

        # # Apply Camera Calibration to Undistort 2D Coordinates
        # common_targets_2d_undistorted_1 = cv2.undistortPoints(common_targets_2d.reshape(-1, 1, 2),
        #                                                     camera1_intrinsic,
        #                                                     camera1_distortion)
        # common_targets_2d_undistorted_2 = cv2.undistortPoints(common_targets_2d.reshape(-1, 1, 2),
        #                                                     camera2_intrinsic,
        #                                                     camera2_distortion)

        # # Reshape the Undistorted 2D Coordinates
        # common_targets_2d_undistorted_1 = common_targets_2d_undistorted_1.squeeze()
        # common_targets_2d_undistorted_2 = common_targets_2d_undistorted_2.squeeze()

        # Estimate Pose of Camera 2 Relative to Camera 1

        _, rotation_vec, translation_vec,_ = cv2.solvePnPRansac(common_targets_3d,
                                                            common_targets_2d,
                                                            camera_matrix,
                                                            None)


        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        # Define the arrow length for visualization
        arrow_length = 10
        # Calculate the endpoint of the arrow based on camera position and direction
        arrow_end = rotation_matrix.T @ np.array([0, 0, arrow_length])
        origin = translation_vec.flatten()
        arrow_params = [origin, arrow_end]
        self.overall_camera_pose.append(arrow_params)



    #    # Apply Camera Calibration to Undistort 2D Coordinates in Camera 2
    #     targets_2d_undistorted_camera2 = cv2.undistortPoints(camera2_2D_remainder.reshape(-1, 1, 2),
    #                                                         camera2_intrinsic,
    #                                                         camera2_distortion)
    #     targets_2d_undistorted_camera2 = targets_2d_undistorted_camera2.squeeze()

    #     # Transform 2D Coordinates from Camera 2 to Camera 1 Coordinates
    #     rotation_matrix = cv2.Rodrigues(rotation_vec)[0]
    #     translation_vec = translation_vec.flatten()
    #     homogeneous_coords = np.hstack((targets_2d_undistorted_camera2, np.ones((targets_2d_undistorted_camera2.shape[0], 1))))
    #     transformed_homogeneous_coords = rotation_matrix @ homogeneous_coords.T + np.expand_dims(translation_vec, axis=1)
    #     transformed_homogeneous_coords /= transformed_homogeneous_coords[2, :]
    #     targets_2d_transformed_camera1 = transformed_homogeneous_coords[:2, :].T

    #     # Estimate 3D Coordinates of Targets in Camera 1 Coordinates
    #     targets_3d_camera1 = np.zeros((targets_2d_transformed_camera1.shape[0], 3))
    #     for i in range(targets_2d_transformed_camera1.shape[0]):
    #         x, y = targets_2d_transformed_camera1[i]
    #         A = np.dot(rotation_matrix, np.linalg.inv(camera1_intrinsic))
    #         B = translation_vec - np.dot(A, np.array([x, y, 1]))
    #         lambda_ = np.linalg.norm(np.dot(np.linalg.inv(A), B))
    #         targets_3d_camera1[i] = np.dot(np.linalg.inv(A), B) / lambda_


    #     self.overall_3D_points.append(targets_3d_camera1)

        # Plot the 3D points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #ax.scatter(targets_3d_camera1[:, 0], targets_3d_camera1[:, 1], targets_3d_camera1[:, 2], c='blue')

        # print('3D coords:')
        # print(targets_3d_camera1)
        for points in self.overall_3D_points:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue')

        for arrows in self.overall_camera_pose:
            ax.quiver(*arrows[0], *arrows[1], color='red')

        # Set axes labels and display the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()