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
        self.overall_rvec_tvec = []

        # Camera Calibration buttons
        self.calibrate_11_L_btn = QPushButton("Calibrate 11 Left", self)
        self.calibrate_11_L_btn.clicked.connect(self.calibrate_11_L)
        self.original_image_label = QLabel(self)

        self.calibrate_11_R_btn = QPushButton("Calibrate 11 Right", self)
        self.calibrate_11_R_btn.clicked.connect(self.calibrate_11_R)

        self.calibrate_all_btn = QPushButton("Calibrate All", self)
        self.calibrate_all_btn.clicked.connect(self.calibrate_all)

        # Create grid layout
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)

        # Add widgets to grid layout
        self.grid_layout.addWidget(self.calibrate_11_L_btn, 0, 0, 1, 4)
        self.grid_layout.addWidget(self.calibrate_11_R_btn, 1, 0, 1, 4)
        self.grid_layout.addWidget(self.calibrate_all_btn, 2, 0, 1, 4)
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
        return np.array(coordinates, dtype=np.float32)
    
    def calibrate_11_L(self):
        stuff1 = self.widgets['Camera 11 RGB Left'].ret_sorted_clusters()
        sorted_cluster_2D = np.array(self.process_data(stuff1), dtype=np.float32)

        cam_11_L_3D = self.widgets['Camera 11 RGB Left'].ret_3D_coords()
        sorted_cluster_3D = np.array(self.process_data(cam_11_L_3D), dtype=np.float32)

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

        arrow_params = [np.array(origin, dtype=np.float32), np.array(arrow_end, dtype=np.float32)]
        # self.overall_camera_pose.append(arrow_params)

        # Set axes labels and display the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def get_matching_coordinates(self, coords_2D1, coords_2D2, coords_3D1, coords_3D2):
        # Flatten the data structures into lists of labels
        labels_1 = [item['label'] for sublist in coords_2D1 for item in sublist if 'label' in item]
        labels_2 = [item['label'] for sublist in coords_2D2 for item in sublist if 'label' in item]

        # Find matching labels
        matching_labels = set(labels_1).intersection(labels_2)

        coordinates_2D_1 = []
        coordinates_2D_2 = []
        coordinates_2D_remainder = []
        coordinates_3D_1 = []
        coordinates_3D_2 = []

        # Get points which are found in both images
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

            for targets in coords_3D1:
                if targets[0]['label'] == label:
                    for target in targets[1:]:
                        if 'center' in target:
                            coordinates_3D_1.append(target['center'])

            for targets in coords_3D2:
                if targets[0]['label'] == label:
                    for target in targets[1:]:
                        if 'center' in target:
                            coordinates_3D_2.append(target['center'])

        # Get the leftover points from image 2
        for targets in coords_3D2:
            isRemainder = True
            for label in matching_labels:
                if targets[0]['label'] == label:
                    isRemainder = False
            
            if isRemainder:
                for target in targets[1:]:
                    coordinates_2D_remainder.append(target['center'])

        return matching_labels, np.array(coordinates_2D_1, dtype=np.float32), np.array(coordinates_2D_2, dtype=np.float32), np.array(coordinates_2D_remainder, dtype=np.float32), np.array(coordinates_3D_1, dtype=np.float32), np.array(coordinates_3D_2, dtype=np.float32)

    def calibrate_11_R(self):

        # Camera 11_L Intrinsic Parameters
        camera_11_L_json = self.widgets['Camera 11 RGB Left'].jsonName
        json_path_L = os.path.join("data", "camera parameters", camera_11_L_json)

        with open(json_path_L, 'r') as f:
            data1 = json.load(f)

        # Camera 11_L intrinsic parameters
        fx1 = data1["f"]["val"]
        fy1 = data1["f"]["val"]
        cx1 = data1["ocx"]["val"]
        cy1 = data1["ocy"]["val"]
        k1 = data1["ok1"]["val"]
        k2 = data1["ok2"]["val"]
        k3 = data1["ok3"]["val"]
        p1 = data1["op1"]["val"]
        p2 = data1["op2"]["val"]

        # Camera 11_L Intrinsic Parameters
        camera1_intrinsic = np.array([[fx1, 0, cx1],
                                    [0, fy1, cy1],
                                    [0, 0, 1]], dtype=np.float32)

        # Camera 11_L Distortion Parameters
        camera1_distortion = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

        # Camera 11_R Intrinsic Parameters
        camera_11_R_json = self.widgets['Camera 11 RGB Right'].jsonName
        json_path_R = os.path.join("data", "camera parameters", camera_11_R_json)

        with open(json_path_R, 'r') as f:
            data2 = json.load(f)

        # Camera 11_R intrinsic parameters
        fx2 = data2["f"]["val"]
        fy2 = data2["f"]["val"]
        cx2 = data2["ocx"]["val"]
        cy2 = data2["ocy"]["val"]
        k1 = data2["ok1"]["val"]
        k2 = data2["ok2"]["val"]
        k3 = data2["ok3"]["val"]
        p1 = data2["op1"]["val"]
        p2 = data2["op2"]["val"]

        camera2_intrinsic = np.array([[fx2, 0, cx2],
                                    [0, fy2, cy2],
                                    [0, 0, 1]], dtype=np.float32)

        # Camera 2 Distortion Parameters
        camera2_distortion = np.array([k1, k2, p1, p2, k3])

        matched_labels, camera1_2D, camera2_2D, camera_2D_remainder, camera_1_3D, camera_2_3D = self.get_matching_coordinates(self.widgets['Camera 11 RGB Left'].ret_sorted_clusters(), 
                                                                                                                 self.widgets['Camera 11 RGB Right'].ret_sorted_clusters(), 
                                                                                                                 self.widgets['Camera 11 RGB Left'].ret_3D_coords(),
                                                                                                                 self.widgets['Camera 11 RGB Right'].ret_3D_coords())


        # Camera calibration
        width1, height1 = self.widgets['Camera 11 RGB Left'].ret_width_height()
        ret, camera1_matrix, camera1_dist_coeffs, _, _ = cv2.calibrateCamera(np.array([camera_1_3D], dtype=np.float32),
                                                                            np.array([camera1_2D], dtype=np.float32),
                                                                            (width1, height1),
                                                                            camera1_intrinsic,
                                                                            camera1_distortion,
                                                                            flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        
        width2, height2 = self.widgets['Camera 11 RGB Right'].ret_width_height()
        ret, camera2_matrix, camera2_dist_coeffs, _, _ = cv2.calibrateCamera(np.array([camera_2_3D], dtype=np.float32),
                                                                            np.array([camera2_2D], dtype=np.float32),
                                                                            (width2, height2),
                                                                            camera2_intrinsic,
                                                                            camera2_distortion,
                                                                            flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        

        # PnP for Camera 1
        _, camera1_rotation_vector, camera1_translation_vector = cv2.solvePnP(np.array(camera_1_3D, dtype=np.float32),
                                                                                np.array(camera1_2D, dtype=np.float32),
                                                                                np.array(camera1_matrix, dtype=np.float32),
                                                                                np.array(camera1_dist_coeffs, dtype=np.float32))

        # PnP for Camera 2
        _, camera11_R_rotation_vector, camera11_R_translation_vector = cv2.solvePnP(np.array(camera_1_3D, dtype=np.float32),
                                                                                np.array(camera2_2D, dtype=np.float32),
                                                                                np.array(camera2_matrix, dtype=np.float32),
                                                                                np.array(camera2_dist_coeffs, dtype=np.float32))
        
        vectors = [np.array(camera11_R_rotation_vector, dtype=np.float32), np.array(camera11_R_translation_vector, dtype=np.float32)]
        self.overall_rvec_tvec.append(vectors)

        # Relative pose estimation
        relative_rotation_vector, relative_translation_vector, _, _, _, _, _, _, _, _ = cv2.composeRT(np.array(camera1_rotation_vector, dtype=np.float32),
                                                                            np.array(camera1_translation_vector, dtype=np.float32),
                                                                            np.array(camera11_R_rotation_vector, dtype=np.float32),
                                                                            np.array(camera11_R_translation_vector, dtype=np.float32))
   

        # Relative pose transformation
        relative_rotation_matrix, _ = cv2.Rodrigues(np.array(relative_rotation_vector, dtype=np.float32))
        relative_translation_vector = np.array(relative_translation_vector.reshape(3), dtype=np.float32)
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = relative_rotation_matrix
        relative_pose[:3, 3] = relative_translation_vector

        # Define the arrow length for visualization
        arrow_length = 10

        # Convert rotation vector to rotation matrix
        rotation_matrix_1, _ = cv2.Rodrigues(np.array(camera1_rotation_vector, dtype=np.float32))
        # Calculate the endpoint of the arrow based on camera position and direction
        arrow_end_1 = np.array(rotation_matrix_1.T @ np.array([0, 0, arrow_length]), dtype=np.float32)

        # Convert rotation vector to rotation matrix
        rotation_matrix_2, _ = cv2.Rodrigues(camera11_R_rotation_vector)
        # Calculate the endpoint of the arrow based on camera position and direction
        arrow_end_2 = np.array(rotation_matrix_2.T @ np.array([0, 0, arrow_length]), dtype=np.float32)

        # Plot the 3D points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the camera position as a red arrow
        origin_1 = np.array(camera1_translation_vector.flatten(), dtype=np.float32)
        ax.quiver(*origin_1, *arrow_end_1, color='red')

        # Plot the camera position as a red arrow
        origin_2 = np.array(camera11_R_translation_vector.flatten(), dtype=np.float32)
        ax.quiver(*origin_2, *arrow_end_2, color='red')

        arrow_params = [origin_1, arrow_end_1]
        self.overall_camera_pose.append(arrow_params)
        arrow_params = [origin_2, arrow_end_2]
        self.overall_camera_pose.append(arrow_params)

        # Plot the 3D points
        for points in self.overall_3D_points:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue')

        # Set axes labels and display the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


    def calibrate_all(self):
        
        # The order of cameras that we are going to calibrate. 
        cameras = ['Camera 11 RGB Right', 'Camera 71 RGB', 'Camera 74 RGB', 'Camera 73 RGB', 'Camera 72 RGB']

        # Loop starts at 1 because we already calibrated camera 11
        for i, camera in enumerate(cameras[1:], start=1):
            # Intrinsic parameters of camera we are currently calibrating
            camera_current_json = self.widgets[camera].jsonName
            json_path_current = os.path.join("data", "camera parameters", camera_current_json)

            with open(json_path_current, 'r') as f:
                data = json.load(f)

            # Intrinsic parameters
            fx = data["f"]["val"]
            fy = data["f"]["val"]
            cx= data["ocx"]["val"]
            cy = data["ocy"]["val"]
            k1 = data["ok1"]["val"]
            k2 = data["ok2"]["val"]
            k3 = data["ok3"]["val"]
            p1 = data["op1"]["val"]
            p2 = data["op2"]["val"]

            camera_current__intrinsic = np.array([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]], dtype=np.float32)

            # Camera 2 Distortion Parameters
            camera_current_distortion = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

            # Get the point data
            matched_labels, camera1_2D, camera2_2D, camera2_3D_remainder, camera_1_3D, camera_2_3D = self.get_matching_coordinates(self.widgets[cameras[i-1]].ret_sorted_clusters(), 
                                                                                                                    self.widgets[camera].ret_sorted_clusters(), 
                                                                                                                    self.widgets[cameras[i-1]].ret_3D_coords(),
                                                                                                                    self.widgets[camera].ret_3D_coords())

            # Get image size
            width2, height2 = self.widgets[camera].ret_width_height()
            # Get the camera calibration matrices
            ret, current_camera_matrix, current_camera_dist_coeffs, _, _ = cv2.calibrateCamera(np.array([camera_2_3D], dtype=np.float32),
                                                                                np.array([camera2_2D], dtype=np.float32),
                                                                                (width2, height2),
                                                                                np.array(camera_current__intrinsic, dtype=np.float32),
                                                                                np.array(camera_current_distortion, dtype=np.float32),
                                                                                flags=cv2.CALIB_USE_INTRINSIC_GUESS)
            

            # PnP for Current Camera
            _, current_camera_rotation_vector, current_camera_translation_vector = cv2.solvePnP(np.array(camera_1_3D, dtype=np.float32),
                                                                                    np.array(camera2_2D, dtype=np.float32),
                                                                                    np.array(current_camera_matrix, dtype=np.float32),
                                                                                    np.array(current_camera_dist_coeffs, dtype=np.float32))
            
            vectors = [np.array(current_camera_rotation_vector, dtype=np.float32), np.array(current_camera_translation_vector, dtype=np.float32)]
            self.overall_rvec_tvec.append(vectors)

            # Relative pose estimation
            relative_rotation_vector, relative_translation_vector, _, _, _, _, _, _, _, _ = cv2.composeRT(np.array(self.overall_rvec_tvec[i-1][0], dtype=np.float32),
                                                                                np.array(self.overall_rvec_tvec[i-1][1], dtype=np.float32),
                                                                                np.array(current_camera_rotation_vector, dtype=np.float32),
                                                                                np.array(current_camera_translation_vector, dtype=np.float32))
            
            relative_rotation_matrix, _ = cv2.Rodrigues(np.array(relative_rotation_vector, dtype=np.float32))
   

            # Relative pose transformation
            relative_rotation_matrix, _ = cv2.Rodrigues(np.array(relative_rotation_vector, dtype=np.float32))
            relative_translation_vector = np.array(relative_translation_vector.reshape(3), dtype=np.float32)
            relative_pose = np.eye(4)
            relative_pose[:3, :3] = np.array(relative_rotation_matrix, dtype=np.float32)
            relative_pose[:3, 3] = np.array(relative_translation_vector, dtype=np.float32)

            # Transforming 3D coordinates from Camera 2 to Camera 1's coordinate system
            camera2_3d_points_transformed = []

            for point in camera2_3D_remainder:
                # Convert 3D point to homogeneous coordinates
                point_homogeneous = np.hstack((point, 1)).reshape(4, 1)
                
                # Apply relative pose transformation
                point_transformed = np.dot(relative_pose, point_homogeneous)
                
                # Extract transformed 3D coordinates
                point_transformed_3d = point_transformed[:3, 0]
                
                camera2_3d_points_transformed.append(point_transformed_3d.tolist())

            camera2_3d_points_transformed = np.array(camera2_3d_points_transformed, dtype=np.float32)


            print('3D coordinates of: ' + camera)
            print(camera2_3d_points_transformed)

            self.overall_3D_points.append(camera2_3d_points_transformed)

            # Convert rotation vector to rotation matrix
            rotation_matrix_2, _ = cv2.Rodrigues(np.array(current_camera_rotation_vector, dtype=np.float32))
            # Calculate the endpoint of the arrow based on camera position and direction
            # Define the arrow length for visualization
            arrow_length = 10
            arrow_end_2 = rotation_matrix_2.T @ np.array([0, 0, arrow_length])
            origin_2 = current_camera_translation_vector.flatten()

            arrow_params = [np.array(origin_2, dtype=np.float32), np.array(arrow_end_2, dtype=np.float32)]
            self.overall_camera_pose.append(arrow_params)



        # Plot the 3D points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        # Plot the 3D points
        for points in self.overall_3D_points:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue')

        for arrow in self.overall_camera_pose:
            ax.quiver(*arrow[0], *arrow[1], color='red')

        # Set axes labels and display the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()