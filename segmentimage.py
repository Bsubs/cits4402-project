# CITS4402 Project - Joo Kai Tay (22489437), Yusi Zhang (23458522), Runtian Liang (23485011)

from pickle import TRUE
from PyQt5 import QtWidgets
import json
import sys
import os
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


class MaskImage (QtWidgets.QWidget):
      
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Read in tuned hyperparameters from JSON file
        self.tminColor = param_dict.get('tminColor')
        self.tdiffColor = param_dict.get('tdiffColor')
        self.tminArea = param_dict.get('tminArea')     
        self.tmaxArea = param_dict.get('tmaxArea')
        self.tDistance = param_dict.get('tDistance')
        self.taxisRatio = param_dict.get('taxisRatio')
        self.tellipse = param_dict.get('tellipse')
        self.jsonName = param_dict.get('jsonName')
        self.imgName = param_dict.get('imgName')
        self.distance_threshold=1

        # Creates the widgets for colour thresholding
        # Create load image widget & label
        self.original_image_label = QLabel(self)
        self.run_all_btn = QPushButton("Perform Image Segmentation", self)
        self.run_all_btn.clicked.connect(self.run_all)
        # Creates the widgets for tminColor and tdiffColor (label, slider, value label)
        self.tminColor_slider = QSlider(Qt.Horizontal)
        self.tminColor_slider.valueChanged.connect(self.update_tminColor)
        self.tminColor_value_label = QLabel(str(self.tminColor))
        self.tminColor_slider.setMaximum(200)
        self.tdiffColor_slider = QSlider(Qt.Horizontal)
        tminColor_label = QLabel("tminColor: ")
        self.tdiffColor_slider.valueChanged.connect(self.update_tdiffColor)
        self.tdiffColor_slider.setMaximum(200)
        tdiffColor_label = QLabel("tdiffColor: ")
        self.tdiffColor_value_label = QLabel(str(self.tdiffColor))


        # Creates the widgets for connected component analysis
        tminArea_label = QLabel("tminArea: ")
        self.tminArea_slider = QSlider(Qt.Horizontal)
        self.tminArea_slider.setMaximum(200)
        tmaxArea_label = QLabel("tmaxArea: ")
        self.tminArea_slider.valueChanged.connect(self.update_tminArea)
        self.tminArea_value_label = QLabel(str(self.tminArea))
        self.tmaxArea_slider = QSlider(Qt.Horizontal)
        self.tmaxArea_slider.setMaximum(300)
        self.tmaxArea_slider.valueChanged.connect(self.update_tmaxArea)
        self.tmaxArea_value_label = QLabel(str(self.tmaxArea))
        tDistance_label = QLabel("tDistance: ")
        self.tDistance_slider = QSlider(Qt.Horizontal)
        self.tDistance_slider.setMaximum(200)
        self.tDistance_slider.valueChanged.connect(self.update_tDistance)
        self.tDistance_value_label = QLabel(str(self.tDistance))
        self.taxisRatio_slider = QDoubleSpinBox()
        self.taxisRatio_slider.setRange(0, 10)
        self.taxisRatio_slider.setSingleStep(0.1)
        self.taxisRatio_slider.valueChanged.connect(self.update_taxisRatio)
        taxisRatio_label = QLabel("taxisRatio: ")
        self.taxisRatio_value_label = QLabel(str(self.taxisRatio))


        # Creates the widgets for hexagon detection
        self.tellipse_slider = QDoubleSpinBox()
        self.tellipse_slider.setRange(0, 150)
        self.tellipse_slider.setSingleStep(0.1)
        self.hexagon_label = QLabel(self)
        self.tellipse_slider.valueChanged.connect(self.update_tellipse)
        tellipse_label = QLabel("tellipse: ")
        self.tellipse_value_label = QLabel(str(self.tellipse))


        # Creates the widgets for labeling and alignment
        self.lb_label = QLabel(self)
        self.string_label = QLabel(self)
        self.get3D_label=QLabel(self)
        # Create grid layout
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)

        # Add widgets to grid layout
        self.grid_layout.addWidget(self.run_all_btn, 1, 0, 1, 4)
        self.grid_layout.addWidget(tminColor_label, 2, 0)
        self.grid_layout.addWidget(self.tminColor_slider, 2, 1, 1, 3)
        self.grid_layout.addWidget(self.tminColor_value_label, 2, 4)
        self.grid_layout.addWidget(tdiffColor_label, 3, 0)
        self.grid_layout.addWidget(self.tdiffColor_slider, 3, 1, 1, 3)
        self.grid_layout.addWidget(self.tdiffColor_value_label, 3, 4)
        self.grid_layout.addWidget(tminArea_label, 5, 0)
        self.grid_layout.addWidget(self.tminArea_slider, 5, 1, 1, 3)
        self.grid_layout.addWidget(self.tminArea_value_label, 5, 4)
        self.grid_layout.addWidget(tmaxArea_label, 6, 0)
        self.grid_layout.addWidget(self.tmaxArea_slider, 6, 1, 1, 3)
        self.grid_layout.addWidget(self.tmaxArea_value_label, 6, 4)
        self.grid_layout.addWidget(tDistance_label, 7, 0)
        self.grid_layout.addWidget(self.tDistance_slider, 7, 1, 1, 3)
        self.grid_layout.addWidget(self.tDistance_value_label, 7, 4)
        self.grid_layout.addWidget(taxisRatio_label, 8, 0)
        self.grid_layout.addWidget(self.taxisRatio_slider, 8, 1, 1, 3)
        self.grid_layout.addWidget(self.taxisRatio_value_label, 8, 4)
        self.grid_layout.addWidget(tellipse_label, 9, 0)
        self.grid_layout.addWidget(self.tellipse_slider, 9, 1, 1, 3)
        self.grid_layout.addWidget(self.tellipse_value_label, 9, 4)
        self.grid_layout.addWidget(self.original_image_label, 19, 1)
        self.grid_layout.addWidget(self.hexagon_label, 19, 2)
        self.grid_layout.addWidget(self.lb_label, 20, 1)
        self.grid_layout.addWidget(self.string_label, 20, 2)
        self.grid_layout.addWidget(self.get3D_label, 21, 1)
        self.grid_layout.setColumnStretch(2, 1)  

    # Loads the image that we are going to segment and label
    def load_image(self):
        # auto loads the image based on the name provided
        file_name = os.path.join("data", "images", self.imgName)
        self.original_image = cv2.imread(file_name)

        # display the original image
        self.display_image(self.original_image, self.original_image_label)
        # Generate initial segmentation mask
        self.generate_mask()

    # this function displays the specified image in the specified label
    def display_image(self, img, label):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    # Section 1.2 rough target detection, segments image based on tmincolor and tdiffcolor
    def generate_mask(self):

        # Compute minimum and maximum intensities for each pixel
        min_intensities = np.min(self.original_image, axis=2)
        max_intensities = np.max(self.original_image, axis=2)

        # Create mask based on threshold values
        mask = (
            (min_intensities < self.tminColor)
            | (max_intensities - min_intensities > self.tdiffColor)
        ).astype(np.uint8) * 255

        # Apply mask to original original_image
        self.masked_image = cv2.bitwise_and(
            self.original_image, self.original_image, mask=mask
        )

    # Section 1.2 rough target detection, performs connected component analysis on clusters
    def filter_clusters(self):
        # Convert image to grayscale
        gray_image = cv2.cvtColor(self.masked_image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to obtain binary image
        _, self.binary_image = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY)

        # Perform connected component analysis on binary image using skimage.measure.label()
        self.labeled_image = label(self.binary_image, connectivity=2)
        self.props = regionprops(self.labeled_image)

        # Filter out small and large clusters based on area thresholds
        cluster_mask = np.zeros_like(self.binary_image)
        centroids = []
        valid_props = []

        for prop in self.props:
            if self.tminArea <= prop.area <= self.tmaxArea:
                # Compute minor and major axis lengths and their ratio
                sigma_min, sigma_max = np.sqrt(prop.inertia_tensor_eigvals)
                ratio = sigma_min / sigma_max

                # Filter out clusters with minor-to-major axis ratio below threshold
                if ratio < self.taxisRatio:
                    centroids.append(prop.centroid)
                    valid_props.append(prop)

        # Compute pairwise distances between cluster centroids
        centroids = np.array(centroids)
        distances = distance_matrix(centroids, centroids)

        # Filter out clusters with distance to the nearest cluster greater than the threshold
        for i, prop in enumerate(valid_props):
            min_distance = np.min(distances[i, [j for j in range(len(distances)) if i != j]])
            if min_distance <= self.tDistance:
                cluster_mask[self.labeled_image == prop.label] = 255

        # Apply mask to original image
        self.eigen_image = cv2.bitwise_and(
            self.masked_image, self.masked_image, mask=cluster_mask
        )

    # Section 1.2 rough target detection, fitting an ellipse on the closest 6 points
    def find_target_clusters(self):
        # Find the five nearest clusters for each cluster
        centroids = np.array([prop.centroid for prop in self.props])
        kdtree = KDTree(centroids)
        nearest_indices = kdtree.query(centroids, k=6)[1][:, 1:]
        self.nearest_five = nearest_indices

        # computes residual error of the ellipse fit
        def ellipse_residual_error(params, selected_centroids):
            # Helper function to compute residual error of ellipse fitting
            x0, y0, a, b, theta = params
            t = np.linspace(0, 2 * np.pi, 100)
            ellipse_points = np.column_stack((a * np.cos(t), b * np.sin(t)))
            # rotates ellipse points by angle theta
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]]
            )
            # translates them to center coordinates
            ellipse_points_rotated = np.dot(ellipse_points, rotation_matrix) + np.array(
                [x0, y0]
            )
            # distance matrix between rotated ellipse points and selected centroids
            dists = distance_matrix(selected_centroids, ellipse_points_rotated)
            # compute squared distances from each centroid to the closest ellipse point
            min_dists = np.min(dists, axis=1)
            return min_dists**2

        # Fit an ellipse on the six centroids
        target_clusters = []
        for i, nearest in enumerate(nearest_indices):
            selected_centroids = np.vstack((centroids[i], centroids[nearest]))
            x0, y0 = np.mean(selected_centroids, axis=0)
            a, b = (
                np.max(selected_centroids, axis=0) -
                np.min(selected_centroids, axis=0)
            ) / 2
            theta = 0
            initial_params = [x0, y0, a, b, theta]

            # Compute residual error
            result = least_squares(
                ellipse_residual_error, initial_params, args=(
                    selected_centroids,)
            )
            residual_error = np.sum(result.fun)

            # Check if the largest residual error is less than the threshold
            if residual_error < self.tellipse:
                target_clusters.append(i)

        # Apply a mask to the original image to display only the target clusters
        target_mask = np.zeros_like(self.binary_image)
        target_mask[
            np.isin(self.labeled_image, [
                    self.props[i].label for i in target_clusters])
        ] = 255
        target_image = cv2.bitwise_and(
            self.eigen_image, self.eigen_image, mask=target_mask
        )
        self.display_image(target_image, self.hexagon_label)
        self.image = target_image
        self.target_clusters = target_clusters

    # Section 1.3 detected target analysis
    def find_target_lable(self):
        def color_temperature_compensation(image, gain_factors):
            return np.clip(image * gain_factors, 0, 255).astype(np.uint8)

        # Define the reference colors: red, green, and blue
        reference_colors = np.array([
            [0, 0, 255],    # blue
            [0, 255, 0],    # Green
            [255, 0, 0],    # red
        ], dtype=np.float32)

        # Define gain factors for color temperature compensation
        gain_factors = np.array([1.2, 1.5, 1.0])

        # Assuming self.image is a BGR image
        image = self.image.copy()
        height, width, _ = image.shape

        # Create an empty image with the same size as the input image
        output_image = np.zeros_like(image, dtype=np.float32)

        def get_max_color(color, distances, distance_threshold):
            max_index = np.argmax(color)
            if distances[max_index] > distance_threshold:
                max_color = np.zeros_like(color)
                max_color[max_index] = color[max_index]
                return max_color
            else:
                return None

        distance_threshold = 200

        # Iterate through each pixel in the input image
        for y in range(height):
            for x in range(width):
                pixel_color = image[y, x].astype(np.float32)

                # Check if the pixel color is not black
                if np.any(pixel_color != 0):
                    # Calculate the distances between the pixel color and the reference colors
                    distances = np.linalg.norm(
                        reference_colors - pixel_color, axis=1)

                    # Find the index of the closest reference color
                    closest_color_idx = np.argmin(distances)

                    # Choose the closest reference color
                    closest_color = reference_colors[closest_color_idx]

                    # Apply color temperature compensation to the closest reference color
                    compensated_color = color_temperature_compensation(
                        closest_color, gain_factors)

                    # Set the other color components to zero
                    compensated_color[np.argmin(compensated_color)] = 0
                    compensated_color[np.argsort(compensated_color)[1]] = 0

                    # Set a threshold for color assignment
                    color_threshold = 10

                    # Only assign the color if the largest color component is greater than the threshold
                    if np.max(compensated_color) > color_threshold:
                        # If multiple colors are present, choose the one with the largest component
                        if np.count_nonzero(compensated_color) > 1:
                            max_color = get_max_color(
                                compensated_color, distances, distance_threshold)
                            if max_color is not None:
                                output_image[y, x] = max_color
                            else:
                                output_image[y, x] = 0
                        else:
                            output_image[y, x] = compensated_color
                    else:
                        output_image[y, x] = 0
                else:
                    # Keep the pixel color in the output image as black
                    output_image[y, x] = pixel_color

        # Convert the output image back to the uint8 data type
        output_image = output_image.astype(np.uint8)

        def get_neighbors(y, x, height, width, distance=3):
            neighbors = []
            for i in range(-distance, distance + 1):
                for j in range(-distance, distance + 1):
                    if i == 0 and j == 0:
                        continue

                    neighbor_y = y + i
                    neighbor_x = x + j

                    if 0 <= neighbor_y < height and 0 <= neighbor_x < width:
                        neighbors.append((neighbor_y, neighbor_x))

            return neighbors

        def set_to_black(y, x, image, height, width, distance=3):
            if np.all(image[y, x] == [0, 0, 0]) or np.all(image[y, x] == [0, 255, 0]):
                return

            neighbors = get_neighbors(y, x, height, width, distance)
            green_neighbors = [(neighbor_y, neighbor_x) for neighbor_y, neighbor_x in neighbors
                               if np.all(image[neighbor_y, neighbor_x] == [0, 255, 0])]

            if not green_neighbors:
                return

            image[y, x] = [0, 0, 0]

        def process_image(image):
            height, width, _ = image.shape
            output_image = image.copy()

            for y in range(height):
                for x in range(width):
                    set_to_black(y, x, output_image, height, width)


            return output_image

        processed_image = process_image(output_image)
        
        self.processed_image = processed_image

        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)

        # Define color range for blue in HSV
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])

        # Define color ranges for red and green in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([20, 255, 255])

        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])

        # Threshold the image to keep only blue colors
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Create a copy of the original image to draw markers on
        marked_image = self.original_image.copy()

        # Find contours in the mask
        contours1, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours3, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_contour_area = 1  # Set your minimum contour area threshold

        # Create new lists to store the contours that are large enough
        contours1 = [cnt for cnt in contours1 if cv2.contourArea(cnt) > min_contour_area]
        contours2 = [cnt for cnt in contours2 if cv2.contourArea(cnt) > min_contour_area]
        contours3 = [cnt for cnt in contours3 if cv2.contourArea(cnt) > min_contour_area]
        # Convert contours to center points
        def get_centroids(contours):
            centroids = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                centroids.append([cX, cY])
            return np.array(centroids)

        # get the center point of the color
        blue_centroids = get_centroids(contours1)
        red_centroids = get_centroids(contours2)
        green_centroids = get_centroids(contours3)
        self.blue_centroids = blue_centroids
        self.red_centroids = red_centroids
        self.green_centroids = green_centroids
        # Merge the center points of all colors
        all_centroids = np.vstack(
            (blue_centroids, red_centroids, green_centroids))
        cluster_size = 6
        # Group the color points into clusters according to every six groups
        clusters = [all_centroids[i:i + cluster_size]
                    for i in range(0, len(all_centroids), cluster_size)]
        # If the number of color points in the last cluster is less than 6, remove this cluster
        if len(clusters[-1]) < cluster_size:
            clusters.pop()

        # Calculate the center point of each rectangle
        def get_rectangles(contours):
            rectangles = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                centerX = x + w / 2
                centerY = y + h / 2
                center = (int(centerX), int(centerY))

                rectangles.append(
                    {'center': center, 'x': x, 'y': y, 'w': w, 'h': h})

            return rectangles

        # Loop through the contours and draw a rectangle around each pixel and display the label
        for cnt in contours1:
            # Calculate the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw a blue rectangle around the blue pixel
            
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            


        for cnt in contours2:
            # Calculate the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw a red rectangle around the red pixel
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 0, 255), 2)


        for cnt in contours3:
            # Calculate the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw a green rectangle around the green pixel
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # Calculate the center point for each color of the contour and add to the list of rectangles
        blue_rects = get_rectangles(contours1)
        red_rects = get_rectangles(contours2)
        green_rects = get_rectangles(contours3)


        # Put the center points of the rectangle boxes of all colors into a list
        all_rects = blue_rects + red_rects + green_rects

        # Sort from left to right by the x-coordinate of the center point
        sorted_rects = sorted(all_rects, key=lambda rect: rect['center'][0])
        # Every six points are classified into a cluster
        clusters = []
        cluster_size = 6

        for i in range(0, len(sorted_rects), cluster_size):
            cluster = sorted_rects[i:i + cluster_size]
            clusters.append(cluster)

        def clockwise_angle_and_distance(point, origin, reference):
            vector = [point[0] - origin[0], point[1] - origin[1]]
            reference_vector = [reference[0] -
                                origin[0], reference[1] - origin[1]]
            len_vector = np.sqrt(vector[0]**2 + vector[1]**2)
            len_ref_vector = np.sqrt(
                reference_vector[0]**2 + reference_vector[1]**2)

            if len_vector == 0 or len_ref_vector == 0:
                return (2 * np.pi, 0)

            cosine_angle = np.dot(vector, reference_vector) / \
                (len_vector * len_ref_vector)
            angle = np.arccos(np.clip(cosine_angle, -1, 1))
            cross_product = np.cross(vector, reference_vector)

            if cross_product < 0:
                angle = 2 * np.pi - angle

            return (angle, len_vector)

        def sort_points_clockwise(points, blue_point):
            reference_point = [blue_point[0], blue_point[1]-1]
            return sorted(points, key=lambda point: clockwise_angle_and_distance(point, blue_point, reference_point))[::-1]

        def sort_clusters_clockwise(clusters, blue_rects): 
            sorted_clusters = []

            for cluster in clusters:
                for rect in cluster:
                    if rect in blue_rects:
                        blue_rect = rect
                        break

                if blue_rect is None:
                    continue

                blue_center = blue_rect['center']  # Coordinates of blue center point
                other_points = [rect['center']
                                for rect in cluster if rect != blue_rect]  # Other center point coordinates
                sorted_points = sort_points_clockwise(
                    other_points, blue_center)
                sorted_cluster = [blue_rect] + \
                    [{'center': point} for point in sorted_points]
                sorted_clusters.append(sorted_cluster)

            return sorted_clusters

        sorted_clusters_clockwise = sort_clusters_clockwise(
            clusters, blue_rects)
        self.sorted_cluster=sorted_clusters_clockwise
        
        self.display_image(marked_image, self.lb_label)

        for cluster in sorted_clusters_clockwise:
            blue_rect = cluster[0]  # extract blue point
            clockwise_points = cluster[1:]  # Extract the 5 points behind the blue point
            # Initialize label to empty string
            label = ''

            # Traversing the 5 points behind the blue point
            for point in clockwise_points:
                # Checkpoint is in red_rects
                for red_rect in red_rects:
                    if point['center'] == red_rect['center']:
                        label += 'R'
                        break

                # Checkpoint is in green_rects
                for green_rect in green_rects:
                    if point['center'] == green_rect['center']:
                        label += 'G'
                        break


            # Assign the resulting labels to the blue points
            blue_rect['label'] = f"{label}_1"

            # Add a label next to the blue rectangle
            cv2.putText(marked_image, blue_rect['label'], (blue_rect['x'], blue_rect['y'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            # Add labels for other points sorted clockwise
            for i, point in enumerate(clockwise_points, start=2):
                point_rect = next(
                    rect for rect in all_rects if rect['center'] == point['center'])
                point_label = f"{label}_{i}"

                # Calculate text position
                if i == 4:
                    text_position = (
                        point_rect['x'], point_rect['y'] + point_rect['h'] + 20)
                elif i in [5, 6]:
                    text_position = (point_rect['x'] - 90, point_rect['y'])
                else:
                    text_position = (point_rect['x'], point_rect['y'] - 5)

                # Add a label next to the corresponding rectangle
                cv2.putText(marked_image, point_label, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                point_rect = next(
                    rect for rect in all_rects if rect['center'] == point['center'])
                point_label = f"{label}_{i}"

                # Add a label next to the corresponding rectangle
                cv2.putText(marked_image, point_label, (point_rect['x'], point_rect['y'] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        self.display_image(marked_image, self.string_label)

    # Section 1.4 subpixel target alignment
    def align_clusters(self):
        def get_cluster(centroid, image,color, distance_threshold, assigned_pixels):
            cluster = []
            cluster.append(centroid)
            height, width, _ = image.shape
            y, x = centroid
            centroid_color = image[y, x]
            # Iterate through the surrounding points within the distance threshold
            for dy in range(-distance_threshold, distance_threshold + 1):
                for dx in range(-distance_threshold, distance_threshold + 1):
                    if dy == 0 and dx == 0:
                        continue
                    new_y = y + dy
                    new_x = x + dx
                    # Check if the new coordinates are within the image bounds
                    if 0 <= new_y < height and 0 <= new_x < width:
                        pixel = image[new_y, new_x]
                        # Check if the pixel matches the target color and the centroid color,
                        # and if it has not been assigned to a cluster yet
                        if np.all(pixel == color) and np.all(pixel == centroid_color) and not assigned_pixels[new_y, new_x]:
                            distance = np.sqrt(dy**2 + dx**2)
                            # Include the point in the cluster if the distance is within the threshold.
                            if distance <= distance_threshold:
                                cluster.append((new_y, new_x))
                                assigned_pixels[new_y, new_x] = True

            return cluster

        def get_all_clusters(centroids, image,color,min_distance_threshold, max_distance_threshold):
            clusters = []
            height, width, _ = image.shape
            assigned_pixels = np.zeros((height, width), dtype=bool)
            # Iterate through the centroids and get clusters for each centroid.
            for centroid in centroids: 
                for distance_threshold in range(max_distance_threshold, min_distance_threshold - 1, -1):
                    y, x = centroid
                    swapped_centroid = (x, y)
                    cluster = get_cluster(
                        swapped_centroid, image, color, distance_threshold, assigned_pixels)
                    if cluster:
                        clusters.append(cluster)
                        break
            return clusters
        #use processed image to get clusters
        processed_image = self.processed_image
        min_distance_threshold = 1
        max_distance_threshold = 6
        blue_clusters = get_all_clusters(
            self.blue_centroids, processed_image, np.array([255, 0, 0]), min_distance_threshold, max_distance_threshold)
        red_clusters = get_all_clusters(
            self.red_centroids, processed_image, np.array([0, 0, 255]), min_distance_threshold, max_distance_threshold)
        green_clusters = get_all_clusters(
            self.green_centroids, processed_image, np.array([0, 255, 0]), min_distance_threshold, max_distance_threshold)
        #get all clusters
        all_clusters = blue_clusters + red_clusters + green_clusters
        self.clusters = all_clusters

        # use image to do alignment
        # for the reason that the processed image used a mask to make all the colors in the cluster the same
        image = self.image.copy()
        aligned_image = np.zeros_like(image)
        
        # Calculate the weight of a point based on its color distance from the average color
        def weight(image, point, extended_points, average_color):
            y, x = point
            pixel = image[y, x]
            if np.all(pixel == [0, 0, 0]):
                return 0
            distance = np.linalg.norm(pixel - average_color)
            max_distance = np.max(np.linalg.norm(
                extended_points - average_color, axis=1))
            if max_distance == 0:
                return 0
            return 1 - (distance / max_distance)
        
        #Calculate the weighted centroid of a cluster.
        def weighted_centroid(image, cluster):
            extended_points = np.zeros_like(image)
            extended_points[tuple(zip(*cluster))] = image[tuple(zip(*cluster))]
            centroid_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            centroid_mask[tuple(zip(*cluster))] = 1
            centroid_mask = cv2.dilate(
                centroid_mask, np.ones((9, 9), dtype=np.uint8))

            for i, point in enumerate(cluster):
                y, x = point
                if centroid_mask[y, x] == 0:
                    extended_points[tuple(point)] = 0

            extended_points = extended_points[np.nonzero(centroid_mask)]
            average_color = np.mean(extended_points, axis=0)
            weights = np.array(
                [weight(image, p, extended_points, average_color) for p in cluster])
            total_weight = np.sum(weights)
            if total_weight == 0:
                return [186.99513931, 470.34900327]
            else:
                return np.average(cluster, axis=0, weights=weights)

        aligned_centroids = []
        aligned_centroids_float = []
        # Iterate through all clusters
        for cluster in all_clusters:
            # Calculate the weighted centroid of the cluster
            centroid = weighted_centroid(image, cluster)
             # Compute the offset between the weighted centroid and original centroid
            offset = np.round(centroid - np.mean(cluster, axis=0)).astype(int)
            # Calculate the aligned centroid by adding the offset to the original centroid
            aligned_centroid = np.round(centroid + offset).astype(int)
            aligned_centroids.append(aligned_centroid)
            # Compute the offset between the weighted centroid and original centroid
            offset_float = centroid - np.mean(cluster, axis=0)
            # Calculate the aligned centroid by adding the offset to the original centroid
            aligned_centroid_float = centroid + offset_float
            aligned_centroids_float.append(aligned_centroid_float)
            # Move the points in the cluster based on the calculated offset
            for point in cluster:
                y, x = point
                new_y, new_x = np.round(point + offset).astype(int)
                new_x = np.clip(new_x, 0, image.shape[1] - 1)
                new_y = np.clip(new_y, 0, image.shape[0] - 1)
                # Assign the pixel value from the original image to the aligned_image
                if np.all(aligned_image[new_y, new_x] == [0, 0, 0]) or np.linalg.norm(image[y, x]) > np.linalg.norm(aligned_image[new_y, new_x]):
                    aligned_image[new_y, new_x] = image[y, x]

                    
        # Set the color of the centroids in the aligned_image
        for centroid in aligned_centroids:
            new_y, new_x = centroid
            new_x = np.clip(new_x, 0, image.shape[1] - 1)
            new_y = np.clip(new_y, 0, image.shape[0] - 1)
            centroid_color = image[new_y, new_x]
            aligned_image[new_y, new_x] = centroid_color
            #aligned_image[new_y, new_x] = [255, 255, 255]
        # Use the function to sort aligned_centroids
        aligned_centroids_float = [np.array([point[1], point[0]]) for point in aligned_centroids_float]
        aligned_centroids_sorted = sorted(aligned_centroids_float, key=lambda point: point[0])
        # calculate distance 
        def distance(point1, point2):
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        # update the center points in sorted cluster
        for cluster in self.sorted_cluster:
            for point in cluster:
                closest_point = min(aligned_centroids_sorted, key=lambda centroid: distance(centroid, point['center']))
                point['center'] = tuple(closest_point)
        # Convert the aligned_image back to the original image type (e.g., uint8)
        aligned_image = aligned_image.astype(np.uint8)

        self.newimage = aligned_image


    def get3D(self):
        json_path = os.path.join("data", "camera parameters", self.jsonName)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Camera intrinsic parameters
        fx = data["f"]["val"]
        fy = data["f"]["val"]
        ocx = data["ocx"]["val"]
        ocy = data["ocy"]["val"]
        ok1 = data["ok1"]["val"]
        ok2 = data["ok2"]["val"]
        ok3 = data["ok3"]["val"]
        op1 = data["op1"]["val"]
        op2 = data["op2"]["val"]

        # Known 3D coordinates of hexagon's vertices
        # Coordinates of the hexagon points
        points = [
            (0, 90.01599884033203),
            (77.95613861083984, 45.007999420166016),
            (77.95613861083984, -45.007999420166016),
            (0, -90.01599884033203),
            (-77.95613861083984, -45.007999420166016),
            (-77.95613861083984, 45.007999420166016)
        ]

        # Calculate the distance between the first and second point
        x1, y1 = points[0]
        x2, y2 = points[1]
        hexagon_size = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


        # Initialize array to store 3D coordinates for export
        self.coords3D = []

        # Loop through each hexagon
        for hexagon in self.sorted_cluster:
            # Get the center of the hexagon
            center_x, center_y = hexagon[0]['center']
            hex3D = []
            hex3D.append({'label': hexagon[0]['label'] })
            
            # Loop through each point in the hexagon
            for point in hexagon[0:]:
                # Extract image coordinates
                x, y = point['center']
                
                # Calculate normalized coordinates
                u = (x - ocx) / fx
                v = (y - ocy) / fy
                
                # Apply radial and tangential distortion
                r2 = u**2 + v**2
                delta_u = u * (ok1 * r2 + ok2 * r2**2 + ok3 * r2**3) + 2 * op1 * u * v + op2 * (r2 + 2 * u**2)
                delta_v = v * (ok1 * r2 + ok2 * r2**2 + ok3 * r2**3) + op1 * (r2 + 2 * v**2) + 2 * op2 * u * v
                u_distorted = u + delta_u
                v_distorted = v + delta_v
                
                # Convert to camera coordinates
                x_camera = u_distorted * fx
                y_camera = v_distorted * fy
                z_camera = fx  # Assuming the points lie on the image plane
                
                # Scale the camera coordinates
                scale_factor = hexagon_size / math.sqrt(x_camera**2 + y_camera**2 + z_camera**2)
                x_scaled = x_camera * scale_factor
                y_scaled = y_camera * scale_factor
                z_scaled = z_camera * scale_factor
                
                # Append the scaled coordinates to hex3D
                hex3D.append({'center': (x_scaled, y_scaled, z_scaled)})
            
            self.coords3D.append(hex3D)


    def ret_sorted_clusters(self):
        return self.sorted_cluster
    
    def ret_masked_img(self):
        return self.newimage
    
    def ret_3D_coords(self):
        return self.coords3D
        
    def ret_width_height(self):
        height, width, _ = self.original_image.shape
        return width, height
    
    # Slider value update functions
    def update_tminColor(self, value):
        self.tminColor = value
        self.tminColor_value_label.setText(str(value))
        # Regenerate segmentation mask
        self.generate_mask()
    
    # Slider value update functions
    def update_tdiffColor(self, value):
        self.tdiffColor = value
        self.tdiffColor_value_label.setText(str(value))
        # Regenerate segmentation mask
        self.generate_mask()
    
    # Slider value update functions
    def update_tminArea(self, value):
        self.tminArea = value
        self.tminArea_value_label.setText(str(value))
    
    # Slider value update functions
    def update_tmaxArea(self, value):
        self.tmaxArea = value
        self.tmaxArea_value_label.setText(str(value))
    
    # Slider value update functions
    def update_tDistance(self, value):
            self.tDistance = value
            self.tDistance_value_label.setText(str(value))
    
    # Slider value update functions
    def update_taxisRatio(self, value):
        self.taxisRatio = value
        self.taxisRatio_value_label.setText(str(value))
    
    # Slider value update functions
    def update_tellipse(self, value):
        self.tellipse = value
        self.tellipse_value_label.setText(str(value))

    # Used to run all functions
    def run_all(self):
        self.load_image()
        self.filter_clusters()
        self.find_target_clusters()
        self.find_target_lable()
        self.align_clusters()
        self.get3D()

