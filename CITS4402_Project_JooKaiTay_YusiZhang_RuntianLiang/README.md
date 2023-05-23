# CITS4402 Project 2023
CITS4402 Project - Joo Kai Tay (22489437), Yusi Zhang (23458522), Runtian Liang (23485011)

## How to launch the application
1. Open the anaconda terminal
2. Navigate to the directory where you have extracted the zip file into  
3. Enter the command `conda env create -f environment_cits4402_project.yml`
4. Enter the command `conda activate cits4402-2023` to activate the virtual environment 
5. Enter the command `python main.py` to run the application

## Navigating the GUI
- When you open the application you will be on the `Perform Image Segmentation` tab
    - Click the `Perform Image Segmentation for all Cameras`
        - You should see the message `Running image segmentation, please wait patiently` appear on your screen
        - This function is computationally intensive and may take a few minutes to run even on a fast PC. Please do not click on the screen while the message above is still displayed. This may cause a crash of the application. (This has taken over 6 minutes on a slower device)
        - When the message `Image Segmentation Completed` is displayed, the image segmentation is completed.
    - You may then navigate the tabs along the top of the application
        - The tabs are labelled according to which camera the image came from
        - In these tabs you will see 4 images:
            - The original image
            - The result of the rough segmentation mask as of section 1.2 of the thesis
            - The detected targets with squares drawn over the targets as of section 1.3 of the thesis
            - The detected targets with their corresponding strings after the subpixel target alignment as of section 1.4 of the thesis
            - The optimal hyperparameters to achieve this segmentation have been preloaded and are visible on the right of the sliders. You may adjust these as you wish but the results will most likely be worse than with the provided optimal hyperparameters
    - Navigate to the last tab `3D render of room`
        - Click the button `Calibrate Stereo Camera` to see the 3D render of the calibration of the stereo camera 11
        - Click the button `Calibrate All` to see the 3D render of the entire  holographic acquisition rig
            - You should fullscreen this 3D plot as it becomes much easier to see the targets when the plot is larger
            - You can manupulate the plot using your mouse to see the plotted targets from different angles 

## Purpose and Design of the Application
**Purpose**
The purpose of the application is to implement the callibration process for a holographic acquisition rig.

**Design**
The project was implemented in Python using a variety of libraries for image processing. The GUI code was developed using PyQt5. The project was split into 3 source files:
- `main.py`: This code initializes the GUI and creates the various tabs and widgets that the user interacts with
- `segmentimage.py`: This code provides the implementation of tasks 1 & 2.
- `aligncameras.py`: This code provides the implementation of task 3

**Other files and folders**:
The data folder contains the following files:
- `camera parameters`: This folder contains json files describing the intrinsic parameters of the 6 cameras. 
- `images`: This folder contains the 6 images that we will be using to reconstruct the 3D scene of the room
- `tuned_hyperparameters.json`: This json file contains the optimal hyperparameters for each image in the implementation of task 1 & 2. This file will be automatically processed by the code to load these hyperparameters into the application. 

## Implementation of Task 1

### generate_mask()
- This function generates the initial segmentation mask based on certain threshold values
- Compute the minimum and maximum intensity values for each pixel of the original image
- Create a binary mask based on threshold values. The threshold values are defined by the 'tminColor' and 'tdiffColor'
- Apply the mask to the original image using the bitwise and function

### filter_culsters()
- THe masked image from generate_mask() is converted to greyscale and thresholded
- Connected component analysis (CCA) is applied on the thresholded image
    - Each connected component is labeled with a unique integer label and stored in self.labeled_image
- The properties of each connected component is computed using regionprops()
    - This returns values such as area, centroid, bounding box and moments
    - This data is stored in the vairable self.props for use in later functions 
- Clusters are filtered based on size, with thresholds decided by 'tminArea' and 'tmaxArea'
- Compute the minor and major axis lengths of each remaining cluster using the inertia_tensor_eigvals property of the regionprops object. The ratio of the minor and major axes is computed as sigma_min / sigma_max (as per task brief). Clusters whose minor-to-major axis ratio is below the 'taxisRatio' threshold are filtered out.
- A mask is created and applied to the masked image to further fine tune the clusters. 

### find_target_clusters()
- For each centroid of the clusters in self.props (generated in filter_clusters()), add them to a KD Tree
    - Find the 5 nearest neightbours using the query() method 
- For each centroid, compute the residual error of the ellipse fit
- Check if largest residual error for the six centrois is less than a threshold value tellipse
- Add the centroid to the target mask

## Implementation of Task 2

### find_target_label()
- Color Temperature Compensation: This function applies color temperature compensation to the image. This implies that the function corrects the colors in the image to match the colors in the original scene, helping to eliminate color cast caused by different light sources.
- Reference Colors Identification: The function identifies red, green, and blue as the reference colors in the image.
- Pixel Color Analysis: For every pixel in the image, this function determines the closest reference color and applies color temperature compensation to it. If the color component is greater than a set threshold, the color is assigned to the pixel.
- Image Processing: The function processes the image by setting certain pixels to black based on defined conditions.
- HSV Conversion and Masking: The function then converts the processed image to HSV color space and applies masks for blue, red, and green colors.
- Contours and Centroids Calculation: For each color mask, the function calculates contours and their centroids.
- Clustering: The function then clusters these centroids and sorts them from left to right.
- Sorting Points Clockwise: The function sorts points in each cluster clockwise, starting from the blue point.
- Label Assignment: Finally, the function assigns labels to each point in the cluster, based on their position and color.

### align_clusters()
- Use centriods in find_target_lable() to get all the clusters for each color with the get_all_clusters() function.
- Calculate the weighted centroid for each cluster using the weighted_centroid() function.
- Compute the offset between the weighted centroid and original centroid for each cluster.
- Calculate the aligned centroid by adding the offset to the original centroid.
- Move the points in the cluster based on the calculated offset.
- Set the color of the centroids in the aligned image.
- Convert the aligned image back to the original image type (e.g., uint8).
- Display the aligned image using the display_image() function.
- Update self.sorted_cluster with float

## Implementation of Task 3
- In task 3, we started with calculating the 3D coordinates of the targets in the image from camera 11 using the known size of the hexagons.
    - The size of the hexagons was provided to us in an email by Dr Nasir
    - The calculation was done in the `get3D()` function in `segmentimage.py`
- Using the 3D coordinates that we estimated, we used solve pnp to get the pose of camera 11L with respect to these 3D points.
    - This was done in the function `calibrate 11_L()`
    - The other side of the stereo camera was calibrated in `calibrate_11_R()`
    - With this, we had the pose of both stereo cameras.
- The `calibrate_all()` function was used to calibrate the remainder of the cameras
    - We calibrated the cameras in the following sequence: 11L -> 11R -> 71 -> 74 -> 73 -> 72
    - We load the intrinsic parameters (fx, fy, cx, cy) and the distortion parameters (ok1, ok2, ok3, op1, op2) from the json files provided
    - Using the `get_matching_coordinates()` function, we match the targets in the 2 cameras we are calibrating using the strings found in task 2
    - We use the 3D points from the previous camera, the 2D points from the current camera and the camera intrinsic parameters in solvePnP to determine the pose of the current camera relative to the previous camera
    - We then transform the image coordinates from the current camera to the new coordinate system using the rotation and translation vectors from solvePnP
- The 3D coordinates and pose are plotted in the 3D plot


## Tuning Hyperparameters

### Camera 11 RGB Left

- tminColor:   50
- tdiffColor:  100
- tminArea:    40
- tmaxArea:    150
- tDistance:   100
- taxisRatio:  2.5
- tellipse:    7

### Camera 11 RGB Right

- tminColor:   56
- tdiffColor:  75
- tminArea:    40
- tmaxArea:    116
- tDistance:   27
- taxisRatio:  2.5
- tellipse:    7

### Camera 71 RGB 
This one needs fixing.

- tminColor:   10799
- tdiffColor:  30
- tminArea:    10
- tmaxArea:    178
- taxisRatio:  1.8
- tellipse:    120

### Camera 72 RGB 

- tminColor:   80
- tdiffColor:  99
- tminArea:    6
- tmaxArea:    53
- tDistance:   20
- taxisRatio:  2.3
- tellipse:    7

### Camera 73 RGB 

- tminColor:   99
- tdiffColor:  99
- tminArea:    6
- tmaxArea:    92
- tDistance:   20
- taxisRatio:  2.2
- tellipse:    4

### Camera 74 RGB 

- tminColor:   65
- tdiffColor:  40
- tminArea:    20
- tmaxArea:    200
- tDistance:   23
- taxisRatio:  1.4
- tellipse:    30
