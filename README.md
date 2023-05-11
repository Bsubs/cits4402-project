# CITS4402 Project 2023
CITS4402 Project - Joo Kai Tay (22489437), Yusi Zhang (23458522), Runtian Liang (23485011)

## How to launch the application
1. Open the anaconda terminal
2. Navigate to the directory where you have extracted the zip file into  
3. Enter the command `conda env create -f environment_cits4402_project.yml`
4. Enter the command `conda activate cits4402-2023` to activate the virtual environment 
5. Enter the command `python main.py` to run the application

## Navigating the GUI
- Load the image in question using the 'Load Image' button
    - This will display the original image on the left side of the screen
    - This will also display a masked image where pixels of red, blue and green have been segmented
- Use the `tminColor` and `tdiffColor` sliders to adjust the thresholds in the image until your desired segmenting has been achieved 
- Click the `Connected Components Analysis` Button
    - This will display the clusters which have been filtered with connected components analysis on the left hand side of the screen
    - Adjust the settings with `tminArea`, `tmaxArea` and `taxisRatio` to adjust the minimum area of clusters, maximum area of clusters and roundess of clusters detected
- Click the 'Detect Hexagons' button
    - On the right side of the screen, only clusters that are part of a hexagonal target are displayed. 
        - Adjust the `tellipse` field to adjust the sensitivity of this operation

## Purpose and Design of the Application
**Purpose**
The purpose of the application is to implement the callibration process for a holographic acquisition rig.

**Design**
The project was implemented in Python using a variety of libraries for image processing. The GUI code was developed using PyQt5. Detailed explanations of each function will be provided below. 

## Implementation

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
- Update self.sorted_cluster, which can be used in getting depth


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

- tminColor:   99
- tdiffColor:  17
- tminArea:    13
- tmaxArea:    111
- taxisRatio:  1.8
- tellipse:    7

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
