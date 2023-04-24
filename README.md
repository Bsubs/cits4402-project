# CITS4402 Project 2023
CITS4402 Project - Joo Kai Tay (22489437), Yusi Zhang (23458522), Runtian Lian (23485011)

## How to launch the application
1. Open the anaconda terminal
2. Navigate to the directory where you have extracted the zip file into  
3. Enter the command 'conda env create -f environment_cits4402_project.yml'
4. Enter the command 'conda activate cits4402-2023' to activate the virtual environment 
5. Enter the command 'python main.py' to run the application

## Navigating the GUI
- Load the image in question using the 'Load Image' button
    - This will display the original image on the left side of the screen
    - This will also display a masked image where pixels of red, blue and green have been segmented
- Use the 'tminColor' and 'tdiffColor' sliders to adjust the thresholds in the image until your desired segmenting has been achieved 
- Click the 'Connected Components Analysis' Button
    - This will display the clusters which have been filtered with connected components analysis on the left hand side of the screen
    - Adjust the settings with 'tminArea', 'tmaxArea' and 'taxisRatio' to adjust the minimum area of clusters, maximum area of clusters and roundess of clusters detected
    - On the right side of the screen, only clusters that are part of a hexagonal target are displayed. 
        - Adjust the 'tellipse' field to adjust the sensitivity of this operation

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


