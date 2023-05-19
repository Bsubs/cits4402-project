import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

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

# Image coordinates of hexagon's vertices
#hexagon_2d = np.array([[194, 276], [209, 286], [208, 306], [193, 316], [180, 306], [181, 286]], dtype=np.float32)
#hexagon_2d = np.array([[193, 316], [208, 306], [209, 286], [194, 276],[181, 286], [180, 306]], dtype=np.float32)

hexagons = np.array([[[194, 276],
                    [209, 286],
                    [208, 306],
                    [193, 316],
                    [180, 306],
                    [181, 286]],
                    [[394, 271],
                    [412, 283],
                    [410, 303],
                    [392, 313],
                    [375, 302],
                    [376, 281]],
                    [[542, 351],
                    [560, 363],
                    [558, 385],
                    [539, 394],
                    [521, 382],
                    [523, 360]],
                    [[708, 292],
                    [727, 304],
                    [726, 326],
                    [707, 336],
                    [688, 324],
                    [688, 302]],
                    [[869, 385],
                    [887, 398],
                    [886, 420],
                    [867, 430],
                    [849, 418],
                    [850, 395]],
                    [[1123, 345],
                    [1139, 359],
                    [1138, 389],
                    [1122, 403],
                    [1109, 389],
                    [1109, 361]]], dtype=np.float32)

# Camera intrinsic parameters
fx = 640
fy = 640
ocx = 641.63525390625
ocy = 355.5729675292969
ok1 = -0.1568632423877716
ok2 = -0.00861599575728178
ok3 = 0.021558040753006935
op1 = -3.6368699511513114e-05
op2 = -0.0009189021657221019

# Known size of the hexagon in millimeters
hexagon_size = 10

# Data structure with point coordinates
points_data = [
    [{'center': (194, 276), 'x': 192, 'y': 272, 'w': 8, 'h': 9, 'label': 'HexaTarget_RRGRR_1'},
     {'center': (209, 286)}, {'center': (208, 306)}, {'center': (193, 316)}, {'center': (180, 306)},
     {'center': (181, 286)}]
]

# Initialize 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Loop through each hexagon
for hexagon in points_data:
    # Get the center of the hexagon
    center_x, center_y = hexagon[0]['center']
    
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
        
        # Plot the 3D coordinates
        ax.scatter(x_scaled, y_scaled, z_scaled, c='b', marker='o')
        print(f"Point: ({x_scaled:.2f}, {y_scaled:.2f}, {z_scaled:.2f})")

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the aspect ratio of the plot
ax.set_box_aspect([1, 1, 1])

# Show the plot
plt.show()

