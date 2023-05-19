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

# Camera distortion parameters
k1 = -0.1568632423877716
k2 = -0.00861599575728178
k3 = 0.021558040753006935
p1 = -3.6368699511513114e-05
p2 = -0.0009189021657221019

# Construct camera matrix
camera_matrix = np.array([[fx, 0, ocx], [0, fy, ocy], [0, 0, 1]])

# Construct distortion coefficients
dist_coeffs = np.array([k1, k2, p1, p2, k3])

# Plot 2D coordinates in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# List to store estimated poses
poses = []

# Iterate over hexagons
for hexagon in hexagons:
    # Extract image coordinates of hexagon's vertices
    hexagon_2d = np.array(hexagon, dtype=np.float32)
    
    # Undistort image points
    undistorted_points = cv2.undistortPoints(hexagon_2d.reshape(-1, 1, 2), camera_matrix, dist_coeffs)
    undistorted_points = undistorted_points.squeeze()
    
    # Generate 3D coordinates of the hexagon's vertices (assuming planar target)
    hexagon_3d = np.zeros((6, 3), dtype=np.float32)
    hexagon_3d[:, :2] = hexagon_size * hexagon
    # # Plot 3D points
    ax.scatter(hexagon_3d[:, 0], hexagon_3d[:, 1], hexagon_3d[:, 2], c='r', marker='o')

    # Solve for extrinsic parameters
    retval, rvec, tvec = cv2.solvePnP(hexagon_3d, undistorted_points, camera_matrix, dist_coeffs)

    # Plot projection of 3D points onto image plane
    projected_points, _ = cv2.projectPoints(hexagon_3d, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.squeeze()
    ax.scatter(projected_points[:, 0], projected_points[:, 1], 0, c='b', marker='x')

    
    # Store estimated pose
    pose = {'rvec': rvec, 'tvec': tvec}
    poses.append(pose)

# Bundle adjustment to refine poses
obj_points = [hexagon_3d] * len(hexagons)  # 3D coordinates of the hexagon's vertices
img_points = [hexagon for hexagon in hexagons]  # Image coordinates of hexagons' vertices
#retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, camera_matrix, dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3)

# Set plot labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show the plot
plt.show()