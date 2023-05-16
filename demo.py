import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


P1 = np.array([[697.45550537,   0,621.21575928   ,0],
 [  0, 697.45550537 ,353.87438965 ,  0],
 [  0,  0,1,0]])

P2 = np.array([[6.52060686e+02, 9.64968406e-01 ,6.29374077e+02 ,7.67916510e+04],
 [7.48476831e+00, 6.40958664e+02, 3.53762780e+02, 1.49123414e+01],
 [1.89770784e-02 ,2.70399337e-03, 9.99816263e-01 ,3.25487815e-02]] )


sorted_cluster_left = np.array([[194, 276],
                                [209, 286],
                                [208, 306],
                                [193, 316],
                                [180, 306],
                                [181, 286],
                                [394, 271],
                                [412, 283],
                                [410, 303],
                                [392, 313],
                                [375, 302],
                                [376, 281],
                                [542, 351],
                                [560, 363],
                                [558, 385],
                                [539, 394],
                                [521, 382],
                                [523, 360],
                                [708, 292],
                                [727, 304],
                                [726, 326],
                                [707, 336],
                                [688, 324],
                                [688, 302],
                                [869, 385],
                                [887, 398],
                                [886, 420],
                                [867, 430],
                                [849, 418],
                                [850, 395],
                                [1123, 345],
                                [1139, 359],
                                [1138, 389],
                                [1122, 403],
                                [1109, 389],
                                [1109, 361]], dtype=np.float32)

sorted_cluster_right = np.array([[209, 282],
                                 [222, 292],
                                 [221, 311],
                                 [206, 320],
                                 [194, 311],
                                 [196, 292],
                                 [400, 276],
                                 [418, 286],
                                 [416, 308],
                                 [398, 317],
                                 [383, 306],
                                 [383, 286],
                                 [546, 354],
                                 [564, 367],
                                 [562, 388],
                                 [542, 398],
                                 [524, 386],
                                 [527, 363],
                                 [710, 296],
                                 [730, 307],
                                 [729, 329],
                                 [709, 340],
                                 [691, 328],
                                 [691, 306],
                                 [873, 388],
                                 [892, 400],
                                 [890, 424],
                                 [871, 433],
                                 [852, 421],
                                 [853, 398],
                                 [1125, 348],
                                 [1139, 362],
                                 [1139, 392],
                                 [1123, 407],
                                 [1110, 393],
                                 [1111, 364]], dtype=np.float32)

points1 = sorted_cluster_left
points2 = sorted_cluster_right

normalized_points1 = cv2.normalize(points1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
normalized_points2 = cv2.normalize(points2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


points4D_homogeneous = cv2.triangulatePoints(P1, P2, normalized_points1.T, normalized_points2.T)


points3D_homogeneous = points4D_homogeneous / points4D_homogeneous[3]
points3D = points3D_homogeneous[:3].T

print("三维坐标: ", points3D)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()



