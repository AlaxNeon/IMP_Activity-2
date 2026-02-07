import cv2
import numpy as np

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Line detection kernel (horizontal)
line_kernel = np.array([[-1, -1, -1],
                        [ 2,  2,  2],
                        [-1, -1, -1]])

line_detected = cv2.filter2D(img, -1, line_kernel)

# Point detection kernel
point_kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])

point_detected = cv2.filter2D(img, -1, point_kernel)

cv2.imwrite("line_detected.png", line_detected)
cv2.imwrite("point_detected.png", point_detected)
