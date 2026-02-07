import cv2
import numpy as np

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Global Threshold Formula:
# g(x,y) = 1 if f(x,y) >= T else 0
T = 170

thresholded = np.zeros_like(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i, j] >= T:
            thresholded[i, j] = 255
        else:
            thresholded[i, j] = 0

cv2.imwrite("global_threshold.png", thresholded)
