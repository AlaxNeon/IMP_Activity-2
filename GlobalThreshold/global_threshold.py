import cv2
import numpy as np

# Read image
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape

# Threshold value
T = 170

# Output binary image
binary = np.zeros_like(img)

# Manual thresholding
for i in range(rows):
    for j in range(cols):
        if img[i, j] >= T:
            binary[i, j] = 255
        else:
            binary[i, j] = 0

# Save output
cv2.imwrite("global_threshold.png", binary)