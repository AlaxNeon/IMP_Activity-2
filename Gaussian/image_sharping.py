import cv2
import numpy as np

# Read image
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Apply sharpening
sharpened = cv2.filter2D(img, -1, kernel)

# Save output
cv2.imwrite("sharpened.png", sharpened)
