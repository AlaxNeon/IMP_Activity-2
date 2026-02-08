# BLUR
import cv2
import numpy as np

# Read image
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Gaussian kernel as per textbook
kernel = (1/16) * np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])

# Apply convolution
gaussian_blur = cv2.filter2D(img, -1, kernel)

# Save output
cv2.imwrite("gaussian_blur.png", gaussian_blur)

