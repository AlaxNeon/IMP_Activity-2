import cv2
import numpy as np

# Read image
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape

# Gaussian kernel (textbook)
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]])

# Zero padding
padded = np.zeros((rows + 2, cols + 2), dtype=np.uint8)
padded[1:rows+1, 1:cols+1] = img

# Output image
blurred = np.zeros_like(img)

# Manual convolution
for i in range(rows):
    for j in range(cols):
        region = padded[i:i+3, j:j+3]
        value = np.sum(region * kernel) / 16
        blurred[i, j] = int(value)

# Save output
cv2.imwrite("gaussian_blur.png", blurred)