import cv2
import numpy as np

# Read image
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape

# Sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Zero padding
padded = np.zeros((rows + 2, cols + 2), dtype=np.int16)
padded[1:rows+1, 1:cols+1] = img

# Output image
sharpened = np.zeros_like(img)

# Manual convolution
for i in range(rows):
    for j in range(cols):
        region = padded[i:i+3, j:j+3]
        value = np.sum(region * kernel)
        value = min(max(value, 0), 255)
        sharpened[i, j] = value

# Save output
cv2.imwrite("sharpened_image.png", sharpened)