import cv2
import numpy as np

# Read image
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape

# Point detection mask
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Zero padding
padded = np.zeros((rows + 2, cols + 2), dtype=np.int16)
padded[1:rows+1, 1:cols+1] = img

# Output image
point_image = np.zeros_like(img)

# Manual convolution
for i in range(rows):
    for j in range(cols):
        region = padded[i:i+3, j:j+3]
        value = np.sum(region * kernel)
        value = min(max(value, 0), 255)
        point_image[i, j] = value

# Save output
cv2.imwrite("point_detected.png", point_image)