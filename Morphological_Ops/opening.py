import cv2
import numpy as np

# Read image
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Convert to binary image
_, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

rows, cols = binary.shape

# Padding for vertical kernel
padded = np.zeros((rows + 2, cols), dtype=np.uint8)
padded[1:rows+1, :] = binary

# ----- Erosion -----
eroded = np.zeros_like(binary)

for i in range(rows):
    for j in range(cols):
        region = padded[i:i+3, j]
        if np.sum(region) == 3:      # FIT
            eroded[i, j] = 1
        else:
            eroded[i, j] = 0

# Padding eroded image
padded_eroded = np.zeros((rows + 2, cols), dtype=np.uint8)
padded_eroded[1:rows+1, :] = eroded

# ----- Dilation -----
opened = np.zeros_like(binary)

for i in range(rows):
    for j in range(cols):
        region = padded_eroded[i:i+3, j]
        if np.sum(region) >= 1:      # HIT or FIT
            opened[i, j] = 1
        else:
            opened[i, j] = 0

# Save output
cv2.imwrite("opened_image.png", opened * 255)