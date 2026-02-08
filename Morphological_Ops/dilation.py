import cv2
import numpy as np

# Read image
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Convert to binary image
_, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

rows, cols = binary.shape

# Zero padding (top and bottom)
padded = np.zeros((rows + 2, cols), dtype=np.uint8)
padded[1:rows+1, :] = binary

# Output image
dilated = np.zeros_like(binary)

# Manual dilation
for i in range(rows):
    for j in range(cols):
        region = padded[i:i+3, j]

        if np.sum(region) >= 1:   # HIT or FIT
            dilated[i, j] = 1
        else:                     # MISS
            dilated[i, j] = 0

# Save output
cv2.imwrite("dilated_image.png", dilated * 255)