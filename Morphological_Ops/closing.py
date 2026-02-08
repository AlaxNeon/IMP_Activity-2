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

# ----- Dilation -----
dilated = np.zeros_like(binary)

for i in range(rows):
    for j in range(cols):
        region = padded[i:i+3, j]
        if np.sum(region) >= 1:      # HIT or FIT
            dilated[i, j] = 1
        else:
            dilated[i, j] = 0

# Padding dilated image
padded_dilated = np.zeros((rows + 2, cols), dtype=np.uint8)
padded_dilated[1:rows+1, :] = dilated

# ----- Erosion -----
closed = np.zeros_like(binary)

for i in range(rows):
    for j in range(cols):
        region = padded_dilated[i:i+3, j]
        if np.sum(region) == 3:      # FIT
            closed[i, j] = 1
        else:
            closed[i, j] = 0

# Save output
cv2.imwrite("closed_image.png", closed * 255)