import cv2
import numpy as np

# Read image
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape

# Gaussian kernel
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]])

# Zero padding
padded = np.zeros((rows + 2, cols + 2), dtype=np.int16)
padded[1:rows+1, 1:cols+1] = img

# Gaussian blurred image
blurred = np.zeros_like(img)

for i in range(rows):
    for j in range(cols):
        region = padded[i:i+3, j:j+3]
        blurred[i, j] = np.sum(region * kernel) / 16

# Unsharp masking
mask = img - blurred
sharpened = img + mask

# Clip values
sharpened = np.clip(sharpened, 0, 255)

# Save output
cv2.imwrite("unsharp_masking.png", sharpened)