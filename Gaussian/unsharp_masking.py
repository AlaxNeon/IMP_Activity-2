import cv2
import numpy as np

# Read image
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Gaussian blur using textbook kernel
kernel = (1/16) * np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])

blurred = cv2.filter2D(img, -1, kernel)

# Unsharp masking
unsharp = img + (img - blurred)
unsharp = np.clip(unsharp, 0, 255)

# Save output
cv2.imwrite("unsharp_mask.png", unsharp)
