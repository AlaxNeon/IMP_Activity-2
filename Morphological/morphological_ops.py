import cv2
import numpy as np

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Binary conversion
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)

# Dilation
dilation = cv2.dilate(binary, kernel, iterations=1)

# Erosion
erosion = cv2.erode(binary, kernel, iterations=1)

# Opening = Erosion followed by Dilation
opening = cv2.dilate(erosion, kernel, iterations=1)

# Closing = Dilation followed by Erosion
closing = cv2.erode(dilation, kernel, iterations=1)

cv2.imwrite("dilation.png", dilation)
cv2.imwrite("erosion.png", erosion)
cv2.imwrite("opening.png", opening)
cv2.imwrite("closing.png", closing)
