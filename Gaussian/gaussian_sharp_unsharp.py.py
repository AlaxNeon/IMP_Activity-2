import cv2
import numpy as np

# Read image
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# ---------------- GAUSSIAN BLUR ----------------
def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

kernel = gaussian_kernel(5, 1)
blurred = cv2.filter2D(img, -1, kernel)

# ---------------- SHARPENING ----------------
sharpen_kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
sharpened = cv2.filter2D(img, -1, sharpen_kernel)

# ---------------- UNSHARP MASK ----------------
unsharp = img + (img - blurred)
unsharp = np.clip(unsharp, 0, 255)

# Save outputs
cv2.imwrite("gaussian_blur.png", blurred)
cv2.imwrite("sharpened.png", sharpened)
cv2.imwrite("unsharp_mask.png", unsharp)
