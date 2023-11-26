import numpy as np
import cv2 as cv

image = cv.imread('building.jpg', 0)
cv.imshow("demo", image)

# Gaussian filter (5*5, σ = 1.4)
k = 5 // 2
gaussian = np.zeros([5, 5])
for i in range(5):
    for j in range(5):
        gaussian[i, j] = np.exp(-((i - k) ** 2 + (j - k) ** 2) / (2 * 1.4 ** 2))
gaussian /= 2 * np.pi * 1.4 ** 2
# Batch Normalization
gaussian = gaussian / np.sum(gaussian)
# Use Gaussian Filter
W, H = image.shape
gaussian_image = np.zeros([W - k * 2, H - k * 2])
for i in range(W - 2 * k):
    for j in range(H - 2 * k):
        # convolution operation
        gaussian_image[i, j] = np.sum(image[i:i + 5, j:j + 5] * gaussian)
gaussian_image = np.uint8(gaussian_image)
cv.imshow("Smooth", gaussian_image)

# Use Sobel to compute gradients and direction
W_0, H_0 = gaussian_image.shape
Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
gradients = np.zeros([W_0 - 2, H_0 - 2])
directions = np.zeros([W_0 - 2, H_0 - 2])
for i in range(W_0 - 2):
    for j in range(H_0 - 2):
        dx = np.sum(gaussian_image[i:i + 3, j:j + 3] * Gx)
        dy = np.sum(gaussian_image[i:i + 3, j:j + 3] * Gy)
        gradients[i, j] = np.sqrt(dx ** 2 + dy ** 2)
        if dx == 0:
            directions[i, j] = np.pi / 2
        else:
            directions[i, j] = np.arctan(dy / dx)
gradients = np.uint8(gradients)

# Non-Maximum Suppression
W_1, H_1 = gradients.shape
nms = np.copy(gradients[1:-1, 1:-1])
for i in range(1, W_1 - 1):
    for j in range(1, H_1 - 1):
        theta = directions[i, j]
        weight = np.tan(theta)
        if theta > np.pi / 4:
            d1 = [0, 1]
            d2 = [1, 1]
            weight = 1 / weight
        elif theta >= 0:
            d1 = [1, 0]
            d2 = [1, 1]
        elif theta >= - np.pi / 4:
            d1 = [1, 0]
            d2 = [1, -1]
            weight *= -1
        else:
            d1 = [0, -1]
            d2 = [1, -1]
            weight = -1 / weight
        g1 = gradients[i + d1[0], j + d1[1]]
        g2 = gradients[i + d2[0], j + d2[1]]
        g3 = gradients[i - d1[0], j - d1[1]]
        g4 = gradients[i - d2[0], j - d2[1]]
        grade_count1 = g1 * weight + g2 * (1 - weight)
        grade_count2 = g3 * weight + g4 * (1 - weight)
        if grade_count1 > gradients[i, j] or grade_count2 > gradients[i, j]:
            nms[i - 1, j - 1] = 0

# Double Threshold
visited = np.zeros_like(nms)
output_image = nms.copy()
W_2, H_2 = output_image.shape
def dfs(i, j):
    if i >= W_2 or i < 0 or j >= H_2 or j < 0 or visited[i, j] == 1:
        return
    visited[i, j] = 1
    if output_image[i, j] > 50:
        output_image[i, j] = 255
        dfs(i - 1, j - 1)
        dfs(i - 1, j)
        dfs(i - 1, j + 1)
        dfs(i, j - 1)
        dfs(i, j + 1)
        dfs(i + 1, j - 1)
        dfs(i + 1, j)
        dfs(i + 1, j + 1)
    else:
        output_image[i, j] = 0
for w in range(W_2):
    for h in range(H_2):
        if visited[w, h] == 1:
            continue
        if output_image[w, h] >= 100:
            dfs(w, h)
        elif output_image[w, h] <= 50:
            output_image[w, h] = 0
            visited[w, h] = 1
for w in range(W_2):
    for h in range(H_2):
        if visited[w, h] == 0:
            output_image[w, h] = 0
cv.imshow("outputImage", output_image)

# Harris Corner Detector
window_size=3
k=0.04
threshold=0.1
dx = cv.Sobel(output_image, cv.CV_64F, 1, 0, ksize=3)
dy = cv.Sobel(output_image, cv.CV_64F, 0, 1, ksize=3)
Ixx = dx * dx
Iyy = dy * dy
Ixy = dx * dy
Sxx = cv.GaussianBlur(Ixx, (window_size, window_size), 0)
Syy = cv.GaussianBlur(Iyy, (window_size, window_size), 0)
Sxy = cv.GaussianBlur(Ixy, (window_size, window_size), 0)
det_M = Sxx * Syy - Sxy * Sxy
trace_M = Sxx + Syy
R = det_M - k * trace_M * trace_M
corner_points = np.zeros_like(R)
corner_points[R > threshold * R.max()] = 1
image_with_corners = cv.cvtColor(output_image, cv.COLOR_GRAY2BGR)
image_with_corners[corner_points == 1] = [0, 0, 255]

cv.imshow("Harris Corner Detection", image_with_corners)
cv.waitKey(0)