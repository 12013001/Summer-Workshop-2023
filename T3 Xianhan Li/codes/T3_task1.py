import cv2
import numpy as np

img_1 = cv2.imread('15_1.png')
img_2 = cv2.imread('15_2.png')
gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SURF_create()

psd_kp1, psd_des1 = sift.detectAndCompute(gray_1, None)
psd_kp2, psd_des2 = sift.detectAndCompute(gray_2, None)

image_with_kp1 = cv2.drawKeypoints(gray_1, psd_kp1, None, flags=2)
image_with_kp2 = cv2.drawKeypoints(gray_2, psd_kp2, None, flags=2)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(psd_des1, psd_des2, k=2)
goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

goodMatch = np.expand_dims(goodMatch, 1)
img_out = cv2.drawMatchesKnn(image_with_kp1, psd_kp1,
                                image_with_kp2, psd_kp2,
                                goodMatch[:5], None, flags=2)

cv2.imshow('image', img_out)
cv2.waitKey(0)