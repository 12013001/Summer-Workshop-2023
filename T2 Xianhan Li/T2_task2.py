import cv2
import numpy as np
import time

time_start = time.time()  # 开始计时

img_1 = cv2.imread('11_1.png')
img_2 = cv2.imread('11_2.png')
gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# SIFT特征计算
sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.xfeatures2d.SURF_create()
psd_kp1, psd_des1 = sift.detectAndCompute(gray_1, None)
psd_kp2, psd_des2 = sift.detectAndCompute(gray_2, None)
print(len(psd_kp1))
print(len(psd_kp2))

# Flann特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(psd_des1, psd_des2, k=2)
goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)
print(len(goodMatch))

# 增加一个维度
goodMatch = np.expand_dims(goodMatch, 1)
print(goodMatch[:])
img_out = cv2.drawMatchesKnn(img_1, psd_kp1,
                                img_2, psd_kp2,
                                goodMatch[:], None, flags=2)

time_end = time.time()  # 结束计时

time_c = time_end - time_start  # 运行所花时间
print('time cost', time_c, 's')

cv2.imshow('image', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
