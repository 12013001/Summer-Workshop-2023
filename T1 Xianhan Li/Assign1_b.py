import tkinter

import cv2

img = cv2.imread("lena.png")

img_mean = cv2.blur(img,(9,9))
img_Gaussian = cv2.GaussianBlur(img,(9,9),0)
img_median = cv2.medianBlur(img,9)
img_bilater = cv2.bilateralFilter(img,9,75,75)

titles = ['Nomal','Normalized Box Filter','Gaussian Filter','Median Filter','Bilateral Filter']
imgs = [img, img_mean, img_Gaussian, img_median, img_bilater]

for i in range(5):
    cv2.imshow(titles[i],imgs[i])
    cv2.waitKey()

