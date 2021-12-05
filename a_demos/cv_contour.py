# -*- coding: utf-8 -*-
"""
 Created on 2021/4/7 20:51
 Author: ff_wang
 E-mail: 1105936347@qq.com
 Zhihu : https://www.zhihu.com/people/1105936347
 Github: https://github.com/AFei19911012
-------------------------------------------------
"""
# Source: https://github.com/spmallick/learnopencv

from imutils import perspective
from skimage.filters import threshold_local
import cv2 as cv
import imutils
import matplotlib.pyplot as plt

# 读取图像
image = cv.imread(r'images/dog.jpg')
# 比例
orig = image.copy()
plt.figure('opencv contour')
plt.subplot(1, 3, 1)
plt.imshow(imutils.opencv2matplotlib(orig))
plt.axis('off')
plt.title('original')
# 灰度转化
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# 滤波
gray_image = cv.GaussianBlur(gray_image, (5, 5), 0)
# gray_image = cv.bilateralFilter(gray_image, 11, 17, 17)
# 二值化
# ret, gray_image = cv.threshold(gray_image, 100, 255, 0)
# 边缘检测
canny_image = cv.Canny(gray_image, 120, 200)
plt.subplot(1, 3, 2)
plt.imshow(imutils.opencv2matplotlib(canny_image))
plt.axis('off')
plt.title('Canny')
# 查找轮廓
contours = cv.findContours(canny_image.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# 保留最大轮廓
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]
# 在原图上画轮廓
cv.drawContours(orig, contours, -1, (255, 0, 255), 2)
# 在原图上画矩形轮廓
for i in range(0, len(contours)):
    x, y, w, h = cv.boundingRect((contours[i]))
    cv.rectangle(orig, (x, y), (x+w, y+h), (255, 255, 0), 3)

plt.subplot(1, 3, 3)
plt.imshow(imutils.opencv2matplotlib(orig))
plt.axis('off')
plt.title('contour')
plt.show()
