# -*- coding: utf-8 -*-
"""
 Created on 2021/4/7 21:27
 Author: ff_wang
 E-mail: 1105936347@qq.com
 Zhihu : https://www.zhihu.com/people/1105936347
 Github: https://github.com/AFei19911012
-------------------------------------------------
"""
# Source: https://github.com/jrosebr1/imutils

from imutils import perspective
from skimage.filters import threshold_local
import cv2 as cv
import imutils
import matplotlib.pyplot as plt

# 读取图像
image = cv.imread(r'..\images\notecard.png')
# 比例
orig = image.copy()
plt.figure('perspective transform')
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
canny_image = cv.Canny(gray_image, 70, 200)
plt.subplot(1, 3, 2)
plt.imshow(imutils.opencv2matplotlib(canny_image))
plt.axis('off')
plt.title('Canny')
cv.imwrite('../images/Canny.jpg', canny_image)
# 查找轮廓
contours = cv.findContours(canny_image.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# 保留最大轮廓
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:2]
screenCnt = None
for c in contours:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.018*peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    print("No contour detected")
else:
    # 视角
    warped = perspective.four_point_transform(orig, screenCnt.reshape(4, 2))
    # 灰度转换
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    # 阈值分割
    T = threshold_local(warped, 11, offset=10, method='gaussian')
    warped = (warped > T).astype('uint8') * 255
    plt.subplot(1, 3, 3)
    plt.imshow(imutils.opencv2matplotlib(warped))
    plt.axis('off')
    plt.title('Warped')
    plt.show()
    cv.imwrite('../images/Warped.jpg', warped)
