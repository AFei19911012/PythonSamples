# -*- coding: utf-8 -*-
"""
 Created on 2021/9/12 21:22
 Filename   : ex_image_polygon_cut2.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import numpy as np
import cv2

img = cv2.imread("../images/dog.jpg")
pts = np.array([(100, 100), (50, 250), (100, 400), (500, 400), (500, 100)])
# mask
mask = np.zeros(img.shape[:2], np.uint8)
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
foreground = cv2.bitwise_and(img, img, mask=mask)
background = np.ones_like(img, np.uint8) * 255
cv2.bitwise_not(background, background, mask=mask)
new_img = background + foreground
cv2.imwrite("../images/dog_cut2.png", new_img)
