# -*- coding: utf-8 -*-
"""
 Created on 2021/9/12 21:22
 Filename   : cv_image_polygon_cut.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import os

import numpy as np
import cv2


# 图像多边形裁剪
def image_polygon_crop(image, polygon):
    pts = np.array(polygon)
    # 掩膜
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    background = np.ones_like(image, np.uint8) * 255
    cv2.bitwise_not(background, background, mask=mask)
    new_image = background + foreground
    return new_image


if __name__ == '__main__':
    img = cv2.imread("images/dog.jpg")
    module = [(100, 100), (50, 250), (100, 400), (500, 400), (500, 100)]
    img_out = image_polygon_crop(img, module)
    cv2.imwrite("images/dog_cut2.png", img_out)
