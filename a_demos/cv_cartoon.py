# -*- coding: utf-8 -*-
"""
 Created on 2021/6/28 22:33
 Filename   : cv_cartoon.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: 卡通图像
"""

# =======================================================
import cv2


def main():
    image_path = 'images/obama.jpg'
    img_rgb = cv2.imread(image_path)
    height, width = img_rgb.shape[0:2]
    img_color = img_rgb.copy()
    """ 先对图像进行高斯平滑，然后再进行降采样 """
    for _ in range(2):
        img_color = cv2.pyrDown(img_color)
    """ 重复使用小的双边滤波代替一个大的滤波 """
    for _ in range(7):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
    """ 升采样图片到原始大小 """
    for _ in range(2):
        img_color = cv2.pyrUp(img_color)
    img_color = cv2.resize(img_color, (width, height))
    """ 转换成灰度图 """
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    """ 模糊 """
    img_blur = cv2.medianBlur(img_gray, 7)
    """ 边缘检测 """
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    """ 转换回彩图 """
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    """ 二进制 “与“ 操作 """
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    """ 显示 """
    cv2.imshow('Cartoon', img_cartoon)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
