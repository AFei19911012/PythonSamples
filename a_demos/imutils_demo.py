# -*- coding: utf-8 -*-
"""
 Created on 2021/4/7 20:07
 Author: ff_wang
 E-mail: 1105936347@qq.com
 Zhihu : https://www.zhihu.com/people/1105936347
 Github: https://github.com/AFei19911012
-------------------------------------------------
"""
# Source: https://github.com/jrosebr1/imutils

import cv2
import imutils
import matplotlib.pyplot as plt

# load image
image = cv2.imread(r'../images/dog.jpg')
# show image
cv2.imshow('Original image', image)
# 0: 只显示当前帧图像，相当于视频暂停
# 1： 延时 1ms 切换到下一帧图像，针对视频
# cv2.waitKey(0)
# delete figure
# cv2.destroyAllWindows()

# translation: left -> right 100, up -> down 200
translation_image = imutils.translate(image, 100, 200)
# save image
cv2.imwrite(r'../images/translation_image.jpg', translation_image)

# rotation
rotation_image = imutils.rotate(image, 90)
rotation_bound_image = imutils.rotate_bound(image, 45)
cv2.imwrite(r'../images/rotation_image_90.jpg', rotation_image)
cv2.imwrite(r'../images/rotation_bound_image_45.jpg', rotation_bound_image)

# resizing: 等比例缩放
resized_image = imutils.resize(image, width=400)
cv2.imwrite(r'../images/resized_image_400.jpg', resized_image)

# skeletonization: 提取拓扑骨架（topological skeleton）
# size：粒度，越小越耗时
# 转成灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
skeleton_image = imutils.skeletonize(gray_image, size=(3, 3))
skeleton_Canny_image = cv2.Canny(gray_image, 60, 200)
cv2.imwrite(r'../images/gray_image.jpg', gray_image)
cv2.imwrite(r'../images/skeleton_image.jpg', skeleton_image)
cv2.imwrite(r'../images/skeleton_Canny_image.jpg', skeleton_Canny_image)

# matplotlib 显示
# cv2.cvtColor 将 BGR 序列转换为 RGB 序列
# 使用 imutils.opencv2matplotlib
plt.figure('Opencv vs Matplotlib')
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')
plt.title('Incorrect')
plt.subplot(1, 2, 2)
plt.imshow(imutils.opencv2matplotlib(image))
plt.axis('off')
plt.title('Correct')
plt.show()
