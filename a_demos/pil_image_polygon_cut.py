# -*- coding: utf-8 -*-
"""
 Created on 2021/9/12 19:48
 Filename   : pil_image_polygon_cut.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import numpy as np
from PIL import Image, ImageDraw

# RGB 或 RGBA
img = Image.open("images/dog.jpg").convert("RGB")
img_array = np.asarray(img)
# mask
polygon = [(100, 100), (50, 250), (100, 400), (500, 400), (500, 100)]
# 单通道
mask_img = Image.new('1', (img_array.shape[1], img_array.shape[0]), 0)
ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
mask = np.array(mask_img)
new_img_array = np.empty(img_array.shape, dtype='uint8')
# RGB
new_img_array[:, :, :3] = img_array[:, :, :3]
# 每个通道裁剪
new_img_array[:, :, 0] = new_img_array[:, :, 0] * mask
new_img_array[:, :, 1] = new_img_array[:, :, 1] * mask
new_img_array[:, :, 2] = new_img_array[:, :, 2] * mask
# 用透明度来裁剪
# new_img_array[:, :, 3] = mask * 255
new_img = Image.fromarray(new_img_array, "RGB")
new_img.save("images/dog_cut.png")
