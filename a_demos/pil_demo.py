# -*- coding: utf-8 -*-
"""
 Created on 2021/5/24 18:03
 Filename   : pil_demo.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: 
# =======================================================
from PIL import Image
from PIL import ImageFilter
import numpy as np

# open: 加载图片
image = Image.open('images/dog.jpg')
# 用 Windows 自带的图片浏览器打开
# image.show()

# save: 保存图片
image.save('images/dog.png')
print(image.format)
print(image.size)
print(image.mode)
print(image.palette)
'''
JPEG
(768, 576)
RGB
None
'''

# gray: L = R * 299/1000 + G * 587/1000 + B * 114/1000
image_new = image.convert('L')
# image_new.show()

# new: 新建图片
# 不设置颜色默认为 0 即黑色
image_new = Image.new('RGB', (256, 256), '#FF0000')
# image_new.show()

# copy: 拷贝
image_copy = image.copy()

# crop: 局部拷贝
rect = (300, 300, 500, 500)
image_crop = image.crop(rect)

# paste: 粘贴，填充
# 注意 rect 和 box 的 size 一致
box = (100, 100, 300, 300)
image.paste(image_crop, box)
# image.show()
# 黄色填充
image.paste((255, 255, 0), box)
# image.show()
# 蓝色填充
image.paste('blue', box)
# image.show()

# filter: 滤波
# BLUR、CONTOUR、DETAIL、EDGE_ENHANCE、EDGE_ENHANCE_MORE、EMBOSS、FIND_EDGES、SMOOTH、SMOOTH_MORE、SHARPEN
# 模糊，均值滤波
image_blur = image.filter(ImageFilter.BLUR)
# 轮廓
image_contour = image.filter(ImageFilter.CONTOUR)
# 边缘检测
image_edge = image.filter(ImageFilter.FIND_EDGES)
# image_blur.show()
# image_contour.show()
# image_edge.show()

# blend: 融合，按照透明度
# out = image1 * (1 - alpha) + image2 * alpha
image_background = Image.open('images/background.jpg')
image_blend = Image.blend(image, image_background, 0.2)
# image_blend.show()

# split: 图片通道分离
image_r, image_g, image_b = image.split()
print(image_r.mode)
print(image_r.size)
'''
L
(768, 576)
'''

# composite: 图像融合
image_new = Image.composite(image, image_background, image_b)
# image_new.show()


# eval: 对图像进行函数处理
def fun1(x):
    return x * 0.5


def fun2(x):
    return x*2


image_eval1 = Image.eval(image, fun1)
image_eval2 = Image.eval(image, fun2)
# image_eval1.show()
# image_eval2.show()

# merge: 图像融合
image_r2, image_g2, image_b2 = image_background.split()
image_merge = Image.merge('RGB', [image_r, image_g2, image_b2])
# image_merge.show()

# getbbox: 图像非零区域的包围盒
print(image.getbbox())
'''
(0, 0, 768, 576)
'''

# getdata: 像素值
sequence = image.getdata()
sequence = list(sequence)
print(sequence[0])
print(sequence[1])
print(sequence[2])
'''
(57, 58, 50)
(58, 59, 51)
(60, 61, 53)
'''

# getpixel: 指定位置像素值
print(image.getpixel((0, 0)))
print(image_b.getpixel((0, 0)))
'''
(57, 58, 50)
50
'''

# histogram: 直方图，如果有多个通道，则拼接起来
image_hist = image.histogram()
print(len(image_hist))
print(image_hist[0])
print(image_hist[150])
print(image_hist[300])
'''
768   # 三个通道 256*3
40202
3773
416
'''

# resize: 图像尺寸
image_resize = image.resize((400, 400))
# image_resize.show()

# rotate: 图像旋转，逆时针
image_30 = image.rotate(30)
image_45 = image.rotate(45)
print(image_30.size)
print(image_45.size)
'''
(768, 576)
(768, 576)
'''
# image_30.show()
# image_45.show()

# transpose: 图像翻转、旋转
image.transpose(Image.FLIP_TOP_BOTTOM)  # 上下翻转
image.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转
image.transpose(Image.ROTATE_90)        # 旋转 90 度
image.transpose(Image.ROTATE_180)       # 旋转 180 度
image.transpose(Image.ROTATE_270)       # 旋转 270 度

# seek: 查找指定帧
image_gif = Image.open('images/ball.gif')
print(image_gif.mode)
'''
P
'''
# image_gif.show()    # 第 0 帧
image_gif.seek(10)
# image_gif.show()
image_gif.seek(30)
# image_gif.show()

# tell: 第几帧
print(image_gif.tell())
'''
30
'''

# thumbnail: 缩略图，等比例
# filter: NEAREST(default)、BILINEAR、BICUBIC、ANTIALIAS
image_copy.thumbnail((100, 100), Image.NEAREST)
# image_copy.show()

# transform: 图像变换
# image 的 (100, 100, 500, 500) 缩放到 (100, 200)
image_extent = image.transform((100, 200), Image.EXTENT, (100, 100, 500, 500))
print(image_extent.size)
'''
(100, 200)
'''
# image_extent.show()

# (a,b,c,d,e,f): 仿射变换 (x, y) --> (ax+by+c, dx+ey+f)
image_affine = image.transform((200, 200), Image.AFFINE, (1, 2, 3, 2, 1, 4))
# image_affine.show()

# 四个点围成的区域映射到指定尺寸
image_quad = image.transform((200, 200), Image.QUAD, (0, 0, 0, 400, 500, 400, 500, 0))
# image_quad.show()


def cal_coefs(original_coords, warped_coords):
    matrix = []
    for p1, p2 in zip(original_coords, warped_coords):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(warped_coords).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


'''
coefs = cal_coefs([(867,652), (1020,580), (1206,666), (1057,757)], [(700,732), (869,754), (906,916), (712,906)])
coefs_inv = cal_coefs([(700,732), (869,754), (906,916), (712,906)], [(867,652), (1020,580), (1206,666), (1057,757)])
'''
# (a, b, c, d, e, f, g, h): (x, y) --> (ax + by + c)/(gx + hy + 1), (dx+ ey + f)/(gx + hy + 1)
image_perspective = image.transform((1000, 500), Image.PERSPECTIVE, (1, 2, -900, -1, 1, 500, 0, 0))
image_perspective.show()
