# -*- coding: utf-8 -*-
"""
 Created on 2021/5/19 19:10
 Filename   : digital_makeup.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: https://github.com/ageitgey/face_recognition
# =======================================================
import numpy as np
from PIL import Image, ImageDraw
import face_recognition

# 加载图片
image = face_recognition.load_image_file("resources/images/five_people.jpg")
# 检测人脸特征点
face_landmarks_list = face_recognition.face_landmarks(image)
# 就是玩
# image = np.zeros(image.shape, np.uint8) + 255
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image, 'RGBA')
for face_landmarks in face_landmarks_list:
    # 眉毛上色
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
    # 嘴唇上色
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)
    # 眼睛上色
    d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
    # 眼线
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
    # 下吧
    # d.line(face_landmarks['chin'], fill=(0, 0, 0, 110), width=5)
# 显示结果
pil_image.show()
