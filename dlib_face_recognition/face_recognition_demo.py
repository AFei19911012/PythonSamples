# -*- coding: utf-8 -*-
"""
 Created on 2021/5/17 19:59
 Filename   : face_recognition_demo.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: https://github.com/ageitgey/face_recognition
# =======================================================
import face_recognition
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 加载数据到 numpy 矩阵
image = face_recognition.load_image_file("resources/images/obama.jpg")
plt.figure('face-recognition')
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')

# 检测图像中所有的人脸所在矩形区域 (top, right, bottom, left)
face_locations = face_recognition.face_locations(image)
for top, right, bottom, left in face_locations:
    cv.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 5)

# 检测图像中所有人脸特征：下巴、左眼眉毛、右眼眉毛、鼻梁、鼻尖、左眼、右眼、上唇、下唇
face_landmarks_list = face_recognition.face_landmarks(image)
# 背景调成黑色，就是玩
image = np.zeros(image.shape, np.uint8)
for landmarks in face_landmarks_list:
    chin = landmarks['chin']
    left_eyebrow = landmarks['left_eyebrow']
    right_eyebrow = landmarks['right_eyebrow']
    nose_bridge = landmarks['nose_bridge']
    nose_tip = landmarks['nose_tip']
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    top_lip = landmarks['top_lip']
    bottom_lip = landmarks['bottom_lip']
    pts = np.array(chin, np.int32)
    cv.polylines(image, [pts], False, (255, 255, 255), 5)
    pts = np.array(left_eyebrow, np.int32)
    cv.polylines(image, [pts], False, (255, 255, 255), 5)
    pts = np.array(right_eyebrow, np.int32)
    cv.polylines(image, [pts], False, (255, 255, 255), 5)
    pts = np.array(nose_bridge, np.int32)
    cv.polylines(image, [pts], False, (255, 255, 255), 5)
    pts = np.array(nose_tip, np.int32)
    cv.polylines(image, [pts], False, (255, 255, 255), 5)
    pts = np.array(left_eye, np.int32)
    cv.polylines(image, [pts], False, (255, 255, 255), 5)
    pts = np.array(right_eye, np.int32)
    cv.polylines(image, [pts], False, (255, 255, 255), 5)
    pts = np.array(top_lip, np.int32)
    cv.polylines(image, [pts], False, (255, 255, 255), 5)
    pts = np.array(bottom_lip, np.int32)
    cv.polylines(image, [pts], False, (255, 255, 255), 5)

plt.subplot(1, 2, 2)
plt.imshow(image)
plt.axis('off')
plt.show()

# 人脸对比
# 检测已知图像的 face encoding
obama = face_recognition.load_image_file("resources/images/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama)[0]
biden = face_recognition.load_image_file("resources/images/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden)[0]
# 已知人脸库
face_libs = [obama_face_encoding, biden_face_encoding]
face_names = ['obama', 'biden']

# 单目标对比
image = face_recognition.load_image_file('resources/images/obama2.jpg')
unknown_face_encoding = face_recognition.face_encodings(image)[0]
# 对比，默认 tolerance = 0.6
matches = face_recognition.compare_faces(face_libs, unknown_face_encoding, tolerance=0.6)
# 如果有匹配的，选第一个即可
name = 'unknown'
if True in matches:
    first_match_idx = matches.index(True)
    name = face_names[first_match_idx]
top, right, bottom, left = face_recognition.face_locations(image)[0]
# 画人脸框
cv.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 5)
# 画标签
cv.rectangle(image, (left, bottom + 40), (right, bottom), (0, 0, 255), cv.FILLED)
cv.putText(image, name, (left, bottom + 35), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv.LINE_AA)
plt.figure('face-compare')
plt.imshow(image)
plt.axis('off')
plt.show()

# 多目标对比
# 待检测图像中所有的 face encodings
image = face_recognition.load_image_file('resources/images/obama_and_biden.jpg')
unknown_face_locations = face_recognition.face_locations(image)
unknown_face_encodings = face_recognition.face_encodings(image, unknown_face_locations)
for (top, right, bottom, left), unknown_face_encoding in zip(unknown_face_locations, unknown_face_encodings):
    # 检查是否和人脸库匹配
    matches = face_recognition.compare_faces(face_libs, unknown_face_encoding)
    name = 'unknown'
    # 如果有匹配的，选第一个即可
    # if True in matches:
    #     first_match_idx = matches.index(True)
    #     name = face_names[first_match_idx]
    # 或者最小距离法
    face_distances = face_recognition.face_distance(face_libs, unknown_face_encoding)
    best_match_idx = np.argmin(face_distances)
    if matches[best_match_idx]:
        name = face_names[best_match_idx]
    # 画人脸框
    cv.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 5)
    # 画标签
    cv.rectangle(image, (left, bottom + 40), (right, bottom), (0, 0, 255), cv.FILLED)
    cv.putText(image, name, (left, bottom + 35), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv.LINE_AA)
plt.figure('faces-compare')
plt.imshow(image)
plt.axis('off')
plt.show()
