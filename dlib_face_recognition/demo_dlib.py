# -*- coding: utf-8 -*-
"""
 Created on 2021/5/17 15:38
 Filename   : demo_dlib.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: https://github.com/davisking/dlib
# =======================================================
import cv2 as cv
import dlib
import numpy as np


def main():
    # 读取图片
    image_path = r"resources\images\obama2.jpg"
    image = cv.imread(image_path)
    # image = cv.resize(image, [550, 800])
    # 人脸检测
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"models\shape_predictor_68_face_landmarks.dat")
    # 检测出的人脸数
    faces = detector(image, 0)
    # 绘制人脸区域，每个 face 是个 rect
    for rect in faces:
        # 绘制人脸边框
        cv.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 0, 255))
        # 检测人脸标定点，共有 68 个点
        face = predictor(image, rect)
        # # 绘制人脸标定点
        # for i in range(face.num_parts):
        #     pos = (face.part(i).x, face.part(i).y)
        #     # 标定点
        #     cv.circle(image, pos, 1, color=(255, 0, 0))
        #     # 标定点序号
        #     # cv.putText(image, str(i + 1), pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        # 画轮廓线
        # 脸庞轮廓
        face_pos = []
        for i in range(17):
            face_pos.append((face.part(i).x, face.part(i).y))
        pts = np.array(face_pos, np.int32)
        cv.polylines(image, [pts], False, (0, 255, 255), 1)
        # 眉毛
        face_pos = []
        for i in range(17, 22):
            face_pos.append((face.part(i).x, face.part(i).y))
        pts = np.array(face_pos, np.int32)
        cv.polylines(image, [pts], False, (0, 255, 255), 1)
        face_pos = []
        for i in range(22, 27):
            face_pos.append((face.part(i).x, face.part(i).y))
        pts = np.array(face_pos, np.int32)
        cv.polylines(image, [pts], False, (0, 255, 255), 1)
        # 鼻子
        face_pos = []
        for i in range(27, 36):
            face_pos.append((face.part(i).x, face.part(i).y))
        face_pos.append((face.part(30).x, face.part(30).y))
        pts = np.array(face_pos, np.int32)
        cv.polylines(image, [pts], False, (0, 255, 255), 1)
        # 眉毛
        face_pos = []
        for i in range(36, 42):
            face_pos.append((face.part(i).x, face.part(i).y))
        pts = np.array(face_pos, np.int32)
        cv.polylines(image, [pts], True, (0, 255, 255), 1)
        face_pos = []
        for i in range(42, 48):
            face_pos.append((face.part(i).x, face.part(i).y))
        pts = np.array(face_pos, np.int32)
        cv.polylines(image, [pts], True, (0, 255, 255), 1)
        # 眉毛
        face_pos = []
        for i in range(36, 42):
            face_pos.append((face.part(i).x, face.part(i).y))
        pts = np.array(face_pos, np.int32)
        cv.polylines(image, [pts], True, (0, 255, 255), 1)
        # 嘴唇
        face_pos = []
        for i in range(48, 60):
            face_pos.append((face.part(i).x, face.part(i).y))
        pts = np.array(face_pos, np.int32)
        cv.polylines(image, [pts], True, (0, 255, 255), 1)
        face_pos = []
        for i in range(60, 68):
            face_pos.append((face.part(i).x, face.part(i).y))
        pts = np.array(face_pos, np.int32)
        cv.polylines(image, [pts], True, (0, 255, 255), 1)
    # 显示结果图
    cv.imshow("face-dlib", image)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
