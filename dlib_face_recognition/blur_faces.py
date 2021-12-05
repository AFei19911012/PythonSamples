# -*- coding: utf-8 -*-
"""
 Created on 2021/5/19 15:09
 Filename   : blur_faces.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: https://github.com/ageitgey/face_recognition
# =======================================================
import face_recognition
import cv2

# 加载视频或者从 webcam 中获取
video_path = 'resources/videos/short_hamilton_clip.mp4'
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)
# 缩放系数
scale = 2
while cap.isOpened():
    # 逐帧处理
    ret, frame = cap.read()
    # 处理完后释放，退出
    if not ret:
        cap.release()
        break
    # 尺寸减小加快人脸识别速度
    small_frame = cv2.resize(frame, (0, 0), fx=1/scale, fy=1/scale)
    # 检测当前帧所有的人脸位置
    face_locations = face_recognition.face_locations(small_frame, model="cnn")
    # 显示结果
    for top, right, bottom, left in face_locations:
        # 位置还原
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale
        # 提取脸部图像
        face_image = frame[top:bottom, left:right]
        # 高斯模糊
        face_image = cv2.GaussianBlur(face_image, (19, 19), 20)
        # 替换原图人脸
        frame[top:bottom, left:right] = face_image
    # 显示当前帧
    cv2.imshow('Video', frame)
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
