# -*- coding: utf-8 -*-
"""
 Created on 2021/5/19 15:58
 Filename   : demo_find_faces_in_batches.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: https://github.com/ageitgey/face_recognition
# =======================================================
import face_recognition
import cv2

# 加载视频
cap = cv2.VideoCapture("resources/videos/short_hamilton_clip.mp4")
# 初始化
frames = []
frame_count = 0
while cap.isOpened():
    # 逐帧处理
    ret, frame = cap.read()
    # 处理完后释放，退出
    if not ret:
        cap.release()
        break
    # BGR (OpenCV 使用) --> RGB ( face_recognition 使用)
    frame = frame[:, :, ::-1]
    # 每一帧保存到列表
    frame_count += 1
    frames.append(frame)
    # 每 64 帧处理一次，128 帧 我电脑配置报错，number_of_times_to_upsample=1 也报错
    if len(frames) == 64:
        batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)
        # 打印 64 帧中所有检测出来的人脸信息
        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
            number_of_faces_in_frame = len(face_locations)
            frame_number = frame_count - 64 + frame_number_in_batch
            print(f"第 {frame_number+1} 帧检测到 {number_of_faces_in_frame} 个人脸：")
            for idx, face_location in enumerate(face_locations):
                # 打印人脸位置
                top, right, bottom, left = face_location
                print(f"  人脸 {idx+1} 位置 Top: {top:4d}, Left: {left:4d}, Bottom: {bottom:4d}, Right: {right:4d}")
        # 清空待下一次批处理
        frames = []
cv2.destroyAllWindows()
'''
第 1 帧检测到 0 个人脸：
第 2 帧检测到 0 个人脸：
第 3 帧检测到 0 个人脸：
第 4 帧检测到 0 个人脸：
第 5 帧检测到 0 个人脸：
...
第 256 帧检测到 3 个人脸：
  人脸 1 位置 Top:   50, Left:  312, Bottom:  145, Right:  407
  人脸 2 位置 Top:   92, Left:  164, Bottom:  171, Right:  243
  人脸 3 位置 Top:  100, Left:  380, Bottom:  179, Right:  459
'''
