# -*- coding: utf-8 -*-
"""
 Created on 2021/5/19 17:21
 Filename   : demo_sleep_detection.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: https://github.com/ageitgey/face_recognition
# =======================================================
import face_recognition
import cv2
from scipy.spatial import distance as dist

EYES_CLOSED_SECONDS = 5


def main():
    # 从 webcam 中获取视频
    cap = cv2.VideoCapture(0)
    # video_path = 'resources/videos/short_hamilton_clip.mp4'
    # cap = cv2.VideoCapture(video_path)
    # 缩放系数
    scale = 1
    # 连续闭眼帧数
    closed_count = 0
    # 是否处理
    process = True
    while cap.isOpened():
        # 逐帧处理
        ret, frame = cap.read()
        # 处理完后释放，退出
        if not ret:
            cap.release()
            break
        # 缩放图像并转换格式
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process:
            # 检测人脸
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
            for face_landmark in face_landmarks_list:
                # 获取眼部数据
                left_eye = face_landmark['left_eye']
                right_eye = face_landmark['right_eye']
                # 绘图
                cv2.rectangle(small_frame, left_eye[0], right_eye[-1], (255, 0, 0), 2)
                cv2.imshow('Video', small_frame)
                # 计算眼纵横比
                ear_left = get_ear(left_eye)
                ear_right = get_ear(right_eye)
                # 闭眼判断
                closed = ear_left < 0.2 and ear_right < 0.2
                if closed:
                    closed_count += 1
                else:
                    closed_count = 0
                # 连续闭眼超过阈值判断为睡觉
                if closed_count >= EYES_CLOSED_SECONDS:
                    asleep = True
                    while asleep:
                        print("EYES CLOSED")
                        # 等待空格键唤醒
                        if cv2.waitKey(1) == 32:
                            asleep = False
                            print("EYES OPENED")
                    closed_count = 0

        process = not process
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# 计算眼纵横比
def get_ear(eye):
    # 计算欧氏距离：两个纵向、一个横向
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    # 计算眼纵横比
    ear = (A + B) / (2.0 * C)
    return ear


if __name__ == "__main__":
    main()
