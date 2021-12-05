# -*- coding: utf-8 -*-
"""
 Created on 2021/5/20 9:19
 Filename   : face_recognition_video_demo.py
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
cap = cv2.VideoCapture('resources/videos/short_hamilton_clip.mp4')
# 统计帧数
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 创建视频流写入对象
vid_writer = cv2.VideoWriter('resources/videos/output.avi', cv2.VideoWriter_fourcc(*'XVID'), 29.97, (640, 360))
# output_movie = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (640, 360))
# fourcc 意为四字符代码（Four-Character Codes）
# cv2.VideoWriter_fourcc('I', '4', '2', '0'), 该参数是 YUV 编码类型，文件名后缀为 .avi
# cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'), 该参数是 MPEG-1 编码类型，文件名后缀为 .avi
# cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 该参数是 MPEG-4 编码类型，文件名后缀为 .avi
# cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'), 该参数是 Ogg Vorbis，文件名后缀为 .ogv
# cv2.VideoWriter_fourcc('F', 'L', 'V', '1'), 该参数是 Flash 视频，文件名后缀为 .flv
# 加载样例图片并识别
lmm_image = face_recognition.load_image_file("resources/images/lin-manuel-miranda.png")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]
al_image = face_recognition.load_image_file("resources/images/alex-lacamoire.png")
al_face_encoding = face_recognition.face_encodings(al_image)[0]
# 已知人脸
known_faces = [lmm_face_encoding, al_face_encoding]
known_names = ['Lin-Manuel Miranda', 'Alex Lacamoire']
# 初始化
face_locations = []
face_encodings = []
face_names = []
frame_number = 0
while True:
    # 逐帧处理
    ret, frame = cap.read()
    frame_number += 1
    # 处理完后释放，退出
    if not ret:
        cap.release()
        break
    # BGR (OpenCV) --> RGB (face_recognition)
    rgb_frame = frame[:, :, ::-1]
    # 检测当前帧人脸
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # 人脸对比
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
        # 两个人
        # name = None
        # if match[0]:
        #     name = "Lin-Manuel Miranda"
        # elif match[1]:
        #     name = "Alex Lacamoire"
        # 多个人
        name = 'unknown'
        if True in match:
            first_match_idx = match.index(True)
            name = known_names[first_match_idx]
        face_names.append(name)
    # 标签和结果
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue
        # 人脸框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # 人名
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    # 写入视频流
    print(f"Writing frame {frame_number} / {length}")
    vid_writer.write(frame)
cv2.destroyAllWindows()
