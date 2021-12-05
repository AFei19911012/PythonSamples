# -*- coding: utf-8 -*-
"""
 Created on 2021/5/20 17:25
 Filename   : face_recognition_svm.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: https://github.com/ageitgey/face_recognition
# =======================================================
import face_recognition
from sklearn import svm
import os

# 文件夹结构
"""
test_image.jpg
<train_dir>/
    <person_1>/
        person_1_face-1.jpg
        person_1_face-2.jpg
        .
        person_1_face-n.jpg
   <person_2>/
        person_2_face-1.jpg
        person_2_face-2.jpg
        .
        person_2_face-n.jpg
    .
    person_n/
        person_n_face-1.jpg
        person_n_face-2.jpg
        .
        <person_n_face-n>.jpg
"""
# 训练 SVC 分类器
# 训练数据为 face encodings，标签为名字
encodings = []
names = []
# 训练集目录
base_dir = 'resources/images/train_dir/'
train_dir = os.listdir(base_dir)
# 每个人循环
for person in train_dir:
    pix = os.listdir(base_dir + person)
    # 每个人有多张图片
    for person_img in pix:
        # 获取 face encodings
        face = face_recognition.load_image_file(base_dir + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)
        # 仅一个人脸的图片能用于训练
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            # 训练集添加数据
            encodings.append(face_enc)
            names.append(person)
        else:
            print('忽略：' + person + "/" + person_img)

# 创建 SVC 分类器并训练
clf = svm.SVC(gamma='scale')
clf.fit(encodings, names)
# 加载未知图像
test_image = face_recognition.load_image_file('resources/images/test_image.jpg')
# 检测人脸，基于 HOG 模型
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("检测到人脸数: ", no)
# 利用训练好的分类器预测人脸
print("检测到:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    print(*name)

'''
忽略：Jay/3.jpg
检测到人脸数:  3
检测到:
ChenHe
Jay
KunLin
'''