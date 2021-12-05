# -*- coding: utf-8 -*-
"""
 Created on 2021/4/6 10:14
 Author: Taosy
 E-mail: 1105936347@qq.com
 Zhihu : https://www.zhihu.com/people/1105936347
 Github: https://github.com/AFei19911012
 Describe: use darknet_cpp_dll.dll without darknet.py
"""

# Source: https://github.com/AlexeyAB/darknet

import os
import cv2 as cv
import ctypes
from ctypes import *
import random
import tempfile
import numpy as np

MAX_OBJECTS = 1000
FONT_SIMPLEX = cv.FONT_HERSHEY_SIMPLEX
LINE_AA = cv.LINE_AA
# opencv: B G R
COLOR_R = (0, 0, 255)
COLOR_G = (0, 255, 0)
COLOR_B = (255, 0, 0)
COLOR_K = (0, 0, 0)
COLOR_W = (255, 255, 255)
COLOR_M = (255, 0, 255)
COLOR_Y = (0, 255, 255)


class BBOX(Structure):
    _fields_ = [
        ("x", c_uint),
        ("y", c_uint),
        ("w", c_uint),
        ("h", c_uint),
        ("prob", c_float),
        ("obj_id", c_int),
        ("track_id", c_int),
        ("frames_counter", c_int),
        ("x_3d", c_float),
        ("y_3d", c_float),
        ("z_3d", c_float)
    ]


class BBOX_CONTAINER(Structure):
    _fields_ = [
        ("candidates", BBOX * MAX_OBJECTS)
    ]


def get_classes(classes_file):
    with open(classes_file, 'rt') as f:
        names = f.read().rstrip('\n').split('\n')
    return names


def get_colors(names):
    return {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in names}


def remove_low_confidence(container, num, threshold):
    box_list = []
    for idx in range(num):
        if container.candidates[idx].prob > threshold:
            box_list.append(container.candidates[idx])
    box_list = sorted(box_list, key=lambda x: x.prob, reverse=True)
    return box_list


def draw_image(image, classes, predictions, colors):
    for i in range(len(predictions)):
        obj_id = predictions[i].obj_id
        x = predictions[i].x
        y = predictions[i].y
        w = predictions[i].w
        h = predictions[i].h
        prob = predictions[i].prob
        label = classes[obj_id]
        print(f'{obj_id:2d} {label.rjust(10)} {prob:.2f} {x:3d} {y:3d} {w:3d} {h:3d}')
        # Draw a bounding box.
        # (255, 178, 50)
        cv.rectangle(image, (x, y), (x + w, y + h), colors[label], 2)
        # Display the label at the top of the bounding box
        label = f'{label}: {prob:.2f}'
        label_size, base_line = cv.getTextSize(label, FONT_SIMPLEX, 0.5, 2)
        # cv.rectangle(image, (x+1, y+1), (x+label_size[0], y+label_size[1]+base_line), COLOR_W, cv.FILLED)
        # cv.putText(image, label, (x+1, y+label_size[1]), FONT_SIMPLEX, 0.5, (0, 0, 0), 1, LINE_AA)
        cv.putText(image, label, (x, y + label_size[1]), FONT_SIMPLEX, 0.5, COLOR_R, 2, LINE_AA)


if __name__ == '__main__':
    # Initialization
    names_path = '../config/coco.names'
    cfg_path = '../config/coco.cfg'
    weights_path = '../weights/yolov4.weights'
    image_path = '../images/xiyou.jpg'
    video_path = '../videos/run.mp4'
    video_save = '../videos/run_.mp4'
    confidence_threshold = 0.5
    # Load dll (gpu or cpu)
    # win_lib = ctypes.windll.LoadLibrary('yolo_cpp_dll.dll')
    win_lib = CDLL('yolo_cpp_dll.dll', RTLD_GLOBAL)
    # Initialize yolov4 network
    win_lib.init(cfg_path.encode('utf8'), weights_path.encode('utf8'), 0)
    # Get classes
    classes = get_classes(names_path)
    colors = get_colors(classes)
    winName = 'Deep learning object detection in YOLO'
    # image_path = ''
    if image_path:
        cap = cv.VideoCapture(image_path)
    else:
        cap = cv.VideoCapture(video_path)
        vid_writer = cv.VideoWriter(video_save, cv.VideoWriter_fourcc(*'mp4v'), 30,
                                    (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                     round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    state = True
    while state:
        state, frame = cap.read()
        if not state:
            break
        box_container = BBOX_CONTAINER()
        if image_path:
            temp_name = image_path
        else:
            # video detection: a stupid method
            temp = tempfile.NamedTemporaryFile()
            temp_name = temp.name + '.jpg'
            cv.imwrite(temp_name, frame)
        # file path needed
        class_count = win_lib.detect_image(temp_name.encode('utf8'), byref(box_container))
        # remove the low confidence objection
        predictions = remove_low_confidence(box_container, class_count, confidence_threshold)
        if not image_path:
            os.remove(temp_name)
            temp.close()
        # Draw
        draw_image(frame, classes, predictions, colors)
        cv.imshow(winName, frame)
        if image_path:
            cv.waitKey(0)
        else:
            cv.waitKey(1)
        # Save
        if not image_path:
            vid_writer.write(frame.astype(np.uint8))
    cap.release()
    # Release
    win_lib.dispose()
