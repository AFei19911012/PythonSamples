# -*- coding: utf-8 -*-
"""
 Created on 2021/4/15 10:31
 Filename: yolov4_darknet_api.py
 Author  : Taosy
 Zhihu   : https://www.zhihu.com/people/1105936347
 Github  : https://github.com/AFei19911012
 Describe: use darknet_cpp_dll.dll with darknet.py
"""

# Source: https://github.com/AlexeyAB/darknet

import cv2 as cv
import darknet
import numpy as np
import random


def get_colors(names):
    return {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in names}


def image_detection(image, net, class_names, thresh):
    width = image.shape[1]
    height = image.shape[0]
    darknet_image = darknet.make_image(width, height, 3)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_resized = cv.resize(image_rgb, (width, height), interpolation=cv.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(net, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    return detections


def draw_box(image, detections, colors):
    for label, confidence, box in detections:
        x, y, w, h = box
        left = int(x - w / 2)
        right = int(x + w / 2)
        top = int(y - h / 2)
        bottom = int(y + h / 2)
        # Draw box
        cv.rectangle(image, (left, top), (right, bottom), colors[label], 2)
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv.putText(image, label, (left, top + label_size[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    return image


if __name__ == '__main__':
    # Initialization
    names_path = '../config/coco.names'
    cfg_path = '../config/coco.cfg'
    data_path = '../config/coco.data'
    weights_path = '../weights/yolov4.weights'
    image_path = '../images/xiyou.jpg'
    video_path = '../videos/run.mp4'
    video_save = '../videos/run.avi'
    threshold = 0.5
    # Initialize yolov4 network
    network, classes, class_colors = darknet.load_network(cfg_path, data_path, weights_path, 1)
    colors = get_colors(classes)
    winName = 'Deep learning object detection in YOLO'
    # image_path = ''
    if image_path:
        cap = cv.VideoCapture(image_path)
    else:
        cap = cv.VideoCapture(video_path)
        # vid_writer = cv.VideoWriter(video_save, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
        #                             (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        #                              round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    state = True
    while state:
        state, frame = cap.read()
        if not state:
            break
        detections = image_detection(frame, network, classes, threshold)
        draw_box(frame, detections, colors)
        cv.imshow(winName, frame)
        if image_path:
            cv.waitKey(0)
        else:
            cv.waitKey(1)
        # Save
        # if not image_path:
        #     vid_writer.write(frame.astype(np.uint8))
    cap.release()
