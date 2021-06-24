# -*- encoding: utf-8 -*-
"""
 Created on 2021/4/14 11:27
 Filename   : yolov5_tracker.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: 判断叉车+货物出入库
"""

import os
import numpy as np
import darknet
import random
from math import sqrt

import cv2 as cv

MAX_OBJECTS = 1000
# CV property
FONT_SIMPLEX = cv.FONT_HERSHEY_SIMPLEX
LINE_AA = cv.LINE_AA
COLOR_R = (255, 0, 0)
COLOR_G = (0, 255, 0)
COLOR_B = (0, 0, 255)
COLOR_W = (255, 255, 255)
COLOR_K = (0, 0, 0)
COLOR_M = (255, 0, 255)
COLOR_Y = (255, 255, 0)
COLOR_TEXT = (255, 178, 50)
# Trace
MAX_TRACE_LEN = 50
MOVING_LEN = 3
DISTANCE_LEN = 20
# Image size
IMAGE_WIDTH = 1088
IMAGE_HEIGHT = 520
# ROI
ROI_XMIN = 250
ROI_XMAX = 825
ROI_YMIN = 1
ROI_YMAX = 520

# Sort GPU device from 0 according to PCI_BUS_ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Attributes:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.left = 0
        self.right = 0
        self.is_exist = False
        self.is_moving = False
        self.trace_points = []

    # attribute = Attributes(pos, time, is_exist)
    # print(attribute)
    def __str__(self):
        return f'{self.x}, {self.y}'


def get_classes(classes_path):
    with open(classes_path, 'rt') as fid:
        classes = fid.read().rstrip('\n').split('\n')
    return classes


def get_colors(names):
    return {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in names}


def distance_cal(trace_points):
    if len(trace_points) > DISTANCE_LEN:
        pt1 = trace_points[-DISTANCE_LEN]
    else:
        pt1 = trace_points[0]
    pt2 = trace_points[-1]
    return sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))


def image_detection(image, network, class_names, thresh):
    width = image.shape[1]
    height = image.shape[0]
    darknet_image = darknet.make_image(width, height, 3)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_resized = cv.resize(image_rgb, (width, height), interpolation=cv.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    return detections


def draw_trace(image, detections, colors, show_rect=False):
    for label, confidence, box in detections:
        x, y, w, h = box
        left = int(x - w / 2)
        right = int(x + w / 2)
        top = int(y - h / 2)
        bottom = int(y + h / 2)
        # Draw ROI
        cv.rectangle(image, (ROI_XMIN, ROI_YMIN), (ROI_XMAX, ROI_YMAX), COLOR_R, 3)
        # Draw box
        if show_rect:
            cv.rectangle(image, (left, top), (right, bottom), colors[label], 2)
            label_size, base_line = cv.getTextSize(label, FONT_SIMPLEX, 0.5, 2)
            cv.putText(image, label, (left, top + label_size[1]), FONT_SIMPLEX, 0.5, COLOR_R, 2, LINE_AA)
        # Draw trace
        if Target_Objections[label].trace_points:
            pts = np.array(Target_Objections[label].trace_points, np.int32)
            cv.polylines(image, [pts], False, COLOR_B, 5)
    return image


def refresh_attributes(detections):
    for name, prob, box in detections:
        x, y, w, h = box
        Target_Objections[name].x = int(x)
        Target_Objections[name].y = int(y)
        Target_Objections[name].left = int(x - w / 2)
        Target_Objections[name].right = int(x + w / 2)
        Target_Objections[name].is_exist = True
        Target_Objections[name].trace_points.append([x, y])
        # Trace range
        while len(Target_Objections[name].trace_points) > MAX_TRACE_LEN:
            Target_Objections[name].trace_points.pop(0)
        # Moving or not
        distance = distance_cal(Target_Objections[name].trace_points)
        if distance > MOVING_LEN:
            Target_Objections[name].is_moving = True
        else:
            Target_Objections[name].is_moving = False
        # Cargo moving --> forklift moving
        if Target_Objections['cargo'].is_moving:
            Target_Objections['forklift'].is_moving = True

    return Target_Objections


def get_direction():
    forklift_l = Target_Objections['forklift'].left
    forklift_r = Target_Objections['forklift'].right
    cargo_l = Target_Objections['cargo'].left
    cargo_r = Target_Objections['cargo'].right
    dist1 = abs(forklift_r - cargo_l)
    dist2 = abs(forklift_l - cargo_r)
    direction = ''
    if Target_Objections['cargo'].is_moving:
        x = Target_Objections['cargo'].x
        if dist1 < dist2:
            direction = '-->'
            if x > ROI_XMAX:
                direction = direction + 'out'
            elif x < ROI_XMIN:
                direction = direction + 'in'
            else:
                direction = direction + 'loading'
        else:
            direction = '<--'
            if x < ROI_XMIN:
                direction = direction + ' out'
            elif x > ROI_XMAX:
                direction = direction + ' in'
            else:
                direction = direction + 'loading'
    return direction


def show_state(image, detections):
    label = []
    for name, prob, box in detections:
        if Target_Objections[name].is_moving:
            label.append(f'{name}: moving')
        else:
            label.append(f'{name}: stopped')
    label = sorted(label)
    label = str(label) + get_direction()
    cv.putText(image, str(label), (5, 40), FONT_SIMPLEX, 1, COLOR_B, 2, LINE_AA)
    return image


if __name__ == '__main__':
    # Initialization
    names_path = '../config/forklift.names'
    cfg_path = '../config/forklift.cfg'
    data_path = '../config/forklift.data'
    weighs_path = '../weights/forklift.weights'
    video_path = '../videos/forklift.mp4'
    threshold = 0.5
    # Initialize yolov4 network
    net, names, colors = darknet.load_network(cfg_path, data_path, weighs_path, 1)
    winName = 'Deep learning object detection in YOLO'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    cv.resizeWindow(winName, IMAGE_WIDTH, IMAGE_HEIGHT)
    cv.moveWindow(winName, 100, 100)
    while True:
        # Initialization
        Target_Objections = {'forklift': Attributes(), 'cargo': Attributes()}
        cap = cv.VideoCapture(video_path)
        while cv.waitKey(1) < 0:
            state, frame = cap.read()
            if not state:
                break
            # Detect
            detections = image_detection(frame, net, names, threshold)
            # Draw trace
            frame = draw_trace(frame, detections, colors, True)
            # Refresh attributes
            Target_Objections = refresh_attributes(detections)
            frame = show_state(frame, detections)
            # Show
            cv.imshow(winName, frame)
        cap.release()
        # if cv.waitKey(0) == 32:
        #     break
    cv.destroyAllWindows()
