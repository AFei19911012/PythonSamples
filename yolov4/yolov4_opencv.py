# -*- encoding: utf-8 -*-
"""
 Created on 2021/4/14 21:27
 Author     : Taosy
 E-mail     : 1105936347@qq.com
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: YOLO-OpenCV
"""

# Source: https://github.com/spmallick/learnopencv/tree/master/ObjectDetection-YOLO

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import random


def get_classes(classes_path):
    with open(classes_path, 'rt') as fid:
        classes = fid.read().rstrip('\n').split('\n')
    return classes


def get_colors(names):
    return {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in names}


# Get the names of the output layers
def get_output_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def draw_image(image, predictions, classes, colors):
    for x, y, w, h, prob, obj_id in predictions:
        label = classes[obj_id]
        # print(f'{obj_id:2d} {label.rjust(10)} {prob:.2f} {x:3d} {y:3d} {w:3d} {h:3d}')
        # Draw a bounding box.
        # (255, 178, 50)
        cv.rectangle(image, (x, y), (x + w, y + h), colors[label], 2)
        # Display the label at the top of the bounding box
        label = f'{label}: {prob:.2f}'
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # cv.rectangle(image, (x+1, y+1), (x+label_size[0], y+label_size[1]+base_line), COLOR_W, cv.FILLED)
        # cv.putText(image, label, (x+1, y+label_size[1]), FONT_SIMPLEX, 0.5, (0, 0, 0), 1, LINE_AA)
        cv.putText(image, label, (x, y + label_size[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
    return frame


# Remove the bounding boxes with low confidence using non-maximum suppression
def detect_image(frame, net, input_width, input_height, confidence_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (input_width, input_height), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    out_layers = net.forward(get_output_names(net))
    # Candidates
    class_ids = []
    confidences = []
    boxes = []
    for out in out_layers:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    # Perform non maximum suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    predictions = []
    for idx in indices:
        i = idx[0]
        box = boxes[i]
        predictions.append([box[0], box[1], box[2], box[3], confidences[i], class_ids[i]])
    return predictions


if __name__ == '__main__':
    # Initialization
    confidence_threshold = 0.5
    nms_threshold = 0.4
    input_width = 320
    input_height = 320
    device = 'gpu'
    names_path = "config/helmet.names"
    cfg_path = "config/helmet.cfg"
    weights_path = "weights/helmet.weights"
    video_path = 'videos/helmet.mp4'
    video_save_path = "../videos/result.avi"
    # Load names
    classes = get_classes(names_path)
    colors = get_colors(classes)
    # Create net
    net = cv.dnn.readNetFromDarknet(cfg_path, weights_path)
    # cpu or gpu
    if device == 'gpu':
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print('Using GPU device.')
    else:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        print('Using CPU device.')
    # Process inputs
    winName = 'Deep learning object detection in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    # Open the video
    cap = cv.VideoCapture(video_path)
    image_width = round(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    image_height = round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cv.resizeWindow(winName, image_width, image_height)
    cv.moveWindow(winName, 50, 50)
    # Initialize the video writer
    # vid_writer = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (image_width, image_height))
    while cv.waitKey(1) < 0:
        state, frame = cap.read()
        if not state:
            break
        # Detect
        predictions = detect_image(frame, net, input_width, input_height, confidence_threshold, nms_threshold)
        # Draw
        frame = draw_image(frame, predictions, classes, colors)
        t, _ = net.getPerfProfile()
        label = f'Inference time: {t * 1000.0 / cv.getTickFrequency():.2f} ms'
        cv.putText(frame, label, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
        # Save
        # vid_writer.write(frame.astype(np.uint8))
        cv.imshow(winName, frame)
    cap.release()
