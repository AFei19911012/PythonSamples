# -*- encoding: utf-8 -*-
"""
 Created on 2021/4/14 21:27
 Author     : Taosy
 E-mail     : 1105936347@qq.com
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: YOLO-OpenCV
"""

import cv2
import numpy as np
import random


class YOLOv4DetectorCV:
    """ YOLOv4-OpenCV 图像检测 """

    def __init__(self, weights='weights/yolov4.weights', class_file='config/coco.names', cfg='config/coco.cfg', conf_thres=0.25, nms_thres=0.45, imgsz=512,
                 device='gpu'):
        """ Initialization """
        self.weights = weights
        self.class_file = class_file
        self.cfg = cfg
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        """ 320 速度更快 """
        self.imgsz = imgsz
        """ gpu 或者 cpu """
        self.device = device
        """ net """
        self.net = self.init_net()
        """ names """
        self.classes = self.get_names()
        """ colors """
        self.colors = self.get_colors()

    def init_net(self):
        """ 初始化网络 """
        """ Give the configuration and weight files for the model and load the network using them """
        net = cv2.dnn.readNetFromDarknet(self.cfg, self.weights)
        """ 设置 gpu 或者 cpu """
        if self.device == 'cpu':
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print('Using CPU device.')
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print('Using GPU device.')
        return net

    def get_names(self):
        """ 读取类别文件 """
        with open(self.class_file, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
            return classes

    def get_colors(self):
        """ 生成颜色 """
        return {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in self.classes}

    def get_output_names(self):
        """ Get the names of the output layers """
        """ Get the names of all the layers in the network """
        layersNames = self.net.getLayerNames()
        """ Get the names of the output layers, i.e. the layers with unconnected outputs """
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def draw_image(self, frame, classId, conf, left, top, right, bottom):
        """ Draw the predicted bounding box """
        """ line thickness """
        tl = 2 or round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1
        c1 = (left, top)
        c2 = (right, bottom)
        color = self.colors[self.classes[classId]]
        """ Draw a bounding box """
        cv2.rectangle(frame, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        """ Get the label for the class name and its confidence """
        label_show = f'{self.classes[classId]}_{conf:.2f}'
        """ font thickness """
        tf = max(tl - 1, 1)
        """ Display the label at the top of the bounding box """
        t_size = cv2.getTextSize(label_show, cv2.FONT_HERSHEY_SIMPLEX, fontScale=tl/3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        """ filled """
        cv2.rectangle(frame, c1, c2, color, cv2.FILLED, cv2.LINE_AA)
        cv2.putText(frame, label_show, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def non_max_suppression(self, frame, outs):
        """ Remove the bounding boxes with low confidence using non-maxima suppression """
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        """ Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores. Assign the box's class label as 
        the class with the highest score """
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.conf_thres:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        """ Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences """
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.nms_thres)
        bbox_container = []
        for i in indices:
            i = i[0]
            [left, top, width, height] = boxes[i]
            self.draw_image(frame, classIds[i], confidences[i], left, top, left + width, top + height)
            bbox = {'class': self.classes[classIds[i]], 'confidence': round(confidences[i], 2), 'box': [left, top, left + width, top + height]}
            bbox_container.append(bbox)
        return bbox_container

    def __call__(self, image, *args, **kwargs):
        """ main function """
        frame = image.copy()
        """ Create a 4D blob from an image """
        blob = cv2.dnn.blobFromImage(frame, 1/255, (self.imgsz, self.imgsz), [0, 0, 0], 1, crop=False)
        """ Sets the input to the network """
        self.net.setInput(blob)
        """ Runs the forward pass to get output of the output layers """
        output_layers = self.get_output_names()
        outs = self.net.forward(output_layers)
        """ Remove the bounding boxes with low confidence """
        bbox_container = self.non_max_suppression(frame, outs)
        """ Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in 
        layersTimes) """
        t, _ = self.net.getPerfProfile()
        label = f'Inference time: {t * 1000.0 / cv2.getTickFrequency():.2f} ms'
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return frame, bbox_container


def detect_demo():
    image_name = 'dog.jpg'
    image_path = f'images/{image_name}'
    image_save_path = f'images/_{image_name}'
    video_name = 'run.mp4'
    video_path = f'videos/{video_name}'
    video_save_path = f'videos/_{video_name}'
    # image_path = ''
    if image_path:
        cap = cv2.VideoCapture(image_path)
    else:
        cap = cv2.VideoCapture(video_path)
        fource = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(video_save_path, fource, 30, (width, height))
    """ 图像检测器 """
    yolov4_detector = YOLOv4DetectorCV()
    window_name = 'YOLOv4 detector'
    while True:
        state, frame = cap.read()
        if not state:
            break
        image_result, bbox_container = yolov4_detector(frame)
        for info in bbox_container:
            print(info)
        print('---')
        """ 显示图像 """
        cv2.imshow(window_name, image_result)
        if image_path:
            cv2.imwrite(image_save_path, image_result)
            cv2.waitKey(3000)
        else:
            vid_writer.write(image_result)
            cv2.waitKey(1)
        """ 点 x 退出 """
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 1:
            break
    cap.release()
    if not image_path:
        vid_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_demo()
