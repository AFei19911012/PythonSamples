# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import os
import math

from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from yolov5_detector import YOLOv5Detector

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 定义一些变量
threshold_distance = 300
path_normal_walker = []
path_walkers = []


def cal_distance(track_id):
    """ 计算距离 """
    dist = 0
    if len(path_walkers) >= track_id:
        length = min(len(path_normal_walker), len(path_walkers[track_id - 1]))
        for i in range(0, length):
            x_len = (path_walkers[track_id-1][i][0] - path_normal_walker[i][0]) * (path_walkers[track_id-1][i][0] - path_normal_walker[i][0])
            y_len = (path_walkers[track_id-1][i][1] - path_normal_walker[i][1]) * (path_walkers[track_id-1][i][1] - path_normal_walker[i][1])
            dist = dist + math.sqrt(x_len + y_len) / length
    return dist


def draw_image(image, bbox_container, obj_ids):
    """ 绘制人标签 """
    """ 线宽 """
    tl = 2 or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    for i, bbox in enumerate(bbox_container):
        label = bbox['class']
        x1, y1, x2, y2 = bbox['box']
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, (255, 0, 0), thickness=tl, lineType=cv2.LINE_AA)
        """ 字体宽度 """
        tf = max(tl - 1, 1)
        label_show = f'{label}-{obj_ids[i]}'
        """ 判断是否异常行为 """
        distance = cal_distance(obj_ids[i])
        if distance > threshold_distance:
            label_show = label_show + '-NG '
        t_size = cv2.getTextSize(label_show, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        """ filled """
        cv2.rectangle(image, c1, c2, (255, 0, 0), cv2.FILLED, cv2.LINE_AA)
        cv2.putText(image, label_show, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    """ 画路径 """
    if path_normal_walker:
        pts = np.array(path_normal_walker, np.int32)
        cv2.polylines(image, [pts], False, (0, 0, 255), 3)


def xyxy_to_xywh(box):
    """ 目标框转换 """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x_c = int(x1 + w / 2)
    y_c = int(y1 + h / 2)
    return [x_c, y_c, w, h]


def cut_bbox_container(bbox_container):
    """ 只保留人信息 """
    container = []
    for bbox in bbox_container:
        label = bbox['class']
        confidence = bbox['confidence']
        box = bbox['box']
        if label == 'person':
            container.append({'class': label, 'confidence': confidence, 'box': box})
    return container


def main():
    video_name = 'walker_detection.mp4'
    cap = cv2.VideoCapture(f'data/videos/{video_name}')
    fource = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(f'runs/track/{video_name}.mp4', fource, 30, (width, height))
    """ yolov5 目标检测器 """
    yolov5_detector = YOLOv5Detector()
    """ deepsort 追踪器 """
    cfg = get_config()
    cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE,
                        n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    window_name = 'Abnormal walker detection'
    while True:
        state, frame = cap.read()
        if not state:
            break
        """ 检测目标 """
        image, bbox_container = yolov5_detector(frame)
        """ 仅保留人信息"""
        bbox_container = cut_bbox_container(bbox_container)
        """ 初始化一些变量 """
        xywh_bboxs = []
        labels = []
        confs = []
        for bbox in bbox_container:
            xywh_bboxs.append(xyxy_to_xywh(bbox['box']))
            labels.append(bbox['class'])
            confs.append(bbox['confidence'])
        """ detections --> deepsort """
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
        outputs = deepsort.update(xywhs, confss, labels, frame)
        obj_ids = []
        bbox_draw = []
        if len(outputs) > 0:
            for (x1, y1, x2, y2, label, track_id) in outputs:
                bbox_draw.append({'class': label, 'box': [x1, y1, x2, y2]})
                obj_ids.append(track_id)
                # 正常路径
                if track_id == 9:
                    if not path_normal_walker:
                        path_normal_walker.append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))
                    else:
                        if 0.5 * (y1 + y2) < path_normal_walker[-1][1]:
                            path_normal_walker.append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))
                # 其它路径
                while track_id > len(path_walkers):
                    path_walkers.append([])
                path_walkers[track_id - 1].append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))
            """ 绘图显示 """
            draw_image(frame, bbox_draw, obj_ids)
        """ 输出一些信息 """
        for info in bbox_draw:
            print(info)
        print(obj_ids)
        print('---')
        cv2.imshow(window_name, frame)
        vid_writer.write(frame)
        cv2.waitKey(1)
        """ 点 x 退出 """
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 1:
            break
    cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
