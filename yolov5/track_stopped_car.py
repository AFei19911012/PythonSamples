# -*- coding: utf-8 -*-
"""
 Created on 2021/11/22 21:42
 Filename   : track_stopped_car.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import time

import cv2
import numpy as np
import torch
import os


from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from yolov5_detector import YOLOv5Detector

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def draw_image(image, bbox_container, obj_ids):
    """ 绘制车标签 """
    """ 线宽 """
    tl = 2 or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    for i, bbox in enumerate(bbox_container):
        label = bbox['class']
        x1, y1, x2, y2 = bbox['box']
        c1, c2 = (x1, y1), (x2, y2)
        color = (0, 0, 255)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        """ 字体宽度 """
        tf = max(tl - 1, 1)
        label_show = f'{label}{obj_ids[i]}'
        """ 判断行进方向 """

        t_size = cv2.getTextSize(label_show, 0, fontScale=tl/3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        """ filled """
        cv2.rectangle(image, c1, c2, color, cv2.FILLED, cv2.LINE_AA)
        cv2.putText(image, label_show, (c1[0], c1[1] - 2), 0, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_image(image, bbox_container, obj_ids, path_cars, target_point_count):
    """ 绘制结果 """
    """ 线宽 """
    tl = 2 or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

    """ 判断是否滞留 """
    num = 0
    dist_max = 25
    stop_obj = []
    for track_id in obj_ids:
        car = path_cars[track_id - 1]
        """ 满足帧数的目标作判断 """
        if len(car) == target_point_count:
            """ 计算距离是否超过阈值 """
            x0 = car[0][0]
            y0 = car[0][1]
            flag = True
            for x, y in car:
                if (x - x0) * (x - x0) + (y - y0) * (y - y0) > dist_max * dist_max:
                    flag = False
                    break
            if flag:
                num = num + 1
                stop_obj.append(track_id)

    for i, bbox in enumerate(bbox_container):
        label = bbox['class']
        x1, y1, x2, y2 = bbox['box']
        c1, c2 = (x1, y1), (x2, y2)
        """ 正常颜色 """
        color = (0, 0, 255)
        """ 异常颜色 """
        if obj_ids[i] in stop_obj:
            color = (255, 0, 0)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        """ 字体宽度 """
        tf = max(tl - 1, 1)
        label_show = f'{label}{obj_ids[i]}'
        """ 判断行进方向 """

        t_size = cv2.getTextSize(label_show, 0, fontScale=tl/3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        """ filled """
        cv2.rectangle(image, c1, c2, color, cv2.FILLED, cv2.LINE_AA)
        cv2.putText(image, label_show, (c1[0], c1[1] - 2), 0, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    """ 返回滞留数量 """
    return num


def xyxy_to_xywh(box):
    """ 目标框转换 """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x_c = int(x1 + w/2)
    y_c = int(y1 + h/2)
    return [x_c, y_c, w, h]


def cut_bbox_container(bbox_container):
    """ 只保留车信息 """
    container = []
    for bbox in bbox_container:
        label = bbox['class']
        confidence = bbox['confidence']
        box = bbox['box']

        if label in ['car', 'bus', 'truck']:
            """ 再限制一下条件 面积 宽高比例"""
            w = box[2] - box[0]
            h = box[3] - box[1]
            flag = True
            if w * h < 900 or w / h > 4:
                flag = False
            if flag:
                container.append({'class': 'car', 'confidence': confidence, 'box': box})
    return container


def main():
    video_name = '1.avi'
    cap = cv2.VideoCapture(f'data/videos/{video_name}')
    fource = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(f'runs/track/{video_name}.mp4', fource, 30, (width, height))
    """ yolov5 目标检测器 """
    yolov5_detector = YOLOv5Detector(weights='weights/yolov5s.pt', conf_thres=0.6)
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
    window_name = 'Anti Running Car Tracking'

    # 每个目标记录 target_point_count 个点，据此判断目标是否滞留
    target_point_count = 50
    path_cars = []

    while True:
        state, frame = cap.read()
        if not state:
            break
        prev_time = time.time()
        """ 检测目标 """
        image, bbox_container = yolov5_detector(frame)
        """ 仅保留车信息"""
        bbox_container = cut_bbox_container(bbox_container)

        """ 初始化一些变量 """
        xywh_bboxs = []
        labels = []
        confs = []
        for bbox in bbox_container:
            xywh_bboxs.append(xyxy_to_xywh(bbox['box']))
            labels.append(bbox['class'])
            confs.append(bbox['confidence'])
        """ 检测到目标后才有追踪 """
        if labels:
            """ detections --> deepsort """
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)
            outputs = deepsort.update(xywhs, confss, labels, frame)
            obj_ids = []
            bbox_draw = []
            num = 0
            if len(outputs) > 0:
                for (x1, y1, x2, y2, label, track_id) in outputs:
                    bbox_draw.append({'class': label, 'box': [x1, y1, x2, y2]})
                    obj_ids.append(track_id)

                    """ 记录所有目标的路径 每个目标记录点数为 target_point_count """
                    while track_id > len(path_cars):
                        path_cars.append([])
                    path_cars[track_id - 1].append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))

                    """ 超过的点数从首点删除 """
                    while len(path_cars[track_id - 1]) > target_point_count:
                        path_cars[track_id - 1].remove(path_cars[track_id - 1][0])

                """ 绘图显示 """
                num = draw_image(frame, bbox_draw, obj_ids, path_cars, target_point_count)
            """ 输出一些信息 """
            for info in bbox_draw:
                print(info)
            print(obj_ids)
            print('---')

            """ fps"""
            fps = int(1 / (time.time() - prev_time))
            cv2.putText(frame, f'fps={fps} max_id={len(path_cars)}', (10, 40), 0, 1, [0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
            """ 滞留"""
            cv2.putText(frame, f'stop={num}', (10, 80), 0, 1, [0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
            """ 拥堵"""
            if num >= 3:
                cv2.putText(frame, f'crowded{num}', (10, 120), 0, 1, [0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
            else:
                cv2.putText(frame, f'normal', (10, 120), 0, 1, [0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
            """ 车的总数"""
            cv2.putText(frame, f'total number={len(bbox_container)}', (10, 160), 0, 1, [0, 255, 0], thickness=1, lineType=cv2.LINE_AA)

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
