# -*- coding: utf-8 -*-
"""
 Created on 2021/6/23 16:55
 Filename   : yolov5_tracker.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: 人车检测追踪，参考源码：https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git
"""

# =======================================================
import cv2
import torch

from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from detector import YOLOv5Detector


def draw_image(image, bbox_container, obj_ids):
    """ 绘制人车标签 """
    """ 线宽 """
    tl = 2 or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    for i, bbox in enumerate(bbox_container):
        label = bbox['class']
        x1, y1, x2, y2 = bbox['box']
        c1, c2 = (x1, y1), (x2, y2)
        if label == 'person':
            color = (255, 0, 0)
        elif label == 'car':
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        """ 字体宽度 """
        tf = max(tl - 1, 1)
        label_show = f'{label}-{obj_ids[i]}'
        t_size = cv2.getTextSize(label_show, 0, fontScale=tl/3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        """ filled """
        cv2.rectangle(image, c1, c2, color, cv2.FILLED, cv2.LINE_AA)
        cv2.putText(image, label_show, (c1[0], c1[1] - 2), 0, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def xyxy_to_xywh(box):
    """ 目标框转换 """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x_c = int(x1 + w/2)
    y_c = int(y1 + h/2)
    return [x_c, y_c, w, h]


def cut_bbox_container(bbox_container):
    """ 只保留人车信息 """
    container = []
    for bbox in bbox_container:
        label = bbox['class']
        confidence = bbox['confidence']
        box = bbox['box']
        if label in ['person', 'car']:
            container.append({'class': label, 'confidence': confidence, 'box': box})
    return container


def main():
    video_name = 'car.mp4'
    # video_name = 'ADL-Rundle-6-raw.webm'
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
    window_name = 'Real-time tracking'
    while True:
        state, frame = cap.read()
        if not state:
            break
        """ 检测目标 """
        image, bbox_container = yolov5_detector(frame)
        """ 仅保留人车信息"""
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
