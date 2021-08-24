# -*- coding: utf-8 -*-
"""
 Created on 2021/8/24 22:26
 Filename   : capure_screen_detector.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""

# =======================================================

import cv2
import imutils
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box
from PIL import ImageGrab


class YOLOv5Detector:
    """ YOLOv5 object detection """

    def __init__(self, weights='weights/yolov5s.pt', conf_thres=0.25, iou_thres=0.45, imgsz=640):
        """ Initialization """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.half_precision = True if torch.cuda.is_available() else False
        """ Load model """
        self.model = attempt_load(weights, map_location=self.device)
        self.model.to(self.device).eval()
        """ check image size """
        self.image_size = check_img_size(imgsz, s=int(self.model.stride.max()))
        """ get class names """
        self.names = self.model.names
        """ half precision only supported on CUDA """
        if self.half_precision:
            self.model.half()

    def image_preprocess(self, image):
        img0 = image.copy()
        """ Padded resize """
        img = letterbox(img0, self.image_size, stride=int(self.model.stride.max()))[0]
        """ Convert """
        """ BGR to RGB, to 3x416x416 """
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        """ uint8 to fp16/32 """
        img = img.half() if self.half_precision else img.float()
        """ 0 - 255 to 0.0 - 1.0 """
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def __call__(self, image, *args, **kwargs):
        img = self.image_preprocess(image)
        """ Inference """
        pred = self.model(img)[0]
        """ Apply NMS """
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]
        """ Process detections """
        im0 = image.copy()
        s = ''
        bbox_container = []
        if len(det):
            """ Rescale boxes from img_size to im0 size """
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            """ Print results """
            """ detections per class """
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                """ add to string """
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
            """ Write results """
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{self.names[c]} {conf:.2f}'
                """ xyxy: LU --> RD """
                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=2)
                bbox = {'class': self.names[c], 'confidence': round(conf.item(), 2), 'box': [int(v.item()) for v in xyxy]}
                bbox_container.append(bbox)
        print(s)
        return im0, bbox_container


def detect_demo():
    """ 图像检测器 """
    yolov5_detector = YOLOv5Detector(weights='weights/yolov5s.pt', conf_thres=0.25)
    window_name = 'YOLOv5 detector'
    while True:
        im = ImageGrab.grab(bbox=(0, 0, 600, 800))
        frame = np.array(im)
        image_result, bbox_container = yolov5_detector(frame)
        for info in bbox_container:
            print(info)
        print('---')
        """ 显示图像 """
        cv2.imshow(window_name, imutils.opencv2matplotlib(image_result))
        cv2.waitKey(1)
        """ 点 x 退出 """
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_demo()
