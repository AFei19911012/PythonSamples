# -*- coding: utf-8 -*-
"""
 Created on 2021/9/30 22:03
 Filename   : head_detector.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box


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
        return bbox_container


# 图像多边形裁剪
def image_polygon_crop(image, polygon):
    pts = np.array(polygon)
    # 掩膜
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    background = np.ones_like(image, np.uint8) * 255
    cv2.bitwise_not(background, background, mask=mask)
    new_image = background + foreground
    return new_image


# 画方框、统计人数
def draw_image(image, bbox_container, polygon):
    tl = 2 or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)
    for box in bbox_container:
        x = box['box']
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(image, c1, c2, (255, 0, 0), thickness=tl, lineType=cv2.LINE_AA)
    cv2.putText(image, f'total person: {len(bbox_container)}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, tl/3, [0, 0, 255], thickness=tf,
                lineType=cv2.LINE_AA)
    # 画多边形区域
    cv2.polylines(image, [np.array(polygon)], True, (255, 0, 255), 2)


def detect_demo():
    image_name = 'rubbish.jpg'
    image_path = f'data/images/{image_name}'
    image_save_path = f'runs/detect/exp/{image_name}'
    video_name = '1.mp4'
    video_path = f'data/videos/{video_name}'
    video_save_path = f'runs/detect/exp/{video_name}'
    image_path = ''
    polygon_area = [(300, 2), (200, 150), (100, 320), (620, 320), (600, 200), (620, 2)]
    # polygon_area = [(200, 2), (200, 1050), (1900, 1050), (1900, 2)]
    if image_path:
        cap = cv2.VideoCapture(image_path)
    else:
        cap = cv2.VideoCapture(video_path)
        fource = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(video_save_path, fource, 30, (width, height))
    """ 图像检测器 """
    yolov5_detector = YOLOv5Detector(weights='weights/head.pt', conf_thres=0.25, iou_thres=0.15)
    window_name = 'YOLOv5 detector'
    while True:
        state, frame = cap.read()
        if not state:
            break
        # 裁剪图像
        new_frame = image_polygon_crop(frame, polygon_area)
        bbox_container = yolov5_detector(new_frame)
        for info in bbox_container:
            print(info)
        print('---')
        """ 显示图像 """
        draw_image(frame, bbox_container, polygon_area)
        cv2.imshow(window_name, frame)
        if image_path:
            cv2.imwrite(image_save_path, frame)
            cv2.waitKey(1000)
        else:
            vid_writer.write(frame)
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
