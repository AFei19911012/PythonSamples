# -*- coding: utf-8 -*-
"""
 Created on 2021/9/13 21:29
 Filename   : rubbish_detector.py
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
        return im0, bbox_container


# ?????????????????????
def image_polygon_crop(image, polygon):
    pts = np.array(polygon)
    # ??????
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    background = np.ones_like(image, np.uint8) * 255
    cv2.bitwise_not(background, background, mask=mask)
    new_image = background + foreground
    return new_image


# ?????????
def draw_image(image, boxes_info, polygon):
    image0 = image.copy()
    pts = []
    for boxes in boxes_info:
        pts.append(boxes[0])
    cv2.polylines(image0, [np.array(pts, np.int32)], False, (0, 0, 255), 2)
    # ??????????????????
    cv2.polylines(image0, [np.array(polygon)], True, (255, 0, 0), 2)
    return image0


# ?????????
def draw_image_rect(image, rects_info, boxes_info, polygon):
    image0 = image.copy()
    color = (0, 0, 255)
    pts = []
    for i in range(0, len(rects_info)):
        pts.append(boxes_info[i][0])
        left, top, right, bottom = rects_info[i][0]
        c1, c2 = (left, top), (right, bottom)
        cv2.rectangle(image0, c1, c2, color, 2, cv2.LINE_AA)
    cv2.polylines(image, [np.array(pts, np.int32)], False, color, 2)
    # ??????????????????
    cv2.polylines(image0, [np.array(polygon)], True, (255, 0, 0), 2)
    return image0


def detect_demo():
    video_name = 'rubbish4.mp4'
    video_path = f'data/videos/{video_name}'
    cap = cv2.VideoCapture(video_path)
    fource = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid = cv2.VideoWriter(f'runs_out/{video_name}', fource, 30, (width, height))
    """ ??????????????? """
    yolov5_detector = YOLOv5Detector(weights='weights/rubbish.pt', conf_thres=0.4)
    window_name = 'YOLOv5 detector'

    # ?????????????????????????????????
    # polygon_area = [(300, 2), (200, 150), (100, 320), (620, 320), (600, 200), (620, 2)]
    polygon_area = [(200, 2), (200, 1050), (1900, 1050), (1900, 2)]
    # ??????????????????????????????????????????????????????????????????????????? h = 0.5 * g * t * t
    time_threshold = 4
    has_target = False
    boxes_info = []
    rects_info = []
    image_throwing = None
    vid_writer = None
    name_id = 1

    while True:
        state, frame = cap.read()
        if not state:
            break
        # ?????????????????????
        frame_crop = image_polygon_crop(frame, polygon_area)
        # ??????
        image_result, bbox_container = yolov5_detector(frame_crop)
        box_info = []
        rect_info = []
        if bbox_container:
            box = bbox_container[0]['box']
            box_info.append([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])
            rect_info.append(box)

        # ???????????????
        if box_info:
            has_target = True

            # ??????????????????????????????
            if image_throwing is None:
                image_throwing = frame

            # ????????????
            if vid_writer is None:
                vid_writer = cv2.VideoWriter(f'runs_out/video_{name_id:02d}.mp4', fource, 30, (width, height))

            # ??????????????????????????????????????????
            boxes_info.append(box_info)
            rects_info.append(rect_info)
            # ??????????????????????????????
            time_start = time.time()
            # # ??????????????????
            # if not boxes_info:
            #     boxes_info.append(box_info)
            #     rects_info.append(rect_info)
            #     # ??????????????????????????????
            #     time_start = time.time()
            # else:
            #     # ????????????????????????????????????????????????
            #     y2 = box_info[0][1]
            #     y1 = boxes_info[-1][0][1]
            #     # ????????????????????????????????????????????????
            #     if y2 > y1:
            #         boxes_info.append(box_info)
            #         rects_info.append(rect_info)

        # ???????????????????????????????????????????????????????????????
        if has_target:
            frame_rect = draw_image_rect(frame, rects_info, boxes_info, polygon_area)
            vid_writer.write(frame_rect)
            # ???????????? ???????????????????????????
            frame_line = draw_image(image_throwing, boxes_info, polygon_area)
            cv2.imwrite(f'runs_out/{name_id}.bmp', frame_line)

            # ??????????????????
            time_end = time.time()
            # ??????????????????
            if time_end - time_start > time_threshold:
                name_id = name_id + 1
                image_throwing = None
                has_target = False
                vid_writer.release()
                vid_writer = None
                boxes_info.clear()
                rects_info.clear()

        frame_show = draw_image_rect(frame, rects_info, boxes_info, polygon_area)
        vid.write(frame_show)
        """ ???????????? """
        cv2.imshow(window_name, frame_show)
        cv2.waitKey(1)
        print('---')
        """ ??? x ?????? """
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 1:
            break
    cap.release()
    vid.release()
    if vid_writer:
        vid_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_demo()
