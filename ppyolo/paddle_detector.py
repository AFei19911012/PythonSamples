# -*- coding: utf-8 -*-
"""
 Created on 2021/6/24 21:24
 Filename   : paddle_detector.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: Paddle 目标检测
"""

# =======================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import tempfile
import warnings
import cv2
import paddle
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser

""" ignore warning log """
warnings.filterwarnings('ignore')


class PaddleDetector:
    """ Paddle 目标检测器 """

    def __init__(self, weights='resources/weights/ppyolo_r50vd_dcn_1x_coco.pdparams', cfg_path='configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml',
                 conf_thres=0.2, use_gpu=False):
        """ 初始化 """
        self.weights = weights
        self.cfg_path = cfg_path
        self.conf_thres = conf_thres
        self.use_gpu = use_gpu
        self.classes = []
        self.colors = []
        self.trainer = self.init_model()

    def init_model(self):
        parser = ArgsParser()
        parser.add_argument("-c", "--config", type=str, default=self.cfg_path, help="configuration file to use")
        parser.add_argument("--draw_threshold", type=float, default=self.conf_thres, help="Threshold to reserve the result for visualization.")
        args = parser.parse_args()
        cfg = load_config(args.config)
        cfg.weights = self.weights
        cfg.use_gpu = self.use_gpu
        paddle.set_device('gpu' if cfg.use_gpu else 'cpu')
        if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
            cfg['norm_type'] = 'bn'
        check_config(cfg)
        check_gpu(cfg.use_gpu)
        check_version()
        """ build trainer """
        trainer = Trainer(cfg, mode='test')
        """ load weights """
        trainer.load_weights(cfg.weights)
        return trainer

    def get_colors(self):
        """ 生成颜色 """
        random.seed(0)
        return {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in self.classes}

    def draw_image(self, image, classes, bbox_res, conf_thres):
        """ 绘制检测结果 """
        """ Draw the predicted bounding box """
        """ line thickness """
        tl = 2 or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        colors = self.get_colors()
        bbox_container = []
        for value in bbox_res:
            category_id = value['category_id']
            score = value['score']
            """ 得分阈值限制 """
            if score < conf_thres:
                continue
            x, y, w, h = value['bbox']
            left = int(x)
            top = int(y)
            right = int(x + w)
            bottom = int(y + h)
            c1 = (left, top)
            c2 = (right, bottom)
            label = classes[category_id]
            bbox_container.append({'class': label, 'confidence': score, 'box': [left, top, right, bottom]})
            """ line color """
            color = colors[label]
            """ Draw a bounding box """
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            """ Get the label for the class name and its confidence """
            label_show = f'{label}_{score:.2f}'
            """ font thickness """
            tf = max(tl - 1, 1)
            """ Display the label at the top of the bounding box """
            t_size = cv2.getTextSize(label_show, cv2.FONT_HERSHEY_SIMPLEX, fontScale=tl / 4, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            """ filled """
            cv2.rectangle(image, c1, c2, color, cv2.FILLED, cv2.LINE_AA)
            cv2.putText(image, label_show, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return bbox_container

    def __call__(self, image_path, *args, **kwargs):
        """ 目标检测调用 """
        """ inference """
        info = self.trainer.predict_image([image_path])
        """ 绘制结果信息 """
        self.classes = info['classes']
        bbox_res = info['bbox_res']
        image = cv2.imread(image_path)
        bbox_container = self.draw_image(image, self.classes, bbox_res, self.conf_thres)
        return image, bbox_container


def detect_demo():
    image_name = '000000570688.jpg'
    image_path = f'resources/images/{image_name}'
    image_save_path = f'resources/runs/{image_name}'
    video_name = 'run.mp4'
    video_path = f'resources/videos/{video_name}'
    video_save_path = f'resources/runs/{video_name}'
    image_path = ''
    """ 图像检测器 """
    paddle_detector = PaddleDetector()
    window_name = 'Paddle detector'
    if image_path:
        image, bbox_container = paddle_detector(image_path)
        cv2.imwrite(image_save_path, image)
        for info in bbox_container:
            print(info)
        cv2.imshow(window_name, image)
        cv2.waitKey(2000)
    else:
        cap = cv2.VideoCapture(video_path)
        fource = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(video_save_path, fource, 30, (width, height))
        while True:
            state, frame = cap.read()
            if not state:
                break
            """ 本地加载图片方式 """
            # video detection: a stupid method
            temp = tempfile.NamedTemporaryFile()
            image_path_temp = temp.name + '.jpg'
            cv2.imwrite(image_path_temp, frame)
            image, bbox_container = paddle_detector(image_path_temp)
            os.remove(image_path_temp)
            for info in bbox_container:
                print(info)
            print('---')
            """ 显示图像 """
            cv2.imshow(window_name, image)
            vid_writer.write(image)
            cv2.waitKey(1)
            """ 点 x 退出 """
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 1:
                break
        cap.release()
        vid_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_demo()
