# -*- encoding: utf-8 -*-
"""
 Created on 2021/6/17 17:13
 Filename   : predictor.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: https://github.com/JDAI-CV/fast-reid
"""

# =======================================================
import os
from os.path import abspath
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer


def setup_cfg():
    cfg_ = get_cfg()
    cfg_.MODEL.DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    """ Torch not compiled with CUDA enabled """
    # cfg_.MODEL.DEVICE = 'cuda'
    """ car """
    cfg_.merge_from_file(abspath('configs/VehicleID/bagtricks_R50-ibn.yml'))
    cfg_.merge_from_list(['MODEL.WEIGHTS', abspath('models/vehicleid_bot_R50-ibn.pth')])
    # cfg_.merge_from_file(abspath('configs/VERIWild/bagtricks_R50-ibn.yml'))
    # cfg_.merge_from_list(['MODEL.WEIGHTS', abspath('models/veriwild_bot_R50-ibn.pth')])
    cfg_.freeze()
    return cfg_


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config.
    The predictor takes an BGR image, resizes it to the specified resolution,
    runs the model and produces a dict of predictions.
    This predictor takes care of model loading and input preprocessing for you.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
    Examples:
    .. code-block:: python
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg_):
        self.cfg = cfg_.clone()  # cfg can be modified by model
        self.cfg.defrost()
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(cfg_.MODEL.WEIGHTS)

    def __call__(self, image):
        """
        Args:
            image (torch.tensor): an image tensor of shape (B, C, H, W).
        Returns:
            predictions (torch.tensor): the output features of the model
        """
        inputs = {"images": image.to(self.model.device)}
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            predictions = self.model(inputs)
            # Normalize feature to compute cosine distance
            features = F.normalize(predictions)
            features = features.data
        return features


class FeatureExtraction(object):
    def __init__(self, cfg_):
        self.cfg = cfg_
        self.predictor = DefaultPredictor(cfg_)

    def run_on_image(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (np.ndarray): normalized feature of the model.
        """
        # the model expects RGB inputs
        original_image = original_image[:, :, ::-1]
        # Apply pre-processing to image.
        image = cv2.resize(original_image, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
        # Make shape with a new batch dimension which is adapted for network input
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
        predictions = self.predictor(image)
        return predictions

    def run_on_image_list(self, original_image_list):
        predictions_list = []
        for image in original_image_list:
            predictions_list.append(self.run_on_image(image))
        return predictions_list

    def __call__(self, original_image_list):
        predictions_list = []
        for image in original_image_list:
            predictions_list.append(self.run_on_image(image))
        return predictions_list


def cal_cosine_dist(feat_, feats_):
    """
    Computes cosine distance
    """
    """ list --> tensor """
    feats_ = torch.cat(feats_, dim=0)
    feats_ = feats_.t()
    dist_ = 1 - torch.mm(feat_, feats_)
    dist_ = dist_.cpu().numpy().tolist()[0]
    # numpy
    # dist_ = 1 - np.sum(feat_ * feats_) / np.linalg.norm(feat_) * np.linalg.norm(feats_)
    # np.save('test.npy', feat)
    return dist_


def test_car():
    cfg = setup_cfg()
    pred = FeatureExtraction(cfg)
    image_list = []
    file_list = os.listdir('images')
    for file in file_list:
        if 'car' in file:
            image = cv2.imread(f'images/{file}')
            image_list.append(image)
    feat1 = pred.run_on_image(image_list[0])
    feat2 = pred.run_on_image(image_list[3])
    feat3 = pred.run_on_image(image_list[6])
    feat4 = pred.run_on_image(image_list[12])
    feats = pred.run_on_image_list(image_list)
    # feats = pred(image_list)
    for i in range(0, 4):
        dist = cal_cosine_dist(eval(f'feat{i+1}'), feats)
        for value in dist:
            print(format(value, '.4f'), end=', ')
        print('')


if __name__ == '__main__':
    test_car()
