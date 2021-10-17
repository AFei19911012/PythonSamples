# -*- coding: utf-8 -*-
"""
 Created on 2021/10/17 15:32
 Filename   : get_person_feat.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import cv2
import numpy as np
import torch
import os
import math
import sys

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir + '\\fast_reid')

from fast_reid.predictor import FeatureExtraction

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def cosine_similarity(feat_, feats_):
    dists = []
    for feat in feats_:
        dist = np.sum(feat_ * feat) / np.linalg.norm(feat_) * np.linalg.norm(feat)
        dists.append(dist)
    return dists


def get_person_feat():
    pred = FeatureExtraction()
    image_list = []
    file_list = os.listdir('fast_reid/data/person')
    names = []
    for i, file in enumerate(file_list):
        image = cv2.imread(f'fast_reid/data/person/{file}')
        image_list.append(image)
        names.append(f'id-{i + 1}')
    names.append('none')
    feats = pred.run_on_image_list(image_list)
    np.save('fast_reid/data/features.npy', feats)
    np.save('fast_reid/data/names.npy', names)
    print(len(feats))
    print(feats[0].shape)


def test_person():
    feats = np.load('fast_reid/data/features.npy')
    names = np.load('fast_reid/data/names.npy')
    image_list = []
    file_list = os.listdir('fast_reid/data/person')
    for i, file in enumerate(file_list):
        image = cv2.imread(f'fast_reid/data/person/{file}')
        image_list.append(image)
    pred = FeatureExtraction()
    for file in file_list:
        image = cv2.imread(f'fast_reid/data/person/{file}')
        feat = pred.run_on_image(image)
        similarity = cosine_similarity(feat, feats)
        max_index = np.argmax(similarity, axis=0)
        print(names[max_index])


if __name__ == '__main__':
    get_person_feat()
    test_person()
