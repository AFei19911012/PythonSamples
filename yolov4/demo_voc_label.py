# -*- coding: utf-8 -*-
"""
 Created on 2021/4/11 21:40
 Filename: demo_voc_label.py
 Author  : Taosy
 Zhihu   : https://www.zhihu.com/people/1105936347
 Github  : https://github.com/AFei19911012
 Describe:
"""

import os

if __name__ == '__main__':
    wd = os.getcwd()
    # Get train and val
    file_list = os.listdir('data/helmet_2021_04_11')
    train_list = []
    for f in file_list:
        if '.jpg' in f:
            train_list.append(f[:-4])
    # Write to train.txt
    list_file = open(f'config/helmet_train.txt', 'w')
    for image_id in train_list:
        list_file.write(f'{wd}/data/helmet_2021_04_11/{image_id}.jpg\n')
    list_file.close()
