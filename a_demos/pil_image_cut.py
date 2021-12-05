# -*- coding: utf-8 -*-
"""
 Created on 2021/10/30 11:48
 Filename   : pil_image_cut.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
from PIL import Image
import os


def image_cut_save():
    fold_path = 'D:/MyPrograms/DataSet/car/images'
    files = os.listdir(fold_path)
    for file in files:
        fullname = os.path.join(fold_path, file)
        img = Image.open(fullname)
        width, height = img.size
        if width > 1000:
            # 缩放
            img = img.resize((848, 518))
            # 裁剪
            box = (2, 15, 848, 518)
            img = img.crop(box)
            img.save(fullname)
            print(fullname)


if __name__ == '__main__':
    image_cut_save()
