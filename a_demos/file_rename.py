# -*- coding: utf-8 -*-
"""
 Created on 2021/10/30 11:08
 Filename   : file_rename.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import os


def rename_file():
    fold_path = 'D:/MyPrograms/DataSet/car/images'
    files = os.listdir(fold_path)
    idx = 1
    for file in files:
        if 'jpg' in file:
            new_name = f'{idx:06d}.jpg'
            old_fullname = os.path.join(fold_path, file)
            new_fullname = os.path.join(fold_path, new_name)
            if not os.path.isfile(new_fullname):
                os.rename(old_fullname, new_fullname)
            idx = idx + 1


if __name__ == '__main__':
    rename_file()
