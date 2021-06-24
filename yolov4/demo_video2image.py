# -*- coding: utf-8 -*-
"""
 Created on 2021/4/11 17:53
 Filename: demo_image2image.py
 Author  : Taosy
 Zhihu   : https://www.zhihu.com/people/1105936347
 Github  : https://github.com/AFei19911012
 Describe: video --> image
"""

import cv2 as cv
import time
import os
import sys


def main():
    # file path
    video_name = 'helmet.mp4'
    video_path = 'videos/' + video_name
    image_path = 'data/' + video_name[:-4] + '_' + time.strftime('%Y_%m_%d', time.localtime())
    # generate a fold
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # read video
    if not os.path.isfile(video_path):
        print(video_path + ' not exist')
        sys.exit(1)
    cap = cv.VideoCapture(video_path)

    # frame --> image
    has_frame = True
    idx = 0
    while has_frame:
        has_frame, frame = cap.read()
        if has_frame and idx % 5 == 1:
            # file_name = f'{idx//5}'.zfill(6) + '.jpg'
            file_name = f'{idx // 5:06d}.jpg'
            cv.imwrite(os.path.join(image_path, file_name), frame)
        idx += 1
    cap.release()
    print('Done processing.')


if __name__ == '__main__':
    main()
