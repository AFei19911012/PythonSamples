# -*- coding: utf-8 -*-
"""
 Created on 2021/5/29 13:12
 Filename   : ex_camera_rtsp.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: 
# =======================================================
import cv2
import time
import multiprocessing as mp


def run_camera_url():
    # 大华
    url = 'rtsp://admin:Zhzt2026@44.195.14.202/cam/realmonitor?channel=1&subtype=0'
    # 华为
    # url = 'rtsp://admin:Zhzt2026@44.195.12.168:554/LiveMedia/ch1/Media1'
    # 海康
    # url = 'rtsp://admin:ztjjdd123@44.199.84.216:554/Streaming/Channels/1'
    cap = cv2.VideoCapture(url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def image_put(q, url):
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        print('Camera opened ...')
    while True:
        ret, frame = cap.read()
        q.put(frame)
        time.sleep(0.001)


def image_get(q):
    while True:
        frame = q.get()
        cv2.imshow('DaHua', frame)
        cv2.waitKey(1)


def run_camera_url_multiprocessing():
    url = 'rtsp://admin:Zhzt2026@44.195.14.202/cam/realmonitor?channel=1&subtype=0'
    queue = mp.Queue(maxsize=2)
    processes = [mp.Process(target=image_put, args=(queue, url)),
                 mp.Process(target=image_get, args=(queue,))]
    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    mp.set_start_method(method='spawn')
    run_camera_url()
    # run_camera_url_multiprocessing()
