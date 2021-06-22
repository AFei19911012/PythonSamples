# -*- coding: utf-8 -*-
"""
 Created on 2021/6/18 22:13
 Filename   : ex_camera_huawei.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: 
# =======================================================
import requests
from pkg_resources import resource_filename
from requests.auth import HTTPDigestAuth
import json
from wsgiref.simple_server import make_server
import re
import numpy as np
import cv2
import base64
import ssl


# requests.packages.urllib3.disable_warnings()


def subscriber(ip, host='44.195.12.227', passwd='Zhzt2026', port=6666):
    url = f'https://{ip}/SDCAPI/V1.0/Metadata/Subscription/Subscriber?Topic=all&ID=1'
    data = {
        "topic": "all",
        "address": host,
        "port": port,
        "timeOut": port,
        "httpsEnable": 1,
        "alarmURL": "",
        "digUserName": "",
        "digUserPwd": ""
    }
    response = requests.post(url, json=data, verify=False, auth=requests.auth.HTTPDigestAuth('ApiAdmin', passwd))
    print(response.content)


def check_subscriber(ip, passwd='Zhzt2026'):
    url = f'https://{ip}/SDCAPI/V1.0/Metadata/Subscription'
    response = requests.get(url, verify=False, auth=requests.auth.HTTPDigestAuth('ApiAdmin', passwd))
    if response.status_code == 401:
        print('Password Error ... try HuaWei123')


def application(environ, start_response):
    """
    定义函数，参数都是 python 本身定义的，默认就行
    """
    # 定义文件请求的类型和当前请求成功的code
    start_response('200 OK', [('Content-Type', 'application/json')])
    # environ 是当前请求的所有数据，包括 Header 和 URL，body
    request_body = environ["wsgi.input"].read(int(environ.get("CONTENT_LENGTH", 0)))
    json_str = request_body.decode('utf-8')  # byte 转 str
    json_str = re.sub('\'', '\"', json_str)  # 单引号转双引号, json.loads 必须使用双引号
    json_dict = json.loads(json_str)         # （注意：key值必须双引号）
    image_list = json_dict['metadataObject']['subImageList']
    image = decode_image(image_list[0]['data'])
    cv2.imwrite('test.jpg', image)
    return [json.dumps(json_dict)]


def decode_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img_data = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(img_data, 1)  # 以彩色模式读入为 1，灰色为 0，又变回二进制
    return image


def test_receive_server():
    """
    向华为相机发送请求以获取数据
    """
    httpd = make_server('0.0.0.0', 6666, application)
    print('serving http on 0.0.0.0:6666')
    httpd.socket = ssl.wrap_socket(httpd.socket, server_side=True, certfile=resource_filename(__name__, "localhost.pem"), ssl_version=ssl.PROTOCOL_TLSv1_2)
    httpd.serve_forever()


def test_subscriber():
    url = 'https://44.195.12.145/SDCAPI/V1.0/Metadata/Subscription'
    response = requests.get(url, verify=False, auth=requests.auth.HTTPDigestAuth('ApiAdmin', 'HuaWei123'))
    print(response)
    subscriber(ip='44.195.12.104', passwd='HuaWei123')


if __name__ == "__main__":
    test_subscriber()
    # test_receive_server()
