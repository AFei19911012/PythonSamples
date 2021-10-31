# -*- coding: utf-8 -*-
"""
 Created on 2021/4/19 11:47
 Filename   : ex_spider_image_baidu.py
 Author     : Taosy
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: Spider - get images from baidu
"""

import requests
import os
import re
import hashlib


def get_images_from_baidu(keyword, page_num, save_dir):
    # UA 伪装：当前爬取信息伪装成浏览器
    # 将 User-Agent 封装到一个字典中
    # 【（网页右键 → 审查元素）或者 F12】 → 【Network】 → 【Ctrl+R】 → 左边选一项，右边在 【Response Hearders】 里查找
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'}
    # 请求的 url
    url = 'https://image.baidu.com/search/acjson?'
    n = 0
    for pn in range(0, 30 * page_num, 30):
        # 请求参数
        param = {'tn': 'resultjson_com',
                 # 'logid': '7603311155072595725',
                 'ipn': 'rj',
                 'ct': 201326592,
                 'is': '',
                 'fp': 'result',
                 'queryWord': keyword,
                 'cl': 2,
                 'lm': -1,
                 'ie': 'utf-8',
                 'oe': 'utf-8',
                 'adpicid': '',
                 'st': -1,
                 'z': '',
                 'ic': '',
                 'hd': '',
                 'latest': '',
                 'copyright': '',
                 'word': keyword,
                 's': '',
                 'se': '',
                 'tab': '',
                 'width': '',
                 'height': '',
                 'face': 0,
                 'istype': 2,
                 'qc': '',
                 'nc': '1',
                 'fr': '',
                 'expermode': '',
                 'force': '',
                 'cg': '',    # 这个参数没公开，但是不可少
                 'pn': pn,    # 显示：30-60-90
                 'rn': '30',  # 每页显示 30 条
                 'gsm': '1e',
                 '1618827096642': ''
                 }
        request = requests.get(url=url, headers=header, params=param)
        if request.status_code == 200:
            print('Request success.')
        request.encoding = 'utf-8'
        # 正则方式提取图片链接
        html = request.text
        image_url_list = re.findall('"thumbURL":"(.*?)",', html, re.S)
        print(image_url_list)
        # # 换一种方式
        # request_dict = request.json()
        # info_list = request_dict['data']
        # # 看它的值最后多了一个，删除掉
        # info_list.pop()
        # image_url_list = []
        # for info in info_list:
        #     image_url_list.append(info['thumbURL'])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for image_url in image_url_list:
            image_data = requests.get(url=image_url, headers=header).content
            # with open(os.path.join(save_dir, f'{n:06d}.jpg'), 'wb') as fp:
            #     fp.write(image_data)
            # 用唯一的文件名
            md = hashlib.md5()
            md.update(bytes(image_url, encodings='utf-8'))
            filename = md.hexdigest()
            with open(os.path.join(save_dir, f'{filename}.jpg'), 'wb') as fp:
                fp.write(image_data)
            n = n + 1


if __name__ == '__main__':
    keywords = ['电梯 电动车']
    for keyword in keywords:
        save_dir = keyword
        page_num = 5
        get_images_from_baidu(keyword, page_num, save_dir)
    print('Get images finished.')
