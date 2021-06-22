# -*- coding: utf-8 -*-
"""
 Created on 2021/4/19 14:22
 Filename   : ex_spider_UA.py
 Author     : Taosy
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: User-Agent
"""

import requests

if __name__ == '__main__':
    url = 'https://www.baidu.com/'
    # UA 伪装：当前爬取信息伪装成浏览器
    # 将 User-Agent 封装到一个字典中
    # 【（网页右键 → 审查元素）或者 F12】 → 【Network】 → 【Ctrl+R】 → 左边选一项，右边在 【Response Hearders】 里查找
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'}
    # 处理 url 携带的参数：封装到字典中
    key_word = '先知大数据'
    param = {'query': key_word}
    response = requests.get(url=url, params=param, headers=header)
    page_text = response.text
    # 打印爬取到的信息
    print(page_text)
