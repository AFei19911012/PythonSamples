# -*- coding: utf-8 -*-
"""
 Created on 2021/5/6 19:47
 Filename   : ex_010_json_requests.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""

import json

# python dict --> JSON
dict1 = {'Name': 'Taosy.W', 'url': 'https://www.zhihu.com/people/1105936347'}
json_str = json.dumps(dict1)
print(f'python 原始数据：{dict1}')
print(f'JSON 对象：{json_str}')
'''
python 原始数据：{'Name': 'Taosy.W', 'url': 'https://www.zhihu.com/people/1105936347'}
JSON 对象：{"Name": "Taosy.W", "url": "https://www.zhihu.com/people/1105936347"}
'''

# JSON --> python dict
dict2 = json.loads(json_str)
print(f"dict2['Name']: {dict2['Name']}")
print(f"dict2['url']: {dict2['url']}")
'''
dict2['Name']: Taosy.W
dict2['url']: https://www.zhihu.com/people/1105936347
'''

# 处理文件
# 写入 JSON 数据
with open('data/data.json', 'w') as f:
    json.dump(dict1, f)

# 读取 JSON 数据
with open('data/data.json', 'r') as f:
    data = json.load(f)


import requests

# 发送请求得到 response 响应，从中获取信息
response = requests.get('https://www.zhihu.com/people/1105936347')

# 传递 URL 参数，值为 None 的键不会被添加到 URL 的查询字符串里
payload = {'key': 'value1', 'key2': 'value2', 'key3': None}
response = requests.get('https://www.zhihu.com/people/1105936347', params=payload)
print(response.url)
'''
https://www.zhihu.com/people/1105936347?key=value1&key2=value2
'''
payload = {'key1': 'value1', 'key2': ['value2', 'value3']}
response = requests.get('https://www.zhihu.com/people/1105936347', params=payload)
print(response.url)
'''
https://www.zhihu.com/people/1105936347?key1=value1&key2=value2&key2=value3
'''

# 定制请求头
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'}
response = requests.get('https://api.github.com/events', headers=header)
print(f'StatusCode: {response.status_code}')
# print(response.text)
# UnicodeEncodeError: 'gbk' codec can't encode character '\u200b' in position 8437: illegal multibyte sequence
with open('../zhihu.html', 'w', encoding='utf-8') as fid:
    fid.write(response.text)
# JSON 响应内容，不一定能成功，这里返回的是一个列表，每个元素是一个字典
response_list = response.json()
# 比如提取每个元素里键为 id 的值
response_id = [dict_temp['id'] for dict_temp in response_list]
print(response_id)
'''
['16255003591', '16255003586', '16255003576', '16255003587', '16255003550', '16255003582', '16255003579', '16255003578', '16255003573', 
 '16255003580', '16255003575', '16255003567', '16255003568', '16255003564', '16255003560', '16255003557', '16255003556', '16255003555', 
 '16255003549', '16255003548', '16255003551', '16255003541', '16255003543', '16255003547', '16255003544', '16255003540', '16255003517', 
 '16255003535', '16255003536', '16255003525'] 
'''