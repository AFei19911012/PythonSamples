# -*- coding: utf-8 -*-
"""
 Created on 2021/6/18 22:21
 Filename   : ex_redis.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: 
# =======================================================
import redis


def test1():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.set('name', 'seer')   # 设置 name 对应的值
    print(r['name'])
    print(r.get('name'))    # 取出 name 对应的值


def test2():
    # 共享连接池
    pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    r = redis.Redis(connection_pool=pool)
    r.set('name', 'seer', ex=3)
    print(r.get('name'))


def test3():
    pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    r = redis.Redis(connection_pool=pool)
    # 批量设置
    # r.mget({'k1': 'v1', 'k2': 'v2'})
    r.mset({'k1': 'v1', 'k2': 'v2'})
    # 批量获取
    print(r.mget('k1', 'k2'))


def test4():
    pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    r = redis.Redis(connection_pool=pool)
    # 基本命令 hash
    r.hset("hash1", "k1", "v1")
    r.hset("hash1", "k2", "v2")
    print(r.hkeys("hash1"))              # 取 hash 中所有的 key
    print(r.hget("hash1", "k1"))         # 单个取 hash 的 key 对应的值
    print(r.hmget("hash1", "k1", "k2"))  # 多个取 hash 的 key 对应的值
    r.hsetnx("hash1", "k2", "v3")        # 只能新建
    print(r.hget("hash1", "k2"))

    for i in range(10):
        r.hset('hash2', f'key{i}', f'value{i}')
    # 大数据量获取
    for h in r.hscan_iter('hash2'):
        print(h)


if __name__ == '__main__':
    test4()
