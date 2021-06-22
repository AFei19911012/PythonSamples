# -*- coding: utf-8 -*-
"""
 Created on 2021/4/17 11:36
 Filename: ex_queue_thread.py
 Author  : Taosy
 Zhihu   : https://www.zhihu.com/people/1105936347
 Github  : https://github.com/AFei19911012
 Describe: example for queue and thread
"""

from threading import Thread
import time
from queue import Queue, LifoQueue, PriorityQueue

# 先入先出
q = Queue(maxsize=5)
# 先入后出
lq = LifoQueue(maxsize=6)
# 优先级
pq = PriorityQueue(maxsize=5)

# 写入队列
for i in range(5):
    q.put(i)
    lq.put(i)
    pq.put(i)
print(f'先入先出队列：{q.queue}；队列大小：{q.qsize()}；是否为空：{q.empty()}；是否为满：{q.full()}')
print(f'先入后出队列：{lq.queue}；队列大小：{lq.qsize()}；是否为空：{lq.empty()}；是否为满：{lq.full()}')
print(f'优先级队列：{pq.queue}；队列大小：{pq.qsize()}；是否为空：{pq.empty()}；是否为满：{pq.full()}')
'''
先入先出队列：deque([0, 1, 2, 3, 4])；队列大小：5；是否为空：False；是否为满：True
先入后出队列：[0, 1, 2, 3, 4]；队列大小：5；是否为空：False；是否为满：False
优先级队列：[0, 1, 2, 3, 4]；队列大小：5；是否为空：False；是否为满：True
'''

# 获取队列
print(q.get())
print(lq.get())
print(pq.get())
'''
0
4
0
'''
print(f'先入先出队列：{q.queue}；队列大小：{q.qsize()}；是否为空：{q.empty()}；是否为满：{q.full()}')
print(f'先入后出队列：{lq.queue}；队列大小：{lq.qsize()}；是否为空：{lq.empty()}；是否为满：{lq.full()}')
print(f'优先级队列：{pq.queue}；队列大小：{pq.qsize()}；是否为空：{pq.empty()}；是否为满：{pq.full()}')
'''
先入先出队列：deque([1, 2, 3, 4])；队列大小：4；是否为空：False；是否为满：False
先入后出队列：[0, 1, 2, 3]；队列大小：4；是否为空：False；是否为满：False
优先级队列：[1, 3, 2, 4]；队列大小：4；是否为空：False；是否为满：False
'''

# 不限大小
q = Queue(maxsize=0)


def producer(name):
    count = 1
    while True:
        q.put(f'ball-{count}')
        print(f'{name} produce {count} balls')
        count += 1
        time.sleep(1)


def consumer(name):
    while True:
        print(f'{name} takes {q.get()}')
        time.sleep(0.2)
        # finish the task
        q.task_done()


Thread(target=producer, args=('Boss',)).start()
Thread(target=consumer, args=('member-1',)).start()
Thread(target=consumer, args=('member-2',)).start()
