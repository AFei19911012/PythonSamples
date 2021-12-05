# -*- coding: utf-8 -*-
"""
 Created on 2021/5/6 18:27
 Filename   : thread_demo.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
import threading
import time


# 定义线程函数
def print_time(thread_name, delay):
    count = 0
    while count < 5:
        time.sleep(delay)
        count += 1
        print(f"{thread_name}: {time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}")


# 创建两个线程
# Thread(target=print_time, args=('Thread-1', 1)).start()
# Thread(target=print_time, args=('Thread-2', 2)).start()
# thread1 = threading.Thread(target=print_time, args=('Thread-1', 1))
# thread2 = threading.Thread(target=print_time, args=('Thread-2', 2))
# 按顺序执行
# thread1.run()
# thread2.run()
'''
Thread-1: 2021-05-06 19-22-01
Thread-1: 2021-05-06 19-22-02
Thread-1: 2021-05-06 19-22-03
Thread-1: 2021-05-06 19-22-04
Thread-1: 2021-05-06 19-22-05
Thread-2: 2021-05-06 19-22-07
Thread-2: 2021-05-06 19-22-09
Thread-2: 2021-05-06 19-22-11
Thread-2: 2021-05-06 19-22-13
Thread-2: 2021-05-06 19-22-15
'''
# 同步执行
# thread1.start()
# thread2.start()
# thread1.join()
# thread2.join()
'''
Thread-1: 2021-05-06 19-23-11
Thread-2: 2021-05-06 19-23-12
Thread-1: 2021-05-06 19-23-12
Thread-1: 2021-05-06 19-23-13
Thread-2: 2021-05-06 19-23-14
Thread-1: 2021-05-06 19-23-14
Thread-1: 2021-05-06 19-23-15
Thread-2: 2021-05-06 19-23-16
Thread-2: 2021-05-06 19-23-18
Thread-2: 2021-05-06 19-23-20
'''


class MyThread(threading.Thread):
    def __init__(self, thread_name, delay):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.delay = delay

    def run(self):
        print(f'开始线程 {self.name}')
        print_time(self.name, self.delay)
        print(f'退出线程 {self.name}')


# 创建新线程
thread1 = MyThread('Thread-1', 1)
thread2 = MyThread('Thread-2', 2)
# 开启新线程
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print("退出主线程")
'''
开始线程 Thread-1
开始线程 Thread-2
Thread-1: 2021-05-06 19-35-36
Thread-2: 2021-05-06 19-35-37
Thread-1: 2021-05-06 19-35-37
Thread-1: 2021-05-06 19-35-38
Thread-1: 2021-05-06 19-35-39
Thread-2: 2021-05-06 19-35-39
Thread-1: 2021-05-06 19-35-40
退出线程 Thread-1
Thread-2: 2021-05-06 19-35-41
Thread-2: 2021-05-06 19-35-43
Thread-2: 2021-05-06 19-35-45
退出线程 Thread-2
退出主线程
'''