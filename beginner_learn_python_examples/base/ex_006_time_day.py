# -*- coding: utf-8 -*-
"""
 Created on 2021/4/4 14:59
 Author: ff_wang
 E-mail: 1105936347@qq.com
 Zhihu : https://www.zhihu.com/people/1105936347
 Github: https://github.com/AFei19911012
-------------------------------------------------
"""

import time
import calendar

# 时间戳以自 1970-1-1 午夜经过了多长时间来表示
# 获取当前时间戳
ticks = time.time()
print(f'当前时间戳为：{ticks}')
'''
当前时间戳为：1617519853.3761892
'''

# 获取当前时间
'''
0	tm_year	2008
1	tm_mon	1 到 12
2	tm_mday	1 到 31
3	tm_hour	0 到 23
4	tm_min	0 到 59
5	tm_sec	0 到 61 (60或61 是闰秒)
6	tm_wday	0到6 (0是周一)
7	tm_yday	1 到 366(儒略历)
8	tm_isdst	-1, 0, 1, -1是决定是否为夏令时的旗帜
'''
localtime = time.localtime(time.time())
year = localtime.tm_year
mon = localtime.tm_mon
day = localtime.tm_mday
hour = localtime.tm_hour
minu = localtime.tm_min
sec = localtime.tm_sec
print(f'本地时间为：{year, mon, day, hour, minu, sec}')
'''
本地时间为：(2021, 4, 4, 15, 19, 29)
'''

# 格式化时间
localtime = time.asctime(time.localtime())
print(f'本地时间为：{localtime}')
'''
本地时间为：Sun Apr  4 15:20:31 2021
'''
# 格式化成 2021-04-04 15:22:50 形式
print(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()))
'''
2021-04-04 15-23-23
'''
# 格式化成 Sun Apr 4 15:20:31 2021 的形式
print(time.strftime('%a %b %d %H:%M:%S %Y', time.localtime()))
'''
Sun Apr 04 15:28:42 2021
'''
# 将格式字符串转换为时间戳
time_str = 'Sun Apr 04 15:28:42 2021'
localtime = time.strptime(time_str, '%a %b %d %H:%M:%S %Y')
print(localtime)
'''
time.struct_time(tm_year=2021, tm_mon=4, tm_mday=4, tm_hour=15, tm_min=28, tm_sec=42, tm_wday=6, tm_yday=94, tm_isdst=-1)
'''
print(time.mktime(localtime))
'''
1617521322.0
'''

# 打印日历
cal = calendar.month(2021, 4)
print(f'2021年4月份日历：\n{cal}')
'''
2021年4月份日历：
     April 2021
Mo Tu We Th Fr Sa Su
          1  2  3  4
 5  6  7  8  9 10 11
12 13 14 15 16 17 18
19 20 21 22 23 24 25
26 27 28 29 30
'''

# 判断闰年
print(f'2000 年是闰年：{calendar.isleap(2000)}')
'''
2000 年是闰年：True
'''
print(f'1900 年是闰年：{calendar.isleap(1900)}')
'''
1900 年是闰年：False
'''