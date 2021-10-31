# -*- coding: utf-8 -*-
"""
Created on 2021/3/28 21:05
author: ff_wang
E-mail: 1105936347@qq.com
Zhihu: https://www.zhihu.com/people/1105936347
Github: https://github.com/AFei19911012
"""

import random

# Fibonacci sequence: 斐波那契数列
# 0、1、1、2、3、5、8、13、21、34
# end 关键字，可用于将结果输出到同一行
a, b = 0, 1
while b < 100:
    print(b, end=', ')
    a, b = b, a + b
'''
1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 
'''

# 加载 random 包
# 产生一个 1 到 10 之间的整数
answer = random.randint(1, 10)
print("游戏开始：")
temp = input("猜一个数字：")
guess = int(temp)
if guess == answer:
    print("真厉害，一下就猜中了。")
else:
    # 这里用 while 循环，猜不中会一直猜下去
    while guess != answer:
        if guess < answer:
            temp = input("太小了，重新猜：")
            guess = int(temp)
        elif guess > answer:
            temp = input("太大了，重新猜：")
            guess = int(temp)
        # 这里也要判断一下是否猜中了
        if guess == answer:
            print('恭喜你猜中了')
print("游戏结束！")
'''
猜一个数字：5
太大了，重新猜：3
恭喜你猜中了
游戏结束！
'''

for ch in 'LOVE':
    print(ch, end=', ')
print('\n')
'''
L, O, V, E, 
'''

language = ['C', 'C++', 'Pyhton', 'Matlab']
for x in language:
    print(x, end=', ')
print('\n')
'''
C, C++, Pyhton, Matlab, 
'''

# break 跳出循环体
# continue跳出当前循环继续下一次循环
code = 'ff_wang'
answer = input('Input the code: ')
while True:
    if answer == code:
        break
    answer = input('Error code, input again: ')
print('Congratulation!')
'''
Input the code: mima
Error code, input again: ff_wang
Congratulation!
'''

for i in range(5):
    if i % 2 != 0:
        print(i)
        continue
    i += 3
    print(i)
'''
3  # i = 0 不满足 if 条件，执行 i = i + 3 打印 3
1  # i = 1 满足 if 条件，打印 1 不执行后续语句
5
3
7
'''