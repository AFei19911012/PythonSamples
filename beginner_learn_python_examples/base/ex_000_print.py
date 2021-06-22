# -*- coding: utf-8 -*-
"""
Created on 2021/3/27 17:47
author: ff_wang
E-mail: 1105936347@qq.com
Zhihu: https://www.zhihu.com/people/1105936347
Github: https://github.com/AFei19911012
"""

# 表示注释
"""
三个成对双引号多行注释；
Tab 键自动补全；
"""
'''
三个成对单引号多行注释；
'''

# print 函数打印结果；
print("Hello World")
"""
Hello World
"""
print(1 + 2 * 3)
"""
7
"""

# 字符串 + 字符串 表示字符串拼接；
print("Hello " + "World")
"""
Hello World
"""

# 字符串 * 数字 表示字符串重复；
print("Hello World\n" * 4)
"""
Hello World
Hello World
Hello World
Hello World
"""

# 打印单引号双引号
print("Let's go!")
# 转义
print('Let\'s go!')
"""
Let's go!
"""
print("Let\"s go!")
print('Let\"s go!')
"""
Let"s go!
"""
print('Let''s play a game')
"""
Lets play a game
"""
print('Let"s play a game')
"""
Let"s play a game
"""

temp_str = input('Input a number:')
print(temp_str)
"""
输出的是字符串 str
5
"""
num = int(temp_str)
print(num)
"""
数字 int，如果 temp_str 不是数字字符串会报错
5
"""