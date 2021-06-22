# -*- coding: utf-8 -*-
"""
Created on 2021/3/27 19:39
author: ff_wang
E-mail: 1105936347@qq.com
Zhihu: https://www.zhihu.com/people/1105936347
Github: https://github.com/AFei19911012
"""

# 算术运算符
a = 21.1
b = 10
c = a + b
print('a + b = ', c)
"""
a + b =  31.1
"""
c = a - b
print('a - b = ', c)
"""
a - b =  11.100000000000001
"""
c = a * b
print('a * b = ', c)
"""
a * b =  211.0
"""
c = a / b
print('a / b = ', c)
"""
a / b =  2.1100000000000003
"""
# 求余数
c = a % b
print('a % b = ', c)
"""
a % b =  1.1000000000000014
"""
# 向下取整
c = a // b
print('a // b = ', c)
"""
a // b =  2.0
"""
a = 2
b = 3
c = a**b
print("a**b = ", c)
"""
a**b =  8
"""

# 比较运算符，返回 bool 值 False 或 True
a = 10
b = 20
result = a == b
print(result)
"""
False
"""
print(a != b)
"""
True
"""
print(a > b)
"""
False
"""
print(a < b)
"""
True
"""
print(a >= b)
"""
False
"""
print(a <= b)
"""
True
"""

# 赋值运算符
'''
c = a + b
c += a 等效 c = c + a
c -= a 等效 c = c - a
c *= a 等效 c = c * a
c /= a 等效 c = c / a
c %= a 等效 c = c % a
c **= a 等效 c = c ** a
c //= a 等效 c = c // a
'''

# 位运算符
# 将数字视为二进制来进行计算
'''
a = 60 
b = 13
a = 0011 1100
b = 0000 1101
a&b = 0000 1100
a|b = 0011 1101
a^b = 0011 0001
~a  = 1100 0011
'''
a = 60
b = 13
# 按位与
print('a & b = ', a & b)
'''
a & b =  12   # 12 = 0000 1100
'''
# 按位或
print('a | b = ', a | b)
'''
a | b =  61   # 61 = 0011 1101
'''
# 按位异或
print('a ^ b = ', a ^ b)
'''
a ^ b =  49   # 49 = 0011 0001
'''
# 按位取反
print('~a = ', ~a)
'''
60 → 0011 1100 → 1100 0011 → 1100 0010 → 1011 1101
~a =  -61
'''
# 左移
print('a << 2 = ', a << 2)
'''
a << 2 =  240   # 240 = 1111 0000
'''
# 右移
print('a >> 2 = ', a >> 2)
'''
a >> 2 =  15   # 15 = 0000 1111
'''

# 逻辑运算符
a = 10
b = 20
print(a and b)
'''
20
'''
print(a or b)
'''
10
'''
print(not a)
'''
False
'''
# 成员运算符，返回 bool 值 False 或 True
a = 10
list0 = [1, 2, 3, 4, 5]
print(a in list0)
'''
False
'''
print(a not in list0)
'''
True
'''

# 身份运算符，返回 bool 值 False 或 True
# is 判断两个标识符是否引用自一个对象
# id() 函数用于获取对象内存地址
a = 20
b = 20
print(a is b)
'''
True
'''
print(id(a) == id(b))
'''
True
'''
