# -*- coding: utf-8 -*-
"""
 Created on 2021/4/4 22:22
 Author: ff_wang
 E-mail: 1105936347@qq.com
 Zhihu : https://www.zhihu.com/people/1105936347
 Github: https://github.com/AFei19911012
-------------------------------------------------
"""


def divide(x, y):
    try:
        result = x / y
    # try 发生异常，执行 except
    except ZeroDivisionError:
        print('division by zero.')
    # try 没有发生异常，执行 else
    else:
        print(f'result is: {result}')
    # 不管 try 有没有发生异常，finally 都会执行
    finally:
        print('executing finally clause\n')


# 正确结果
result = divide(2, 1)
'''
result is: 2.0
executing finally clause
'''
# 处理的异常
result = divide(2, 0)
'''
division by zero.
executing finally clause
'''
# 未处理的异常类型
result = divide('2', '0')
'''
executing finally clause
TypeError: unsupported operand type(s) for /: 'str' and 'str'
'''

