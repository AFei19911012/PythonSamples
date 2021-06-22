# -*- coding: utf-8 -*-
"""
Created on 2021/3/28 23:07
author: ff_wang
E-mail: 1105936347@qq.com
Zhihu: https://www.zhihu.com/people/1105936347
Github: https://github.com/AFei19911012
"""

import sys

# 没有输入参数
def my_fun1():
    print('This is my first function.')


# 一个输入参数
def my_fun2(name):
    print(name, 'is happy')


# 两个参数，一个返回值
def my_sum(num1, num2):
    return num1 + num2


my_fun1()
'''
This is my first function.
'''
my_fun2('Taosy')
'''
Taosy is happy
'''
total = my_sum(1, 2)
print(total)
'''
3
'''


# 函数文档
def my_fun_doc(name):
    """
    函数定义过程中的 name 叫形参
    param name: name
    return: none
    """
    print('传递进来的 ' + name + ' 叫实参')


# 显示函数文档
help(my_fun_doc)
'''
Help on function my_fun_doc in module __main__:

my_fun_doc(name)
    函数定义过程中的 name 叫形参
    param name: name
    return: none
'''


# 关键字参数，输入参数较多时有用，防止出错
def my_fun(name, word):
    print(name + '-->' + word)


my_fun('Hello', 'Python')
my_fun('Python', 'Hello')
my_fun(word='Python', name='Hello')
'''
Hello-->Python
Python-->Hello
Hello-->Python
'''


# 默认参数
def my_fun(name='Hello', word='Python'):
    print(name + '-->' + word)


my_fun()
my_fun(word='World', name='Hello')
'''
Hello-->Python
Hello-->World
'''


# 不定长参数
def my_fun(*params):
    print(f'总共有 {len(params)} 个参数')
    print(f'第二个参数是：{params[1]}')


my_fun(1, 'Hello', 2, 3)
'''
总共有 4 个参数
第二个参数是：Hello
'''


# 不定长参数之外，还有参数，建议用默认参数，防止出错
def my_fun(*params, ext=111):
    print(f'总共有 {len(params) + 1} 个参数')
    print(f'第二个参数是：{params[1]}')
    print(f'最后一个参数是：{ext}')


my_fun(1, 2, 3, 'a', ext=999)
'''
总共有 5 个参数
第二个参数是：2
最后一个参数是：999
'''


# 传不可变对象，局部变量和全局变量
def my_fun(price, rate):
    final_price = price * rate
    return final_price


old_price = 100
rate = 0.8
new_price = my_fun(old_price, rate)
# final_price 只在函数体生效，属于局部变量，函数体外不可访问
print(new_price)
'''
80.0
'''


def my_fun(price, rate):
    final_price = price * rate
    old_price = 50
    print(f'修改后的 old_price 的值是：{old_price}')
    return final_price


old_price = 100
fate = 0.8
new_price = my_fun(old_price, rate)
print(f'修改后的 old_price 的值是：{old_price}')
print(f'打折后的价格是：{new_price}')
'''
修改后的 old_price 的值是：50   # 函数体内修改函数外的变量值，仅函数体内有效，在函数体外依然保持不变
修改后的 old_price 的值是：100
打折后的价格是：80.0
'''

x = 5


def my_fun():
    return x * x


print(my_fun())
'''
25  # 调用外部的 x 值
'''

# 传可变对象
def my_fun(mylist):
    """
    修改传入的列表
    :param mylist:
    :return:
    """
    mylist.append([10, 20])
    print(f'函数体内取值：{mylist}')
    return


mylist = [1, 2, 3]
my_fun(mylist)
'''
函数体内取值：[1, 2, 3, [10, 20]]
'''
print(f'函数体外取值：{mylist}')
'''
函数体内取值：[1, 2, 3, [10, 20]]    # 实例体内 mylist 添加新内容的对象和传入对象是同一个引用 
'''

# 匿名函数，用 lambda 来创建
my_sum = lambda arg1, arg2: arg1 + arg2
# 调用 my_sum
print(f'相加后的值为: {my_sum(10, 20)}')
'''
相加后的值为: 30
'''


# 嵌套函数
def my_fun1():
    print('my_fun1 正在执行...')

    def my_fun2():
        print('my_fun2 正在执行...')

    my_fun2()
    return


my_fun1()
'''
my_fun1 正在执行...
my_fun2 正在执行...
'''


# nonlocal 关键字
def my_fun1():
    x = 5

    def my_fun2():
        nonlocal x   # 如果不用 nonlocal 关键字，则无法识别 x
        x *= x
        return x

    return my_fun2()


print(my_fun1())
'''
25
'''

# filter：过滤 False 的元素
f1 = filter(None, [1, 0, False, True])
print(list(f1))
'''
[1, True]
'''


# 奇数
def odd(x):
    return x % 2


result = filter(odd, range(10))
print(list(result))
'''
[1, 3, 5, 7, 9]
'''
# lambda 形式
print(list(filter(lambda x: x % 2, range(10))))
'''
[1, 3, 5, 7, 9]
'''

# map：每个元素执行对应的操作
print(list(map(lambda x: x*2, range(10))))
'''
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
'''

# 递归函数
# 导入 sys，设置递归深度
# 慎用递归，效率低、耗内存
sys.setrecursionlimit(1000)


# 阶乘
def my_factorial(n):
    result = n
    for i in range(1, n):
        result *= i
    return result


print(my_factorial(5))
'''
120
'''


# 用递归
def my_factorial(n):
    if n == 1:
        return 1
    else:
        return n * my_factorial(n - 1)


print(my_factorial(5))
'''
120
'''


# 汉诺塔游戏
def hanoi(n, x, y, z):
    if n == 1:
        print(x, '-->', z)
    else:
        hanoi(n - 1, x, z, y)  # 将前 n - 1 个盘子从 x 移动到 y 上
        print(x, '-->', z)     # 将最底下的最后一个盘子从 x 移动到 z 上
        hanoi(n - 1, y, x, z)  # 将 y 上的 n - 1 个盘子移动到 z 上


hanoi(3, 'x', 'y', 'z')
'''
x --> z
x --> y
z --> y
x --> z
y --> x
y --> z
x --> z
'''