# -*- coding: utf-8 -*-
"""
Created on 2021/3/28 14:04
author: ff_wang
E-mail: 1105936347@qq.com
Zhihu: https://www.zhihu.com/people/1105936347
Github: https://github.com/AFei19911012
"""

# 索引访问
list1 = ['red', 'green', 'blue', 'yellow', 'white', 'black']
# 第一个索引为 0
print(list1[0])
'''
red
'''
# 反向索引：最后一个索引为 -1，往前一位为 -2
print(list1[-1])
'''
black
'''
print(list1[-2])
'''
white
'''

# 切片，可用来进行列表拷贝
list1 = [10, 20, 30, 40, 50, 60, 70, 80]
# 注意左闭右开
print(list1[0:4])
'''
[10, 20, 30, 40]
'''
print(list1[0:-2])
'''
[10, 20, 30, 40, 50, 60]
'''
print(list1[:3])
'''
[10, 20, 30]
'''
print(list1[1:])
'''
[20, 30, 40, 50, 60, 70, 80]
'''
print(list1[:])
'''
[10, 20, 30, 40, 50, 60, 70, 80]
'''

# 列表添加元素
list1 = [1, 2, 3, 4]
# 添加列表到最后
list1.extend([6, 7])
print(list1)
'''
[1, 2, 3, 4, 5, 6, 7]
'''
# 添加元素到最后
list1.append(5)
print(list1)
'''
[1, 2, 3, 4, 5]
'''
# 指定位置添加元素
list1.insert(1, 10)
print(list1)
'''
[1, 10, 2, 3, 4, 5, 6, 7]
'''

# 删除元素
list1 = ['a', 'b', 'c', 'd', 'e']
list1.remove('b')
print(list1)
'''
['a', 'c', 'd', 'e']
'''
del list1[1]
print(list1)
'''
['a', 'd', 'e']
'''
# 删除最后一个元素
list1 = ['a', 'b', 'c', 'd', 'e']
list1.pop()
print(list1)
'''
['a', 'b', 'c', 'd']
'''
# 删除第二个元素
list1.pop(1)
print(list1)
'''
['a', 'c', 'd']
'''

# 列表拼接
print([1, 2, 3] + [4, 5])
'''
[1, 2, 3, 4, 5]
'''
# 列表重复
print([1, 2, 3] * 2)
'''
[1, 2, 3, 1, 2, 3]
'''

# count() 某个元素个数
# index() 元素索引位置
# reverse() 逆序
# clear() 清空
# copy() 复制
list1 = [1, 1, 2, 3]
print(list1.count(1))
'''
2
'''
print(list1.index(2))
'''
2
'''
list1.reverse()
print(list1)
'''
[3, 2, 1, 1]
'''
list1.clear()
print(list1)
'''
[]
'''
list1 = [1, 1, 2, 3]
list2 = list1.copy()
list2[1] = 5
print(list2)
'''
[1, 5, 2, 3]
'''
print(list1)
'''
[1, 1, 2, 3]
'''
# 等号赋值，相当于两边访问的是同一个内存地址
list1 = [1, 1, 2, 3]
list2 = list1
list2[1] = 5
print(list2)
'''
[1, 5, 2, 3]
'''
print(list1)
'''
[1, 5, 2, 3]
'''

# 将元组或字符串转换为列表
tup1 = (1, 2, 3)
str1 = 'Hello python'
list1 = list(tup1)
list2 = list(str1)
print(list1)
'''
[1, 2, 3]
'''
print(list2)
'''
['H', 'e', 'l', 'l', 'o', ' ', 'p', 'y', 't', 'h', 'o', 'n']
'''

############################################################
# 创建元组
tup1 = (1, 2, 3, 4, 5)
tup2 = 1, 2, 3, 4, 5
tup3 = ()
# 元组只包含一个元素时，需要再元素后面添加逗号，否则括号会被当做运算符使用
tup4 = (1,)
print(tup1)
print(tup2)
'''
(1, 2, 3, 4, 5)
'''

# 访问元组：索引、切片
print("tup1[0]: ", tup1[0])
'''
tup1[0]:  1
'''
print("tup2[1:5]: ", tup2[1:5])
'''
tup2[1:5]:  (2, 3, 4, 5)
'''

# 修改元组：元组中的元素值不可修改、删除，可进行拼接
tup1 = (1, 2, 3)
tup2 = ('a', 'b', 'c')
tup3 = tup1 + tup2
print(tup3)
'''
(1, 2, 3, 'a', 'b', 'c')
'''

# 可迭代系列转换为元组
list1 = [1, 2, 3]
tup1 = tuple(list1)
print(tup1)
'''
(1, 2, 3)
'''

############################################################
# 创建字典
dict1 = {'name': 'ff_wang', 'url': 'https://www.zhihu.com/people/1105936347'}
# 访问字典
print("dict1['name']:", dict1['name'])
'''
dict1['name']: ff_wang
'''
print("dict1['url']:", dict1['url'])
'''
dict1['url']: https://www.zhihu.com/people/1105936347
'''

# 修改字典
# 添加新的键/值对
dict1['age'] = 20
print(dict1)
'''
{'name': 'ff_wang', 'url': 'https://www.zhihu.com/people/1105936347', 'age': 20}
'''
# 更新 age
dict1['age'] = 30
print(dict1)
'''
{'name': 'ff_wang', 'url': 'https://www.zhihu.com/people/1105936347', 'age': 30}
'''
# 删除字典元素
del dict1['age']
print(dict1)
'''
{'name': 'ff_wang', 'url': 'https://www.zhihu.com/people/1105936347'}
'''
# 清空字典
dict1.clear()
print(dict1)
'''
{}
'''

############################################################
# 创建集合：无序、不重复元素序列
set1 = {'a', 'b', 'c'}
set1 = set(('a', 'b', 'c'))
# 集合运算
a = set('abcde')
b = set('abcg')
# 差集
print(a - b)
'''
{'d', 'e'}
'''
# 并集
print(a | b)
'''
{'a', 'c', 'd', 'b', 'g', 'e'}
'''
# 交集
print(a & b)
'''
{'c', 'b', 'a'}
'''
# 并集 - 交集
print(a ^ b)
'''
{'d', 'g', 'e'}
'''

# 添加元素
set1 = set(('a', 'b', 'c'))
# 添加元素
set1.add(1)
print(set1)
'''
{1, 'b', 'c', 'a'}
'''
set1.update({3, 4})
print(set1)
'''
{1, 'b', 3, 'c', 'a', 4}
'''
set1.update(['aa', 'bb'], [5, 6])
print(set1)
'''
{1, 3, 4, 5, 6, 'c', 'a', 'bb', 'b', 'aa'}
'''

# 移除元素
# 如果元素不存在则报错
set1.remove('aa')
print(set1)
'''
{1, 3, 4, 'bb', 5, 6, 'a', 'b', 'c'}
'''
# 元素不存在不报错
set1.discard('bb')
print(set1)
'''
{1, 3, 4, 5, 6, 'a', 'b', 'c'}
'''
# 随机移除：pop 方法对集合进行无序排列，然后删除左边第一个元素
set1 = set(('a', 'b', 'c'))
x = set1.pop()
print(x)
'''
c
'''