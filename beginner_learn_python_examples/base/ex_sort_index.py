# -*- coding: utf-8 -*-
"""
 Created on 2021/4/15 11:44
 Filename: ex_sort_index.py
 Author  : Taosy
 Zhihu   : https://www.zhihu.com/people/1105936347
 Github  : https://github.com/AFei19911012
 Describe:
"""

import random

# dict
nums = {num: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for num in range(3)}
print(nums)
'''
{0: (133, 20, 142), 1: (93, 135, 243), 2: (186, 135, 9)}
'''

# list
nums = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(3)]
print(nums)
'''
[(46, 141, 231), (120, 218, 90), (3, 127, 57)]
'''


def get_sort_index(num_list, reverse=False):
    num_enum = enumerate(num_list)
    num_sort = sorted(num_enum, key=lambda x: x[1], reverse=reverse)
    return [num[0] for num in num_sort], [num[1] for num in num_sort]


nums = [1, 2, 4, 3, 5]
idx, nums_sort = get_sort_index(nums)
print(idx)
print(nums_sort)
'''
[0, 1, 3, 2, 4]
[1, 2, 3, 4, 5]
'''

idx, nums_sort = get_sort_index(nums, reverse=True)
print(idx)
print(nums_sort)
'''
[4, 2, 3, 1, 0]
[5, 4, 3, 2, 1]
'''
