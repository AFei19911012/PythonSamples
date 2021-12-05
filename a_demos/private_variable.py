# -*- coding: utf-8 -*-
"""
 Created on 2021/5/14 14:40
 Filename   : private_variable.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""

# =======================================================
import numpy as np


class Complex:
    # 内置，初始化时调用
    def __init__(self, realpart, imagpart):
        self.re = realpart    # 公有变量
        self.im = imagpart
        self.re_ = realpart   # 防止和关键词重名
        self.im_ = imagpart
        self.__re = realpart  # 私有化属性或方法，无法在外部直接访问
        self.__im = imagpart

    def abs(self):
        return np.sqrt(self.__re * self.__re + self._im * self._im)


class ComplexChild(Complex):
    def __init__(self, x, y, z):
        super(ComplexChild, self).__init__(x, y)
        # Complex.__init__(self, x, y)
        self.z = z

    def real(self):
        return self.re


if __name__ == '__main__':
    # 类的实例化
    cp = Complex(3, 4)
    cp_child = ComplexChild(3, 4, 5)
    # 无法访问 __re
    print(f'cp.re = {cp.re}')
    print(f'cp.re_ = {cp.re_}')
    print(f'cp_child.real() = {cp_child.real()}')

'''
cp.re = 3
cp.re_ = 3
cp_child.real() = 3
'''