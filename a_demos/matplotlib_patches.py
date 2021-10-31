# -*- coding: utf-8 -*-
"""
 Created on 2021/6/21 1:52
 Filename   : ex_patches.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: 形状填充图
"""

# =======================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch


def main():
    xy1 = np.array([0.2, 0.2])
    xy2 = np.array([0.2, 0.8])
    xy3 = np.array([0.8, 0.2])
    xy4 = np.array([0.8, 0.8])
    ax = plt.subplot()
    """ 圆 """
    circle = patch.Circle(xy1, 0.1)
    ax.add_patch(circle)
    """ 长方形 """
    rect = patch.Rectangle(xy2, 0.2, 0.1, color='r')
    ax.add_patch(rect)
    """ 多边形 """
    polygon = patch.RegularPolygon(xy3, 6, 0.1, color='g')
    ax.add_patch(polygon)
    """ 椭圆 """
    ellipse = patch.Ellipse(xy4, 0.4, 0.2, color='c')
    ax.add_patch(ellipse)
    ax.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
