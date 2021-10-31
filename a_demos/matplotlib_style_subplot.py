# -*- coding: utf-8 -*-
"""
 Created on 2021/6/21 1:59
 Filename   : ex_style_subplot.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: 子图和样式
"""

# =======================================================
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 查看支持的样式
    print(plt.style.available)
    """ 应用样式 """
    plt.style.use('seaborn')
    """ 新建 4 个子图 """
    fig, axes = plt.subplots(2, 2)
    ax1, ax2, ax3, ax4 = axes.ravel()
    """ 第一个图 """
    x, y = np.random.normal(size=(2, 100))
    ax1.plot(x, y, 'o')
    """ 第二个图 """
    x = np.arange(0, 10)
    y = np.arange(0, 10)
    colors = plt.rcParams['axes.prop_cycle']
    length = np.linspace(0, 10, len(colors))
    for s in length:
        ax2.plot(x, y + s, '-')
    """ 第三个图 """
    x = np.arange(5)
    y1, y2, y3 = np.random.randint(1, 25, size=(3, 5))
    width = 0.25
    ax3.bar(x, y1, width)
    ax3.bar(x + width, y2, width)
    ax3.bar(x + 2 * width, y3, width)
    """ 第四个图 """
    for i, color in enumerate(colors):
        xy = np.random.normal(size=2)
    ax4.add_patch(plt.Circle(xy, radius=0.3, color=color['color']))
    ax4.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
