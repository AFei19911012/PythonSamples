# -*- coding: utf-8 -*-
"""
 Created on 2021/6/21 0:30
 Filename   : matplotlib_tips.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: 基本用法
"""

# =======================================================
import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.arange(0, 10, 0.5)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = 0.1*(x - 5)**2
    plt.plot(x, y1)
    plt.plot(x, y2)

    """ 中文标题和坐标轴 """
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('中文标题')

    """ 添加文字 """
    plt.text(3, 0.5, '添加文字：y = sin(x)')

    """ 注释 """
    plt.annotate('注释1', xy=(1.57, 1), xytext=(2, 0.95))
    plt.annotate('注释2', xy=(4.71, -1), xytext=(4, -0.25), arrowprops={'headwidth': 10, 'facecolor': 'r'})

    """ 坐标轴名称 """
    plt.xlabel('x 轴')
    plt.ylabel('y 轴')

    """ 图例 """
    plt.legend(['y = sin(x)', 'y = cos(x)'])
    """ 
    颜色、标记 
    ‘.’：点(point marker)
    ‘,’：像素点(pixel marker)
    ‘o’：圆形(circle marker)
    ‘v’：朝下三角形(triangle_down marker)
    ‘^’：朝上三角形(triangle_up marker)
    ‘<‘：朝左三角形(triangle_left marker)
    ‘>’：朝右三角形(triangle_right marker)
    ‘1’：(tri_down marker)
    ‘2’：(tri_up marker)
    ‘3’：(tri_left marker)
    ‘4’：(tri_right marker)
    ‘s’：正方形(square marker)
    ‘p’：五边星(pentagon marker)
    ‘*’：星型(star marker)
    ‘h’：1号六角形(hexagon1 marker)
    ‘H’：2号六角形(hexagon2 marker)
    ‘+’：+号标记(plus marker)
    ‘x’：x号标记(x marker)
    ‘D’：菱形(diamond marker)
    ‘d’：小型菱形(thin_diamond marker)
    ‘|’：垂直线形(vline marker)
    ‘_’：水平线形(hline marker)
    """
    plt.plot(x, y3, color='k', marker='o')
    # plt.plot(x, y3, color='0')
    # plt.plot(x, y3, color='#000000')
    # plt.plot(x, y3, color=(0, 0, 0))

    """ 数学公式 """
    plt.text(3.14, 0.75, r'$ \alpha \beta \pi $', size=25)

    """ 显示网格 """
    plt.grid()

    """ 坐标轴范围 """
    plt.xlim(xmin=-1, xmax=10)

    """ 坐标轴刻度 """
    plt.locator_params(nbins=30)
    # plt.locator_params('y', nbins=30)

    """ 坐标轴刻度自适应位置 """
    plt.gcf().autofmt_xdate()

    """ 双坐标轴 """
    y4 = x*x
    plt.twinx()
    plt.plot(x, y4, 'm')

    """ 填充 """
    y5 = 0.5*x*x + x
    plt.fill(x, y5, 'g')
    plt.fill_between(x, y4, y5, where=y4 > y5, color='r', interpolate=True)

    """ 必不可少 """
    plt.show()


if __name__ == '__main__':
    main()
