# -*- coding: utf-8 -*-
"""
 Created on 2021/7/2 18:20
 Filename   : matplotlib_plotting_tutorial.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description: https://www.machinelearningplus.com/plots/matplotlib-plotting-tutorial/
"""

# =======================================================
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
import matplotlib.image as mpimg
import matplotlib.animation as animation


def my_first_graph():
    """ 第一个绘图，默认以 0, 1, 2, ··· 为横轴刻度 """
    plt.plot([1, 2, 4, 9, 5, 3])
    plt.show()


def plot_x_y():
    """ 绘制 x 和 y 的关系曲线 """
    plt.plot([-2, -1, 3, 0], [1, 2, 4, 0])
    plt.show()


def limit_x_y():
    """ 设置 x 和 y 轴范围 """
    plt.plot([-2, -1, 3, 0], [1, 2, 4, 0])
    plt.axis([-4, 6, 0, 7])
    plt.show()


def numpy_function():
    """ 使用 numpy 绘制函数曲线 """
    x = np.linspace(-2, 2, 500)
    y = x ** 2
    plt.plot(x, y)
    plt.show()


def title_x_y_label():
    """ 添加标题、x 和 y 标签、网格线 """
    x = np.linspace(-2, 2, 500)
    y = x ** 2
    plt.plot(x, y)
    plt.title("Square function")
    plt.xlabel("x")
    plt.ylabel("y = x**2")
    plt.grid(True)
    plt.show()


def line_style_color():
    """ 线型、颜色、子图 """
    plt.subplot(3, 2, 1)
    plt.plot([0, 100, 100, 0, 0, 100, 50, 0, 100], [0, 0, 100, 100, 0, 100, 130, 100, 0])
    plt.axis([-10, 110, -10, 140])

    plt.subplot(3, 2, 2)
    plt.plot([0, 100, 100, 0, 0, 100, 50, 0, 100], [0, 0, 100, 100, 0, 100, 130, 100, 0], "g--")
    plt.axis([-10, 110, -10, 140])

    plt.subplot(3, 2, 3)
    plt.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], "r-", [0, 100, 50, 0, 100], [0, 100, 130, 100, 0], "g--")
    plt.axis([-10, 110, -10, 140])

    plt.subplot(3, 2, 4)
    plt.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], "r-")
    plt.plot([0, 100, 50, 0, 100], [0, 100, 130, 100, 0], "g--")
    plt.axis([-10, 110, -10, 140])

    plt.subplot(3, 2, 5)
    x = np.linspace(-1.4, 1.4, 30)
    plt.plot(x, x, 'g--', x, x ** 2, 'r:', x, x ** 3, 'b^')

    plt.subplot(3, 2, 6)
    x = np.linspace(-1.4, 1.4, 30)
    line1, line2, line3 = plt.plot(x, x, 'g--', x, x ** 2, 'r:', x, x ** 3, 'b^')
    line1.set_linewidth(3.0)
    line1.set_dash_capstyle("round")
    line3.set_alpha(0.2)

    plt.show()


def save_figure():
    """ 保存图片 """
    x = np.linspace(-1.4, 1.4, 30)
    plt.plot(x, x ** 2)
    plt.savefig("images/my_square_function.png", transparent=True)


def subplot_figure():
    """ 子图 """
    x = np.linspace(-1.4, 1.4, 30)
    plt.subplot(2, 2, 1)  # 2 rows, 2 columns, 1st subplot = top left
    plt.plot(x, x)
    plt.subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd subplot = top right
    plt.plot(x, x ** 2)
    plt.subplot(2, 1, 2)  # 2 rows, *1* column, 2nd subplot = bottom
    plt.plot(x, x ** 3)
    plt.show()


def subplot_figure_complex():
    """ 更复杂的子图 """
    x = np.linspace(-1.4, 1.4, 30)
    plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
    plt.plot(x, x ** 2)
    plt.subplot2grid((3, 3), (0, 2))
    plt.plot(x, x ** 3)
    plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    plt.plot(x, x ** 4)
    plt.subplot2grid((3, 3), (2, 0), colspan=2)
    plt.plot(x, x ** 5)
    plt.show()


def multiple_figures():
    """ 多图 """
    x = np.linspace(-1.4, 1.4, 30)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(x, x ** 2)
    plt.title("Square and Cube")
    plt.subplot(212)
    plt.plot(x, x ** 3)

    plt.figure(2, figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x, x ** 4)
    plt.title("y = x**4")
    plt.subplot(122)
    plt.plot(x, x ** 5)
    plt.title("y = x**5")

    plt.figure(1)  # back to figure 1, current subplot is 212 (bottom)
    plt.plot(x, -x ** 3, "r:")

    plt.show()


def plotting_explicit():
    """ 显式绘图 """
    x = np.linspace(-2, 2, 200)
    fig1, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True)
    fig1.set_size_inches(10, 5)
    line1, line2 = ax_top.plot(x, np.sin(3 * x ** 2), "r-", x, np.cos(5 * x ** 2), "b-")
    line3, = ax_bottom.plot(x, np.sin(3 * x), "r-")
    ax_top.grid(True)

    fig2, ax = plt.subplots(1, 1)
    ax.plot(x, x ** 2)
    plt.show()


def drawing_text():
    """ 绘制文本 """
    x = np.linspace(-1.5, 1.5, 30)
    px = 0.8
    py = px ** 2

    plt.plot(x, x ** 2, "b-", px, py, "ro")

    plt.text(0, 1.5, "Square function\n$y = x^2$", fontsize=20, color='blue', horizontalalignment="center")
    plt.text(px - 0.08, py, "Beautiful point", ha="right", weight="heavy")
    plt.text(px, py, "x = %0.2f\ny = %0.2f" % (px, py), rotation=50, color='gray')

    plt.show()


def drawing_annotation():
    """ 绘制箭头 """
    x = np.linspace(-1.5, 1.5, 30)
    px = 0.8
    py = px ** 2
    plt.plot(x, x ** 2, px, py, "ro")
    plt.annotate("Beautiful point", xy=(px, py), xytext=(px - 1.3, py + 0.5),
                 color="green", weight="heavy", fontsize=14,
                 arrowprops={"facecolor": "lightgreen"})
    plt.show()


def drawing_bounding_box():
    """ 绘制注释边框 """
    x = np.linspace(-1.5, 1.5, 30)
    px = 0.8
    py = px ** 2
    plt.plot(x, x ** 2, px, py, "ro")

    bbox_props = dict(boxstyle="rarrow,pad=0.3", ec="b", lw=2, fc="lightblue")
    plt.text(px - 0.2, py, "Beautiful point", bbox=bbox_props, ha="right")

    bbox_props = dict(boxstyle="round4,pad=1,rounding_size=0.2", ec="black", fc="#EEEEFF", lw=5)
    plt.text(0, 1.5, "Square function\n$y = x^2$", fontsize=20, color='black', ha="center", bbox=bbox_props)

    plt.show()


def drawing_bounding_box_xkcd():
    """ 手绘样式 """
    x = np.linspace(-1.5, 1.5, 30)
    px = 0.8
    py = px ** 2
    with plt.xkcd():
        plt.plot(x, x ** 2, px, py, "ro")

    bbox_props = dict(boxstyle="rarrow,pad=0.3", ec="b", lw=2, fc="lightblue")
    plt.text(px - 0.2, py, "Beautiful point", bbox=bbox_props, ha="right")

    bbox_props = dict(boxstyle="round4,pad=1,rounding_size=0.2", ec="black", fc="#EEEEFF", lw=5)
    plt.text(0, 1.5, "Square function\n$y = x^2$", fontsize=20, color='black', ha="center", bbox=bbox_props)

    plt.show()


def plotting_legend():
    """ 图例 """
    x = np.linspace(-1.4, 1.4, 50)
    plt.plot(x, x ** 2, "r--", label="Square function")
    plt.plot(x, x ** 3, "g-", label="Cube function")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def plotting_logarithmic():
    """ 对数刻度 """
    x = np.linspace(0.1, 15, 500)
    y = x ** 3 / np.exp(2 * x)
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('log')
    plt.grid(True)
    plt.show()


def plotting_ticks_tickers():
    """ 刻度标签 """
    x = np.linspace(-2, 2, 100)

    plt.figure(1, figsize=(15, 10))
    plt.subplot(131)
    plt.plot(x, x ** 3)
    plt.grid(True)
    plt.title("Default ticks")

    ax = plt.subplot(132)
    plt.plot(x, x ** 3)
    ax.xaxis.set_ticks(np.arange(-2, 2, 1))
    plt.grid(True)
    plt.title("Manual ticks on the x-axis")

    ax = plt.subplot(133)
    plt.plot(x, x ** 3)
    plt.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom='off')
    ax.xaxis.set_ticks([-2, 0, 1, 2])
    ax.yaxis.set_ticks(np.arange(-5, 5, 1))
    ax.yaxis.set_ticklabels(["min", -4, -3, -2, -1, 0, 1, 2, 3, "max"])
    plt.title("Manual ticks and tick labels\n(plus minor ticks) on the y-axis")

    plt.grid(True)

    plt.show()


def plotting_polar():
    """ 极坐标 """
    radius = 1
    theta = np.linspace(0, 2 * np.pi * radius, 1000)

    plt.subplot(111, projection='polar')
    plt.plot(theta, np.sin(5 * theta), "g-")
    plt.plot(theta, 0.5 * np.cos(20 * theta), "b-")
    plt.show()


def plotting_surface():
    """ 3d 曲面图 """
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    figure = plt.figure(1, figsize=(12, 4))
    subplot3d = plt.subplot(111, projection='3d')
    surface = subplot3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0.1)
    plt.show()


def plotting_contourf():
    """ 3d 等高线图 """
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    plt.contourf(X, Y, Z, cmap=matplotlib.cm.coolwarm)
    plt.colorbar()
    plt.show()


def plotting_scatter():
    """散点图 """
    x, y = rand(2, 100)
    plt.scatter(x, y)
    plt.show()


def plotting_scatter_pointsize():
    """ 散点图：点大小 """
    x, y, scale = rand(3, 100)
    scale = 500 * scale ** 5
    plt.scatter(x, y, s=scale)
    plt.show()


def plotting_scatter_attributes():
    """ 散点图：点大小、颜色、透明度 """
    for color in ['red', 'green', 'blue']:
        n = 100
        x, y = rand(2, n)
        scale = 500.0 * rand(n) ** 5
        plt.scatter(x, y, s=scale, c=color, alpha=0.3, edgecolors='blue')
    plt.grid(True)
    plt.show()


def plotting_histograms():
    """ 绘制直方图 """
    data = [1, 1.1, 1.8, 2, 2.1, 3.2, 3, 3, 3, 3]
    plt.subplot(211)
    plt.hist(data, bins=10, rwidth=0.8)

    plt.subplot(212)
    plt.hist(data, bins=[1, 1.5, 2, 2.5, 3], rwidth=0.95)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.show()


def plotting_histograms_complex():
    """ 复杂一点的直方图 """
    data1 = np.random.randn(400)
    data2 = np.random.randn(500) + 3
    data3 = np.random.randn(450) + 6
    data4a = np.random.randn(200) + 9
    data4b = np.random.randn(100) + 10

    plt.hist(data1, bins=5, color='g', alpha=0.75, label='bar hist')  # default histtype='bar'
    plt.hist(data2, color='b', alpha=0.65, histtype='stepfilled', label='stepfilled hist')
    plt.hist(data3, color='r', histtype='step', label='step hist')
    plt.hist((data4a, data4b), color=('r', 'm'), alpha=0.55, histtype='barstacked', label=('barstacked a', 'barstacked b'))

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


def plotting_image():
    """ 读取图像 """
    img = mpimg.imread('images/my_square_function.png')
    print(img.shape, img.dtype)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def plotting_image_genetated():
    """ 图像数据 """
    img = np.arange(100 * 100).reshape(100, 100)
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="hot")

    img = np.empty((20, 30, 3))
    img[:, :10] = [0, 0, 0.6]
    img[:, 10:20] = [1, 1, 1]
    img[:, 20:] = [0.6, 0, 0]
    plt.subplot(1, 2, 2)
    plt.imshow(img, interpolation="bilinear")

    plt.show()


def plotting_animations():
    """ 动画 """

    """ this function will be called at every iteration """
    """ we only plot the first `num` data points """
    def update_line(num, data, line):
        line.set_data(data[..., :num] + np.random.rand(2, num) / 25)
        return line

    x = np.linspace(-1, 1, 100)
    y = np.sin(x ** 2 * 25)
    data = np.array([x, y])

    fig = plt.figure()
    line, = plt.plot([], [], "r-")  # start with an empty plot
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.plot([-0.5, 0.5], [0, 0], "b-", [0, 0], [-0.5, 0.5], "b-", 0, 0, "ro")
    plt.grid(True)
    plt.title("Marvelous animation")

    """ make animation """
    line_ani = animation.FuncAnimation(fig, update_line, frames=50, fargs=(data, line), interval=100)
    plt.close()  # call close() to avoid displaying the static plot

    """ save as *.gif """
    line_ani.save('videos/my_wiggly_animation.gif', writer='imagemagick', fps=100)
    plt.show()


if __name__ == '__main__':
    my_first_graph()
    plot_x_y()
    limit_x_y()
    numpy_function()
    title_x_y_label()
    line_style_color()
    save_figure()
    subplot_figure()
    subplot_figure_complex()
    multiple_figures()
    plotting_explicit()
    drawing_text()
    drawing_annotation()
    drawing_bounding_box()
    drawing_bounding_box_xkcd()
    plotting_legend()
    plotting_logarithmic()
    plotting_ticks_tickers()
    plotting_polar()
    plotting_surface()
    plotting_contourf()
    plotting_scatter()
    plotting_scatter_pointsize()
    plotting_scatter_attributes()
    plotting_histograms()
    plotting_histograms_complex()
    plotting_image()
    plotting_image_genetated()
    plotting_animations()
