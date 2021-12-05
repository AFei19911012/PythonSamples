# -*- coding: utf-8 -*-
"""
 Created on 2021/6/29 23:58
 Filename   : numpy_fractal_Mandelbrot_set.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description: Mandelbrot set 分形
"""

# =======================================================
import numpy as np
import matplotlib.pyplot as plt


def mandelbrot(h, w, maxit=20):
    """ Returns an image of the Mandelbrot fractal of size (h,w) """
    y, x = np.ogrid[-2:2:h*1j, -2:0.8:w*1j]
    c = x + y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)
    for i in range(maxit):
        z = z ** 2 + c
        """ 发散 """
        diverge = z * np.conj(z) > 2 ** 2
        div_now = diverge & (divtime == maxit)
        divtime[div_now] = i
        z[diverge] = 2
    return divtime


if __name__ == '__main__':
    image = mandelbrot(400, 400)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
