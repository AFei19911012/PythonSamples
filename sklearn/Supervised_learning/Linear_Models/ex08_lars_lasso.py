# -*- coding: utf-8 -*-
"""
 Created on 2021/7/5 23:03
 Filename   : ex08_lars_lasso.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def lars_lasso():
    reg = linear_model.LassoLars(alpha=.1)
    reg.fit([[0, 0], [1, 1]], [0, 1])
    print(reg.coef_)


def plot_lasso_lars():
    X, y = datasets.load_diabetes(return_X_y=True)

    print("Computing regularization path using the LARS ...")
    _, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)

    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]

    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('LASSO Path')
    plt.axis('tight')
    plt.show()


if __name__ == '__main__':
    lars_lasso()
