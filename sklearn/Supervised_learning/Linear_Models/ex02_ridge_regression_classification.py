# -*- coding: utf-8 -*-
"""
 Created on 2021/7/4 18:15
 Filename   : ex02_ridge_regression_classification.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description: 岭回归、分类
"""

# =======================================================
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

""" min{norm(Xw - y, 2)^2 + α*norm(w, 2)^2) """


def ridge_regression():
    """ 岭回归 """
    reg = linear_model.Ridge(alpha=.5)
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    print(reg.coef_)
    print(reg.intercept_)


def plot_ridge_path():
    """ 正则化参数和权系数关系 """

    # X is the 10x10 Hilbert matrix
    X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
    y = np.ones(10)

    # Compute paths
    n_alphas = 200
    alphas = np.logspace(-10, -2, n_alphas)
    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)

    # Display results
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()


def plot_document_classification_20newsgroups():
    """ https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document
    -classification-20newsgroups-py """
    pass


def plot_linear_model_coefficient_interpretation():
    """ https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples
    -inspection-plot-linear-model-coefficient-interpretation-py """
    pass


def ridge_cv():
    """ 交叉校验 """
    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    print(reg.alpha_)


if __name__ == '__main__':
    # ridge_regression()
    # plot_ridge_path()
    ridge_cv()
