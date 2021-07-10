# -*- coding: utf-8 -*-
"""
 Created on 2021/7/5 23:49
 Filename   : ex17_polynomial_regression.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron

""" y(w, x) = w0 + w1*x1 + w2*x2 + w3*x1*x2 + w4*x1^2 + w5*x2^2 """


def polynomial_features():
    X = np.arange(6).reshape(3, 2)
    poly = PolynomialFeatures(degree=2)
    print(poly.fit_transform(X))


def plot_pipeline():
    model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
    # fit to an order-3 polynomial data
    x = np.arange(5)
    y = 3 - 2 * x + x ** 2 - x ** 3
    model = model.fit(x[:, np.newaxis], y)
    print(model.named_steps['linear'].coef_)


def plot_perceptron():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = X[:, 0] ^ X[:, 1]
    X = PolynomialFeatures(interaction_only=True).fit_transform(X).astype(int)
    clf = Perceptron(fit_intercept=False, max_iter=10, tol=None, shuffle = False).fit(X, y)
    print(clf.predict(X))
    print(clf.score(X, y))


if __name__ == '__main__':
    plot_perceptron()
