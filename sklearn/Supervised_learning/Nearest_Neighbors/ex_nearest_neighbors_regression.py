# -*- coding: utf-8 -*-
"""
 Created on 2021/7/9 23:33
 Filename   : ex_nearest_neighbors_regression.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors


def plot_regression():
    np.random.seed(0)
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    T = np.linspace(0, 5, 500)[:, np.newaxis]
    y = np.sin(X).ravel()

    # Add noise to targets
    y[::5] += 1 * (0.5 - np.random.rand(8))

    # #############################################################################
    # Fit regression model
    n_neighbors = 5

    for i, weights in enumerate(['uniform', 'distance']):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        y_ = knn.fit(X, y).predict(T)

        plt.subplot(2, 1, i + 1)
        plt.scatter(X, y, color='darkorange', label='data')
        plt.plot(T, y_, color='navy', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_regression()
