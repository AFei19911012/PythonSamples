# -*- coding: utf-8 -*-
"""
 Created on 2021/7/10 0:21
 Filename   : ex_plot_multi_layer_perceptron.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
from sklearn.neural_network import MLPClassifier


def plot_multi_layer_perceptron():
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(X, y)

    clf.predict([[2., 2.], [-1., -2.]])

    """ probability estimates """
    clf.predict_proba([[2., 2.], [1., 2.]])

    X = [[0., 0.], [1., 1.]]
    y = [[0, 1], [1, 1]]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(15,), random_state=1)

    clf.fit(X, y)

    clf.predict([[1., 2.]])

    clf.predict([[0., 0.]])


if __name__ == '__main__':
    plot_multi_layer_perceptron()
