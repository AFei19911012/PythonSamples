# -*- coding: utf-8 -*-
"""
 Created on 2021/7/10 13:55
 Filename   : ex_plot_rbm_logistic_classification.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import minmax_scale
from sklearn.base import clone


# Setting up
def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()

    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


def plot_rbm_logistic_classification():
    # Load Data
    X, y = datasets.load_digits(return_X_y=True)
    X = np.asarray(X, 'float32')
    X, Y = nudge_dataset(X, y)
    X = minmax_scale(X, feature_range=(0, 1))  # 0-1 scaling

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)

    # Models we will use
    logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
    rbm = BernoulliRBM(random_state=0, verbose=True)

    rbm_features_classifier = Pipeline(
        steps=[('rbm', rbm), ('logistic', logistic)])

    # #############################################################################
    # Training

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 10
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    logistic.C = 6000

    # Training RBM-Logistic Pipeline
    rbm_features_classifier.fit(X_train, Y_train)

    # Training the Logistic regression classifier directly on the pixel
    raw_pixel_classifier = clone(logistic)
    raw_pixel_classifier.C = 100.
    raw_pixel_classifier.fit(X_train, Y_train)

    # #############################################################################
    # Evaluation

    Y_pred = rbm_features_classifier.predict(X_test)
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(Y_test, Y_pred)))

    Y_pred = raw_pixel_classifier.predict(X_test)
    print("Logistic regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(Y_test, Y_pred)))

    # #############################################################################
    # Plotting

    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(rbm.components_):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()


if __name__ == '__main__':
    plot_rbm_logistic_classification()
